"""Parsed-mesh LRU cache for the Ninja-Nj / Xj parsers.

Phase D Win 4 (2026-04-25). PSOBB.IO ships a small handful of "dragon
class" models — Dragon (~4 MB body NJ, 9677 verts × 124 bones × 7539
triangles), De Rol Le, Sil Dragon, Mericarol, plAbdy00 (player body) —
that re-cost ~1.1 s of pure-Python parse time every time the user opens
them. The user opens the same model many times in a session (variant
picker, motion preview, paint, sculpt), so the second-and-onwards opens
are wasted work.

This module wraps the four public parser entry points
(``parse_nj_file``, ``parse_xj_file``, ``parse_nj_skinned``,
``parse_skeleton``) with an LRU cache that holds parsed model objects in
memory and (optionally) on disk. Cold parse populates the cache; warm
parse hits in <5 ms (a dict lookup + reference handoff).

Cache layering (top-to-bottom — outermost serves first):

  ``manifest._NEWEST_MTIME_CACHE``   — install-tree mtime, 60 s TTL
  ``formats.bml._PRS_INNER_CACHE``   — decompressed BML inner blobs (64 MB)
  **THIS MODULE** ``_PARSE_CACHE``   — parsed XjMesh / XjBone lists (256 MB)
  ``server._SKINNED_PAYLOAD_CACHE``  — skinned wire dict + on-disk JSON (128 MB / 256)
  ``server._BINDING_CACHE``          — NJTL→XVMH binding dicts (32 MB / 256)
  ``server._TILE_PNG_CACHE``         — per-tile PNG bytes (128 MB / 256)
  ``server._SUBDIVIDE_CACHE``        — per-(model, subdivision) tessellated meshes
  ``server._NJM_DECOMPRESS_CACHE``   — decompressed NJM blobs (32 MB)

Each layer keys on something the layer below can't see:
  - PRS cache:    (path, mtime_ns, inner_name) → bytes
  - parse cache:  (path, mtime_ns, size, inner_name?, parser_id) → parsed
  - skinned:      (parser_id, path, mtime_ns, size, inner_name) → dict
  - binding:      (path, mtime_ns, size, outer_ext, inner_name) → dict
  - tile_png:     (path, mtime_ns, size, tile_idx, request_path) → bytes
  - subdivide:    (model_path, level)
The PRS layer feeds the parse layer (parse cache calls into the parser
that consumes already-decompressed bytes). They invalidate independently
on mtime change because each layer's key includes ``mtime_ns``.

Key design: when callers know a stable identity (path + mtime + size +
inner-name) we use that as the cache key — zero hashing cost. When they
don't, ``cached_call`` falls back to ``hashlib.sha1`` of the input bytes
(roughly 5 MB/ms — still negligible vs. the 1.1 s parse cost).

Disk cache layout: ``cache/parse_cache/v<schema>/<sha2>.pkl`` with
schema-versioned subdirs so an editor upgrade that changes the dataclass
shape invalidates old caches automatically (bump
``_DISK_CACHE_SCHEMA``).
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from . import xj as _xj_mod
from . import xj_descriptor as _xj_desc_mod


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default 256 MB total cache footprint. Each parsed dragon-class model is
# 5-20 MB of Python object overhead (list[XjMesh] with thousands of
# vertices, each a 3-tuple + 3-tuple + 2-tuple + int dataclass), so 256
# MB holds ~15-50 dragons.  Users on low-RAM systems should lower this
# via PSO_PARSE_CACHE_MB (set to 0 to disable in-memory caching entirely
# — the disk cache continues to operate).
PARSE_CACHE_MAX_BYTES = int(
    os.environ.get("PSO_PARSE_CACHE_MB", "256")
) * 1024 * 1024

# Per-entry hard limit. A single parse result above this size still gets
# served (we always keep one entry even if oversize) but we won't try to
# stash it on disk — pickling a 100+ MB dataclass tree is slow and the
# disk hit is no longer a clear win.
_DISK_PERSIST_MAX_BYTES = 32 * 1024 * 1024

# Bump when the in-memory dataclass shapes change in a way that breaks
# pickle round-trip. A schema bump invalidates every prior on-disk pkl
# without manual cleanup — the lookup just won't find any v<old> files.
# v2 (2026-06-20): XjVertex gained a `color` RGBA field — old v1 pickles
# unpickle WITHOUT it and crash `_xj_meshes_to_payload`'s v.color read.
# v3 (2026-06-21): the descriptor-XJ parser now POPULATES XjMesh.blend_mode
# / alpha_blend from the type-2 (src_alpha,dst_alpha) material entry
# (additive FX — e.g. bm_eff_ice). v2 pickles carry the old default
# blend_mode="none" for every descriptor mesh, so effect models would keep
# rendering dark-opaque off a stale pkl until this bump forces a re-parse.
_DISK_CACHE_SCHEMA = 3

# Disk cache root. Created on first write; the ``ensure_cache_dir``
# helper in the server points us at the real location.
_disk_cache_dir: Optional[Path] = None


def configure(*, cache_dir: Optional[Path] = None,
              max_bytes: Optional[int] = None) -> None:
    """Optionally override defaults at process start.

    ``cache_dir`` selects the on-disk parse cache root. The server calls
    this from startup so the cache lands under ``<repo>/cache/`` rather
    than the per-user temp dir. ``max_bytes`` overrides the in-memory
    cap (typically read from PSO_PARSE_CACHE_MB env var instead).
    """
    global _disk_cache_dir, PARSE_CACHE_MAX_BYTES
    if cache_dir is not None:
        _disk_cache_dir = Path(cache_dir).resolve()
    if max_bytes is not None:
        PARSE_CACHE_MAX_BYTES = int(max_bytes)


def _resolve_disk_dir() -> Optional[Path]:
    """Return the disk cache root, creating the schema-versioned subdir.

    Returns None if disk persistence is disabled (no cache_dir set and no
    PSO_DISABLE_DISK_PARSE_CACHE override). Errors during directory
    creation degrade gracefully — the in-memory cache still works.
    """
    if os.environ.get("PSO_DISABLE_DISK_PARSE_CACHE", "0") in ("1", "true", "True"):
        return None
    base = _disk_cache_dir
    if base is None:
        return None
    try:
        d = base / f"v{_DISK_CACHE_SCHEMA}"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError as e:
        log.warning("parse_cache: disk dir creation failed at %s: %s", base, e)
        return None


# ---------------------------------------------------------------------------
# In-memory LRU
# ---------------------------------------------------------------------------
# Value shape: (parsed_object, byte_estimate, hit_count, last_access_ts)
# ``parsed_object`` is whatever the wrapped parser returned (list[XjMesh],
# list[XjBone], or tuple). ``byte_estimate`` is the pickled size — a
# tighter upper bound on real RAM use than ``sys.getsizeof`` (which only
# walks one level deep). We pickle once at insert and keep the bytes
# count for the eviction loop.

_PARSE_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_PARSE_CACHE_LOCK = threading.Lock()
_PARSE_CACHE_BYTES = 0
# Counters surfaced by the /api/parse_cache/stats endpoint.
_HITS_INMEMORY = 0
_HITS_DISK = 0
_MISSES = 0
# Map of cache_key → number of times THIS specific model was hit. Used
# by /api/parse_cache/stats "top entries" so the user can see whether
# the dragon really is the top consumer.
_KEY_HIT_COUNTS: "dict[tuple, int]" = {}


def _purge_until_under_cap_locked() -> None:
    """Evict LRU entries until total bytes <= cap.

    Called with ``_PARSE_CACHE_LOCK`` held. We always keep at least one
    entry so the working set survives even when a single parse exceeds
    the cap (the alternative is to spend forever evicting the entry we
    just inserted).
    """
    global _PARSE_CACHE_BYTES
    while (_PARSE_CACHE_BYTES > PARSE_CACHE_MAX_BYTES
           and len(_PARSE_CACHE) > 1):
        try:
            _evicted_key, value = _PARSE_CACHE.popitem(last=False)
        except KeyError:
            break
        _PARSE_CACHE_BYTES -= int(value[1])


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------
# We pickle (parsed_object, key_repr) so a corrupted file's key mismatch
# is caught at load time rather than serving stale data. The filename
# is the sha2 of (key_repr || schema), giving us collision-resistant
# stable names without leaking path strings to the filesystem.

def _disk_path_for_key(key: tuple, base: Path) -> Path:
    """Compute on-disk pickle path for a cache key.

    The key gets canonicalised to bytes via ``repr()`` then hashed with
    sha256. We don't truncate — the full 64-char hex name keeps the
    accidental-collision rate at vanishing levels (we have ~660
    distinct models max).
    """
    h = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return base / f"{h}.pkl"


def _try_load_from_disk(key: tuple) -> Optional[Tuple[Any, int]]:
    """Read a pickled parse result from disk, or None on miss/corrupt.

    Returns (parsed, byte_estimate) on hit. Corrupted files are deleted
    in-place so the next parse can repopulate cleanly — we never raise
    out of this function (a disk problem must not break the parse).
    """
    base = _resolve_disk_dir()
    if base is None:
        return None
    p = _disk_path_for_key(key, base)
    if not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            stored_key, parsed = pickle.load(f)
    except (pickle.PickleError, EOFError, ValueError, OSError) as e:
        log.warning("parse_cache: corrupt pickle %s removed: %s", p.name, e)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    # Defensive sanity-check — schema/version mismatch landed somehow.
    if stored_key != key:
        log.warning("parse_cache: pickle key mismatch on %s; deleting", p.name)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    return parsed, int(size)


def _try_write_to_disk(key: tuple, parsed: Any, byte_estimate: int) -> None:
    """Persist a parse result to disk; silent no-op on any error.

    Skips entries above ``_DISK_PERSIST_MAX_BYTES`` — pickling a >32 MB
    dataclass tree takes ~hundreds of ms which would blow the cold-parse
    budget for a marginal win on disk hits.

    Writes go atomic-rename: tmp → final, so a kill-9 mid-write can't
    leave a half-pickle that we'd subsequently mistake for a valid
    cache entry.
    """
    if byte_estimate > _DISK_PERSIST_MAX_BYTES:
        return
    base = _resolve_disk_dir()
    if base is None:
        return
    final = _disk_path_for_key(key, base)
    tmp = final.with_suffix(".tmp")
    try:
        with tmp.open("wb") as f:
            pickle.dump((key, parsed), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, final)
    except (OSError, pickle.PickleError) as e:
        log.warning("parse_cache: disk write failed for %s: %s", final.name, e)
        try:
            tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Core entry point: cached_call
# ---------------------------------------------------------------------------

def _make_key(parser_id: str,
              file_key: Optional[Tuple[Any, ...]],
              data: Optional[bytes]) -> tuple:
    """Build the in-cache lookup key for one parse call.

    ``file_key`` is preferred — when the caller has a stable
    (path, mtime_ns, size, inner_name?) tuple we use that directly.
    Falling through to a sha1 of the input bytes keeps the cache correct
    for synthetic/in-memory inputs (e.g. tests, future inline-buffer
    callers) but costs ~5 ms per 4 MB of input so we avoid it on the
    hot path.
    """
    if file_key is not None:
        return ("fkey", parser_id, file_key)
    if data is None:
        raise ValueError("parse_cache._make_key: need file_key or data")
    digest = hashlib.sha1(data).hexdigest()
    return ("hash", parser_id, digest, len(data))


def cached_call(
    parser_id: str,
    parser_fn: Callable[..., Any],
    *,
    data: bytes,
    file_key: Optional[Tuple[Any, ...]] = None,
    extra_kwargs: Optional[dict] = None,
) -> Any:
    """Run ``parser_fn(data, **extra_kwargs)`` with parse-result caching.

    Lookup order:
      1. In-memory LRU keyed on (parser_id, file_key OR sha1).
      2. On-disk pkl under ``cache/parse_cache/v<schema>/``.
      3. Cold parse via ``parser_fn``.

    On a cold parse we always populate the in-memory cache; we ALSO
    persist to disk if the result is below the per-entry persist cap.
    Eviction is LRU-by-bytes — we keep popping the least recently used
    entry until we're back under the configured cap.

    ``extra_kwargs`` is passed verbatim to the wrapped parser
    (e.g. ``ignore_hide=False``). It does NOT participate in the cache
    key — the small flag space (one bool today) and the cache-isolation
    constraints make it not worth the key bloat. If a caller flips
    ``ignore_hide``, the cache will (correctly) return whatever was
    cached first; tests should call ``cache_clear()`` between runs.
    """
    global _HITS_INMEMORY, _HITS_DISK, _MISSES, _PARSE_CACHE_BYTES
    key = _make_key(parser_id, file_key, data)

    # --- L1: in-memory LRU
    with _PARSE_CACHE_LOCK:
        ent = _PARSE_CACHE.get(key)
        if ent is not None:
            _PARSE_CACHE.move_to_end(key)
            ent[2] = ent[2] + 1                       # bump hit count
            ent[3] = time.time()
            _HITS_INMEMORY += 1
            _KEY_HIT_COUNTS[key] = _KEY_HIT_COUNTS.get(key, 0) + 1
            return ent[0]

    # --- L2: on-disk pickle
    disk_hit = _try_load_from_disk(key)
    if disk_hit is not None:
        parsed, byte_estimate = disk_hit
        with _PARSE_CACHE_LOCK:
            ent = _PARSE_CACHE.get(key)
            if ent is None:
                _PARSE_CACHE[key] = [parsed, byte_estimate, 1, time.time()]
                _PARSE_CACHE_BYTES += byte_estimate
                _purge_until_under_cap_locked()
                _HITS_DISK += 1
            else:
                # Race: another caller landed first.
                _PARSE_CACHE.move_to_end(key)
                ent[2] += 1
                ent[3] = time.time()
                parsed = ent[0]
                _HITS_INMEMORY += 1
            _KEY_HIT_COUNTS[key] = _KEY_HIT_COUNTS.get(key, 0) + 1
        return parsed

    # --- Cold parse (the work this whole module exists to avoid)
    kw = extra_kwargs or {}
    parsed = parser_fn(data, **kw)

    # Estimate footprint via pickle size (a tight upper bound on real
    # RAM use). For the 4 MB dragon NJ this is ~5-10 MB of pickle, which
    # tracks the actual Python object cost reasonably well. We also
    # use the pickled bytes for the disk write so we only serialise once.
    try:
        pickled = pickle.dumps(parsed, protocol=pickle.HIGHEST_PROTOCOL)
        byte_estimate = len(pickled)
    except (pickle.PickleError, TypeError):
        # Unpicklable result — keep in-memory only.
        pickled = None
        byte_estimate = 1024 * 1024  # conservative 1 MB placeholder

    with _PARSE_CACHE_LOCK:
        # Race-safe: re-check before insert.
        ent = _PARSE_CACHE.get(key)
        if ent is None:
            _PARSE_CACHE[key] = [parsed, byte_estimate, 1, time.time()]
            _PARSE_CACHE_BYTES += byte_estimate
            _purge_until_under_cap_locked()
            _MISSES += 1
        else:
            _PARSE_CACHE.move_to_end(key)
            ent[2] += 1
            ent[3] = time.time()
            parsed = ent[0]
            _HITS_INMEMORY += 1
        _KEY_HIT_COUNTS[key] = _KEY_HIT_COUNTS.get(key, 0) + 1

    # Disk persist (outside the lock — disk I/O could be slow and we
    # don't want to serialise other cache reads behind it). Honor the
    # per-entry size cap.
    if pickled is not None and byte_estimate <= _DISK_PERSIST_MAX_BYTES:
        # We already have the pickled bytes from the size estimate, so
        # write them directly rather than re-pickling inside _try_write.
        base = _resolve_disk_dir()
        if base is not None:
            final = _disk_path_for_key(key, base)
            tmp = final.with_suffix(".tmp")
            try:
                # The on-disk format is (key, parsed) — re-pickle to
                # include the key-tag for corruption detection. The
                # estimate-pickle didn't include the key.
                with tmp.open("wb") as f:
                    pickle.dump((key, parsed), f,
                                protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, final)
            except (OSError, pickle.PickleError) as e:
                log.warning("parse_cache: disk write failed for %s: %s",
                            final.name, e)
                try:
                    tmp.unlink()
                except OSError:
                    pass

    return parsed


# ---------------------------------------------------------------------------
# Convenience wrappers — one per public parser entry point
# ---------------------------------------------------------------------------

def parse_nj_file_cached(
    data: bytes,
    *,
    file_key: Optional[Tuple[Any, ...]] = None,
    ignore_hide: Optional[bool] = None,
) -> list:
    """Cached ``formats.xj.parse_nj_file``."""
    return cached_call(
        "nj_file",
        _xj_mod.parse_nj_file,
        data=data,
        file_key=file_key,
        extra_kwargs={"ignore_hide": ignore_hide},
    )


def parse_xj_file_cached(
    data: bytes,
    *,
    file_key: Optional[Tuple[Any, ...]] = None,
    ignore_hide: Optional[bool] = None,
) -> list:
    """Cached ``formats.xj_descriptor.parse_xj_file``."""
    return cached_call(
        "xj_descriptor_file",
        _xj_desc_mod.parse_xj_file,
        data=data,
        file_key=file_key,
        extra_kwargs={"ignore_hide": ignore_hide},
    )


def parse_skeleton_cached(
    data: bytes,
    *,
    file_key: Optional[Tuple[Any, ...]] = None,
) -> list:
    """Cached ``formats.xj.parse_skeleton``.

    Unlike the mesh parsers this one is fast (~10-50 ms even on dragon)
    because it only walks the 52-byte mesh-tree nodes. We still cache it
    because the model-bundle endpoint hits it 2-3 times per open and
    every shaved millisecond compounds across the page-load critical path.
    """
    return cached_call(
        "nj_skeleton",
        _xj_mod.parse_skeleton,
        data=data,
        file_key=file_key,
    )


def parse_nj_skinned_cached(
    data: bytes,
    *,
    file_key: Optional[Tuple[Any, ...]] = None,
    ignore_hide: Optional[bool] = None,
) -> tuple:
    """Cached ``formats.xj.parse_nj_skinned``.

    Returns ``(meshes, bones)``. The skinned path is what the Phase 0.5
    perf agent flagged as dragon-slow (the BONE bake walks every node's
    vertex chunk twice — once to populate slots, once to emit strips).
    """
    return cached_call(
        "nj_skinned",
        _xj_mod.parse_nj_skinned,
        data=data,
        file_key=file_key,
        extra_kwargs={"ignore_hide": ignore_hide},
    )


# ---------------------------------------------------------------------------
# Stats / clear
# ---------------------------------------------------------------------------

def cache_stats() -> dict:
    """Return a snapshot of cache health for the /api/parse_cache/stats route.

    Shape:
      ``entries``         - in-memory LRU entry count
      ``bytes``           - in-memory total (pickled-size estimate)
      ``max_bytes``       - configured cap
      ``disk_entries``    - on-disk pickle count (None if disk disabled)
      ``disk_bytes``      - on-disk total bytes (None if disk disabled)
      ``hits_inmemory``   - lifetime L1 hit count
      ``hits_disk``       - lifetime L2 hit count
      ``misses``          - lifetime cold parse count
      ``hit_rate``        - (hits_inmemory + hits_disk) / total
      ``top_entries``     - top-10 entries by hit count
      ``schema``          - on-disk schema version
    """
    with _PARSE_CACHE_LOCK:
        entries = len(_PARSE_CACHE)
        total = _PARSE_CACHE_BYTES
        # Build top-10 by hit count (live counters from _KEY_HIT_COUNTS).
        top: list = []
        for key, hits in sorted(_KEY_HIT_COUNTS.items(),
                                key=lambda kv: kv[1], reverse=True)[:10]:
            ent = _PARSE_CACHE.get(key)
            top.append({
                "key": _stringify_key(key),
                "hits": hits,
                "bytes": int(ent[1]) if ent else 0,
                "in_memory": ent is not None,
            })
        h_mem = _HITS_INMEMORY
        h_disk = _HITS_DISK
        miss = _MISSES

    # Disk usage — outside the lock; this is just stat-walks.
    disk_entries: Optional[int] = None
    disk_bytes: Optional[int] = None
    base = _resolve_disk_dir()
    if base is not None and base.is_dir():
        try:
            disk_entries = 0
            disk_bytes = 0
            for child in base.iterdir():
                if child.is_file() and child.suffix == ".pkl":
                    disk_entries += 1
                    try:
                        disk_bytes += child.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass

    total_calls = h_mem + h_disk + miss
    hit_rate = (h_mem + h_disk) / total_calls if total_calls else 0.0
    return {
        "entries": entries,
        "bytes": total,
        "max_bytes": PARSE_CACHE_MAX_BYTES,
        "disk_entries": disk_entries,
        "disk_bytes": disk_bytes,
        "hits_inmemory": h_mem,
        "hits_disk": h_disk,
        "misses": miss,
        "hit_rate": hit_rate,
        "top_entries": top,
        "schema": _DISK_CACHE_SCHEMA,
    }


def _stringify_key(key: tuple) -> str:
    """Render a cache key as a short readable string for stats output.

    Avoids leaking absolute paths verbatim into the JSON response —
    truncates the path component to its basename.
    """
    if not key:
        return "<empty>"
    kind = key[0]
    if kind == "fkey" and len(key) >= 3:
        parser_id = key[1]
        fkey = key[2]
        path_basename = ""
        if isinstance(fkey, tuple) and fkey:
            head = fkey[0]
            if isinstance(head, str):
                # Last segment in a Windows or POSIX path.
                path_basename = head.replace("\\", "/").rsplit("/", 1)[-1]
        rest = "..." if isinstance(fkey, tuple) and len(fkey) > 1 else ""
        return f"{parser_id}:{path_basename}{rest}"
    if kind == "hash" and len(key) >= 3:
        return f"{key[1]}:hash{str(key[2])[:12]}"
    return str(key)[:80]


def cache_clear(*, drop_disk: bool = True) -> dict:
    """Drop the in-memory cache, and (by default) the on-disk pickles too.

    Returns a small summary so callers can confirm what got cleared.
    Tests that want to keep the disk cache around (e.g. to exercise the
    L2 hit path) pass ``drop_disk=False``.
    """
    global _PARSE_CACHE_BYTES, _HITS_INMEMORY, _HITS_DISK, _MISSES
    with _PARSE_CACHE_LOCK:
        cleared_entries = len(_PARSE_CACHE)
        cleared_bytes = _PARSE_CACHE_BYTES
        _PARSE_CACHE.clear()
        _PARSE_CACHE_BYTES = 0
        _HITS_INMEMORY = 0
        _HITS_DISK = 0
        _MISSES = 0
        _KEY_HIT_COUNTS.clear()

    disk_files = 0
    disk_bytes_freed = 0
    if drop_disk:
        base = _resolve_disk_dir()
        if base is not None and base.is_dir():
            try:
                for child in base.iterdir():
                    if child.is_file() and child.suffix in (".pkl", ".tmp"):
                        try:
                            sz = child.stat().st_size
                        except OSError:
                            sz = 0
                        try:
                            child.unlink()
                            disk_files += 1
                            disk_bytes_freed += sz
                        except OSError:
                            pass
            except OSError:
                pass

    return {
        "cleared_entries": cleared_entries,
        "cleared_bytes": cleared_bytes,
        "cleared_disk_files": disk_files,
        "cleared_disk_bytes": disk_bytes_freed,
    }


__all__ = [
    "PARSE_CACHE_MAX_BYTES",
    "configure",
    "cached_call",
    "parse_nj_file_cached",
    "parse_xj_file_cached",
    "parse_skeleton_cached",
    "parse_nj_skinned_cached",
    "cache_stats",
    "cache_clear",
]
