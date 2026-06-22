"""Global cross-archive texture index for the PSOBB Texture Editor.

Builds a dict keyed by texture name (the value carried by an NJTL slot)
mapping to every (archive, inner_name, xvr_index) location that supplies
the named texture. Used by ``server.py::_build_model_texture_binding``
to resolve NJTL references whose texture lives in a SIBLING archive
rather than the model's own inline XVM.

Sources walked
--------------

* **BML archives** — each BML's textured ``.nj`` / ``.xj`` inners carry
  an NJTL chunk with one or more named slots. The matching XVMH archive
  (sibling of the inner) holds the same number of XVRT records in
  positional order. We index every (name -> location) pair.

* **AFS archives** — PSOBB packs player textures (``pl[A-Z]tex.afs``)
  as flat AFS archives where every entry is a single bare ``XVRT``
  record. Each entry has NO embedded name, so we synthesise one per
  slot with a stable ``<archive_stem>_<NNNN>`` form. This gives the
  cross-archive resolver a uniform name -> location map across both
  source kinds; the consumer chooses how to bind based on the
  ``TextureLocation.kind`` discriminator.

  AFS archives that wrap PRS-compressed XVMH blobs (e.g.
  ``ItemTexture.afs``) are also walked: each inner XVM contributes one
  record per XVR, named ``<archive_stem>_<NNNN>_<MMMM>`` where the inner
  index is NNNN and the XVR index inside the XVM is MMMM.

PSOBB.IO ships ~60 NJTL refs (across ~119 affected models) whose name
appears in some OTHER BML's NJTL but not in the host's inline XVM, plus
~140 player class body / hair / head texture slots that resolve only
through ``pl[A-Z]tex.afs``. The runtime resolves both via its global
texture cache; the editor needs the same lookup table so the model
viewer doesn't render those submeshes with the wrong (or missing) tile.

Design notes
------------

* The index is built lazily on first call to ``get_texture_index``,
  cached in-memory on the module, and persisted to
  ``cache/texture_index.json`` so cold starts are fast (hot cache: ~5
  ms; cold: ~10 s for the full PSOBB.IO walk).
* Cache key: the data directory's path + the max mtime of any
  ``*.bml`` OR ``*.afs`` in that directory. If a BML / AFS is rebuilt
  out-of-tree the cache invalidates automatically.
* The index does NOT carry texture bytes — only locations. The actual
  texture extraction happens lazily at binding time via
  ``extract_bml_texture`` (BML) or ``afs_reader.materialize_inner``
  (AFS).
* Atomic write: we write to a tempfile then rename, so a partially-
  built cache cannot poison the next run.

Public API
----------

``TextureLocation`` — dataclass.
``build_texture_index(data_dir)`` — fresh BML-only build (no cache).
``index_afs_archives(data_dir, archive_filter=None)`` — fresh AFS-only walk.
``build_global_texture_index(data_dir)`` — fresh combined BML+AFS build.
``get_texture_index(data_dir)`` — cached entry point used by server.
``lookup(data_dir, name)`` — convenience name lookup.
``lookup_player_class_textures(data_dir, model_filename)`` — positional
    fallback for player NJ models that ship without an NJTL chunk.
"""
from __future__ import annotations

import json
import logging
import os
import re
import struct
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence

from formats.bml import parse_bml, decompress_prs_cached
from formats.iff import parse_iff
from formats.njtl import find_and_parse_njtl

log = logging.getLogger("psobb_editor.texture_index")


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class TextureLocation:
    """Where a named texture can be found inside a PSOBB archive.

    Attributes
    ----------
    bml_name:
        BML/AFS filename relative to the data dir
        (e.g. ``bm_boss3_volopt.bml`` or ``plAtex.afs``). We store the
        basename, not the absolute path, so the index is portable
        between machines.
    inner_name:
        Entry name inside the archive whose paired XVM/XVR holds this
        texture (e.g. ``boss3_v.xj`` for BML or ``0042_plAtex_0042.xvr``
        for AFS).
    xvr_index:
        Position in the XVMH archive (0-based). Lines up with the NJTL
        slot of the same model that registered the name. For AFS-XVR
        entries (one record per inner) this is always 0.
    kind:
        ``"bml"`` for BML inline-XVM locations, ``"afs"`` for AFS
        entries. Defaults to ``"bml"`` so the field is optional in
        legacy callers and serialised payloads (a missing ``kind`` =>
        ``"bml"``).
    archive:
        Synonym for ``bml_name`` carried for AFS entries so callers can
        read the more accurate field name without a discriminator
        check. Always set to the same value as ``bml_name``.
    inner_index:
        AFS-only: 0-based blob index inside the AFS archive. ``-1`` for
        BML entries (which use ``inner_name`` instead). Used by the
        frontend to synthesise the ``<archive>#NNNN_<basename>`` URL.
    """
    bml_name: str
    inner_name: str
    xvr_index: int
    kind: str = "bml"
    archive: str = ""
    inner_index: int = -1

    def __post_init__(self) -> None:
        # Mirror bml_name into archive so AFS callers have a clear
        # accessor without losing the legacy field name.
        if not self.archive:
            self.archive = self.bml_name


# ---------------------------------------------------------------------------
# Cache state
# ---------------------------------------------------------------------------


_INDEX_LOCK = Lock()
_INDEX_CACHE: Optional[Dict[str, List[TextureLocation]]] = None
_INDEX_CACHE_KEY: Optional[tuple] = None


# Persistent cache lives next to the editor root so it survives across
# server restarts.
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_CACHE_FILE = _CACHE_DIR / "texture_index.json"


# Player-class AFS naming convention. ``plAbdy00.nj`` => texture archive
# ``plAtex.afs``. The single capital letter at index 2 of the model
# stem identifies the class. Any model whose stem matches
# ``pl[A-Z]<rest>`` is assumed to bind positionally against the matching
# ``pl<class>tex.afs`` when its NJTL is missing or empty.
#
# The same convention applies to inners packed inside ``pl[A-Z]nj.bml``
# (e.g. ``plAnj.bml#plAbdy00.nj``) — the BML wrapper is just a packing
# container, the underlying model still binds against ``pl[A-Z]tex.afs``.
_PLAYER_NJ_RE = re.compile(r"^pl([A-Z])(bdy|hai|hed|cap|fac|sho|hen|leg|frm|nai|nan)", re.IGNORECASE)


# ItemModel → ItemTexture pairing. PSOBB items (weapons, mags, units,
# shields, frames) live in ``ItemModel.afs`` (Ep1-3) and
# ``ItemModelEp4.afs`` (Ep4 add-ons). Each model is a chunk-Ninja inner
# with an NJTL chunk that names its textures, but the texture XVMs live
# in a SIBLING archive (``ItemTexture.afs`` / ``ItemTextureEp4.afs``),
# bound POSITIONALLY: ItemModel.afs#NNNN's K-th NJTL slot resolves to
# ItemTexture.afs#NNNN's K-th XVR record. The runtime resolves this by
# (archive, inner_index, xvr_index) tuple via PSOBB's global texture
# registrar; the editor mirrors the same lookup table.
_ITEM_MODEL_TO_TEXTURE: Dict[str, str] = {
    "ItemModel.afs": "ItemTexture.afs",
    "ItemModelEp4.afs": "ItemTextureEp4.afs",
}


# Per-BML XVMH-bearing inner cache, keyed on (bml_basename, mtime_ns).
# Maps to a list of (inner_name, xvr_count) pairs in the order the XVMs
# appear in the BML. Used by the same-BML cross-inner positional
# fallback in ``_build_model_texture_binding`` — when a sibling inner
# has no inline XVM but other inners IN THE SAME BML do, we treat the
# missing entries as positional references into the largest sibling
# XVMH archive.
#
# Cache lives in-process only; rebuild on demand. The work is one
# extract_bml_texture call per inner-with-tex per BML, paid once per
# server start (or when a BML is rebuilt out-of-tree).
_BML_XVMH_CACHE: Dict[tuple, List[tuple]] = {}
_BML_XVMH_LOCK = Lock()


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _cache_key(data_dir: Path) -> tuple:
    """Compute a (path, max_mtime) tuple for cache invalidation.

    Walks BOTH ``*.bml`` and ``*.afs`` under ``data_dir`` so the cache
    invalidates when any source changes — a BML rebuild out-of-tree or
    an AFS swap both trigger a fresh index build.
    """
    max_mtime = 0
    for pattern in ("*.bml", "*.afs"):
        for f in data_dir.glob(pattern):
            try:
                m = f.stat().st_mtime_ns
            except OSError:
                continue
            if m > max_mtime:
                max_mtime = m
    return (str(data_dir.resolve()), max_mtime)


def _index_one_bml(
    bml_path: Path,
    out: Dict[str, List[TextureLocation]],
) -> None:
    """Walk one BML and contribute its NJTL→XVR mappings to ``out``."""
    try:
        buf = bml_path.read_bytes()
    except OSError as e:
        log.warning("texture_index: cannot read %s: %s", bml_path, e)
        return
    try:
        entries = parse_bml(buf)
    except Exception as e:
        log.warning("texture_index: parse_bml failed for %s: %s", bml_path, e)
        return

    try:
        st_mtime_ns = bml_path.stat().st_mtime_ns
    except OSError:
        st_mtime_ns = 0

    for ent in entries:
        # Only inner entries that COULD carry NJTL refs.
        ext_lower = ent.name.lower()
        if not (ext_lower.endswith(".nj") or ext_lower.endswith(".xj")):
            continue
        # Skip entries without paired textures — they can't *provide*
        # textures to anyone (only consume them). Note: an entry with
        # has_texture=False can still reference textures from another
        # BML; we want PROVIDERS here, so the XVM bearing entries are
        # the only ones we index.
        if not ent.has_texture or ent.tex_size_compressed == 0:
            continue
        try:
            inner_blob = decompress_prs_cached(
                bml_path, st_mtime_ns, ent.name,
                lambda: bytes(buf[ent.offset:ent.offset + ent.size_compressed]),
            )
        except Exception as e:
            log.debug("texture_index: PRS fail %s#%s: %s", bml_path.name, ent.name, e)
            continue
        # Parse the NJTL chunk to enumerate the texture names this
        # entry provides via its inline XVM.
        try:
            njtl_entries = find_and_parse_njtl(inner_blob) or []
        except Exception as e:
            log.debug("texture_index: NJTL parse fail %s#%s: %s",
                      bml_path.name, ent.name, e)
            continue
        if not njtl_entries:
            continue
        # NjtlEntry positional convention: slot i ↔ XVR record i in the
        # paired XVMH archive.
        for nj in njtl_entries:
            name = (nj.name or "").strip()
            if not name:
                continue
            out.setdefault(name, []).append(
                TextureLocation(
                    bml_name=bml_path.name,
                    inner_name=ent.name,
                    xvr_index=int(nj.slot),
                    kind="bml",
                )
            )


# ---------------------------------------------------------------------------
# AFS walk
# ---------------------------------------------------------------------------


def _afs_synth_name(archive_stem: str, idx: int, sub_idx: Optional[int] = None) -> str:
    """Synthesise a stable name for an AFS-inner texture record.

    Form: ``<stem>_<NNNN>`` for one-record-per-blob archives (pl?tex.afs);
    ``<stem>_<NNNN>_<MMMM>`` when the blob itself wraps an XVM with multiple
    XVR records (Item*.afs).
    """
    base = f"{archive_stem}_{idx:04d}"
    if sub_idx is None:
        return base
    return f"{base}_{sub_idx:04d}"


def _afs_inner_logical_name(archive_stem: str, idx: int, ext: Optional[str]) -> str:
    """Match the convention used by ``server.py::_parse_afs_inner_name``.

    The server-side route ``<archive>#NNNN_<basename>.<ext>`` uses a
    leading 4-digit index + ``_`` + sniffed-extension basename. We
    mirror that so a frontend URL built from ``TextureLocation`` round-
    trips through ``_materialize_inner_for_extract``.
    """
    suffix = ext or ""
    return f"{idx:04d}_{archive_stem}_{idx:04d}{suffix}"


def _count_xvmh_records(xvm: bytes) -> int:
    """Count texture records inside a decompressed texture archive.

    Magic-routes (2026-06-21, psov2 multivariant) so Dreamcast (PVMH) and
    GameCube (GVMH) inline archives are counted as texture PROVIDERS, not
    silently skipped:

        XVMH  -> declared u32 LE count at +0x08 (Xbox / PSOBB.IO)
        PVMH  -> declared u16 LE count at +0x0A (Dreamcast)
        GVMH  -> declared u16 BE count at +0x0A (GameCube)

    Returns 0 for unrecognised / malformed input; never raises. Used by
    the deep-walk + sibling-fallback paths so we can emit one row per
    sub-record of a textured inner without parsing the whole record table.
    """
    if not xvm or len(xvm) < 0x10:
        return 0
    head = xvm[:4]
    try:
        if head == b"XVMH":
            if len(xvm) < 0x40:
                return 0
            n = struct.unpack_from("<I", xvm, 0x08)[0]
        elif head == b"PVMH":
            n = struct.unpack_from("<H", xvm, 0x0A)[0]
        elif head == b"GVMH":
            n = struct.unpack_from(">H", xvm, 0x0A)[0]
        else:
            return 0
    except struct.error:
        return 0
    # Sanity-cap to defend against malformed counts. PSOBB items rarely
    # carry more than 16 textures per inner; 256 is a generous ceiling.
    if 0 < n < 256:
        return int(n)
    return 0


def _index_one_afs(
    afs_path: Path,
    out: Dict[str, List[TextureLocation]],
    *,
    deep_xvm: bool = False,
    cache_dir: Optional[Path] = None,
) -> None:
    """Walk one AFS archive and contribute texture locations to ``out``.

    Two layouts are recognised:

    * Bare-XVR (``pl[A-Z]tex.afs``): each blob's first 4 bytes are
      ``XVRT``. The blob IS one texture; ``xvr_index`` is always 0 and
      the synthesised name encodes the AFS index.
    * Packed-XVM (``ItemTexture.afs``, ``ItemTextureEp4.afs``): each
      blob is an ``XVMH`` archive (PRS-compressed). When ``deep_xvm`` is
      False (the default) we record only the OUTER (archive,
      inner_index) and let the consumer decompress on demand —
      ``xvr_index`` is left at -1 to signal "the inner is itself a
      multi-record archive". When ``deep_xvm`` is True we additionally
      decompress each blob and emit one extra row per XVR record so the
      positional resolver can index by sub-position. Deep walks are
      reserved for the ItemTexture archives — the decompression cost is
      tolerable there (~370 + 502 inners) and the extra rows are how we
      give ItemModel cross_afs lookup its per-NJTL-slot precision.

    Other AFS layouts (NJ-only model archives, audio) contribute
    nothing — we only emit rows whose blob magic matches a texture
    container.
    """
    # Local import keeps the module load order clean: afs.py and
    # afs_reader.py both depend on bml.py; importing them at top level
    # would create a small cycle.
    from formats import afs as afs_mod
    try:
        buf = afs_path.read_bytes()
    except OSError as e:
        log.warning("texture_index: cannot read %s: %s", afs_path, e)
        return
    try:
        blobs = afs_mod.parse_afs(buf)
    except Exception as e:
        log.warning("texture_index: parse_afs failed for %s: %s", afs_path, e)
        return

    archive_stem = afs_path.stem
    archive_name = afs_path.name

    # Lazy import of afs_reader for the deep-walk path: we only need it
    # when we have to materialize (and PRS-decompress) every inner blob.
    afs_reader = None
    if deep_xvm:
        try:
            from formats import afs_reader as _afs_reader  # noqa: F401
            afs_reader = _afs_reader
        except ImportError as e:  # pragma: no cover
            log.warning(
                "texture_index: afs_reader unavailable for deep walk of %s: %s",
                archive_name, e,
            )
            afs_reader = None

    def _emit_xvm_record(idx: int, n_xvr: int) -> None:
        """Emit the outer XVMH row + (optionally) per-XVR sub-rows.

        Outer row keeps the legacy ``xvr_index=-1`` shape so tooling that
        looks at the archive-keyed name (``ItemTexture_NNNN``) keeps
        working. Per-XVR rows use a 4-digit sub-index suffix
        (``ItemTexture_NNNN_MMMM``) so positional resolvers can do
        ``f"{stem}_{NNNN:04d}_{MMMM:04d}"`` lookups.
        """
        outer_name = _afs_synth_name(archive_stem, idx)
        outer_inner = _afs_inner_logical_name(archive_stem, idx, ".xvm")
        out.setdefault(outer_name, []).append(
            TextureLocation(
                bml_name=archive_name,
                inner_name=outer_inner,
                xvr_index=-1,
                kind="afs",
                archive=archive_name,
                inner_index=idx,
            )
        )
        for k in range(n_xvr):
            sub_name = _afs_synth_name(archive_stem, idx, k)
            out.setdefault(sub_name, []).append(
                TextureLocation(
                    bml_name=archive_name,
                    inner_name=outer_inner,
                    xvr_index=k,
                    kind="afs",
                    archive=archive_name,
                    inner_index=idx,
                )
            )

    for i, blob in enumerate(blobs):
        if not blob:
            continue
        head = blob[:4]
        if head == b"XVRT":
            # One texture per blob: name = synthesised slot string.
            name = _afs_synth_name(archive_stem, i)
            inner_logical = _afs_inner_logical_name(archive_stem, i, ".xvr")
            out.setdefault(name, []).append(
                TextureLocation(
                    bml_name=archive_name,
                    inner_name=inner_logical,
                    xvr_index=0,
                    kind="afs",
                    archive=archive_name,
                    inner_index=i,
                )
            )
            continue
        # PRS-wrapped XVMH (ItemTexture-style). The first byte is the PRS
        # control bitmap; ``XVMH`` typically appears verbatim within the
        # first ~32 bytes since PRS leaves long literal runs unencoded.
        # Direct (uncompressed) XVMH is rare but handled the same way.
        is_packed_xvm = b"XVMH" in blob[:64]
        is_direct_xvm = head == b"XVMH"
        if not (is_packed_xvm or is_direct_xvm):
            continue
        n_xvr = 0
        if deep_xvm and afs_reader is not None and cache_dir is not None:
            try:
                cache_path, _info = afs_reader.materialize_inner(
                    afs_path, i, cache_dir,
                )
                xvm_bytes = cache_path.read_bytes()
                n_xvr = _count_xvmh_records(xvm_bytes)
            except Exception as e:  # pragma: no cover - defensive
                log.debug(
                    "texture_index: deep walk of %s#%04d failed: %s",
                    archive_name, i, e,
                )
                n_xvr = 0
        _emit_xvm_record(i, n_xvr)


def index_afs_archives(
    data_dir: Path,
    archive_filter: Optional[Sequence[str]] = None,
    *,
    deep_xvm_archives: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[TextureLocation]]:
    """Walk every (or filtered) ``.afs`` in ``data_dir`` and return its index.

    Args
    ----
    data_dir:
        Directory holding the AFS archives (typically
        ``~/PSOBB.IO/data``).
    archive_filter:
        Optional iterable of basename glob patterns
        (e.g. ``["pl*.afs"]``). Default ``None`` walks every
        ``.afs``.
    deep_xvm_archives:
        Optional iterable of basenames whose XVMH-wrapping inners should
        be decompressed during the walk so per-XVR sub-rows are emitted.
        Defaults to ``("ItemTexture.afs", "ItemTextureEp4.afs")`` so
        weapon / item positional cross_afs lookup works out of the box.
        Pass an empty list to disable deep walking entirely.
    cache_dir:
        Where ``afs_reader.materialize_inner`` may stash decompressed
        XVMs during a deep walk. Defaults to ``cache/`` next to the
        editor root. Required only when ``deep_xvm_archives`` is non-
        empty.

    Returns
    -------
    Dict mapping synthesised texture-name -> list of TextureLocation.

    Does NOT consult the on-disk cache. Suitable for force-rebuild
    paths and for tests.
    """
    out: Dict[str, List[TextureLocation]] = {}
    if archive_filter is None:
        archives = sorted(data_dir.glob("*.afs"))
    else:
        seen: set[Path] = set()
        archives = []
        for pat in archive_filter:
            for p in sorted(data_dir.glob(pat)):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    archives.append(p)
    if deep_xvm_archives is None:
        deep_xvm_archives = ("ItemTexture.afs", "ItemTextureEp4.afs")
    deep_set = {name for name in deep_xvm_archives}
    if cache_dir is None:
        cache_dir = _CACHE_DIR
    log.info("texture_index: walking %d AFS archives in %s", len(archives), data_dir)
    for p in archives:
        deep = p.name in deep_set
        _index_one_afs(p, out, deep_xvm=deep, cache_dir=cache_dir if deep else None)
    for name in out:
        out[name].sort(key=lambda loc: (loc.bml_name, loc.inner_index, loc.xvr_index))
    log.info(
        "texture_index: %d unique AFS texture names indexed across %d archives",
        len(out), len(archives),
    )
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_texture_index(data_dir: Path) -> Dict[str, List[TextureLocation]]:
    """Walk every BML in ``data_dir`` and return a fresh BML-only index.

    Does NOT consult the on-disk cache. Suitable for force-rebuild
    paths and for tests.

    Note: this is the LEGACY builder — for the full cross-archive map
    (BML + AFS) call ``build_global_texture_index`` instead.
    """
    out: Dict[str, List[TextureLocation]] = {}
    bmls = sorted(data_dir.glob("*.bml"))
    log.info("texture_index: building from %d BMLs in %s", len(bmls), data_dir)
    for p in bmls:
        _index_one_bml(p, out)
    # Sort each location list by (bml_name, inner_name) for determinism.
    for name in out:
        out[name].sort(key=lambda loc: (loc.bml_name, loc.inner_name))
    log.info(
        "texture_index: %d unique texture names indexed across %d BMLs",
        len(out), len(bmls),
    )
    return out


def build_global_texture_index(data_dir: Path) -> Dict[str, List[TextureLocation]]:
    """Walk every BML AND every AFS in ``data_dir``; return the merged index.

    This is the single source-of-truth builder used by
    ``get_texture_index``. The two scans contribute disjoint key spaces:
    BML scans yield real NJTL names, AFS scans yield synthesised
    ``<archive_stem>_<NNNN>`` names. A name can appear in BOTH (if a
    user happens to register a texture with the same synthesised stem)
    — locations from both kinds are kept so the consumer can pick the
    in_bml vs cross_bml vs cross_afs path it wants.
    """
    out = build_texture_index(data_dir)
    afs_index = index_afs_archives(data_dir)
    for name, locs in afs_index.items():
        out.setdefault(name, []).extend(locs)
    # Re-sort each merged location list deterministically: BML rows
    # first (kind=='bml'), then AFS rows. This keeps the legacy
    # cross_bml fast-path identical to before for names that were
    # already in the BML index.
    for name in out:
        out[name].sort(key=lambda loc: (
            0 if loc.kind == "bml" else 1,
            loc.bml_name,
            loc.inner_index,
            loc.xvr_index,
        ))
    log.info(
        "texture_index: combined %d unique names (BML+AFS) in %s",
        len(out), data_dir,
    )
    return out


def _save_cache(
    index: Dict[str, List[TextureLocation]],
    cache_key: tuple,
) -> None:
    """Persist the index + key atomically."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Convert TextureLocation lists to plain dicts for JSON.
        serialised = {
            name: [asdict(loc) for loc in locs]
            for name, locs in index.items()
        }
        payload = {
            # v1: BML-only (no kind / archive / inner_index).
            # v2: BML + AFS, with kind / archive / inner_index — outer
            #     XVMH rows only (xvr_index = -1 for packed-XVM blobs).
            # v3: + deep walk of ItemTexture / ItemTextureEp4: per-XVR
            #     sub-rows (``<stem>_<NNNN>_<MMMM>``) so positional
            #     cross_afs lookup can pin a real XVR record.
            "version": 3,
            "key": [cache_key[0], cache_key[1]],
            "index": serialised,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(_CACHE_DIR),
            prefix=".texture_index_", suffix=".tmp", delete=False,
        ) as tf:
            tmpname = tf.name
            json.dump(payload, tf, ensure_ascii=False, indent=0)
        os.replace(tmpname, _CACHE_FILE)
    except Exception as e:  # pragma: no cover - cache is opportunistic
        log.warning("texture_index: cache save failed: %s", e)


def _load_cache(cache_key: tuple) -> Optional[Dict[str, List[TextureLocation]]]:
    """Try to load a cached index whose key matches ``cache_key``.

    Accepts both v1 (BML-only, no ``kind`` field) and v2 (BML+AFS, with
    ``kind``/``archive``/``inner_index``). v1 entries are upgraded
    in-flight by the dataclass defaults.
    """
    if not _CACHE_FILE.exists():
        return None
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:  # pragma: no cover
        log.warning("texture_index: cache load failed: %s", e)
        return None
    saved_key = payload.get("key")
    if not isinstance(saved_key, list) or len(saved_key) != 2:
        return None
    if saved_key[0] != cache_key[0] or int(saved_key[1]) != cache_key[1]:
        return None
    # v1 cache files (pre-AFS) are missing the AFS half of the index.
    # v2 cache files lack per-XVR sub-rows for ItemTexture archives, so
    # ItemModel positional cross_afs lookup can't bind. Force a rebuild
    # instead of returning a stale half-index.
    schema = int(payload.get("version", 1))
    if schema < 3:
        log.info(
            "texture_index: cache schema v%d found; forcing rebuild "
            "(need v3 for ItemTexture deep walk)", schema,
        )
        return None
    raw = payload.get("index", {})
    out: Dict[str, List[TextureLocation]] = {}
    for name, locs in raw.items():
        rebuilt: list[TextureLocation] = []
        for loc in locs:
            # Accept v1 dicts (no kind/archive/inner_index) gracefully.
            kwargs = dict(loc)
            kwargs.setdefault("kind", "bml")
            kwargs.setdefault("archive", kwargs.get("bml_name", ""))
            kwargs.setdefault("inner_index", -1)
            rebuilt.append(TextureLocation(**kwargs))
        out[name] = rebuilt
    log.info("texture_index: loaded cached index (%d names)", len(out))
    return out


def get_texture_index(data_dir: Path) -> Dict[str, List[TextureLocation]]:
    """Cached entry point used by server.py for cross-archive lookup.

    Builds the index lazily on first call; caches in-process for the
    lifetime of the python process (and on disk under ``cache/`` for
    future cold starts). Walks BOTH BMLs and AFS archives — see
    ``build_global_texture_index``.

    Thread-safe: serialised behind a single lock.
    """
    global _INDEX_CACHE, _INDEX_CACHE_KEY
    cache_key = _cache_key(data_dir)
    with _INDEX_LOCK:
        if _INDEX_CACHE is not None and _INDEX_CACHE_KEY == cache_key:
            return _INDEX_CACHE
        # Try disk cache first.
        disk = _load_cache(cache_key)
        if disk is not None:
            _INDEX_CACHE = disk
            _INDEX_CACHE_KEY = cache_key
            return disk
    # Cold build outside the lock so concurrent callers can still
    # serve hot-cache reads. Race condition: two cold callers may both
    # build; the second one wins on assignment, both results are
    # equivalent.
    fresh = build_global_texture_index(data_dir)
    _save_cache(fresh, cache_key)
    with _INDEX_LOCK:
        _INDEX_CACHE = fresh
        _INDEX_CACHE_KEY = cache_key
    return fresh


def clear_cache() -> None:
    """Drop the in-memory + on-disk index. Test-only helper."""
    global _INDEX_CACHE, _INDEX_CACHE_KEY
    with _INDEX_LOCK:
        _INDEX_CACHE = None
        _INDEX_CACHE_KEY = None
    try:
        _CACHE_FILE.unlink()
    except FileNotFoundError:
        pass


def lookup(
    data_dir: Path,
    name: str,
) -> List[TextureLocation]:
    """Convenience: return all known locations for ``name`` (lower-cased
    comparison)."""
    if not name:
        return []
    idx = get_texture_index(data_dir)
    out = idx.get(name)
    if out:
        return list(out)
    # Fall back to a case-insensitive sweep — PSOBB names are
    # mostly-but-not-always lower-case.
    lower = name.lower()
    for k, v in idx.items():
        if k.lower() == lower:
            return list(v)
    return []


# ---------------------------------------------------------------------------
# Player-class positional fallback
# ---------------------------------------------------------------------------


def player_class_for(model_filename: str) -> Optional[str]:
    """Return the class letter (A..Z) for a player NJ model, or None.

    ``plAbdy00.nj`` -> ``"A"``, ``plChed01.nj`` -> ``"C"``, etc. Files
    that don't match the player-class naming convention return None.
    """
    if not model_filename:
        return None
    stem = Path(model_filename).stem
    m = _PLAYER_NJ_RE.match(stem)
    if not m:
        return None
    return m.group(1).upper()


def list_bml_xvmh_inners(bml_path: Path) -> List[tuple]:
    """Return ``[(inner_name, xvr_count), ...]`` for every XVMH-bearing inner.

    Walks one BML and reports every ``.nj`` / ``.xj`` inner whose paired
    XVM archive decompresses to a valid XVMH with at least one XVR
    record. The list preserves the on-disk order of inners so the first
    entry tends to be the BML's "main" model (which usually owns the
    fullest texture set).

    Used by ``_build_model_texture_binding`` for the same-BML cross-
    inner positional fallback. PSOBB packs many small accessory models
    (Vol Opt monitors' bezels, De Rol Le's helm/shell shards, item
    boxes, player class secondaries) into the same BML as their
    "parent", with only the parent owning the inline XVM. The accessory
    inners' material_ids index into the parent's XVMH archive
    positionally — the runtime resolves these via shared NJTL state in
    PSOBB's texture allocator. We mirror that behaviour by emitting a
    ``cross_bml`` location pointing at the BML's largest XVMH-bearing
    sibling.

    Returns an empty list when no inner in the BML carries a paired
    XVMH (which is the common case for animation-only BMLs).

    Cached in-process by (bml_basename, mtime_ns); thread-safe.
    """
    # Local imports to avoid a top-level cycle (server.py imports
    # texture_index at startup, but bml.extract_bml_texture pulls in
    # several other modules).
    from formats.bml import extract_bml_texture

    try:
        st = bml_path.stat()
    except OSError:
        return []
    cache_key = (bml_path.name, int(st.st_mtime_ns))
    with _BML_XVMH_LOCK:
        cached = _BML_XVMH_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    try:
        buf = bml_path.read_bytes()
    except OSError:
        return []
    try:
        entries = parse_bml(buf)
    except Exception:
        return []

    out: List[tuple] = []
    for ent in entries:
        ext_lower = ent.name.lower()
        if not (ext_lower.endswith(".nj") or ext_lower.endswith(".xj")):
            continue
        if not ent.has_texture or ent.tex_size_compressed == 0:
            continue
        # Read the inline archive and count its texture records. Magic-
        # routes XVMH (Xbox) / PVMH (Dreamcast) / GVMH (GameCube) via
        # ``_count_xvmh_records`` so a Dreamcast BML's textured inners are
        # surfaced for the cross-inner positional fallback instead of
        # being silently skipped (2026-06-21, psov2 multivariant).
        try:
            xvm = extract_bml_texture(buf, ent.name)
        except Exception:
            continue
        if xvm is None:
            continue
        xvr_count = _count_xvmh_records(xvm)
        # Sanity-cap is already applied inside _count_xvmh_records (real
        # models have <=64 textures; PSOBB's runtime tops out there).
        if xvr_count > 0:
            out.append((ent.name, int(xvr_count)))

    with _BML_XVMH_LOCK:
        _BML_XVMH_CACHE[cache_key] = out
    return list(out)


def best_sibling_xvmh_for(
    bml_path: Path,
    exclude_inner: Optional[str],
    min_records: int = 1,
) -> Optional[tuple]:
    """Pick the best sibling-inner XVMH source for cross-inner fallback.

    Returns ``(inner_name, xvr_count)`` for the inner in this BML with
    the LARGEST XVR record count (so a missing material_id N has the
    best chance of being in-range), excluding ``exclude_inner`` and any
    inner whose record count is below ``min_records``.

    Returns ``None`` when no qualifying sibling exists.
    """
    inners = list_bml_xvmh_inners(bml_path)
    candidates = [
        (name, count) for (name, count) in inners
        if name != exclude_inner and count >= min_records
    ]
    if not candidates:
        return None
    # Pick the inner with the most records (most likely to cover any
    # given mid). Ties broken by source order (first-listed wins) which
    # tracks PSOBB's "main" inner convention.
    candidates.sort(key=lambda c: (-c[1], inners.index(c)))
    return candidates[0]


def lookup_player_class_textures(
    data_dir: Path,
    model_filename: str,
) -> List[TextureLocation]:
    """Return the positional texture map for a player NJ model.

    Player class models (``plAbdy00.nj``, ``plKhai00.nj``, ...) ship
    WITHOUT an NJTL chunk; their material slots bind positionally
    against the matching ``pl<class>tex.afs`` archive. The runtime
    knows this from the model's filename prefix; the editor needs the
    same mapping table.

    Returns the AFS locations sorted by ``inner_index``, so the
    consumer can index the list by ``material_id`` directly:
    ``locations[material_id]`` is the texture for that slot.

    Returns an empty list if:
      - the filename doesn't match the player-class naming convention,
      - the matching ``pl<class>tex.afs`` is missing,
      - the AFS contains no XVR-magic blobs (i.e. it's not a player
        texture archive).
    """
    cls = player_class_for(model_filename)
    if cls is None:
        return []
    archive_stem = f"pl{cls}tex"
    archive_name = f"{archive_stem}.afs"
    archive_path = data_dir / archive_name
    if not archive_path.exists():
        return []
    # Filter the AFS index for this specific archive's bare-XVR rows.
    # We can't just scan get_texture_index() because the AFS rows are
    # keyed on synthesised names; instead pull them direct.
    out: Dict[str, List[TextureLocation]] = {}
    _index_one_afs(archive_path, out)
    # Flatten to a positional list; the synthesised name encodes the
    # inner_index so we can sort robustly.
    flat: list[TextureLocation] = []
    for locs in out.values():
        flat.extend(locs)
    flat.sort(key=lambda loc: loc.inner_index)
    # Filter to bare-XVR entries only (xvr_index == 0). The AFS could
    # in principle hold mixed records; player tex archives in practice
    # are uniform XVRs.
    return [loc for loc in flat if loc.xvr_index == 0 and loc.kind == "afs"]


# ---------------------------------------------------------------------------
# ItemModel positional fallback
# ---------------------------------------------------------------------------


def item_archive_for(model_archive: str) -> Optional[str]:
    """Map an ItemModel-style archive name to its sibling texture archive.

    ``ItemModel.afs`` -> ``ItemTexture.afs``;
    ``ItemModelEp4.afs`` -> ``ItemTextureEp4.afs``.

    Returns None for unrecognised archives so callers can quickly skip
    the lookup. Case-insensitive on the basename so URL-encoded paths
    that arrive lower-case still resolve.
    """
    if not model_archive:
        return None
    name = Path(model_archive).name
    # Direct match first (preserves case-correct PSOBB.IO basenames).
    if name in _ITEM_MODEL_TO_TEXTURE:
        return _ITEM_MODEL_TO_TEXTURE[name]
    # Case-insensitive fallback for callers that lower-cased the name.
    lower = name.lower()
    for k, v in _ITEM_MODEL_TO_TEXTURE.items():
        if k.lower() == lower:
            return v
    return None


def lookup_item_textures(
    data_dir: Path,
    model_archive: str,
    inner_index: int,
) -> List[TextureLocation]:
    """Positional cross-AFS lookup for an ItemModel inner's textures.

    Returns the per-NJTL-slot texture locations for
    ``<model_archive>#<inner_index>``. The K-th element of the returned
    list is the texture record for NJTL slot K of that inner.

    The lookup is purely positional: the matching ItemTexture archive's
    inner at the SAME ``inner_index`` is opened and its XVR records are
    enumerated in order. This mirrors the runtime convention — PSOBB
    pairs ItemModel.afs#NNNN with ItemTexture.afs#NNNN and uses the
    NJTL slot to index into the XVR record table.

    Returns an empty list when:
      - ``model_archive`` is not in the ItemModel→ItemTexture map,
      - the matching ItemTexture archive is missing from ``data_dir``,
      - the texture-side inner is missing or not an XVMH.

    The result is pulled directly from the global texture index
    (``get_texture_index``) so a fresh rebuild populates new entries
    without needing per-call deep walks.
    """
    if inner_index < 0:
        return []
    tex_archive_name = item_archive_for(model_archive)
    if not tex_archive_name:
        return []
    tex_archive_path = data_dir / tex_archive_name
    if not tex_archive_path.exists():
        return []
    # Use the global cached index — the deep walk emits per-XVR rows
    # under the synth name ``<stem>_<NNNN>_<MMMM>``, so we can pull the
    # full table by scanning rows whose inner_index matches.
    idx = get_texture_index(data_dir)
    flat: list[TextureLocation] = []
    for locs in idx.values():
        for loc in locs:
            if (
                loc.kind == "afs"
                and loc.archive == tex_archive_name
                and loc.inner_index == inner_index
                and loc.xvr_index >= 0  # skip outer XVMH placeholder rows
            ):
                flat.append(loc)
    flat.sort(key=lambda loc: loc.xvr_index)
    return flat


# ---------------------------------------------------------------------------
# Sibling-BML stem-family lookup (NJTL-less inner fallback)
# ---------------------------------------------------------------------------

# Common stem fragments shared across many PSOBB inners (prefix tokens
# that don't disambiguate a model — e.g. fe_obj_, de_obj_, fd_obj_,
# bm_obj_, swarp_, etc.). Used to filter out low-signal tokens when
# computing inner-name overlap for the sibling-BML stem-family lookup
# below — a match on "obj" or "df" is meaningless, but a match on
# "warp_gawa" or "monitor_bezel" is a high-confidence hit.
_GENERIC_INNER_TOKENS = frozenset({
    "obj", "obj0", "obj1", "obj2", "obj3", "obj4", "obj5",
    "fe", "fd", "de", "dd", "bm", "fs", "n", "s",
    "bm_obj", "fs_obj", "fe_obj", "fd_obj", "de_obj", "dd_obj",
    "warp", "df", "boss", "ene", "eff",
    "n", "ne", "se", "sw", "nw",  # cardinal markers (e.g. Vol Opt monitor)
    "low", "high", "main", "sub",  # LOD markers
    "a", "b", "c", "d", "ab", "ac", "ad",  # variant markers
    "01", "02", "03", "04", "05", "06", "07", "08", "09",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
})


def _inner_stem_tokens(inner_name: str) -> set[str]:
    """Decompose an inner filename into a set of meaningful stem tokens.

    Splits on ``_`` then drops generic markers (see
    ``_GENERIC_INNER_TOKENS``). The returned set is suitable for
    Jaccard-style overlap scoring against another inner's tokens — a
    non-empty intersection of NON-generic tokens is treated as a
    sibling-family hit.

    Examples:
      ``fe_obj_df_warp_gawa.xj``  -> ``{"warp_gawa"}``  (after dropping fe/obj/df)
      ``fd_obj1_swarp_gawa.xj``   -> ``{"swarp_gawa"}`` (after dropping fd/obj1)
      ``boss1_s_nb_dragon.nj``    -> ``{"dragon"}``     (after dropping boss1/s/nb)
    """
    if not inner_name:
        return set()
    stem = Path(inner_name).stem.lower()
    parts = [p for p in re.split(r"[_\.]+", stem) if p]
    # Drop strict-generic single tokens; KEEP multi-letter unique words.
    meaningful = [p for p in parts if p not in _GENERIC_INNER_TOKENS and len(p) > 1]
    if not meaningful:
        return set()
    # Emit BOTH the joined-meaningful stem (for tight matching) AND each
    # individual token. The joined form gives "warp_gawa" which only
    # matches a sibling that also contains both warp+gawa; the
    # per-token set also catches partial matches like "gawa"-only.
    out = set(meaningful)
    if len(meaningful) > 1:
        out.add("_".join(meaningful))
    return out


def find_sibling_bml_by_inner_stem(
    bml_path: Path,
    inner_name: str,
    *,
    min_xvr_count: int = 1,
) -> Optional[tuple]:
    """Find a sibling BML whose inner shares a stem-family with ``inner_name``.

    Walks every ``*.bml`` in ``bml_path``'s parent directory, looking
    for one whose XVMH-bearing inners contain a stem-family match for
    the requested inner. Used by ``_build_model_texture_binding`` as a
    NJTL-less fallback: when our inner has 0 inline tiles AND no NJTL
    chunk (so cross-archive name lookup can't fire), the most plausible
    texture provider is a sibling BML whose inner shares a meaningful
    stem token (e.g. "gawa" between
    ``bm_obj_warpboss_ancient.bml#fe_obj_df_warp_gawa.xj`` and
    ``bm_o_warp_ancient.bml#fd_obj1_swarp_gawa.xj``).

    Returns
    -------
    ``(sibling_bml_basename, sibling_inner_name, sibling_xvr_count)`` for
    the BEST candidate (highest token-overlap), or ``None`` if no
    sibling BML in the directory contains a matching inner.

    Excludes ``bml_path`` itself from the search. Excludes inners with
    fewer than ``min_xvr_count`` XVR records (default 1 = "must have
    AT LEAST one tile to bind").
    """
    if not inner_name:
        return None
    src_tokens = _inner_stem_tokens(inner_name)
    if not src_tokens:
        return None

    parent = bml_path.parent
    if not parent.is_dir():
        return None

    src_basename = bml_path.name
    best: Optional[tuple] = None
    best_score: int = 0

    for sibling in sorted(parent.glob("*.bml")):
        if sibling.name == src_basename:
            continue
        try:
            inners = list_bml_xvmh_inners(sibling)
        except Exception:  # pragma: no cover - defensive
            continue
        for sib_inner_name, sib_xvr_count in inners:
            if sib_xvr_count < min_xvr_count:
                continue
            sib_tokens = _inner_stem_tokens(sib_inner_name)
            if not sib_tokens:
                continue
            overlap = src_tokens & sib_tokens
            if not overlap:
                continue
            # Score: count of overlapping tokens, with a tiebreaker on
            # the sibling's tile count (richer texture pool wins).
            score = len(overlap) * 1000 + sib_xvr_count
            if score > best_score:
                best_score = score
                best = (sibling.name, sib_inner_name, int(sib_xvr_count))

    return best


__all__ = [
    "TextureLocation",
    "build_texture_index",
    "build_global_texture_index",
    "index_afs_archives",
    "get_texture_index",
    "clear_cache",
    "lookup",
    "list_bml_xvmh_inners",
    "best_sibling_xvmh_for",
    "lookup_player_class_textures",
    "lookup_item_textures",
    "item_archive_for",
    "player_class_for",
    "find_sibling_bml_by_inner_stem",
]
