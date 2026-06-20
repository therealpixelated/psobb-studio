"""Asset manifest layer for the PSOBB Modding Suite (Phase A — Agent 1).

Walks an install root (or any data directory) and produces a normalized
metadata record per file conforming to ``MASTER_PLAN/manifest.schema.json``.

Public entry points:
  - ``walk_install(root)``      yield non-backup files
  - ``classify(path, root)``    -> AssetEntry dict
  - ``build_manifest(root)``    -> Manifest dict
  - ``cache_manifest(root)``    -> cached Manifest dict, rebuilt if stale

Idempotency: the manifest is sorted by relative path and contains no
clock-derived fields except ``generated_at`` (an integer epoch). Two runs
over the same input tree at the same wall-clock time produce byte-identical
JSON; even across different times the only diff is the top-level timestamp.
This lets downstream consumers (frontend, diff tools) treat the manifest
as a stable snapshot.

Atomic writes: ``cache/manifest.json`` is written via ``<path>.tmp`` +
``os.replace`` so a crash mid-write never leaves a half-baked file in
place.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("psobb_editor.manifest")

# Phase A integration: the texture<->model multi-rule matcher (Agent 3).
# Imported lazily-ish: a missing or broken matcher must NOT poison the
# whole manifest build. We import once at module load and feature-flag the
# call site.
try:
    from formats import match as _match_mod  # type: ignore
    _HAS_MATCHER = (
        hasattr(_match_mod, "match_textures")
        and hasattr(_match_mod, "matches_to_manifest_field")
    )
except Exception:  # pragma: no cover - defensive
    _match_mod = None  # type: ignore[assignment]
    _HAS_MATCHER = False

# AFS unpacker: synthesizes one manifest entry per archived inner blob
# (ItemModel.afs#0042_sword.nj etc.). Optional — if it fails to import,
# AFS archives just appear as their top-level entries with no inner
# expansion (existing behaviour).
try:
    from formats import afs_reader as _afs_reader_mod  # type: ignore
    _HAS_AFS_READER = (
        hasattr(_afs_reader_mod, "list_inner_blobs")
        and hasattr(_afs_reader_mod, "iter_afs_archives")
    )
except Exception:  # pragma: no cover - defensive
    _afs_reader_mod = None  # type: ignore[assignment]
    _HAS_AFS_READER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Manifest schema version. Bump whenever an AssetEntry shape changes in a
# non-backwards-compatible way.
MANIFEST_VERSION = 1

# How many bytes to read from each file to compute the magic prefix.
# 16 keeps us well under any sniff-needed offset (longest format prefix is
# AFS file table, but we only need the magic itself for classification).
MAGIC_PROBE_BYTES = 16

# How many bytes to read for inner-format sniffing on PRS files. PRS's
# leading byte is a bitmap; the next ~6 bytes are usually literal output,
# which contain the inner magic when the inner is a tagged format like
# XVMH or NJCM. 64 bytes gives us plenty of headroom.
PRS_SNIFF_BYTES = 64

# Substrings (case-insensitive) that flag a file as a backup / quarantine
# sibling. Matching either the *suffix* or as a *substring of the name*
# (so "Foo.afs.SUSPECT_crash_20260424" is excluded). Mirrors the
# heuristic in ``server.BACKUP_FRAGMENTS`` plus the spec's extra forms.
BACKUP_PATTERNS_SUFFIX = (
    ".bak",
    ".disabled",
)
BACKUP_PATTERNS_SUBSTRING = (
    ".pre_",
    ".suspect_",
    ".parked_",
    ".bad_",
    ".not_og_",
    ".disabled",
)
BACKUP_PATTERNS_PREFIX = (
    "pre_",
)

# Categorical map: extension -> (category, format, parsable).
# When the magic doesn't match the extension we record a warning but keep
# the extension's mapping (extensions are the user-facing label).
#
# parsable values reflect the editor's *current* decode capability:
#   yes      = round-trips (PRS via PuyoTools, XVM/XVR via xvr_codec)
#   partial  = read header / list contents only (BML, NJ_IFF — Phase B)
#   no       = format known but no decoder (REL, DAT_QUEST, ...).
#
# Any extension not in this table falls through to UNKNOWN/unknown/no.
_EXT_MAP: dict[str, tuple[str, str, str]] = {
    ".prs": ("texture",   "PRS",       "yes"),
    ".xvm": ("texture",   "XVM",       "yes"),
    ".xvr": ("texture",   "XVR",       "yes"),
    ".afs": ("container", "AFS",       "no"),
    ".bml": ("model",     "BML",       "partial"),
    ".nj":  ("model",     "NJ_IFF",    "partial"),
    ".rel": ("map",       "REL",       "no"),
    ".dat": ("quest",     "DAT_QUEST", "no"),
    ".bin": ("quest",     "BIN_QUEST", "no"),
    ".evt": ("script",    "EVT",       "no"),
    ".pae": ("cinematic", "PAE",       "no"),
    ".gsl": ("script",    "GSL",       "no"),
    ".pr2": ("metadata",  "PR2",       "no"),
    ".pr3": ("metadata",  "PR3",       "no"),
    ".prc": ("container", "PRC",       "no"),
    ".lst": ("script",    "LST",       "yes"),
    ".png": ("ui",        "PNG",       "yes"),
    ".ogg": ("audio",     "OGG",       "yes"),
    ".txt": ("metadata",  "TXT",       "yes"),
}

# Magic byte signatures keyed by canonical format identifier.
# Order matters for ambiguous extensions: longer / more specific first.
_MAGIC_TABLE: list[tuple[bytes, str]] = [
    (b"XVMH",   "XVM"),
    (b"XVRT",   "XVR"),
    (b"NJCM",   "NJ_IFF"),
    (b"AFS\x00", "AFS"),
    (b"\x89PNG", "PNG"),
    (b"OggS",   "OGG"),
]


# ---------------------------------------------------------------------------
# Inferred-category mapping (Phase B) — JSON-driven
#
# The categorizer now reads its rules from ``_reports/categorization_db.json``,
# the canonical asset-categorization database produced and maintained by
# the research-agent pipeline. The DB carries 100+ prefix patterns plus
# subcategory + in_game_name annotations, all sourced from PSOBB.exe
# disasm + Phantasmal World + newserv. See ``_reports/categorization_db.md``
# for the human-readable companion.
#
# Each rule has the shape::
#     {
#       "pattern":      "bm_boss1_dragon*.bml",
#       "category":     "Bosses",
#       "subcategory":  "EP1 Forest boss",
#       "in_game_name": "Sil Dragon (Dragon)",
#       "source":       "PsoBB.exe @0x4faeec ..."
#     }
#
# Pattern syntax:
#   - "Foo.afs#*"     → match by parent_archive (AFS inner blobs)
#   - "scene/*"       → match by parent directory path
#   - "bm_*"          → fnmatch-style basename glob
#
# Rule precedence: the first matching rule wins, so more-specific patterns
# MUST appear before less-specific ones in the JSON. The DB is already
# vetted in that order; we don't reorder at load time.
#
# Backwards compatibility: the public ``infer_category()`` still returns a
# single category-string keyed by the JSON's ``category`` field, so
# existing consumers (tree.js etc.) reading ``entry.inferred_category``
# keep working. The richer ``infer_category_full()`` accessor returns
# the full dict for callers that want subcategory / in-game name.
# ---------------------------------------------------------------------------

# Canonical location of the rule DB. Keep this as the single source of
# truth — do NOT inline the rules into Python.
_CATEGORY_DB_PATH = Path(__file__).parent / "_reports" / "categorization_db.json"

# Lazy-loaded cache: ``_load_category_db()`` populates this on first call,
# subsequent calls return the cached dict. Tests that mutate the DB on
# disk can call ``_category_db_cache_clear()`` to force a reload.
_CATEGORY_DB: Optional[dict] = None


def _load_category_db() -> dict:
    """Load and cache the categorization DB JSON. Returns the parsed dict.

    On any read / parse failure we degrade to an empty rule-set with the
    fallback "Uncategorized" — the manifest still builds, every entry
    just lacks an ``inferred_category`` field.

    Perf 2026-04-30: stamps each rule with a ``"_pattern_l"`` key holding
    the lowercase pattern, so ``_match_pattern`` skips ~1.5M redundant
    ``.lower()`` calls per full manifest rebuild (171 rules × 9k entries).
    The original ``"pattern"`` field is preserved unchanged for callers
    that read it back.
    """
    global _CATEGORY_DB
    if _CATEGORY_DB is None:
        try:
            with open(_CATEGORY_DB_PATH, "r", encoding="utf-8") as f:
                _CATEGORY_DB = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning(
                "categorization_db unreadable, falling back to empty rules: %s", e
            )
            _CATEGORY_DB = {"rules": [], "fallback": "Uncategorized"}
        # One-time precompute of pattern.lower() per rule. Stored under
        # a leading-underscore key to make clear it's an in-memory cache,
        # never written back to disk.
        for _rule in _CATEGORY_DB.get("rules") or []:
            if isinstance(_rule, dict) and "pattern" in _rule:
                _rule["_pattern_l"] = (_rule.get("pattern") or "").lower()
    return _CATEGORY_DB


def _category_db_cache_clear() -> None:
    """Drop the cached DB so the next ``infer_category()`` call reloads
    from disk. Used by tests that swap the DB file."""
    global _CATEGORY_DB
    _CATEGORY_DB = None


def _match_pattern_lowered(
    name_l: str, parent_l: str, archive_l: str, pat_l: str,
) -> bool:
    """Hot-path matcher: all four inputs are trusted to be lowercase.

    Used by ``infer_category_full`` to avoid recomputing ``.lower()`` on
    every (entry, rule) pair during a full manifest rebuild — at 9k
    entries × 171 rules that's 6M redundant lower() calls. The public
    ``_match_pattern`` is a thin wrapper that lowers and delegates.
    """
    if not pat_l:
        return False

    # AFS-inner pattern (archive-name match).
    if "afs#" in pat_l:
        # Strip the "#..." tail and compare the head against parent_archive.
        archive_part = pat_l.split("#", 1)[0]
        if not archive_l:
            return False
        # Pattern head may itself be a glob like "ItemKT*.afs"
        return fnmatch.fnmatchcase(archive_l, archive_part)

    # Path-fragment pattern (e.g. "scene/*", "ogg/*").
    if "/" in pat_l:
        joined = f"{parent_l}/{name_l}" if parent_l else name_l
        if fnmatch.fnmatchcase(joined, pat_l):
            return True
        # Also accept "scene/*" when the parent is exactly "scene" or has
        # a "scene" anywhere in its path (matches the legacy
        # ``_is_in_scene_dir`` behaviour for nested scene/ subtrees).
        head = pat_l.rstrip("*").rstrip("/")
        if head and (parent_l == head or parent_l.startswith(head + "/")
                     or f"/{head}/" in parent_l or parent_l.endswith(f"/{head}")):
            return True
        return False

    # Plain glob on basename. ``fnmatchcase`` honors *, ?, [seq].
    if fnmatch.fnmatchcase(name_l, pat_l):
        return True
    # Synthesised AFS-inner blobs have an empty parent path and a name
    # that contains the archive name plus a "#NNNN_inner" tail. The DB
    # often labels the archive itself (e.g. "pl?tex.afs", "plZsmpnj.afs")
    # so we also try the pattern against ``parent_archive`` — this lets
    # the same rule annotate both the archive top-level entry AND every
    # one of its inner-blob children without duplicating rows in the DB.
    if archive_l and fnmatch.fnmatchcase(archive_l, pat_l):
        return True
    return False


def _match_pattern(name: str, parent: str, parent_archive: str, pattern: str) -> bool:
    """Match a glob-style ``pattern`` against the (name, parent, archive) triple.

    Pattern dispatch (in order):
      1. ``"<archive>.afs#*"``  — the entry is an AFS inner blob; match by
         comparing the archive prefix (case-insensitive). This catches
         all of e.g. ``ItemModel.afs#0042_sword.nj``.
      2. ``"<dir>/*"``  — path-fragment glob; match by joining parent +
         name and running ``fnmatch`` against the joined string.
      3. plain glob   — ``fnmatch`` against the basename.

    All comparisons are case-insensitive (the DB stores lowercase
    patterns; we lower the inputs). External callers (tests + any future
    direct integrators) keep the original API; the hot internal path
    skips the redundant lower() through ``_match_pattern_lowered``.
    """
    if not pattern:
        return False
    return _match_pattern_lowered(
        name.lower(), parent.lower(), parent_archive.lower(), pattern.lower(),
    )


def infer_category_full(
    rel_path: str, parent_archive: Optional[str] = None
) -> Optional[dict]:
    """Look up the rich category record for ``rel_path``.

    Returns a dict with keys ``category``, ``subcategory``, ``in_game_name``,
    ``pattern`` (the matching pattern), or None when no rule matches.
    Walks the JSON DB's ``rules`` list in order — the first match wins,
    so ordering in the DB is load-bearing.

    Pure function; no I/O beyond the lazy DB load on first call.
    """
    if not rel_path:
        return None
    db = _load_category_db()
    rules = db.get("rules") or []
    if not rules:
        return None
    name = Path(rel_path).name.lower()
    # Path before the basename. For inner-blob synth paths like
    # "ItemModel.afs#0042_inner.nj" the parent is "" (no dir component);
    # parent_archive carries the archive name instead.
    parent_path = "/".join(
        rel_path.replace("\\", "/").split("/")[:-1]
    ).lower()
    arch = (parent_archive or "").lower()
    # Hot loop: dispatch to the lowered fast path so we don't redo
    # name/parent/arch/pattern .lower() per-rule. The rule's
    # ``_pattern_l`` cache key is populated by _load_category_db on
    # first load (so the per-rule .lower() runs at most once across the
    # process lifetime).
    for rule in rules:
        pat_l = rule.get("_pattern_l")
        pat = rule.get("pattern", "")
        if pat_l is None:
            pat_l = (pat or "").lower()
        try:
            if _match_pattern_lowered(name, parent_path, arch, pat_l):
                return {
                    "category": rule.get("category"),
                    "subcategory": rule.get("subcategory"),
                    "in_game_name": rule.get("in_game_name"),
                    "pattern": pat,
                }
        except Exception:  # pragma: no cover — defensive
            continue
    return None


def infer_category(
    rel_path: str, parent_archive: Optional[str] = None
) -> Optional[str]:
    """Bucket ``rel_path`` into a user-facing inferred category.

    Backwards-compatible wrapper around :func:`infer_category_full`: returns
    only the top-level category string (the value emitted as
    ``inferred_category`` on each AssetEntry) so existing consumers
    (``tree.js`` etc.) keep working unchanged. New consumers wanting the
    subcategory or in-game name should call :func:`infer_category_full`.

    Returns ``None`` when no rule matches; the asset tree groups those
    under the canonical ``category`` instead.
    """
    info = infer_category_full(rel_path, parent_archive)
    return info.get("category") if info else None


# AFS-inner blob → top-level format key map. Used by ``classify_inner_blob``
# to project the format-id sniffed by ``afs_reader`` to the same enum
# values the rest of the manifest pipeline uses (so consumers that read
# ``format`` don't have to special-case AFS rows).
_INNER_FORMAT_TO_CATEGORY: dict[str, tuple[str, str]] = {
    "NJ_IFF":    ("model",     "partial"),
    "XVM":       ("texture",   "yes"),
    "XVR":       ("texture",   "yes"),
    "PVR":       ("texture",   "no"),
    "PNG":       ("ui",        "yes"),
    "OGG":       ("audio",     "yes"),
    "NJM":       ("animation", "no"),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_backup_name(name: str) -> bool:
    """True if ``name`` looks like a backup / quarantine sibling.

    Matches on suffix, on lowercased substring, and on lowercased prefix.
    Conservative on purpose: false positives just hide one entry, false
    negatives leak backup files into the manifest where the UI would
    surface them as live assets.
    """
    nl = name.lower()
    for suf in BACKUP_PATTERNS_SUFFIX:
        if nl.endswith(suf):
            return True
    for sub in BACKUP_PATTERNS_SUBSTRING:
        if sub in nl:
            return True
    for pre in BACKUP_PATTERNS_PREFIX:
        if nl.startswith(pre):
            return True
    return False


def _read_magic(path: Path, n: int = MAGIC_PROBE_BYTES) -> bytes:
    """Read up to ``n`` bytes from the head of ``path``. Returns b'' on
    failure (the entry will then carry a warning + UNKNOWN format)."""
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError as e:
        log.debug("could not read magic from %s: %s", path, e)
        return b""


def _scan_inner_magic(prs_head: bytes) -> Optional[str]:
    """For a PRS file, peek at the head bytes and try to find a known
    inner-format magic. PRS bitmap byte + literal-heavy openings mean the
    inner magic almost always appears verbatim in the first ~16 bytes."""
    if not prs_head:
        return None
    for magic, fmt in _MAGIC_TABLE:
        if magic in prs_head[:PRS_SNIFF_BYTES]:
            return fmt
    return None


def _ascii_safe(buf: bytes, n: int = 4) -> str:
    """Render the first ``n`` bytes as printable ASCII; non-printable
    bytes are replaced with '.'. Matches the JSON Schema requirement of
    a max-16 string and ensures all entries can round-trip through
    ``json.dumps`` cleanly."""
    if not buf:
        return ""
    out_chars = []
    for b in buf[:n]:
        if 0x20 <= b < 0x7F:
            out_chars.append(chr(b))
        else:
            out_chars.append(".")
    return "".join(out_chars)


def _rel_path(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` using forward slashes (POSIX
    style) — matches the schema's example shape."""
    return path.relative_to(root).as_posix()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def walk_install(root: Path) -> Iterator[Path]:
    """Yield every regular file under ``root`` that is NOT a backup/
    quarantine sibling.

    Matching rule (any condition triggers exclusion):
      - filename ends with one of ``BACKUP_PATTERNS_SUFFIX`` (`.bak`,
        `.disabled`)
      - filename contains one of ``BACKUP_PATTERNS_SUBSTRING`` (`.pre_`,
        `.suspect_`, `.parked_`, `.bad_`, `.not_og_`, `.disabled`)
      - filename starts with one of ``BACKUP_PATTERNS_PREFIX`` (`pre_`)

    Order is deterministic: directories are walked depth-first with each
    level sorted lexicographically (case-insensitive). Two walks over
    the same tree on the same machine yield identical orderings, which
    feeds into manifest idempotency.
    """
    root = root.resolve()
    if not root.exists():
        return
    if root.is_file():
        if not _is_backup_name(root.name):
            yield root
        return
    # Walk directories sorted; files sorted; backup-skip on each.
    for dirpath, dirnames, filenames in os.walk(root):
        # in-place sort makes os.walk yield in deterministic order
        dirnames.sort(key=str.lower)
        filenames.sort(key=str.lower)
        d = Path(dirpath)
        for fn in filenames:
            if _is_backup_name(fn):
                continue
            p = d / fn
            try:
                if p.is_file():
                    yield p
            except OSError:
                # broken symlink / permission issue — skip
                continue


def classify(path: Path, root: Optional[Path] = None) -> dict:
    """Return an AssetEntry dict for ``path`` matching ``manifest.schema.json``.

    ``root`` is used to render the ``path`` field as a forward-slash
    relative path. If omitted, the bare filename is used (still schema-
    legal — there is no required prefix in the schema).

    Side-effects: reads up to ``MAGIC_PROBE_BYTES`` bytes from the file.
    Never opens the file for write. Errors during stat/read are degraded
    into ``warnings`` rather than raised, so a single bad file cannot
    poison the whole manifest.
    """
    warnings: list[str] = []

    # ------- size + mtime --------------------------------------------------
    try:
        st = path.stat()
        size = int(st.st_size)
        mtime = int(st.st_mtime)
    except OSError as e:
        warnings.append(f"stat failed: {e}")
        size = 0
        mtime = 0

    # ------- relative path -------------------------------------------------
    if root is not None:
        try:
            rel = _rel_path(path, root)
        except ValueError:
            # path was not under root — fall back to the filename.
            warnings.append("path is outside root; using bare name")
            rel = path.name
    else:
        rel = path.name

    # ------- extension -----------------------------------------------------
    ext = path.suffix.lower()
    if not ext:
        # Schema requires extension to be ".[a-z0-9]+" — when there's no
        # extension we record an empty placeholder via the warning system
        # and use ".unknown" so the manifest still validates.
        warnings.append("file has no extension")
        ext = ".unknown"

    # Ensure schema-legal extension shape (lowercase alphanumerics only).
    # Some files have extensions with periods / odd chars — coerce to
    # ".unknown" with a warning.
    if not ext.startswith(".") or not ext[1:].isalnum() or not ext[1:].islower():
        warnings.append(f"non-canonical extension {ext!r}")
        ext = ".unknown"

    # ------- magic ---------------------------------------------------------
    head = _read_magic(path, MAGIC_PROBE_BYTES)
    magic_hex = head.hex()
    magic_ascii = _ascii_safe(head, 4)

    # ------- format / category / parsable ---------------------------------
    cat, fmt, parsable = _EXT_MAP.get(ext, ("unknown", "UNKNOWN", "no"))

    # Magic-byte cross-check. For tagged formats (XVM/XVR/NJ_IFF/AFS/PNG/
    # OGG) a mismatch is a strong signal something's wrong; record a
    # warning but keep the extension-derived classification (the user's
    # mental model is filename-driven).
    detected_fmt: Optional[str] = None
    for magic, m_fmt in _MAGIC_TABLE:
        if head.startswith(magic):
            detected_fmt = m_fmt
            break

    if detected_fmt is not None and fmt != "PRS" and detected_fmt != fmt:
        # Extension claims one format, magic says another. This actually
        # happens in the wild for AFS-wrapped XVM streams.
        if not (fmt == "UNKNOWN" or fmt == "AFS" and detected_fmt == "AFS"):
            warnings.append(
                f"magic {detected_fmt!r} disagrees with extension format {fmt!r}"
            )

    # ------- compression / inner_format -----------------------------------
    compressed = False
    inner_format: Optional[str] = None
    if fmt == "PRS":
        compressed = True
        # Sniff the head for a recognizable inner magic. We never run a
        # full PRS decompress here — that's a job for the per-file
        # endpoint when the user actually opens the file.
        inner = _scan_inner_magic(head)
        if inner is None:
            # Read a slightly bigger window for files whose first literal
            # run is shorter (rare, but cheap to handle).
            bigger = _read_magic(path, PRS_SNIFF_BYTES)
            inner = _scan_inner_magic(bigger)
        if inner is not None:
            inner_format = inner
        else:
            inner_format = "UNKNOWN"
            warnings.append("PRS inner format not detected by head sniff")

    # ------- siblings ------------------------------------------------------
    # Phase A: leave empty. The matcher (Agent 3) will populate this in
    # a later pass; we keep the field present (required by schema) but
    # blank. Future agents wire in via a re-classification step rather
    # than re-walking.
    siblings: list[str] = []

    # ------- matched_textures (Agent 3) -----------------------------------
    # For models (BML / NJ) ask the multi-rule matcher for ranked texture
    # candidates. The matcher is optional; if it's not loaded or it raises
    # on this particular file we silently emit no annotation (the schema
    # field is itself optional, so the entry still validates).
    matched_textures: list[dict] = []
    if _HAS_MATCHER and _match_mod is not None and root is not None:
        if ext in (".bml", ".nj"):
            try:
                ms = _match_mod.match_textures(path, root)
                matched_textures = _match_mod.matches_to_manifest_field(ms, root)
            except Exception as e:  # pragma: no cover - defensive
                warnings.append(f"matcher failed: {type(e).__name__}: {e}")
                matched_textures = []

    entry: dict = {
        "path": rel,
        "size": size,
        "mtime": mtime,
        "extension": ext,
        "magic_hex": magic_hex,
        "magic_ascii": magic_ascii,
        "category": cat,
        "format": fmt,
        "parsable": parsable,
        "siblings": siblings,
    }

    # Schema-optional fields: only emit when meaningful so unchanged
    # files produce identical entries between runs.
    if compressed:
        entry["compressed"] = True
    if inner_format is not None:
        entry["inner_format"] = inner_format
    if matched_textures:
        entry["matched_textures"] = matched_textures
    if warnings:
        entry["warnings"] = warnings

    # Inferred user-facing category (Enemies / Bosses / Weapons / ...).
    # Optional — only emitted when a rule matched, so the field stays
    # absent (rather than equal to the canonical category) when no
    # bucket fits. Tree consumers that read this fall back to the
    # canonical category when None.
    inf = infer_category(rel)
    if inf is not None:
        entry["inferred_category"] = inf

    return entry


def _synthesize_afs_entries(
    afs_path: Path, root: Path, *, cache_dir: Optional[Path] = None,
) -> list[dict]:
    """Walk one ``.afs`` archive and synthesise one AssetEntry per inner blob.

    Each entry's ``path`` is ``<archive_relpath>#<NNNN>_<inner_name>``,
    its ``parent_archive`` field points back at the archive, and its
    ``inner_index`` is the slot index inside the AFS table. Cache files
    are NOT materialised here — that happens lazily on first
    ``GET /api/file/<archive>#<inner>`` (and only the inner blob the
    user asked for, so a manifest rebuild stays fast).

    Errors during AFS parsing are degraded into an empty list; the
    archive itself still appears as a top-level entry via ``classify()``.
    """
    if not _HAS_AFS_READER or _afs_reader_mod is None:
        return []
    try:
        rows = _afs_reader_mod.list_inner_blobs(afs_path)
    except (ValueError, OSError) as e:
        log.warning("AFS list failed for %s: %s", afs_path, e)
        return []
    try:
        archive_rel = _rel_path(afs_path, root)
    except ValueError:
        archive_rel = afs_path.name
    archive_name = afs_path.name
    out: list[dict] = []
    for row in rows:
        idx = int(row.get("index", 0))
        # Build a 4-digit-prefixed inner name so alpha-sort matches
        # archive order. Synth path: "ItemModel.afs#0042_inner.nj"
        # — the `#` separator matches the editor's existing inner-path
        # syntax (see formats/match.py R2).
        inner_name = row.get("name") or f"{afs_path.stem}_{idx:04d}"
        # Strip path components defensively (the AFS filename table
        # MAY embed a relative path; we want a flat synthesised name).
        inner_basename = Path(inner_name).name
        synth = f"{archive_rel}#{idx:04d}_{inner_basename}"
        fmt = row.get("inner_format") or "UNKNOWN"
        cat, parsable = _INNER_FORMAT_TO_CATEGORY.get(fmt, ("unknown", "no"))
        ext_guess = (row.get("inner_ext") or "").lower() or ".unknown"
        warnings: list[str] = []
        if fmt == "UNKNOWN":
            warnings.append("inner format not detected by sniff")
        entry: dict = {
            "path": synth,
            "size": int(row.get("size", 0)),
            # Inherit mtime from the parent archive — there's no
            # per-entry mtime in the AFS spec without the optional
            # filename table's mtime block (which we don't decode).
            "mtime": int(afs_path.stat().st_mtime),
            "extension": ext_guess,
            "magic_hex": row.get("magic_hex", ""),
            "magic_ascii": _ascii_safe(bytes.fromhex(row.get("magic_hex") or "00"), 4),
            "category": cat,
            "format": fmt,
            "parsable": parsable,
            "siblings": [],
            "parent_archive": archive_rel,
            "inner_index": idx,
        }
        if row.get("compressed"):
            entry["compressed"] = True
        if warnings:
            entry["warnings"] = warnings
        # Inferred bucket: dispatch on archive name (e.g. ItemModel.afs
        # → "Weapons / Items"). Pass parent_archive so the rule list
        # can use the archive name as the discriminator.
        inf = infer_category(synth, parent_archive=archive_name)
        if inf is not None:
            entry["inferred_category"] = inf
        out.append(entry)
    return out


def build_manifest(root: Path) -> dict:
    """Walk ``root`` and return the full Manifest dict.

    Output structure (per ``manifest.schema.json``)::

        {
          "version":      1,
          "generated_at": <epoch seconds>,
          "install_root": "C:/.../PSOBB.IO",
          "entries":      [ AssetEntry, ... ]   # sorted by `path`
        }

    Idempotency: ``entries`` is sorted by relative path (case-insensitive
    lexicographic) so re-runs over the same tree produce the same
    serialized JSON modulo ``generated_at``.
    """
    root = root.resolve()
    entries: list[dict] = []
    afs_paths: list[Path] = []
    for p in walk_install(root):
        try:
            entries.append(classify(p, root=root))
        except OSError as e:
            log.warning("classify failed for %s: %s", p, e)
            continue
        # Collect AFS archives for second-pass inner-blob synthesis. We
        # do this in two passes so a failed AFS parse can't disturb the
        # main entry stream — every archive still has its top-level
        # entry from classify() above.
        if p.suffix.lower() == ".afs":
            afs_paths.append(p)

    # Second pass: synthesise per-inner-blob entries for each AFS.
    # Only Item* (weapons/items) and pl?tex (player textures) carry
    # inner blobs we can address individually; the per-blob count is
    # bounded (~400 max per archive) so we can afford to expand all of
    # them.
    afs_inner_count = 0
    for afs_path in afs_paths:
        rows = _synthesize_afs_entries(afs_path, root)
        if rows:
            entries.extend(rows)
            afs_inner_count += len(rows)
    if afs_inner_count:
        log.info(
            "manifest: %d AFS archive(s) expanded to %d inner-blob entries",
            len(afs_paths), afs_inner_count,
        )

    entries.sort(key=lambda e: e["path"].lower())
    return {
        "version":      MANIFEST_VERSION,
        "generated_at": int(time.time()),
        "install_root": str(root).replace("\\", "/"),
        "entries":      entries,
    }


def _newest_mtime_under(root: Path) -> int:
    """Return the largest mtime (epoch seconds) among non-backup files
    under ``root``. Used to decide whether the cached manifest is stale.
    Returns 0 for empty / missing roots.

    Fast path: uses ``os.scandir`` directly so each directory entry only
    pays one stat (the DirEntry batches type+stat from the OS readdir
    call). Going through ``walk_install`` previously cost two stats per
    entry (``Path.is_file`` then ``Path.stat``) on top of the
    ``Path``-object construction overhead — for a 2.3 k-file install
    this is the difference between ~3 ms and ~250 ms.

    Backup-name skip mirrors ``walk_install`` exactly so a freshness
    check sees the same file set the manifest builder will.
    """
    root = root.resolve()
    if not root.exists():
        return 0
    if root.is_file():
        if _is_backup_name(root.name):
            return 0
        try:
            return int(root.stat().st_mtime)
        except OSError:
            return 0
    newest = 0
    stack: list[str] = [str(root)]
    while stack:
        cur = stack.pop()
        try:
            it = os.scandir(cur)
        except OSError:
            continue
        with it:
            for ent in it:
                try:
                    if ent.is_dir(follow_symlinks=False):
                        stack.append(ent.path)
                        continue
                    if not ent.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                if _is_backup_name(ent.name):
                    continue
                try:
                    m = int(ent.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    continue
                if m > newest:
                    newest = m
    return newest


# ---------------------------------------------------------------------------
# Cached _newest_mtime_under (perf win)
#
# Each /api/manifest hit on the warm path used to do a *second* full walk of
# the install root (~9k files) just to decide if the cached JSON was still
# fresh. With this cache the walk runs at most once every
# ``_NEWEST_MTIME_TTL_SECONDS`` regardless of request volume. On a warm
# request we now do exactly one stat (the cache file) plus one timestamp
# compare.
#
# Process-local; not shared across server restarts. The cached value is
# keyed on the resolved install-root path so dev / live mirrors stay
# isolated.
# ---------------------------------------------------------------------------
_NEWEST_MTIME_CACHE: dict[str, tuple[int, float]] = {}
_NEWEST_MTIME_TTL_SECONDS = 60.0  # match user expectations for "saved a file → manifest sees it"


def _newest_mtime_cached(root: Path, *, force: bool = False) -> int:
    """Return ``_newest_mtime_under(root)``, cached for
    ``_NEWEST_MTIME_TTL_SECONDS`` seconds.

    ``force=True`` bypasses the cache and always rewalks. Tests that
    mutate the install tree between calls should pass ``force=True``
    (or call the cache reset helper) to avoid spurious "not yet stale"
    responses inside the TTL window.
    """
    key = str(root.resolve()).replace("\\", "/")
    now = time.time()
    if not force:
        ent = _NEWEST_MTIME_CACHE.get(key)
        if ent is not None:
            value, ts = ent
            if now - ts < _NEWEST_MTIME_TTL_SECONDS:
                return value
    value = _newest_mtime_under(root)
    _NEWEST_MTIME_CACHE[key] = (value, now)
    return value


def _newest_mtime_cache_clear(root: Optional[Path] = None) -> None:
    """Drop the cached mtime for ``root`` (or all roots when omitted)."""
    if root is None:
        _NEWEST_MTIME_CACHE.clear()
        return
    key = str(root.resolve()).replace("\\", "/")
    _NEWEST_MTIME_CACHE.pop(key, None)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically (tmp + os.replace).

    Uses ``sort_keys=True`` so two manifests with identical content
    produce byte-identical files. Any partial write left from a crash
    will be on the ``.tmp`` sibling, never on the live path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, sort_keys=True, indent=2)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
    os.replace(tmp, path)


# In-memory single-slot cache for the parsed manifest dict, keyed on
# the on-disk cache file's stat (mtime_ns, size). Avoids JSON-parsing
# 3.8 MB on every /api/manifest, /api/asset/<path>, /api/manifest/
# categories request. The slot is invalidated implicitly when a manifest
# rebuild changes the cache file's stat.
_PARSED_MANIFEST_CACHE: dict = {}
_PARSED_MANIFEST_LOCK = threading.Lock()

# In-memory single-slot cache for the parsed manifest-LITE dict, keyed on
# the on-disk lite cache file's stat (mtime_ns, size, install_root).
# Mirrors _PARSED_MANIFEST_CACHE so the dominant /api/manifest_lite warm
# path skips the ~12 ms JSON parse of the 1.65 MB lite file. Invalidated
# implicitly when an atomic-rename rewrite changes the file's stat.
_PARSED_LITE_CACHE: dict = {}
_PARSED_LITE_LOCK = threading.Lock()

# Cheap (entry-count, last_built) summary for /api/health, keyed on the
# full manifest cache file's stat. Avoids json.load-ing 3.8 MB on every
# health poll just to report len(entries) + mtime. Re-reads only when the
# cache file's (mtime_ns, size) changes.
_MANIFEST_SUMMARY_CACHE: dict = {}
_MANIFEST_SUMMARY_LOCK = threading.Lock()


def manifest_summary(install_root: Path, cache_dir: Optional[Path] = None) -> dict:
    """Return ``{"entries": int, "last_built": int, "path": str}`` cheaply.

    Powers /api/health. Reports the entry count + last-built epoch of the
    on-disk full manifest WITHOUT json.load-ing the 3.8 MB file on every
    call: the result is memoized on the cache file's (mtime_ns, size) and
    the full parse only happens when the file actually changes (a manifest
    rebuild). ``last_built`` always reflects the live mtime (a cheap stat).

    Never raises — a missing / unreadable cache reports zeroes so the rest
    of /api/health stays green.

    Fast-path ordering avoids a redundant parse: if cache_manifest already
    holds the parsed dict in its in-memory slot for the SAME stat, we read
    len(entries) from there instead of re-opening the file.
    """
    cf = cache_path_for(install_root, cache_dir=cache_dir)
    out = {"entries": 0, "last_built": 0, "path": str(cf)}
    try:
        st = cf.stat()
    except OSError:
        return out
    out["last_built"] = int(st.st_mtime)
    key = (int(st.st_mtime_ns), int(st.st_size))

    with _MANIFEST_SUMMARY_LOCK:
        slot = _MANIFEST_SUMMARY_CACHE.get("slot")
        if slot is not None and slot[0] == key:
            out["entries"] = slot[1]
            return out

    # Miss: derive the entry count once for this revision. Prefer the
    # already-parsed manifest slot (no I/O); fall back to a targeted JSON
    # parse only when the parsed dict isn't resident for this stat.
    entries = None
    with _PARSED_MANIFEST_LOCK:
        pslot = _PARSED_MANIFEST_CACHE.get("slot")
        if pslot is not None and pslot[0][0] == key[0] and pslot[0][1] == key[1]:
            ents = pslot[1].get("entries") if isinstance(pslot[1], dict) else None
            if isinstance(ents, list):
                entries = len(ents)
    if entries is None:
        try:
            with open(cf, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
                entries = len(payload["entries"])
        except (OSError, json.JSONDecodeError) as e:
            log.debug("manifest_summary parse failed: %s", e)
            entries = 0

    with _MANIFEST_SUMMARY_LOCK:
        _MANIFEST_SUMMARY_CACHE["slot"] = (key, int(entries))
    out["entries"] = int(entries)
    return out


def cache_manifest(install_root: Path, cache_dir: Optional[Path] = None,
                   *, force: bool = False) -> dict:
    """Return the cached manifest for ``install_root``, rebuilding if stale.

    The cache lives at ``<cache_dir>/manifest.json``. ``cache_dir``
    defaults to ``./cache/`` relative to this module (matches the
    editor's existing layout). Stale = the cache file is older than the
    newest non-backup file under the install root, or doesn't exist, or
    was generated for a different ``install_root``.

    ``force=True`` bypasses the staleness mtime cache, forcing a full
    walk of the install tree to decide whether the cached manifest can
    still be served. The on-disk cache itself is still respected unless
    its content is corrupt or the install root drifted; the force flag
    only invalidates the in-memory mtime memoization.

    Always returns a Manifest dict — never raises on cache I/O issues
    (a failed read just triggers a rebuild).

    Perf 2026-04-30: results are memoized in-memory by cache-file stat
    so repeated calls on the warm path skip the 3.8 MB JSON parse
    (~25-40 ms cold). The slot is keyed on (mtime_ns, size, install_root)
    so a rebuild atomically invalidates it via the file's new mtime.
    """
    install_root = install_root.resolve()
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    cache_dir = cache_dir.resolve()
    cache_path = cache_dir / "manifest.json"
    install_root_str = str(install_root).replace("\\", "/")

    # Fast path: in-memory cache hit. Single stat() call, then a dict
    # lookup. Skips JSON parse on warm calls — the dominant case.
    if not force and cache_path.exists():
        try:
            st = cache_path.stat()
            slot_key = (int(st.st_mtime_ns), int(st.st_size), install_root_str)
        except OSError:
            slot_key = None
        if slot_key is not None:
            with _PARSED_MANIFEST_LOCK:
                slot = _PARSED_MANIFEST_CACHE.get("slot")
                if slot is not None and slot[0] == slot_key:
                    cached_dict = slot[1]
                    # Still need to verify install-tree freshness: a
                    # newer file in the install dir means our cached
                    # manifest is stale even if the cache file hasn't
                    # been rewritten yet (e.g. another tool dropped a
                    # file). We mirror the staleness check below.
                    newest = _newest_mtime_cached(install_root, force=False)
                    if newest <= int(st.st_mtime):
                        return cached_dict

    cached: Optional[dict] = None
    cache_mtime = 0
    cache_stat: Optional[os.stat_result] = None
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cache_stat = cache_path.stat()
            cache_mtime = int(cache_stat.st_mtime)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("manifest cache unreadable, rebuilding: %s", e)
            cached = None
            cache_mtime = 0

    # Rebuild conditions (any one triggers):
    #   - no usable cache
    #   - install_root drifted (different machine / data dir)
    #   - install_root has a newer file than the cache
    rebuild = False
    if cached is None:
        rebuild = True
    elif cached.get("version") != MANIFEST_VERSION:
        rebuild = True
    elif cached.get("install_root") != install_root_str:
        rebuild = True
    else:
        # Use the cached mtime walker so we don't repeat the second full
        # tree walk on every warm request. The cache TTL (60 s) matches
        # the typical "save → reload" cycle; ?force=1 from the API
        # routes around this when the user explicitly wants a refresh.
        newest = _newest_mtime_cached(install_root, force=force)
        if newest > cache_mtime:
            rebuild = True

    if not rebuild:
        # Populate the in-memory slot for the next call.
        if cache_stat is not None:
            with _PARSED_MANIFEST_LOCK:
                _PARSED_MANIFEST_CACHE["slot"] = (
                    (int(cache_stat.st_mtime_ns), int(cache_stat.st_size),
                     install_root_str),
                    cached,
                )
        return cached  # type: ignore[return-value]

    log.info("rebuilding manifest cache for %s", install_root)
    fresh = build_manifest(install_root)
    try:
        _atomic_write_json(cache_path, fresh)
    except OSError as e:
        log.warning("could not write manifest cache: %s", e)
    # Drop the in-memory slot so the next call repopulates from the
    # newly-written file.
    with _PARSED_MANIFEST_LOCK:
        _PARSED_MANIFEST_CACHE.pop("slot", None)
    # Force a fresh mtime read on the next /api/manifest call so the
    # in-memory cache picks up the just-written manifest.json mtime.
    _newest_mtime_cache_clear(install_root)
    return fresh


def cache_path_for(install_root: Path, cache_dir: Optional[Path] = None) -> Path:
    """Helper for callers (server.py, tests) that need to know where the
    cache file lives — single source of truth."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    return (cache_dir / "manifest.json").resolve()


# ---------------------------------------------------------------------------
# Manifest-lite (Phase 0.5 perf win)
#
# /api/manifest serves a 3.8 MB payload on every cold load; tree.js needs
# only a fraction of that to render the sidebar. The lite shape strips
# every entry to its identity columns:
#
#   path | category | inferred_category | size | parent_archive
#
# At ~50 B per entry x 9357 entries this is ~470 KB raw, ~110 KB gzipped
# — an order of magnitude smaller than the full manifest. Detail
# (matched_textures, warnings, format pill) lazy-loads via
# ``GET /api/asset/<path>`` when the user clicks an entry.
#
# Schema: {version, generated_at, install_root, entries: [LiteEntry]}.
# LiteEntry contains exactly the keys above (parent_archive only when
# the row is an AFS-inner blob); everything else is stripped.
# ---------------------------------------------------------------------------

LITE_KEYS = ("path", "category", "inferred_category", "size", "parent_archive")


def _to_lite_entry(entry: dict) -> dict:
    """Project a full AssetEntry to its lite shape.

    Drops every optional field except the small identity columns that
    drive the asset tree's category / size labels. Returns a fresh
    dict — does not mutate the input.
    """
    out: dict = {}
    for k in LITE_KEYS:
        v = entry.get(k)
        if v is not None and v != "":
            out[k] = v
    return out


def build_manifest_lite_from(full: dict) -> dict:
    """Project a full Manifest to a Manifest-Lite dict in-memory.

    Used internally by ``cache_manifest_lite`` so we can derive the
    lite payload from the on-disk full manifest without a second walk.
    """
    return {
        "version":      MANIFEST_VERSION,
        "generated_at": int(full.get("generated_at") or time.time()),
        "install_root": full.get("install_root", ""),
        "entries":      [_to_lite_entry(e) for e in (full.get("entries") or []) if e],
    }


def cache_manifest_lite(install_root: Path, cache_dir: Optional[Path] = None,
                        *, force: bool = False) -> dict:
    """Return the lite manifest, rebuilding the cache when stale.

    Lite cache lives at ``<cache_dir>/manifest_lite.json``. The cache is
    derived from the on-disk full manifest (``manifest.json``) so a
    rebuild here NEVER does a second filesystem walk — the lite cache is
    invalidated alongside the full cache.

    Returns a dict with the same {version, generated_at, install_root,
    entries[]} shape as ``cache_manifest`` but with each entry trimmed
    to ``LITE_KEYS``.

    Fast path (added 2026-04-25 perf): if the lite cache file already
    exists, version-matches, install-root-matches, and is fresh relative
    to the install tree's newest file, serve it WITHOUT loading the
    full 3.9 MB manifest. This is the dominant call shape — every page
    load hits ``/api/manifest_lite``, and the legacy implementation
    paid ~25 ms warm / ~280 ms cold just to satisfy a freshness check
    that the lite file alone can answer.
    """
    install_root = install_root.resolve()
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    cache_dir = cache_dir.resolve()
    full_path = cache_dir / "manifest.json"
    lite_path = cache_dir / "manifest_lite.json"
    install_root_str = str(install_root).replace("\\", "/")

    # FAST PATH: serve directly from the lite cache when it's fresh
    # relative to the install root. Skips the full manifest load AND
    # the second cache_manifest staleness check entirely.
    if not force and lite_path.exists():
        try:
            lite_st = lite_path.stat()
            lite_mtime = int(lite_st.st_mtime)
            lite_slot_key = (int(lite_st.st_mtime_ns), int(lite_st.st_size),
                             install_root_str)
        except OSError:
            lite_mtime = 0
            lite_slot_key = None
        if lite_mtime > 0:
            newest = _newest_mtime_cached(install_root, force=False)
            if lite_mtime >= newest:
                # In-memory memo hit: return the parsed dict without a
                # json.load of the 1.65 MB lite file. Keyed on the same
                # (mtime_ns, size, install_root) tuple cache_manifest uses,
                # so an atomic-rename rewrite invalidates it automatically.
                if lite_slot_key is not None:
                    with _PARSED_LITE_LOCK:
                        slot = _PARSED_LITE_CACHE.get("slot")
                        if slot is not None and slot[0] == lite_slot_key:
                            return slot[1]  # type: ignore[return-value]
                try:
                    with open(lite_path, "r", encoding="utf-8") as f:
                        cached_lite = json.load(f)
                    if (
                        isinstance(cached_lite, dict)
                        and cached_lite.get("version") == MANIFEST_VERSION
                        and cached_lite.get("install_root") == install_root_str
                        and isinstance(cached_lite.get("entries"), list)
                    ):
                        # Populate the in-memory slot for the next call.
                        if lite_slot_key is not None:
                            with _PARSED_LITE_LOCK:
                                _PARSED_LITE_CACHE["slot"] = (
                                    lite_slot_key, cached_lite,
                                )
                        return cached_lite  # type: ignore[return-value]
                except (OSError, json.JSONDecodeError):
                    pass  # fall through to slow path

    # SLOW PATH: lite cache is missing / stale / wrong-root / corrupt.
    # Rebuild it from the (possibly-also-stale) full manifest.
    full = cache_manifest(install_root, cache_dir=cache_dir, force=force)

    rebuild_lite = True
    if lite_path.exists() and full_path.exists():
        try:
            if lite_path.stat().st_mtime >= full_path.stat().st_mtime:
                with open(lite_path, "r", encoding="utf-8") as f:
                    cached_lite = json.load(f)
                if (
                    isinstance(cached_lite, dict)
                    and cached_lite.get("version") == MANIFEST_VERSION
                    and cached_lite.get("install_root") == full.get("install_root")
                ):
                    return cached_lite  # type: ignore[return-value]
        except (OSError, json.JSONDecodeError):
            rebuild_lite = True

    if rebuild_lite:
        lite = build_manifest_lite_from(full)
        try:
            _atomic_write_json(lite_path, lite)
        except OSError as e:
            log.warning("could not write manifest_lite cache: %s", e)
        # Drop the in-memory slot so the next call repopulates against the
        # just-written file's fresh stat (the rename changed mtime+size).
        with _PARSED_LITE_LOCK:
            _PARSED_LITE_CACHE.pop("slot", None)
        return lite
    # Unreachable; guarded above.
    return build_manifest_lite_from(full)


def lite_cache_path_for(install_root: Path,
                        cache_dir: Optional[Path] = None) -> Path:
    """Path to the cached manifest-lite JSON file."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    return (cache_dir / "manifest_lite.json").resolve()


# Cached path→entry index, keyed on the manifest cache file's
# (mtime_ns, size). Avoids rebuilding the index on every /api/asset/<path>
# click. The cache_manifest() helper re-reads + JSON-parses the on-disk
# manifest.json on every call (no in-memory dict cache there yet — see
# the perf note in cache_manifest), so we anchor our index on the
# stat() of the cache FILE, which is stable until a manifest rebuild
# touches it. (mtime_ns, size) is unique enough — atomic-rename writes
# guarantee both fields change together.
_LOOKUP_INDEX_CACHE: dict = {}
_LOOKUP_INDEX_LOCK = threading.Lock()


def lookup_entry(install_root: Path, target_path: str,
                 cache_dir: Optional[Path] = None) -> Optional[dict]:
    """Return the full AssetEntry for ``target_path`` from the cached
    manifest, or None when the path is unknown.

    Used by ``GET /api/asset/<path>`` to lazy-fetch the full entry shape
    (matched_textures, warnings, format) after the lite manifest seeded
    the asset tree with identity columns only.

    Trades latency on the first click of an entry (one map lookup over
    the in-memory full manifest) for ~3.4 MB removed from the cold-load
    payload.  No network in the warm case — the manifest dict itself is
    already in process memory after the lite build.

    Index: a path→entry dict is built once per cache-file revision and
    held in a process-wide single-slot cache keyed on the cache file's
    (mtime_ns, size). Avoids the O(N) scan over ~9k entries on every
    click, and survives the on-each-call JSON re-read inside
    cache_manifest. The index reseeds automatically when a rebuild
    rewrites manifest.json (mtime / size changes).
    """
    full = cache_manifest(install_root, cache_dir=cache_dir)
    cache_path = cache_path_for(install_root, cache_dir=cache_dir)
    try:
        st = cache_path.stat()
        revision = (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        # If we can't stat the cache file (rare — it was just produced
        # by cache_manifest), fall back to a cheap O(N) scan rather
        # than build an index we can't invalidate.
        for ent in full.get("entries") or []:
            if ent and ent.get("path") == target_path:
                return ent
        return None
    with _LOOKUP_INDEX_LOCK:
        slot = _LOOKUP_INDEX_CACHE.get("slot")
        if slot is None or slot[0] != revision:
            index = {
                ent["path"]: ent
                for ent in (full.get("entries") or [])
                if ent and "path" in ent
            }
            _LOOKUP_INDEX_CACHE["slot"] = (revision, index)
        else:
            index = slot[1]
    return index.get(target_path)
