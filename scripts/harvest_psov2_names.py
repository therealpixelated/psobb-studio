"""Harvest curated display-names from the psov2 reference asset plugins.

``_reference/psov2/public/js/Asset{Stage,Rooms,Player,Enemies,Weapons,
Objects}.js`` are GPL-licensed (DashGL) ordered JS objects of the shape::

    const AssetEnemies = {
        "Booma" : async function() {
            let ark = await NinjaFile.API.load("gsl_forest01.gsl");
            let gsl = NinjaFile.API.gsl(ark);
            let bml = NinjaFile.API.bml(gsl["bm_ene_re8_b_beast.bml"]);
            ...
        },
        ...
    };

The object KEY is a curated human display name ("Booma"); the function body
``load(...)`` / ``bml[...]`` / ``gsl[...]`` references + ``mdlList[N]``
indices identify which on-disk asset (or AFS inner-blob index) that name
belongs to. Declaration ORDER is the curated sort order.

This script parses those files (text-level, no JS engine needed) into a
committed JSON table::

    data_meta/psov2_names.json = {
      "version": 1,
      "source": "_reference/psov2 (DashGL, GPLv3)",
      "by_file":    { "<lowercased asset filename>": {name, category, order} },
      "by_archive": { "itemmodel.afs#0042":          {name, category, order} }
    }

``manifest.py`` prefers these curated names + order over filename inference.
Only NAMES + ORDER are adopted; the curated category is mapped onto our
richer 18-category structure via ``_PSOV2_CATEGORY_MAP``.

Re-run:  python scripts/harvest_psov2_names.py
(writes data_meta/psov2_names.json; idempotent).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# ``_reference/`` is .gitignored (large, third-party GPL sources) so a fresh
# clone won't have it — the COMMITTED artifact is data_meta/psov2_names.json.
# Both inputs and output are env-overridable so this can run against another
# checkout that still has _reference/ (e.g. from a worktree that doesn't).
_PSOV2_JS_DIR = Path(
    os.environ.get("PSOV2_JS_DIR", _ROOT / "_reference" / "psov2" / "public" / "js")
)
_OUT_PATH = Path(
    os.environ.get("PSOV2_NAMES_OUT", _ROOT / "data_meta" / "psov2_names.json")
)

# psov2's flat 6 "Asset<X>.js" buckets -> our richer canonical categories.
# We adopt psov2's NAMES + ORDER only; the category comes from this map so
# the harvested table slots into OUR 18-category structure, not psov2's
# coarser 6.
_PSOV2_CATEGORY_MAP = {
    "Stage":   "Maps / Terrain",
    "Rooms":   "Maps / Terrain",
    "Player":  "Player Bodies",
    "Enemies": "Enemies",
    "Weapons": "Weapons",
    "Objects": "Objects",
}

_ASSET_FILES = ["Stage", "Rooms", "Player", "Enemies", "Weapons", "Objects"]

# Model/asset file extensions worth keying on (lowercased). We index a
# curated name against every distinct model/archive file it references so a
# manifest entry for any of those files can find the name. Textures (.pvr/
# .pvm/.xvm) are intentionally excluded — they're shared across many named
# entries and would collide. ``.afs`` is excluded too: the whole archive
# (itemmodel.afs / itemtexture.afs) backs hundreds of named weapons, so a
# by-FILE key on it would mis-label the container; weapons are keyed by
# archive INDEX (itemmodel.afs#NNNN) instead.
_KEYABLE_EXTS = (".nj", ".bml", ".rel", ".bin", ".rlc", ".njm")


def _iter_top_level_entries(src: str):
    """Yield (display_name, body_text, order) for each top-level object key.

    The Asset<X>.js objects are one flat level of ``"Name" : async ...``
    members. We locate each quoted key followed by ``:`` and ``async``,
    then slice its body up to the next sibling key (or end of object). Pure
    text scan — robust to both ``function()`` and ``() =>`` member forms.
    """
    # Match a member header:  "Display Name" : async  (function|()=>)
    header = re.compile(
        r'"((?:[^"\\]|\\.)+)"\s*:\s*async\b', re.MULTILINE
    )
    matches = list(header.finditer(src))
    for order, m in enumerate(matches):
        name = m.group(1)
        body_start = m.end()
        body_end = matches[order + 1].start() if order + 1 < len(matches) else len(src)
        yield name, src[body_start:body_end], order


_LOAD_RE = re.compile(r'\.load\(\s*["\']([^"\']+)["\']\s*\)')
_INNER_RE = re.compile(r'(?:bml|gsl|afs|prc)\[\s*["\']([^"\']+)["\']\s*\]')
_MDL_IDX_RE = re.compile(r'mdlList\[\s*(\d+)\s*\]')
_AFS_LOAD_RE = re.compile(r'\.load\(\s*["\']([^"\']+\.afs)["\']\s*\)', re.IGNORECASE)


def _basename_lower(p: str) -> str:
    """Last path component, lowercased (psov2 paths use '/' and mixed case)."""
    return p.replace("\\", "/").split("/")[-1].lower()


def harvest() -> dict:
    by_file: dict[str, dict] = {}
    by_archive: dict[str, dict] = {}

    for stem in _ASSET_FILES:
        path = _PSOV2_JS_DIR / f"Asset{stem}.js"
        if not path.is_file():
            print(f"  skip (missing): {path}")
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        category = _PSOV2_CATEGORY_MAP[stem]
        n_file = 0
        n_arch = 0
        for name, body, order in _iter_top_level_entries(src):
            record = {"name": name, "category": category, "order": order}

            # --- archive-index keys (Weapons): mdlList[N] into itemmodel.afs.
            afs_loads = [_basename_lower(a) for a in _AFS_LOAD_RE.findall(body)]
            mdl_idxs = [int(x) for x in _MDL_IDX_RE.findall(body)]
            if mdl_idxs and any("itemmodel" in a for a in afs_loads):
                idx = mdl_idxs[0]
                key = f"itemmodel.afs#{idx:04d}"
                # First writer wins (preserve declaration order on collision).
                by_archive.setdefault(key, record)
                n_arch += 1

            # --- filename keys: every model/archive file the entry pulls in.
            refs = set()
            for f in _LOAD_RE.findall(body):
                refs.add(_basename_lower(f))
            for f in _INNER_RE.findall(body):
                refs.add(_basename_lower(f))
            for ref in refs:
                ext = "." + ref.rsplit(".", 1)[-1] if "." in ref else ""
                if ext not in _KEYABLE_EXTS:
                    continue
                # First curated name to claim a file wins (preserves order).
                if ref not in by_file:
                    by_file[ref] = record
                    n_file += 1
        print(f"  Asset{stem}.js: {n_file} file-keys, {n_arch} archive-keys")

    return {
        "version": 1,
        "source": "_reference/psov2 (DashGL Ninja plugin, GPLv3) — names + order only",
        "category_map": _PSOV2_CATEGORY_MAP,
        "by_file": dict(sorted(by_file.items())),
        "by_archive": dict(sorted(by_archive.items())),
    }


def main() -> int:
    print("Harvesting psov2 display-names ...")
    table = harvest()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(
        json.dumps(table, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {_OUT_PATH} "
        f"({len(table['by_file'])} file-keys, "
        f"{len(table['by_archive'])} archive-keys)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
