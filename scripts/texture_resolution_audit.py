"""Cross-BML texture-name resolution audit.

For every (model, NJTL_entry) reference in the install:
  - resolve in-BML: does THIS BML's inner XVM (or sibling .xvm) carry it?
  - resolve cross-BML: does ANY OTHER BML's NJTL list this name?
  - or is the name unresolved everywhere?

Outputs a CSV at TEXTURE_COVERAGE.csv listing one row per (BML, inner,
NJTL_index, name, resolution) and a Markdown summary at
TEXTURE_COVERAGE.md with per-class counts and 5+ sample resolutions of
each kind.

Run from the editor root::

    python scripts/texture_resolution_audit.py

Reads PSOBB.IO data dir read-only. Uses the same pso-blender PRS
side-loader trick as ``coverage_audit.py``.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import os
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = Path(os.path.expanduser("~/PSOBB.IO/data"))

sys.path.insert(0, str(ROOT))
from formats.bml import parse_bml  # noqa: E402
from formats.iff import parse_iff  # noqa: E402

_PRS_PATH = ROOT / "_modelwork" / "pso-blender" / "pso_blender" / "prs.py"
_spec = importlib.util.spec_from_file_location("pso_blender_prs", str(_PRS_PATH))
_mod = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_mod)  # type: ignore
prs_decompress = _mod.decompress


def _align_up(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


def _safe_decompress(raw: bytes) -> Optional[bytes]:
    """Wrap PRS decompression to tolerate the 'XVMH-stub' truncation case.

    The stub texture replicated across many BMLs (~52 KB compressed,
    ~262 KB decompressed, no terminator) raises IndexError at the very
    end of the stream. We catch that and return whatever was produced.
    """
    try:
        return bytes(prs_decompress(bytearray(raw)))
    except IndexError:
        return None
    except Exception:
        return None


def _parse_njtl(payload: bytes) -> List[str]:
    """Same NJTL decoder as coverage_audit.py."""
    if len(payload) < 8:
        return []
    try:
        elements_ptr, count = struct.unpack_from("<II", payload, 0)
    except struct.error:
        return []
    if count == 0 or count > 256:
        return []
    if elements_ptr + count * 12 > len(payload):
        return []
    names: List[str] = []
    for i in range(count):
        entry_off = elements_ptr + i * 12
        try:
            name_ptr = struct.unpack_from("<I", payload, entry_off)[0]
        except struct.error:
            break
        if name_ptr == 0 or name_ptr >= len(payload):
            names.append("")
            continue
        end = name_ptr
        while end < len(payload) and end < name_ptr + 64 and payload[end] != 0:
            end += 1
        try:
            names.append(payload[name_ptr:end].decode("ascii", errors="replace"))
        except Exception:
            names.append("")
    return names


def _xvm_xvr_count(blob: Optional[bytes]) -> int:
    if not blob or len(blob) < 12 or blob[:4] != b"XVMH":
        return -1
    try:
        return struct.unpack_from("<I", blob, 8)[0]
    except struct.error:
        return -1


@dataclass
class TexRow:
    container: str = ""
    inner: str = ""
    njtl_index: int = -1
    njtl_name: str = ""
    in_bml_xvr_count: int = -1
    resolution: str = ""           # "in_bml" / "cross_bml" / "external_afs" / "unresolved"
    cross_bml_count: int = 0       # how many other BMLs ALSO advertise this name
    cross_bml_first: str = ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default=str(ROOT / "TEXTURE_COVERAGE.csv"))
    ap.add_argument("--out-md", default=str(ROOT / "TEXTURE_COVERAGE.md"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    # Pass 1: gather every NJTL entry across every BML, plus the
    # in-BML xvr_count. Index by name.
    by_name: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # name -> [(bml, inner)]
    refs: List[Tuple[str, str, int, str, int]] = []  # (bml, inner, idx, name, in_bml_xvr_count)

    bmls = sorted(data_dir.glob("*.bml"))
    print(f"# scanning {len(bmls)} BMLs", file=sys.stderr)

    for bi, p in enumerate(bmls):
        if bi % 30 == 0:
            print(f"  pass1 [{bi+1}/{len(bmls)}] {p.name}", file=sys.stderr)
        try:
            buf = p.read_bytes()
            ents = parse_bml(buf)
        except Exception:
            continue
        ct = buf[8] if len(buf) > 8 else 0
        ht_flag = buf[9] if len(buf) > 9 else 0
        # Detect the "lying has_textures" case: header says 1 but every
        # entry has tex_size==0. Use the corrected heuristic: 0x800
        # alignment when no entry actually has a texture.
        any_tex = any(e.tex_size_compressed > 0 for e in ents)
        align = 0x20 if (ht_flag and any_tex) else 0x800

        for e in ents:
            raw = buf[e.offset:e.offset + e.size_compressed]
            if ct == 0x50:
                inner = _safe_decompress(raw)
            else:
                inner = raw
            if inner is None:
                continue
            try:
                chunks = parse_iff(inner)
            except Exception:
                continue
            njtl = next((c for c in chunks if c.type == "NJTL"), None)
            if njtl is None:
                continue
            names = _parse_njtl(njtl.data)
            if not names:
                continue
            # In-BML xvr count
            xvm_bytes: Optional[bytes] = None
            if e.has_texture and e.tex_size_compressed > 0:
                tex_off = e.offset + _align_up(e.size_compressed, align)
                tex_raw = buf[tex_off:tex_off + e.tex_size_compressed]
                xvm_bytes = _safe_decompress(tex_raw)
            xvr_n = _xvm_xvr_count(xvm_bytes)

            for ni, nm in enumerate(names):
                if not nm:
                    continue
                refs.append((p.name, e.name, ni, nm, xvr_n))
                by_name[nm].append((p.name, e.name))

    print(f"# pass1 done: {len(refs)} references over {len(by_name)} unique names", file=sys.stderr)

    # Pass 2: classify each ref. The in_bml xvr_count we have is just
    # the COUNT — we don't have per-XVR names. Heuristic: if an entry's
    # has_textures flag is true AND xvr_count >= the entry's NJTL count,
    # assume the name is in_bml. If there's a sibling _tex.xvm file
    # anywhere, that's external. Cross-BML hit when name shows up in
    # OTHER BMLs.
    rows: List[TexRow] = []
    cls_counter: Counter = Counter()
    for (bml, inner, idx, nm, in_bml_xvr_count) in refs:
        # Other BMLs that mention this name (excluding ourselves)
        other_bmls = sorted({b for (b, _i) in by_name.get(nm, []) if b != bml})

        # Classification (heuristic, based on common-sense):
        if in_bml_xvr_count > 0:
            # Has an in-BML XVM — assume index -> name mapping is positional
            # (NJTL entry i corresponds to xvr i in the XVM). Many BMLs
            # have NJTL count == xvr count, in which case it's in_bml.
            # If xvr_count < njtl_count, the higher-indexed names must
            # come from another archive.
            # We don't know per-name exactly, but this is informative.
            if idx < in_bml_xvr_count:
                resolution = "in_bml"
            elif other_bmls:
                resolution = "cross_bml"
            else:
                resolution = "external_afs_or_missing"
        else:
            # No in-BML XVM. Either cross-BML or external (player AFS).
            if other_bmls:
                resolution = "cross_bml"
            else:
                resolution = "external_afs_or_missing"
        cls_counter[resolution] += 1
        rows.append(TexRow(
            container=bml,
            inner=inner,
            njtl_index=idx,
            njtl_name=nm,
            in_bml_xvr_count=in_bml_xvr_count,
            resolution=resolution,
            cross_bml_count=len(other_bmls),
            cross_bml_first=other_bmls[0] if other_bmls else "",
        ))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"# wrote {out_csv} ({len(rows)} rows)", file=sys.stderr)

    md = io.StringIO()
    md.write("# PSOBB Texture Resolution Audit\n\n")
    md.write(f"- BMLs scanned: {len(bmls)}\n")
    md.write(f"- NJTL references total: {len(refs)}\n")
    md.write(f"- Unique texture names: {len(by_name)}\n\n")

    md.write("## Resolution class distribution\n\n")
    md.write("| Class | Count | % |\n|---|---|---|\n")
    for cls, n in cls_counter.most_common():
        pct = (n / len(refs)) * 100 if refs else 0
        md.write(f"| `{cls}` | {n} | {pct:.1f}% |\n")
    md.write("\n")

    # Cross-BML hot-list: names referenced by 5+ different BMLs
    cross_hot: List[Tuple[str, int]] = []
    for nm, locs in by_name.items():
        bs = {b for (b, _i) in locs}
        if len(bs) >= 5:
            cross_hot.append((nm, len(bs)))
    cross_hot.sort(key=lambda kv: -kv[1])
    md.write(f"## Hot-list: textures referenced by 5+ BMLs ({len(cross_hot)} names)\n\n")
    md.write("These are shared assets; a working texture binder MUST search "
             "across BMLs (or a global texture index) to find them.\n\n")
    md.write("| Name | BMLs | Sample BMLs |\n|---|---|---|\n")
    for nm, n in cross_hot[:30]:
        sample = sorted({b for (b, _i) in by_name[nm]})[:3]
        md.write(f"| `{nm}` | {n} | {', '.join(sample)} |\n")
    md.write("\n")

    # External candidates: names with NO cross-BML and no in_bml resolution
    ext = [r for r in rows if r.resolution == "external_afs_or_missing"]
    md.write(f"## external_afs_or_missing samples ({len(ext)} refs)\n\n")
    md.write("Texture names with no in-BML XVM and not referenced by any "
             "other BML's NJTL. These typically resolve via the player "
             "AFS (`pl?tex.afs`) or are entirely unbundled.\n\n")
    seen: Set[str] = set()
    rendered = 0
    for r in ext:
        if r.njtl_name in seen:
            continue
        seen.add(r.njtl_name)
        md.write(f"- `{r.njtl_name}` (used by `{r.container}#{r.inner}`)\n")
        rendered += 1
        if rendered >= 25:
            break
    md.write("\n")

    Path(args.out_md).write_text(md.getvalue(), encoding="utf-8")
    print(f"# wrote {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
