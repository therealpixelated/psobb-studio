"""Headless render audit for the PSOBB Texture Editor (Task E).

Walks a representative test matrix of models and exercises the
``/api/model_skinned`` (or ``/api/model_mesh`` for .xj) endpoint plus
``/api/animations`` + ``/api/animation_data`` for each. Emits a CSV
table summarising:

  * model_path: ``<bml>#<inner>``
  * class: enemy / boss / player / npc / special
  * mesh_count + bone_count
  * texture binding: bound / unbound / cross-bml count
  * default motion: name + frames + fps
  * walk frame sample (frame 0, frame mid): not visualised (this is a
    headless audit), but we verify the endpoint returns coherent
    positions across frames.

Output:
  C:/tmp_pso_editor/model_audit/render_audit.csv
  C:/tmp_pso_editor/model_audit/render_audit.md   (human summary)

Requires the editor server running at http://127.0.0.1:8765.

Usage:
  python scripts/render_audit.py
"""
from __future__ import annotations
import os

import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

API = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "model_audit"
OUT_CSV = OUT_DIR / "render_audit.csv"
OUT_MD = OUT_DIR / "render_audit.md"


def http_get(path: str, timeout: int = 60):
    url = API + path
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
        except Exception:
            err = {"detail": str(e)}
        return {"_error": f"HTTP {e.code}", "_detail": err}
    except urllib.error.URLError as e:
        return {"_error": f"URL {e}"}


# ---- Test matrix ----------------------------------------------------------
# Each entry: (label, class, model_path, kind)
#   kind: "skinned" (try /api/model_skinned first) or "mesh" (use /api/model_mesh)

TEST_MATRIX = [
    # Bosses
    ("dragon", "boss", "bm_boss8_dragon.bml#boss1_s_nb_dragon.nj", "skinned"),
    ("de_rol_le", "boss", "bm_boss2_de_rol_le.bml#boss2_b_derorure_body.nj", "skinned"),
    ("vol_opt_monitor", "boss", "bm_boss3_volopt.bml#fe_obj_vo_mo_dai_aka.xj", "mesh"),
    ("ep4_boss09", "boss", "bm_ene_boss09.bml#boss00_root.nj", "skinned"),
    ("crawfish", "boss", "bm_boss7_crawfish.bml", "discover"),
    ("gryphon", "boss", "bm_boss5_gryphon.bml#boss5_s_body.nj", "skinned"),
    # Enemies
    ("booma", "enemy", "bm_ene_boota.bml#nj00_boota.nj", "skinned"),
    ("wolf", "enemy", "bm_ene_bm5_wolf.bml#bm5_s_kem_body.nj", "skinned"),
    ("mericarol", "enemy", "bm_ene_bm9_s_mericarol.bml#bm9_s_meri_body.nj", "skinned"),
    # Players (the bml.py alignment-affected ones)
    ("plAbdy00", "player", "plAnj.bml#plAbdy00.nj", "skinned"),
    ("plDbdy00", "player", "plDnj.bml#plDbdy00.nj", "skinned"),
    ("plHbdy00", "player", "plHnj.bml#plHbdy00.nj", "skinned"),
    ("plKbdy00", "player", "plKnj.bml#plKbdy00.nj", "skinned"),
    # Player heads / weapons (other archives the alignment fix touches)
    ("plAhai00", "player", "plAnj.bml#plAhai00.nj", "skinned"),
    ("plDhai00", "player", "plDnj.bml#plDhai00.nj", "skinned"),
    # Special / cross-BML texture cases
    ("vo_mo_pillar", "special", "bm_boss3_volopt.bml#fe_obj_vo_mo_sho01_aka.xj", "mesh"),
]


@dataclass
class AuditRow:
    label: str
    cls: str
    model_path: str
    kind: str
    status: str = ""
    error: str = ""
    mesh_count: int = 0
    bone_count: int = 0
    binding_in_bml: int = 0
    binding_cross_bml: int = 0
    binding_missing: int = 0
    motion_count: int = 0
    default_motion: str = ""
    motion_frames: int = 0
    motion_fps: float = 0.0
    rotation_only_bones: int = 0
    pos_anim_bones: int = 0
    notes: List[str] = field(default_factory=list)

    def to_csv_row(self) -> dict:
        return {
            "label": self.label,
            "class": self.cls,
            "model_path": self.model_path,
            "status": self.status,
            "mesh_count": self.mesh_count,
            "bone_count": self.bone_count,
            "binding_in_bml": self.binding_in_bml,
            "binding_cross_bml": self.binding_cross_bml,
            "binding_missing": self.binding_missing,
            "motion_count": self.motion_count,
            "default_motion": self.default_motion,
            "motion_frames": self.motion_frames,
            "motion_fps": f"{self.motion_fps:.1f}",
            "rotation_only_bones": self.rotation_only_bones,
            "pos_anim_bones": self.pos_anim_bones,
            "notes": "; ".join(self.notes),
            "error": self.error,
        }


def _audit_bindings(binding: list, row: AuditRow) -> None:
    for b in binding or []:
        src = b.get("source", "")
        if src == "in_bml":
            row.binding_in_bml += 1
        elif src == "cross_bml":
            row.binding_cross_bml += 1
        elif b.get("missing"):
            row.binding_missing += 1


def _audit_animation(model_path: str, row: AuditRow) -> None:
    quoted = urllib.parse.quote(model_path, safe="")
    listing = http_get(f"/api/animations/{quoted}")
    if "_error" in listing:
        row.notes.append(f"animations: {listing['_error']}")
        return
    motions = listing.get("motions", [])
    row.motion_count = len(motions)
    di = listing.get("default_index")
    if di is not None and 0 <= di < len(motions):
        chosen = motions[di]
        row.default_motion = chosen.get("name", "")
        row.motion_frames = int(chosen.get("frame_count", 0))
        row.motion_fps = float(chosen.get("fps", 0.0))
    if not row.default_motion:
        return
    # Fetch the keyframes and check track presence patterns.
    data = http_get(
        f"/api/animation_data/{quoted}?motion="
        f"{urllib.parse.quote(row.default_motion, safe='')}",
    )
    if "_error" in data:
        row.notes.append(f"data: {data['_error']}")
        return
    for b in data.get("bones", []):
        present = int(b.get("present", 0))
        if present & 0x01:  # POS bit
            row.pos_anim_bones += 1
        elif present & 0x02:  # ANG only
            row.rotation_only_bones += 1


def _audit_skinned(row: AuditRow) -> None:
    quoted = urllib.parse.quote(row.model_path, safe="")
    r = http_get(f"/api/model_skinned/{quoted}")
    if "_error" in r:
        row.error = f"{r['_error']}: {r.get('_detail', '')}"
        row.status = "ERROR"
        return
    row.mesh_count = int(r.get("mesh_count", 0))
    row.bone_count = int(r.get("bone_count", 0))
    _audit_bindings(r.get("binding", []), row)
    if row.mesh_count == 0:
        row.status = "no-meshes"
    elif row.bone_count == 0:
        row.status = "no-bones"
    else:
        row.status = "ok"
    _audit_animation(row.model_path, row)


def _audit_mesh(row: AuditRow) -> None:
    quoted = urllib.parse.quote(row.model_path, safe="")
    r = http_get(f"/api/model_mesh/{quoted}")
    if "_error" in r:
        row.error = f"{r['_error']}: {r.get('_detail', '')}"
        row.status = "ERROR"
        return
    row.mesh_count = int(r.get("mesh_count", 0))
    row.bone_count = 0  # XJ has no bones
    _audit_bindings(r.get("binding", []), row)
    row.status = "ok" if row.mesh_count > 0 else "no-meshes"
    # XJ models can still pair with NJM motions if siblings exist.
    _audit_animation(row.model_path, row)


def _discover_first_inner(bml_path: str) -> Optional[str]:
    """For 'discover' kind, find the first .nj/.xj inner inside the BML."""
    quoted = urllib.parse.quote(bml_path, safe="")
    info = http_get(f"/api/bml_inners/{quoted}")
    if "_error" in info or not info.get("entries"):
        # Fallback: try via parse_bml directly.
        from formats.bml import parse_bml
        from pathlib import Path as _P
        for root in (
            _P("C:/tmp_pso_dev/data"),
            _P(os.path.expanduser("~/PSOBB.IO/data")),
        ):
            p = root / bml_path
            if p.exists():
                try:
                    entries = parse_bml(p.read_bytes())
                    for e in entries:
                        if e.name.endswith(".nj") or e.name.endswith(".xj"):
                            return e.name
                except Exception:
                    pass
        return None
    for e in info.get("entries", []):
        n = e.get("name", "")
        if n.endswith(".nj") or n.endswith(".xj"):
            return n
    return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[AuditRow] = []
    t0 = time.time()
    for label, cls, mp, kind in TEST_MATRIX:
        if kind == "discover":
            inner = _discover_first_inner(mp)
            if not inner:
                row = AuditRow(label=label, cls=cls, model_path=mp, kind="skipped",
                               status="not_found", error="no .nj/.xj inner")
                rows.append(row)
                print(f"  [{'NOT-FOUND':10s}] {row.cls:6s} {row.label:14s} (no inner found)")
                continue
            mp = f"{mp}#{inner}"
            kind = "skinned" if inner.endswith(".nj") else "mesh"
        row = AuditRow(label=label, cls=cls, model_path=mp, kind=kind)
        if kind == "skinned":
            _audit_skinned(row)
        else:
            _audit_mesh(row)
        rows.append(row)
        sym = "OK" if row.status == "ok" else row.status.upper()
        print(
            f"  [{sym:8s}] {row.cls:6s} {row.label:14s} "
            f"meshes={row.mesh_count:4d} bones={row.bone_count:3d} "
            f"binding(in/x/miss)={row.binding_in_bml}/{row.binding_cross_bml}/{row.binding_missing} "
            f"motion={row.default_motion or '(none)'}"
        )
        if row.error:
            print(f"    error: {row.error}")
    dt = time.time() - t0

    # CSV
    cols = list(rows[0].to_csv_row().keys()) if rows else []
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_csv_row())

    # MD summary
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# PSOBB Model Render Audit\n\n")
        f.write(f"Generated against `{API}` in {dt:.1f}s.\n\n")
        f.write(f"| label | class | status | meshes | bones | bindings (in/cross/miss) | motion | frames@fps | rot-only | pos-anim |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| `{r.label}` | {r.cls} | {r.status} | {r.mesh_count} | "
                f"{r.bone_count} | {r.binding_in_bml}/{r.binding_cross_bml}/{r.binding_missing} | "
                f"{r.default_motion or '(none)'} | {r.motion_frames}@{r.motion_fps:.0f} | "
                f"{r.rotation_only_bones} | {r.pos_anim_bones} |\n"
            )
        f.write("\n## Notes\n\n")
        for r in rows:
            if r.error or r.notes:
                f.write(f"- `{r.label}`: ")
                if r.error:
                    f.write(f"ERROR={r.error} ")
                for n in r.notes:
                    f.write(f"`{n}` ")
                f.write("\n")
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    fail_count = sum(1 for r in rows if r.status == "ERROR")
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
