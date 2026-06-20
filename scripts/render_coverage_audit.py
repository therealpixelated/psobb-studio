"""Render-side coverage audit — exercises /api/model_bundle for every
inner model asset and grades the response.

Companion to ``scripts/coverage_audit.py``. The parser-side audit
tells us which files our format modules can decode; this script tells
us whether the *render* path (server endpoints + texture binder + NJM
animation list) actually surfaces them to the model viewer.

The grade for a row is one of:

  ok                — skinned mesh has bones+meshes, has bound textures
                      (or no NJTL refs at all), and has animations OR is
                      a no-anim prop family.
  ok_no_textures    — skinned mesh ok but the model declares NJTL names
                      and the binder didn't resolve any.
  ok_no_animations  — skinned mesh ok and binder ok, but the family has
                      no NJM siblings AND the family is not on the
                      "props don't need anims" allowlist.
  missing_skinned   — bundle returned but ``skinned`` is null/empty.
  missing_bundle    — endpoint 4xx/5xx'd entirely.
  parse_error       — server returned 400 with a parser error string.
  unsupported_route — endpoint refuses the path form (the AFS '#' hash
                      form is the most common case).

We compare against MODEL_COVERAGE.csv at the end so the operator can see
"parsed fine but no render" deltas.

Usage:
    python scripts/render_coverage_audit.py [--server URL] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urlreq, error as urlerr, parse as urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SERVER = "http://127.0.0.1:8765"
OUT_CSV = ROOT / "_reports" / "render_coverage.csv"

# Families we don't expect to ship NJM siblings.  These are static props
# and effects.  ``ok_no_animations`` is downgraded to ``ok`` for them.
NO_ANIM_PREFIXES = (
    "bm_obj_", "bm_eff_", "set_", "np_obj_", "kazari_",
    "abeniji_", "biri_", "ItemModel", "ItemModelEp4",
    "fs_obj_", "fe_obj_", "kakusi_", "mokei_",
    "lobby_", "lo_", "wp_obj_", "wp_pal_", "tm_obj_", "tm_lobby_",
    "scene_", "ti_", "ts_", "tk_obj_", "fs_e1_obj_",
    "ot_obj_", "ev_obj_", "lp_obj_", "te_obj_",
    # Static NPCs (citizens / shopkeepers) — only NpcApcMot.bml has
    # animations, and only for pxuG/pxuR/pxuS skins. The bm_n_* bodies
    # are static decoration.
    "bm_n_", "bm_nc", "bm_npc_", "bm_gunsinei", "bm_kenkyuw",
    # Story NPCs ("Rico", "Heathcliff" etc.) ship as standalone BMLs
    # without sibling NJMs.
    "rico_", "heathcliff_", "lico_", "rio_", "scarlett_",
)
# Player-class parts (body, head, hair, cap, arm, lwr, ext, fac, acc).
# Animations live in plBdyMot.bml; the per-part BMLs only carry meshes.
# Generated programmatically — every alphabet char A..Z plus their
# lowercase form, since the manifest mixes both.
_PL_BASES = ("bdy", "hed", "hai", "cap", "arm", "lwr", "ext", "fac", "acc",
             "ear", "tail")
NO_ANIM_PREFIXES = NO_ANIM_PREFIXES + tuple(
    f"pl{c}{base}"
    for c in "abcdefghijklmnopqrstuvwxyz"
    for base in _PL_BASES
)

# Families that legitimately have no per-NJTL textures (meshes are
# vertex-coloured or untextured, OR their textures are runtime-only and
# not present anywhere on disk).  Skip the no-textures grade.
NO_TEXTURE_PREFIXES = (
    "bm_eff_",  # effects often have no NJTL

    # plZsmpnj.afs is PSOBB's character-preview archive (the small
    # rotating doll on the class-select / shop / counter screens). Its
    # 434 inners ship NJTL chunks naming textures like ``pxtAb_*`` /
    # ``w32_okhhmf*``, but those names DON'T appear in any on-disk
    # archive — the runtime allocates the textures procedurally from
    # the equipped character's body / cloak / weapon and registers
    # them under the synthetic preview names. The on-disk index can't
    # bind these without re-implementing the runtime's per-character
    # equip simulation, which is out of scope. The viewer renders the
    # untextured mesh, which is correct for an on-disk lookup; the
    # missing textures are flagged to the operator via the manifest's
    # render_audit reports rather than this coverage audit.
    "plzsmpnj",  # NB: family_key is lower-cased before the startswith check
)


def _fetch(server: str, route: str, timeout: float = 30.0) -> Tuple[int, Optional[dict]]:
    url = server.rstrip("/") + route
    req = urlreq.Request(url, headers={"Accept": "application/json"})
    try:
        with urlreq.urlopen(req, timeout=timeout) as r:
            body = r.read()
            try:
                return r.status, json.loads(body.decode("utf-8"))
            except Exception:
                return r.status, None
    except urlerr.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            payload = None
        return e.code, payload
    except (urlerr.URLError, TimeoutError, OSError) as e:
        return -1, {"detail": f"transport: {e}"}


def _enumerate_targets(server: str) -> List[Dict[str, Any]]:
    """Return one target dict per inner-asset.  Each has at least:
       {key, container, inner, ext, infered_category}
    """
    status, mlite = _fetch(server, "/api/manifest_lite")
    if status != 200 or not mlite:
        raise RuntimeError(f"manifest_lite fetch failed: {status}")
    entries = mlite.get("entries") or []
    targets: List[Dict[str, Any]] = []

    seen_keys: set[str] = set()

    # 1) AFS hash-form entries -> 1 target each (route may not support
    # them today; we still log to surface the gap).
    for e in entries:
        if e.get("category") != "model":
            continue
        path = e.get("path") or ""
        if "#" not in path:
            continue
        # Skip non-.nj inners (only .nj observed today).
        if not path.lower().endswith(".nj"):
            continue
        container, inner = path.split("#", 1)
        key = path
        if key in seen_keys:
            continue
        seen_keys.add(key)
        targets.append({
            "key": key,
            "path": path,
            "container": container,
            "inner": inner,
            "ext": ".nj",
            "infered_category": e.get("inferred_category") or "",
        })

    # 2) Top-level BMLs -> N targets each (one per .nj / .xj inner).
    for e in entries:
        if e.get("category") != "model":
            continue
        path = e.get("path") or ""
        if "#" in path:
            continue
        if not path.lower().endswith(".bml"):
            # Standalone .nj at top level is rare in PSOBB.IO/data; cover
            # only what manifest_lite already classified as model.
            if path.lower().endswith(".nj") or path.lower().endswith(".xj"):
                key = path
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                targets.append({
                    "key": key,
                    "path": path,
                    "container": "",
                    "inner": "",
                    "ext": Path(path).suffix.lower(),
                    "infered_category": e.get("inferred_category") or "",
                })
            continue
        # BML — fan out via /api/bml/<path>/list
        status, lst = _fetch(server, f"/api/bml/{urlparse.quote(path, safe='')}/list")
        if status != 200 or not lst:
            # Surface the failure as one row so the operator knows.
            targets.append({
                "key": path,
                "path": path,
                "container": path,
                "inner": "",
                "ext": ".bml",
                "infered_category": e.get("inferred_category") or "",
                "_list_failed": True,
                "_list_status": status,
            })
            continue
        for ent in lst.get("entries") or []:
            name = ent.get("name") or ""
            ext = Path(name).suffix.lower()
            # Geometry models only — animations are graded as a
            # by-product of the parent .nj entry, not a separate row.
            if ext not in (".nj", ".xj"):
                continue
            key = f"{path}#{name}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            targets.append({
                "key": key,
                "path": key,
                "container": path,
                "inner": name,
                "ext": ext,
                "infered_category": e.get("inferred_category") or "",
            })

    return targets


def _grade_mesh_only(target: Dict[str, Any], status: int, body: Optional[dict]) -> Dict[str, Any]:
    """Score one /api/model_mesh response (mesh-only path).

    Used as a fall-through for .xj inners that can't be skinned but still
    render fine as static geometry.  Returns ``ok`` if the mesh blob is
    populated and the binding has at least one resolved texture (or no
    NJTL refs at all).
    """
    out: Dict[str, Any] = {
        "path": target["key"],
        "container": target.get("container") or "",
        "inner": target.get("inner") or "",
        "ext": target.get("ext") or "",
        "infered_category": target.get("infered_category") or "",
        "status": "",
        "note": "",
        "n_textures": 0,
        "n_animations": 0,
        "has_skinned": False,
    }
    if status == -1 or status >= 500:
        out["status"] = "missing_bundle"
        out["note"] = (body or {}).get("detail") or f"transport/5xx ({status})"
        return out
    if status == 404:
        out["status"] = "missing_bundle"
        out["note"] = (body or {}).get("detail") or "404"
        return out
    if status == 400:
        out["status"] = "parse_error"
        out["note"] = ((body or {}).get("detail") or "400")[:200]
        return out
    if not body or not isinstance(body, dict):
        out["status"] = "missing_bundle"
        out["note"] = "no JSON"
        return out

    meshes = body.get("meshes") or []
    if not meshes:
        out["status"] = "missing_skinned"
        out["note"] = "mesh response has no meshes"
        return out

    bd = body.get("binding_data") or {}
    binding = bd.get("binding") or []
    bound = sum(
        1 for r in binding
        if isinstance(r, dict) and not r.get("missing")
        and (r.get("source") or "") not in ("", "unmatched", "unknown")
    )
    njtl = bd.get("njtl") or []
    out["n_textures"] = bound
    if njtl and bound == 0:
        out["status"] = "ok_no_textures"
    else:
        out["status"] = "ok"
    return out


def _grade_scene(target: Dict[str, Any], status: int, body: Optional[dict]) -> Dict[str, Any]:
    """Score one /api/map/asset response."""
    out: Dict[str, Any] = {
        "path": target["key"],
        "container": target.get("container") or "",
        "inner": target.get("inner") or "",
        "ext": target.get("ext") or "",
        "infered_category": target.get("infered_category") or "",
        "status": "",
        "note": "scene asset (Map Editor route)",
        "n_textures": 0,
        "n_animations": 0,
        "has_skinned": False,
    }
    if status == -1 or status >= 500:
        out["status"] = "missing_bundle"
        out["note"] = (body or {}).get("detail") or f"transport/5xx ({status})"
        return out
    if status == 404:
        out["status"] = "missing_bundle"
        out["note"] = (body or {}).get("detail") or "404"
        return out
    if status == 400:
        out["status"] = "parse_error"
        out["note"] = ((body or {}).get("detail") or "400")[:200]
        return out
    if not body or not isinstance(body, dict):
        out["status"] = "missing_bundle"
        out["note"] = "no JSON"
        return out
    meshes = body.get("meshes") or []
    out["has_skinned"] = bool(meshes)  # scene meshes have no skeleton
    if not meshes:
        out["status"] = "missing_skinned"
        out["note"] = "scene asset returned no meshes"
        return out
    # Scene assets share one .xvm sibling at runtime; binding is empty
    # by design.  No animations, ever.
    out["status"] = "ok"
    out["note"] = f"scene asset ({len(meshes)} meshes)"
    return out


def _grade_bundle(target: Dict[str, Any], status: int, body: Optional[dict]) -> Dict[str, Any]:
    """Score one bundle response."""
    out: Dict[str, Any] = {
        "path": target["key"],
        "container": target.get("container") or "",
        "inner": target.get("inner") or "",
        "ext": target.get("ext") or "",
        "infered_category": target.get("infered_category") or "",
        "status": "",
        "note": "",
        "n_textures": 0,
        "n_animations": 0,
        "has_skinned": False,
    }

    if target.get("_list_failed"):
        out["status"] = "missing_bundle"
        out["note"] = f"BML list endpoint failed: {target.get('_list_status')}"
        return out

    if status == -1:
        out["status"] = "missing_bundle"
        out["note"] = (body or {}).get("detail") or "transport"
        return out
    if status >= 500:
        out["status"] = "missing_bundle"
        out["note"] = f"5xx: {(body or {}).get('detail') or status}"
        return out
    if status == 404:
        out["status"] = "missing_bundle"
        out["note"] = (body or {}).get("detail") or "404"
        return out
    if status == 400:
        detail = (body or {}).get("detail") or ""
        # Special-case the AFS / .xj-skinned routing complaints.
        if (
            "extension '.afs'" in detail
            or detail.startswith("unsupported model extension '.afs'")
        ):
            out["status"] = "unsupported_route"
            out["note"] = "AFS '#' bundle path not supported"
            return out
        if "skinned mesh requires .nj inner" in detail or ".xj does not carry" in detail:
            out["status"] = "unsupported_route"
            out["note"] = ".xj inner cannot be skinned"
            return out
        if "BML model requires" in detail:
            out["status"] = "missing_bundle"
            out["note"] = "BML inner missing in URL"
            return out
        out["status"] = "parse_error"
        out["note"] = detail[:200]
        return out

    if not body or not isinstance(body, dict):
        out["status"] = "missing_bundle"
        out["note"] = "no JSON body"
        return out

    sk = body.get("skinned") or {}
    errs = body.get("errors") or {}
    anims = body.get("animations") or {}
    motions = anims.get("motions") or []
    out["n_animations"] = len(motions)

    skin_err = errs.get("skinned")
    if skin_err:
        # Sub-call failed.
        if "extension '.afs'" in str(skin_err):
            out["status"] = "unsupported_route"
            out["note"] = "AFS skinned not supported"
        elif "skinned mesh requires .nj inner" in str(skin_err) or ".xj does not carry" in str(skin_err):
            out["status"] = "unsupported_route"
            out["note"] = ".xj cannot be skinned (use mesh-only path)"
        elif "no entry named" in str(skin_err):
            out["status"] = "missing_bundle"
            out["note"] = f"BML inner not found: {skin_err}"
        elif "BML model requires" in str(skin_err):
            out["status"] = "missing_bundle"
            out["note"] = "bundle handed off without inner"
        else:
            out["status"] = "missing_skinned"
            out["note"] = str(skin_err)[:200]
        return out

    # Skinned shape check.
    has_meshes = isinstance(sk.get("meshes"), list) and len(sk["meshes"]) > 0
    has_bones = isinstance(sk.get("bones"), list) and len(sk["bones"]) > 0
    out["has_skinned"] = bool(has_meshes and has_bones)
    if not has_meshes:
        out["status"] = "missing_skinned"
        out["note"] = "skinned response has no meshes"
        return out

    # Texture binding.
    bd = sk.get("binding_data") or {}
    binding_rows = bd.get("binding") or []
    bound_count = 0
    for row in binding_rows:
        if not isinstance(row, dict):
            continue
        if row.get("missing"):
            continue
        src = row.get("source") or ""
        if src and src not in ("unmatched", "unknown"):
            bound_count += 1
    njtl = bd.get("njtl") or []
    out["n_textures"] = bound_count
    declares_textures = len(njtl) > 0

    # Grade the result.
    status_str = "ok"

    inner_lower = (target.get("inner") or "").lower()
    container_lower = (target.get("container") or "").lower()
    path_lower = (target.get("path") or "").lower()
    family_key = container_lower or inner_lower or path_lower
    no_anim_family = any(family_key.startswith(p) for p in NO_ANIM_PREFIXES)
    no_tex_family = any(family_key.startswith(p) for p in NO_TEXTURE_PREFIXES)

    if declares_textures and bound_count == 0 and not no_tex_family:
        status_str = "ok_no_textures"

    if out["n_animations"] == 0 and not no_anim_family:
        # If we already flagged no-textures, keep the more interesting
        # status; otherwise mark no-animations.
        if status_str == "ok":
            status_str = "ok_no_animations"
        else:
            out["note"] = (out["note"] + " | also: no animations").strip(" |")

    out["status"] = status_str
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--limit", type=int, default=0,
                    help="Audit only the first N targets (smoke test).")
    ap.add_argument("--motion", default="walk",
                    help="Motion to request via include_motion (substring match).")
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    args = ap.parse_args()

    print(f"# server: {args.server}", file=sys.stderr)
    print(f"# enumerating targets...", file=sys.stderr)
    targets = _enumerate_targets(args.server)
    print(f"# targets: {len(targets)}", file=sys.stderr)
    if args.limit:
        targets = targets[: args.limit]
        print(f"# limited to {len(targets)}", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, t in enumerate(targets):
        # Bundle URL — quote the whole path (including '#') so that the
        # FastAPI {path:path} matcher sees it as a single segment.
        if t.get("_list_failed"):
            rows.append(_grade_bundle(t, -1, {"detail": "list-failed"}))
            continue
        # Scene/ paths go through the scene endpoint; their renderer is
        # the Map Editor, not the model viewer.
        path = t["path"]
        if path.startswith("scene/"):
            url_path = "/api/map/asset/" + urlparse.quote(path, safe="/")
            status, body = _fetch(args.server, url_path, timeout=60.0)
            row = _grade_scene(t, status, body)
            rows.append(row)
            if i % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{i+1}/{len(targets)}] {row['status']:18s} {row['path']} "
                      f"(elapsed {elapsed:.1f}s)", file=sys.stderr)
            continue
        url_path = "/api/model_bundle/" + urlparse.quote(path, safe="")
        if args.motion:
            url_path += f"?include_motion={urlparse.quote(args.motion)}"
        status, body = _fetch(args.server, url_path, timeout=60.0)
        row = _grade_bundle(t, status, body)
        # If bundle was unsupported_route (mostly .xj inners that can't
        # be skinned), fall back to mesh-only verification.  This mirrors
        # what the frontend's tryLoadRealMesh does after tryLoadSkinnedMesh
        # fails.
        if (
            row["status"] == "unsupported_route"
            and t.get("ext") == ".xj"
            and t.get("container", "").endswith(".bml")
        ):
            mesh_url = (
                "/api/model_mesh/"
                + urlparse.quote(t["container"], safe="")
                + "?inner="
                + urlparse.quote(t["inner"])
            )
            mst, mb = _fetch(args.server, mesh_url, timeout=60.0)
            mesh_row = _grade_mesh_only(t, mst, mb)
            if mesh_row["status"] in ("ok", "ok_no_textures"):
                # The .xj inner renders fine via the mesh path.  Mark
                # as ok with a note clarifying the route.
                row = mesh_row
                row["note"] = (
                    f"xj mesh-only (no skin); "
                    f"{row.get('note') or 'rendered via /api/model_mesh'}"
                )
        rows.append(row)
        if i % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(targets)}] {row['status']:18s} {row['path']} "
                  f"(elapsed {elapsed:.1f}s)", file=sys.stderr)

    # Aggregate
    by_status: Dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"# done. status breakdown:", file=sys.stderr)
    for s, n in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"    {s:24s} {n:5d}", file=sys.stderr)

    # CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path", "container", "inner", "ext", "infered_category",
        "status", "note", "n_textures", "n_animations", "has_skinned",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"# wrote {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
