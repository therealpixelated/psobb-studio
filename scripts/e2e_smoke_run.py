"""End-to-end smoke matrix for the PSOBB asset editor.

Walks the manifest_lite snapshot, samples up to N BMLs per inferred
category, then for every (BML, inner) pair issues:

  - GET /api/bml/<bml>/list
  - GET /api/model_mesh/<bml>?inner=<inner>
  - GET /api/model_bundle/<bml>?inner=<inner>     (when --bundle)
  - GET /api/tile_png/<archive>/<idx>             (one per binding row)

Categories of failure are emitted into the report:

  MESH_EMPTY        mesh_count == 0
  INNER_REJECTED    /api/model_mesh returns 4xx with model-rejection
  BINDING_EMPTY     has_texture=True but binding_data.binding == []
  MATERIAL_SKEW     >95% of submeshes share a single material_id
  TILE_404          tile_png returns 404
  TILE_500          tile_png returns 5xx
  MOTION_EMPTY      BML has .njm inners but motions list is empty

The script is concurrent (ThreadPoolExecutor) and writes a Markdown
report at ``_reports/e2e_smoke_matrix.md`` plus a structured JSON dump
at ``_e2e_work/smoke_matrix.json`` that the pytest layer consumes.

Usage:
  python scripts/e2e_smoke_run.py               # 20 BMLs/category, 16 cats
  python scripts/e2e_smoke_run.py --per-cat 5   # smaller sample
  python scripts/e2e_smoke_run.py --bundle      # also fetch bundle endpoint
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.request
import urllib.parse


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_LITE = ROOT / "_e2e_work" / "manifest_lite.json"
WORK_DIR = ROOT / "_e2e_work"
REPORT_PATH = ROOT / "_reports" / "e2e_smoke_matrix.md"
JSON_DUMP_PATH = WORK_DIR / "smoke_matrix.json"


def _http_get_json(url: str, timeout: float = 60.0, retries: int = 2) -> Tuple[int, Optional[Any], Optional[bytes]]:
    """GET ``url`` and JSON-decode the body.

    Returns (status_code, parsed_json_or_None, raw_bytes_or_None).
    Status -1 indicates a connection / transport failure (retries
    exhausted). The retry loop is needed because the live server can
    momentarily refuse new connections under load (the model parser
    holds the GIL for hundreds of ms on cold parses).
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                try:
                    return resp.status, json.loads(body), body
                except Exception:
                    return resp.status, None, body
        except urllib.error.HTTPError as e:
            try:
                body = e.read()
            except Exception:
                body = b""
            try:
                return e.code, json.loads(body), body
            except Exception:
                return e.code, None, body
        except Exception as e:
            last_err = e
            # Brief backoff between retries.
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return -1, None, None


def _http_head_size(url: str, timeout: float = 30.0, retries: int = 1) -> Tuple[int, int]:
    """GET ``url`` (HEAD-equivalent) and return (status, body_size).

    Same retry-on-transport-error semantics as ``_http_get_json``.
    """
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                return resp.status, len(data)
        except urllib.error.HTTPError as e:
            return e.code, 0
        except Exception:
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return -1, 0


def _derive_archive_from_inner_binding_row(bml: str, inner: str, row: dict) -> Optional[str]:
    """Mirror ``deriveTextureArchive`` on the frontend.

    The viewer fetches tiles via ``/api/tile_png/<archive>/<idx>``. The
    archive is derived from the binding row's ``source`` and the BML
    path. We support:

      * source=in_bml         -> "<bml>#<inner>.xvm"
      * source=cross_bml      -> row['cross_bml']['bml'] + '#' + cross_bml.inner + '.xvm'
      * source=sibling_archive-> row['sibling']['archive']  (path)
      * source=missing        -> None
    """
    src = row.get("source")
    if src == "in_bml":
        return f"{bml}#{inner}.xvm"
    if src == "cross_bml":
        cb = row.get("cross_bml") or {}
        cb_bml = cb.get("bml")
        cb_inner = cb.get("inner")
        if cb_bml and cb_inner:
            return f"{cb_bml}#{cb_inner}.xvm"
    if src == "sibling_archive":
        sib = row.get("sibling") or {}
        ar = sib.get("archive")
        if ar:
            return ar
    return None


def _classify_inner(name: str) -> str:
    """Mirror the frontend ``_classifyInner`` heuristic."""
    low = name.lower()
    if low.endswith(".nj") or low.endswith(".xj"):
        stem = name.rsplit(".", 1)[0].lower()
        if "_lod" in stem or stem.endswith("_low"):
            return "lod"
        if "_shadow" in stem or stem.endswith("_shd"):
            return "shadow"
        if "_hahen" in stem or "_destroy" in stem or "_dest" in stem or "_break" in stem:
            return "destroyed"
        return "primary"
    return "other"


# ---------------------------------------------------------------------------
# Per-BML probe.
# ---------------------------------------------------------------------------


def probe_bml(base_url: str, bml: str, *, do_bundle: bool, max_tile_probes: int = 4) -> dict:
    """Run the full smoke probe against a single BML.

    Returns a dict that matches the JSON-dump schema. Failures are
    classified by code in the ``failures`` field; pure success returns
    an empty failures list.
    """
    out: dict = {
        "bml": bml,
        "category": None,
        "inners": [],            # one entry per inner (name, ext, model_status, ...)
        "njm_count": 0,          # how many .njm inners exist
        "failures": [],          # list[(code, detail)]
        "elapsed_ms": 0,
        "list_status": 0,
    }
    t0 = time.time()

    list_url = f"{base_url}/api/bml/{urllib.parse.quote(bml)}/list"
    code, listing, _ = _http_get_json(list_url)
    out["list_status"] = code
    if code != 200 or not isinstance(listing, dict):
        out["failures"].append(("LIST_FAIL", f"http {code}"))
        out["elapsed_ms"] = int((time.time() - t0) * 1000)
        return out

    entries = listing.get("entries") or []
    nj_inners = []
    njm_inners = []
    for e in entries:
        nm = (e.get("name") or "").strip()
        if not nm:
            continue
        low = nm.lower()
        if low.endswith(".nj") or low.endswith(".xj"):
            nj_inners.append(e)
        elif low.endswith(".njm"):
            njm_inners.append(e)
    out["njm_count"] = len(njm_inners)

    for e in nj_inners:
        inner_name = e["name"]
        inner_low = inner_name.lower()
        ext = inner_low.rsplit(".", 1)[-1] if "." in inner_low else ""
        kind = _classify_inner(inner_name)
        has_tex = bool(e.get("has_texture"))
        rec = {
            "inner": inner_name,
            "ext": ext,
            "kind": kind,
            "has_texture": has_tex,
            "model_status": 0,
            "mesh_count": 0,
            "vert_total": 0,
            "tri_total": 0,
            "binding_rows": 0,
            "binding_missing_rows": 0,
            "material_id_top_share": 0.0,
            "material_id_distinct": 0,
            "tile_probes": [],     # [(archive, idx, status, size)]
            "errors": [],
        }

        mesh_url = (
            f"{base_url}/api/model_mesh/{urllib.parse.quote(bml)}"
            f"?inner={urllib.parse.quote(inner_name)}"
        )
        code, payload, _ = _http_get_json(mesh_url)
        rec["model_status"] = code

        if code != 200 or not isinstance(payload, dict):
            # Pull a useful detail string for triage.
            detail = ""
            if isinstance(payload, dict) and payload.get("detail"):
                detail = str(payload["detail"])
            rec["errors"].append({"stage": "model_mesh", "code": code, "detail": detail})
            # Specific failure-class detection.
            if code == 400 and "is not a model" in detail.lower():
                out["failures"].append(("INNER_REJECTED", f"{inner_name}: {detail}"))
            elif code in (404, 500):
                out["failures"].append(("MESH_EMPTY", f"{inner_name}: http {code}: {detail}"))
            else:
                out["failures"].append(("MESH_EMPTY", f"{inner_name}: http {code}"))
            out["inners"].append(rec)
            continue

        meshes = payload.get("meshes") or []
        rec["mesh_count"] = len(meshes)
        rec["vert_total"] = sum((m.get("vertex_count") or 0) for m in meshes)
        rec["tri_total"] = sum((m.get("triangle_count") or 0) for m in meshes)

        # Known-honest-gap inners: NJTL-only "shared name list" files
        # that other BMLs reference for cross-archive texture lookup.
        # These intentionally ship without an NJCM mesh tree — they are
        # not models, just lists of texture names. Skip the empty-mesh
        # failure for these but keep the inner record so the report
        # still shows them.
        # Reference: bm_ene_common_all.bml#ene_common_all.nj is a 3-tex
        # NJTL of (s064_brad_*) names that Barba/Bardas BMLs cross-ref.
        is_njtl_only_marker = (
            inner_name.lower() in (
                "ene_common_all.nj",
            )
        )
        if not meshes:
            if not is_njtl_only_marker:
                out["failures"].append(("MESH_EMPTY", f"{inner_name}: 0 meshes"))
            else:
                # Tag for the report; not a failure.
                rec.setdefault("notes", []).append("NJTL-only marker (no NJCM)")

        # Binding rows. We need this BEFORE the skew check so we can
        # distinguish "model legitimately uses one texture" (1 binding
        # row + 1 material_id = healthy) from "parser collapsed every
        # strip onto material_id=0" (multi binding rows + 1 material_id).
        binding = (payload.get("binding_data") or {}).get("binding") or []
        rec["binding_rows"] = len(binding)
        rec["binding_missing_rows"] = sum(1 for b in binding if b.get("missing"))

        if meshes:
            ids = Counter(m.get("material_id", 0) for m in meshes)
            top = ids.most_common(1)[0]
            rec["material_id_distinct"] = len(ids)
            rec["material_id_top_share"] = top[1] / len(meshes)
            # MATERIAL_SKEW fires when:
            #   - the model has 4+ submeshes (smaller models commonly
            #     use one trivial material);
            #   - 95%+ of submeshes share a single material_id;
            #   - the model exposes 2+ distinct binding rows;
            #   - AND only ONE distinct material_id appears in the
            #     emitted submeshes. The last clause filters out player
            #     bodies and similar where one texture dominates 90+%
            #     legitimately while a second texture is used for a
            #     small accessory strip (plUbdy00.nj: 92/93 share id=1
            #     with id=0 used by 1 trinket strip — genuine binding
            #     intent, not a propagation bug).
            if (
                len(meshes) >= 4
                and rec["material_id_top_share"] > 0.95
                and rec["binding_rows"] >= 2
                and rec["material_id_distinct"] == 1
            ):
                out["failures"].append((
                    "MATERIAL_SKEW",
                    f"{inner_name}: {top[1]}/{len(meshes)} submeshes share "
                    f"material_id={top[0]} (binding has "
                    f"{rec['binding_rows']} rows, distinct ids={len(ids)})",
                ))

        if has_tex and not binding:
            out["failures"].append((
                "BINDING_EMPTY",
                f"{inner_name}: has_texture but 0 binding rows",
            ))

        # Probe up to N tile_png URLs (skip missing rows, dedupe by archive+idx).
        seen = set()
        probes = 0
        for b in binding:
            if b.get("missing"):
                continue
            archive = _derive_archive_from_inner_binding_row(bml, inner_name, b)
            if not archive:
                continue
            idx = int(b.get("tile_index") or 0)
            key = (archive, idx)
            if key in seen:
                continue
            seen.add(key)
            tile_url = f"{base_url}/api/tile_png/{urllib.parse.quote(archive)}/{idx}"
            tcode, tsize = _http_head_size(tile_url, timeout=20.0)
            rec["tile_probes"].append({
                "archive": archive, "idx": idx, "status": tcode, "size": tsize,
            })
            if tcode == 404:
                out["failures"].append(("TILE_404", f"{inner_name}: {archive}#{idx}"))
            elif 500 <= tcode < 600:
                out["failures"].append(("TILE_500", f"{inner_name}: {archive}#{idx} http {tcode}"))
            elif tcode == 200 and tsize == 0:
                out["failures"].append(("TILE_EMPTY", f"{inner_name}: {archive}#{idx}"))
            probes += 1
            if probes >= max_tile_probes:
                break

        out["inners"].append(rec)

    # Composite-mode sanity (multi-primary BMLs). When 2+ inners are
    # primary (i.e. not _hahen / _lod / _shadow), the frontend picks
    # composite-by-default. We verify EVERY primary inner produced a
    # non-empty mesh — partial composites (some inners 0 meshes) are
    # one of the user-reported failure modes from the warp set.
    primaries = [r for r in out["inners"] if r["kind"] == "primary"]
    if len(primaries) >= 2:
        empty_primaries = [r for r in primaries if r["mesh_count"] == 0]
        if empty_primaries:
            for r in empty_primaries:
                out["failures"].append((
                    "COMPOSITE_PARTIAL",
                    f"{r['inner']}: 0 meshes in multi-primary composite "
                    f"({len(primaries)} primaries total)",
                ))

    # Optional bundle probe (animations + skinned).
    if do_bundle and out["njm_count"] > 0:
        # For animation discovery we only need ONE successful inner; pick
        # the first .nj if any, else first .xj.
        anim_inner = None
        for r in out["inners"]:
            if r["ext"] == "nj":
                anim_inner = r["inner"]
                break
        if anim_inner is None and out["inners"]:
            anim_inner = out["inners"][0]["inner"]
        if anim_inner:
            burl = (
                f"{base_url}/api/model_bundle/{urllib.parse.quote(bml)}"
                f"?inner={urllib.parse.quote(anim_inner)}"
            )
            bcode, bpayload, _ = _http_get_json(burl)
            if bcode == 200 and isinstance(bpayload, dict):
                anims = (bpayload.get("animations") or {})
                mc = int(anims.get("motion_count") or 0)
                if mc == 0:
                    out["failures"].append((
                        "MOTION_EMPTY",
                        f"{anim_inner}: bml has {out['njm_count']} .njm but motion_count=0",
                    ))

    out["elapsed_ms"] = int((time.time() - t0) * 1000)
    return out


# ---------------------------------------------------------------------------
# Sampling driver.
# ---------------------------------------------------------------------------


def sample_bmls(per_cat: int) -> Dict[str, List[str]]:
    """Group manifest entries by inferred_category and pick up to N BMLs each."""
    if not MANIFEST_LITE.exists():
        raise SystemExit(
            f"manifest_lite snapshot missing: {MANIFEST_LITE}\n"
            "Run: curl http://127.0.0.1:8765/api/manifest_lite -o "
            f"{MANIFEST_LITE}"
        )
    mfest = json.loads(MANIFEST_LITE.read_text(encoding="utf-8"))
    by_cat: Dict[str, List[str]] = defaultdict(list)
    for e in mfest.get("entries", []):
        path = e.get("path", "")
        if not path.endswith(".bml"):
            continue
        cat = e.get("inferred_category", "?")
        by_cat[cat].append(path)
    # Stable sample: sort then slice (so reruns hit the same BMLs).
    out: Dict[str, List[str]] = {}
    for cat, paths in by_cat.items():
        paths = sorted(paths)
        if per_cat > 0 and len(paths) > per_cat:
            # Spread the sample across the alphabetical range to dilute
            # name-prefix bias (e.g. "bm_obj_warp*" clustering).
            step = len(paths) / per_cat
            picked = [paths[int(i * step)] for i in range(per_cat)]
            out[cat] = picked
        else:
            out[cat] = paths
    return out


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def render_report(results: List[dict], cat_index: Dict[str, str]) -> str:
    """Render the per-category pass/fail table + failure-mode breakdown."""
    lines = []
    lines.append("# E2E asset smoke matrix")
    lines.append("")
    total_failures = sum(len(r["failures"]) for r in results)
    total_inners = sum(len(r["inners"]) for r in results)
    lines.append(
        f"BMLs probed: {len(results)}  |  inners probed: {total_inners}  |  "
        f"failure rows: {total_failures}"
    )
    lines.append("")

    # Per-category pass/fail.
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        cat = cat_index.get(r["bml"], "?")
        r["category"] = cat
        by_cat[cat].append(r)

    lines.append("## Per-category summary")
    lines.append("")
    lines.append("| Category | BMLs | Inners | Pass | Fail | Pass% | Top fail |")
    lines.append("|---|--:|--:|--:|--:|--:|---|")
    for cat in sorted(by_cat.keys(), key=lambda k: (k == "?", k)):
        recs = by_cat[cat]
        n_b = len(recs)
        n_in = sum(len(r["inners"]) for r in recs)
        n_fail = sum(1 for r in recs if r["failures"])
        n_pass = n_b - n_fail
        pp = (100.0 * n_pass / n_b) if n_b else 0.0
        # Most common failure code in this cat.
        codes = Counter()
        for r in recs:
            for code, _ in r["failures"]:
                codes[code] += 1
        top = codes.most_common(1)
        top_label = f"{top[0][0]}={top[0][1]}" if top else "-"
        lines.append(
            f"| {cat} | {n_b} | {n_in} | {n_pass} | {n_fail} | {pp:.1f}% | {top_label} |"
        )
    lines.append("")

    # Aggregate failure-mode counts across the whole sample.
    overall = Counter()
    for r in results:
        for code, _ in r["failures"]:
            overall[code] += 1
    lines.append("## Failure modes (overall)")
    lines.append("")
    if not overall:
        lines.append("(no failures — everything green)")
    else:
        lines.append("| Code | Count |")
        lines.append("|---|--:|")
        for code, n in overall.most_common():
            lines.append(f"| {code} | {n} |")
    lines.append("")

    # Per-mode top offenders (first 25 of each).
    by_mode: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in results:
        for code, detail in r["failures"]:
            by_mode[code].append((r["bml"], detail))
    lines.append("## Failures by mode (first 25 each)")
    lines.append("")
    for code in sorted(by_mode.keys()):
        rows = by_mode[code]
        lines.append(f"### {code}  ({len(rows)} total)")
        lines.append("")
        for bml, detail in rows[:25]:
            lines.append(f"- `{bml}` — {detail}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bundle", action="store_true",
                    help="also probe /api/model_bundle for animation flags")
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=str(REPORT_PATH))
    ap.add_argument("--json", default=str(JSON_DUMP_PATH))
    args = ap.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Sanity-ping the server (5s).
    code, _, _ = _http_get_json(f"{args.base}/api/health", timeout=5.0)
    if code != 200:
        raise SystemExit(f"server not reachable: GET /api/health -> {code}")

    sampled = sample_bmls(args.per_cat)
    cat_index: Dict[str, str] = {}
    bmls: List[str] = []
    for cat, paths in sampled.items():
        for p in paths:
            cat_index[p] = cat
            bmls.append(p)
    print(f"[smoke] {len(bmls)} BMLs across {len(sampled)} categories", flush=True)

    t0 = time.time()
    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_map = {
            ex.submit(probe_bml, args.base, bml, do_bundle=args.bundle): bml
            for bml in bmls
        }
        done = 0
        for fut in as_completed(future_map):
            try:
                rec = fut.result()
            except Exception as e:
                bml = future_map[fut]
                rec = {
                    "bml": bml, "inners": [], "failures": [("PROBE_EXC", str(e))],
                    "elapsed_ms": 0, "list_status": -1, "njm_count": 0,
                }
            results.append(rec)
            done += 1
            if done % 25 == 0 or done == len(bmls):
                print(
                    f"[smoke] {done}/{len(bmls)} done  "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"[smoke] done in {elapsed:.1f}s", flush=True)

    Path(args.json).write_text(
        json.dumps({"results": results, "cat_index": cat_index}, indent=2),
        encoding="utf-8",
    )

    rep = render_report(results, cat_index)
    Path(args.out).write_text(rep, encoding="utf-8")
    print(f"[smoke] report -> {args.out}", flush=True)
    print(f"[smoke] json   -> {args.json}", flush=True)

    # Quick top-of-stdout summary so the operator can see at a glance.
    overall = Counter()
    for r in results:
        for code, _ in r["failures"]:
            overall[code] += 1
    print("[smoke] failure mode tally:")
    if not overall:
        print("  (none — fully green)")
    else:
        for code, n in overall.most_common():
            print(f"  {code:18s} {n}")


if __name__ == "__main__":
    main()
