#!/usr/bin/env python3
"""Headless batch upscale: walk PSOBB.IO/data candidates → upscale → deploy to install.

Pipeline per file:
  1. Sync from LIVE (~/PSOBB.IO/data) → DEV (C:/tmp_pso_dev/data) when DEV
     stub is empty / smaller than LIVE.
  2. For .xvm/.prs files: GET /api/tiles/{filename} → upscale each → repack → deploy.
     For .afs files: GET /api/afs/{archive}/list → for each inner XVM, upscale every
     tile and POST /api/repack_afs_inner with deploy=True to splice + atomic-replace
     the parent archive in one shot.
     For .bml files: GET /api/bml/{bml}/list → for each entry where has_texture=True,
     upscale every tile of `<bml>#<entry>.xvm` and POST /api/repack_bml_inner with
     deploy=True to PRS-recompress + splice into the parent BML.

Skips on errors and continues. Logs per-file outcome to stdout (unbuffered when
run with `python -u`).

Usage:
  python -u scripts/batch_upscale_install.py [scale=2] [filter=xvm,prs,bml,afs]

Default: scale=2, filter=xvm.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests


def _path_encode(filename: str) -> str:
    """URL-encode a filename for use in a path segment.

    `requests.get(f'{BASE}/api/tiles/{filename}')` strips '#' as a URL
    fragment delimiter — that breaks AFS-inner paths like 'ItemKT.afs#0000'.
    Encode '#' (and other reserved chars) in the path component."""
    return quote(filename, safe="")

# Force line-buffered stdout so background runs produce a live log.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

BASE = "http://127.0.0.1:8765"
LIVE_DATA = Path(os.path.expanduser("~/PSOBB.IO/data"))
DEV_DATA = Path("C:/tmp_pso_dev/data")
CANDIDATES_FILE = Path(__file__).parent / "upscale_candidates.txt"

UPSCALE_TIMEOUT = 600  # 10 min per tile (cascade may run multiple passes at 4x)
REPACK_TIMEOUT = 120
DEPLOY_TIMEOUT = 60

# Subdirs we'll search for a candidate's LIVE source. Order matters — root
# wins when the same name exists in both root and a subdir.
SEARCH_SUBDIRS: tuple[str, ...] = ("", "scene")

# Per-file tracking: filename → original LIVE subdir ("" for root, "scene", …).
# Populated by sync_to_dev(), consumed by post-deploy relocation in
# repack_and_deploy(). Module-global so process_file() doesn't have to thread
# the value through the whole call graph; the driver runs files serially so
# concurrency isn't a concern.
_ORIGINAL_SUBDIR: dict[str, str] = {}


def sync_to_dev(filename: str) -> bool:
    """Copy LIVE/<sub>/<filename> → DEV/<filename> (always at DEV root).

    The server's `/api/tiles` and `/api/repack` endpoints validate filenames
    as bare basenames (no path components), so files that live in
    `data/scene/` etc. are inaccessible via those endpoints unless we stage
    them at the DEV root. We track the original subdir in `_ORIGINAL_SUBDIR`
    so the post-deploy step can restore the file to its original location
    inside LIVE.

    Returns True if DEV now has real content for `filename`. Records the
    discovered subdir in `_ORIGINAL_SUBDIR[filename]`.
    """
    for sub in SEARCH_SUBDIRS:
        src = LIVE_DATA / sub / filename if sub else LIVE_DATA / filename
        if not src.exists():
            continue

        # Always stage at DEV root so the API can address it by bare name.
        dst = DEV_DATA / filename

        # Collision guard: if we're staging a subdir file into DEV root,
        # refuse if a different file with the same name already lives at
        # the root in either LIVE or DEV. This shouldn't happen with the
        # current candidate list (verified zero collisions) but guard
        # anyway so a future candidate doesn't silently overwrite an
        # unrelated root file.
        if sub:
            live_root_clash = (LIVE_DATA / filename).exists()
            dev_root_clash = dst.exists() and not _is_our_staged_copy(dst, src)
            if live_root_clash or dev_root_clash:
                print(
                    f"  SKIP: '{filename}' lives in LIVE/{sub}/ but a different "
                    f"file with the same name exists at the root "
                    f"(live={live_root_clash} dev={dev_root_clash})",
                    flush=True,
                )
                return False

        dst.parent.mkdir(parents=True, exist_ok=True)
        # Sync if dest missing OR dest much smaller than source (stub).
        if not dst.exists() or dst.stat().st_size < src.stat().st_size * 0.5:
            shutil.copy2(src, dst)
        _ORIGINAL_SUBDIR[filename] = sub
        return True
    return False


def _is_our_staged_copy(dst: Path, src: Path) -> bool:
    """Best-effort check: is `dst` already a copy of `src` we staged earlier?

    Returns True if the two files have identical size + mtime (which is what
    `shutil.copy2` produces). Used by the collision guard so re-running the
    driver against an interrupted previous run doesn't trip the guard.
    """
    try:
        ds = dst.stat()
        ss = src.stat()
    except OSError:
        return False
    return ds.st_size == ss.st_size and abs(ds.st_mtime - ss.st_mtime) < 2.0


def get_tiles(filename: str) -> list[dict] | None:
    try:
        r = requests.get(f"{BASE}/api/tiles/{_path_encode(filename)}", timeout=60)
        if r.status_code != 200:
            print(f"    /api/tiles {r.status_code}: {r.text[:120]}", flush=True)
            return None
        return r.json().get("tiles", [])
    except Exception as e:
        print(f"    /api/tiles error: {e}", flush=True)
        return None


def upscale_tile(filename: str, tile_index: int, scale: int, model: str) -> str | None:
    body = {
        "filename": filename,
        "tile_index": tile_index,
        "model": model,
        "scale": scale,
        "keep_native_dims": False,
    }
    try:
        r = requests.post(f"{BASE}/api/upscale", json=body, timeout=UPSCALE_TIMEOUT)
        if r.status_code != 200:
            print(f"    upscale tile {tile_index} HTTP {r.status_code}: {r.text[:120]}")
            return None
        return r.json().get("out_b64")
    except Exception as e:
        print(f"    upscale tile {tile_index} error: {e}")
        return None


def repack_and_deploy(filename: str, edits: list[dict]) -> bool:
    """edits = [{tile_index, png_b64}, ...]

    Three deploy paths, dispatched on `filename` shape:

      1. AFS inner (`<archive>.afs#NNNN`): POST /api/repack_afs_inner with
         deploy=True. The server splices the rebuilt XVM back into the
         parent AFS, re-PRS-compresses if the source was PRS, and
         atomic-replaces both LIVE and DEV with a backup. Single round-trip,
         no separate /api/deploy call needed.

      2. BML inner (`<bml>.bml#<entry>.xvm`): POST /api/repack_bml_inner
         with deploy=True. Same idea — server PRS-recompresses the rebuilt
         XVMH bytes, splices into the matched BML entry's texture slot, and
         atomic-replaces both LIVE and DEV.

      3. Plain XVM/PRS (top-level `<file>.xvm` / `<file>.prs`): legacy two-
         step. /api/repack writes the rebuilt bytes to DEV/<filename> (root),
         /api/deploy copies DEV → LIVE. For files originally in a subdir
         (e.g. `scene/foo.xvm`), we then *relocate* LIVE/<filename> →
         LIVE/<subdir>/<filename> so the file ends up where the game
         actually loads it from.
    """
    # AFS-inner branch — single-shot splice + deploy via the new endpoint.
    # /api/repack 404s on '#'-paths; this is the canonical path for inners.
    if "#" in filename:
        outer, _, inner_tail = filename.partition("#")
        if outer.lower().endswith(".afs"):
            try:
                inner_index = int(inner_tail)
            except ValueError:
                print(f"    AFS inner_index parse failed: {inner_tail!r}")
                return False
            body = {
                "archive": outer,
                "inner_index": inner_index,
                "tiles": edits,
                "deploy": True,
            }
            try:
                r = requests.post(
                    f"{BASE}/api/repack_afs_inner",
                    json=body,
                    timeout=REPACK_TIMEOUT,
                )
                if r.status_code != 200:
                    print(
                        f"    /api/repack_afs_inner HTTP {r.status_code}: "
                        f"{r.text[:200]}"
                    )
                    return False
            except Exception as e:
                print(f"    /api/repack_afs_inner error: {e}")
                return False
            return True
        if outer.lower().endswith(".bml"):
            body = {
                "bml": outer,
                "inner_name": inner_tail,
                "tiles": edits,
                "deploy": True,
            }
            try:
                r = requests.post(
                    f"{BASE}/api/repack_bml_inner",
                    json=body,
                    timeout=REPACK_TIMEOUT,
                )
                if r.status_code != 200:
                    print(
                        f"    /api/repack_bml_inner HTTP {r.status_code}: "
                        f"{r.text[:200]}"
                    )
                    return False
            except Exception as e:
                print(f"    /api/repack_bml_inner error: {e}")
                return False
            return True
        # Unknown '#'-shape — fall through to /api/repack which will 4xx
        # cleanly. (No-op today; defensive against future inner formats.)

    body = {"filename": filename, "tiles": edits, "deploy": True}
    try:
        r = requests.post(f"{BASE}/api/repack", json=body, timeout=REPACK_TIMEOUT)
        if r.status_code != 200:
            print(f"    /api/repack HTTP {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"    /api/repack error: {e}")
        return False

    # Now copy DEV → LIVE
    try:
        r = requests.post(f"{BASE}/api/deploy/{_path_encode(filename)}", json={"create_backup": True}, timeout=DEPLOY_TIMEOUT)
        if r.status_code != 200:
            print(f"    /api/deploy HTTP {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"    /api/deploy error: {e}")
        return False

    # Post-deploy: if the file originated in a subdir, move LIVE/foo.xvm
    # back to LIVE/<sub>/foo.xvm so the game can find it.
    sub = _ORIGINAL_SUBDIR.get(filename, "")
    if sub:
        try:
            _relocate_after_deploy(filename, sub)
        except Exception as e:
            print(f"    relocate to {sub}/ error: {e}", flush=True)
            return False

    return True


def _relocate_after_deploy(filename: str, sub: str) -> None:
    """Move LIVE/<filename> → LIVE/<sub>/<filename> after a successful deploy.

    Backs up the existing LIVE/<sub>/<filename> with a `.pre_promote_subdir_<TS>`
    suffix (matching the server's backup naming convention) so the original
    subdir-resident file isn't lost. Also cleans the DEV/<filename> root
    staging copy so a later root-only candidate with the same name doesn't
    accidentally pick up the scene/* file's bytes.
    """
    live_root = LIVE_DATA / filename
    live_target = LIVE_DATA / sub / filename
    if not live_root.exists():
        raise RuntimeError(f"LIVE/{filename} missing after deploy — server didn't write it?")

    live_target.parent.mkdir(parents=True, exist_ok=True)
    if live_target.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        bak = live_target.with_suffix(live_target.suffix + f".pre_promote_subdir_{ts}")
        counter = 0
        while bak.exists():
            counter += 1
            bak = live_target.with_suffix(
                live_target.suffix + f".pre_promote_subdir_{ts}_{counter}",
            )
        shutil.copy2(live_target, bak)

    # Atomic rename when source and dest share a volume; on Windows
    # os.replace clobbers the dest, which is what we want.
    import os as _os
    _os.replace(live_root, live_target)
    print(
        f"    relocated LIVE/{filename} → LIVE/{sub}/{filename} "
        f"({live_target.stat().st_size // 1024} KiB)",
        flush=True,
    )

    # Clean the DEV root staging copy so the next driver run / a future
    # root-named candidate doesn't see a stale subdir file masquerading
    # as a root file. Best-effort — failure here doesn't break the deploy.
    dev_root = DEV_DATA / filename
    if dev_root.exists():
        try:
            dev_root.unlink()
        except OSError as e:
            print(f"    (warn) could not clean DEV/{filename}: {e}", flush=True)


def process_file(filename: str, scale: int, model: str, is_inner: bool = False) -> bool:
    if not is_inner:
        print(f"\n[{filename}] sync...", flush=True)
        if not sync_to_dev(filename):
            print("  SKIP: source not found in LIVE", flush=True)
            return False

    tiles = get_tiles(filename)
    if tiles is None:
        return False
    if not tiles:
        print("  SKIP: no tiles")
        return False
    print(f"  {len(tiles)} tiles")

    edits: list[dict] = []
    for t in tiles:
        idx = t["index"]
        out_b64 = upscale_tile(filename, idx, scale, model)
        if out_b64 is None:
            print(f"  ABORT (tile {idx} failed)")
            return False
        edits.append({"tile_index": idx, "png_b64": out_b64})
        print(f"    tile {idx} ok ({len(out_b64) // 1024} KiB)")

    print("  repack + deploy...")
    if repack_and_deploy(filename, edits):
        # For an inner (`<archive>#<inner>`) the on-disk artifact is the
        # parent archive in LIVE; for a top-level file we may have been
        # relocated into a subdir.
        if "#" in filename:
            outer = filename.split("#", 1)[0]
            final = LIVE_DATA / outer
            sz = final.stat().st_size if final.exists() else 0
            print(f"  OK ({sz // 1024} KiB parent archive → {outer})")
        else:
            sub = _ORIGINAL_SUBDIR.get(filename, "")
            final = LIVE_DATA / sub / filename if sub else LIVE_DATA / filename
            sz = final.stat().st_size if final.exists() else 0
            loc = f"{sub}/{filename}" if sub else filename
            print(f"  OK ({sz // 1024} KiB deployed → {loc})")
        return True
    return False


def list_afs_inners(filename: str) -> list[dict] | None:
    """Return inner archive entries inside an .afs container.
    Each entry: {index, name, size, inner_ext, ...}
    """
    try:
        r = requests.get(f"{BASE}/api/afs/{_path_encode(filename)}/list", timeout=60)
        if r.status_code != 200:
            print(f"    /api/afs/list HTTP {r.status_code}: {r.text[:120]}", flush=True)
            return None
        data = r.json()
        entries = data.get("entries") or data.get("inners") or []
        return entries
    except Exception as e:
        print(f"    /api/afs/list error: {e}", flush=True)
        return None


def process_afs(filename: str, scale: int, model: str) -> bool:
    """Upscale every inner XVM inside an AFS container.

    Per server.py:_materialize_inner_for_extract / _parse_afs_inner_name, the
    inner path syntax is `<afs>#NNNN` (4-digit index) — NOT the inner's own
    name. Full names like `ItemKT_0000.xvm` are rejected with HTTP 400.
    """
    print(f"\n[{filename}] (AFS) sync...", flush=True)
    if not sync_to_dev(filename):
        print("  SKIP: source not found in LIVE", flush=True)
        return False

    entries = list_afs_inners(filename)
    if not entries:
        print("  SKIP: no inners listed", flush=True)
        return False

    # Filter to image-y inners (XVM / PRS) — skip NJ / NJM / RAW.
    image_entries = []
    for e in entries:
        ext = (e.get("inner_ext") or "").lower()
        if ext in (".xvm", ".prs"):
            image_entries.append(e)
    print(f"  {len(image_entries)} image inners (of {len(entries)} total)", flush=True)

    any_processed = False
    any_failed = False
    for e in image_entries:
        idx = e.get("index")
        if idx is None:
            continue
        inner_path = f"{filename}#{idx:04d}"
        print(f"  inner [{idx}]: {e.get('name')}", flush=True)
        if process_file(inner_path, scale, model, is_inner=True):
            any_processed = True
        else:
            any_failed = True

    return any_processed


def list_bml_inners(filename: str) -> list[dict] | None:
    """Return inner entries inside a .bml container.

    Each entry: {name, size_compressed, size_decompressed, has_texture,
    tex_size_compressed}. Only entries with `has_texture=True` carry a
    per-entry XVM texture archive that's addressable as `<bml>#<name>.xvm`
    (the inner-name syntax that /api/repack_bml_inner expects).
    """
    try:
        r = requests.get(
            f"{BASE}/api/bml/{_path_encode(filename)}/list", timeout=60,
        )
        if r.status_code != 200:
            print(
                f"    /api/bml/list HTTP {r.status_code}: {r.text[:120]}",
                flush=True,
            )
            return None
        data = r.json()
        return data.get("entries") or []
    except Exception as e:
        print(f"    /api/bml/list error: {e}", flush=True)
        return None


def process_bml(filename: str, scale: int, model: str) -> bool:
    """Upscale every textured inner XVM inside a BML container.

    BML structure (per server.py:_extract_bml_inner_bytes): each entry is
    a paired `<name>.nj`/`<name>.xj` mesh + an optional texture archive at
    `<name>.xvm`. We only repack textures here — mesh-payload edits go
    through /api/import/replace which has different semantics.

    For each textured entry we POST /api/repack_bml_inner with deploy=True;
    the server PRS-recompresses the rebuilt XVMH bytes and atomic-replaces
    the parent BML in LIVE + DEV. Concurrent inner repacks of the same
    BML are serialized server-side (per-BML lock).
    """
    print(f"\n[{filename}] (BML) sync...", flush=True)
    if not sync_to_dev(filename):
        print("  SKIP: source not found in LIVE", flush=True)
        return False

    entries = list_bml_inners(filename)
    if entries is None:
        return False
    if not entries:
        print("  SKIP: no inners listed", flush=True)
        return False

    textured = [e for e in entries if e.get("has_texture")]
    print(
        f"  {len(textured)} textured inners (of {len(entries)} total)",
        flush=True,
    )
    if not textured:
        print("  SKIP: no textured inners (mesh-only BML)", flush=True)
        return False

    any_processed = False
    any_failed = False
    for e in textured:
        ent_name = e.get("name")
        if not ent_name:
            continue
        # Inner-name syntax for /api/repack_bml_inner is '<entry>.xvm';
        # this also matches the path /api/tiles + /api/upscale expect.
        inner_path = f"{filename}#{ent_name}.xvm"
        print(f"  inner: {ent_name} (.xvm)", flush=True)
        if process_file(inner_path, scale, model, is_inner=True):
            any_processed = True
        else:
            any_failed = True

    return any_processed


def main() -> int:
    scale = int(sys.argv[1]) if len(sys.argv) >= 2 else 2
    filt = sys.argv[2] if len(sys.argv) >= 3 else "xvm"

    # Model selection: server cascades for scales > native model max.
    # x4 model with scale=8 cascades through 4x then resizes; quality acceptable.
    if scale <= 2:
        model = "realesr-animevideov3-x2"
    elif scale <= 4:
        model = "realesr-animevideov3-x4"
    else:
        # 6, 8, 12, 16 — use x4 model + cascade
        model = "realesr-animevideov3-x4"

    extensions = filt.split(",")

    if not CANDIDATES_FILE.exists():
        print(f"missing {CANDIDATES_FILE}", file=sys.stderr)
        return 1
    candidates = [c.strip() for c in CANDIDATES_FILE.read_text().splitlines() if c.strip()]
    candidates = [c for c in candidates if c.split(".")[-1].lower() in extensions]
    print(f"Processing {len(candidates)} files (scale={scale}, model={model}, filter={filt})", flush=True)

    ok_count = 0
    fail_count = 0
    skip_count = 0
    t0 = time.time()
    for i, fn in enumerate(candidates, 1):
        print(f"\n=== [{i}/{len(candidates)}] {fn} ===", flush=True)
        try:
            low = fn.lower()
            if low.endswith(".afs"):
                ok = process_afs(fn, scale, model)
            elif low.endswith(".bml"):
                ok = process_bml(fn, scale, model)
            else:
                ok = process_file(fn, scale, model)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}", flush=True)
            fail_count += 1
        elapsed = time.time() - t0
        eta = (elapsed / i) * (len(candidates) - i)
        print(f"  [progress] {i}/{len(candidates)} ok={ok_count} fail={fail_count} elapsed={int(elapsed)}s eta={int(eta)}s", flush=True)

    print(f"\n=== DONE ok={ok_count} fail={fail_count} skip={skip_count} elapsed={int(time.time()-t0)}s ===", flush=True)
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
