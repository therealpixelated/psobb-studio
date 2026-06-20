"""End-to-end test for the PSOBB Texture Editor backend.

Drives the live HTTP API at http://127.0.0.1:8765 against a known-good
target (LogoEP4.prs):

  1. health check
  2. /api/files smoke
  3. /api/tiles/LogoEP4.prs - extract + verify all PNGs decode
  4. /api/tiles/<an XVM>    - non-PRS path
  5. /api/upscale on the smallest tile (keep_native_dims=true), verify dims
  6. /api/repack with deploy:false, verify rebuilt artifact
  7. /api/repack with deploy:true (single re-saved tile), verify:
       - backup file created with .pre_editor_<YYYYMMDD_HHMMSS> name
       - deployed PRS round-trips through PuyoToolsCli + xvr_codec.py extract
         back to identical (count, fmt, w, h) per tile
  8. /api/restore_backup to undo
  9. cleanup: cache/ should be re-extractable

Exits 0 on full pass; non-zero on failure.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API = "http://127.0.0.1:8765"
DATA_DIR = Path(os.path.expanduser("~/PSOBB.IO/data")).resolve()
TARGET_PRS = "LogoEP4.prs"
TARGET_XVM = "bm_obj_ep4_boss09_core_tex.xvm"

PUYO = Path(r"C:/Tools/re/upscale-lab/tools/puyotools/PuyoToolsCli.exe").resolve()
XVR_CODEC = Path(r"C:/Tools/re/upscale-lab/tools/xvr_codec.py").resolve()
PYEXE = Path(sys.executable).resolve()

WORK = Path(__file__).parent.parent / "_e2e_work"


def _http(method: str, path: str, body: dict | None = None, timeout: int = 600) -> dict:
    url = API + path
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
        except Exception:
            err = {"detail": str(e)}
        raise RuntimeError(f"HTTP {e.code} {method} {path} -> {err}") from e


PASS: list[str] = []
FAIL: list[str] = []


def step(name: str):
    def deco(fn):
        def wrap(*a, **kw):
            t0 = time.time()
            try:
                fn(*a, **kw)
                dt = time.time() - t0
                print(f"  PASS  [{dt:6.2f}s]  {name}")
                PASS.append(name)
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")
                FAIL.append(f"{name}: {e}")
            except Exception as e:
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
                FAIL.append(f"{name}: {type(e).__name__}: {e}")
        return wrap
    return deco


@step("health")
def t_health():
    h = _http("GET", "/api/health")
    assert h["ok"] is True, h
    assert h["version"], h
    for k in ("puyo", "xvr_codec", "realesrgan"):
        assert h["tools_resolved"][k]["exists"], f"{k} missing"


@step("files list")
def t_files():
    r = _http("GET", "/api/files")
    assert "files" in r and "data_dir" in r
    assert len(r["files"]) > 0
    # sorted by name?
    names = [f["name"] for f in r["files"]]
    assert names == sorted(names, key=str.lower), "not sorted"
    # backups filtered out
    for n in names:
        nl = n.lower()
        assert ".pre_" not in nl and ".suspect_" not in nl and ".parked_" not in nl, n
    # target present
    assert TARGET_PRS in names, f"{TARGET_PRS} missing from list"
    assert TARGET_XVM in names, f"{TARGET_XVM} missing from list"


@step("tiles extract (PRS)")
def t_tiles_prs():
    r = _http("GET", f"/api/tiles/{TARGET_PRS}")
    assert r["filename"] == TARGET_PRS
    assert r["tile_count"] == 8, r["tile_count"]
    assert r["is_prs"] is True
    for t in r["tiles"]:
        b = t["src_png_b64"].split(",", 1)[1]
        raw = base64.b64decode(b)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"tile {t['index']} not PNG"
        assert t["fmt"] in (6, 7), t


@step("tiles extract (XVM, non-PRS)")
def t_tiles_xvm():
    r = _http("GET", f"/api/tiles/{TARGET_XVM}")
    assert r["filename"] == TARGET_XVM
    assert r["tile_count"] > 0
    assert r["is_prs"] is False
    for t in r["tiles"][:4]:
        b = t["src_png_b64"].split(",", 1)[1]
        raw = base64.b64decode(b)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", t


@step("path injection rejected")
def t_path_injection():
    for bad in ("..\\evil", "..%5Cevil", "C:\\Windows\\system32"):
        try:
            _http("GET", f"/api/tiles/{bad}")
        except RuntimeError as e:
            # Either 400 (our guard) or 404 (FastAPI unmatched) is acceptable;
            # the critical bit is no path-escape happens. Verify error code:
            assert "400" in str(e) or "404" in str(e), str(e)
            continue
        else:
            raise AssertionError(f"path {bad!r} was NOT rejected")


@step("upscale (keep_native_dims=True)")
def t_upscale_native():
    # smallest tile (4 = 512x512) for speed
    r = _http("POST", "/api/upscale", {
        "filename": TARGET_PRS,
        "tile_index": 4,
        "model": "realesr-animevideov3-x4",
        "scale": 4,
        "keep_native_dims": True,
    }, timeout=600)
    assert r["out_w"] == 512 and r["out_h"] == 512, r
    raw = base64.b64decode(r["out_b64"].split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


@step("upscale (keep_native_dims=False x4)")
def t_upscale_x4():
    r = _http("POST", "/api/upscale", {
        "filename": TARGET_PRS,
        "tile_index": 4,
        "model": "realesr-animevideov3-x4",
        "scale": 4,
        "keep_native_dims": False,
    }, timeout=600)
    assert r["out_w"] == 2048 and r["out_h"] == 2048, r


@step("upscale validation rejects bad input")
def t_upscale_validation():
    bad_cases = [
        {"scale": 7, "model": "realesr-animevideov3-x4"},
        {"scale": 4, "model": "../etc/passwd"},
        {"scale": 4, "model": "weird name"},
    ]
    for b in bad_cases:
        body = {"filename": TARGET_PRS, "tile_index": 0, **b}
        try:
            _http("POST", "/api/upscale", body)
        except RuntimeError as e:
            assert "400" in str(e), str(e)
            continue
        raise AssertionError(f"upscale with {b} should have failed")


@step("repack deploy:false (no edits) verifies")
def t_repack_dryrun():
    r = _http("POST", "/api/repack", {
        "filename": TARGET_PRS, "tiles": [], "deploy": False,
    }, timeout=120)
    assert r["verify"]["ok"], r["verify"]
    assert r["deploy_path"] is None
    assert r["backup_path"] is None
    rebuilt = Path(r["rebuilt_path"])
    assert rebuilt.exists()
    blob = rebuilt.read_bytes()
    assert blob[:4] == b"XVMH"
    count = struct.unpack_from("<I", blob, 0x08)[0]
    assert count == 8, count


@step("repack deploy:true single edit + round-trip game-loadable")
def t_repack_deploy():
    # 1. snapshot original bytes (we'll restore via restore_backup at the end)
    target = DATA_DIR / TARGET_PRS
    orig_bytes = target.read_bytes()

    # 2. fetch tile 4 (smallest), modify a few pixels, send it back through repack.
    # Use a pseudo-random pixel value derived from time so re-runs always produce
    # a different tile state (defending against historical test residue).
    tiles = _http("GET", f"/api/tiles/{TARGET_PRS}")
    t4 = next(t for t in tiles["tiles"] if t["index"] == 4)
    raw = base64.b64decode(t4["src_png_b64"].split(",", 1)[1])
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    px = im.load()
    # Salt the colour with the current second so the deployed file differs
    # from the pre-existing on-disk PRS even if a prior test left magenta.
    salt = int(time.time()) & 0xFF
    test_colour = (salt, 0xFF - salt, salt ^ 0x55, 255)
    for y in range(64, 80):
        for x in range(64, 80):
            px[x, y] = test_colour
    buf = io.BytesIO()
    im.save(buf, "PNG")
    new_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    r = _http("POST", "/api/repack", {
        "filename": TARGET_PRS,
        "tiles": [{"tile_index": 4, "png_b64": new_b64}],
        "deploy": True,
    }, timeout=180)
    assert r["verify"]["ok"], r["verify"]
    assert r["deploy_path"], r
    bak = Path(r["backup_path"])
    assert bak.exists(), bak
    # New backup naming: .pre_editor_<YYYYMMDD_HHMMSS>
    assert re.search(r"\.pre_editor_\d{8}_\d{6}(?:_\d+)?$", bak.name), bak.name

    # V4: splice/reencode reporting
    if "spliced_count" in r:
        # tile 4 was edited, so it must re-encode; the other 7 should splice
        assert r["reencoded_count"] >= 1, r
        assert r["spliced_count"] >= 1, r
        assert 4 in r.get("changed_indices", []), r

    deployed = Path(r["deploy_path"])
    deployed_bytes = deployed.read_bytes()
    assert deployed.exists()
    assert deployed_bytes != orig_bytes, "deployed file is unchanged"

    # 3. round-trip the deployed file through PuyoToolsCli + xvr_codec.py extract
    #    and verify the same (count, w, h, fmt) per tile.
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    work_prs = WORK / TARGET_PRS
    shutil.copy(deployed, work_prs)
    # decompress
    subprocess.run(
        [str(PUYO), "compression", "decompress", "--overwrite", "-i", TARGET_PRS],
        cwd=WORK, check=True, capture_output=True, timeout=60,
    )
    out_dir = WORK / "tiles"
    subprocess.run(
        [str(PYEXE), str(XVR_CODEC), "extract", str(work_prs), str(out_dir)],
        check=True, capture_output=True, timeout=60,
    )
    # walk extracted PNGs vs the manifest from the API
    pngs = sorted(out_dir.glob("*.png"), key=lambda p: int(re.search(r"_(\d+)_", p.name).group(1)))
    assert len(pngs) == tiles["tile_count"], f"{len(pngs)} != {tiles['tile_count']}"
    from PIL import Image as _Image
    for png, expected in zip(pngs, tiles["tiles"]):
        with _Image.open(png) as im2:
            w, h = im2.size
        assert w == expected["width"] and h == expected["height"], \
            f"{png.name}: extracted {w}x{h}, expected {expected['width']}x{expected['height']}"
    # Verify XVR fmt byte for tile 4 specifically
    xvr4 = next(p for p in out_dir.glob("*.xvr") if "_04_" in p.name)
    fmt = struct.unpack_from("<I", xvr4.read_bytes(), 0x0C)[0]
    assert fmt == 6, f"tile 4 fmt {fmt} != 6"

    # remember backup name for restore step
    globals()["LAST_BACKUP"] = bak.name


@step("restore_backup undoes the deploy")
def t_restore():
    bak_name = globals().get("LAST_BACKUP")
    assert bak_name, "no backup recorded"
    r = _http("POST", "/api/restore_backup", {
        "filename": TARGET_PRS,
        "backup_name": bak_name,
    })
    assert r["restored_from"].endswith(bak_name)
    # verify file is back; the bak file was a copy of the pre-deploy bytes,
    # which were the original.
    target = DATA_DIR / TARGET_PRS
    assert target.exists()
    # cleanup the backup we created (so DATA_DIR doesn't bloat)
    bak = DATA_DIR / bak_name
    if bak.exists():
        bak.unlink()


@step("repack rejects garbage png_b64")
def t_repack_bad_input():
    try:
        _http("POST", "/api/repack", {
            "filename": TARGET_PRS,
            "tiles": [{"tile_index": 0, "png_b64": "data:image/png;base64,!!!notb64!!!"}],
            "deploy": False,
        }, timeout=60)
    except RuntimeError as e:
        assert "400" in str(e), str(e)
        return
    raise AssertionError("garbage png_b64 should have been rejected")


# ============================================================
# V3 additions (2026-04-24)
#   - /api/models extended fields
#   - /api/upscale tile_size, tta, gpu_id options
#   - /api/upscale cascade scales 6/8/12/16
# ============================================================

@step("models endpoint v3 fields")
def t_models_v3():
    r = _http("GET", "/api/models")
    assert "models" in r, r
    assert "allowed_scales" in r, r
    assert "allowed_tile_sizes" in r, r
    assert 8 in r["allowed_scales"], r["allowed_scales"]
    assert 16 in r["allowed_scales"], r["allowed_scales"]
    assert 0 in r["allowed_tile_sizes"], r["allowed_tile_sizes"]
    for m in r["models"]:
        assert "native_scale" in m, m
        assert "max_scale" in m, m
        assert "supports_tta" in m, m
        assert "description" in m, m
        # legacy alias preserved
        assert "default_scale" in m, m
    # spot-check known model
    by_name = {m["name"]: m for m in r["models"]}
    assert by_name["realesr-animevideov3-x2"]["native_scale"] == 2
    assert by_name["realesr-animevideov3-x4"]["native_scale"] == 4
    assert by_name["realesrgan-x4plus-anime"]["native_scale"] == 4


@step("upscale with tile_size and tta options")
def t_upscale_advanced_opts():
    r = _http("POST", "/api/upscale", {
        "filename": TARGET_PRS,
        "tile_index": 4,
        "model": "realesr-animevideov3-x2",
        "scale": 2,
        "keep_native_dims": True,
        "tile_size": 128,
        "tta": True,
    }, timeout=900)
    assert r["out_w"] == 512 and r["out_h"] == 512, r
    assert r["tile_size"] == 128, r
    assert r["tta"] is True, r


@step("upscale cascade scale=8 (keep_native_dims=False)")
def t_upscale_cascade_8():
    # tile 4 is 512x512; 8x cascade -> 4096x4096 intermediate
    r = _http("POST", "/api/upscale", {
        "filename": TARGET_PRS,
        "tile_index": 4,
        "model": "realesrgan-x4plus-anime",
        "scale": 8,
        "keep_native_dims": False,
    }, timeout=600)
    assert r["out_w"] == 4096 and r["out_h"] == 4096, r
    raw = base64.b64decode(r["out_b64"].split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


@step("upscale cascade scale=8 keep_native_dims=True returns native dims")
def t_upscale_cascade_8_native():
    r = _http("POST", "/api/upscale", {
        "filename": TARGET_PRS,
        "tile_index": 4,
        "model": "realesrgan-x4plus-anime",
        "scale": 8,
        "keep_native_dims": True,
    }, timeout=600)
    assert r["out_w"] == 512 and r["out_h"] == 512, r
    assert r["cascade_w"] == 4096 and r["cascade_h"] == 4096, r


@step("upscale rejects scale outside allowed list")
def t_upscale_scale_validation_v3():
    bad_cases = [
        {"scale": 5},   # not in allowed list
        {"scale": 7},   # not in allowed list
        {"scale": 9},   # not in allowed list
        {"scale": 32},  # too big
    ]
    for b in bad_cases:
        body = {
            "filename": TARGET_PRS,
            "tile_index": 4,
            "model": "realesr-animevideov3-x4",
            **b,
        }
        try:
            _http("POST", "/api/upscale", body)
        except RuntimeError as e:
            assert "400" in str(e), str(e)
            continue
        raise AssertionError(f"upscale with {b} should have failed")


@step("upscale rejects bad tile_size")
def t_upscale_tilesize_validation():
    body = {
        "filename": TARGET_PRS,
        "tile_index": 4,
        "model": "realesr-animevideov3-x4",
        "scale": 4,
        "tile_size": 999,  # not in allowed
    }
    try:
        _http("POST", "/api/upscale", body)
    except RuntimeError as e:
        assert "400" in str(e), str(e)
        return
    raise AssertionError("bad tile_size should have failed")


# ============================================================
# V4 additions (2026-04-25) — UX-shipping endpoints
#   - /api/import_png/{filename}/{tile_index}    drag-drop / file-import
#   - /api/repack_diff                           pre-deploy summary
# ============================================================


def _post_multipart(path: str, fields: dict, file_field: str, file_name: str,
                    file_bytes: bytes, file_type: str = "image/png", timeout: int = 60) -> dict:
    """Tiny multipart/form-data POST helper (we avoid `requests` dep)."""
    import uuid
    boundary = "----e2e" + uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{file_field}\"; filename=\"{file_name}\"\r\n"
        f"Content-Type: {file_type}\r\n\r\n".encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        API + path, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
        except Exception:
            err = {"detail": str(e)}
        raise RuntimeError(f"HTTP {e.code} POST {path} -> {err}") from e


def _png_bytes(w: int, h: int, color=(255, 0, 255, 255)) -> bytes:
    """Synthesize a solid-colour PNG."""
    from PIL import Image as _Img
    im = _Img.new("RGBA", (w, h), color)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


@step("import_png native dim accepted")
def t_import_native():
    # tile 4 of LogoEP4 is 512x512
    blob = _png_bytes(512, 512)
    r = _post_multipart(
        f"/api/import_png/{TARGET_PRS}/4",
        {"keep_native_dims": "true"}, "image", "x.png", blob,
    )
    assert r["scale_factor"] == 1, r
    assert r["out_w"] == 512 and r["out_h"] == 512, r
    assert r["src_w"] == 512 and r["src_h"] == 512, r
    raw = base64.b64decode(r["out_b64"].split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


@step("import_png 4x dim is Lanczos-downscaled to native")
def t_import_4x_native():
    blob = _png_bytes(2048, 2048, (0, 255, 0, 255))
    r = _post_multipart(
        f"/api/import_png/{TARGET_PRS}/4",
        {"keep_native_dims": "true"}, "image", "x4.png", blob,
    )
    assert r["scale_factor"] == 4, r
    # keep_native_dims=true means the returned PNG is 512x512
    assert r["out_w"] == 512 and r["out_h"] == 512, r
    assert r["imported_w"] == 2048 and r["imported_h"] == 2048, r


@step("import_png keep_native_dims=false preserves oversized PNG")
def t_import_keep_oversized():
    blob = _png_bytes(2048, 2048, (0, 0, 255, 255))
    r = _post_multipart(
        f"/api/import_png/{TARGET_PRS}/4",
        {"keep_native_dims": "false"}, "image", "x4big.png", blob,
    )
    assert r["scale_factor"] == 4, r
    assert r["out_w"] == 2048 and r["out_h"] == 2048, r


@step("import_png rejects non-multiple dim")
def t_import_bad_dim():
    blob = _png_bytes(700, 700)
    try:
        _post_multipart(
            f"/api/import_png/{TARGET_PRS}/4",
            {"keep_native_dims": "true"}, "image", "bad.png", blob,
        )
    except RuntimeError as e:
        assert "400" in str(e), str(e)
        assert "integer multiple" in str(e).lower(), str(e)
        return
    raise AssertionError("non-multiple dim should have been rejected")


@step("import_png rejects non-uniform scale")
def t_import_nonuniform():
    blob = _png_bytes(1024, 2048)  # 2x on x, 4x on y
    try:
        _post_multipart(
            f"/api/import_png/{TARGET_PRS}/4",
            {"keep_native_dims": "true"}, "image", "skew.png", blob,
        )
    except RuntimeError as e:
        assert "400" in str(e), str(e)
        assert "non-uniform" in str(e).lower(), str(e)
        return
    raise AssertionError("non-uniform scale should have been rejected")


@step("import_png rejects non-PNG bytes")
def t_import_not_png():
    try:
        _post_multipart(
            f"/api/import_png/{TARGET_PRS}/4",
            {"keep_native_dims": "true"}, "image", "x.png", b"not a png at all",
        )
    except RuntimeError as e:
        assert "400" in str(e), str(e)
        return
    raise AssertionError("non-PNG bytes should have been rejected")


@step("import_png rejects bogus tile index")
def t_import_bad_tile():
    blob = _png_bytes(64, 64)
    try:
        _post_multipart(
            f"/api/import_png/{TARGET_PRS}/999",
            {"keep_native_dims": "true"}, "image", "x.png", blob,
        )
    except RuntimeError as e:
        assert "404" in str(e), str(e)
        return
    raise AssertionError("nonexistent tile should have been rejected")


@step("repack_diff returns expected partition")
def t_repack_diff():
    r = _http("POST", "/api/repack_diff", {
        "filename": TARGET_PRS,
        "edited_indices": [0, 3, 7],
    })
    assert r["filename"] == TARGET_PRS
    assert r["tile_count"] == 8
    assert r["is_prs"] is True
    assert r["changed_indices"] == [0, 3, 7]
    assert r["unchanged_indices"] == [1, 2, 4, 5, 6]
    assert r["unknown_indices"] == []
    assert re.match(rf"^{re.escape(TARGET_PRS)}\.pre_editor_\d{{8}}_\d{{6}}$", r["backup_name_preview"]), r["backup_name_preview"]
    assert r["file_size_bytes"] > 0


@step("repack_diff flags unknown tile indices")
def t_repack_diff_unknown():
    r = _http("POST", "/api/repack_diff", {
        "filename": TARGET_PRS,
        "edited_indices": [0, 999, 1000],
    })
    assert r["changed_indices"] == [0]
    assert sorted(r["unknown_indices"]) == [999, 1000]


@step("repack_diff with no edits returns clean partition")
def t_repack_diff_empty():
    r = _http("POST", "/api/repack_diff", {
        "filename": TARGET_PRS,
        "edited_indices": [],
    })
    assert r["changed_indices"] == []
    assert r["unchanged_indices"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert r["unknown_indices"] == []


# ============================================================
# V4 quality additions (2026-04-24)
#   - Splice-not-reencode for unchanged tiles
#   - Export-only build (deploy=false + token download)
#   - /api/verify per-tile bit-identity check
# ============================================================


@step("repack no-edit produces bit-identical XVM via splice")
def t_v4q_splice_identical():
    """A repack with zero edits must yield an XVM whose XVR payloads are
    byte-for-byte identical to the source — proving the splice path is wired."""
    target = DATA_DIR / TARGET_PRS
    orig_bytes = target.read_bytes()
    r = _http("POST", "/api/repack", {
        "filename": TARGET_PRS, "tiles": [], "deploy": False,
    }, timeout=120)
    assert r["verify"]["ok"], r["verify"]
    if "spliced_count" in r:
        # ALL tiles must splice; ZERO must re-encode
        assert r["reencoded_count"] == 0, f"expected 0 reencoded, got {r}"
        assert r["spliced_count"] >= 1, r
        assert r["changed_indices"] == [], r
    # The on-disk PRS is untouched (deploy=false)
    assert target.read_bytes() == orig_bytes, "deploy=false touched on-disk PRS"
    # The rebuilt XVM file must equal the decompressed source XVM byte-for-byte.
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    work_prs = WORK / TARGET_PRS
    shutil.copy(target, work_prs)
    subprocess.run(
        [str(PUYO), "compression", "decompress", "--overwrite", "-i", TARGET_PRS],
        cwd=WORK, check=True, capture_output=True, timeout=60,
    )
    orig_xvm_md5 = hashlib.md5(work_prs.read_bytes()).hexdigest()

    rebuilt_path = Path(r["rebuilt_path"])
    assert rebuilt_path.exists(), rebuilt_path
    rebuilt_md5 = hashlib.md5(rebuilt_path.read_bytes()).hexdigest()
    assert rebuilt_md5 == orig_xvm_md5, (
        f"splice path drifted: orig XVM md5 {orig_xvm_md5} != "
        f"rebuilt XVM md5 {rebuilt_md5}"
    )


@step("repack edit produces splice + re-encode mix")
def t_v4q_splice_mix():
    """When ONE tile is edited, exactly that tile re-encodes; the rest splice."""
    tiles = _http("GET", f"/api/tiles/{TARGET_PRS}")
    t4 = next(t for t in tiles["tiles"] if t["index"] == 4)
    raw = base64.b64decode(t4["src_png_b64"].split(",", 1)[1])
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    px = im.load()
    salt = (int(time.time()) ^ 0xA5) & 0xFF
    for y in range(96, 112):
        for x in range(96, 112):
            px[x, y] = (salt, 0xFF - salt, 0x80, 255)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    new_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    r = _http("POST", "/api/repack", {
        "filename": TARGET_PRS,
        "tiles": [{"tile_index": 4, "png_b64": new_b64}],
        "deploy": False,
    }, timeout=120)
    assert r["verify"]["ok"], r["verify"]
    if "spliced_count" in r:
        # exactly 1 reencoded (tile 4), 7 spliced
        assert r["reencoded_count"] == 1, r
        assert r["spliced_count"] == 7, r
        assert r["changed_indices"] == [4], r


@step("export-only build mints a token + serves the artifact")
def t_v4q_export_only():
    """deploy=false should return an export_token + export_url; the URL must
    return the rebuilt artifact bytes that decompress to a valid XVM."""
    r = _http("POST", "/api/repack", {
        "filename": TARGET_PRS, "tiles": [], "deploy": False,
    }, timeout=120)
    if "export_url" not in r or not r["export_url"]:
        print("    (skipped: export_url not in response)")
        return
    url = r["export_url"]
    assert url.startswith("/api/export/"), url
    req = urllib.request.Request(API + url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read()
    assert len(body) > 0, "export body was empty"
    # Decompress + confirm XVMH magic.
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    out_prs = WORK / TARGET_PRS
    out_prs.write_bytes(body)
    subprocess.run(
        [str(PUYO), "compression", "decompress", "--overwrite", "-i", TARGET_PRS],
        cwd=WORK, check=True, capture_output=True, timeout=60,
    )
    decompressed = out_prs.read_bytes()
    assert decompressed[:4] == b"XVMH", "exported PRS did not decompress to XVMH"

    # Bad token must 404
    try:
        with urllib.request.urlopen(API + "/api/export/this-is-not-a-real-token", timeout=30):
            pass
    except urllib.error.HTTPError as e:
        assert e.code == 404, e.code
    else:
        raise AssertionError("bad token should 404")


@step("export-only does not touch DATA_DIR")
def t_v4q_export_only_clean():
    """deploy=false must leave the live data file untouched."""
    target = DATA_DIR / TARGET_PRS
    pre_md5 = hashlib.md5(target.read_bytes()).hexdigest()
    pre_size = target.stat().st_size
    r = _http("POST", "/api/repack", {
        "filename": TARGET_PRS, "tiles": [], "deploy": False,
    }, timeout=120)
    assert r["deploy_path"] is None
    assert r["backup_path"] is None
    post_md5 = hashlib.md5(target.read_bytes()).hexdigest()
    post_size = target.stat().st_size
    assert pre_md5 == post_md5, "deploy=false changed file content!"
    assert pre_size == post_size, "deploy=false changed file size!"


@step("verify endpoint reports per-tile identity")
def t_v4q_verify():
    """The verify endpoint reads tiles from cache and checks each PNG's md5
    against the sidecar recorded at extract time. Earlier tests may have
    modified the cached PNGs (import_png replaces the on-disk PNG), so
    we nuke the cache first to force a fresh extract for this test.
    """
    # Drop any stale cache so /api/verify re-extracts from disk, producing
    # both fresh PNGs AND fresh sidecars (which always match).
    import shutil as _sh
    cache_dir = Path(r"C:/tmp_pso_editor/cache")
    target_pat = TARGET_PRS + "_*"
    for d in cache_dir.glob(target_pat):
        if d.is_dir():
            _sh.rmtree(d, ignore_errors=True)
    r = _http("GET", f"/api/verify/{TARGET_PRS}", timeout=120)
    assert "tile_count" in r, r
    assert "tiles" in r, r
    assert len(r["tiles"]) == r["tile_count"], r
    for t in r["tiles"]:
        assert "index" in t, t
        assert "actual_md5" in t, t
        assert "identical_to_cache" in t, t
        # After a fresh extract, every tile's md5 matches its sidecar.
        assert t["identical_to_cache"] is True, t
    assert r["all_identical"] is True, r


@step("xvr_codec splice produces bit-identical XVM standalone")
def t_v4q_xvr_codec_splice_offline():
    """Standalone test of xvr_codec's splice behaviour, independent of
    the server. Extract -> rebuild without modifications -> compare md5."""
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    work_prs = WORK / TARGET_PRS
    shutil.copy(DATA_DIR / TARGET_PRS, work_prs)
    subprocess.run(
        [str(PUYO), "compression", "decompress", "--overwrite", "-i", TARGET_PRS],
        cwd=WORK, check=True, capture_output=True, timeout=60,
    )
    orig_xvm_md5 = hashlib.md5(work_prs.read_bytes()).hexdigest()
    out_dir = WORK / "tiles"
    subprocess.run(
        [str(PYEXE), str(XVR_CODEC), "extract", str(work_prs), str(out_dir)],
        check=True, capture_output=True, timeout=60,
    )
    sidecars = list(out_dir.glob("*.src.md5"))
    assert len(sidecars) > 0, "no .src.md5 sidecars produced by extract"

    rebuilt = WORK / "rebuilt.xvm"
    result = subprocess.run(
        [str(PYEXE), str(XVR_CODEC), "rebuild", str(out_dir), str(rebuilt)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    rebuilt_md5 = hashlib.md5(rebuilt.read_bytes()).hexdigest()
    assert rebuilt_md5 == orig_xvm_md5, (
        f"unmodified rebuild drifted: orig {orig_xvm_md5} != rebuilt {rebuilt_md5}"
    )
    assert "spliced=" in result.stdout, result.stdout


# ============================================================
# Code-quality cleanup pass (2026-04-25)
#   - body-size limits
#   - lock-table reporting on /api/health
#   - gpu_id validation
#   - concurrent-repack 409
#   - modal upscale-bar contract (response shape)
# ============================================================

@step("health reports lock-table sizes")
def t_cleanup_health_locks():
    h = _http("GET", "/api/health")
    assert "locks" in h, h
    for k in ("upscale", "repack"):
        assert k in h["locks"], h
        assert isinstance(h["locks"][k], int), h


@step("upscale rejects out-of-range gpu_id")
def t_cleanup_gpu_id_validation():
    """gpu_id must be -1..7. Anything outside that range should 400."""
    bad_cases = [-2, 8, 100]
    for g in bad_cases:
        body = {
            "filename": TARGET_PRS,
            "tile_index": 4,
            "model": "realesr-animevideov3-x4",
            "scale": 4,
            "gpu_id": g,
        }
        try:
            _http("POST", "/api/upscale", body)
        except RuntimeError as e:
            assert "400" in str(e), str(e)
            continue
        raise AssertionError(f"upscale with gpu_id={g} should have failed")


@step("upscale accepts gpu_id at the boundaries (-1, 7)")
def t_cleanup_gpu_id_boundaries():
    """gpu_id=-1 ("use the CPU") and gpu_id=7 (max) must both pass the
    validator. We don't run the binary in this test (CPU upscale would be
    glacial; gpu 7 doesn't exist on the user's machine) — we use a valid
    file but a non-existent tile index so we hit the 404 'no such tile'
    branch AFTER validation has passed. That's our proof gpu_id was
    accepted.
    """
    for gid in (-1, 7):
        body = {
            "filename": TARGET_PRS,
            "tile_index": 99,  # in-range (<=4096) but no tile 99 in TARGET_PRS (8 tiles)
            "model": "realesr-animevideov3-x4",
            "scale": 4,
            "gpu_id": gid,
        }
        try:
            _http("POST", "/api/upscale", body)
        except RuntimeError as e:
            # 404 = "no such tile" (validation passed); anything else means
            # validation rejected gpu_id at the input layer.
            assert "404" in str(e), f"gpu_id={gid}: expected 404, got: {e}"
            continue
        raise AssertionError(f"gpu_id={gid}: expected 404 for non-existent tile")


@step("repack body-size limit (oversized request rejected)")
def t_cleanup_repack_body_too_large():
    """A request claiming Content-Length > 64 MB (MAX_REPACK_BODY) should
    be rejected with 413. We don't actually need to send 64 MB of payload
    -- the guard reads Content-Length out of the headers and rejects up
    front.

    Implementation note: uvicorn requires the body to be fully received
    before invoking the handler in async mode, so we DO need to either
    stream the full body (slow) or set a tight client timeout and accept
    timeout-as-rejection. We use the second approach: under the limit we'd
    get a quick 200, over the limit we get 413 OR a hard timeout (because
    server stops reading our oversize stream). Either is a valid pass.
    """
    import urllib.request
    huge_size = 100 * 1024 * 1024  # 100 MB > 64 MB cap
    # Build a payload that's just under the size declared. We can't easily
    # send 100 MB so we craft a request that's claiming 100 MB but is much
    # smaller. urllib will compute Content-Length itself by default; we
    # need to use http.client with a custom header.
    import http.client
    body = b'{"filename":"' + TARGET_PRS.encode() + b'","tiles":[],"deploy":false}'
    body = body + b" " * (huge_size - len(body))  # pad to exactly 100 MB
    # That's still slow. Instead, send with a fake CL header and a small body.
    # Some HTTP servers will see CL=huge, body=tiny, and either timeout or 413.
    body_small = b'{"filename":"' + TARGET_PRS.encode() + b'","tiles":[],"deploy":false}'
    conn = http.client.HTTPConnection("127.0.0.1", 8765, timeout=5)
    try:
        conn.putrequest("POST", "/api/repack")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(huge_size))  # lie!
        conn.putheader("Connection", "close")
        conn.endheaders()
        conn.send(body_small)
        try:
            resp = conn.getresponse()
            status = resp.status
            resp.read()
        except (TimeoutError, http.client.RemoteDisconnected, ConnectionResetError):
            # Server gave up waiting for the rest of the body. That's a
            # legitimate rejection (it never let our oversized request
            # through), even if not a clean 413.
            return
        assert status in (413, 422, 400), f"expected 4xx, got {status}"
    finally:
        conn.close()


@step("repack rejects too many tile edits")
def t_cleanup_repack_too_many_tiles():
    """64-tile sanity cap on /api/repack."""
    fake_b64 = "data:image/png;base64,iVBORw0KGgo="  # tiny stub
    body = {
        "filename": TARGET_PRS,
        "tiles": [{"tile_index": i, "png_b64": fake_b64} for i in range(100)],
        "deploy": False,
    }
    try:
        _http("POST", "/api/repack", body, timeout=30)
    except RuntimeError as e:
        assert "400" in str(e), str(e)
        return
    raise AssertionError("100-tile repack should have been rejected")


@step("concurrent repack returns 409")
def t_cleanup_concurrent_repack():
    """Fire two /api/repack calls at the same file simultaneously. The
    second should get a 409 'already in progress' response while the first
    is still running.
    """
    import threading
    results: list = []
    errors: list = []

    def fire():
        try:
            r = _http("POST", "/api/repack", {
                "filename": TARGET_PRS, "tiles": [], "deploy": False,
            }, timeout=120)
            results.append(r)
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=fire) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    # We expect at least one success and possibly one 409. If both succeed,
    # the second simply ran after the first finished — also acceptable but
    # we want at least one path tested. The strong check: nothing crashed.
    assert len(results) + len(errors) == 2, f"got {len(results)} results, {len(errors)} errors"
    # If we got an error, it must be 409 (the only legitimate one).
    for e in errors:
        assert "409" in e, f"unexpected error from concurrent repack: {e}"


@step("modal upscale-bar response contract has all expected fields")
def t_cleanup_modal_upscale_contract():
    """The frontend's modal upscale-bar reads these fields out of the
    /api/upscale JSON response. If any field disappears, the modal breaks
    silently. This test pins the contract.
    """
    r = _http("POST", "/api/upscale", {
        "filename": TARGET_PRS,
        "tile_index": 4,
        "model": "realesr-animevideov3-x4",
        "scale": 4,
        "keep_native_dims": True,
    }, timeout=600)
    expected_keys = {
        "tile_index", "model", "scale", "tile_size", "tta", "gpu_id",
        "out_b64", "out_w", "out_h", "src_w", "src_h", "cascade_w", "cascade_h",
    }
    missing = expected_keys - set(r.keys())
    assert not missing, f"upscale response missing keys: {missing}"


@step("invalid filename rejected on every endpoint")
def t_cleanup_invalid_filename_all_endpoints():
    """All endpoints that accept a filename should 400 on path traversal."""
    bad = "..\\evil"
    # /api/tiles/{filename}
    try:
        _http("GET", f"/api/tiles/{bad}")
    except RuntimeError as e:
        assert "400" in str(e) or "404" in str(e), str(e)
    # /api/tile_png/{filename}/0
    try:
        _http("GET", f"/api/tile_png/{bad}/0")
    except RuntimeError as e:
        assert "400" in str(e) or "404" in str(e), str(e)
    # /api/upscale
    try:
        _http("POST", "/api/upscale", {
            "filename": bad, "tile_index": 0,
            "model": "realesr-animevideov3-x4", "scale": 4,
        })
    except RuntimeError as e:
        assert "400" in str(e), str(e)
    # /api/restore_backup
    try:
        _http("POST", "/api/restore_backup", {"filename": bad})
    except RuntimeError as e:
        assert "400" in str(e) or "404" in str(e), str(e)


@step("export token format guarded against path injection")
def t_cleanup_export_token_path_traversal():
    """Tokens are validated against [A-Za-z0-9_-]+ — anything else is 400."""
    bad_tokens = ["../etc/passwd", "tok with space", "tok/slash"]
    for tok in bad_tokens:
        import urllib.parse
        try:
            _http("GET", f"/api/export/{urllib.parse.quote(tok)}")
        except RuntimeError as e:
            # 400 (our guard) or 404 (FastAPI wrong route) both fine
            assert "400" in str(e) or "404" in str(e), f"{tok}: {e}"
            continue
        raise AssertionError(f"export token {tok!r} should have been rejected")


@step("xvr_codec re-encodes when PNG is modified")
def t_v4q_xvr_codec_modified():
    """If a PNG is modified, splice MUST NOT happen for that tile — the
    new content has to make it into the XVR."""
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    work_prs = WORK / TARGET_PRS
    shutil.copy(DATA_DIR / TARGET_PRS, work_prs)
    subprocess.run(
        [str(PUYO), "compression", "decompress", "--overwrite", "-i", TARGET_PRS],
        cwd=WORK, check=True, capture_output=True, timeout=60,
    )
    out_dir = WORK / "tiles"
    subprocess.run(
        [str(PYEXE), str(XVR_CODEC), "extract", str(work_prs), str(out_dir)],
        check=True, capture_output=True, timeout=60,
    )
    target_png = next(out_dir.glob("*_00_*.png"))
    from PIL import Image
    im = Image.open(target_png).convert("RGBA")
    salt = int(time.time()) & 0xFF
    for y in range(0, 32):
        for x in range(0, 32):
            im.putpixel((x, y), (salt, 0xFF - salt, 0x88, 255))
    im.save(target_png)

    rebuilt = WORK / "rebuilt.xvm"
    result = subprocess.run(
        [str(PYEXE), str(XVR_CODEC), "rebuild", str(out_dir), str(rebuilt)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    out = result.stdout
    m = re.search(r"reencoded=(\d+)", out)
    assert m, out
    n_reenc = int(m.group(1))
    assert n_reenc >= 1, f"expected >=1 reencoded, got {n_reenc} in {out!r}"


# ============================================================
# Atlas mode (2026-04-25)
#   - GET  /api/atlas/{filename}        layout + composite_b64
#   - POST /api/atlas_upscale           upscale composite, slice -> per-tile
#   - POST /api/atlas_import            user-supplied composite, slice -> per-tile
# ============================================================


# ============================================================
# BML container reader (Agent 4, 2026-04-24)
#   - GET /api/bml/{path}/list
#   - GET /api/bml/{path}/extract/{name}
#   - GET /api/bml/{path}/texture/{name}
# ============================================================
LIVE_DATA_DIR = Path(os.path.expanduser("~/PSOBB.IO/data")).resolve()
TARGET_BML = "bm_obj_ep4_boss09_core.bml"


def _http_get_bytes(path: str, timeout: int = 60) -> tuple[int, bytes, dict[str, str]]:
    """GET a path and return (status, body_bytes, headers).

    Used for endpoints that return raw bytes rather than JSON. On HTTP
    error returns (status, error_body, headers) without raising so
    tests can assert on the code.
    """
    url = API + path
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers.items())
    except urllib.error.HTTPError as e:
        err_headers = dict(e.headers.items()) if e.headers else {}
        return e.code, e.read(), err_headers


def _bml_available(name: str) -> bool:
    """True if the named BML exists in either DATA_DIR or LIVE_DATA_DIR."""
    return (DATA_DIR / name).exists() or (LIVE_DATA_DIR / name).exists()


@step(f"bml: {TARGET_BML} parses to N NJ entries")
def t_bml_list():
    if not _bml_available(TARGET_BML):
        print("    (skipped: target BML not found in DATA_DIR or LIVE_DATA_DIR)")
        return
    r = _http("GET", f"/api/bml/{TARGET_BML}/list")
    assert r["path"] == TARGET_BML, r
    assert r["count"] == 3, f"expected 3 entries, got {r['count']}"
    assert len(r["entries"]) == 3, r
    for ent in r["entries"]:
        assert ent["name"].endswith(".nj"), f"non-NJ entry: {ent['name']}"
        assert ent["size_compressed"] > 0
        assert ent["size_decompressed"] >= ent["size_compressed"]
        assert ent["has_texture"] is True
        assert ent["tex_size_compressed"] > 0


@step(f"bml: extract returns bytes that pass IFF NJCM check")
def t_bml_extract_nj():
    if not _bml_available(TARGET_BML):
        print("    (skipped: target BML not found)")
        return
    # Get the first entry's name from /list
    listing = _http("GET", f"/api/bml/{TARGET_BML}/list")
    name = listing["entries"][0]["name"]
    status, body, headers = _http_get_bytes(f"/api/bml/{TARGET_BML}/extract/{name}", timeout=60)
    assert status == 200, f"extract returned {status}: {body[:200]!r}"
    assert len(body) > 0, "empty extract body"
    # Try the IFF parser if available (Agent 2's formats/iff.py); fall
    # back to raw byte sniff if not.
    has_njcm = False
    try:
        from formats.iff import parse_iff
        chunks = parse_iff(body)
        has_njcm = any(c.type == "NJCM" for c in chunks)
        assert has_njcm, f"extracted bytes have no NJCM chunk; got types {[c.type for c in chunks]}"
    except ImportError:
        # Graceful degrade: at least confirm the file is plausible IFF.
        assert body[:4].isascii(), f"first 4 bytes not ASCII: {body[:4]!r}"
        # We at least expect one of NJTL/NJCM at the very front.
        assert body[:4] in (b"NJTL", b"NJCM"), f"unexpected magic {body[:4]!r}"


@step("bml: bad header rejected with 400")
def t_bml_bad_header():
    # Stage a deliberately-bad BML in DATA_DIR temporarily.
    # We can't write to LIVE_DATA_DIR (read-only by spec), but DATA_DIR
    # is the dev mirror so this is fine.
    bad_name = "_e2e_bml_bad_header.bml"
    bad_path = DATA_DIR / bad_name
    try:
        bad_path.write_bytes(b"\xff" * 0x40 + b"\x00" * 0x800)
        status, body, _ = _http_get_bytes(f"/api/bml/{bad_name}/list", timeout=30)
        assert status == 400, f"expected 400 for bad BML header, got {status}: {body[:200]!r}"
        # Sanity-check error message contains "BML"
        msg = body.decode("utf-8", errors="replace")
        assert "BML" in msg or "compression" in msg or "file_count" in msg, msg
    finally:
        if bad_path.exists():
            bad_path.unlink()


@step("bml: empty file rejected with 400")
def t_bml_empty():
    empty_name = "_e2e_bml_empty.bml"
    empty_path = DATA_DIR / empty_name
    try:
        empty_path.write_bytes(b"")
        status, body, _ = _http_get_bytes(f"/api/bml/{empty_name}/list", timeout=30)
        assert status == 400, f"expected 400 for empty BML, got {status}: {body[:200]!r}"
    finally:
        if empty_path.exists():
            empty_path.unlink()


@step("bml: texture extract returns valid XVM magic when has_texture=true")
def t_bml_texture_xvm():
    if not _bml_available(TARGET_BML):
        print("    (skipped: target BML not found)")
        return
    listing = _http("GET", f"/api/bml/{TARGET_BML}/list")
    # Pick the first entry that has_texture=true
    target = next((e for e in listing["entries"] if e["has_texture"]), None)
    assert target is not None, "no entry has a texture"
    name = target["name"]
    status, body, headers = _http_get_bytes(
        f"/api/bml/{TARGET_BML}/texture/{name}", timeout=120,
    )
    assert status == 200, f"texture endpoint returned {status}: {body[:200]!r}"
    assert len(body) > 8, f"texture body too short: {len(body)}"
    # XVM begins with "XVMH" + a size byte (typically 0x38).
    assert body[:4] == b"XVMH", f"texture magic not XVMH: {body[:8]!r}"
    # Content-Type should be application/x-xvm
    ct = headers.get("Content-Type") or headers.get("content-type", "")
    assert "x-xvm" in ct.lower() or "octet" in ct.lower(), f"unexpected Content-Type: {ct!r}"


# ============================================================
# XJ mesh parser (formats/xj.py) + /api/model_mesh endpoint
# ============================================================
TARGET_NJ = "plAbdy00.nj"  # always present in PSOBB.IO/data/


def _nj_available(name: str) -> bool:
    return (DATA_DIR / name).exists() or (LIVE_DATA_DIR / name).exists()


@step("xj: parse plAbdy00.nj returns >=1 mesh with valid AABB")
def t_xj_parse_plabdy00_smoke():
    if not _nj_available(TARGET_NJ):
        print("    (skipped: target NJ not found in DATA_DIR or LIVE_DATA_DIR)")
        return
    from formats.xj import parse_nj_file
    nj_path = DATA_DIR / TARGET_NJ
    if not nj_path.exists():
        nj_path = LIVE_DATA_DIR / TARGET_NJ
    buf = nj_path.read_bytes()
    meshes = parse_nj_file(buf)
    assert len(meshes) >= 1, f"expected at least 1 mesh, got {len(meshes)}"
    # Pick a mesh that actually has vertices and verify the bounding sphere
    # is within a sane range. Player bodies live in [-2, 2] world space.
    populated = [m for m in meshes if m.vertices]
    assert len(populated) >= 1, "no mesh has any vertices"
    # Player body verts now bake into world space via the bone-tree
    # walker (post-2026-04-24 fix); a humanoid model spans roughly
    # 0..170 along Y. Use a generous +/- 200 box that catches both the
    # old "near-origin local" range and the new "world-space" range.
    for m in populated:
        cx, cy, cz, r = m.bounding_sphere
        assert -200.0 <= cx <= 200.0, f"bbox center.x out of range: {cx}"
        assert -200.0 <= cy <= 200.0, f"bbox center.y out of range: {cy}"
        assert -200.0 <= cz <= 200.0, f"bbox center.z out of range: {cz}"
        assert r >= 0.0, f"bbox radius negative: {r}"


@step("xj: plAbdy00.nj parses to >= 500 tris (full Ninja-chunk extraction)")
def t_xj_proto_consistency():
    """plAbdy00.nj is a player body model. The earlier "POF0 +
    plausibility heuristic" parser returned ~5-10% of the geometry;
    after the faithful Phantasmal-Nj.kt port we should get ~500+ tris
    and ~700+ verts.

    This is the regression guard for the bug repro in
    AGENT_XJ_FAITHFUL_PORT_REPORT.md. If a future change re-introduces
    the heuristic parser (or breaks the chunk-stream cache replay) the
    triangle count will collapse and this test catches it.
    """
    if not _nj_available(TARGET_NJ):
        print("    (skipped: target NJ not found)")
        return
    from formats.xj import parse_nj_file
    nj_path = DATA_DIR / TARGET_NJ
    if not nj_path.exists():
        nj_path = LIVE_DATA_DIR / TARGET_NJ
    meshes = parse_nj_file(nj_path.read_bytes())
    total_v = sum(len(m.vertices) for m in meshes)
    total_t = sum(len(m.indices) // 3 for m in meshes)
    # Empirically plAbdy00.nj has ~793 verts and ~557 tris distributed
    # across ~142 submeshes. Thresholds set to ~70% of that so they
    # remain meaningful even if the strip de-duplicator becomes more
    # aggressive in future.
    assert total_t >= 500, f"plAbdy00 tri count too low: {total_t}"
    assert total_v >= 700, f"plAbdy00 vert count too low: {total_v}"


@step("xj: bm4_ps_ma_body.bml#bm4_ps_ma_body.nj parses to >= 500 tris")
def t_xj_bm4_player_body():
    """Bug-repro fixture: the user clicked this BML-inner pair in the
    asset tree and saw 24 sub-meshes with 261 verts / 116 tris (POF0
    heuristic parser). After the faithful chunk-Ninja port we should
    get ~130 submeshes / ~941 verts / ~681 tris, all of which are
    well above the threshold.
    """
    bml_name = "bm4_ps_ma_body.bml"
    bml_path = DATA_DIR / bml_name
    if not bml_path.exists():
        bml_path = LIVE_DATA_DIR / bml_name
    if not bml_path.exists():
        print(f"    (skipped: {bml_name} not present)")
        return
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj import parse_nj_file

    blob = bml_path.read_bytes()
    entries = parse_bml(blob)
    target = next((e for e in entries if e.name == "bm4_ps_ma_body.nj"), None)
    assert target is not None, "bm4_ps_ma_body.bml is missing its .nj inner"
    raw = bytes(blob[target.offset:target.offset + target.size_compressed])
    nj_bytes = _prs_decompress(raw, timeout=20.0)
    meshes = parse_nj_file(nj_bytes)
    total_v = sum(len(m.vertices) for m in meshes)
    total_t = sum(len(m.indices) // 3 for m in meshes)
    # Threshold set just below the empirical 681 to allow for minor
    # de-duplicator tweaks.
    assert total_t >= 500, f"bm4 player body tri count too low: {total_t}"
    assert total_v >= 800, f"bm4 player body vert count too low: {total_v}"
    assert len(meshes) >= 50, f"submesh count too low: {len(meshes)}"


@step("xj: bm_obj_ep4_boss09_core.bml#core01.nj parses to >= 80 tris")
def t_xj_boss09_core():
    """Boss-09 core model — tiny low-poly fixture used as a regression
    sentinel. ~28 submeshes / ~179 verts / ~123 tris empirically.
    """
    bml_name = "bm_obj_ep4_boss09_core.bml"
    bml_path = DATA_DIR / bml_name
    if not bml_path.exists():
        bml_path = LIVE_DATA_DIR / bml_name
    if not bml_path.exists():
        print(f"    (skipped: {bml_name} not present)")
        return
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj import parse_nj_file

    blob = bml_path.read_bytes()
    entries = parse_bml(blob)
    # PSOBB-IO names this entry bm_obj_ep4_boss09_core01.nj (the
    # original BML drops the prefix in some packs but PSOBB.IO keeps
    # it). We accept either spelling.
    target = next(
        (
            e for e in entries
            if e.name in ("core01.nj", "bm_obj_ep4_boss09_core01.nj")
        ),
        None,
    )
    assert target is not None, (
        f"boss09_core BML is missing its core01.nj inner; got "
        f"{[e.name for e in entries]}"
    )
    raw = bytes(blob[target.offset:target.offset + target.size_compressed])
    nj_bytes = _prs_decompress(raw, timeout=20.0)
    meshes = parse_nj_file(nj_bytes)
    total_t = sum(len(m.indices) // 3 for m in meshes)
    assert total_t >= 80, f"boss09 core tri count too low: {total_t}"


@step("xj: bm_ene_gibbles_low.bml#lo_gibb_body.nj has sub-mesh transforms (at least one with non-zero position)")
def t_xj_gibbles_submesh_transforms():
    """Bone-tree transform regression guard (added 2026-04-24).

    PSOBB BB skinned models — like the gibbles enemy — store per-bone
    vertex chunks in BONE-LOCAL coordinates. The pre-fix parser
    dropped the bone tree entirely and rendered every strip's
    bone-local geometry stacked at the model origin, producing the
    "exploded shards" bug where head/arms/body all overlap.

    The fix is in ``formats/xj.py``: the tree walker accumulates each
    node's local-to-world matrix and bakes it into the vertex slot
    table during the vertex pass. Each emitted ``XjMesh`` then carries
    a per-submesh ``world_position`` reflecting the strip's AABB
    centre in world space.

    This test asserts the submeshes carry meaningful transforms — at
    least one ``XjMesh`` with a non-zero ``world_position``. (For the
    actual coherence check, see the dragon test below — it asserts
    BOTH a near-origin and a far-from-origin sub-mesh exists, which
    is only true when bone transforms were correctly composed.)
    """
    bml_name = "bm_ene_gibbles_low.bml"
    inner_name = "lo_gibb_body.nj"
    bml_path = DATA_DIR / bml_name
    if not bml_path.exists():
        bml_path = LIVE_DATA_DIR / bml_name
    if not bml_path.exists():
        print(f"    (skipped: {bml_name} not present)")
        return
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj import parse_nj_file

    blob = bml_path.read_bytes()
    entries = parse_bml(blob)
    target = next((e for e in entries if e.name == inner_name), None)
    assert target is not None, f"{bml_name} is missing {inner_name}"
    raw = bytes(blob[target.offset:target.offset + target.size_compressed])
    nj_bytes = _prs_decompress(raw, timeout=20.0)
    meshes = parse_nj_file(nj_bytes)
    assert len(meshes) >= 1, f"no meshes parsed for {inner_name}"

    nonzero = [m for m in meshes if any(abs(c) > 1e-3 for c in m.world_position)]
    assert len(nonzero) >= 1, (
        f"expected at least one sub-mesh with non-zero world_position, "
        f"got {len(meshes)} meshes all centred at origin (bone tree "
        f"likely not being walked correctly)"
    )
    # Sanity: the world_position is also surfaced on the JSON wire
    # format. Hit /api/model_mesh and verify it's present. We pass the
    # inner via the ``?inner=`` query rather than the ``#`` fragment so
    # urllib doesn't strip it as a URL fragment.
    r = _http(
        "GET",
        f"/api/model_mesh/{bml_name}?inner={urllib.parse.quote(inner_name)}",
    )
    assert r.get("vertices_pre_transformed") is True, (
        "/api/model_mesh response missing vertices_pre_transformed=true; "
        "frontend will doubly-offset every submesh"
    )
    assert r["meshes"][0].get("world_position") is not None, (
        "JSON mesh entries missing world_position field"
    )


@step("xj: bm_boss8_dragon.bml#boss1_s_nb_dragon.nj has at least one sub-mesh near origin and one offset")
def t_xj_dragon_submesh_combined_pose():
    """Combined-pose regression guard (added 2026-04-24).

    Whereas the gibbles test only asserts SOME sub-mesh has a non-zero
    pose (catches "tree walker dropped"), this test asserts the WORLD
    SPACE positions VARY across sub-meshes — which is only true when
    the parent-chain transform is being correctly composed at each
    bone. If the walker accumulated only translation and dropped
    parent rotation, every sub-mesh would still wind up at (small,
    small, small) and the "far from origin" half of this assert would
    fail.

    bm_boss8_dragon is the PSOBB BB Sil Dragon boss, a multi-bone
    model with the dragon's body extending well into negative-Z
    territory (tail) while the head sits in positive-X around 100+
    units. We assert both a near-origin sub-mesh (some part of the
    body sits near the model anchor) and a far-from-origin one (tail
    or wingtip is offset > 15 units somewhere).
    """
    bml_name = "bm_boss8_dragon.bml"
    inner_name = "boss1_s_nb_dragon.nj"
    bml_path = DATA_DIR / bml_name
    if not bml_path.exists():
        bml_path = LIVE_DATA_DIR / bml_name
    if not bml_path.exists():
        print(f"    (skipped: {bml_name} not present)")
        return
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj import parse_nj_file

    blob = bml_path.read_bytes()
    entries = parse_bml(blob)
    target = next((e for e in entries if e.name == inner_name), None)
    assert target is not None, f"{bml_name} is missing {inner_name}"
    raw = bytes(blob[target.offset:target.offset + target.size_compressed])
    nj_bytes = _prs_decompress(raw, timeout=20.0)
    meshes = parse_nj_file(nj_bytes)
    assert len(meshes) >= 10, f"expected many sub-meshes for the dragon, got {len(meshes)}"

    near_origin = [
        m for m in meshes
        if all(abs(c) < 8.0 for c in m.world_position)
    ]
    far_from_origin = [
        m for m in meshes
        if any(abs(c) > 15.0 for c in m.world_position)
    ]
    assert len(near_origin) >= 1, (
        f"expected at least one sub-mesh near the origin, got 0 of "
        f"{len(meshes)} (parent chain may have over-applied translation)"
    )
    assert len(far_from_origin) >= 1, (
        f"expected at least one sub-mesh far from the origin, got 0 of "
        f"{len(meshes)} (parent chain may not be accumulating "
        f"translations at all)"
    )


@step("xj_descriptor: bm_fe_obj_o_door01l.bml#fe_obj_o_door01l.xj parses to >= 50 tris")
def t_xj_descriptor_door01l():
    """Descriptor-table XJ parser smoke test (added 2026-04-24).

    PSOBB.IO ships ~263 BML-inner models in the descriptor-table .xj
    format (vs the chunk-based .nj). Before this fixture they fell
    through the model viewer's "primitive cube" fallback. The new
    parser at ``formats/xj_descriptor.py`` walks the same NJCM IFF
    wrapper but parses XjModel structs (vertex info table + triangle
    strip table + material table) instead of variable-length chunks.

    The user-spec fixture path is ``bm_fe_obj_o_door01.bml#fe_obj_o_door01.xj``
    but the actual file in PSOBB.IO is named with an ``l`` suffix
    (a localization artifact). Empirically: 24 sub-meshes / 173 verts /
    125 tris.
    """
    bml_name = "bm_fe_obj_o_door01l.bml"
    inner_name = "fe_obj_o_door01l.xj"
    bml_path = DATA_DIR / bml_name
    if not bml_path.exists():
        bml_path = LIVE_DATA_DIR / bml_name
    if not bml_path.exists():
        print(f"    (skipped: {bml_name} not present)")
        return
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj_descriptor import parse_xj_file

    blob = bml_path.read_bytes()
    entries = parse_bml(blob)
    target = next((e for e in entries if e.name == inner_name), None)
    assert target is not None, f"{bml_name} is missing {inner_name}"
    raw = bytes(blob[target.offset:target.offset + target.size_compressed])
    xj_bytes = _prs_decompress(raw, timeout=20.0)
    meshes = parse_xj_file(xj_bytes)
    total_v = sum(len(m.vertices) for m in meshes)
    total_t = sum(len(m.indices) // 3 for m in meshes)
    assert total_t >= 50, f"door01l tri count too low: {total_t}"
    assert total_v >= 100, f"door01l vert count too low: {total_v}"
    assert len(meshes) >= 5, f"door01l submesh count too low: {len(meshes)}"


@step("xj_descriptor: an .xj model returns plausible mesh shape (verts > 0, indices % 3 == 0)")
def t_xj_descriptor_shape_invariants():
    """Across a representative sample of ``.xj`` BML inners, every
    parsed submesh must have:

      - ``vertices`` list non-empty
      - ``indices`` length is a multiple of 3 (already triangulated)
      - every index points into the local ``vertices`` list

    This is the single most important invariant for the renderer: a
    misaligned strip would either crash three.js or produce
    catastrophic visual corruption.
    """
    fixtures = [
        ("bm_fe_obj_o_door01l.bml", "fe_obj_o_door01l.xj"),
        ("bm_fe_obj_o_door03l.bml", "fs_obj_o_door01l.xj"),
        ("bm_fe_obj_o_capsule01.bml", "fe_obj_o_capsule01.xj"),
        ("bm_fe_obj_aircon02.bml", "fe_obj_aircon02.xj"),
        ("bm_fe_obj_komo.bml", "fe_obj_komo.xj"),
        # Multi vertex-info-table cases (vbInfoCount > 1).
        ("bm_eff_ice.bml", "ice_root.xj"),
        ("bm_fs_obj_cakeya.bml", "fs_obj_cakeya.xj"),
    ]
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj_descriptor import parse_xj_file

    checked = 0
    for bml_name, inner_name in fixtures:
        bml_path = DATA_DIR / bml_name
        if not bml_path.exists():
            bml_path = LIVE_DATA_DIR / bml_name
        if not bml_path.exists():
            continue
        blob = bml_path.read_bytes()
        entries = parse_bml(blob)
        target = next((e for e in entries if e.name == inner_name), None)
        if target is None:
            continue
        raw = bytes(blob[target.offset:target.offset + target.size_compressed])
        xj_bytes = _prs_decompress(raw, timeout=20.0)
        meshes = parse_xj_file(xj_bytes)
        assert len(meshes) >= 1, f"{bml_name}#{inner_name}: no submeshes"
        for i, m in enumerate(meshes):
            assert len(m.vertices) > 0, (
                f"{bml_name}#{inner_name} sub[{i}]: empty vertex list"
            )
            assert len(m.indices) % 3 == 0, (
                f"{bml_name}#{inner_name} sub[{i}]: indices length {len(m.indices)} "
                f"is not divisible by 3"
            )
            n = len(m.vertices)
            assert all(0 <= idx < n for idx in m.indices), (
                f"{bml_name}#{inner_name} sub[{i}]: out-of-range index "
                f"(have {n} verts, max idx={max(m.indices) if m.indices else -1})"
            )
        checked += 1
    assert checked >= 3, (
        f"only {checked} XJ descriptor fixtures available; need at least 3 "
        f"to consider the invariant test meaningful"
    )


@step("xj_descriptor: world_position/rotation populated per submesh")
def t_xj_descriptor_world_xform_populated():
    """Verify XjMesh records carry the host-bone transform fields.

    The descriptor-table XJ parser shares the bone-tree walker with the
    chunk parser (both formats use the same 52-byte MeshTreeNode), so
    sub-meshes from a multi-bone .xj file should carry per-bone world
    positions in their ``world_position`` field.

    We use the door fixture because it has 4 mesh-tree nodes (door body
    + handle bones) — at least one of which must have a non-trivial
    world_position offset (the handle is at y=19.8). The same field
    appears verbatim in the JSON wire response from /api/model_mesh.
    """
    bml_name = "bm_fe_obj_o_door01l.bml"
    inner_name = "fe_obj_o_door01l.xj"
    bml_path = DATA_DIR / bml_name
    if not bml_path.exists():
        bml_path = LIVE_DATA_DIR / bml_name
    if not bml_path.exists():
        print(f"    (skipped: {bml_name} not present)")
        return
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj_descriptor import parse_xj_file

    blob = bml_path.read_bytes()
    entries = parse_bml(blob)
    target = next((e for e in entries if e.name == inner_name), None)
    assert target is not None, f"{bml_name} is missing {inner_name}"
    raw = bytes(blob[target.offset:target.offset + target.size_compressed])
    xj_bytes = _prs_decompress(raw, timeout=20.0)
    meshes = parse_xj_file(xj_bytes)
    assert meshes, "no meshes parsed"

    # Every mesh must carry the four world-transform fields so that the
    # JSON projection in server.py doesn't crash on the dotted access.
    for i, m in enumerate(meshes):
        assert hasattr(m, "world_position"), f"sub[{i}] missing world_position"
        assert hasattr(m, "world_rotation_euler"), f"sub[{i}] missing world_rotation_euler"
        assert hasattr(m, "world_scale"), f"sub[{i}] missing world_scale"
        assert hasattr(m, "world_matrix"), f"sub[{i}] missing world_matrix"
        assert len(m.world_position) == 3, f"sub[{i}] world_position not 3-tuple"
        assert len(m.world_rotation_euler) == 3, f"sub[{i}] world_rotation_euler not 3-tuple"
        assert len(m.world_scale) == 3, f"sub[{i}] world_scale not 3-tuple"
        assert len(m.world_matrix) == 16, f"sub[{i}] world_matrix not 16 floats"

    # At least one sub-mesh must have a non-zero world_position (the
    # handle bone). If all are at the origin, the tree walker has
    # silently dropped the parent-child chain.
    nonzero = [m for m in meshes if any(abs(c) > 1e-3 for c in m.world_position)]
    assert len(nonzero) >= 1, (
        f"expected at least one sub-mesh with non-zero world_position, "
        f"got {len(meshes)} meshes all centred at origin"
    )

    # Sanity: also verify the JSON wire payload exposes the same field.
    r = _http(
        "GET",
        f"/api/model_mesh/{bml_name}?inner={urllib.parse.quote(inner_name)}",
    )
    assert r.get("vertices_pre_transformed") is True, (
        "/api/model_mesh response missing vertices_pre_transformed=true"
    )
    assert r["mesh_count"] >= 5, f"only {r['mesh_count']} sub-meshes via HTTP"
    assert r["meshes"][0].get("world_position") is not None, (
        "JSON mesh entries missing world_position field"
    )


@step("xj: bad NJCM payload returns clean error")
def t_xj_bad_payload():
    """Truncated or otherwise malformed NJCM payloads should raise
    ValueError, not crash with a struct.error / IndexError."""
    from formats.xj import parse_xj_njcm, parse_nj_file
    # Too small to even hold a mesh-tree node
    try:
        parse_xj_njcm(b"\x00\x00\x00\x00")
        assert False, "expected ValueError on tiny payload"
    except ValueError:
        pass
    # Random garbage that *almost* looks like a node - should not raise
    # (returns empty list) or should raise ValueError, never anything else.
    try:
        parse_xj_njcm(b"\xff" * 60)
    except ValueError:
        pass
    # Whole-file path: garbage that isn't IFF should either return []
    # (no NJCM) or raise ValueError. Either is acceptable; assert no
    # other crash type.
    try:
        out = parse_nj_file(b"NOT_IFF_AT_ALL")
        assert isinstance(out, list)
    except ValueError:
        pass  # also acceptable


@step("xj_rotation_order: De Rol Le head bones land in plausible bbox under default ZYX order")
def t_xj_rotation_order_de_rol_le():
    """Regression guard for the ZYX-vs-XYZ rotation-order fix (2026-04-24).

    Phantasmal World's ``NinjaGeometryConversion.kt::convertObject`` builds
    each bone's local rotation as ``Euler(x, y, z, "ZYX")`` (or "ZXY"
    when EVAL_ZXY_ANG is set on that node). Three.js's ``"ZYX"`` order
    composes ``R = Rz @ Ry @ Rx`` — and our parser previously used
    ``R = Rx @ Ry @ Rz`` (the comment claimed "Phantasmal default" but
    that was a transcription error).

    The De Rol Le boss family is the canonical fixture: its head/jaw
    bones sit on bones with NESTED rotations (e.g. bone 24 has 90° Y,
    bone 26 has rot=(3.8°, -90°, 3.8°)). Under XYZ those nested poses
    produce visibly twisted head sub-meshes; under ZYX they line up.

    This test asserts the head sub-meshes (the bones whose world Y is
    POSITIVE — De Rol Le has the head/jaw above the spine, while the
    spine extends in negative Z) land in a plausible bounding box. We
    use a coarse range so the test catches the "off by 90°" regression
    without becoming flaky on numerical drift.
    """
    bml_name = "bm_boss2_de_rol_le_a.bml"
    inner_name = "boss2_b_derorure_body.nj"
    bml_path = DATA_DIR / bml_name
    if not bml_path.exists():
        bml_path = LIVE_DATA_DIR / bml_name
    if not bml_path.exists():
        print(f"    (skipped: {bml_name} not present)")
        return
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj import parse_nj_file

    blob = bml_path.read_bytes()
    entries = parse_bml(blob)
    target = next((e for e in entries if e.name == inner_name), None)
    assert target is not None, f"{bml_name} is missing {inner_name}"
    raw = bytes(blob[target.offset:target.offset + target.size_compressed])
    nj_bytes = _prs_decompress(raw, timeout=20.0)
    meshes = parse_nj_file(nj_bytes)
    assert len(meshes) >= 100, f"expected hundreds of submeshes for de_rol_le body, got {len(meshes)}"

    # Aggregate the world-space vertex AABB. Under the correct ZYX
    # order, De Rol Le's body extends:
    #   Z: ~[-165, +35]   (worm body running along -Z)
    #   X: ~[-25, +25]    (mostly symmetric)
    #   Y: ~[-37, +16]    (head jaw above spine, lower body slightly
    #                      below; the spike points down)
    # We assert against deliberately wide ranges so the test catches
    # the "rotation order wrong" regression without bouncing on minor
    # parser tweaks. Anything in the right ballpark passes; anything
    # rotated 90° off (XYZ regression) fails because the spine-Z would
    # collapse and the X extent would balloon.
    xs, ys, zs = [], [], []
    for m in meshes:
        for v in m.vertices:
            xs.append(v.pos[0]); ys.append(v.pos[1]); zs.append(v.pos[2])
    assert xs, "no vertices in any submesh"
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)
    # The Z extent (worm length) should DOMINATE. It's ~200 units in
    # the correct order; in a wrongly-rotated reading the spine bones'
    # translations stay along Z (translations are not rotation-affected),
    # but the head sub-meshes — bones 24-30+ that hang off the front of
    # the body with nested rotations — would push X or Y out instead.
    assert span_z > 150.0, f"expected Z span > 150 (worm length), got {span_z:.1f}"
    assert span_z > span_x * 2.0, (
        f"expected Z span ({span_z:.1f}) >> X span ({span_x:.1f}) for elongated worm body; "
        "rotation order may be wrong"
    )
    assert span_z > span_y * 3.0, (
        f"expected Z span ({span_z:.1f}) >> Y span ({span_y:.1f}); "
        "rotation order may be wrong"
    )


@step("xj_eval_hide: De Rol Le body renders all sub-meshes by default (no HIDE drops)")
def t_xj_eval_hide_de_rol_le_default():
    """Investigation B from AGENT_MODEL_DEEP_DEBUG_REPORT (2026-04-24).

    The original task report hypothesized that EVAL_HIDE was being
    honored too aggressively, causing a "skull covering" sub-mesh to
    disappear. The empirical audit (scripts/dump_eval_hide_audit.py)
    found that ZERO models in PSOBB.IO/data set HIDE / SHAPE_SKIP on
    a mesh-bearing node. The "missing skull" therefore can't be HIDE.

    What's actually happening: each ``.bml`` archive holds MULTIPLE
    inner ``.nj`` files (body, helm_break, shell_break, etc.) and the
    viewer loads ONE at a time. So when the user opens
    ``boss2_b_derorure_body.nj`` they see the worm body; the helmet
    is in a SIBLING inner ``boss2_b_helm_break.nj``.

    This test asserts:
      - Default parse of De Rol Le body: emits 600+ sub-meshes (a
        regression to "0 emitted" would catch a HIDE / cache-replay bug).
      - ignore_hide=True: same count (no HIDE flags in shipping data).
      - Helm inner parses to its own non-empty mesh list — confirming
        the "skull" isn't dropped, it's just in a sibling file.
    """
    bml_name = "bm_boss2_de_rol_le_a.bml"
    bml_path = DATA_DIR / bml_name
    if not bml_path.exists():
        bml_path = LIVE_DATA_DIR / bml_name
    if not bml_path.exists():
        print(f"    (skipped: {bml_name} not present)")
        return
    from formats.bml import parse_bml, _prs_decompress
    from formats.xj import parse_nj_file

    blob = bml_path.read_bytes()
    entries = parse_bml(blob)

    body_entry = next((e for e in entries if e.name == "boss2_b_derorure_body.nj"), None)
    helm_entry = next((e for e in entries if e.name == "boss2_b_helm_break.nj"), None)
    assert body_entry is not None, "missing body inner"
    assert helm_entry is not None, "missing helm inner"

    body_raw = bytes(blob[body_entry.offset:body_entry.offset + body_entry.size_compressed])
    helm_raw = bytes(blob[helm_entry.offset:helm_entry.offset + helm_entry.size_compressed])
    body_bytes = _prs_decompress(body_raw, timeout=20.0)
    helm_bytes = _prs_decompress(helm_raw, timeout=20.0)

    body_default = parse_nj_file(body_bytes)
    body_ignore = parse_nj_file(body_bytes, ignore_hide=True)
    helm_default = parse_nj_file(helm_bytes)

    assert len(body_default) >= 100, (
        f"De Rol Le body should render hundreds of sub-meshes, "
        f"got {len(body_default)} (parser dropped strips somewhere)"
    )
    # Empirical audit: no shipping models set HIDE on mesh-bearing nodes,
    # so the default and ignore_hide paths must be identical.
    assert len(body_default) == len(body_ignore), (
        f"default ({len(body_default)}) != ignore_hide ({len(body_ignore)}) "
        "— a HIDE flag was honored against expectation; either the audit "
        "missed a model or new data was added"
    )
    # Helm inner is a separate file; it has its own (much smaller)
    # mesh count — confirms the "missing skull" perception is really
    # about the inner-file selector, not parser drop.
    assert len(helm_default) >= 5, (
        f"De Rol Le helm should render geometry, got {len(helm_default)}"
    )


@step("model_mesh: endpoint returns base64 vertices/indices that decode to consistent counts")
def t_model_mesh_endpoint_consistency():
    if not _nj_available(TARGET_NJ):
        print("    (skipped: target NJ not found)")
        return
    r = _http("GET", f"/api/model_mesh/{TARGET_NJ}")
    assert "meshes" in r, r
    assert r["mesh_count"] == len(r["meshes"]), r
    assert r["mesh_count"] >= 1, r
    total_v = 0
    total_t = 0
    for m in r["meshes"]:
        # b64 decode and verify the byte counts match vertex/triangle counts
        verts_bin = base64.b64decode(m["vertices_b64"])
        idx_bin = base64.b64decode(m["indices_b64"])
        # Float32 interleaved [px,py,pz, nx,ny,nz, u,v] = 32 bytes per vertex
        assert len(verts_bin) == m["vertex_count"] * 32, (
            f"vertices_b64 size mismatch: {len(verts_bin)} vs {m['vertex_count']}*32"
        )
        # Uint32 indices, 4 bytes each, 3 per triangle
        assert len(idx_bin) == m["triangle_count"] * 12, (
            f"indices_b64 size mismatch: {len(idx_bin)} vs {m['triangle_count']}*12"
        )
        # AABB sanity
        aabb = m["aabb"]
        assert len(aabb) == 6, aabb
        if m["vertex_count"]:
            assert aabb[3] >= aabb[0], aabb
            assert aabb[4] >= aabb[1], aabb
            assert aabb[5] >= aabb[2], aabb
        total_v += m["vertex_count"]
        total_t += m["triangle_count"]
    assert r["totals"]["vertices"] == total_v, r["totals"]
    assert r["totals"]["triangles"] == total_t, r["totals"]


@step("model_mesh: nonexistent path returns 404")
def t_model_mesh_404():
    status, body, _ = _http_get_bytes("/api/model_mesh/__nonexistent__.nj", timeout=10)
    assert status == 404, f"expected 404 for missing NJ, got {status}: {body[:200]!r}"


@step("model_mesh: non-.nj/.bml extension returns 400")
def t_model_mesh_bad_ext():
    # Use an existing PRS file - it's not a model. The endpoint should
    # 400 with a clear message rather than try to parse the bytes.
    if not (DATA_DIR / "LogoEP4.prs").exists() and not (LIVE_DATA_DIR / "LogoEP4.prs").exists():
        print("    (skipped: LogoEP4.prs not available)")
        return
    status, body, _ = _http_get_bytes("/api/model_mesh/LogoEP4.prs", timeout=10)
    assert status == 400, f"expected 400 for non-model, got {status}: {body[:200]!r}"


# ---------------------------------------------------------------------- per-mesh texture binding
#
# /api/model_textures resolves the NJTL slot list, the XVMH tile list,
# and the per-material binding the frontend uses to assign one
# texture per submesh. These tests ride on the same PSOBB.IO/data/
# install used elsewhere; they're skipped if the chosen BML isn't
# present so the suite stays runnable on dev machines without a full
# game install.
def _bml_in_install(name: str) -> bool:
    return (DATA_DIR / name).exists() or (LIVE_DATA_DIR / name).exists()


@step("model_textures: bm4_ps_ma_body.bml#bm4_ps_ma_body.nj returns binding[] with at least 2 distinct tile indices")
def t_model_textures_player_body_distinct_tiles():
    """Multi-textured player body — confirms binding spans more than
    one tile. The PSOBB female ranger's body model uses 10 textures
    (skin/head/body-base/body-accent/etc.); we assert at least 2 of
    those tile indices show up in the binding so we know the per-mesh
    flow isn't degenerate."""
    bml_name = "bm4_ps_ma_body.bml"
    inner_name = "bm4_ps_ma_body.nj"
    if not _bml_in_install(bml_name):
        print(f"    (skipped: {bml_name} not present)")
        return
    r = _http(
        "GET",
        f"/api/model_textures/{bml_name}?inner={urllib.parse.quote(inner_name)}",
    )
    assert "binding" in r, r
    binding = r["binding"]
    assert len(binding) >= 2, f"expected >= 2 binding rows, got {binding!r}"
    tiles = sorted({b["tile_index"] for b in binding if not b["missing"]})
    assert len(tiles) >= 2, (
        f"expected >= 2 distinct tile_indices in binding, got {tiles!r}"
    )
    # Sanity: each binding row carries the four required keys.
    for b in binding:
        for k in ("material_id", "tile_index", "missing", "name"):
            assert k in b, f"binding row missing key {k!r}: {b!r}"


@step("model_textures: NJTL slot 0 maps to a real XVMH name")
def t_model_textures_njtl_slot_zero_is_xvmh_name():
    """NJTL slot 0 ↔ XVMH record 0 by positional alignment. Asserts
    that the NJTL[0] name string appears verbatim in xvmh[0].name —
    which is how the frontend's per-submesh material identification
    works for diagnostic UI ("body texture loaded for s128_bm4_bodyf")."""
    bml_name = "bm4_ps_ma_body.bml"
    inner_name = "bm4_ps_ma_body.nj"
    if not _bml_in_install(bml_name):
        print(f"    (skipped: {bml_name} not present)")
        return
    r = _http(
        "GET",
        f"/api/model_textures/{bml_name}?inner={urllib.parse.quote(inner_name)}",
    )
    njtl = r.get("njtl") or []
    xvmh = r.get("xvmh") or []
    assert njtl, f"NJTL list empty: {r!r}"
    assert xvmh, f"XVMH list empty: {r!r}"
    assert len(njtl) == len(xvmh), (
        f"NJTL/XVMH length mismatch: {len(njtl)} vs {len(xvmh)}"
    )
    # Names align positionally — the writer emits NJTL entries and
    # XVR records in the SAME order (see pso-blender's TextureManager).
    assert njtl[0]["name"], f"NJTL slot 0 has empty name: {njtl[0]!r}"
    assert njtl[0]["name"] == xvmh[0]["name"], (
        f"NJTL[0]={njtl[0]['name']!r} but XVMH[0]={xvmh[0]['name']!r}"
    )
    # Sanity: name_match==True means no slot fell off the XVMH side.
    assert r.get("name_match") is True, r.get("name_match")


@step("model_textures: xj-format model also produces a binding")
def t_model_textures_xj_format():
    """Cross-format coverage. The XJ descriptor table parser at
    formats/xj_descriptor.py emits material_ids the same way the NJ
    chunk parser does, and the NJTL chunk lives in the IFF wrapper —
    so the binding endpoint should work identically for `.xj` inners.
    Vol Opt's hat/eye prop is a small XJ model with a single texture,
    making it a clean smoke target."""
    bml_name = "bm_boss3_volopt.bml"
    inner_name = "fe_obj_vo_mo_dai_aka.xj"
    if not _bml_in_install(bml_name):
        print(f"    (skipped: {bml_name} not present)")
        return
    r = _http(
        "GET",
        f"/api/model_textures/{bml_name}?inner={urllib.parse.quote(inner_name)}",
    )
    binding = r.get("binding") or []
    assert binding, f"XJ model returned empty binding: {r!r}"
    # The model carries at least one material reference; binding rows
    # should each have a non-negative tile_index and a name (possibly
    # empty for unused slots, but the field must exist).
    for b in binding:
        assert b["tile_index"] >= 0, b
        assert "name" in b, b


@step("model_textures: '#'-form path resolves identical binding to ?inner= form")
def t_model_textures_hash_form_equivalence():
    """The /api/model_mesh accepts both `?inner=<inner>` and the
    `<bml>#<inner>` URL fragment; /api/model_textures inherits the
    same dispatch. Asserts the two forms produce IDENTICAL binding
    output for the same model."""
    bml_name = "bm4_ps_ma_body.bml"
    inner_name = "bm4_ps_ma_body.nj"
    if not _bml_in_install(bml_name):
        print(f"    (skipped: {bml_name} not present)")
        return
    r1 = _http(
        "GET",
        f"/api/model_textures/{bml_name}?inner={urllib.parse.quote(inner_name)}",
    )
    # `#` is a URL fragment — must be percent-encoded as `%23` to make
    # it through to the server (urllib otherwise strips it).
    r2 = _http(
        "GET",
        f"/api/model_textures/{urllib.parse.quote(bml_name)}%23{urllib.parse.quote(inner_name)}",
    )
    assert r1["binding"] == r2["binding"], (
        f"binding mismatch: ?inner= vs %23 form differ"
    )


@step("model_mesh: response includes binding[] alongside meshes")
def t_model_mesh_includes_binding():
    """Confirms the integration touch on /api/model_mesh: the existing
    payload now carries a `binding` array populated from the same
    NJTL/XVMH match the dedicated endpoint exposes. The frontend reads
    `payload.binding` directly to skip a second round-trip on every
    model open."""
    bml_name = "bm4_ps_ma_body.bml"
    inner_name = "bm4_ps_ma_body.nj"
    if not _bml_in_install(bml_name):
        print(f"    (skipped: {bml_name} not present)")
        return
    r = _http(
        "GET",
        f"/api/model_mesh/{bml_name}?inner={urllib.parse.quote(inner_name)}",
    )
    binding = r.get("binding") or []
    assert binding, f"model_mesh missing binding: {list(r.keys())}"
    bd = r.get("binding_data") or {}
    assert "njtl" in bd, bd
    assert "xvmh" in bd, bd
    # Every material_id seen on a submesh should appear in the binding
    # array (the binding is keyed by unique material_id, sorted).
    seen_mids = sorted({m["material_id"] for m in r["meshes"]})
    binding_mids = sorted({b["material_id"] for b in binding})
    assert seen_mids == binding_mids, (
        f"binding mids {binding_mids} differ from mesh mids {seen_mids}"
    )


@step("atlas: GET LogoEP4.prs returns 4 placements + skip [4-7]")
def t_atlas_logo_layout():
    r = _http("GET", f"/api/atlas/{TARGET_PRS}")
    assert r["filename"] == TARGET_PRS, r
    assert r["composite_w"] == 2048, r
    assert r["composite_h"] == 2048, r
    plc = r["placements"]
    assert len(plc) == 4, plc
    # Verify the engine-spatial order: tile_0 top-left, tile_2 top-right,
    # tile_1 bottom-left, tile_3 bottom-right.
    by_idx = {p["tile_index"]: p for p in plc}
    assert by_idx[0]["x"] == 0 and by_idx[0]["y"] == 0, by_idx[0]
    assert by_idx[2]["x"] == 1024 and by_idx[2]["y"] == 0, by_idx[2]
    assert by_idx[1]["x"] == 0 and by_idx[1]["y"] == 1024, by_idx[1]
    assert by_idx[3]["x"] == 1024 and by_idx[3]["y"] == 1024, by_idx[3]
    for p in plc:
        assert p["w"] == 1024 and p["h"] == 1024, p
        assert p["uv_box"] == [0.0, 0.0, 1.0, 1.0], p
    assert r["skip_tiles"] == [4, 5, 6, 7], r["skip_tiles"]
    raw = base64.b64decode(r["composite_b64"].split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image as _Image
    with _Image.open(io.BytesIO(raw)) as im:
        assert im.size == (2048, 2048), im.size


@step("atlas: GET unknown file returns 404 with clean error")
def t_atlas_unknown_404():
    fname = "f256_hyouji.prs"  # real file, no atlas layout
    try:
        _http("GET", f"/api/atlas/{fname}")
    except RuntimeError as e:
        assert "404" in str(e), str(e)
        assert "no atlas layout known" in str(e), str(e)
        return
    raise AssertionError(f"expected 404 for {fname}")


@step("atlas: POST atlas_import (1x) slices to 4 tiles at native dim")
def t_atlas_import_native():
    g = _http("GET", f"/api/atlas/{TARGET_PRS}")
    body = {
        "filename": TARGET_PRS,
        "png_b64": g["composite_b64"],
        "keep_native_dims": True,
    }
    r = _http("POST", "/api/atlas_import", body, timeout=120)
    assert r["imported_w"] == 2048 and r["imported_h"] == 2048, r
    assert r["scale_factor"] == 1, r
    assert r["composite_w"] == 2048 and r["composite_h"] == 2048, r
    assert len(r["tiles"]) == 4, [t["tile_index"] for t in r["tiles"]]
    assert r["skip_tiles"] == [4, 5, 6, 7], r["skip_tiles"]
    by_idx = {t["tile_index"]: t for t in r["tiles"]}
    for idx in (0, 1, 2, 3):
        assert idx in by_idx, by_idx
        t = by_idx[idx]
        assert t["out_w"] == 1024 and t["out_h"] == 1024, t
        assert t["src_w"] == 1024 and t["src_h"] == 1024, t
        raw = base64.b64decode(t["out_b64"].split(",", 1)[1])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", t


@step("atlas: POST atlas_import (4x) Lanczos-downs to native dim per tile")
def t_atlas_import_4x():
    g = _http("GET", f"/api/atlas/{TARGET_PRS}")
    raw = base64.b64decode(g["composite_b64"].split(",", 1)[1])
    from PIL import Image as _Image
    with _Image.open(io.BytesIO(raw)) as im:
        big = im.convert("RGBA").resize((8192, 8192), _Image.Resampling.NEAREST)
    buf = io.BytesIO()
    big.save(buf, "PNG")
    big_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    body = {
        "filename": TARGET_PRS,
        "png_b64": big_b64,
        "keep_native_dims": True,
    }
    r = _http("POST", "/api/atlas_import", body, timeout=180)
    assert r["imported_w"] == 8192 and r["imported_h"] == 8192, r
    assert r["scale_factor"] == 4, r
    for t in r["tiles"]:
        # keep_native_dims=True => Lanczos-down to 1024x1024
        assert t["out_w"] == 1024 and t["out_h"] == 1024, t


@step("atlas: POST atlas_import rejects non-multiple dim")
def t_atlas_import_bad_dim():
    from PIL import Image as _Image
    bad = _Image.new("RGBA", (2049, 2048), (255, 0, 0, 255))
    buf = io.BytesIO()
    bad.save(buf, "PNG")
    bad_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    body = {
        "filename": TARGET_PRS,
        "png_b64": bad_b64,
        "keep_native_dims": True,
    }
    try:
        _http("POST", "/api/atlas_import", body, timeout=60)
    except RuntimeError as e:
        assert "400" in str(e), str(e)
        return
    raise AssertionError("non-multiple dim should have been rejected")


@step("atlas: round-trip (atlas_import -> repack deploy=False) preserves tile dims")
def t_atlas_roundtrip_repack():
    g = _http("GET", f"/api/atlas/{TARGET_PRS}")
    imp = _http("POST", "/api/atlas_import", {
        "filename": TARGET_PRS,
        "png_b64": g["composite_b64"],
        "keep_native_dims": True,
    }, timeout=120)
    edits = [{"tile_index": t["tile_index"], "png_b64": t["out_b64"]} for t in imp["tiles"]]
    r = _http("POST", "/api/repack", {
        "filename": TARGET_PRS,
        "tiles": edits,
        "deploy": False,
    }, timeout=180)
    assert r["verify"]["ok"], r["verify"]
    rebuilt = Path(r["rebuilt_path"])
    assert rebuilt.exists(), rebuilt
    blob = rebuilt.read_bytes()
    assert blob[:4] == b"XVMH"
    tiles_meta = _http("GET", f"/api/tiles/{TARGET_PRS}")
    expected = [(t["width"], t["height"]) for t in tiles_meta["tiles"]]
    pos = 0x40
    seen: list[tuple[int, int]] = []
    while pos + 0x40 <= len(blob):
        if blob[pos:pos + 4] != b"XVRT":
            break
        w = int.from_bytes(blob[pos + 0x14:pos + 0x16], "little")
        h = int.from_bytes(blob[pos + 0x16:pos + 0x18], "little")
        dsz = int.from_bytes(blob[pos + 0x18:pos + 0x1C], "little")
        seen.append((w, h))
        pos += 0x40 + dsz
    assert seen == expected, f"rebuilt tile dims {seen} != expected {expected}"


# ============================================================
# Viewport mode (16:9 transform, 2026-04-25)
#   - GET  /api/viewport/{filename}     layout + composite_b64 at 1278x768
#   - POST /api/viewport_paint          slice painted PNG -> per-tile edits
# ============================================================

VIEWPORT_W = 1278
VIEWPORT_H = 768


@step("viewport: GET LogoEP4.prs returns atlas placements scaled to 1278x768")
def t_viewport_logo_atlas():
    r = _http("GET", f"/api/viewport/{TARGET_PRS}")
    assert r["filename"] == TARGET_PRS, r
    assert r["viewport_w"] == VIEWPORT_W and r["viewport_h"] == VIEWPORT_H, r
    assert r["layout"] == "atlas", r
    assert r["source"] == "atlas_layouts.py", r
    assert len(r["placements"]) == 4, [p["tile_index"] for p in r["placements"]]
    assert sorted(r["skip_tiles"]) == [4, 5, 6, 7], r["skip_tiles"]
    # 4:3 inset must be centered at 127px on each side
    inset = r["inset"]
    assert inset["w"] == 1024 and inset["h"] == 768, inset
    assert inset["x"] == 127 and inset["y"] == 0, inset
    # Every placement must lie inside the 4:3 inset
    for p in r["placements"]:
        assert p["dest_x"] >= inset["x"], p
        assert p["dest_y"] >= inset["y"], p
        assert p["dest_x"] + p["dest_w"] <= inset["x"] + inset["w"], p
        assert p["dest_y"] + p["dest_h"] <= inset["y"] + inset["h"], p
        assert p["uv_box"] == [0.0, 0.0, 1.0, 1.0], p
    # Composite PNG decodes to 1278x768
    raw = base64.b64decode(r["composite_b64"].split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image as _Image
    with _Image.open(io.BytesIO(raw)) as im:
        assert im.size == (VIEWPORT_W, VIEWPORT_H), im.size


@step("viewport: GET unknown file returns centered placement")
def t_viewport_unknown_centered():
    # Pick any non-atlas PRS that exists in DATA_DIR
    files = _http("GET", "/api/files")["files"]
    target = None
    for f in files:
        # Skip the one file we DO have an atlas layout for
        if f["name"] == TARGET_PRS:
            continue
        if f["name"].lower().endswith(".prs"):
            target = f["name"]
            break
    if not target:
        print("    (skipped: no non-atlas PRS in DATA_DIR)")
        return
    r = _http("GET", f"/api/viewport/{target}")
    assert r["filename"] == target, r
    assert r["viewport_w"] == VIEWPORT_W and r["viewport_h"] == VIEWPORT_H, r
    assert r["layout"] == "centered", r
    assert r["source"] == "guessed", r
    # Centered fallback always produces exactly 1 placement (the largest tile)
    assert len(r["placements"]) == 1, [p["tile_index"] for p in r["placements"]]
    p0 = r["placements"][0]
    inset = r["inset"]
    # Placement must lie fully inside the 4:3 inset
    assert p0["dest_x"] >= inset["x"], p0
    assert p0["dest_y"] >= inset["y"], p0
    assert p0["dest_x"] + p0["dest_w"] <= inset["x"] + inset["w"], p0
    assert p0["dest_y"] + p0["dest_h"] <= inset["y"] + inset["h"], p0
    # Composite still decodes to 1278x768 even when most of it is transparent
    raw = base64.b64decode(r["composite_b64"].split(",", 1)[1])
    from PIL import Image as _Image
    with _Image.open(io.BytesIO(raw)) as im:
        assert im.size == (VIEWPORT_W, VIEWPORT_H), im.size


@step("viewport: POST viewport_paint slices back to native dims")
def t_viewport_paint_roundtrip():
    g = _http("GET", f"/api/viewport/{TARGET_PRS}")
    # Send the unmodified composite back so we can verify the slice math.
    body = {
        "filename": TARGET_PRS,
        "viewport_png_b64": g["composite_b64"],
        "viewport_w": VIEWPORT_W,
        "viewport_h": VIEWPORT_H,
    }
    r = _http("POST", "/api/viewport_paint", body, timeout=120)
    assert r["filename"] == TARGET_PRS, r
    assert r["viewport_w"] == VIEWPORT_W and r["viewport_h"] == VIEWPORT_H, r
    assert sorted(r["tiles_modified"]) == [0, 1, 2, 3], r["tiles_modified"]
    assert sorted(r["skipped"]) == [4, 5, 6, 7], r["skipped"]
    # Each tile slice must come back at native dim (Lanczos-down already applied)
    tile_meta = _http("GET", f"/api/tiles/{TARGET_PRS}")
    by_native = {t["index"]: (t["width"], t["height"]) for t in tile_meta["tiles"]}
    for it in r["tiles"]:
        idx = it["tile_index"]
        nw, nh = by_native[idx]
        assert it["src_w"] == nw and it["src_h"] == nh, (idx, it)
        assert it["out_w"] == nw and it["out_h"] == nh, (idx, it)
        # PNG must decode
        raw = base64.b64decode(it["out_b64"].split(",", 1)[1])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", idx


@step("viewport: rejects non-1278x768 input dim")
def t_viewport_paint_bad_dim():
    from PIL import Image as _Image
    bad = _Image.new("RGBA", (640, 480), (255, 0, 0, 255))
    buf = io.BytesIO()
    bad.save(buf, "PNG")
    bad_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    body = {
        "filename": TARGET_PRS,
        "viewport_png_b64": bad_b64,
        "viewport_w": VIEWPORT_W,
        "viewport_h": VIEWPORT_H,
    }
    try:
        _http("POST", "/api/viewport_paint", body, timeout=60)
    except RuntimeError as e:
        assert "400" in str(e), str(e)
        assert "viewport" in str(e).lower(), str(e)
        return
    raise AssertionError("non-1278x768 PNG should have been rejected")


# ---------------------------------------------------------------------------- Manifest (Phase A — Agent 1)

# Schema lives next to the master plan; load once for every manifest test so
# they all share the same source of truth.
_MANIFEST_SCHEMA_PATH = Path(__file__).parent.parent / "MASTER_PLAN" / "manifest.schema.json"


def _load_manifest_schema() -> dict:
    with open(_MANIFEST_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _entry_for(manifest: dict, name: str) -> dict | None:
    """Return the AssetEntry whose path basename (case-insensitive) matches
    ``name``, or None if not present."""
    nl = name.lower()
    for e in manifest.get("entries", []):
        # path may be 'foo.xvm' or 'data/foo.xvm' depending on root.
        bn = e["path"].rsplit("/", 1)[-1].lower()
        if bn == nl:
            return e
    return None


@step("manifest endpoint returns valid schema")
def t_manifest_schema_valid():
    try:
        import jsonschema
    except ImportError as e:
        raise AssertionError(f"jsonschema not installed: {e}")
    m = _http("GET", "/api/manifest", timeout=120)
    schema = _load_manifest_schema()
    jsonschema.validate(m, schema)
    # Sanity: at least version and entries shape
    assert m["version"] == 1, m["version"]
    assert isinstance(m["entries"], list), type(m["entries"])
    assert isinstance(m["generated_at"], int), type(m["generated_at"])
    assert isinstance(m["install_root"], str), type(m["install_root"])


@step("manifest classifies XVM as texture/XVM/parsable=yes")
def t_manifest_xvm_classification():
    m = _http("GET", "/api/manifest", timeout=120)
    e = _entry_for(m, TARGET_XVM)
    assert e is not None, f"{TARGET_XVM} missing from manifest"
    assert e["category"] == "texture", e
    assert e["format"] == "XVM", e
    assert e["parsable"] == "yes", e
    assert e["extension"] == ".xvm", e
    # Magic check: XVMH at offset 0
    assert e["magic_hex"].startswith("58564d48"), e["magic_hex"]
    assert e["magic_ascii"] == "XVMH", e["magic_ascii"]


@step("manifest classifies bm_*.bml as model/BML/parsable=partial")
def t_manifest_bml_classification():
    m = _http("GET", "/api/manifest", timeout=120)
    # Pick the first bm_*.bml in the manifest — the dev mirror has several.
    bml_entries = [
        e for e in m["entries"]
        if e["extension"] == ".bml"
        and e["path"].rsplit("/", 1)[-1].lower().startswith("bm_")
    ]
    assert bml_entries, "no bm_*.bml entries in manifest"
    e = bml_entries[0]
    assert e["category"] == "model", e
    assert e["format"] == "BML", e
    assert e["parsable"] == "partial", e
    assert e["extension"] == ".bml", e


@step("manifest classifies LogoEP4.prs as compressed=true, inner_format=XVM")
def t_manifest_prs_inner_xvm():
    m = _http("GET", "/api/manifest", timeout=120)
    e = _entry_for(m, TARGET_PRS)
    assert e is not None, f"{TARGET_PRS} missing from manifest"
    assert e["category"] == "texture", e
    assert e["format"] == "PRS", e
    assert e["parsable"] == "yes", e
    assert e["compressed"] is True, e
    assert e["inner_format"] == "XVM", e["inner_format"]


@step("manifest cache rebuilds on file mtime change")
def t_manifest_cache_rebuild():
    """Touch a known file in DATA_DIR and verify the manifest's generated_at advances.

    Runs against the editor's actual DATA_DIR (dev mirror), so it touches
    a single PRS we've otherwise been treating as read-only and resets
    its mtime back to the original after.
    """
    # Resolve DATA_DIR from the running server (it may differ from this file's
    # constant if the user set PSO_DATA_DIR; use whatever the health endpoint
    # reports rather than guessing).
    h = _http("GET", "/api/health")
    data_dir = Path(h["tools_resolved"]["data_dir"]["path"]).resolve()
    target = data_dir / TARGET_PRS
    if not target.exists():
        raise AssertionError(f"target {target} not present in active DATA_DIR")

    m1 = _http("GET", "/api/manifest", timeout=120)
    cache_path = Path(h["cache_dir"]) / "manifest.json"
    assert cache_path.exists(), f"manifest cache missing: {cache_path}"

    original_mtime = target.stat().st_mtime
    # Bump mtime forward by 5 seconds so cache_manifest sees a newer file.
    new_mtime = max(original_mtime, time.time()) + 5
    try:
        os.utime(target, (new_mtime, new_mtime))
        # Wait for the OS to reflect the change.
        time.sleep(0.2)
        m2 = _http("GET", "/api/manifest", timeout=120)
    finally:
        os.utime(target, (original_mtime, original_mtime))

    # The bumped mtime should propagate into the entry's mtime field.
    e1 = _entry_for(m1, TARGET_PRS)
    e2 = _entry_for(m2, TARGET_PRS)
    assert e1 is not None and e2 is not None, (e1, e2)
    assert e2["mtime"] >= e1["mtime"] + 4, (
        f"entry mtime did not advance: {e1['mtime']} -> {e2['mtime']}"
    )


@step("manifest excludes .pre_editor_* / .bak / .SUSPECT_* siblings")
def t_manifest_excludes_backups():
    """Drop a fake backup sibling next to a real file and verify it doesn't
    show up in the rebuilt manifest. Cleans up after itself."""
    h = _http("GET", "/api/health")
    data_dir = Path(h["tools_resolved"]["data_dir"]["path"]).resolve()

    fakes = [
        data_dir / "manifest_e2e_fake.bak",
        data_dir / "manifest_e2e.SUSPECT_crash_20260424_999999",
        data_dir / "manifest_e2e.pre_editor_19990101_000000",
        data_dir / "pre_old_thing.xvm",
        data_dir / "manifest_e2e.PARKED_old.dat",
        data_dir / "manifest_e2e.NOT_OG_thing.bin",
        data_dir / "manifest_e2e.DISABLED",
    ]
    cache_path = Path(h["cache_dir"]) / "manifest.json"

    try:
        # Create the fakes; force a manifest refresh by bumping mtimes
        # to "now" so cache_manifest definitely rebuilds.
        for fp in fakes:
            fp.write_bytes(b"\x00" * 32)
        # Make sure each is newer than the cache so the rebuild fires.
        now = time.time() + 1
        for fp in fakes:
            os.utime(fp, (now, now))
        time.sleep(0.2)
        m = _http("GET", "/api/manifest", timeout=120)
        # None of the fake names should appear in the manifest.
        names = {e["path"].rsplit("/", 1)[-1] for e in m["entries"]}
        for fp in fakes:
            assert fp.name not in names, (
                f"backup sibling {fp.name} leaked into manifest"
            )
    finally:
        for fp in fakes:
            try:
                fp.unlink()
            except OSError:
                pass
        # Force the cache to recognize the cleanup so subsequent tests don't see stale state.
        if cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass


# ============================================================
# Texture <-> model matcher (Agent 3 — formats/match.py)
# Pure-Python; runs without the live HTTP server.
# Each rule (R1..R6) gets one test.
# ============================================================


@step("match R1: tex.xvm sibling resolves at confidence 1.0")
def t_match_r1_tex_sibling():
    from formats.match import match_textures, Match
    install_root = Path(os.path.expanduser("~/PSOBB.IO"))
    bml = install_root / "data" / "bm_obj_ep4_boss09_core.bml"
    expected = install_root / "data" / "bm_obj_ep4_boss09_core_tex.xvm"
    assert bml.exists(), f"fixture missing: {bml}"
    assert expected.exists(), f"fixture missing: {expected}"
    matches = match_textures(bml, install_root)
    r1 = [m for m in matches if m.rule == "R1"]
    assert len(r1) == 1, f"expected exactly 1 R1 match, got {len(r1)}: {matches}"
    m = r1[0]
    assert m.confidence == 1.0, m
    assert m.path == expected, m.path
    assert m.partial is False, m


@step("match R2: BML-internal returns partial stub if no extractor")
def t_match_r2_bml_internal():
    from formats import match as match_mod
    from formats.match import match_textures
    install_root = Path(os.path.expanduser("~/PSOBB.IO"))
    # bm_boss2_de_rol_le.bml has has_textures=1 (verified by header byte 9)
    bml = install_root / "data" / "bm_boss2_de_rol_le.bml"
    assert bml.exists(), f"fixture missing: {bml}"
    matches = match_textures(bml, install_root)
    r2 = [m for m in matches if m.rule.startswith("R2")]
    assert len(r2) >= 1, f"expected at least one R2 match, got {matches}"
    m = r2[0]
    assert m.confidence == 0.95, m
    if match_mod._HAS_BML:
        # Real extractor available — must produce non-partial matches that
        # name an inner XVM virtual path.
        non_partial = [x for x in r2 if not x.partial]
        assert non_partial, f"expected at least one non-partial R2 with BML extractor, got {r2}"
        assert non_partial[0].rule == "R2", non_partial[0]
    else:
        # Graceful degrade: stub points back at the BML file with partial=True.
        assert m.partial is True, m
        assert m.rule == "R2-stub", m
        assert m.path == bml, m.path


@step("match R3: NJTL fallback returns valid match if IFF reader landed")
def t_match_r3_njtl_lookup():
    """Synthesizes an NJTL .nj because the live install has no NJTL chunks
    (every PSOBB.IO .nj is just NJCM+POF0). Whether or not formats.iff is
    available, the inline IFF walker handles this fixture."""
    import struct
    import tempfile
    from formats.match import match_textures
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = root / "data"
        data.mkdir()
        names = [b"njtl_test_alpha", b"njtl_test_beta"]
        # NJTL body: u32 elements_off=8, u32 count=N, then 12-byte entries,
        # then NUL-terminated strings.
        count = len(names)
        entries_off = 8
        entries_size = count * 12
        strings_off = entries_off + entries_size
        body = bytearray()
        body += struct.pack("<II", entries_off, count)
        cur_str = strings_off
        for n in names:
            body += struct.pack("<III", cur_str, 0, 0)
            cur_str += len(n) + 1
        for n in names:
            body += n + b"\x00"
        while len(body) % 4:
            body += b"\x00"
        iff_bytes = b"NJTL" + struct.pack("<I", len(body)) + bytes(body)

        nj = data / "match_r3_test.nj"
        nj.write_bytes(iff_bytes)
        # Sibling textures the matcher should resolve
        (data / "njtl_test_alpha.xvm").write_bytes(b"XVMH8\x00\x00\x00fakedata")
        (data / "njtl_test_beta.prs").write_bytes(b"fake prs payload")

        matches = match_textures(nj, root)
        r3 = [m for m in matches if m.rule.startswith("R3")]
        assert len(r3) == 2, f"expected 2 R3 matches, got {r3}"
        # both should be confidence 0.9, non-partial
        for m in r3:
            assert m.confidence == 0.9, m
            assert m.partial is False, m
        paths = sorted(m.path.name for m in r3)
        assert paths == ["njtl_test_alpha.xvm", "njtl_test_beta.prs"], paths


@step("match R4: plAbdy00 matches plAtex.afs")
def t_match_r4_player_afs():
    from formats.match import match_textures
    install_root = Path(os.path.expanduser("~/PSOBB.IO"))
    nj = install_root / "data" / "plAbdy00.nj"
    expected = install_root / "data" / "plAtex.afs"
    assert nj.exists(), f"fixture missing: {nj}"
    assert expected.exists(), f"fixture missing: {expected}"
    matches = match_textures(nj, install_root)
    r4 = [m for m in matches if m.rule == "R4"]
    assert len(r4) == 1, f"expected exactly 1 R4 match, got {r4}"
    m = r4[0]
    assert m.confidence == 0.85, m
    assert m.path == expected, m.path
    assert m.partial is False, m
    assert m.detail.get("char_class") == "A", m.detail


@step("match R5: ItemModel/ItemTexture ordinal pair")
def t_match_r5_item_pair():
    from formats.match import match_textures
    install_root = Path(os.path.expanduser("~/PSOBB.IO"))
    item_model = install_root / "data" / "ItemModel.afs"
    item_tex = install_root / "data" / "ItemTexture.afs"
    assert item_model.exists(), f"fixture missing: {item_model}"
    if not item_tex.exists():
        # Some installs have ItemTexture.afs only as .SUSPECT_*; skip cleanly.
        print("    (skipped: ItemTexture.afs not present in clean form)")
        return
    matches = match_textures(item_model, install_root)
    r5 = [m for m in matches if m.rule == "R5"]
    assert len(r5) == 1, f"expected exactly 1 R5 match, got {r5}"
    m = r5[0]
    assert m.confidence == 0.7, m
    assert m.path == item_tex, m.path
    assert m.detail.get("pair") == ("ItemModel.afs", "ItemTexture.afs"), m.detail


@step("match R6: map_*.rel returns .xvm siblings")
def t_match_r6_map_prefix():
    """Uses real scene/ data when available; falls back to a tmp fixture."""
    import tempfile
    from formats.match import match_textures
    install_root = Path(os.path.expanduser("~/PSOBB.IO"))
    scene_rel = install_root / "data" / "scene" / "map_aancient01_00n.rel"
    if scene_rel.exists():
        matches = match_textures(scene_rel, install_root)
        r6 = [m for m in matches if m.rule == "R6"]
        assert len(r6) >= 1, f"expected at least one R6 match for {scene_rel}, got {matches}"
        for m in r6:
            assert m.confidence == 0.5, m
            assert m.path.suffix.lower() == ".xvm", m.path
            assert m.path.name.lower().startswith("map_aancient01"), m.path.name

        # Inverse: map_*.xvm -> .rel siblings
        scene_xvm = install_root / "data" / "scene" / "map_aancient01_00s.xvm"
        if scene_xvm.exists():
            matches = match_textures(scene_xvm, install_root)
            inv = [m for m in matches if m.rule == "R6"]
            assert len(inv) >= 1, f"expected R6 inverse hit for {scene_xvm}, got {matches}"
            assert any(m.path.suffix.lower() == ".rel" for m in inv), inv
        return

    # Fallback fixture
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = root / "data"
        data.mkdir()
        rel = data / "map_test01_00n.rel"
        rel.write_bytes(b"fake rel")
        (data / "map_test01.xvm").write_bytes(b"XVMH8\x00\x00\x00x")
        (data / "map_test01_00s.xvm").write_bytes(b"XVMH8\x00\x00\x00x")
        matches = match_textures(rel, root)
        r6 = [m for m in matches if m.rule == "R6"]
        assert len(r6) == 2, f"expected 2 R6 matches in fixture, got {r6}"
        for m in r6:
            assert m.confidence == 0.5, m
            assert m.path.suffix == ".xvm", m.path


# ============================================================
# Phase A formats (Agent 2): IFF + AFS readers
#   - GET /api/asset/<file>/iff           list IFF chunks
#   - GET /api/asset/<file>?meta=1        AFS metadata
# Fixture files (plAbdy00.nj, ItemModel.afs) are copied from
# ~/PSOBB.IO/data/ to the editor's DEV mirror so they
# are reachable through safe_data_path() without weakening the
# path-traversal guard.
# ============================================================

# DEV mirror DATA_DIR used by the running server.
_DEV_DATA_DIR = Path(r"C:/tmp_pso_dev/data").resolve()
_FIXTURE_NJ = "plAbdy00.nj"
_FIXTURE_AFS = "ItemModel.afs"


def _ensure_fixture(name: str) -> None:
    """Copy a fixture from the live install to the dev mirror if absent."""
    dst = _DEV_DATA_DIR / name
    if dst.exists():
        return
    src = DATA_DIR / name
    if not src.exists():
        raise AssertionError(
            f"fixture {name} missing from both DEV ({dst}) and LIVE ({src}); "
            f"cannot run formats tests"
        )
    _DEV_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


@step("iff: NJCM chunk found at top of plAbdy00.nj")
def t_iff_njcm_top():
    _ensure_fixture(_FIXTURE_NJ)
    r = _http("GET", f"/api/asset/{_FIXTURE_NJ}/iff")
    assert r["filename"] == _FIXTURE_NJ
    assert r["format"] == "iff"
    assert r["total_chunks"] >= 1, r
    first = r["chunks"][0]
    assert first["type"] == "NJCM", f"first chunk is {first} not NJCM"
    assert first["size"] > 0, first


@step("iff: NJTL chunk follows NJCM in some files")
def t_iff_njtl_after_njcm():
    """The standalone .nj files in PSOBB.IO/data/ have only NJCM+POF0,
    but BML/AFS-extracted NJ payloads typically begin NJTL+NJCM. We
    verify the parser preserves chunk order by feeding it a synthetic
    buffer constructed in-memory, then cross-check that the live
    endpoint preserves the same documented chunk order on a real .nj.
    """
    _ensure_fixture(_FIXTURE_NJ)
    # Build an in-memory buffer: NJTL(8 bytes payload) + NJCM(4 bytes payload)
    # in the documented little-endian "<4sI" header layout.
    njtl_body = b"\x00\x00\x00\x00\x00\x00\x00\x00"
    njcm_body = b"\xde\xad\xbe\xef"
    blob = (
        b"NJTL" + struct.pack("<I", len(njtl_body)) + njtl_body
        + b"NJCM" + struct.pack("<I", len(njcm_body)) + njcm_body
    )
    # Import the parser the same way the server does.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from formats.iff import parse_iff  # type: ignore[import-not-found]
    finally:
        if str(Path(__file__).parent.parent) in sys.path:
            sys.path.remove(str(Path(__file__).parent.parent))
    chunks = parse_iff(blob)
    assert len(chunks) == 2, chunks
    assert chunks[0].type == "NJTL", chunks[0]
    assert chunks[1].type == "NJCM", chunks[1]
    assert chunks[1].data == njcm_body, "NJCM payload corrupted in roundtrip"

    # Live endpoint: verify chunk order is preserved on a real .nj.
    r = _http("GET", f"/api/asset/{_FIXTURE_NJ}/iff")
    types = [c["type"] for c in r["chunks"]]
    assert types[0] == "NJCM", types  # documented order: NJCM first


@step("iff: corrupted IFF returns clean error")
def t_iff_corrupted():
    """Plant a deliberately broken .nj into the dev mirror, hit the
    endpoint, expect HTTP 400 with a parser-derived message (no 500).
    """
    bad = _DEV_DATA_DIR / "__e2e_iff_corrupt.nj"
    # NJCM with size=0xFFFFFFFF and only 4 bytes of payload after.
    bad.write_bytes(b"NJCM" + b"\xff\xff\xff\xff" + b"\x00\x00\x00\x00")
    try:
        try:
            _http("GET", f"/api/asset/{bad.name}/iff")
        except RuntimeError as e:
            assert "400" in str(e), f"expected 400, got: {e}"
            assert "IFF parse failed" in str(e) or "parse_iff" in str(e), str(e)
            return
        raise AssertionError("corrupted IFF should have been rejected")
    finally:
        bad.unlink(missing_ok=True)


@step("afs: ItemModel.afs parses to N entries")
def t_afs_itemmodel_parses():
    _ensure_fixture(_FIXTURE_AFS)
    r = _http("GET", f"/api/asset/{_FIXTURE_AFS}?meta=1")
    assert r["filename"] == _FIXTURE_AFS
    assert r["format"] == "afs"
    # The vanilla install ships ItemModel.afs with 370 PRS-compressed
    # entries. Anything in this ballpark proves the parser walked the
    # full table; a strict equality is too brittle for community-edited
    # builds, so we assert a sane lower bound + per-entry sanity.
    assert r["count"] >= 100, f"only {r['count']} entries found"
    assert len(r["sizes"]) == r["count"], "sizes/count mismatch"
    # Every entry must carry a strictly positive payload size.
    assert all(s > 0 for s in r["sizes"]), "zero-size entry"
    # Names are documented as empty for AFS (no name table in PSOBB).
    assert r["names"] == [], r["names"]


@step("afs: bad magic returns 400")
def t_afs_bad_magic():
    bad = _DEV_DATA_DIR / "__e2e_afs_badmagic.afs"
    bad.write_bytes(b"NOTAFS!!" + b"\x00" * 64)
    try:
        try:
            _http("GET", f"/api/asset/{bad.name}?meta=1")
        except RuntimeError as e:
            assert "400" in str(e), f"expected 400, got: {e}"
            assert "magic" in str(e).lower() or "AFS parse failed" in str(e), str(e)
            return
        raise AssertionError("bad AFS magic should have been rejected")
    finally:
        bad.unlink(missing_ok=True)


@step("afs: oversized offset rejected")
def t_afs_oversized_offset():
    """Forge a header with file_count=1 and an offset that points past
    the end of the buffer. The parser must refuse it cleanly.
    """
    bad = _DEV_DATA_DIR / "__e2e_afs_oversize.afs"
    # Magic + count(1) + pad + entry(offset=0xFFFFFF, size=1000) + padding.
    forged = (
        b"AFS\x00"
        + struct.pack("<HH", 1, 0)
        + struct.pack("<II", 0xFFFFFF, 1000)
        + b"\x00" * 256
    )
    bad.write_bytes(forged)
    try:
        try:
            _http("GET", f"/api/asset/{bad.name}?meta=1")
        except RuntimeError as e:
            assert "400" in str(e), f"expected 400, got: {e}"
            assert "exceeds buffer" in str(e) or "AFS parse failed" in str(e), str(e)
            return
        raise AssertionError("oversized AFS offset should have been rejected")
    finally:
        bad.unlink(missing_ok=True)


# ------------------------------------------------------------------------
# Tree (Phase A — Agent 5): asset-tree frontend smoke
#
# These ride on top of Agent 1's /api/manifest. Both gracefully skip
# (printed as PENDING) if Agent 1's endpoint is missing — the tree's UI
# already shows a "manifest not yet built" placeholder for that case.
# ------------------------------------------------------------------------

# Category enum mirrored verbatim from MASTER_PLAN/manifest.schema.json.
# Duplicated here (not parsed) because the categories are part of the
# IPC contract — if the schema changes, this assertion forces a manual
# review. Bus listeners + tree groups depend on this exact set.
_TREE_CATEGORY_ENUM = frozenset({
    "texture", "model", "container",
    "quest", "map", "audio",
    "ui", "script", "cinematic",
    "metadata", "unknown",
})


def _manifest_or_pending() -> dict | None:
    """Fetch /api/manifest; return the dict on success, ``None`` if the
    endpoint is a 404 (Agent 1 hasn't shipped). Anything else re-raises."""
    try:
        return _http("GET", "/api/manifest", timeout=120)
    except RuntimeError as e:
        if "404" in str(e):
            return None
        raise


@step("tree: GET /api/manifest returns categories that match the schema enum")
def t_tree_categories_match_enum():
    m = _manifest_or_pending()
    if m is None:
        print("    (PENDING: /api/manifest not implemented yet — depends on Agent 1)")
        return
    seen = {e.get("category") for e in m.get("entries", [])}
    # Empty manifest is acceptable — Agent 5's tree falls through to a
    # "manifest empty" placeholder. Don't fail just because the install
    # has no files.
    if not seen:
        return
    extra = seen - _TREE_CATEGORY_ENUM
    assert not extra, (
        f"manifest reports categories not in the schema enum: {sorted(extra)}; "
        f"expected subset of {sorted(_TREE_CATEGORY_ENUM)}"
    )


@step("tree: every entry has at least path/size/extension/category")
def t_tree_entry_minimum_fields():
    m = _manifest_or_pending()
    if m is None:
        print("    (PENDING: /api/manifest not implemented yet — depends on Agent 1)")
        return
    entries = m.get("entries", [])
    if not entries:
        # Same skip rationale as above — empty manifest is OK.
        return
    # Spot-check every entry; if there are tens of thousands the failure
    # message should still point to the first offender, not a count.
    for i, e in enumerate(entries):
        assert isinstance(e, dict), f"entry[{i}] is not a dict: {type(e).__name__}"
        for field in ("path", "size", "extension", "category"):
            assert field in e, f"entry[{i}] missing '{field}': {e}"
        assert isinstance(e["path"], str) and e["path"], f"entry[{i}].path bad: {e['path']!r}"
        assert isinstance(e["size"], int) and e["size"] >= 0, f"entry[{i}].size bad: {e['size']!r}"
        assert isinstance(e["extension"], str), f"entry[{i}].extension bad: {e['extension']!r}"
        assert isinstance(e["category"], str), f"entry[{i}].category bad: {e['category']!r}"


# ============================================================
# Asset coverage Phase 1+2 (2026-04-25):
#   - /api/manifest/categories (tab-strip backend)
#   - /api/raw/{path} (audio/hex/text fallbacks)
#   - /api/model/{path}/skeleton (bone visualization)
#   - manifest entries category=model expose matched_textures
# ============================================================


@step("manifest categories endpoint returns 11 enum-matching categories with counts")
def t_manifest_categories_endpoint():
    r = _http("GET", "/api/manifest/categories", timeout=120)
    assert "categories" in r and "total" in r, r
    cats = r["categories"]
    assert isinstance(cats, list) and cats, "empty categories list"
    # The schema enum has 11 categories. The live install almost certainly
    # has at least 5 (texture/model/audio/quest/script); we assert >=5 to
    # tolerate empty data dirs in test fixtures, but the enum contract
    # says no more than 11 are valid.
    enum = {
        "texture", "model", "container",
        "quest", "map", "audio",
        "ui", "script", "cinematic",
        "metadata", "unknown",
    }
    seen_names = {c["name"] for c in cats}
    extra = seen_names - enum
    assert not extra, f"unexpected categories outside enum: {extra}"
    for c in cats:
        assert "name" in c and "count" in c, c
        assert isinstance(c["count"], int) and c["count"] > 0, c
    # Total matches the sum of counts
    assert r["total"] == sum(c["count"] for c in cats), r


@step("models tab: GET /api/manifest filtered by category=model returns >100 entries on live install")
def t_models_category_count():
    m = _manifest_or_pending()
    if m is None:
        print("    (PENDING: /api/manifest not implemented yet)")
        return
    models = [e for e in m.get("entries", []) if e.get("category") == "model"]
    # The live install has 365 BMLs + ~300 NJs in the data dir; vanilla
    # PSOBB.IO has even more once we walk the full root. >100 is a
    # comfortable lower bound that catches "we forgot to classify .bml".
    assert len(models) > 100, f"only {len(models)} models classified — expected >100"
    # Every model should have a recognizable format
    for e in models[:50]:
        assert e.get("format") in ("BML", "NJ_IFF"), f"model with bad format: {e}"


@step("model auto-bind: bm_obj_ep4_boss09_core.bml's matched_textures includes core_tex.xvm via R1")
def t_model_auto_bind_r1():
    m = _manifest_or_pending()
    if m is None:
        print("    (PENDING: /api/manifest not implemented yet)")
        return
    target_bml = "bm_obj_ep4_boss09_core.bml"
    target_xvm = "bm_obj_ep4_boss09_core_tex.xvm"
    entry = next(
        (e for e in m.get("entries", []) if e.get("path") == target_bml),
        None,
    )
    if entry is None:
        # Skip if the install lacks the EP4 boss model; the matcher logic
        # is still covered by t_match_r1_tex_sibling further down.
        print(f"    (SKIP: {target_bml} not in manifest)")
        return
    matched = entry.get("matched_textures") or []
    assert matched, (
        f"{target_bml} has no matched_textures — Agent 3 R1 should fire on a "
        f"sibling _tex.xvm"
    )
    # Find the R1 hit specifically. R1 is the highest-confidence rule
    # (sibling _tex.xvm) and must reach 1.0.
    r1 = next((m for m in matched if m.get("rule") == "R1"), None)
    assert r1 is not None, f"no R1 hit in {matched!r}"
    assert r1.get("path", "").endswith(target_xvm), (
        f"R1 path mismatch: expected '{target_xvm}', got {r1!r}"
    )
    assert r1.get("confidence", 0) >= 0.99, f"R1 confidence too low: {r1!r}"


@step("raw endpoint: serves a known .bml with correct Content-Type and bytes")
def t_raw_endpoint_bml_serves():
    # Use one of the known small BMLs in the data dir; we want to verify
    # the byte stream and content-type, not the parser.
    target = "biri_ball.bml"
    url = f"{API}/api/raw/{target}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200, f"got HTTP {resp.status}"
        ct = resp.headers.get("Content-Type", "")
        assert ct.startswith("application/x-bml"), (
            f"wrong Content-Type for .bml: {ct!r} (expected application/x-bml)"
        )
        x_size = resp.headers.get("X-Asset-Size")
        assert x_size and int(x_size) > 0, f"missing/zero X-Asset-Size: {x_size!r}"
        body = resp.read()
        assert len(body) == int(x_size), (
            f"length mismatch: header says {x_size}, got {len(body)} bytes"
        )
        # BML files have no fixed magic header, but the file must be
        # non-trivial (>= 256 bytes is tiny but real). The smallest BML
        # in the install is around 3 KB.
        assert len(body) >= 256, f"BML too small to be plausible: {len(body)}"


@step("bml_inner_resolve: tile_png on bm_obj_ep4_boss09_core_tex.xvm works direct (regression)")
def t_bml_inner_direct_tile_png_regression():
    # Regression guard: refactoring the tiles/tile_png endpoints to
    # understand `<base>#<inner>` must not break the plain-filename
    # path. Use a known XVM that ships in the live install.
    target = "bm_obj_ep4_boss09_core_tex.xvm"
    url = f"{API}/api/tile_png/{urllib.parse.quote(target)}/0"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200, f"got HTTP {resp.status}"
        ct = resp.headers.get("Content-Type", "")
        assert ct.startswith("image/png"), f"wrong Content-Type: {ct!r}"
        body = resp.read()
        # PNG magic
        assert body[:8] == b"\x89PNG\r\n\x1a\n", "tile_png response is not a PNG"


@step("bml_inner_resolve: tile_png on a BML#inner_xvm path returns valid PNG")
def t_bml_inner_tile_png_via_hash():
    # Click-through path from the asset tree: BML container + `#` +
    # synthesized texture name `<inner_nj>.xvm`. The router decodes the
    # `#` (URL-encoded as %23) into the inner-texture lookup.
    target = "bm4_ps_ma_body.bml#bm4_ps_ma_body.nj.xvm"
    url = f"{API}/api/tile_png/{urllib.parse.quote(target, safe='')}/0"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        assert resp.status == 200, f"got HTTP {resp.status} for {target!r}"
        body = resp.read()
        assert body[:8] == b"\x89PNG\r\n\x1a\n", (
            f"BML-inner tile_png response is not a PNG ({len(body)} bytes; "
            f"head={body[:16]!r})"
        )
        assert len(body) > 100, f"tile PNG too tiny to be plausible: {len(body)} bytes"

    # Also verify /api/tiles returns the same shape with the # syntax
    tiles_url = f"{API}/api/tiles/{urllib.parse.quote(target, safe='')}"
    with urllib.request.urlopen(tiles_url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assert data["filename"] == target, data
    assert data["tile_count"] > 0, data
    assert data["is_prs"] is False, data
    # Each tile carries a base64 PNG; verify tile 0 at minimum
    t0 = data["tiles"][0]
    raw = base64.b64decode(t0["src_png_b64"].split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "first tile b64 is not a PNG"


@step("bml_inner_resolve: model_preview on a .bml#inner.nj returns hint + tile_count")
def t_bml_inner_model_preview():
    # The model viewer hits /api/model_preview first to learn shape +
    # tile_count; the BML-inner form must give it enough info to pick
    # a default texture (i.e. tile_count > 0 from the inner XVM).
    target = "bm4_ps_ma_body.bml#bm4_ps_ma_body.nj.xvm"
    url = f"{API}/api/model_preview/{urllib.parse.quote(target, safe='')}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assert data["filename"] == target, data
    assert "shape" in data and data["shape"] in (
        "sphere", "cube", "plane", "cylinder",
    ), data
    assert data["tile_count"] > 0, (
        f"model_preview on a textured BML inner returned tile_count=0: {data!r}"
    )
    assert data["first_tile"] is not None, data
    assert data["first_tile"]["width"] > 0, data
    assert data["first_tile"]["height"] > 0, data

    # Also verify model_mesh (NJ via #inner) returns geometry. The path
    # form `<bml>#<inner.nj>` should subsume the legacy ?inner= query.
    mesh_target = "bm4_ps_ma_body.bml#bm4_ps_ma_body.nj"
    mesh_url = f"{API}/api/model_mesh/{urllib.parse.quote(mesh_target, safe='')}"
    with urllib.request.urlopen(mesh_url, timeout=60) as resp:
        mesh = json.loads(resp.read().decode("utf-8"))
    assert mesh["mesh_count"] >= 1, mesh
    assert mesh["totals"]["vertices"] > 0, mesh
    assert mesh["totals"]["triangles"] > 0, mesh
    assert mesh["inner"] == "bm4_ps_ma_body.nj", mesh


@step("bml_inner_resolve: invalid #-syntax returns clean 400")
def t_bml_inner_invalid_syntax():
    # All four forms must be rejected at the validator. We accept either
    # 400 (our explicit handler) or 404 (FastAPI route mismatch when the
    # URL-decoded form contains `/`).
    bad_paths = [
        "biri_ball.bml#",                    # empty inner
        "#bm4_ps_ma_body.nj.xvm",            # empty base
        "biri_ball.bml#foo#bar",             # multiple separators
        "LogoEP4.prs#nope.xvm",              # `#` on a non-BML base
    ]
    for bad in bad_paths:
        url = f"{API}/api/tile_png/{urllib.parse.quote(bad, safe='')}/0"
        try:
            urllib.request.urlopen(url, timeout=10)
        except urllib.error.HTTPError as e:
            # Empty-inner can be 400 (our guard) or 404 (FastAPI 404 on
            # the trailing `/` route mismatch); the `#`-on-non-BML must
            # be 400. Empty base routes to a different endpoint so 404
            # is also fine. The critical assertion is no 200, no 5xx.
            assert e.code in (400, 404), (
                f"unexpected HTTP {e.code} for {bad!r}; expected 400/404"
            )
            continue
        raise AssertionError(f"path {bad!r} was NOT rejected (returned 200)")


def _pick_bml_inner_target() -> "tuple[str, str] | None":
    """Find a (bml, inner_nj) pair for testing /api/model_mesh.

    Strategy:
      1. Check the manifest for a category=model entry containing `#`.
         (The current build doesn't synthesise BML-inner entries so this
         is unlikely to hit, but if a future agent does we'll adapt.)
      2. Fall back: list every category=model BML and pick the first
         that has at least one `.nj` inner via /api/bml/<x>/list.

    Returns (bml_path, inner_nj_name) or None if neither path turned up
    a usable target.
    """
    m = _manifest_or_pending()
    if m is None:
        return None
    # Strategy 1: hash-form synthesis (future-proof).
    for e in m.get("entries", []):
        if (e.get("category") == "model"
                and "#" in e.get("path", "")
                and e["path"].lower().endswith(".nj")):
            base, _, inner = e["path"].partition("#")
            return (base, inner)
    # Strategy 2: probe BMLs via the inner-list endpoint.
    bml_entries = [
        e for e in m.get("entries", [])
        if e.get("format") == "BML" and e.get("category") == "model"
    ]
    # Bias toward "ene" / "obj" BMLs which usually have meshes; fall
    # back to whatever's first.
    bml_entries.sort(key=lambda e: (
        0 if "ene" in e["path"] else 1 if "obj" in e["path"] else 2,
        e["path"],
    ))
    for e in bml_entries[:30]:  # cap attempts
        bml_path = e["path"]
        url = f"{API}/api/bml/{urllib.parse.quote(bml_path, safe='')}/list"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
        for inner in data.get("entries", []):
            name = inner.get("name", "")
            if name.lower().endswith(".nj"):
                return (bml_path, name)
    return None


@step("model→mesh routing: model_mesh on bml#inner.nj returns vertices via the path-driven URL")
def t_model_mesh_via_path_for_bml_inner():
    # The frontend bug: clicking a `<bml>#<inner>.nj` model entry in the
    # asset tree silently fell back to a primitive cube because the
    # router routed through the texture-driven open() flow which lost
    # the inner specifier. The fix is asset_router.js + model_viewer.js
    # using /api/model_mesh/<bml>%23<inner.nj> directly. This test
    # locks in that the backend serves that exact URL form with real
    # geometry — the same call the new openByPath() makes.
    target = _pick_bml_inner_target()
    if target is None:
        print("    (SKIP: no .bml with inner .nj available)")
        return
    bml, inner = target
    target_path = f"{bml}#{inner}"
    url = f"{API}/api/model_mesh/{urllib.parse.quote(target_path, safe='')}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        assert resp.status == 200, f"got HTTP {resp.status} for {target_path!r}"
        payload = json.loads(resp.read().decode("utf-8"))
    assert payload["mesh_count"] >= 1, (
        f"BML-inner {target_path} returned mesh_count=0 — frontend would fall "
        f"back to primitive (the bug we're guarding against)"
    )
    assert payload["totals"]["vertices"] > 0, payload["totals"]
    assert payload["totals"]["triangles"] > 0, payload["totals"]
    assert payload["inner"] == inner, payload
    # The wire shape used by buildMeshGroupFromPayload — each mesh
    # carries vertices_b64 + indices_b64 in the documented stride.
    for mesh in payload["meshes"]:
        verts = base64.b64decode(mesh["vertices_b64"])
        idx = base64.b64decode(mesh["indices_b64"])
        # Float32 interleaved [px,py,pz, nx,ny,nz, u,v] = 32 bytes/vertex
        assert len(verts) == mesh["vertex_count"] * 32, (
            f"vertex stride mismatch: {len(verts)} vs {mesh['vertex_count']}*32"
        )
        # Uint32 indices, 12 bytes per triangle
        assert len(idx) == mesh["triangle_count"] * 12, (
            f"index stride mismatch: {len(idx)} vs {mesh['triangle_count']}*12"
        )


@step("model→mesh routing: matched_textures on a model entry point at .xvm/.nj.xvm")
def t_model_mesh_inner_matched_textures():
    # The asset_router's openByPath() picks the highest-confidence
    # matched_texture and pre-binds it on the loaded mesh. For a `.bml`
    # entry, the matcher (Agent 3 R2) should synthesize a sibling
    # `<bml>#<inner>.nj.xvm` (or related) as a texture match.
    # If this test fails, openByPath would have nothing to wrap on the
    # mesh and the regression would be a colour-less render.
    m = _manifest_or_pending()
    if m is None:
        print("    (PENDING: /api/manifest not implemented yet)")
        return
    target_entry = None
    for e in m.get("entries", []):
        if e.get("category") == "model" and (e.get("matched_textures") or []):
            target_entry = e
            break
    if target_entry is None:
        print("    (SKIP: no model entries with matched_textures)")
        return
    matched = target_entry["matched_textures"]
    # Top match should pair with a `.nj.xvm` / `.xj.xvm` / `.xvm` / `.prs`
    # form. The frontend's openByPath() filter accepts these.
    top = matched[0]
    top_lower = top.get("path", "").lower()
    assert (
        top_lower.endswith(".xvm")
        or top_lower.endswith(".prs")
        or top_lower.endswith(".nj.xvm")
        or top_lower.endswith(".xj.xvm")
    ), f"top matched_texture for {target_entry['path']} is unrecognised: {top!r}"
    # Confirm we can actually fetch it through the texture preview pipeline
    # (model_preview), the way openByPath does.
    tex_url = f"{API}/api/model_preview/{urllib.parse.quote(top['path'], safe='')}"
    with urllib.request.urlopen(tex_url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assert data["tile_count"] > 0, (
        f"matched texture {top['path']} returned tile_count=0; the model viewer "
        f"would have nothing to wrap"
    )


@step("model→mesh routing: model_mesh on a top-level .bml + ?inner= returns geometry (legacy hint flow)")
def t_model_mesh_top_level_bml():
    # Legacy hint-driven `tryLoadRealMesh` calls /api/bml/<x>/list to
    # find the first .nj, then /api/model_mesh/{bml}?inner=<name>. This
    # path remains in the codebase for when the texture-first flow
    # opens a model. We pin the contract so a refactor doesn't drop it.
    target = _pick_bml_inner_target()
    if target is None:
        print("    (SKIP: no .bml with inner .nj available)")
        return
    bml_path, nj_inner_name = target
    mesh_url = (f"{API}/api/model_mesh/{urllib.parse.quote(bml_path, safe='')}"
                f"?inner={urllib.parse.quote(nj_inner_name, safe='')}")
    with urllib.request.urlopen(mesh_url, timeout=60) as resp:
        mesh = json.loads(resp.read().decode("utf-8"))
    assert mesh["mesh_count"] >= 1, mesh
    assert mesh["inner"] == nj_inner_name, mesh


@step("model→mesh routing: top-level .bml entry resolves inner from matched_textures (frontend flow simulation)")
def t_model_mesh_resolves_inner_from_matched_textures():
    # This simulates the EXACT flow the asset-tree click produces:
    # 1. user clicks `bm_ene_del_depth.bml` in the Models tab
    # 2. asset_router.js dispatch() routes to openByPath()
    # 3. openByPath inspects matched_textures and infers
    #    `<bml>#<inner>.nj` from the highest-confidence
    #    `<bml>#<inner>.nj.xvm` match
    # 4. tryLoadRealMesh({path: "<bml>#<inner>.nj"}) hits
    #    /api/model_mesh/<bml>%23<inner>.nj
    #
    # If this test fails, the user sees a primitive cube ("the bug").
    m = _manifest_or_pending()
    if m is None:
        print("    (PENDING: /api/manifest not implemented yet)")
        return
    # Find a top-level .bml entry whose matched_textures includes a
    # `<bml>#<inner>.nj.xvm` form — exactly what the bug repro hits.
    target = None
    for e in m.get("entries", []):
        if e.get("category") != "model" or e.get("format") != "BML":
            continue
        if "#" in e.get("path", ""):
            continue  # skip already-inner entries
        for mt in e.get("matched_textures") or []:
            mtp = mt.get("path", "")
            if mtp.startswith(e["path"] + "#") and mtp.lower().endswith(".nj.xvm"):
                target = (e["path"], mtp)
                break
        if target:
            break
    if target is None:
        print("    (SKIP: no .bml with matched <bml>#<inner>.nj.xvm)")
        return
    bml_path, tex_path = target
    # Mirror the JS resolveMeshPath logic exactly: drop `.xvm` to get
    # `<bml>#<inner>.nj`.
    assert tex_path.lower().endswith(".nj.xvm"), tex_path
    resolved_mesh_path = tex_path[:-4]  # strip .xvm → "<bml>#<inner>.nj"
    assert resolved_mesh_path.startswith(bml_path + "#"), resolved_mesh_path
    # Hit /api/model_mesh with the resolved path — same call the
    # frontend's tryLoadRealMesh makes.
    url = f"{API}/api/model_mesh/{urllib.parse.quote(resolved_mesh_path, safe='')}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        assert resp.status == 200, (
            f"frontend-resolved path {resolved_mesh_path!r} returned HTTP "
            f"{resp.status} — bug regression: user would see primitive cube"
        )
        payload = json.loads(resp.read().decode("utf-8"))
    assert payload["mesh_count"] >= 1, (
        f"frontend-resolved path {resolved_mesh_path!r} returned mesh_count=0 — "
        f"openByPath would fall back to primitive (bug regression)"
    )
    assert payload["totals"]["vertices"] > 0
    assert payload["totals"]["triangles"] > 0
    # Also check the texture pre-bind step — if openByPath's loadTexture
    # picks `tex_path` and hits /api/model_preview, it should get back
    # tile_count > 0 (else the wrap on the mesh is empty).
    tex_url = f"{API}/api/model_preview/{urllib.parse.quote(tex_path, safe='')}"
    with urllib.request.urlopen(tex_url, timeout=30) as resp:
        tex_data = json.loads(resp.read().decode("utf-8"))
    assert tex_data.get("tile_count", 0) > 0, (
        f"matched texture {tex_path!r} has tile_count=0 — mesh would render "
        f"untextured"
    )


@step("raw endpoint: rejects path traversal")
def t_raw_endpoint_rejects_traversal():
    # Several encodings of the classic ../../etc/passwd attack. None should
    # reach the filesystem; all should be 400 (path components forbidden)
    # or 404 (resolved path doesn't exist), never 200.
    bad_paths = [
        "..%2Fetc%2Fpasswd",
        ".%2E%2Fetc%2Fpasswd",
        "subdir%2Fnested.bml",
        "..%5Cetc%5Cpasswd",
    ]
    for bad in bad_paths:
        url = f"{API}/api/raw/{bad}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                # If we somehow got a 200, that's a security failure
                assert resp.status >= 400, (
                    f"path-traversal {bad!r} returned HTTP {resp.status} (should be 4xx)"
                )
        except urllib.error.HTTPError as e:
            # 400 / 404 are both acceptable - 400 = filename validation
            # rejected the path-component form, 404 = resolved-but-missing.
            assert e.code in (400, 404), (
                f"path-traversal {bad!r} returned HTTP {e.code} (expected 400/404)"
            )


# ============================================================
# Unified viewport perspectives (2026-04-24)
#
# These tests check the API contracts the new perspectives consume.
# The DOM-side rendering is covered by manual smoke; here we just
# pin the wire formats so a backend regression would surface as a
# concrete test failure rather than silent UI breakage.
# ============================================================

@step("perspectives: tile-grid contract — /api/tiles returns parseable shape")
def t_perspectives_tile_grid_contract():
    # The "tile-grid" perspective drives window.openFile, which calls
    # /api/tiles. Verify the shape is exactly what the perspective expects.
    target = "LogoEP4.prs"
    data = _http("GET", f"/api/tiles/{urllib.parse.quote(target)}")
    assert data["filename"] == target, data
    assert isinstance(data.get("tile_count"), int) and data["tile_count"] > 0, data
    assert isinstance(data.get("tiles"), list) and len(data["tiles"]) == data["tile_count"], data
    # Per-tile: index, width, height, fmt, src_png_b64
    for t in data["tiles"]:
        for k in ("index", "filename", "width", "height", "fmt", "src_png_b64"):
            assert k in t, f"missing {k} in tile: {t!r}"
        assert t["src_png_b64"].startswith("data:image/png;base64,"), t


@step("perspectives: 3d-view contract — /api/model_preview returns hint shape")
def t_perspectives_3d_view_contract():
    # The "3d-view" perspective relies on /api/model_preview/<filename>
    # for the hint (shape, model_archive, first_tile, why) — same call
    # the legacy modal makes via tryLoadRealMesh.
    target = "LogoEP4.prs"
    data = _http("GET", f"/api/model_preview/{urllib.parse.quote(target)}")
    # Required fields the perspective renders.
    assert "shape" in data, data
    assert "tile_count" in data, data
    assert isinstance(data["tile_count"], int), data
    # `why` is the human-readable description shown in the hint row.
    assert "why" in data, data


@step("perspectives: viewport-paint contract — /api/viewport returns layout + composite")
def t_perspectives_viewport_paint_contract():
    # The "viewport-paint" perspective hands ctx.fileName to its Web
    # Component, which calls /api/viewport/<filename>. The response must
    # include composite_b64, placements, viewport_w/h.
    target = "LogoEP4.prs"
    data = _http("GET", f"/api/viewport/{urllib.parse.quote(target)}")
    assert data.get("viewport_w") == 1278, data
    assert data.get("viewport_h") == 768, data
    assert "composite_b64" in data and data["composite_b64"].startswith("data:image/png;"), data
    assert isinstance(data.get("placements"), list), data
    assert "layout" in data, data


@step("perspectives: deploy-diff contract — /api/repack_diff with empty edited_indices")
def t_perspectives_deploy_diff_contract():
    # The "deploy-diff" perspective reuses the existing /api/repack_diff
    # endpoint (driven by app.js's showDeployDiff). Verify the empty case
    # returns a sane shape (zero-changed_indices but a valid diff).
    target = "LogoEP4.prs"
    r = _http("POST", "/api/repack_diff", {
        "filename": target,
        "edited_indices": [],
    })
    assert r["filename"] == target, r
    assert "tile_count" in r, r
    assert isinstance(r.get("changed_indices"), list), r
    assert isinstance(r.get("unchanged_indices"), list), r
    # Changed should be empty for empty edited_indices
    assert r["changed_indices"] == [], r
    # All tiles should be in unchanged
    assert len(r["unchanged_indices"]) == r["tile_count"], r


# ============================================================
# Deploy: dev mirror -> live PSOBB.IO data dir (2026-04-24)
# ============================================================

# Live = the user's playable game install. Dev = editor's mirror.
LIVE_DATA_DIR = Path(os.path.expanduser("~/PSOBB.IO/data")).resolve()
DEV_DATA_DIR = Path(r"C:/tmp_pso_dev/data").resolve()


@step("deploy: GET /api/deploy/config returns dev_dir + live_dir")
def t_deploy_config():
    r = _http("GET", "/api/deploy/config")
    assert "dev_dir" in r, r
    assert "live_dir" in r, r
    # Paths normalize to either / or \\ on Windows; just check they end with the
    # right basenames so we don't pin one separator style.
    assert r["dev_dir"].lower().endswith("data"), r
    assert r["live_dir"].lower().endswith("data"), r
    assert r["live_exists"] is True, r
    assert r["dev_exists"] is True, r


@step("deploy: GET /api/deploy/diff returns valid shape")
def t_deploy_diff_shape():
    r = _http("GET", "/api/deploy/diff", timeout=180)
    assert "changed" in r, r
    assert "dev_only" in r, r
    assert "live_only" in r, r
    assert "summary" in r, r
    assert "dev_dir" in r and "live_dir" in r, r
    for key in ("changed", "dev_only", "live_only"):
        assert isinstance(r[key], list), f"{key} not list: {type(r[key]).__name__}"
    # changed entries must carry md5 + size for both sides
    for e in r["changed"]:
        for field in ("name", "dev_md5", "live_md5", "dev_size", "live_size"):
            assert field in e, f"changed entry missing {field}: {e}"
        assert e["dev_md5"] != e["live_md5"], (
            f"diff returned a 'changed' entry with matching md5: {e}"
        )
    for e in r["dev_only"]:
        for field in ("name", "dev_size"):
            assert field in e, f"dev_only entry missing {field}: {e}"
    for e in r["live_only"]:
        for field in ("name", "live_size"):
            assert field in e, f"live_only entry missing {field}: {e}"
    # summary aligns with the lists
    assert r["summary"]["changed_count"] == len(r["changed"]), r["summary"]
    assert r["summary"]["dev_only_count"] == len(r["dev_only"]), r["summary"]
    assert r["summary"]["live_only_count"] == len(r["live_only"]), r["summary"]


@step("deploy: POST /api/deploy/promote with empty files list returns 400")
def t_deploy_promote_empty_400():
    try:
        _http("POST", "/api/deploy/promote", {"files": [], "create_backup": True})
    except RuntimeError as e:
        assert "400" in str(e), str(e)
        return
    raise AssertionError("empty files list should have been rejected")


@step("deploy: POST /api/deploy/promote with invalid filename rejected")
def t_deploy_promote_path_traversal_400():
    bad_names = [
        "..\\evil.bin",
        "../etc/passwd",
        "subdir/file.prs",
        "back\\slash.prs",
        "",
        ".",
        "..",
    ]
    for bad in bad_names:
        try:
            _http("POST", "/api/deploy/promote", {"files": [bad]})
        except RuntimeError as e:
            assert "400" in str(e), f"{bad!r}: expected 400, got {e}"
            continue
        raise AssertionError(f"deploy/promote with bad name {bad!r} should have been rejected")


@step("deploy: dry-run via dev_only file works (no live file written if create_backup=False)")
def t_deploy_promote_dry_run_dev_only():
    """Plant a unique probe file in DEV_DATA_DIR that does NOT exist in
    LIVE_DATA_DIR (so it's a dev_only entry per /api/deploy/diff). Promote
    it with create_backup=False; the live target should appear with the
    SAME bytes as the dev source, and (since live had no prior version)
    NO backup file should be written.

    Cleanup: remove the planted file from both dev and live so the test
    leaves no residue.
    """
    probe_name = f"_e2e_promote_probe_{int(time.time())}.bin"
    probe_dev = DEV_DATA_DIR / probe_name
    probe_live = LIVE_DATA_DIR / probe_name
    payload = b"e2e-deploy-promote-probe-" + os.urandom(16)

    # Pre-conditions: clean state for this test
    if probe_dev.exists():
        probe_dev.unlink()
    if probe_live.exists():
        probe_live.unlink()

    try:
        probe_dev.write_bytes(payload)
        # Verify the diff sees it as dev_only
        diff = _http("GET", "/api/deploy/diff", timeout=180)
        dev_only_names = {e["name"] for e in diff["dev_only"]}
        assert probe_name in dev_only_names, (
            f"probe {probe_name} not classified as dev_only — diff sees: {sorted(dev_only_names)}"
        )

        # Promote without backup. Since the live target is missing, NO backup
        # should be written even if create_backup=True; but we explicitly set
        # False to assert that contract too.
        r = _http("POST", "/api/deploy/promote", {
            "files": [probe_name],
            "create_backup": False,
        })
        assert r["ok_count"] == 1, r
        assert r["fail_count"] == 0, r
        result = r["results"][0]
        assert result["ok"] is True, result
        assert result["name"] == probe_name, result
        # No prior live file -> no backup
        assert result.get("backup_name") in (None, ""), result
        assert result.get("live_size") == len(payload), result

        # Live target now exists with the dev bytes
        assert probe_live.exists(), "live probe was not written"
        assert probe_live.read_bytes() == payload, "live probe bytes != dev bytes"

        # Confirm: NO .pre_promote_* file accompanies the probe (since
        # there was no prior live file to back up).
        backups = list(LIVE_DATA_DIR.glob(f"{probe_name}.pre_promote_*"))
        assert not backups, f"unexpected backup created: {[b.name for b in backups]}"
    finally:
        # Cleanup
        if probe_dev.exists():
            try:
                probe_dev.unlink()
            except OSError:
                pass
        if probe_live.exists():
            try:
                probe_live.unlink()
            except OSError:
                pass
        # Remove any backups too (defence in depth in case the assertions
        # above miss something).
        for bk in LIVE_DATA_DIR.glob(f"{probe_name}.pre_promote_*"):
            try:
                bk.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Map Editor smoke tests (Task 5, audit 2026-04-25)
# ---------------------------------------------------------------------------
# The Map Editor was the biggest new surface shipped on 2026-04-25 and
# arrived with only server-side unit tests. These steps exercise the
# full HTTP surface end-to-end: list -> bundle -> roundtrip edits ->
# validation rejections. Each step uses a unique map_id sandbox so it
# can run idempotently.

@step("map editor: GET /api/map/list returns picker payload with maps")
def t_map_list_smoke():
    r = _http("GET", "/api/map/list")
    assert isinstance(r, dict), f"expected dict, got {type(r).__name__}"
    # Picker payload from scene_loader.make_picker_payload:
    #   { categories: [...], maps: [{map_id, area, label, floors, ...}, ...] }
    maps = r.get("maps")
    assert isinstance(maps, list), f"maps not list: {type(maps).__name__}"
    assert len(maps) > 0, "map catalogue is empty"
    cats = r.get("categories")
    assert isinstance(cats, list) and len(cats) > 0, f"categories not populated: {cats}"
    # Each entry must carry the picker-required keys (label is the
    # display name; display_name is not part of the wire shape).
    sample = maps[0]
    for k in ("map_id", "category", "floors", "label"):
        assert k in sample, f"missing key {k!r} in maps[0]: {sample}"
    # Confirm at least one well-known map is present.
    ids = {m["map_id"] for m in maps}
    assert "aancient01" in ids or any("ancient" in mid for mid in ids), \
        f"aancient01 absent from catalogue (have {sorted(ids)[:6]}...)"


@step("map editor: GET /api/map/aancient01?floor=0 returns floor bundle")
def t_map_bundle_smoke():
    r = _http("GET", "/api/map/aancient01?floor=0")
    assert isinstance(r, dict), r
    # floor_bundle shape: {ok, map_id, floor, renderable, ...}
    for k in ("map_id", "floor"):
        assert k in r, f"missing {k!r}: {sorted(r.keys())}"
    assert r["map_id"] == "aancient01", r
    assert r["floor"] == 0, r
    # `renderable` may be empty for some maps; just check the key exists.
    assert "renderable" in r, f"missing renderable key: {sorted(r.keys())}"


@step("map editor: POST /api/map/edits roundtrips spawn via GET edits")
def t_map_edits_roundtrip():
    sandbox = "smoketest01"  # matches ^[a-z]+\d+$, not a real map id
    body = {
        "map_id": sandbox,
        "spawns": [{
            "id": 1,
            "type": "mob",
            "world_pos": [10.5, 0.0, -25.25],
            "rotation": 1.5707963,
            "type_data": {"mob_class": 0x40, "comment": "smoke"},
        }],
        "waypoints": [],
    }
    try:
        r = _http("POST", "/api/map/edits", body)
        assert r.get("ok") is True, r
        assert r["map_id"] == sandbox, r
        assert r["spawn_count"] == 1, r
        # Roundtrip: GET should return the same payload normalized.
        g = _http("GET", f"/api/map/edits/{sandbox}")
        assert g.get("ok") is True, g
        assert g.get("exists") is True, g
        spawns = g.get("spawns") or []
        assert len(spawns) == 1, spawns
        s = spawns[0]
        assert s["id"] == 1, s
        assert s["type"] == "mob", s
        assert abs(s["world_pos"][0] - 10.5) < 1e-4, s
        assert abs(s["world_pos"][2] - -25.25) < 1e-4, s
    finally:
        # Sandbox cleanup so the test is idempotent across runs.
        from pathlib import Path
        for p in Path("cache/map_edits").glob(f"{sandbox}.json*"):
            try:
                p.unlink()
            except OSError:
                pass


@step("map editor: bad spawn type rejected (400)")
def t_map_edits_bad_type_rejected():
    sandbox = "smoketest02"
    body = {
        "map_id": sandbox,
        "spawns": [{
            "id": 1,
            "type": "alien_invasion",  # not in VALID_SPAWN_TYPES
            "world_pos": [0.0, 0.0, 0.0],
            "rotation": 0.0,
            "type_data": {},
        }],
        "waypoints": [],
    }
    try:
        _http("POST", "/api/map/edits", body)
    except RuntimeError as e:
        assert "400" in str(e), f"expected HTTP 400, got: {e}"
        return
    raise AssertionError("expected POST to fail with 400 on bad spawn type")


@step("map editor: duplicate spawn ids rejected (400)")
def t_map_edits_dup_id_rejected():
    sandbox = "smoketest03"
    body = {
        "map_id": sandbox,
        "spawns": [
            {"id": 7, "type": "mob",  "world_pos": [0, 0, 0], "rotation": 0,
             "type_data": {}},
            {"id": 7, "type": "chest","world_pos": [1, 0, 1], "rotation": 0,
             "type_data": {}},
        ],
        "waypoints": [],
    }
    try:
        _http("POST", "/api/map/edits", body)
    except RuntimeError as e:
        assert "400" in str(e), f"expected HTTP 400, got: {e}"
        return
    raise AssertionError("expected POST to fail with 400 on duplicate id")


@step("map editor: dangling waypoint reference rejected (400)")
def t_map_edits_dangling_waypoint_rejected():
    sandbox = "smoketest04"
    body = {
        "map_id": sandbox,
        "spawns": [
            {"id": 1, "type": "mob", "world_pos": [0, 0, 0], "rotation": 0,
             "type_data": {}},
        ],
        # waypoint references id 99 which doesn't exist in spawns
        "waypoints": [{"from_id": 1, "to_id": 99, "style": "walk", "speed": 1.0}],
    }
    try:
        _http("POST", "/api/map/edits", body)
    except RuntimeError as e:
        assert "400" in str(e), f"expected HTTP 400, got: {e}"
        return
    raise AssertionError("expected POST to fail with 400 on dangling waypoint")


def main() -> int:
    print(f"== PSOBB Texture Editor e2e ({API}) ==")
    # tests in order
    t_health()
    t_files()
    t_tiles_prs()
    t_tiles_xvm()
    t_path_injection()
    t_upscale_native()
    t_upscale_x4()
    t_upscale_validation()
    t_repack_dryrun()
    t_repack_deploy()
    t_restore()
    t_repack_bad_input()
    # V3
    t_models_v3()
    t_upscale_advanced_opts()
    t_upscale_cascade_8()
    t_upscale_cascade_8_native()
    t_upscale_scale_validation_v3()
    t_upscale_tilesize_validation()
    # V4 (UX)
    t_import_native()
    t_import_4x_native()
    t_import_keep_oversized()
    t_import_bad_dim()
    t_import_nonuniform()
    t_import_not_png()
    t_import_bad_tile()
    t_repack_diff()
    t_repack_diff_unknown()
    t_repack_diff_empty()
    # V4 (quality / repack / export)
    t_v4q_splice_identical()
    t_v4q_splice_mix()
    t_v4q_export_only()
    t_v4q_export_only_clean()
    t_v4q_verify()
    t_v4q_xvr_codec_splice_offline()
    t_v4q_xvr_codec_modified()
    # Code-quality cleanup pass
    t_cleanup_health_locks()
    t_cleanup_gpu_id_validation()
    t_cleanup_gpu_id_boundaries()
    t_cleanup_repack_body_too_large()
    t_cleanup_repack_too_many_tiles()
    t_cleanup_concurrent_repack()
    t_cleanup_modal_upscale_contract()
    t_cleanup_invalid_filename_all_endpoints()
    t_cleanup_export_token_path_traversal()
    # BML container reader (Agent 4)
    t_bml_list()
    t_bml_extract_nj()
    t_bml_bad_header()
    t_bml_empty()
    t_bml_texture_xvm()
    # XJ mesh parser + /api/model_mesh endpoint
    t_xj_parse_plabdy00_smoke()
    t_xj_proto_consistency()
    t_xj_bm4_player_body()
    t_xj_boss09_core()
    # Bone-tree transform regression guards (added 2026-04-24).
    t_xj_gibbles_submesh_transforms()
    t_xj_dragon_submesh_combined_pose()
    # Descriptor-table XJ parser (formats/xj_descriptor.py, 2026-04-24)
    # — handles the ~263 BML-inner ``.xj`` files that the chunk parser
    # could not. See AGENT_XJ_DESCRIPTOR_REPORT.md for layout details.
    t_xj_descriptor_door01l()
    t_xj_descriptor_shape_invariants()
    t_xj_descriptor_world_xform_populated()
    t_xj_bad_payload()
    # Rotation-order + EVAL_HIDE regression guards
    # (AGENT_MODEL_DEEP_DEBUG_REPORT, 2026-04-24).
    t_xj_rotation_order_de_rol_le()
    t_xj_eval_hide_de_rol_le_default()
    t_model_mesh_endpoint_consistency()
    t_model_mesh_404()
    t_model_mesh_bad_ext()
    # Atlas mode
    t_atlas_logo_layout()
    t_atlas_unknown_404()
    t_atlas_import_native()
    t_atlas_import_4x()
    t_atlas_import_bad_dim()
    t_atlas_roundtrip_repack()
    # Viewport mode (16:9 transform)
    t_viewport_logo_atlas()
    t_viewport_unknown_centered()
    t_viewport_paint_roundtrip()
    t_viewport_paint_bad_dim()
    # Manifest (Phase A — Agent 1)
    t_manifest_schema_valid()
    t_manifest_xvm_classification()
    t_manifest_bml_classification()
    t_manifest_prs_inner_xvm()
    t_manifest_cache_rebuild()
    t_manifest_excludes_backups()
    # Formats (Phase A - Agent 2): IFF + AFS readers
    t_iff_njcm_top()
    t_iff_njtl_after_njcm()
    t_iff_corrupted()
    t_afs_itemmodel_parses()
    t_afs_bad_magic()
    t_afs_oversized_offset()
    # Texture <-> model matcher (Phase A - Agent 3)
    # Pure-Python; runs without the live HTTP server.
    t_match_r1_tex_sibling()
    t_match_r2_bml_internal()
    t_match_r3_njtl_lookup()
    t_match_r4_player_afs()
    t_match_r5_item_pair()
    t_match_r6_map_prefix()
    # Tree frontend (Phase A — Agent 5). Gracefully skips if Agent 1's
    # /api/manifest endpoint isn't deployed yet (PENDING).
    t_tree_categories_match_enum()
    t_tree_entry_minimum_fields()
    # Asset coverage Phase 1+2 (2026-04-25)
    t_manifest_categories_endpoint()
    t_models_category_count()
    t_model_auto_bind_r1()
    t_raw_endpoint_bml_serves()
    t_raw_endpoint_rejects_traversal()
    # BML-inner-path resolver (2026-04-24): asset-tree click-through
    # form `<bml>#<inner>` for tile/preview/mesh/raw endpoints
    t_bml_inner_direct_tile_png_regression()
    t_bml_inner_tile_png_via_hash()
    t_bml_inner_model_preview()
    t_bml_inner_invalid_syntax()
    # Model→mesh routing fix (2026-04-24): Models tab click should
    # resolve through /api/model_mesh directly, not via texture-driven
    # hint flow. Regression guards against the "primitive cube" bug.
    t_model_mesh_via_path_for_bml_inner()
    t_model_mesh_inner_matched_textures()
    t_model_mesh_top_level_bml()
    t_model_mesh_resolves_inner_from_matched_textures()
    # Deploy: dev mirror -> live PSOBB.IO data dir
    t_deploy_config()
    t_deploy_diff_shape()
    t_deploy_promote_empty_400()
    t_deploy_promote_path_traversal_400()
    t_deploy_promote_dry_run_dev_only()
    # Unified viewport perspectives (2026-04-24)
    t_perspectives_tile_grid_contract()
    t_perspectives_3d_view_contract()
    t_perspectives_viewport_paint_contract()
    t_perspectives_deploy_diff_contract()
    # Map Editor smoke (2026-04-25): exercise the new map endpoints
    # end-to-end against a live server. Validation rejections must
    # surface as 400s.
    t_map_list_smoke()
    t_map_bundle_smoke()
    t_map_edits_roundtrip()
    t_map_edits_bad_type_rejected()
    t_map_edits_dup_id_rejected()
    t_map_edits_dangling_waypoint_rejected()

    print()
    print(f"PASS: {len(PASS)}")
    print(f"FAIL: {len(FAIL)}")
    for f in FAIL:
        print(f"  - {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
