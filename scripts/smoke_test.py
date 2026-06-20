#!/usr/bin/env python3
"""Smoke test — proves the committed source is COMPLETE and the app boots.

Why this exists: a `.gitignore` rule once dropped package files (`__init__.py`,
`_imageutil.py`, ...) from the repo, so a fresh clone could not import and the
server would not start — yet it worked fine locally (the files were still on
disk). This test runs the import + boot checks against *only what is committed*,
so CI (which checks out a clean tree) catches that class of breakage before a
push goes out.

Two gates:
  1. import-all  — import every first-party module (server, manifest,
     atlas_layouts, formats.*, aigen.*). A missing file / missing package
     __init__ fails here. Optional heavy third-party deps (torch, diffusers,
     ...) that aren't installed are tolerated; missing FIRST-PARTY modules are
     hard failures.
  2. boot        — start the uvicorn server and require /api/health to answer
     200 within 60s, then shut it down.

Exit code 0 on success, non-zero on any first-party import failure or boot
failure. Run locally with:  python scripts/smoke_test.py
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Third-party deps that are intentionally optional (lazy-imported by feature
# code). If absent in CI, a module that imports one is skipped, not failed.
OPTIONAL_DEPS = {
    "torch", "diffusers", "transformers", "accelerate", "safetensors",
    "cv2", "scipy", "trimesh", "pygltflib",
}

TOP_LEVEL = ["server", "manifest", "atlas_layouts"]
FIRST_PARTY_PACKAGES = ["formats", "aigen"]


def _discover_modules() -> list[str]:
    mods = list(TOP_LEVEL)
    for pkg in FIRST_PARTY_PACKAGES:
        mods.append(pkg)
        pkg_dir = ROOT / pkg
        if pkg_dir.is_dir():
            for info in pkgutil.iter_modules([str(pkg_dir)]):
                mods.append(f"{pkg}.{info.name}")
    return mods


def import_all() -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    for mod in _discover_modules():
        try:
            importlib.import_module(mod)
            print(f"  ok    {mod}")
        except Exception as e:  # noqa: BLE001 — we classify below
            missing = getattr(e, "name", None)
            root_pkg = missing.split(".")[0] if missing else None
            if isinstance(e, ModuleNotFoundError) and root_pkg in OPTIONAL_DEPS:
                print(f"  skip  {mod} (optional dep '{missing}' not installed)")
            else:
                print(f"  FAIL  {mod}: {type(e).__name__}: {e}")
                failures.append((mod, f"{type(e).__name__}: {e}"))
    return failures


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def boot() -> tuple[bool, str]:
    port = _free_port()
    env = dict(os.environ)
    # A data dir that merely EXISTS is enough for /api/health to answer 200.
    env.setdefault("PSO_DATA_DIR", str(ROOT))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 60
        url = f"http://127.0.0.1:{port}/api/health"
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read().decode("utf-8", "replace") if proc.stdout else ""
                return False, f"server exited early (code {proc.returncode}):\n{out[-2000:]}"
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return True, f"/api/health -> 200 on :{port}"
            except Exception:
                time.sleep(0.5)
        return False, "server did not answer /api/health within 60s"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def main() -> int:
    print("== import smoke (catches dropped/missing first-party source) ==")
    failures = import_all()
    if failures:
        print(f"\nIMPORT SMOKE FAILED — {len(failures)} module(s) could not import:")
        for mod, err in failures:
            print(f"  - {mod}: {err}")
        print("\nLikely a source file is missing from the commit "
              "(e.g. swallowed by .gitignore).")
        return 1

    print("\n== boot smoke (uvicorn server starts + /api/health answers) ==")
    ok, msg = boot()
    print(f"  {'ok' if ok else 'FAIL'}  {msg}")
    if not ok:
        return 1

    print("\nSMOKE PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
