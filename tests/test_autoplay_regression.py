"""Regression test for the model-viewer auto-play (walk-on-load).

PSOBB.IO models that ship NJM motions are expected to start playing their
"default" motion (walk > move > swim/fly > idle/wait/stand > first) the
moment the user opens them in the 3D viewer. This contract was originally
established with the 2026-04-24 NJM animation work and re-confirmed by
the 2026-04-25 ``pick_default_motion`` tier change for ``move_*`` BMLs.

Regression 2026-04-25 (this file's motivation): ``populateAnimationPanel``
in ``static/model_viewer.js`` had an early ``if (!sel) return`` that
bailed out BEFORE the ``await loadMotion(default_name)`` call when the
``#modelAnimSel`` dropdown wasn't in the DOM (e.g. unified-viewport or
tile-detail perspectives that hide ``#modelModal``). The result: dragons,
monsters, and bm4 props loaded in bind pose instead of walking. Server-
side picker + animation_data wire format were unaffected — the auto-play
trigger was being short-circuited on the client side.

The fix: decouple auto-play from the dropdown's existence. The dropdown
is a UI affordance; ``state.anim.playing`` is the source of truth for
the render loop. The two regression dimensions exercised here:

  1. Dragon BML auto-plays walk on load (the canonical "model is moving"
     case).
  2. bm4_ps_ma_body auto-plays a SAME-STEM motion. Pre-2026-04-26 the
     picker was verb-only and landed on ``move_bm4_ps_mb_body`` — a
     1-bone track for the wrong sub-form rig. The new four-tier
     resolver demotes that to Tier 3 and picks the 43-bone
     ``wait2_bm4_ps_ma_body`` (idle, but correct rig).
  3. Mericarol BML auto-plays ``wait_*`` (no locomotion; tier-4 fallback).
  4. Auto-play STILL fires when the dropdown is missing (the actual bug).
  5. Scrubber-driven ``psoSeekAnimationToFrame`` still pauses (the v4
     motion-editor's "stay paused on release" semantics are preserved).

The test runs in jsdom under Node — pytest spawns ``node`` and asserts
on the exit code. The Node script itself (test_autoplay_jsdom.mjs)
handles the actual jsdom + THREE shimming.

Skips when:
  * Node isn't on PATH
  * jsdom isn't installed (``npm install jsdom`` in repo root)
  * The local server isn't running on 127.0.0.1:8765 (the test needs
    real HTTP responses — we don't fake them)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NODE_TEST = REPO / "tests" / "test_autoplay_jsdom.mjs"


def _node_available() -> bool:
    return shutil.which("node") is not None


def _jsdom_installed() -> bool:
    return (REPO / "node_modules" / "jsdom").is_dir()


def _server_running() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


@pytest.mark.skipif(not _node_available(), reason="node not on PATH")
@pytest.mark.skipif(not _jsdom_installed(), reason="jsdom not installed (run `npm install jsdom` in repo root)")
@pytest.mark.skipif(not _server_running(), reason="server not running on 127.0.0.1:8765")
def test_model_viewer_autoplay_regression():
    """Run the jsdom-based auto-play regression suite.

    The Node script exits 0 on pass, 1 on regression, 2-3 on infra
    failure (jsdom load error etc.). We surface the script's stdout/stderr
    on failure so the regression details are visible in pytest output.
    """
    assert NODE_TEST.is_file(), f"missing test driver: {NODE_TEST}"
    proc = subprocess.run(
        ["node", str(NODE_TEST)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        msg = (
            f"Auto-play regression test failed (exit {proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )
        pytest.fail(msg)
    # Sanity-check the output mentions the 5 cases we expect
    out = proc.stdout
    assert "Test 1 PASS" in out, "missing Test 1 (dragon walk)"
    assert "Test 2 PASS" in out, "missing Test 2 (no-dropdown auto-play)"
    assert "Test 3 PASS" in out, "missing Test 3 (bm4 same-stem)"
    assert "Test 4 PASS" in out, "missing Test 4 (mericarol wait)"
    assert "Test 5 PASS" in out, "missing Test 5 (v4 scrubber pause)"
