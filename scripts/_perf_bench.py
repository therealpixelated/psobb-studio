#!/usr/bin/env python3
"""Ad-hoc perf benchmark for the psov2 model load path (NOT committed-test).

Boots a headless Chromium against an already-running server (PORT env),
forces a real 1400x900 viewport + sized model modal, drives real opens via
window.psoOpenModelByPath(path,{},[]) exactly as asset_router does, and times
the mesh-visible signal (#modelMeshStats showing "verts N").

Measures:
  cold   — first open of each model (server + client cold)
  reopen — re-open Dark Bringer (should be ~instant once a client cache lands)
  abc    — rapid A->B->C with NO awaits between opens; asserts the FINAL
           painted mesh matches C (abort-on-switch) and reports total time.

Reports JSON to stdout.
"""
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

PORT = int(os.environ.get("PORT", "8785"))
BASE = f"http://127.0.0.1:{PORT}"

DARK_BRINGER = "bm_ene_df2_bringer_a.bml#bm8_s_kb_body.nj"
BITER = "bm_ene_biter_body.bml#biter_body.nj"
ASTARK = "bm_ene_astark.bml#nj00_astark.nj"

# Marker that proves the Dark Bringer's native walk clip is auto-playing.
DB_WALK_HINT = "bm8_s_kb_body"


def _force_modal(page):
    page.evaluate(
        """() => {
          const m = document.querySelector('#modelModal');
          if (m) { m.hidden = false; m.style.display='block'; }
          const c = document.querySelector('#modelCanvas') || document.querySelector('#modelModal canvas');
          if (c) { c.style.width='900px'; c.style.height='700px'; }
          const stage = document.querySelector('#modelStage') || document.querySelector('#modelModalBody');
          if (stage) { stage.style.width='900px'; stage.style.height='700px'; }
        }"""
    )


def _open_and_time(page, path, timeout_ms=30000):
    """Open one model and return ms until #modelMeshStats shows 'verts'."""
    page.evaluate(
        """(p) => {
          window.__benchStats = null;
          const el = document.querySelector('#modelMeshStats');
          if (el) el.textContent = '';
          window.__benchOpen = window.psoOpenModelByPath(p, {}, []);
        }""",
        path,
    )
    _force_modal(page)
    t0 = time.time()
    deadline = t0 + timeout_ms / 1000.0
    last = ""
    while time.time() < deadline:
        last = page.evaluate(
            "() => { const e=document.querySelector('#modelMeshStats'); return e ? e.textContent : ''; }"
        )
        if last and "verts" in last:
            return (time.time() - t0) * 1000.0, last
        time.sleep(0.02)
    return None, last


def _open_no_wait(page, path):
    page.evaluate(
        """(p) => {
          const el = document.querySelector('#modelMeshStats');
          if (el) el.textContent = '';
          window.psoOpenModelByPath(p, {}, []);
        }""",
        path,
    )
    _force_modal(page)


def _title(page):
    return page.evaluate(
        "() => { const e=document.querySelector('#modelModalTitle'); return e ? e.textContent : ''; }"
    )


def _anim_state(page):
    """Return {playing, motion, clipCount} from the live psov2 anim state."""
    return page.evaluate(
        """() => {
          try {
            const sel = document.querySelector('#modelAnimSel');
            const scrubLbl = document.querySelector('#modelAnimStatus');
            return {
              sel: sel ? sel.value : null,
              status: scrubLbl ? scrubLbl.textContent : null,
              barHidden: (document.querySelector('#modelAnimBar')||{}).hidden,
            };
          } catch(e){ return {err:String(e)}; }
        }"""
    )


def main():
    out = {"port": PORT}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True, args=["--disable-gpu", "--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(BASE, wait_until="domcontentloaded")
        # let module scripts + three.js import settle
        page.wait_for_function(
            "() => typeof window.psoOpenModelByPath === 'function'", timeout=20000
        )
        time.sleep(0.5)

        # ---- COLD opens (each model first time) ----
        cold = {}
        for label, p in [("dark_bringer", DARK_BRINGER), ("biter", BITER), ("astark", ASTARK)]:
            ms, stats = _open_and_time(page, p)
            cold[label] = {"ms": round(ms, 1) if ms else None, "stats": stats}
            time.sleep(0.3)

        # capture Dark Bringer animation correctness AFTER a fresh cold open
        _open_and_time(page, DARK_BRINGER)
        time.sleep(0.5)
        db_anim_cold = _anim_state(page)

        # ---- REOPEN (cache check): re-open Dark Bringer ----
        reopen = []
        for _ in range(3):
            ms, stats = _open_and_time(page, DARK_BRINGER)
            reopen.append(round(ms, 1) if ms else None)
            time.sleep(0.25)

        # ---- RAPID A->B->C (no awaits) ----
        # Fire all three back to back; only C should ultimately paint.
        # We SAMPLE the painted verts signature over a settle window to catch
        # a stale A/B commit clobbering C (the documented non-deterministic
        # bug). astark(C)=2994, biter(B)=2760, darkbringer(A)=2694.
        astark_stats = cold["astark"]["stats"]
        biter_stats = cold["biter"]["stats"]
        c_verts = _verts(astark_stats)
        b_verts = _verts(biter_stats)
        a_verts = _verts(cold["dark_bringer"]["stats"])
        _open_no_wait(page, DARK_BRINGER)
        _open_no_wait(page, BITER)
        t0 = time.time()
        _open_no_wait(page, ASTARK)  # C = astark
        # Sample the painted signature continuously for a fixed settle window.
        deadline = t0 + 8
        samples = []
        final_stats = ""
        c_first_seen_ms = None
        stale_after_c = False
        while time.time() < deadline:
            final_stats = page.evaluate(
                "() => { const e=document.querySelector('#modelMeshStats'); return e ? e.textContent : ''; }"
            )
            if final_stats and "verts" in final_stats:
                v = _verts(final_stats)
                now_ms = (time.time() - t0) * 1000.0
                samples.append((round(now_ms, 0), v))
                if v == c_verts and c_first_seen_ms is None:
                    c_first_seen_ms = now_ms
                # if C already showed and then a non-C (A/B) signature appears,
                # that's the clobber bug.
                if c_first_seen_ms is not None and v in (a_verts, b_verts) and v != c_verts:
                    stale_after_c = True
            time.sleep(0.03)
        abc_settle_ms = (time.time() - t0) * 1000.0
        final_title = _title(page)
        # collapse samples to a transition sequence (de-dup consecutive)
        seq = []
        for _t, v in samples:
            if not seq or seq[-1] != v:
                seq.append(v)
        abc_final_matches_c = (final_title.endswith(ASTARK) or ASTARK in final_title)
        # confirm the painted mesh signature equals astark's (not biter/db's)
        abc_mesh_is_c = bool(astark_stats and final_stats and
                             _verts(final_stats) == c_verts)

        # ---- confirm animation STILL plays on Dark Bringer after all churn ----
        _open_and_time(page, DARK_BRINGER)
        time.sleep(0.6)
        db_anim_after = _anim_state(page)

        browser.close()

    out["cold"] = cold
    out["reopen_ms"] = reopen
    out["abc"] = {
        "settle_ms": round(abc_settle_ms, 1),
        "final_title": final_title,
        "final_stats": final_stats,
        "final_matches_c": abc_final_matches_c,
        "mesh_is_c": abc_mesh_is_c,
        "c_first_seen_ms": round(c_first_seen_ms, 1) if c_first_seen_ms else None,
        "stale_clobber_after_c": stale_after_c,
        "vert_transition_seq": seq,
        "signature": {"A_db": a_verts, "B_biter": b_verts, "C_astark": c_verts},
    }
    out["db_anim_cold"] = db_anim_cold
    out["db_anim_after"] = db_anim_after
    out["page_errors"] = errs[:10]
    print(json.dumps(out, indent=2))


def _verts(stats):
    import re
    m = re.search(r"verts\s+(\d+)", stats or "")
    return int(m.group(1)) if m else -1


if __name__ == "__main__":
    main()
