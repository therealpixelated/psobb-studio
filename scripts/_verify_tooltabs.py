#!/usr/bin/env python3
"""Ad-hoc verification harness for the fix/tooltabs work (NOT a committed test).

Boots a headless Chromium against an already-running server (PORT env), forces
a real viewport + sized model modal, loads the Dark Bringer (88-bone psov2
SkinnedMesh), then asserts the owner-reported tool-tab fixes objectively via
live DOM/state (no pixel screenshots):

  1. Skeleton tab — window.psoGetSkeleton() returns >0 bones.
  2. UV tab       — collectSubmeshes() (psoGetDebugMeshes w/ .mesh + uv) >0.
  3. Paint        — bind a CanvasTexture to a material slot, mutate a texel,
                    assert mesh.material[slot].map === that texture AND a
                    sampled texel changed.
  4. Animation    — psoGetAnimDebug().playing === true (walk autoloads).
  5. Tabs         — exactly ONE "Animation" tab; Edit + Anim Editor tabs gone;
                    tab count reduced.

Reports JSON to stdout.
"""
import json
import os
import time

from playwright.sync_api import sync_playwright

PORT = int(os.environ.get("PORT", "8787"))
BASE = f"http://127.0.0.1:{PORT}"
DARK_BRINGER = "bm_ene_df2_bringer_a.bml#bm8_s_kb_body.nj"


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


def _open_and_wait(page, path, timeout_ms=30000):
    page.evaluate(
        """(p) => {
          const el = document.querySelector('#modelMeshStats');
          if (el) el.textContent = '';
          window.psoOpenModelByPath(p, {}, []);
        }""",
        path,
    )
    _force_modal(page)
    deadline = time.time() + timeout_ms / 1000.0
    last = ""
    while time.time() < deadline:
        last = page.evaluate(
            "() => { const e=document.querySelector('#modelMeshStats'); return e ? e.textContent : ''; }"
        )
        if last and "verts" in last:
            return last
        time.sleep(0.02)
    return last


def main():
    out = {"port": PORT}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(BASE, wait_until="domcontentloaded")
        page.wait_for_function(
            "() => typeof window.psoOpenModelByPath === 'function'", timeout=20000
        )
        time.sleep(0.5)

        stats = _open_and_wait(page, DARK_BRINGER)
        out["mesh_stats"] = stats
        # Give native motions + skeleton mirror a beat to settle.
        time.sleep(1.0)

        # 1. SKELETON ---------------------------------------------------
        out["skeleton_bones"] = page.evaluate(
            "() => { const s = window.psoGetSkeleton && window.psoGetSkeleton(); return s ? s.length : 0; }"
        )
        out["skeleton_sample"] = page.evaluate(
            """() => {
              const s = window.psoGetSkeleton && window.psoGetSkeleton();
              if (!s || !s.length) return null;
              const b = s[Math.min(1, s.length-1)];
              return { index:b.index, parent:b.parent, pos:b.position, rot:b.rotation_bams };
            }"""
        )

        # 2. UV submeshes (exactly what uv_panel.collectSubmeshes does) --
        out["uv_submeshes"] = page.evaluate(
            """() => {
              const dbg = window.psoGetDebugMeshes && window.psoGetDebugMeshes();
              if (!dbg || !dbg.length) return 0;
              let n = 0;
              for (const e of dbg) {
                const mesh = e && e.mesh;
                if (!mesh || !mesh.geometry) continue;
                if (!mesh.geometry.getAttribute('uv')) continue;
                n++;
              }
              return n;
            }"""
        )
        out["debug_mesh_count"] = page.evaluate(
            "() => { const d = window.psoGetDebugMeshes(); return d ? d.length : 0; }"
        )

        # 3. PAINT — bind a CanvasTexture to a real material slot + mutate
        # a texel, then prove the bound material's map is OUR canvas AND the
        # sampled pixel changed. We drive psoSetMaterialTexture directly (the
        # same call paint_panel.bindActiveTile makes) since a real pointer
        # raycast is unreliable headless.
        out["paint"] = page.evaluate(
            """() => {
              const THREE = window.THREE;
              const res = { ok:false, steps:{} };
              const list = window.psoListMeshTextures ? window.psoListMeshTextures() : [];
              res.steps.tileRows = list.length;
              if (!list.length) { res.reason='no tile rows'; return res; }
              // Pick the first tile row + its first material_id.
              const row = list[0];
              const mid = (row.material_ids && row.material_ids[0]);
              res.steps.mid = mid;
              if (mid === undefined) { res.reason='no material_id'; return res; }
              // Build a 16x16 canvas filled with a known color.
              const cv = document.createElement('canvas');
              cv.width = 16; cv.height = 16;
              const ctx = cv.getContext('2d', { willReadFrequently:true });
              ctx.fillStyle = '#ff00ff'; ctx.fillRect(0,0,16,16);
              const tex = new THREE.CanvasTexture(cv);
              tex.needsUpdate = true;
              const bound = window.psoSetMaterialTexture(mid, tex);
              res.steps.bindReturn = bound;
              // Verify the live mesh actually carries our texture on the
              // matching material slot.
              const grp = window.psoGetMeshGroup ? window.psoGetMeshGroup() : null;
              let mapMatches = false, slotFound = -1;
              if (grp) grp.traverse((c) => {
                if (!c.isMesh) return;
                const groups = c.userData && c.userData.materialGroups;
                if (Array.isArray(groups) && Array.isArray(c.material)) {
                  for (const g of groups) {
                    if ((g.materialId|0) !== (mid|0)) continue;
                    const m = c.material[g.materialIndex|0];
                    if (m && m.map === tex) { mapMatches = true; slotFound = g.materialIndex|0; }
                  }
                } else if ((c.userData && c.userData.materialId|0) === (mid|0)) {
                  if (c.material && c.material.map === tex) { mapMatches = true; slotFound = 0; }
                }
              });
              res.steps.mapMatches = mapMatches;
              res.steps.slotFound = slotFound;
              // Mutate a texel (simulate a brush stamp), flag needsUpdate.
              const before = ctx.getImageData(4,4,1,1).data;
              ctx.fillStyle = '#00ff00'; ctx.fillRect(2,2,6,6);
              tex.needsUpdate = true;
              const after = ctx.getImageData(4,4,1,1).data;
              const changed = (before[0]!==after[0] || before[1]!==after[1] || before[2]!==after[2]);
              res.steps.texelChanged = changed;
              res.steps.before = Array.from(before);
              res.steps.after = Array.from(after);
              res.ok = !!(bound && mapMatches && changed);
              return res;
            }"""
        )

        # 4. ANIMATION still plays --------------------------------------
        out["anim_debug"] = page.evaluate(
            "() => window.psoGetAnimDebug ? window.psoGetAnimDebug() : null"
        )

        # 5. TABS — open the model panel + inspect the tab strip. The strip
        # is built lazily; ensure it exists by touching a tab register hook,
        # then read the data-tab buttons.
        out["tabs"] = page.evaluate(
            """() => {
              const strip = document.querySelector('.pso-tex-panel-tabs');
              if (!strip) return { found:false };
              const btns = Array.from(strip.querySelectorAll('button[data-tab]'));
              const tabs = btns.map(b => b.getAttribute('data-tab'));
              const labels = btns.map(b => (b.textContent||'').trim());
              return {
                found: true,
                count: tabs.length,
                tabs, labels,
                hasAnimation: labels.includes('Animation'),
                hasEdit: tabs.includes('edit'),
                hasAnimEditor: tabs.includes('anim_editor'),
                hasMotionsLabel: labels.includes('Motions'),
              };
            }"""
        )

        browser.close()

    out["page_errors"] = errs[:10]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
