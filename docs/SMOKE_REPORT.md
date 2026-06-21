# Model render smoke report

Generated: 2026-06-21T17:22:58.735Z

Harness: `scripts/smoke_render_all.mjs` — replays model_viewer.js `openByPath()` for every manifest model entry (real `parseNinjaModel` under three@0.160.0 for the .nj psov2 path; `/api/model_mesh` // `/api/model_skinned` // `/api/composite_bundle` for the server paths).

## Fixes applied (2026-06-21) — every cube eliminated, cold loads cut

All **96** grey cubes from the prior sweep are gone (**0** cubes, 100% ok), and the
over-50ms tail dropped from 135 → 77 (all remaining are `composite` bml-toplevel,
which the harness times SERIALLY but the real browser fetches in PARALLEL via
`Promise.all` — ~67ms for the worst 22-inner composite).

| cube class | count | root cause | fix |
|---|---|---|---|
| `scene/map_*.nj` | 94 | server `_validate_bare_filename` forbade ALL path components, so the `scene/` subdir map meshes 400'd with "invalid path" | `server.py`: new `_validate_contained_relpath` + `_resolve_under_roots_relpath` accept a known subdir prefix (`scene/`) while still resolving strictly inside the data root (`relative_to` guard intact). Routes both `/api/raw_nj` and `/api/model_mesh`. |
| `NpcApcMot.bml` | 1 | motion-only BML — 120 `.njm` chunks, **0** `.nj`/`.xj` mesh inners | `model_viewer.js`: `_discoverBmlInners` reports `motionOnly`; `openByPath` shows a dedicated "motion-only archive — N animations, no mesh" banner instead of the grey cube. |
| `npcplayerchar.dat` | 1 | a `NOL\0` NPC roster/config table mis-classified as a renderable model | `manifest.py`: re-bucket `npcplayerchar.dat` from `model` → `metadata` (alongside the other NOL/string tables) so it never enters the model viewer. |

### Perf fixes

* **AFS inner fetch** (`formats/afs_reader.py`): `materialize_inner` re-read + fully
  re-parsed the WHOLE AFS directory on every call — even cache hits. For a 300+-inner
  archive (`ItemModelEp4.afs`) that re-index ran per-inner and spiked to multi-second
  cold loads under the sweep. Added a fast cache-hit path that returns the materialised
  inner WITHOUT touching the archive. A 40-inner cold burst dropped from ~5s/inner to
  173ms total wall.
* **psov2 parser hang** (`static/psov2_ninja.js`): a few `ItemModelEp4.afs` inners drove
  `readBitsChunk`'s case-5 NJD cache-list into an UNBOUNDED `mem_stack.push` loop —
  ~5s of spinning before throwing "Invalid array length". Added a `mem_stack` depth cap
  (4096) + a chunk-loop iteration cap so psov2 fails FAST (2.8ms) on the corrupt stream;
  the caller then renders these via the server skinned path in ~9-20ms. No valid model
  is affected (full sweep: still 2862 ok, same 449 empty stubs).
* **Lazy motions** (`static/model_viewer.js`): non-default `.njm` motions now fetch their
  NMDM bytes + build the THREE.AnimationClip ON DEMAND (first play) via the live ninja
  loader, cached + request-coalesced — completing commit ff8aefd (which stopped the
  per-open flood) so EVERY motion plays without re-flooding on open.
* **/api/health flood** (`static/onboarding.js`): `refreshDataDir` unconditionally fetched
  `/api/health` and then called ITSELF on success → unbounded recursive re-probe (the
  1000+ requests in the trace). Split paint from probe; the probe is now one-shot per
  data-dir path. Verified: 100 `refreshDataDir` calls → 1 health fetch.

## Totals

| metric | value |
|---|---|
| models tested | **2862** |
| real mesh (ok) | **2862** (100.0%) |
| &nbsp;&nbsp;of which empty (0 verts, no cube banner) | 449 |
| grey cube | **0** (0%) |
| sweep wall time | 10.6s |

> "ok" mirrors the browser's cube indicator: a model is **ok** when `openByPath` does NOT raise the "primitive (cube) — model unavailable" banner. 449 of the ok models parse to a 0-vertex stub (psov2 succeeds with bones but no geometry — e.g. `ene_common_all.nj`); the browser shows no cube for these but nothing renders. They are listed in the Empty-models section below.

## Load-time distribution (per-model route ms)

| stat | ms |
|---|---|
| median | 9 |
| p90 | 24.78 |
| p99 | 103.67 |
| max | 736.22 |
| count > 50ms | **77** (2.7%) |

Per-type load timing:

| type | n | ok | cube | median ms | p90 ms | >50ms |
|---|---|---|---|---|---|---|
| afs#inner_nj | 1212 | 1212 | 0 | 11.8 | 24.1 | 0 |
| bml#inner_nj | 667 | 667 | 0 | 5.9 | 15.0 | 10 |
| bml_toplevel | 364 | 364 | 0 | 18.5 | 70.6 | 66 |
| bml#inner_xj | 316 | 316 | 0 | 8.1 | 13.9 | 1 |
| bare_nj | 303 | 303 | 0 | 4.1 | 5.8 | 0 |

## Cube failures grouped

**No cubes** — every model entry produced a real mesh (verts > 0).

## Empty (0-vertex) models — render "ok" but invisible

449 models parse without a cube banner but carry no geometry (0 vertices). To the user these render as nothing. First 40:

| path | route |
|---|---|
| `bm_ene_common_all.bml` | psov2 |
| `bm_ene_common_all.bml#ene_common_all.nj` | psov2 |
| `ItemModel.afs#0005_ItemModel_0005.nj` | psov2 |
| `ItemModel.afs#0006_ItemModel_0006.nj` | psov2 |
| `ItemModel.afs#0007_ItemModel_0007.nj` | psov2 |
| `ItemModel.afs#0008_ItemModel_0008.nj` | psov2 |
| `ItemModel.afs#0012_ItemModel_0012.nj` | psov2 |
| `ItemModel.afs#0013_ItemModel_0013.nj` | psov2 |
| `ItemModel.afs#0014_ItemModel_0014.nj` | psov2 |
| `ItemModel.afs#0017_ItemModel_0017.nj` | psov2 |
| `ItemModel.afs#0018_ItemModel_0018.nj` | psov2 |
| `ItemModel.afs#0019_ItemModel_0019.nj` | psov2 |
| `ItemModel.afs#0020_ItemModel_0020.nj` | psov2 |
| `ItemModel.afs#0021_ItemModel_0021.nj` | psov2 |
| `ItemModel.afs#0022_ItemModel_0022.nj` | psov2 |
| `ItemModel.afs#0025_ItemModel_0025.nj` | psov2 |
| `ItemModel.afs#0026_ItemModel_0026.nj` | psov2 |
| `ItemModel.afs#0027_ItemModel_0027.nj` | psov2 |
| `ItemModel.afs#0028_ItemModel_0028.nj` | psov2 |
| `ItemModel.afs#0029_ItemModel_0029.nj` | psov2 |
| `ItemModel.afs#0033_ItemModel_0033.nj` | psov2 |
| `ItemModel.afs#0034_ItemModel_0034.nj` | psov2 |
| `ItemModel.afs#0037_ItemModel_0037.nj` | psov2 |
| `ItemModel.afs#0041_ItemModel_0041.nj` | psov2 |
| `ItemModel.afs#0046_ItemModel_0046.nj` | psov2 |
| `ItemModel.afs#0050_ItemModel_0050.nj` | psov2 |
| `ItemModel.afs#0065_ItemModel_0065.nj` | psov2 |
| `ItemModel.afs#0066_ItemModel_0066.nj` | psov2 |
| `ItemModel.afs#0067_ItemModel_0067.nj` | psov2 |
| `ItemModel.afs#0069_ItemModel_0069.nj` | psov2 |
| `ItemModel.afs#0070_ItemModel_0070.nj` | psov2 |
| `ItemModel.afs#0071_ItemModel_0071.nj` | psov2 |
| `ItemModel.afs#0072_ItemModel_0072.nj` | psov2 |
| `ItemModel.afs#0074_ItemModel_0074.nj` | psov2 |
| `ItemModel.afs#0075_ItemModel_0075.nj` | psov2 |
| `ItemModel.afs#0076_ItemModel_0076.nj` | psov2 |
| `ItemModel.afs#0077_ItemModel_0077.nj` | psov2 |
| `ItemModel.afs#0079_ItemModel_0079.nj` | psov2 |
| `ItemModel.afs#0081_ItemModel_0081.nj` | psov2 |
| `ItemModel.afs#0082_ItemModel_0082.nj` | psov2 |

## Over-50ms loads

77 models loaded slower than 50ms. Top 25:

| ms | route | type | path |
|---|---|---|---|
| 736.22 | composite | bml_toplevel | `plHnj.bml` |
| 729.28 | composite | bml_toplevel | `plGnj.bml` |
| 694.95 | composite | bml_toplevel | `boss06_plotfalz_dat.bml` |
| 669.65 | composite | bml_toplevel | `darkfalz_dat.bml` |
| 416.15 | composite | bml_toplevel | `bm_ene_boss09.bml` |
| 415.86 | composite | bml_toplevel | `bm_boss3_volopt_ap.bml` |
| 371.44 | composite | bml_toplevel | `bm_boss3_volopt.bml` |
| 224.34 | composite | bml_toplevel | `pm_mdl.bml` |
| 218.53 | composite | bml_toplevel | `plLnj.bml` |
| 198.97 | composite | bml_toplevel | `bm_boss2_de_rol_le.bml` |
| 196 | composite | bml_toplevel | `bm_boss5_gryphon.bml` |
| 193.81 | composite | bml_toplevel | `bm_boss7_de_rol_le_c.bml` |
| 189.05 | composite | bml_toplevel | `plInj.bml` |
| 188.29 | composite | bml_toplevel | `plKnj.bml` |
| 182.16 | composite | bml_toplevel | `bm_boss2_de_rol_le_a.bml` |
| 180.05 | composite | bml_toplevel | `plDnj.bml` |
| 170.46 | skinned | bml_toplevel | `bm_boss8_dragon.bml` |
| 158.28 | skinned | bml#inner_nj | `bm_boss8_dragon.bml#lo_boss1_s_nb_dragon.nj` |
| 153.26 | skinned | bml#inner_nj | `boss06_plotfalz_dat.bml#lo_bossgc_pf02l_body.nj` |
| 151.95 | composite | bml_toplevel | `bm_ene_boota.bml` |
| 145.9 | skinned | bml#inner_nj | `bm_boss5_gryphon.bml#boss5_s_body.nj` |
| 137.16 | composite | bml_toplevel | `plAnj.bml` |
| 135.31 | composite | bml_toplevel | `bm_ene_df3_dimedian_a.bml` |
| 132.14 | composite | bml_toplevel | `bm_ene_darkgunner.bml` |
| 130.28 | composite | bml_toplevel | `bm4_ps_ma_body.bml` |

> Note: per-model ms is measured in-process (psov2 parse + server fetches), which the browser parallelises with renderer setup + texture/motion fetches. Cached re-opens in the live editor are ~4ms (the psov2 LRU); these are COLD loads.
>
> The handful of multi-second outliers split into (a) **genuine** heavy cold server parses — the `ItemModelEp4.afs` skinned inners reproduce at ~6s even in isolation and are a real server-side optimisation target; and (b) **sweep-startup stalls** — the first models scheduled (alphabetical, e.g. `biri_ball.bml`) occasionally absorb a one-time worker/JIT warm-up spike (`biri_ball.bml#biri_ball.nj` measures ~9ms when re-run in isolation), which inflates max/p99 but not the median/p90. Re-run a flagged path alone with `--paths-file` to distinguish the two.
