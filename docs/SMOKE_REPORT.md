# Model render smoke report

Generated: 2026-06-21T16:51:08.365Z

Harness: `scripts/smoke_render_all.mjs` — replays model_viewer.js `openByPath()` for every manifest model entry (real `parseNinjaModel` under three@0.160.0 for the .nj psov2 path; `/api/model_mesh` // `/api/model_skinned` // `/api/composite_bundle` for the server paths).

## Totals

| metric | value |
|---|---|
| models tested | **2863** |
| real mesh (ok) | **2767** (96.6%) |
| &nbsp;&nbsp;of which empty (0 verts, no cube banner) | 449 |
| grey cube | **96** (3.35%) |
| sweep wall time | 22.7s |

> "ok" mirrors the browser's cube indicator: a model is **ok** when `openByPath` does NOT raise the "primitive (cube) — model unavailable" banner. 449 of the ok models parse to a 0-vertex stub (psov2 succeeds with bones but no geometry — e.g. `ene_common_all.nj`); the browser shows no cube for these but nothing renders. They are listed in the Empty-models section below.

## Load-time distribution (per-model route ms)

| stat | ms |
|---|---|
| median | 10.14 |
| p90 | 33.17 |
| p99 | 208.18 |
| max | 6415.73 |
| count > 50ms | **135** (4.7%) |

Per-type load timing:

| type | n | ok | cube | median ms | p90 ms | >50ms |
|---|---|---|---|---|---|---|
| afs#inner_nj | 1212 | 1212 | 0 | 10.7 | 32.6 | 15 |
| bml#inner_nj | 667 | 667 | 0 | 6.2 | 18.0 | 20 |
| bml_toplevel | 364 | 363 | 1 | 23.4 | 92.5 | 95 |
| bml#inner_xj | 316 | 316 | 0 | 10.1 | 19.3 | 4 |
| bare_nj | 303 | 209 | 94 | 4.0 | 10.2 | 1 |
| other | 1 | 0 | 1 | 7.7 | 7.7 | 0 |

## Cube failures grouped

### By entry type

| type | cubes | examples |
|---|---|---|
| bare_nj | 94 | `scene/map_aancient01_00s.nj`<br>`scene/map_acave01_00s.nj`<br>`scene/map_acave01_01s.nj`<br>`scene/map_acave01_02s.nj`<br>`scene/map_acave01_03s.nj` |
| bml_toplevel | 1 | `NpcApcMot.bml` |
| other | 1 | `npcplayerchar.dat` |

### By error signature

| count | signature | examples |
|---|---|---|
| 94 | `mesh: invalid path (path components forbidden)` | `scene/map_aancient01_00s.nj`<br>`scene/map_acave01_00s.nj`<br>`scene/map_acave01_01s.nj`<br>`scene/map_acave01_02s.nj`<br>`scene/map_acave01_03s.nj` |
| 1 | `bml has no .nj/.xj inner` | `NpcApcMot.bml` |
| 1 | `mesh: unsupported model extension '…' (expected .nj, .xj, .bml, or .afs)` | `npcplayerchar.dat` |

### By archive family (top 20)

| count | archive | examples |
|---|---|---|
| 94 | `scene/` | `scene/map_aancient01_00s.nj`<br>`scene/map_acave01_00s.nj`<br>`scene/map_acave01_01s.nj` |
| 1 | `NpcApcMot.bml` | `NpcApcMot.bml` |
| 1 | `npcplayerchar.dat` | `npcplayerchar.dat` |

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

135 models loaded slower than 50ms. Top 25:

| ms | route | type | path |
|---|---|---|---|
| 6415.73 | skinned | afs#inner_nj | `ItemModelEp4.afs#0303_ItemModelEp4_0303.nj` |
| 6207.07 | skinned | afs#inner_nj | `ItemModelEp4.afs#0302_ItemModelEp4_0302.nj` |
| 6191.82 | psov2 | afs#inner_nj | `ItemModelEp4.afs#0305_ItemModelEp4_0305.nj` |
| 6186.81 | psov2 | afs#inner_nj | `ItemModelEp4.afs#0304_ItemModelEp4_0304.nj` |
| 5153.83 | psov2 | bml_toplevel | `onlineending_dat.bml` |
| 5141.5 | psov2 | bare_nj | `plAbdy00.nj` |
| 1375.62 | composite | bml_toplevel | `darkfalz_dat.bml` |
| 953.58 | composite | bml_toplevel | `bm_ene_boss09.bml` |
| 934.64 | composite | bml_toplevel | `bm_boss3_volopt.bml` |
| 857.06 | composite | bml_toplevel | `boss06_plotfalz_dat.bml` |
| 753.92 | composite | bml_toplevel | `bm_boss3_volopt_ap.bml` |
| 667.31 | composite | bml_toplevel | `plDnj.bml` |
| 623.54 | mesh | bml#inner_xj | `bm_o_abe_butterfly.bml#abecyou_fe_obj001_cyou.xj` |
| 558.16 | composite | bml_toplevel | `bm_ene_df3_dimedian.bml` |
| 557.98 | composite | bml_toplevel | `bm_boss2_de_rol_le_a.bml` |
| 482.64 | psov2 | bml_toplevel | `bm_ene_dkflower.bml` |
| 481.49 | mesh | bml#inner_xj | `fe_obj_hashi.bml#fe_obj_hashi.xj` |
| 447.42 | psov2 | bml#inner_nj | `plDnj.bml#plDhai03.nj` |
| 436.4 | mesh | bml_toplevel | `fe_obj_kaifuku_moto_2.bml` |
| 391.03 | mesh | bml#inner_xj | `bm_boss3_volopt.bml#fe_obj_vo_mo_sho01_ao.xj` |
| 366.3 | skinned | afs#inner_nj | `ItemModel.afs#0317_ItemModel_0317.nj` |
| 330.14 | skinned | bml_toplevel | `bm_boss8_dragon.bml` |
| 311.81 | mesh | bml_toplevel | `bm_o_door_seabed01.bml` |
| 287.41 | composite | bml_toplevel | `bm_ene_df3_dimedian_a.bml` |
| 269.55 | composite | bml_toplevel | `pm_mdl.bml` |

> Note: per-model ms is measured in-process (psov2 parse + server fetches), which the browser parallelises with renderer setup + texture/motion fetches. Cached re-opens in the live editor are ~4ms (the psov2 LRU); these are COLD loads.
>
> The handful of multi-second outliers split into (a) **genuine** heavy cold server parses — the `ItemModelEp4.afs` skinned inners reproduce at ~6s even in isolation and are a real server-side optimisation target; and (b) **sweep-startup stalls** — the first models scheduled (alphabetical, e.g. `biri_ball.bml`) occasionally absorb a one-time worker/JIT warm-up spike (`biri_ball.bml#biri_ball.nj` measures ~9ms when re-run in isolation), which inflates max/p99 but not the median/p90. Re-run a flagged path alone with `--paths-file` to distinguish the two.
