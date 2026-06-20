# psobb-studio

[![smoke](https://github.com/therealpixelated/psobb-studio/actions/workflows/smoke.yml/badge.svg)](https://github.com/therealpixelated/psobb-studio/actions/workflows/smoke.yml)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![platform](https://img.shields.io/badge/platform-windows-lightgrey.svg)](#setup--run)

> The **smoke** badge above is green only when the latest commit imports cleanly and the server boots on a fresh checkout. If it's red, `main` is broken — see [Continuous integration](#continuous-integration--smoke-test).

A web-based asset studio for **Phantasy Star Online: Blue Burst**. psobb-studio reads the asset files from a local PSOBB install, decodes the game's proprietary model, texture, and archive formats, and lets you view, edit, upscale, and re-pack them from the browser — then stage the rebuilt files back for testing. It pairs a FastAPI backend (`server.py` plus a `formats/` package of binary readers/writers) with a vanilla-JavaScript, Three.js frontend in `static/`. It is a hobbyist modding tool: you supply your own game files, and no game assets are bundled or distributed with it.

---

## Features

### Model viewing & export (NJ / NJM / XJ)
- **3D viewer** for chunk-Ninja `.nj` and descriptor/chunk `.xj` models, including inners extracted from BML and AFS containers. Triangulated, world-baked meshes are served as base64 Float32/Uint32 buffers and rendered with a PSOBB-matched Lambert shader (`static/model_viewer.js`, ~187 KB Three.js).
- **Skinned meshes & skeletons** — bone-local vertex payloads with bind-pose hierarchies for animation; per-bone keyframe motion from `.njm`.
- **Round-trip-preserving export** — encode models back to `.nj` and motions to `.njm` via `formats/nj_writer.py` and `formats/njm_writer.py`, preserving narrow/wide-angle and IFF-chunk metadata so re-encoded files match byte-for-byte where possible.
- **Composite bosses** — multi-part bosses (Dragon, De Rol Le, Vol Opt, Olga Flow) ship as one BML with several `.nj` inners; `/api/composite_bundle` assembles them with a hand-curated TRS placement table.
- **External import** — bring in `.obj` / `.gltf` / `.glb` / `.fbx`, convert to deployable `.nj` with axis-flip and skeleton-template substitution, and splice into a target BML inner.

### Texture decode & encode (XVR / PVR / XVM)
- **XVR/PVR decoding** of XVM texture archives into PNG tiles, with NJTL slot tables resolving model material IDs to texture tiles (`formats/xvr_decode.py`, `formats/pvr_decode.py`, `formats/njtl.py`).
- **Encode / re-pack** edited tiles back into XVM via the `xvr_codec` tool, including S3TC (BC1/BC3) re-encode through `quicktex`.
- **Atlas & viewport modes** — assemble multi-tile files into a single composite for context-aware editing (letters that cross tile seams stay continuous), or preview a 16:9 widescreen transform.

### Archives & compression (AFS / BML / PRS)
- **AFS** Sega archive reader/writer with magic-sniffing classification of inner blobs (`formats/afs.py`, `formats/afs_reader.py`).
- **BML** container parse/pack with an inner-PRS LRU cache; repack individual inner XVMs or NJs and atomic-deploy (`formats/bml.py`).
- **PRS** LZ-style decompress/compress, including an `compress_optimal` path for tight rebuilds (`formats/prs.py`).
- Sibling-archive resolution (BML ↔ AFS) and a cross-archive texture-name index.

### Item & battle-parameter editors
- **BattleParam** (`BattleParam_*.dat`) byte-exact JSON round-trip editor with a higher-level **mob AI DSL** authoring layer and shippable presets (`formats/battle_param.py`, `formats/mob_dsl.py`).
- **ItemPMT** (BB-V4 and legacy) PRS-compressed item-parameter editor with round-trip metadata (`formats/itempmt.py`).
- Both deploy into a newserv install behind a single in-flight deploy lock with timestamped backups.

### Editing panels
- **Paint** — flat and v5 layered texture painting (layers, masks, blend modes, opacity), composited server-side and re-packed into the host archive.
- **Sculpt** — sparse/dense vertex-displacement sculpting persisted as JSON sidecars and baked into an `.nj`.
- **Edit (protools)** — explicit per-vertex transform edits with sparse displacement storage.
- **Rig** — skeleton + per-vertex weights + IK targets, with distance-falloff and heat-diffusion auto-skin algorithms.
- **Animation** — NJM keyframe editor (insert/delete/save with Bézier handles), N-source motion **blend**, and glTF/FBX **retarget** onto a target skeleton with optional FABRIK IK and left/right mirror.
- **UV** — UV inspection alongside the mesh viewer.
- **Map** — map catalogue browser, terrain `.nj`/`.xj`/`.rel` rendering, and spawn/waypoint placement editing.
- **Material** — per-submesh material breakdown and edits (diffuse/alpha/blend/depth/two-sided) with a preset catalog.

### Upscaling pipeline
- **Real-ESRGAN (ncnn-vulkan)** super-resolution of texture tiles. Pick a model and scale; oddball scales cascade through supported steps. `keep_native_dims` Lanczos-downsamples back to the original dimensions afterward — required so the game engine can still load the rebuilt PRS.
- Upscale a **single tile**, the **full atlas composite** (full spatial context across tile seams), or **import** an externally upscaled PNG (e.g. an Upscayl run). Per-`(file,tile,model,scale,settings)` locks serialize duplicate requests.
- A **batch** endpoint runs the upscale across every tile of many assets at once.
- An optional **AI-generation** panel (`/api/aigen/*`) routes img2img / inpaint / text2img / ControlNet jobs to A1111, ComfyUI, or an in-process Diffusers provider.

---

## Architecture

```
~/Repositories/psobb-studio/
├── server.py            # FastAPI app — the HTTP API (137 routes)
├── manifest.py          # asset-tree walker + classifier (disk-cached)
├── atlas_layouts.py     # ground-truth per-file tile layouts
├── formats/             # binary readers/writers — one module per format
├── static/              # vanilla-JS + Three.js frontend (~36 modules)
├── aigen/               # optional AI image-gen providers (a1111, comfy, diffusers)
├── data/                # bundled non-asset data (import templates, mob presets)
├── tests/               # pytest unit + e2e + JSDOM frontend smoke
├── cache/               # layered LRU + on-disk caches and staged export dirs
├── requirements.txt     # Python runtime deps
└── package.json         # Node dep (jsdom) for frontend tests
```

**Backend (`formats/`).** Every PSOBB binary format has a dedicated module: `bml.py`, `afs.py`/`afs_reader.py`, `prs.py`, `nj_writer.py`/`njm.py`/`njm_writer.py`, `xj.py`/`xj_descriptor.py`, `njtl.py`/`xvr_decode.py`/`pvr_decode.py`, `rel.py`, `battle_param.py`/`itempmt.py`/`mob_dsl.py`, plus the editing layers `paint.py`, `sculpt.py`, `rigging.py`, `material.py`, `anim_retarget.py`/`anim_blend.py`, and import paths `import_external.py`/`fbx_reader.py`. New formats slot in by adding a module and wiring its read into the manifest classifier; writes hook the build/repack pipeline.

**Frontend (`static/`).** A single-page app served at `/`, built from focused panel modules (`model_viewer.js`, `texture_panel.js`, `paint_panel.js`, `sculpt_panel.js`, `rig_panel.js`, `anim_editor_panel.js`, `map_panel.js`, `battle_param_panel.js`, `itempmt_panel.js`, `mob_dsl_panel.js`, `import_panel.js`, …). It talks to the backend purely over the HTTP API and renders models with Three.js. The index HTML is served with cache-busted `?v=<sha8>` static URLs so browsers reload only changed assets.

**HTTP API.** ~137 localhost-only routes under `/api/*`, grouped by subsystem: manifest/asset listing, model rendering (`/api/model_mesh`, `/api/model_skinned`, `/api/model_bundle`, `/api/composite_bundle`), tiles & upscale (`/api/tiles`, `/api/upscale`, `/api/atlas_upscale`), repack/deploy (`/api/repack*`, `/api/deploy/*`, `/api/build_*`), the editor panels (`/api/paint/*`, `/api/sculpt/*`, `/api/rig/*`, `/api/protools/*`, `/api/anim_*`, `/api/map/*`, `/api/material/*`), the server-side editors (`/api/battle_param/*`, `/api/itempmt/*`, `/api/mob_dsl/*`), AI generation (`/api/aigen/*`), and live-test / cache / SSE debug routes. Path traversal is rejected at the resolver layer and POST bodies are size-capped. See **`API_REFERENCE.md`** for the per-route reference and **`ARCHITECTURE.md`** for the cache hierarchy, lock topology, and concurrency model.

The editor reads from a dev data dir (shadowing the live install) and writes only there until you explicitly promote a rebuilt file into the playable install.

---

## Setup & run

**Prerequisites**
- Python 3.11
- Node.js (only needed to run the JSDOM-based frontend tests)
- A PSOBB install whose `data/` directory you point the editor at
- *(optional)* `realesrgan-ncnn-vulkan`, the `xvr_codec`/`puyo` tools, and AI providers if you want upscaling / AI generation — `GET /api/health` reports which external tools resolved

**Install**

The project is a standard PEP 621 package (`pyproject.toml`). In a fresh virtualenv:

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -e .            # runtime deps + the `psobb-studio` command
# or: pip install -e ".[dev]"   (adds pytest + ruff)
# or, with make:  make install
# or, with poetry:  poetry install
```

Dependencies are declared and pinned in `pyproject.toml` (`requirements.txt` is the matching lock used by CI). The heavy AI-generation stack (`torch`/`diffusers`/`transformers`) is the optional `[ai]` extra — only needed for the local Diffusers provider.

**Run**

Point it at a PSOBB `data/` directory and launch — via the console entry, `make`, or raw uvicorn:

```bash
psobb-studio --port 8765 --data-dir "$HOME/PSOBB.IO/data"
# or:  make run PORT=8765 DATA="$HOME/PSOBB.IO/data"
# or:  PSO_DATA_DIR="$HOME/PSOBB.IO/data" uvicorn server:app --port 8765
```

Then open <http://127.0.0.1:8765>. All routes are localhost-only by design — the server is an unauthenticated sidecar, so do not expose it to a network.

Before pushing, run the smoke check (the same one CI runs): `make smoke` (or `python scripts/smoke_test.py`).

---

## Continuous integration & smoke test

Every push and pull request runs a **fresh-checkout smoke test** ([`.github/workflows/smoke.yml`](.github/workflows/smoke.yml) — the **smoke** badge at the top of this file). On a clean checkout of *exactly what is committed*, it:

1. **imports every first-party module** (`server`, `manifest`, `atlas_layouts`, all of `formats/*` and `aigen/*`) — so a source file that is missing from the commit (e.g. swallowed by a `.gitignore` rule) fails CI instead of reaching `main`; and
2. **boots the server** and requires `GET /api/health` to answer `200`, then shuts it down.

Optional heavy dependencies (`torch`, `diffusers`, …) are tolerated when absent; only missing *first-party* source is a hard failure.

Run the identical check locally before you push:

```bash
python scripts/smoke_test.py
```

A green **smoke** badge means the committed tree imports and the server starts. A red badge means a clean clone is broken — do not assume "works on my machine," since local files that are git-ignored still import locally but are absent from the commit.

---

## Format correctness

The binary codecs in `formats/` are validated and cross-checked against established, community reverse-engineering work rather than guessed. Decoders/encoders are ported or verified byte-for-byte against the reference implementations below, and round-trip tests assert that re-encoded files reproduce the originals where the format allows. Credit and thanks to:

- **VrSharp / PuyoTools** — Nick Woronekin — PVR/GVR/SVR (and the broader Puyo Tools archive/texture suite). <https://github.com/nickworonekin/puyotools>
- **SA3D.Modeling & X-Hax** — Ninja (NJ/NJM/XJ) model & motion structures. <https://github.com/X-Hax/SA3D.Modeling> / <https://github.com/X-Hax>
- **phantasmal-world** — PSOBB asset/format research, including multi-inner texture-id offsets. <https://github.com/DaanVandenBosch/phantasmal-world>
- **pvr2image** — PVR texture decode reference. <https://github.com/yevgeniy-logachev/pvr2image>
- **libpsoarchive** — PSO archive (AFS/PRS) handling. <https://github.com/Sylverant/libpsoarchive>
- **DashGL (ikaruga / psov2)** — PSO Ninja model rendering references. <https://gitlab.com/dashgl/ikaruga> / <https://gitlab.com/dashgl/psov2>
- **Solybum PSO-Tools** — PRS and multi-game Sega tooling. <https://github.com/Solybum/PSO-Tools>
- **njaPatcher / nja-gen** — Ninja motion/animation tooling references.

---

## References & credits

Broader tooling, servers, and documentation that informed this project:

**Models (NJ/NJM/XJ/GJ)**
- Aqua Toolset / PSO2-Aqua-Library — Shadowth117: <https://github.com/Shadowth117/Aqua-Toolset>, <https://github.com/Shadowth117/PSO2-Aqua-Library>
- Blender-NaomiLib (Naomi/Dreamcast models): <https://github.com/NaomiMod/blender-NaomiLib>
- pso_gc_tools — gered: <https://github.com/gered/pso_gc_tools>
- pso-utils — choogiesaur: <https://github.com/choogiesaur/pso-utils>

**Archives & compression (AFS/GSL/BML/PRS)**
- PSOBMLExtract — Shadowth117: <https://github.com/Shadowth117/PSOBMLExtract>
- prsutil — essen / fuzziqer: <https://github.com/essen/prsutil>
- newserv — fuzziqersoftware (PSO server + RE tools, ItemPMT/rare-table conversion): <https://github.com/fuzziqersoftware/newserv>
- GCFT — LagoLunatic: <https://github.com/LagoLunatic/GCFT>

**Quests (QST/DAT/BIN)**
- Sylverant pso_tools (qst_tool): <https://github.com/Sylverant/pso_tools>
- YAQP — jtuu: <https://github.com/jtuu/yaqp>
- psogc_quest_tool — gered: <https://github.com/gered/pso_gc_tools>

**Items & battle parameters**
- PSO Battle Parameter Editor — johndellarosa: <https://github.com/johndellarosa/pso_battle_parameter_editor>
- Battle-Param-Editor — tofuman0: <https://github.com/tofuman0/Battle-Param-Editor>
- newserv ItemPMT tooling: <https://github.com/fuzziqersoftware/newserv>

**Servers**
- Sylverant: <https://github.com/Sylverant>
- newserv: <https://github.com/fuzziqersoftware/newserv>

**Indexes & wikis**
- Awesome PSO — tcardlab: <https://github.com/tcardlab/awesome-pso>
- PSO Dev Wiki: <http://psodevwiki.sharnoth.com/>

This project stands on the shoulders of the PSO modding and reverse-engineering community. Any omission above is unintentional — corrections welcome.

---

## Contributing

The backend ships with ~494 pytest unit tests, a 121-step end-to-end script, and a JSDOM frontend smoke test.

```bash
# Python unit + integration tests
pip install -r requirements.txt
python -m pytest tests/

# End-to-end script (drives a real asset round-trip)
python e2e_test.py

# Frontend smoke test (needs Node + jsdom)
npm install
node tests/test_autoplay_jsdom.mjs
```

When adding an endpoint, follow the conventions in `ARCHITECTURE.md` ("Adding a new endpoint"): resolve user paths only through the safe resolvers (`safe_data_path`, `_resolve_under_roots`, `_safe_archive_name`, `_validate_bare_filename`), cap POST bodies with `_enforce_body_size`, use the `tmp.write_bytes(...)` + `os.replace(...)` atomic-write pattern for any write that lands in a real data dir, gate expensive idempotent work behind a per-key lock, and add a unit test (plus an e2e step for user-visible flows). A new binary format is a new module in `formats/` wired into the manifest classifier and the build/repack pipeline.

---

## Legal

psobb-studio is an unofficial, non-commercial hobbyist tool created by and for the PSOBB modding community. It is **not** affiliated with, endorsed by, or sponsored by SEGA or Sonic Team. Phantasy Star Online and Blue Burst are trademarks of their respective owners.

**This repository contains no SEGA or PSOBB game assets, and you must not commit or distribute any.** psobb-studio operates only on game files that **you** already legally own and supply from your own installation (via `PSO_DATA_DIR`). Do not check game data — textures, models, archives, parameter files, or any extracted/derived asset — into this or any public repository, and do not redistribute it. The tool reads from your local install and writes rebuilt files back to directories you control; what you do with those files is your responsibility. Use it only with content you are licensed to modify, and keep your edited assets to yourself.
