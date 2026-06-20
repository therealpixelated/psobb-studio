"""PSOBB Studio - FastAPI backend.

Endpoints:
  GET  /api/health                      service health + tool resolution status
  GET  /api/files                       list .prs/.xvm files in PSOBB.IO/data/
  GET  /api/tiles/{filename}            extract & return tile metadata + base64 PNGs
  GET  /api/tile_png/{filename}/{idx}   serve raw extracted tile PNG (no base64)
  POST /api/upscale                     run realesrgan-ncnn-vulkan on a tile, return base64 PNG
  POST /api/import_png/{file}/{idx}     upload an external PNG (e.g. Upscayl) as a tile edit
  POST /api/repack                      repack edited tiles into PRS, optionally deploy
                                        OR build-only (deploy=false) for token-protected download
  POST /api/repack_diff                 dry-run summary: what a repack would touch
  POST /api/restore_backup              restore the most recent .pre_editor_* backup for a file
  GET  /api/export/{token}              download a previously-built artifact (deploy=false output)
  GET  /api/deploy/config               return DEV_DATA_DIR + LIVE_DATA_DIR
  GET  /api/deploy/diff                 diff between dev mirror and live game install
  POST /api/deploy/promote              copy named files from dev -> live with backup
  GET  /api/verify/{filename}           per-tile bit-identity check vs cached source
  GET  /api/atlas/{filename}            return composite layout + assembled PNG b64
                                        (404 if file has no known atlas layout)
  POST /api/atlas_upscale               upscale the assembled composite, slice back
                                        to per-tile cache PNGs (one job → many edits)
  POST /api/atlas_import                accept a user-supplied composite PNG (e.g. from
                                        external Upscayl), slice back same as atlas_upscale
  GET  /api/viewport/{filename}         16:9 viewport transform: layout placements
                                        on a 1278x768 canvas + composite PNG b64
                                        (works for atlas-known and unknown files)
  POST /api/viewport_paint              accept a hand-painted 1278x768 PNG, slice each
                                        placement back to native tile dim, register
                                        as upscaled-equivalent edits
  GET  /api/models                      list available ncnn upscaler models
  GET  /                                serve index.html

Hardening (2026-04-24):
  - Path-injection guard everywhere (safe_data_path).
  - Subprocess timeouts (60s default; upscale gets longer).
  - Cache cleanup on startup (orphans + >24h old subdirs).
  - Per-(file,tile,model,scale,...) lock around upscale to prevent races.
  - Per-file lock around repack.
  - Backup naming uses _<YYYYMMDD_HHMMSS> to avoid same-day collisions.
  - XVM round-trip path verified separately from PRS (xvr_codec rebuild only).
  - /api/repack with deploy:false returns rebuilt artifact size + path so caller
    can sanity-check before committing.

V3 (2026-04-24):
  - /api/upscale gains optional fields: tile_size, tta, gpu_id.
  - Cascade upscaling: scale > 4 chains 4x passes (e.g. 8 = 4*2, 16 = 4*4).
  - /api/models returns native_scale, max_scale, supports_tta, description.
  - Backwards-compatible: existing fields untouched.

V4 quality (2026-04-24):
  - xvr_codec.py extract now writes <name>.src.md5 sidecars, and rebuild
    splices the original .xvr bytes verbatim for any tile whose PNG
    matches its sidecar md5. Untouched tiles are bit-identical through
    the full PRS->XVM->PRS round-trip; only modified tiles re-encode.
    Quantified loss avoided per cycle: ~80-90 dB PSNR on tiles with
    actual content (see REPACK_ANALYSIS.md).
  - /api/repack reports `spliced_count` / `reencoded_count` / `changed_indices`.
  - /api/repack with deploy=false now mints an `export_token` and
    `export_url` (GET /api/export/<token>) so callers can build an
    artifact for download WITHOUT touching DATA_DIR. Tokens expire
    after EXPORT_TTL_SECONDS or on server restart.
  - GET /api/verify/<filename> reads the live deployed file and reports
    per-tile bit-identity vs the cached extract, plus aggregate.

Code-quality pass (2026-04-25):
  - Structured logging via stdlib `logging` (startup banner kept on stdout).
  - Bounded LRU cap on per-file lock dictionaries (no unbounded growth).
  - Request-body size limits enforced on every POST endpoint.
  - Cache cleanup robust against partial / dirty extractions.
  - Magic numbers extracted to named constants (XVM offsets, body limits, ...).
  - Specific exception types replace bare excepts.
  - Atomic manifest.json writes (tmp + os.replace).
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

# Audit S-1 (2026-05-01): cap PIL decode size to defang decompression-bomb
# PNGs. 50 MP fits any legit PSOBB tile at 8x (e.g. 4096x4096 = 16.7 MP)
# with comfortable headroom; anything bigger is either pathological or a
# deliberate bomb. Pillow now raises Image.DecompressionBombError instead
# of merely warning.
Image.MAX_IMAGE_PIXELS = 50_000_000

import atlas_layouts
import manifest as manifest_mod
from formats.iff import parse_iff
from formats.afs import parse_afs, write_afs
from formats.bml import (
    BmlPackEntry,
    COMPRESSION_NONE as BML_COMPRESSION_NONE,
    COMPRESSION_PRS as BML_COMPRESSION_PRS,
    DATA_ALIGNMENT_HAS_TEX as BML_ALIGN_HAS_TEX,
    DATA_ALIGNMENT_NO_TEX as BML_ALIGN_NO_TEX,
    extract_bml,
    extract_bml_texture,
    pack_bml,
    parse_bml,
)
# Inner-discovery validator (Wave 2 / Agent B, 2026-04-26). Returns the
# audited expected ``.nj`` + ``.xj`` count for a BML filename, or None
# when no ground truth is recorded. Used by api_bml_list to log a WARN
# (never fail-closed) on walker regressions.
from formats.psobb_engine_tables import expected_bml_inner_count
# Battle params (Investigation 1+2 server target). Standalone so the
# import doesn't pull in heavyweight model parsers when only the
# battle-param endpoints are reached.
from formats import battle_param as bp_mod
# ItemPMT (V4 BB item parameter table). Standalone import; the parser
# handles its own PRS round-trip via formats.prs.
from formats import itempmt as ipmt_mod
from formats import prs as prs_mod
# PVR (Sega Dreamcast PowerVR) decoder — wired into extract_tiles so
# .pvr / .pvm / .gvm sources (and PRS-wrapped variants) produce per-tile
# PNGs even though the legacy xvr_codec path only handles XVMH/XVRT.
# See formats/pvr_decode.py for the supported pixel/tex-mode coverage.
from formats import pvr_decode as _pvr_decode
# Sibling-archive discovery — magic-sniffed companion files next to a
# model. Used as a last-resort fallback by _build_model_texture_binding
# when neither the inline XVMH nor cross-archive lookup finds the
# texture's tile.
from formats import sibling_archives as _sibling_archives
# In-process XVMH/XVRT extractor (2026-04-26). Replaces the
# subprocess-based xvr_codec.py extract path on the hot extract_tiles
# code path. Removes the ~150 ms/archive Python-startup cost; for
# multi-texture XBOX assets that was the dominant cold-load latency.
# xvr_codec.py is still the canonical reference and remains the
# canonical *rebuild* path (the round-trip md5-splice logic lives there).
from formats import xvr_decode as _xvr_decode
# In-process PRS (Sega LZ77) decompressor — verified byte-exact against
# newserv/libpsoarchive. Replaces the dead PuyoToolsCli.exe subprocess on the
# .prs tile path (every .prs texture was 500ing on the missing binary).
from formats import prs as _prs
# Sculpt deltas (2026-04-25). Per-vertex displacement persistence + brush
# math (encode/decode + apply_displacement_to_payload). The frontend
# does the live-stroke math itself; the server only handles save / fetch.
from formats import sculpt as sculpt_mod
# Rigging (2026-04-25). Skeleton edits + per-vertex bone weights + IK
# targets persistence. Like sculpt_mod, frontend (rig_panel.js) drives
# live interaction; server validates + persists JSON sidecars and
# computes auto-skin weights / FABRIK IK on demand.
from formats import rigging as rigging_mod
from formats.njtl import (
    find_and_parse_njtl,
)
# Cross-BML texture lookup. PSOBB.IO ships ~60 NJTL refs whose name
# resolves only in a SIBLING BML's inline XVM (e.g. ts008_siro is
# referenced by Vol Opt's monitor sub-parts but lives in the boss-cylinder
# room BML). The texture index walks every BML once on first use,
# caches the (name -> [(bml, inner, xvr_index), ...]) map, and lets
# _build_model_texture_binding resolve cross-BML refs at bind time.
from formats import texture_index as _texture_index
from formats.xj import (
    parse_nj_file as _xj_parse_nj_file,
)
# Composite multi-inner BML placement table (2026-04-30). Curated
# per-part TRS + parent linkage for multi-part bosses (De Rol Le,
# Dragon, ...) so /api/composite_bundle can return a coherent layout
# instead of stacking every inner at the world origin. Identity-
# fallback for unknown BMLs.
from formats.composite_assembly import (
    CompositeAssembly,
    CompositePart,
    lookup_composite as _lookup_composite_assembly,
)
# Descriptor-table XJ parser. Different format from the chunk-based
# Ninja-Nj that ``formats/xj.py`` handles (see formats/xj_descriptor.py
# header). Of the 656 BML-inner models, ~263 are descriptor-table .xj
# and 393 are chunk-Nj .nj. The /api/model_mesh endpoint dispatches on
# the inner-file extension to pick the right parser.
from formats.xj_descriptor import parse_xj_file as _xj_parse_xj_descriptor_file
# Phase D Win 4 parsed-mesh LRU. Wraps the four parsers above so that
# repeated opens of the same model (variant picker, motion preview,
# paint, sculpt) hit a parsed-object cache instead of re-walking the NJ
# chunk stream. Cold parse populates an in-memory LRU + on-disk pickle;
# warm parse returns in <5 ms. See formats/parse_cache.py for layering.
from formats import parse_cache as _parse_cache
# NJM (Ninja Motion) parser — used by /api/animations and
# /api/animation_data to surface skeletal-animation keyframes for the
# model viewer's playback. Keyframes are returned in BAMS for rotation
# (raw Sega Ninja angles) and float for position/scale; the frontend
# converts BAMS → radians at apply time.
from formats.njm import (
    parse_njm as _njm_parse,
    parse_njm_header_only as _njm_parse_header,
    pick_default_motion as _njm_pick_default,
    guess_motion_fps as _njm_guess_fps,
)
# Tier-ranked motion pairing — handles multi-form BMLs (Pan Arms, De
# Rol Le, Vol Opt, Pouilly Slime) where the legacy verb-only picker
# auto-played a motion authored for the wrong inner-model rig. Tier
# weights come from the on-disk inventory in
# ``_reports/motion_inventory.md``; see ``formats/motion_pairing.py``
# header for the four-tier taxonomy.
from formats.motion_pairing import (
    resolve_motions_for_model as _resolve_motion_pairing,
    extract_action_hint as _motion_action_hint,
)
# Variant detector — finds color/state sibling BMLs and intra-BML NJTL
# slot groups so the frontend can offer a "variant picker" (Mericarol →
# Mericus → Merikle, Booma → Gobooma, etc.). See
# formats/variant_detector.py for the heuristic + family table.
from formats.variant_detector import (
    VariantInfo,
    detect_variants as _detect_variants,
)
# Map Editor scene catalogue (2026-04-25) — groups every scene/map_*
# manifest entry by (area, area_num, floor) so the Map perspective can
# pick a map + floor and load every renderable NJ/XJ in parallel. See
# formats/scene_loader.py for the wire shape + spawn/waypoint validators.
from formats import scene_loader as _scene_loader

# Floor copy/create editor (2026-06-20). rel_writer carries the size
# budgets + the RelWriteError raised on overflow; the floor endpoints map
# that to HTTP 422. Imported here so the budget constants are available at
# module scope. NOTE: the floor module deliberately does NOT import
# safe_live_path / reference LIVE_DATA_DIR for any write.
from formats import rel_writer as _rw

# Audio suite (2026-06-20). Pure-Python byte-exact .pac PCM SFX-bank codec
# (audio_pac) + optional ffmpeg-backed .ogg/.sfd decode (audio_codec), behind
# the `audio` facade. The Replace verb here is DEV-ONLY: every output is
# hard-asserted (via _floor_assert_not_live) to resolve inside DEV_DATA_DIR
# and NEVER under LIVE_DATA_DIR — unlike /api/repack_afs_inner which writes
# LIVE. ffmpeg absence degrades to HTTP 501, never 500.
from formats import audio as audio_mod

# AI generation provider plugins (additive; never imported at request-time
# critical paths, so an ImportError here only kills /api/aigen/* — the
# rest of the editor keeps working).
import aigen as aigen_mod

# AI-gen MVP provider abstraction + budget guard (P5). Separate from the
# legacy aigen v1 WebUI providers above: this layer adds a cost model and a
# spend guard so a fresh, key-less install can run the free local-upscale
# path but is BLOCKED from any paid provider until a budget is configured.
from aigen.budget import BudgetExceeded as AigenBudgetExceeded
from aigen.budget import BudgetGuard as AigenBudgetGuard
from aigen.providers import ImageRequest as AigenImageRequest
from aigen.providers import default_registry as _aigen_default_registry

VERSION = "1.2.1-cleanup"

ROOT = Path(__file__).parent.resolve()
# Editor operates on a DEV mirror of the game install so the user can keep playing
# while we iterate. Override with the PSO_DATA_DIR env var if needed.
LIVE_DATA_DIR = Path(os.path.expanduser(os.environ.get("PSO_LIVE_DATA_DIR") or "~/PSOBB.IO/data")).resolve()
DEV_DATA_DIR = Path(os.path.expanduser(os.environ.get("PSO_DEV_DATA_DIR") or "C:/tmp_pso_dev/data")).resolve()
DATA_DIR = Path(os.environ.get("PSO_DATA_DIR") or DEV_DATA_DIR).resolve()
CACHE_DIR = ROOT / "cache"
STATIC_DIR = ROOT / "static"
EXPORT_DIR = ROOT / "exports"  # token-protected artifacts available via /api/export/...
CACHE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# Parsed-mesh LRU root. Lives under cache/ so it survives across server
# restarts (so dragon-class first-open after restart hits a 50-150 ms
# pickle load instead of a 1.1 s parse). Schema-versioned subdir is
# created on demand inside formats/parse_cache.py.
PARSE_CACHE_DIR = CACHE_DIR / "parse_cache"
PARSE_CACHE_DIR.mkdir(exist_ok=True)
_parse_cache.configure(cache_dir=PARSE_CACHE_DIR)

# Tile-PNG LRU root (Phase D Win 5, 2026-04-25). Per-tile texture renders
# are XVR→PIL→PNG conversions costing ~50-100 ms each on cold pour. Dragon
# has 16 tiles, so first-open of a dragon over a cold tile cache eats
# 0.8-1.6 s of texture decoding wall-time. The bytes themselves are
# small (tens of KB per PNG), so we cache the PNG bytes both in-memory
# (fast warm) and on-disk (fast cold-after-restart). See the cache layer
# diagrams below /api/binding_cache for the full ordering.
_TILE_PNG_CACHE_DIR_ENV = os.environ.get("PSO_TILE_PNG_CACHE_DIR")
TILE_PNG_CACHE_DIR = (
    Path(_TILE_PNG_CACHE_DIR_ENV).resolve()
    if _TILE_PNG_CACHE_DIR_ENV
    else CACHE_DIR / "tile_png"
)
# `parents=True` so a per-worker test override at e.g.
# ~/Repositories/psobb-studio/cache/tile_png_test_xdist/gw0 doesn't 500 on first
# write because the parent dir wasn't pre-created. The default install
# path under cache/ already exists.
TILE_PNG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TILE_PNG_CACHE_SCHEMA = 1  # bump if PNG-key shape changes

# Skinned-payload LRU root (Phase 0.5 perf, 2026-04-25). After the parse
# cache and binding cache, the remaining wall time on a dragon-class
# /api/model_skinned cold open is ~1.1-1.3 s. The function
# `_xj_meshes_to_skinned_payload` accounts for ~33-100 ms of that on its
# own (b64 encoding + dict assembly per submesh) — and crucially, its
# output is fully deterministic in (model_path, inner, mtime_ns, size),
# exactly the file-key shape parse_cache uses. Caching the PAYLOAD dict
# (geometry + skeleton, NOT the binding which has its own cache) on
# disk lets cold-after-restart skip both the parse_cache disk-load AND
# the conversion. JSON-on-disk is the right serialisation choice — the
# payload is already JSON-shaped (b64 strings + small dicts), so we
# avoid pickle's pre-load cost and a corrupted file is human-readable.
SKINNED_PAYLOAD_CACHE_DIR = CACHE_DIR / "skinned_payload"
SKINNED_PAYLOAD_CACHE_DIR.mkdir(exist_ok=True)
SKINNED_PAYLOAD_CACHE_SCHEMA = 2  # bump on payload-shape change (v2: +RGBA color, 12-float interleave)

# Binding-cache LRU disk root (Phase 0.5 perf, Item 5 of finishing-line
# 2026-04-25). The in-memory cache wraps `_build_model_texture_binding`
# which spends ~1 s on dragon-class first-open (PRS decompress + NJTL
# parse + cross-archive XVMH lookup). Cache hits skip all of that and
# return in <5 ms — but only WITHIN a single process lifetime. Adding
# disk persistence at `cache/binding/v<schema>/<sha>.json` lets a
# server restart hit a 30-50 ms json load instead of redoing the full
# work. Mirrors the skinned-payload disk layout: JSON for human-
# readable diagnostics + cheap re-encode, atomic tmp+rename writes,
# corruption-recovery on parse fail, per-entry size cap below.
BINDING_CACHE_DIR = CACHE_DIR / "binding"
BINDING_CACHE_DIR.mkdir(exist_ok=True)
# v2 (2026-04-25): texture_index gained per-XVR sub-rows for ItemTexture
# archives + a positional ItemModel resolver. AFS-resident model inners
# previously cached an empty cross_afs map; bumping the schema rotates
# the cache directory so those stale entries don't get served.
BINDING_CACHE_SCHEMA = 3  # bump on binding-payload-shape change

# Battle-param staging directory: edited variants land here as raw .dat
# blobs before the user clicks "Deploy to newserv". Never overwrite the
# user's newserv install directly.
BATTLE_PARAM_STAGE_DIR = CACHE_DIR / "battle_param_export"
BATTLE_PARAM_STAGE_DIR.mkdir(exist_ok=True)

# ItemPMT staging directory. Edited PMTs land here as PRS-compressed
# .prs blobs before the user clicks "Deploy to newserv". The compressor
# is a separate step from JSON edit (POST /api/itempmt → stage,
# POST /api/itempmt/deploy → copy stage to newserv install).
ITEMPMT_STAGE_DIR = CACHE_DIR / "itempmt_export"
ITEMPMT_STAGE_DIR.mkdir(exist_ok=True)
# The shipped name on disk. newserv keeps the V4 file at
# `system/item-tables/ItemPMT-bb-v4.prs`; Booma.Server uses the legacy
# flat name `ItemPMT.prs`. We probe both and write back to whichever was
# the source.
ITEMPMT_STAGE_FILENAME_BB_V4 = "ItemPMT-bb-v4.prs"
ITEMPMT_STAGE_FILENAME_LEGACY = "ItemPMT.prs"

# Sculpt persistence (2026-04-25). The Sculpt tab writes per-model
# vertex-displacement deltas here as JSON sidecars. Filename shape
# `<safe_path>__<sha>.json`, where ``safe_path`` is the source model
# path with `/`, `\`, `#` replaced by `__` so the filename round-trips
# safely on Windows. ``sha`` is the 16-char hash returned by the save
# endpoint and re-used by the fetch endpoint as a stable cache key.
SCULPT_CACHE_DIR = CACHE_DIR / "sculpted_meshes"
SCULPT_CACHE_DIR.mkdir(exist_ok=True)

# Rig persistence (2026-04-25). The Rig tab writes per-model skeleton
# edits + bone weight assignments + IK target stubs here as JSON
# sidecars, same shape as sculpted_meshes/. Filename
# `<safe_path>__<sha>.json`; the on-disk SHA round-trips with the
# frontend's local hash so cache invalidates when the source mesh
# changes.
RIG_CACHE_DIR = CACHE_DIR / "rigs"
RIG_CACHE_DIR.mkdir(exist_ok=True)

# Pro-tools edit-mode sidecar storage (2026-04-26). The Edit tab
# (edit_panel.js) writes vertex-transform deltas here as JSON sidecars,
# same shape as sculpted_meshes/ but with explicit per-vertex indices
# instead of full displacement arrays. The existing
# /api/sculpt/build_archive walker already picks up these sidecars when
# they live in SCULPT_CACHE_DIR, so we keep the protools edits in a
# sibling dir and merge at archive-build time. Filename
# `<safe_path>__<sha>.json`.
PROTOOLS_EDITS_DIR = CACHE_DIR / "protools_edits"
PROTOOLS_EDITS_DIR.mkdir(exist_ok=True)

# Map Editor sidecar storage (2026-04-25). The Map perspective writes
# spawn / waypoint placements here as JSON sidecars, one per map_id.
# Future quest-script editor will pick up the same shape; this dir is
# the single source of truth for "user-edited scene metadata".
MAP_EDITS_DIR = CACHE_DIR / "map_edits"
MAP_EDITS_DIR.mkdir(exist_ok=True)

# newserv install path. Used to read stock BattleParamEntry*.dat files
# for editing. Two candidates ship pre-baked into the editor for the
# common Booma.Server fixtures path; if neither exists, the import
# endpoint falls back to a 404 with a helpful message. Override either
# of these via NEWSERV_PATH (single dir) or NEWSERV_BLUEBURST_DIR
# (specific battle-params subdir).
NEWSERV_PATH = Path(
    os.environ.get("NEWSERV_PATH")
    or os.path.expanduser("~/newserv")
).resolve()
# `system/blueburst/` under a normal newserv install holds the
# BattleParamEntry*.dat files. Booma.Server's data layout is flat
# (everything in `Data/`); we probe both shapes.
NEWSERV_BLUEBURST_CANDIDATES = (
    NEWSERV_PATH / "system" / "blueburst",
    NEWSERV_PATH / "blueburst",
    NEWSERV_PATH / "data",
    NEWSERV_PATH,  # flat
    Path(os.environ.get("NEWSERV_BLUEBURST_DIR") or "").resolve() if os.environ.get("NEWSERV_BLUEBURST_DIR") else None,
    # Fallback to local Booma.Server fixtures so the editor works
    # out-of-the-box even before the user installs newserv.
    Path(os.path.expanduser("~/Repositories/psobb2/Booma.Server/Data")).resolve(),
)


def _resolve_newserv_battleparam_dir() -> Optional[Path]:
    """Return the first directory containing BattleParamEntry*.dat or None."""
    for cand in NEWSERV_BLUEBURST_CANDIDATES:
        if cand is None:
            continue
        try:
            if cand.is_dir() and any(cand.glob("BattleParamEntry*.dat")):
                return cand
        except OSError:
            continue
    return None


# Search candidates for the ItemPMT-bb-v4.prs (or legacy ItemPMT.prs).
# newserv-canonical layout puts it at `system/item-tables/`. Booma.Server
# fixtures keep the flat-name version at `Data/`. We probe both and
# accept whichever exists first.
ITEMPMT_CANDIDATES = (
    NEWSERV_PATH / "system" / "item-tables",
    NEWSERV_PATH / "system" / "blueburst",
    NEWSERV_PATH / "blueburst",
    NEWSERV_PATH / "item-tables",
    NEWSERV_PATH / "data",
    NEWSERV_PATH,  # flat
    Path(os.environ.get("NEWSERV_ITEMPMT_DIR") or "").resolve()
        if os.environ.get("NEWSERV_ITEMPMT_DIR") else None,
    # Fallback: Booma.Server fixtures (always available locally for the
    # editor developer).
    Path(os.path.expanduser("~/Repositories/psobb2/Booma.Server/Data")).resolve(),
)


def _resolve_newserv_itempmt() -> Optional[Path]:
    """Return the first existing ItemPMT(-bb-v4)?.prs path or None."""
    for cand in ITEMPMT_CANDIDATES:
        if cand is None:
            continue
        try:
            if not cand.is_dir():
                continue
            for fname in (
                ITEMPMT_STAGE_FILENAME_BB_V4,
                ITEMPMT_STAGE_FILENAME_LEGACY,
            ):
                p = cand / fname
                if p.is_file():
                    return p
        except OSError:
            continue
    return None

# Map of export tokens to (file path, expiry_ts). Tokens are returned by
# /api/repack with deploy=false; the caller fetches via
# /api/export/<token>. Tokens expire after EXPORT_TTL_SECONDS or the next
# server restart.
#
# Wave 7 (2026-04-26): in addition to the in-memory map, every minted
# token is persisted to a sidecar JSON file under EXPORT_DIR so a
# multi-worker uvicorn launch (workers=4) doesn't lose tokens to
# cross-process invisibility — worker A mints, worker B serves the
# user's GET. The on-disk index is the source of truth; the in-memory
# map is just a startup-time accelerator (avoids per-GET disk reads
# inside the same worker).
_EXPORT_TOKENS: dict[str, dict] = {}
# Audit C-2 (2026-05-01): _gc_export_tokens iterates _EXPORT_TOKENS while
# other request handlers may insert/pop concurrently — would raise
# "dictionary changed size during iteration". Single Lock around every
# read/write/iter snapshot of the dict.
_EXPORT_TOKENS_LOCK = threading.Lock()
EXPORT_TTL_SECONDS = 6 * 3600  # 6h

# Tools (resolved up-front; reported by /api/health)
PUYO = Path(os.environ.get("PSO_PUYOTOOLS") or r"C:/Tools/re/upscale-lab/tools/puyotools/PuyoToolsCli.exe").resolve()
XVR_CODEC = Path(os.environ.get("PSO_XVR_CODEC") or r"C:/Tools/re/upscale-lab/tools/xvr_codec.py").resolve()
REALESRGAN = Path(os.environ.get("PSO_REALESRGAN") or r"C:/Tools/re/upscale-lab/tools/realesrgan_bundle/realesrgan-ncnn-vulkan.exe").resolve()
REALESRGAN_MODELS = REALESRGAN.parent / "models"
PYEXE = Path(sys.executable).resolve()

# Timeouts (seconds)
TIMEOUT_PUYO = 60
TIMEOUT_XVR_REBUILD = 90
TIMEOUT_UPSCALE = 600  # ncnn-vulkan can be slow on big tiles + cold GPU init

# Retry policy for the realesrgan subprocess. Vulkan device creation can fail
# transiently when the game is running on the same GPU (vkCreateDevice -3),
# or the binary can exit clean without producing the output file under load.
# We retry with exponential backoff so a single hiccup doesn't fail the request.
UPSCALE_RETRY_ATTEMPTS = 3
UPSCALE_RETRY_BACKOFF_SECONDS = (0.5, 1.5, 4.0)  # delay BEFORE attempts 2, 3, (4)

# Cache cleanup policy
CACHE_TTL_SECONDS = 24 * 3600  # 24h

# Lock dict caps - LRU eviction prevents unbounded growth across long sessions
MAX_UPSCALE_LOCKS = 512
MAX_REPACK_LOCKS = 64

# Subprocess output capture (truncated for error messages)
SUBPROCESS_OUTPUT_TAIL = 1000

# Request body limits
MAX_REPACK_BODY = 64 * 1024 * 1024  # 64 MB - allows ~8x 4096^2 PNGs
MAX_UPSCALE_BODY = 4 * 1024  # tiny JSON; oversize = client bug
MAX_RESTORE_BODY = 4 * 1024
MAX_REPACK_DIFF_BODY = 16 * 1024
MAX_IMPORT_PNG_BYTES = 64 * 1024 * 1024
MAX_TILES_PER_REPACK = 64
# Sculpt save body limit. A boss-class model (~10k verts) with full
# displacement carries ~120 KB of float32 + base64 inflation = ~160 KB
# per submesh. 32 MB allows ~200 submeshes worth of sculpt data —
# overkill for any real model but cheap.
MAX_SCULPT_SAVE_BODY = 32 * 1024 * 1024
# Rig save body limit. A boss-class skinned model (~10k verts) at 4
# influences/vert carries ~80 KB of int32 + 80 KB of float32 +
# inflation = ~250 KB per submesh, plus a few KB of skeleton/IK
# targets. 32 MB matches the sculpt cap for symmetry; well above any
# real model's needs.
MAX_RIG_SAVE_BODY = 32 * 1024 * 1024
# Viewport (16:9 transform) — body holds a single 1278x768 RGBA PNG at most.
# A wholly opaque 1278x768 RGBA PNG is ~3 MB; 32 MB gives a 10x margin for
# upscaled paint sources we may accept later.
MAX_VIEWPORT_PAINT_BODY = 32 * 1024 * 1024
# Logical 16:9 render canvas dimensions, per the LogoEP4 layout research
# (see MASTER_PLAN/03_render_pipeline_re.md and the per-file
# atlas_layouts.py table). The widescreen ASI fills the right 256-px
# pillar with white tiles when the splash is rendered.
VIEWPORT_W = 1278
VIEWPORT_H = 768
# Native 4:3 inset (centered horizontally) inside the 16:9 canvas. The
# splash atlas is designed for this region; the leftovers are pillar fill.
VIEWPORT_43_W = 1024
VIEWPORT_43_H = 768
# Deploy promote: body is just a list of names + flag; small JSON.
MAX_PROMOTE_BODY = 16 * 1024
# Hard cap on names per single promote call. Defends against accidental
# "deploy everything in dev" requests; the user can chunk if they really
# want to push hundreds of files at once.
MAX_PROMOTE_FILES = 256

# PNG and XVM binary layout constants
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
XVM_HEADER_SIZE = 0x40
XVM_MAGIC = b"XVMH"
XVRT_MAGIC = b"XVRT"

# PSOBB engine has an undocumented internal max texture dimension. Empirically
# 1024 — at 2048 the game silently fails to load the texture, surfacing as
# black geometry / missing trees / missing splash particles. We clamp every
# newly-upscaled output to this cap (only newly-upscaled — original sources
# above the cap are left alone, see _cascade_upscale). Requested scale=8 on
# a 256x source therefore effectively delivers 4x (1024x1024); intentional.
PSOBB_MAX_TEXTURE_DIM = 1024

# File extensions to enumerate
ALLOWED_EXT = (".prs", ".xvm")

# Backup name fragments excluded from listings
BACKUP_FRAGMENTS = (".pre_", ".suspect_", ".parked_", ".bad_", ".disabled")

# Manifest endpoint guard: refuse to serialize a payload above this many bytes.
# At ~5900 entries the live install produces ~1.7 MB; 32 MB gives a 5x margin
# for future growth (siblings, matched_textures populated by Agent 3) before
# we have to chunk or move to a streaming response.
MAX_MANIFEST_RESPONSE_BYTES = 32 * 1024 * 1024

# Raw-bytes endpoint guard: any single file above this is forbidden because
# the response is buffered in memory. Streaming would be a separate endpoint.
# 16 MB covers every audio (.ogg) / model (.bml) / quest (.qst) in the live
# install; the only legitimate file above this is map_aancient03.xvm at
# ~47 MB which already has its own /api/tiles route.
MAX_RAW_RESPONSE_BYTES = 16 * 1024 * 1024

# Filename validation - disallow path separators and traversal markers
_INVALID_FILENAMES = ("", ".", "..")

# Tile filename pattern: tile_<idx>_<W>x<H>.png
TILE_FILENAME_RE = re.compile(r"_(\d+)_(\d+)x(\d+)\.png$")

# Model name allowed pattern (also used for export tokens)
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
EXPORT_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# Tile-index sanity bounds (every PSOBB tilemap is well under this)
MAX_TILE_INDEX = 4096

# Imported-PNG filename sanitization
IMPORT_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Path-component → underscore sanitization for sculpt/protools/rig
# cache filenames (3 identical re.sub call sites used to recompile this
# pattern on every save; precompiled once at module load now).
_CACHE_PATH_SEPS_RE = re.compile(r"[\\/#]")
_CACHE_PATH_SAFE_RE = re.compile(r"[^A-Za-z0-9._\-]")

# Player class texture stem → body model letter. Used by the
# /api/model_preview endpoint per request to map plAtex → plAbdy00.nj.
# Was previously recompiled per-call inside the route handler.
_PLAYER_TEX_STEM_RE = re.compile(r"^pl([a-x])tex(\d{0,2})$", re.IGNORECASE)

# GPU id range
ALLOWED_GPU_ID_RANGE = (-1, 7)

# ----------------------------------------------------------------------------
# Logging - everything except the startup banner goes through this logger.
# ----------------------------------------------------------------------------
log = logging.getLogger("psobb_editor")
if not log.handlers:
    _log_handler = logging.StreamHandler(sys.stdout)
    _log_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_log_handler)
    log.setLevel(logging.INFO)
    log.propagate = False

# Locks
_GLOBAL_LOCK = threading.Lock()
# OrderedDict so we can pop the LRU when the cap is hit
_UPSCALE_LOCKS: "OrderedDict[str, threading.Lock]" = OrderedDict()
_REPACK_LOCKS: "OrderedDict[str, threading.Lock]" = OrderedDict()
# Single global lock around /api/deploy/promote — only one promote in
# flight across the whole process. Promotes touch the user's live game
# install directly, so we keep the concurrency model dead simple.
_PROMOTE_LOCK = threading.Lock()


# Model knowledge base. Hardcoded - the bundled binary's -s caps at 4
# but model NATIVE scale can vary, and chained passes give us 8/16/etc.
MODEL_INFO: dict[str, dict] = {
    "realesr-animevideov3-x2": {
        "native_scale": 2,
        "max_scale": 16,  # via chain 2*2*2*2 etc, but we cascade by 4s
        "supports_tta": True,
        "description": "Anime video model, native 2x output. Fast, good for sharp lineart.",
    },
    "realesr-animevideov3-x3": {
        "native_scale": 3,
        "max_scale": 12,  # 3*4 ceil
        "supports_tta": True,
        "description": "Anime video model, native 3x output. Middle ground for sprite work.",
    },
    "realesr-animevideov3-x4": {
        "native_scale": 4,
        "max_scale": 16,
        "supports_tta": True,
        "description": "Anime video model, native 4x output. Best for HUD icons / pixel sprites.",
    },
    "realesrgan-x4plus": {
        "native_scale": 4,
        "max_scale": 16,
        "supports_tta": True,
        "description": "General photo-realistic model, 4x. Cascade for 8x/16x.",
    },
    "realesrgan-x4plus-anime": {
        "native_scale": 4,
        "max_scale": 16,
        "supports_tta": True,
        "description": "Anime-tuned RealESRGAN, 4x. Best general choice for game art.",
    },
}

ALLOWED_TILE_SIZES = (0, 32, 64, 128, 256, 512)  # 0 = auto
ALLOWED_SCALES = (2, 3, 4, 6, 8, 12, 16)


def model_meta(name: str) -> dict:
    """Return metadata for a model. Defaults assume x4-class generic model."""
    if name in MODEL_INFO:
        return dict(MODEL_INFO[name])
    # heuristic from the name
    native = 4
    m = re.search(r"-x(\d+)", name)
    if m:
        try:
            native = int(m.group(1))
        except ValueError:
            pass
    return {
        "native_scale": native,
        "max_scale": max(16, native * 4),
        "supports_tta": True,
        "description": f"Unknown model; assumed native scale {native}x.",
    }


# ---------------------------------------------------------------------------- helpers
def _get_lock(
    table: "OrderedDict[str, threading.Lock]",
    key: str,
    cap: int,
) -> threading.Lock:
    """Atomically fetch-or-create a per-key lock with LRU eviction.

    The OrderedDict is LRU-ordered: every access moves the key to the end,
    and inserts past `cap` evict from the front. Eviction only fires for
    locks that no other thread currently holds (acquire(blocking=False) test);
    held locks are skipped so concurrent operations are never disturbed.
    """
    with _GLOBAL_LOCK:
        lk = table.get(key)
        if lk is not None:
            table.move_to_end(key)
            return lk
        lk = threading.Lock()
        table[key] = lk
        # Trim down to `cap` by evicting unheld LRU entries.
        while len(table) > cap:
            evicted = False
            # Iterate from oldest to newest, skip the one we just inserted.
            for victim_key in list(table.keys())[:-1]:
                victim = table[victim_key]
                if victim.acquire(blocking=False):
                    try:
                        del table[victim_key]
                    finally:
                        victim.release()
                    evicted = True
                    break
            if not evicted:
                # All other locks busy; can't evict. Bail out, accept temporary growth.
                break
        return lk


def _enforce_body_size(request: Request, limit: int) -> None:
    """Reject oversized bodies up front via Content-Length header.

    Starlette buffers the body before our handler sees it, so we want to
    reject early to avoid memory blowup. Missing or malformed Content-Length
    is treated as 'unknown' and allowed through.
    """
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            n = int(cl)
        except ValueError:
            raise HTTPException(400, "invalid content-length")
        if n > limit:
            raise HTTPException(413, f"request body too large ({n} > {limit} bytes)")


def _validate_bare_filename(name: str, *, label: str = "filename") -> str:
    """Validate `name` is a bare basename with no path components or traversal.

    Returns the validated bare name. Raises HTTPException(400) on missing,
    non-string, or any path-separator / reserved-name input.
    """
    if not name or not isinstance(name, str):
        raise HTTPException(400, f"missing {label}")
    bare = Path(name).name
    if bare != name or bare in _INVALID_FILENAMES or "/" in name or "\\" in name:
        raise HTTPException(400, f"invalid {label} (path components forbidden)")
    return bare


def _validate_inner_name(inner: str, *, msg: str = "invalid inner name", required: bool = False) -> None:
    """Reject path separators or reserved names in a BML/AFS inner entry name.

    With ``required=True``, also rejects empty / falsy input.
    """
    if required and not inner:
        raise HTTPException(400, msg)
    if "/" in inner or "\\" in inner or inner in _INVALID_FILENAMES:
        raise HTTPException(400, msg)


def safe_data_path(name: str) -> Path:
    """Resolve `name` strictly inside DATA_DIR. Reject path traversal."""
    bare = _validate_bare_filename(name, label="filename")
    p = (DATA_DIR / bare).resolve()
    try:
        p.relative_to(DATA_DIR)
    except ValueError:
        raise HTTPException(400, "path escapes data dir")
    return p


# ---------------------------------------------------------------------------
# BML-inner path resolver (2026-04-24)
#
# Asset-tree paths can use a "<base>#<inner>" syntax to reference one entry
# inside a BML container. The frontend (asset_router.js / model_viewer.js)
# already passes these through, so the backend just has to understand them.
#
# Examples:
#   "biri_ball.bml"                          -> regular file
#   "bm4_ps_ma_body.bml#bm4_ps_ma_body.nj"   -> mesh inside BML
#   "bm4_ps_ma_body.bml#bm4_ps_ma_body.nj.xvm"
#                                            -> texture inside BML (the
#                                               trailing ".xvm" maps to
#                                               extract_bml_texture(name=...nj)
#                                               since textures are sibling
#                                               XVM blobs of inner NJs)
#
# Search order is the same as `_resolve_bml_path` / `_resolve_raw_path`:
# DATA_DIR first, LIVE_DATA_DIR fallback (read-only).
# ---------------------------------------------------------------------------
def _split_inner_path(path: str) -> tuple[str, Optional[str]]:
    """Split `<base>#<inner>` into (base, inner). Returns (path, None) if no '#'.

    Raises HTTPException(400) on malformed `#`-syntax (empty base or empty
    inner, multiple `#`, or path-component characters in either side).
    """
    if not path or not isinstance(path, str):
        raise HTTPException(400, "missing path")
    if "#" not in path:
        return path, None
    parts = path.split("#")
    if len(parts) != 2:
        raise HTTPException(400, "invalid path (multiple '#' separators)")
    base, inner = parts
    if not base or not inner:
        raise HTTPException(400, "invalid path (empty base or inner before/after '#')")
    _validate_inner_name(inner, msg="invalid inner name (path components forbidden)")
    return base, inner


def _split_inner_with_query(path: str, query_inner: Optional[str]) -> tuple[str, Optional[str]]:
    """Split `<base>#<inner>` and reconcile with a separate `?inner=` query.

    Both forms (`#fragment` and `?inner=`) are accepted; rejecting only when
    they disagree avoids ambiguity. Returns (base, effective_inner). Used by
    every endpoint that accepts dual-form inner-naming.
    """
    base, hash_inner = _split_inner_path(path)
    if hash_inner is not None and query_inner and query_inner != hash_inner:
        raise HTTPException(400, "conflicting inner: '#' fragment and ?inner= disagree")
    return base, hash_inner or query_inner


def _resolve_under_roots(name: str, roots: tuple[Path, ...], *, label: str, missing_msg: str) -> Path:
    """Validate `name` is a bare filename and locate it in the first matching root.

    Used by the family of dual-root resolvers (DATA_DIR + LIVE_DATA_DIR,
    optionally + a cache subdir). Raises HTTPException(400) on bad input
    and HTTPException(404) with `missing_msg` if not found in any root.
    """
    bare = _validate_bare_filename(name, label=label)
    for root in roots:
        candidate = (root / bare).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.exists() and candidate.is_file():
            return candidate
    raise HTTPException(404, missing_msg)


def _resolve_base_path(base: str) -> Path:
    """Resolve a BASE filename under DATA_DIR or LIVE_DATA_DIR (read-only).

    Same injection guard as `safe_data_path` but extends the search to
    LIVE_DATA_DIR. The DATA_DIR copy wins if both exist (matches the
    precedence used by `_resolve_bml_path`).
    """
    return _resolve_under_roots(
        base,
        (DATA_DIR, LIVE_DATA_DIR),
        label="base",
        missing_msg=f"no such file: {base}",
    )


def _extract_bml_inner_bytes(bml_path: Path, inner: str) -> tuple[bytes, str]:
    """Resolve an inner-name to raw bytes inside a BML container.

    Naming conventions (matching formats/match.py R2 synthesis):
      "<entry>.nj"      -> extract the inner NJ payload (PRS-decompressed)
      "<entry>.njm"     -> same, for animation chunks
      "<entry>.nj.xvm"  -> extract the BML's per-entry XVM texture archive
                           (the BML stores it as a sibling of the NJ)

    Returns (raw_bytes, logical_filename). The logical filename is the
    inner-name as the caller passed it; downstream extract paths use it
    to label tile filenames in the cache.
    """
    sz = bml_path.stat().st_size
    if sz > ASSET_PARSE_MAX_BYTES:
        raise HTTPException(
            413,
            f"BML too large to parse in-memory: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
        )
    blob = bml_path.read_bytes()
    try:
        entries = parse_bml(blob)
    except ValueError as e:
        raise HTTPException(400, f"BML parse failed: {e}")

    inner_lower = inner.lower()
    if inner_lower.endswith(".xvm"):
        # Texture extraction path: strip the `.xvm` suffix to recover the
        # BML entry name, then call extract_bml_texture for the XVM bytes.
        ent_name = inner[: -len(".xvm")]
        if not any(ent.name == ent_name for ent in entries):
            raise HTTPException(404, f"no entry named {ent_name!r} in {bml_path.name}")
        try:
            tex = extract_bml_texture(blob, ent_name, timeout=TIMEOUT_BML_PRS)
        except ValueError as e:
            raise HTTPException(400, f"BML texture extract failed: {e}")
        except RuntimeError as e:
            raise HTTPException(502, f"BML texture extract failed: {e}")
        if tex is None:
            raise HTTPException(404, f"entry {ent_name!r} has no texture")
        return tex, inner

    # Mesh / animation path: extract the raw inner blob (PRS-decompressed).
    target = next((ent for ent in entries if ent.name == inner), None)
    if target is None:
        raise HTTPException(404, f"no entry named {inner!r} in {bml_path.name}")
    try:
        from formats.bml import decompress_prs_cached
        st = bml_path.stat()
        slice_start = target.offset
        slice_end = slice_start + target.size_compressed
        out = decompress_prs_cached(
            bml_path, st.st_mtime_ns, inner,
            lambda: bytes(blob[slice_start:slice_end]),
        )
        return out, inner
    except (RuntimeError, ValueError) as e:
        raise HTTPException(502, f"BML PRS decompress failed: {e}")


def _parse_afs_inner_name(inner: str) -> tuple[Optional[int], str]:
    """Parse an AFS-inner name of the form ``"NNNN_<inner_basename>"``.

    Returns ``(index, inner_basename)``. The 4-digit prefix is what
    ``manifest._synthesize_afs_entries`` writes; the basename tail is
    purely cosmetic (used by the asset tree label) and ignored by the
    lookup — only the index matters for extraction.

    Returns ``(None, inner)`` if the prefix isn't present so callers can
    accept either ``"0042_Sword.nj"`` (manifest synth) or a bare digit
    string ``"42"`` (manual API call).
    """
    s = inner
    # Strict match: 4 digits + '_' + tail.
    if len(s) >= 5 and s[:4].isdigit() and s[4] == "_":
        return int(s[:4]), s[5:]
    # Permissive: a bare digit run as the entire inner.
    if s.isdigit():
        return int(s), ""
    return None, inner


def _extract_afs_inner_bytes(afs_path: Path, inner: str) -> tuple[bytes, str]:
    """Resolve an inner-name to raw bytes inside an AFS container.

    The inner name shape is ``"NNNN_<basename>"`` (manifest synth) or
    ``"NNNN"`` (raw index). We materialise the cached inner via
    ``formats.afs_reader.materialize_inner`` so PRS decompression is
    handled transparently and subsequent reads are O(1).

    Returns ``(bytes, logical_filename)``. The logical filename is the
    cached file's basename — downstream consumers dispatching on
    extension see e.g. ``"0042_sword.nj"`` and route correctly.
    """
    idx, basename = _parse_afs_inner_name(inner)
    if idx is None:
        raise HTTPException(400, f"AFS inner must be 'NNNN_name' or 'NNNN', got {inner!r}")
    try:
        from formats import afs_reader as _afs_reader
    except ImportError as e:
        raise HTTPException(500, f"AFS reader unavailable: {e}")
    try:
        cache_path, info = _afs_reader.materialize_inner(
            afs_path, idx, CACHE_DIR, timeout=TIMEOUT_BML_PRS,
        )
    except IndexError as e:
        raise HTTPException(404, str(e))
    except (ValueError, OSError) as e:
        raise HTTPException(400, f"AFS extract failed: {e}")
    blob = cache_path.read_bytes()
    # Use the manifest-synth basename when present (so e.g.
    # `Foo.afs#0042_sword.nj` round-trips its `.nj` extension), else
    # fall back to the cache filename which already carries the
    # sniffed extension.
    logical = basename if basename else cache_path.name
    return blob, logical


def resolve_asset_bytes(path: str) -> tuple[bytes, str]:
    """Resolve `<file>` or `<container>#<inner>` to raw bytes + a logical filename.

    Falls back to LIVE_DATA_DIR if the base file is not in DATA_DIR.
    Container-inner paths are dispatched by base extension:
      `<bml>#<entry>.nj`           -> raw NJ bytes (PRS-decompressed)
      `<bml>#<entry>.njm`          -> raw NJM bytes (PRS-decompressed)
      `<bml>#<entry>.nj.xvm`       -> raw XVM bytes (entry's texture archive)
      `<archive>.afs#NNNN_name`    -> raw inner blob (AFS, PRS-decompressed)
      `<archive>.afs#NNNN`         -> raw inner blob (AFS, by index)

    The returned logical filename is:
      * the input `path` when there is no `#`
      * the inner name when there is a `#`

    Callers using the logical filename to dispatch on extension (e.g.
    `extract_tiles` checking for `.prs` vs `.xvm`) get the right branch
    in both modes.

    Raises HTTPException with the appropriate code:
      400 - malformed `#`-syntax / path traversal
      404 - file or inner-entry missing
      413 - file too large to parse in-memory
      502 - BML PRS decompress subprocess failed
    """
    base, inner = _split_inner_path(path)
    base_path = _resolve_base_path(base)
    if inner is None:
        sz = base_path.stat().st_size
        if sz > ASSET_PARSE_MAX_BYTES:
            raise HTTPException(
                413,
                f"file too large to parse in-memory: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
            )
        return base_path.read_bytes(), base
    base_ext = base_path.suffix.lower()
    if base_ext == ".bml":
        return _extract_bml_inner_bytes(base_path, inner)
    if base_ext == ".afs":
        return _extract_afs_inner_bytes(base_path, inner)
    raise HTTPException(
        400,
        f"`#` syntax only supported for .bml or .afs bases, got {base_ext!r}",
    )


# Cache subdir for materialized BML-inner blobs. We write one `.xvm` /
# `.nj` per inner here so `extract_tiles` can take a real Path. Files
# are content-keyed by an md5 of (base_name, base_size, base_mtime,
# inner) so changes to the parent BML invalidate the inner cache.
_BML_INNER_CACHE_SUBDIR = "bml_inner"


def _wrap_xvr_as_xvmh(xvr_bytes: bytes) -> bytes:
    """Wrap a bare XVRT record in a single-record XVMH archive.

    PSOBB's ``pl[A-Z]tex.afs`` archives store every player-class
    texture as a flat XVRT blob (no XVMH container). The downstream
    ``xvr_codec.py extract`` path requires XVMH framing; we wrap on the
    fly so the existing tile pipeline (extract_tiles + tile_png) works
    against AFS-resident textures without a special case.

    Layout of the synthesised wrapper:
      0x00  "XVMH"
      0x04  u32 header size  (0x38, matches Sega's authoring tool)
      0x08  u32 texture count (always 1 here)
      0x0C..0x3F  zero padding
      0x40+ the XVRT block verbatim

    Returns the wrapped bytes. Caller is responsible for cache writes.
    """
    if not xvr_bytes or xvr_bytes[:4] != XVRT_MAGIC:
        raise ValueError("expected XVRT bytes for XVMH wrap")
    header = bytearray(XVM_HEADER_SIZE)
    header[0:4] = XVM_MAGIC
    struct.pack_into("<I", header, 4, 0x38)  # header size declared
    struct.pack_into("<I", header, 8, 1)     # texture count
    return bytes(header) + xvr_bytes


def _materialize_inner_for_extract(path: str) -> Path:
    """Materialize a BML-inner or AFS-inner blob to disk; return the path.

    Plain (non-`#`) paths short-circuit to the regular base resolver.
    For `#`-paths, we write the decompressed inner bytes to a stable
    cache subdir so `extract_tiles` (which only takes Path inputs and
    keys its cache on stat) can do its thing. AFS-inner blobs go through
    the dedicated `formats.afs_reader.materialize_inner` cache.

    AFS-inner blobs whose first 4 bytes are ``XVRT`` (player-class
    archives store one bare XVRT per slot) are wrapped in a synthesised
    single-record XVMH archive and cached as ``<NNNN>.xvm`` next to the
    raw ``<NNNN>.xvr``. ``extract_tiles`` is given the ``.xvm`` path so
    its xvr_codec subprocess sees an XVMH-headed input.
    """
    base, inner = _split_inner_path(path)
    if inner is None:
        return _resolve_base_path(base)

    base_path = _resolve_base_path(base)
    base_ext = base_path.suffix.lower()
    if base_ext == ".afs":
        # AFS path: defer to afs_reader's content-keyed cache. The cache
        # path is itself stable across runs (keyed on archive
        # size+mtime), so we don't need our own md5 namespace.
        idx, _basename = _parse_afs_inner_name(inner)
        if idx is None:
            raise HTTPException(400, f"AFS inner must be 'NNNN_name' or 'NNNN', got {inner!r}")
        try:
            from formats import afs_reader as _afs_reader
        except ImportError as e:
            raise HTTPException(500, f"AFS reader unavailable: {e}")
        try:
            cache_path, _info = _afs_reader.materialize_inner(
                base_path, idx, CACHE_DIR, timeout=TIMEOUT_BML_PRS,
            )
        except IndexError as e:
            raise HTTPException(404, str(e))
        except (ValueError, OSError) as e:
            raise HTTPException(400, f"AFS extract failed: {e}")
        # Bare-XVR fast path: wrap in a synthesised XVMH so
        # ``extract_tiles`` can hand it to ``xvr_codec.py extract``
        # unchanged. We side-cache the wrapped bytes next to the raw
        # ``<NNNN>.xvr`` so subsequent fetches skip the wrap.
        try:
            head = cache_path.read_bytes()[:4] if cache_path.stat().st_size >= 4 else b""
        except OSError:
            head = b""
        if head == XVRT_MAGIC:
            wrapped_path = cache_path.with_suffix(".xvm")
            if not (wrapped_path.exists() and wrapped_path.stat().st_size > 0):
                try:
                    xvr_bytes = cache_path.read_bytes()
                    wrapped = _wrap_xvr_as_xvmh(xvr_bytes)
                    tmp = wrapped_path.with_suffix(wrapped_path.suffix + ".tmp")
                    tmp.write_bytes(wrapped)
                    os.replace(tmp, wrapped_path)
                except (ValueError, OSError) as e:
                    log.warning(
                        "xvr->xvmh wrap failed for %s: %s", cache_path, e,
                    )
                    return cache_path
            return wrapped_path
        return cache_path

    if base_ext != ".bml":
        raise HTTPException(
            400,
            f"`#` syntax only supported for .bml or .afs bases, got {base_ext!r}",
        )

    # Stable filename keyed on the parent BML's stat + the inner name so
    # cache_key (size + mtime of the materialized file) re-uses extracts
    # across requests but rebuilds when the BML changes.
    st = base_path.stat()
    digest = hashlib.md5(
        f"{base_path.name}|{st.st_size}|{int(st.st_mtime)}|{inner}".encode("utf-8")
    ).hexdigest()[:16]
    scratch_dir = CACHE_DIR / _BML_INNER_CACHE_SUBDIR / digest
    scratch_dir.mkdir(parents=True, exist_ok=True)
    # Use the inner name as the materialized filename. extract_tiles
    # reads the suffix to decide whether to PRS-decompress; .xvm short-
    # circuits straight to xvr_codec, which is what we want.
    out_path = scratch_dir / inner
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    # Wave 7: serialise materialisation per-inner so concurrent threads
    # (bundle pre-warm + the user's tile_png GET racing for the same
    # inner) don't both try to atomic-rename the same .tmp file.
    # Windows fails the second os.replace with PermissionError.
    lock = _get_extract_lock(f"materialize:{digest}:{inner}")
    with lock:
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        blob, _ = _extract_bml_inner_bytes(base_path, inner)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, out_path)
        return out_path


def sh(cmd: list, cwd: Optional[Path] = None, timeout: int = 60) -> str:
    """Run a command with a hard timeout; raise on failure or timeout.

    On non-zero exit, raises RuntimeError with the last 1 KB of stdout/stderr
    each. On timeout, raises HTTPException(504).
    """
    cmd_str = [str(c) for c in cmd]
    try:
        r = subprocess.run(
            cmd_str,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.warning("subprocess timeout (%ss): %s", timeout, " ".join(cmd_str))
        raise HTTPException(504, f"subprocess timeout ({timeout}s): {' '.join(cmd_str)}")
    if r.returncode != 0:
        raise RuntimeError(
            f"cmd failed: {' '.join(cmd_str)}\n"
            f"STDOUT: {r.stdout[-SUBPROCESS_OUTPUT_TAIL:]}\n"
            f"STDERR: {r.stderr[-SUBPROCESS_OUTPUT_TAIL:]}"
        )
    return r.stdout


def cache_key(prs_path: Path) -> str:
    """Stable per-file cache key: name + size + mtime."""
    st = prs_path.stat()
    return f"{prs_path.name}_{st.st_size}_{int(st.st_mtime)}"


def _is_backup_name(name: str) -> bool:
    """True if `name` looks like a backup / quarantined file."""
    nl = name.lower()
    return any(s in nl for s in BACKUP_FRAGMENTS)


def _live_keys() -> set[str]:
    """Compute the cache_key for every live file currently in DATA_DIR."""
    out: set[str] = set()
    if not DATA_DIR.exists():
        return out
    for p in DATA_DIR.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_EXT:
            continue
        if _is_backup_name(p.name):
            continue
        try:
            out.add(cache_key(p))
        except OSError:
            log.debug("could not stat %s; skipping", p)
    return out


def _is_partial_cache(cdir: Path) -> bool:
    """True if a cache dir was left in a partial state by a crashed extraction.

    A complete cache dir has at minimum: a parseable manifest.json, a tiles/
    subdir, and every PNG referenced by the manifest physically present.
    """
    manifest = cdir / "manifest.json"
    if not manifest.exists():
        return True
    tiles_dir = cdir / "tiles"
    if not tiles_dir.exists():
        return True
    try:
        with open(manifest) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return True
    tiles = m.get("tiles") or []
    if not tiles:
        return True
    for t in tiles:
        fn = t.get("filename")
        if not fn or not (tiles_dir / fn).exists():
            return True
    return False


def cleanup_cache(verbose: bool = True) -> dict:
    """Delete cache subdirs that are stale.

    A cache subdir is deleted if any of:
      - its name is not the cache_key of any live file (orphan)
      - its mtime is older than CACHE_TTL_SECONDS (expired)
      - it's structurally incomplete (partial / crashed extraction)

    Returns stats dict for /api/health observability.
    """
    now = time.time()
    live = _live_keys()
    deleted_orphan: list[str] = []
    deleted_old: list[str] = []
    deleted_partial: list[str] = []
    kept: list[str] = []
    if not CACHE_DIR.exists():
        return {
            "kept": kept, "deleted_orphan": deleted_orphan,
            "deleted_old": deleted_old, "deleted_partial": deleted_partial,
        }
    # Cache subdirectories that are NOT cache_keys but are managed
    # explicitly by the editor (e.g. battle-param staging, model cache).
    # These have stable names and must be preserved across restarts.
    PROTECTED_DIRS = {
        "battle_param_export",
        "itempmt_export",
        # Painted-texture, sculpted-mesh, and parsed-mesh caches are
        # user-authored data the cleanup loop must NOT delete.
        "painted_textures",
        "sculpted_meshes",
        "parse_cache",
        # 2026-04-25: workspace JSON snapshots (UX maturity layer).
        "workspaces",
        # Build/import staging caches. These hold user-authored content
        # (rebuilt BMLs, retargeted .njm + sidecars from /api/import/animation)
        # and MUST NOT be wiped between server restarts. Without this guard
        # the cache cleanup at startup nukes the staged preview animations
        # the editor's "Imported Animations" section depends on.
        "afs_export",
        "bml_export",
        "nj_export",
        "njm_export",
        # Rig + map + skinned-payload + binding caches are also user-
        # authored / index-style data that survives restart by design.
        "rigs",
        "map_edits",
        "skinned_payload",
        "binding",
        "blend_shape_export",
        "live_overrides",
        "afs",
        "bml_inner",
        # 2026-04-26: pro-tools edit-mode vertex transforms (Edit tab).
        # Each save is a sparse-index displacement sidecar; nuking these
        # at startup loses unsaved work between server restarts.
        "protools_edits",
    }
    for cdir in CACHE_DIR.iterdir():
        if not cdir.is_dir():
            continue
        if cdir.name in PROTECTED_DIRS:
            kept.append(cdir.name)
            continue
        try:
            age = now - cdir.stat().st_mtime
        except OSError:
            log.debug("could not stat %s; skipping", cdir)
            continue
        try:
            if cdir.name not in live:
                shutil.rmtree(cdir, ignore_errors=True)
                deleted_orphan.append(cdir.name)
            elif age > CACHE_TTL_SECONDS:
                shutil.rmtree(cdir, ignore_errors=True)
                deleted_old.append(cdir.name)
            elif _is_partial_cache(cdir):
                shutil.rmtree(cdir, ignore_errors=True)
                deleted_partial.append(cdir.name)
            else:
                kept.append(cdir.name)
        except OSError as e:
            log.warning("cache cleanup failed for %s: %s", cdir, e)
    if verbose:
        log.info(
            "cache cleanup: kept=%d orphan=%d old=%d partial=%d",
            len(kept), len(deleted_orphan), len(deleted_old), len(deleted_partial),
        )
    return {
        "kept": kept,
        "deleted_orphan": deleted_orphan,
        "deleted_old": deleted_old,
        "deleted_partial": deleted_partial,
    }


# Audit M-3 (2026-05-01): disk-tier cache aggregate caps.
#
# Both the parsed-mesh disk pickles (cache/parse_cache/v1/) and the
# materialised BML-inner blobs (cache/bml_inner/<digest>/) have per-entry
# but NO aggregate caps. The in-memory LRUs are bounded; the disk side
# grows without bound until the user notices their drive filling up.
#
# `_sweep_cache_dir` walks a directory, deletes files older than
# `max_age_days` first (those are stale by definition), then if the total
# remaining size still exceeds `max_total_bytes` it deletes oldest-first
# until under the cap. Returns a stats dict for observability. Called on
# startup for the two cache dirs; also exposed via POST /api/cache/sweep
# for ad-hoc cleanup without restarting the server.
def _sweep_cache_dir(
    dir_path: Path,
    max_total_bytes: int,
    max_age_days: int,
) -> dict:
    """Sweep `dir_path` by age then by size. Returns delete stats.

    Walks every regular file under `dir_path` recursively. Files whose
    mtime is older than `max_age_days` are deleted unconditionally.
    If total remaining size is still > `max_total_bytes`, the oldest
    surviving files are deleted (oldest-first) until the cap is met.

    Empty dirs left after deletion are NOT cleaned up — leaving the
    directory skeleton means subsequent cache writes don't have to
    re-mkdir. Cheap on disk and avoids racing concurrent writers.

    Returns a dict with keys:
      deleted_count    int   total files removed (age + size combined)
      freed_bytes      int   sum of removed files' sizes
      remaining_bytes  int   on-disk total after sweep
    """
    if not dir_path.exists():
        return {"deleted_count": 0, "freed_bytes": 0, "remaining_bytes": 0}
    cutoff_seconds = max(0, int(max_age_days)) * 86400
    now = time.time()
    # First pass: collect (path, size, mtime) for every file.
    entries: list[tuple[Path, int, float]] = []
    for p in dir_path.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        entries.append((p, st.st_size, st.st_mtime))
    deleted_count = 0
    freed_bytes = 0
    survivors: list[tuple[Path, int, float]] = []
    # Age sweep: anything older than max_age_days goes regardless of total size.
    for p, sz, mt in entries:
        if cutoff_seconds and (now - mt) > cutoff_seconds:
            try:
                p.unlink()
                deleted_count += 1
                freed_bytes += sz
            except OSError:
                # Survivor by failure: leave it on disk and let next sweep retry.
                survivors.append((p, sz, mt))
        else:
            survivors.append((p, sz, mt))
    # Size sweep: oldest-first until under the cap.
    total = sum(sz for _, sz, _ in survivors)
    if total > max_total_bytes:
        # Oldest first — sort ascending by mtime.
        survivors.sort(key=lambda t: t[2])
        for p, sz, _mt in survivors:
            if total <= max_total_bytes:
                break
            try:
                p.unlink()
                deleted_count += 1
                freed_bytes += sz
                total -= sz
            except OSError:
                continue
    remaining = max(0, total)
    return {
        "deleted_count": deleted_count,
        "freed_bytes": freed_bytes,
        "remaining_bytes": remaining,
    }


# Wave 7 (2026-04-26): per-key extract lock.
#
# extract_tiles() writes a manifest.json + tiles/ dir under a key-named
# subdir of CACHE_DIR. Two threads racing on the same key (typical
# scenario: bundle pre-warm + the user's tile_png GET both fire for the
# same model) collide on Windows because os.replace() can't rename a
# file the OS still has open in another handle. The result is a
# PermissionError (WinError 5) bubbling up as 500 Internal Server Error.
#
# Fix: a small dict of per-key locks. Each key gets one threading.Lock;
# extract_tiles takes the lock before touching the cache subdir. The
# lock dict itself is guarded by a mutex so we never create two locks
# for the same key.
_EXTRACT_LOCKS: dict[str, threading.Lock] = {}
_EXTRACT_LOCKS_GUARD = threading.Lock()


def _get_extract_lock(key: str) -> threading.Lock:
    """Lazy-init a per-key Lock and return it. Idempotent across threads."""
    with _EXTRACT_LOCKS_GUARD:
        lk = _EXTRACT_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _EXTRACT_LOCKS[key] = lk
        return lk


def extract_tiles(prs_path: Path) -> dict:
    """Decompress + extract tiles. Caches by (size, mtime). Idempotent.

    Cache layout per file:
      cache/<key>/
        <filename>           (work copy, decompressed in place if PRS)
        tiles/*.png          (one per XVRT block)
        tiles/*.xvr          (raw XVR records, used for fmt readback)
        manifest.json
        upscaled/            (per-tile upscaled outputs, populated by /api/upscale)
        compress_in/         (PRS recompression scratch, populated by /api/repack)

    Re-uses an existing cache if its manifest is intact and every referenced
    PNG still exists; otherwise blows the dir away and rebuilds.

    Wave 7: serialised per-key via _EXTRACT_LOCKS so concurrent threads
    (the bundle pre-warm pool + the user's tile_png GET) don't race on
    manifest.json's atomic write.
    """
    if not prs_path.exists():
        raise FileNotFoundError(prs_path)
    key = cache_key(prs_path)
    # Fast-path: if the cache dir is already populated AND the manifest
    # exists, AND every PNG it references is on disk, return immediately
    # without taking the lock. Avoids a serial bottleneck for warm hits
    # where every reader can read the same finalised manifest in parallel.
    cdir = CACHE_DIR / key
    manifest_path = cdir / "manifest.json"
    if cdir.exists() and manifest_path.exists():
        try:
            with open(manifest_path) as f:
                m = json.load(f)
            tdir = Path(m.get("tiles_dir", ""))
            if tdir.exists():
                if all((tdir / t["filename"]).exists() for t in m.get("tiles", [])):
                    return m
        except (OSError, json.JSONDecodeError):
            pass  # fall through to locked rebuild

    # Slow path: take the lock and (re)build under it. Other threads
    # waiting on the same key will see the finished manifest on the
    # second cdir.exists()/manifest_path.exists() check below (re-check
    # under the lock to avoid the classic double-checked-locking miss).
    lock = _get_extract_lock(key)
    with lock:
        if cdir.exists() and manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    m = json.load(f)
                # quick sanity: tile pngs all still present
                tdir = Path(m.get("tiles_dir", ""))
                if tdir.exists():
                    ok = all((tdir / t["filename"]).exists() for t in m.get("tiles", []))
                    if ok:
                        return m
            except (OSError, json.JSONDecodeError) as e:
                log.warning("corrupt cache for %s (%s); rebuilding", key, e)
            # corrupt cache - rebuild
            shutil.rmtree(cdir, ignore_errors=True)

        cdir.mkdir(parents=True, exist_ok=True)
        work_prs = cdir / prs_path.name
        shutil.copy(prs_path, work_prs)

        is_prs = prs_path.suffix.lower() == ".prs"
        if is_prs:
            # In-process PRS decompress (formats.prs, verified byte-exact) —
            # replaces the dead PuyoToolsCli.exe subprocess that made every
            # .prs texture 500 on a missing binary. Decompress in place so the
            # magic-sniff + decode below sees the inner XVMH/PVRT/PACD/etc.
            try:
                _dec = _prs.decompress(work_prs.read_bytes())
                work_prs.write_bytes(_dec)
            except Exception as e:  # noqa: BLE001 — surface a clean 502
                raise HTTPException(
                    502, f"PRS decompress failed for {prs_path.name}: {e}"
                ) from e
        # else: .xvm has no PRS layer

        tiles_dir = cdir / "tiles"

        # Magic-sniff the (possibly decompressed) work file. The legacy
        # xvr_codec path handles XVMH+XVRT only — Dreamcast/GC PVR/PVM and
        # GVR/GVM containers fall through to the in-process pvr_decode
        # path that produces equivalent per-tile PNGs in the same layout.
        try:
            head = work_prs.read_bytes()[:16]
        except OSError:
            head = b""

        if head[:4] in (b"PVRT", b"GBIX", b"PVMH"):
            # Sega Dreamcast PVR or PVM container.
            tiles = _extract_tiles_via_pvr_decode(
                work_prs, tiles_dir, archive_kind="pvr",
            )
        elif head[:4] == b"GVMH" or head[:4] == b"GVRT":
            # Gamecube GVR / GVM. Currently routed through the same
            # extractor, which records "fmt=-1, no decode" placeholders
            # for unsupported (Gamecube-only) pixel layouts. The frontend
            # surfaces the placeholder PNG so the model still binds.
            tiles = _extract_tiles_via_pvr_decode(
                work_prs, tiles_dir, archive_kind="gvr",
            )
        else:
            # XVMH/XVRT path. Switched 2026-04-26 from a subprocess call to
            # xvr_codec.py to an in-process port (formats/xvr_decode.py).
            # Subprocess spawn was ~150 ms/archive of pure overhead before
            # the actual decode; that dominated cold-load on multi-texture
            # Xbox assets. The on-disk layout is identical, so the existing
            # tile-png cache + rebuild flow are unaffected.
            try:
                blob = work_prs.read_bytes()
            except OSError as e:
                log.warning("xvr extract: read failed for %s: %s", work_prs, e)
                blob = b""
            # Find an XVMH header anywhere in the blob. NJ models occasionally
            # carry their texture archive inline after the model data; we
            # accept that case here so callers don't need to materialize a
            # separate `.nj.xvm` companion to read tiles.
            xvm_off = blob.find(_xvr_decode.XVM_MAGIC) if blob else -1
            if xvm_off >= 0:
                tiles = _xvr_decode.extract_to_dir(
                    blob[xvm_off:], tiles_dir, work_prs.stem, write_md5=True,
                )
            else:
                # No XVMH header found. Don't shell out to xvr_codec.py — it
                # will fail for the same reason. Raise a clean 400 so the
                # frontend can surface a useful diagnostic instead of a 500.
                raise HTTPException(
                    400,
                    f"no XVMH header in {work_prs.name} "
                    f"(magic={blob[:4]!r}); did you mean a `.nj.xvm` inner?",
                )

        manifest = {
            "filename": prs_path.name,
            "is_prs": is_prs,
            "tile_count": len(tiles),
            "tiles": tiles,
            "cache_dir": str(cdir),
            "tiles_dir": str(tiles_dir),
            "work_prs": str(work_prs),
            "extracted_at": int(time.time()),
        }
        # Atomic write: a crash mid-extract leaves either the previous manifest
        # or no manifest, never a half-written one.
        tmp = manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, manifest_path)
        return manifest


def _extract_tiles_via_pvr_decode(
    work_path: Path,
    tiles_dir: Path,
    *,
    archive_kind: str,
) -> list[dict]:
    """Decode a PVR / PVM / GVM container in-process to tile PNGs.

    Mirrors the on-disk layout produced by ``xvr_codec.py extract``:
        tiles/<stem>_<idx:02d>_<W>x<H>.png   per-tile PNG
        tiles/<stem>_<idx:02d>_<W>x<H>.pvr   raw inner-record bytes
                                              (sibling, used by the
                                              tile fmt readback path)

    The .pvr sibling lets the existing fmt-readback in ``extract_tiles``
    surface a usable ``fmt`` field; for PVR we record the px_format byte
    at +0x08 of the inner PVRT chunk as the "format" int (matches
    PSOBB's runtime convention of identifying texture variant by px+tex
    bytes).

    Falls through gracefully for inner records the decoder doesn't yet
    support (GVR / unknown px_format combinations) by emitting a
    1×1 magenta placeholder PNG with ``fmt=-1`` so the frontend has
    SOMETHING to render rather than hard-failing the whole archive.

    Returns the list of tile dicts the caller will store in the manifest.
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    try:
        data = work_path.read_bytes()
    except OSError as e:
        log.warning("pvr extract: read failed for %s: %s", work_path, e)
        return out

    stem = work_path.stem
    head = data[:4]
    inner_records: list[tuple[int, int]] = []

    if head == b"PVMH":
        # PVM container: walk inner PVRT records via sibling-archives helper.
        recs = _sibling_archives._parse_pvm_records(data)
        inner_records = [(o, sz) for (o, sz, _n) in recs]
    elif head == b"GVMH":
        recs = _sibling_archives._parse_gvm_records(data)
        inner_records = [(o, sz) for (o, sz, _n) in recs]
    elif head in (b"PVRT", b"GBIX"):
        # Single-record PVR.
        inner_records = [(0, len(data))]
    elif head == b"GVRT":
        inner_records = [(0, len(data))]

    for idx, (off, sz) in enumerate(inner_records):
        record = data[off:off + sz]
        # Try the in-process PVR decoder. GVR records will fail (no GVR
        # decoder yet) — emit a placeholder rather than aborting the
        # whole archive.
        png_bytes: Optional[bytes] = None
        w = h = 0
        fmt = -1
        try:
            if archive_kind == "pvr" and (
                record[:4] in (b"PVRT", b"GBIX")
                or b"PVRT" in record[:64]
            ):
                w, h, rgba = _pvr_decode.decode_pvr(record)
                # Pick out the px_format byte for the manifest's "fmt"
                # field — same role the XVR fmt int plays for XVMH.
                pvrt_off = record.find(b"PVRT")
                if pvrt_off >= 0 and pvrt_off + 0x10 <= len(record):
                    fmt = record[pvrt_off + 0x08]
                im = Image.frombytes("RGBA", (w, h), rgba)
                buf = BytesIO()
                im.save(buf, format="PNG")
                png_bytes = buf.getvalue()
        except (ValueError, IndexError, NotImplementedError) as e:
            log.info(
                "pvr_decode: tile %d/%d in %s failed: %s",
                idx, len(inner_records), work_path.name, e,
            )
            png_bytes = None

        if png_bytes is None:
            # Placeholder: 1×1 magenta-on-transparent so the frontend
            # has SOMETHING. Width/height stay 0 so callers can detect.
            im = Image.new("RGBA", (1, 1), (255, 0, 255, 255))
            buf = BytesIO()
            im.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            w, h = 1, 1

        # Filename mirrors xvr_codec's pattern so TILE_FILENAME_RE
        # (\d+_\d+x\d+\.png) keeps working downstream.
        name = f"{stem}_{idx:02d}_{w}x{h}.png"
        png_path = tiles_dir / name
        png_path.write_bytes(png_bytes)
        # Sibling raw-record file so future round-trips have the
        # source bytes (parallel to xvr_codec writing .xvr siblings).
        ext = ".pvr" if archive_kind == "pvr" else ".gvr"
        (tiles_dir / f"{stem}_{idx:02d}_{w}x{h}{ext}").write_bytes(record)
        out.append({
            "index": idx,
            "filename": name,
            "width": int(w),
            "height": int(h),
            "fmt": int(fmt),
        })

    return out


def png_to_b64(p: Path) -> str:
    """Read a PNG from disk and return a base64 data URL."""
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def _png_is_readable(p: Path) -> bool:
    """Quick PIL-side sanity check: open + verify the file is a complete PNG.

    Used to detect partial/corrupt cached intermediates from a prior crashed
    realesrgan invocation (e.g. vkCreateDevice -3 mid-write). PIL.Image.verify()
    walks chunks without decoding pixels, so it's cheap. Any exception => bad file.
    """
    try:
        with Image.open(p) as im:
            im.verify()
        # verify() leaves the image in an unusable state; reopen+load to be safe
        with Image.open(p) as im:
            im.load()
        return True
    except Exception:
        return False


def _alpha_is_meaningful(im: Image.Image) -> bool:
    """Return True when ``im`` carries an alpha channel that actually varies.

    A PNG opened in mode RGBA whose alpha plane is uniformly 255 (fully
    opaque) is functionally RGB and going through the split/recombine path
    just adds I/O and a quantisation pass. ``getextrema()`` on the alpha
    band returns ``(min, max)``; if both are 255 the channel carries no
    information.
    """
    if im.mode != "RGBA":
        return False
    extrema = im.getextrema()  # tuple of (min, max) per band; for RGBA the 4th is alpha
    if not extrema or len(extrema) < 4:
        return False
    a_min, a_max = extrema[3]
    # Treat fully-opaque as trivial. Anything else (punch-through, gradient,
    # premultiplied-with-fringe, etc.) goes through the preservation path.
    return not (a_min == 255 and a_max == 255)


def _alpha_is_binary(im: Image.Image) -> bool:
    """Return True iff the alpha channel contains only 0 or 255 values.

    Used after a Lanczos resize to decide whether we need to re-threshold
    the alpha back to binary. Punch-through DXT1 textures arrive with
    binary alpha; pure RGB sources have no alpha; only premultiplied or
    gradient-alpha textures have native fractional alpha — and those are
    rare in PSOBB's DXT1 asset pool. Defensive in either direction: when
    the source's alpha was already fractional, re-threshold would be
    wrong; we only threshold downstream if the *input* to the resize
    was binary.
    """
    if im.mode != "RGBA":
        return False
    alpha = im.split()[3]
    hist = alpha.histogram()
    # All histogram buckets except 0 and 255 must be empty for the
    # alpha to be binary.
    return sum(hist[1:255]) == 0


def _resize_rgba_preserving_binary_alpha(
    im: Image.Image, size: tuple[int, int]
) -> Image.Image:
    """Lanczos-resize an RGBA image; re-threshold alpha to binary if the
    source's alpha was binary.

    PSOBB DXT1 textures use 1-bit punch-through alpha. The realesrgan
    wrapper thresholds alpha after upscale, but ``_cascade_upscale``'s
    tail Lanczos-down step (cumulative overshoot adjust + max-dim cap)
    re-introduces fractional alpha values that DXT1 cannot encode. To
    keep the punch-through edges sharp, we re-threshold to binary at
    128 only when the input was already binary — leaving genuinely
    gradient-alpha (rare) sources untouched.
    """
    rgba = im.convert("RGBA")
    was_binary = _alpha_is_binary(rgba)
    resized = rgba.resize(size, Image.Resampling.LANCZOS)
    if was_binary:
        r, g, b, a = resized.split()
        a = a.point(lambda v: 255 if v >= 128 else 0).convert("L")
        resized = Image.merge("RGBA", (r, g, b, a))
    return resized


def _run_realesrgan(
    src: Path,
    dst: Path,
    model: str,
    binary_scale: int,
    *,
    tile_size: Optional[int] = None,
    tta: bool = False,
    gpu_id: Optional[int] = None,
) -> None:
    """Invoke realesrgan-ncnn-vulkan, preserving alpha if the source has any.

    The bundled animevideov3 model is RGB-only — realesrgan-ncnn-vulkan
    either drops the alpha channel entirely or passes it through at the
    SOURCE dimension, mismatched with the upscaled RGB. Either way the
    downstream xvr_codec rebuild re-encodes as DXT1 assuming opaque, so
    PSOBB DXT1 punch-through textures (foliage, hair edges, particles)
    end up with black holes / invisible particles in-game.

    Fix (Bug A, 2026-05-01): if the source PNG has a non-trivial alpha
    channel, we
      1. split RGB from A into two temp PNGs,
      2. run realesrgan on the RGB-only copy,
      3. Lanczos-resize the alpha channel up to match,
      4. threshold the alpha back to binary at 128 (DXT1 is 1-bit),
      5. recombine into RGBA and save to ``dst``.
    The threshold step is critical: a Lanczos resize can introduce
    intermediate alpha values which DXT1 cannot represent — we'd lose
    the punch-through edge to bilinear blur otherwise.

    Pure-RGB and trivially-opaque RGBA sources skip the wrapper and go
    straight to the binary, identical to the prior behaviour.

    binary_scale must be one of 2/3/4 (the bundled binary's hard cap).
    Caller is responsible for cascading bigger logical scales by chaining outputs.

    Reliability (2026-04-30): writes to a sibling .tmp path and renames on
    success so a crashed subprocess can never leave a partially-written PNG
    at the canonical name (root cause of the "broken PNG chunk b'\\x00\\x00\\x00\\x00'"
    cascade failure). Retries with exponential backoff up to UPSCALE_RETRY_ATTEMPTS
    times on subprocess failure, missing output, or post-write PNG-verify failure
    — covers transient Vulkan device-creation errors during GPU contention with
    the running game.
    """
    # --- Bug A: alpha preservation -------------------------------------
    # Inspect the source. If it has a meaningful alpha channel, run the
    # split-recombine wrapper around the inner RGB-only path.
    try:
        with Image.open(src) as _probe:
            _probe.load()
            mode = _probe.mode
            if mode == "RGBA" and _alpha_is_meaningful(_probe):
                rgb_part = _probe.convert("RGB").copy()
                alpha_part = _probe.split()[3].copy()
            else:
                rgb_part = None
                alpha_part = None
    except Exception:
        # If probe fails for any reason, fall through to the legacy path
        # — _run_realesrgan_inner will surface the real error in a
        # consistent way.
        rgb_part = None
        alpha_part = None

    if rgb_part is not None and alpha_part is not None:
        rgb_tmp = src.with_name(src.stem + "_rgb.png")
        alpha_tmp = src.with_name(src.stem + "_a.png")
        try:
            rgb_part.save(rgb_tmp)
            alpha_part.save(alpha_tmp)
            # Run the upscaler on the RGB-only copy.
            _run_realesrgan_inner(
                rgb_tmp, dst, model, binary_scale,
                tile_size=tile_size, tta=tta, gpu_id=gpu_id,
            )
            # Resize the alpha channel to match the upscaled RGB and
            # threshold back to binary so DXT1 can represent it.
            with Image.open(dst) as up_rgb:
                up_w, up_h = up_rgb.size
                up_rgb_rgb = up_rgb.convert("RGB")
            with Image.open(alpha_tmp) as a_im:
                a_resized = a_im.convert("L").resize(
                    (up_w, up_h), Image.Resampling.LANCZOS
                )
            # Threshold to binary (DXT1 alpha is 1-bit). PIL's point()
            # with a lambda is fine here — we're at upscaled-tile dim
            # (<= a few thousand pixels) and the function runs once per
            # upscale, not in a hot loop.
            a_binary = a_resized.point(lambda v: 255 if v >= 128 else 0).convert("L")
            r, g, b = up_rgb_rgb.split()
            merged = Image.merge("RGBA", (r, g, b, a_binary))
            merged.save(dst)
        finally:
            for stale in (rgb_tmp, alpha_tmp):
                try:
                    if stale.exists():
                        stale.unlink()
                except OSError:
                    pass
        return

    # No meaningful alpha — preserve the legacy fast path verbatim.
    _run_realesrgan_inner(
        src, dst, model, binary_scale,
        tile_size=tile_size, tta=tta, gpu_id=gpu_id,
    )


def _run_realesrgan_inner(
    src: Path,
    dst: Path,
    model: str,
    binary_scale: int,
    *,
    tile_size: Optional[int] = None,
    tta: bool = False,
    gpu_id: Optional[int] = None,
) -> None:
    """RGB-only realesrgan invocation. See _run_realesrgan for the
    alpha-aware wrapper. This is the original implementation before
    Bug A; nothing above the wrapper should call it directly."""
    # Write to a tmp sibling, rename on success. This prevents readers from
    # ever observing a half-written PNG at `dst`. The realesrgan binary
    # validates `-o` extension and rejects `.png.tmp`, so we use a tmp
    # filename whose suffix is still `.png` (`<stem>.tmp.png`).
    tmp_dst = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    cmd: list[str | Path] = [
        str(REALESRGAN),
        "-i", str(src),
        "-o", str(tmp_dst),
        "-n", model,
        "-s", str(binary_scale),
        "-f", "png",
    ]
    if tile_size is not None and tile_size in ALLOWED_TILE_SIZES:
        cmd += ["-t", str(tile_size)]
    if tta:
        cmd += ["-x"]
    if gpu_id is not None:
        cmd += ["-g", str(gpu_id)]

    last_err: Optional[BaseException] = None
    for attempt in range(1, UPSCALE_RETRY_ATTEMPTS + 1):
        # Wipe any leftover from a prior failed attempt before re-running.
        for stale in (tmp_dst, dst):
            try:
                if stale.exists():
                    stale.unlink()
            except OSError:
                pass
        try:
            sh(cmd, timeout=TIMEOUT_UPSCALE)
        except (RuntimeError, HTTPException) as e:
            last_err = e
            log.warning(
                "realesrgan attempt %d/%d failed (%s -> %s): %s",
                attempt, UPSCALE_RETRY_ATTEMPTS, src.name, dst.name,
                str(e).splitlines()[0] if str(e) else type(e).__name__,
            )
        else:
            # Subprocess returned 0. Now confirm a readable PNG actually landed.
            if tmp_dst.exists() and _png_is_readable(tmp_dst):
                try:
                    os.replace(tmp_dst, dst)
                    return
                except OSError as e:
                    last_err = e
                    log.warning("realesrgan rename failed: %s", e)
            else:
                last_err = RuntimeError(
                    f"realesrgan exited 0 but produced no readable PNG at {tmp_dst}"
                )
                log.warning(
                    "realesrgan attempt %d/%d: exit 0 but output missing/corrupt (%s)",
                    attempt, UPSCALE_RETRY_ATTEMPTS, dst.name,
                )
        # Backoff before next attempt (skip after final attempt).
        if attempt < UPSCALE_RETRY_ATTEMPTS:
            backoff = UPSCALE_RETRY_BACKOFF_SECONDS[
                min(attempt - 1, len(UPSCALE_RETRY_BACKOFF_SECONDS) - 1)
            ]
            time.sleep(backoff)

    # All attempts failed. Make sure no garbage is left at dst.
    for stale in (tmp_dst, dst):
        try:
            if stale.exists():
                stale.unlink()
        except OSError:
            pass
    if isinstance(last_err, HTTPException):
        raise last_err
    raise RuntimeError(
        f"realesrgan failed after {UPSCALE_RETRY_ATTEMPTS} attempts: {last_err}"
    )


def _cascade_upscale(
    src: Path,
    target_dir: Path,
    base_name: str,
    model: str,
    requested_scale: int,
    *,
    tile_size: Optional[int] = None,
    tta: bool = False,
    gpu_id: Optional[int] = None,
) -> Path:
    """Run the upscaler at the model's NATIVE scale, repeated as needed,
    then Lanczos-resize to the requested target scale if it overshoots.

    Critical bug fix (2026-04-25): the prior implementation greedy-decomposed
    `requested_scale` into factors {4,3,2} and asked the binary for `-s 2`
    on a 4x-trained model like `realesrgan-x4plus-anime`. The bundled
    `realesrgan-ncnn-vulkan.exe` accepts -s 2/3/4 syntactically, but a model
    trained for 4x produces garbage when asked for `-s 2` (silhouettes + content
    drift were visible in cached step2 outputs). Models like `realesr-animevideov3-x2`
    are SEPARATE binaries for each scale.

    Correct approach: always invoke the model at its native scale. Repeat until
    cumulative >= requested. Then Lanczos-down to the exact requested scale.

    Examples (model_native = 4):
      requested 4  -> 1 pass (4x). Done.
      requested 6  -> 1 pass (4x), then would still need more. -> 2 passes (16x), Lanczos to 6x.
      requested 8  -> 2 passes (16x), Lanczos to 8x.
      requested 12 -> 2 passes (16x), Lanczos to 12x.
      requested 16 -> 2 passes (16x). Done.

    Examples (model_native = 2):
      requested 2  -> 1 pass.
      requested 3  -> 2 passes (4x), Lanczos to 3x.
      requested 4  -> 2 passes.
      requested 8  -> 3 passes.
      requested 16 -> 4 passes.

    Returns path to the final PNG inside `target_dir`, sized to exactly
    src_dim * requested_scale on each axis.
    """
    if requested_scale <= 0:
        raise HTTPException(400, "scale must be positive")

    if requested_scale == 1:
        out = target_dir / f"{base_name}_x1.png"
        if not out.exists():
            shutil.copy(src, out)
        return out

    native = int(model_meta(model).get("native_scale", 4))
    if native < 2:
        native = 2  # safety

    # Read source dim — we need it to compute the exact target after Lanczos
    with Image.open(src) as _im:
        src_w, src_h = _im.size

    # --- Bug B: max-texture-dim cap ------------------------------------
    # PSOBB silently fails to load textures whose dimensions exceed
    # PSOBB_MAX_TEXTURE_DIM (1024) — this caused black trees + missing
    # splash particles when scale=8 produced 2048x2048 outputs.
    #
    # Two clamps below:
    #   (a) If the SOURCE is already at/above the cap on either axis,
    #       there is nothing to gain from running the upscaler: copy the
    #       source through and return it. Per spec we do NOT downscale a
    #       source that's already > cap (rare; preserve user data).
    #   (b) If the upscaled OUTPUT would exceed the cap on either axis,
    #       we let it cascade then Lanczos-down at the end. That keeps
    #       cumulative-aware caching of the full-scale intermediates and
    #       only adds a single resize step at the tail.
    if src_w >= PSOBB_MAX_TEXTURE_DIM or src_h >= PSOBB_MAX_TEXTURE_DIM:
        out = target_dir / f"{base_name}_x1.png"
        if not out.exists():
            shutil.copy(src, out)
        log.info(
            "upscale clamp: %s: source already at cap (%dx%d), no upscale (requested %dx)",
            base_name, src_w, src_h, requested_scale,
        )
        return out

    # Determine number of native-scale passes needed to reach or exceed requested
    passes_needed = 1
    cumulative = native
    while cumulative < requested_scale:
        passes_needed += 1
        cumulative *= native
    # Cap at a sane limit to prevent runaway
    if passes_needed > 6:
        raise HTTPException(400, f"requested_scale {requested_scale} requires {passes_needed} passes (cap 6)")

    cur = src
    cum = 1
    for i in range(passes_needed):
        cum *= native
        out = target_dir / f"{base_name}_step{i+1}_x{cum}.png"
        # Validate any pre-existing cached intermediate. A partial PNG from a
        # prior crashed attempt (e.g. vkCreateDevice -3 mid-write before the
        # atomic-rename guard existed) would otherwise poison the next pass
        # and surface as a "broken PNG chunk" SyntaxError downstream.
        if out.exists() and not _png_is_readable(out):
            log.warning("cascade: discarding corrupt cached intermediate %s", out.name)
            try:
                out.unlink()
            except OSError:
                pass
        if not out.exists():
            _run_realesrgan(
                cur, out, model, native,  # ALWAYS native scale - never asks the model for non-native
                tile_size=tile_size, tta=tta, gpu_id=gpu_id,
            )
            if not out.exists():
                raise HTTPException(500, f"upscaler step {i+1} produced no output")
        cur = out

    # If cumulative overshot the requested, Lanczos-down to the exact requested scale
    if cum != requested_scale:
        target_w = src_w * requested_scale
        target_h = src_h * requested_scale
        adj_out = target_dir / f"{base_name}_x{requested_scale}.png"
        if adj_out.exists() and not _png_is_readable(adj_out):
            log.warning("cascade: discarding corrupt cached overshoot %s", adj_out.name)
            try:
                adj_out.unlink()
            except OSError:
                pass
        if not adj_out.exists():
            with Image.open(cur) as im:
                # Re-threshold binary alpha after Lanczos so DXT1
                # punch-through edges survive the cumulative-overshoot
                # adjustment. See _resize_rgba_preserving_binary_alpha.
                resized = _resize_rgba_preserving_binary_alpha(
                    im, (target_w, target_h)
                )
                resized.save(adj_out)
        cur = adj_out

    # --- Bug B: clamp newly-upscaled output to PSOBB_MAX_TEXTURE_DIM ----
    # The cap applies to outputs the upscaler produced; sources already
    # over the cap are short-circuited at the top of this function.
    # Aspect ratio is preserved so non-square tiles don't squish: we
    # compute a single scale factor based on whichever axis is most
    # over-cap and apply it to both. E.g. 256x128 @ 8x = 2048x1024 →
    # max axis 2048, ratio 1024/2048 = 0.5 → 1024x512.
    with Image.open(cur) as _im:
        out_w, out_h = _im.size
    if out_w > PSOBB_MAX_TEXTURE_DIM or out_h > PSOBB_MAX_TEXTURE_DIM:
        # Aspect-preserving fit-within-cap. Use float math so a 2048-wide
        # tile downscales to exactly 1024 with no rounding error; only
        # the *other* axis can pick up sub-pixel rounding.
        cap = float(PSOBB_MAX_TEXTURE_DIM)
        ratio = min(cap / out_w, cap / out_h)
        capped_w = int(round(out_w * ratio))
        capped_h = int(round(out_h * ratio))
        # Force at least one axis to land exactly on the cap (rounding
        # could otherwise drop 1024.0 to 1023). Clamp the result to the
        # cap so floating-point drift can never produce 1025.
        capped_w = min(capped_w, PSOBB_MAX_TEXTURE_DIM)
        capped_h = min(capped_h, PSOBB_MAX_TEXTURE_DIM)
        # Effective scale (relative to source) after clamp. With aspect
        # preserved both axes share the same effective scale; we just
        # report it once.
        eff_scale = capped_w / src_w if src_w else 0
        log.info(
            "upscale clamp: %s tile: %d× → %.2f× (max dim cap %d, %d×%d → %d×%d)",
            base_name, requested_scale, eff_scale,
            PSOBB_MAX_TEXTURE_DIM, out_w, out_h, capped_w, capped_h,
        )
        capped_out = target_dir / f"{base_name}_capped.png"
        if capped_out.exists() and not _png_is_readable(capped_out):
            try:
                capped_out.unlink()
            except OSError:
                pass
        if not capped_out.exists():
            with Image.open(cur) as im:
                resized = _resize_rgba_preserving_binary_alpha(
                    im, (capped_w, capped_h)
                )
                resized.save(capped_out)
        cur = capped_out

    return cur


# ============================================================================
# Live-reload cache watcher (v5 polish, 2026-04-25)
# ----------------------------------------------------------------------------
# A lightweight mtime-poll thread observes a curated set of cache subdirs and
# dispatches SSE events to every connected /api/events subscriber. This lets
# the frontend refresh bound textures, motion pickers, and recent-imports
# strips without the user having to F5 the browser after a build script
# drops a new file in cache/.
#
# Why a poll loop over watchdog?
#   * watchdog isn't installed on the dev host (and we want zero new deps).
#   * The watched set is small (7 dirs, 0-200 files each) so a one-second
#     scan costs ~1-3 ms — well under the noise floor.
#   * Cross-platform without the Windows ReadDirectoryChangesW dance.
#
# Event coalescing:
#   * The poller batches all changes seen in one sweep into one wake of the
#     dispatcher, but each path produces its own event on the wire — the
#     200 ms throttle on the FRONTEND side handles fast-write spam (per the
#     spec). We DO de-dup multiple events for the same (path, kind) inside
#     one sweep so a save that touches both the .njm and .preview.json
#     within microseconds doesn't double-fire.
# ============================================================================
LIVE_RELOAD_WATCH_DIRS = (
    "painted_textures",
    "sculpted_meshes",
    "njm_export",
    "bml_export",
    "nj_export",
    "itempmt_export",
    "battle_param_export",
)
# Tunables — overridable via env so test suites can crank the poll way
# down for a faster smoke loop. Both env vars are read once at module
# import; tweaking after start is a no-op.
LIVE_RELOAD_POLL_SECONDS = float(os.environ.get("PSO_LIVE_RELOAD_POLL_SECONDS") or 1.0)
LIVE_RELOAD_QUEUE_MAX = int(os.environ.get("PSO_LIVE_RELOAD_QUEUE_MAX") or 256)


class _LiveReloadHub:
    """In-process pub/sub for cache-change events.

    Subscribers receive a per-connection ``asyncio.Queue``; the watcher
    thread pushes events into every queue via the running event loop's
    ``call_soon_threadsafe`` so producers never await. A bounded queue
    drops oldest events when a subscriber stalls, preventing unbounded
    memory growth in the rare case where a panel's SSE handler hangs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        # Watcher thread reads/writes mtime_state via _state_lock. The
        # subscribers list above uses _lock; both are leaf locks so the
        # combination is safe (no nested acquire).
        self._state_lock = threading.Lock()
        self._mtime_state: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._initialized = False
        self._first_scan_done = threading.Event()

    # ----- subscriber lifecycle (called from request handlers) --------
    def subscribe(self) -> tuple[asyncio.AbstractEventLoop, asyncio.Queue]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=LIVE_RELOAD_QUEUE_MAX)
        with self._lock:
            self._subs.append((loop, q))
        return loop, q

    def unsubscribe(self, loop: asyncio.AbstractEventLoop, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._subs.remove((loop, q))
            except ValueError:
                pass

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    # ----- publishing (called from watcher thread + endpoints) --------
    def publish(self, event: dict) -> None:
        """Fan out one event to every active subscriber."""
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(self._enqueue, q, event)
            except RuntimeError:
                # Loop was closed before we could deliver — sweep on next pub.
                self.unsubscribe(loop, q)

    @staticmethod
    def _enqueue(q: asyncio.Queue, event: dict) -> None:
        """Drop-oldest enqueue. Runs ON the subscriber's event loop."""
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            # Bounded above — should be unreachable, but log if so.
            log.warning("live_reload: dropped event for full queue")

    # ----- watcher thread ---------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Seed the mtime_state once synchronously so the first poll iter
        # doesn't fire spurious "create" events for files that already
        # existed at startup. Without this seeding, restarting the
        # server with a non-empty cache/njm_export/ would emit one
        # event per file on every restart.
        self._seed_initial_state()
        self._stop.clear()
        t = threading.Thread(
            target=self._run,
            name="pso-live-reload-watcher",
            daemon=True,
        )
        t.start()
        self._thread = t

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)

    def _seed_initial_state(self) -> None:
        snap = self._snapshot()
        with self._state_lock:
            self._mtime_state = snap
            self._initialized = True
        self._first_scan_done.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:  # never let the watcher die
                log.warning("live_reload watcher tick failed: %s", e)
            # Wait with stop-honour so shutdown is prompt.
            self._stop.wait(LIVE_RELOAD_POLL_SECONDS)

    def _tick(self) -> None:
        new_snap = self._snapshot()
        with self._state_lock:
            old_snap = self._mtime_state
            self._mtime_state = new_snap
        events: list[dict] = []
        for rel, mtime in new_snap.items():
            old = old_snap.get(rel)
            if old is None:
                events.append({"path": rel, "kind": "create"})
            elif old != mtime:
                events.append({"path": rel, "kind": "modify"})
        for rel in old_snap:
            if rel not in new_snap:
                events.append({"path": rel, "kind": "delete"})
        for ev in events:
            self.publish(ev)

    def _snapshot(self) -> dict[str, float]:
        """Walk every watched dir; return {rel_path: mtime_ns_as_float}."""
        out: dict[str, float] = {}
        for sub in LIVE_RELOAD_WATCH_DIRS:
            d = CACHE_DIR / sub
            if not d.is_dir():
                continue
            try:
                for entry in d.iterdir():
                    # Top-level files only — staged exports never nest.
                    # If a future panel writes nested dirs we can revisit.
                    if not entry.is_file():
                        continue
                    try:
                        st = entry.stat()
                    except OSError:
                        continue
                    rel = f"cache/{sub}/{entry.name}"
                    out[rel] = float(st.st_mtime_ns)
            except OSError:
                continue
        return out

    # ----- introspection (used by /api/events/status + tests) ---------
    def snapshot_state(self) -> dict:
        with self._state_lock:
            count = len(self._mtime_state)
            initialized = self._initialized
        return {
            "watched_dirs": list(LIVE_RELOAD_WATCH_DIRS),
            "tracked_files": count,
            "subscribers": self.subscriber_count(),
            "initialized": initialized,
            "poll_seconds": LIVE_RELOAD_POLL_SECONDS,
        }

    def force_rescan(self) -> int:
        """Re-snapshot now and dispatch any deltas. Returns event count."""
        new_snap = self._snapshot()
        with self._state_lock:
            old = self._mtime_state
            self._mtime_state = new_snap
        n = 0
        for rel, mtime in new_snap.items():
            o = old.get(rel)
            if o is None:
                self.publish({"path": rel, "kind": "create"})
                n += 1
            elif o != mtime:
                self.publish({"path": rel, "kind": "modify"})
                n += 1
        for rel in old:
            if rel not in new_snap:
                self.publish({"path": rel, "kind": "delete"})
                n += 1
        return n


_LIVE_RELOAD_HUB = _LiveReloadHub()


# ---------------------------------------------------------------------------- app
def _startup_prewarm() -> None:
    """Warm the manifest caches into memory (runs off the request path).

    Loads the full manifest + lite projection + health summary so the
    in-memory slots (_PARSED_MANIFEST_CACHE, _PARSED_LITE_CACHE,
    _MANIFEST_SUMMARY_CACHE) are populated before the first user request.
    Best-effort: any failure is logged and swallowed — a cold first
    request still works, just without the prewarm head start.
    """
    try:
        t0 = time.perf_counter()
        manifest_mod.cache_manifest(DATA_DIR, cache_dir=CACHE_DIR)
        manifest_mod.cache_manifest_lite(DATA_DIR, cache_dir=CACHE_DIR)
        manifest_mod.manifest_summary(DATA_DIR, cache_dir=CACHE_DIR)
        log.info(
            "startup: manifest prewarm done in %.1fms",
            (time.perf_counter() - t0) * 1000.0,
        )
    except Exception as e:  # pragma: no cover - best-effort warmer
        log.warning("startup: manifest prewarm failed: %s", e)


def _kick_startup_prewarm() -> None:
    """Spawn the manifest prewarm on a daemon thread (non-blocking)."""
    if os.environ.get("PSO_DISABLE_STARTUP_PREWARM", "0") in ("1", "true", "True"):
        return
    threading.Thread(
        target=_startup_prewarm, name="pso-startup-prewarm", daemon=True,
    ).start()


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    """Startup: log banner, run cache cleanup, start cache watcher.

    Shutdown: stop watcher, log clean exit.
    """
    log.info("startup: PSOBB Studio v%s", VERSION)
    log.info("startup: data dir = %s", DATA_DIR)
    log.info("startup: cache dir = %s", CACHE_DIR)
    cleanup_cache(verbose=True)
    # Audit M-3: disk-tier cache aggregate caps. Sweep each on startup;
    # the live in-memory LRUs are unaffected. Caps chosen to comfortably
    # hold a full-coverage parse (parse_cache ≈ ~400 MB observed on a
    # complete dragon walk) plus headroom; bml_inner is materialised
    # blobs of inner xvm/nj bytes so 500 MB is generous.
    try:
        ps = _sweep_cache_dir(PARSE_CACHE_DIR, 1024 * 1024 * 1024, 30)
        log.info(
            "startup: parse_cache sweep deleted=%d freed=%d remaining=%d",
            ps["deleted_count"], ps["freed_bytes"], ps["remaining_bytes"],
        )
    except Exception as e:
        log.warning("startup: parse_cache sweep failed: %s", e)
    try:
        bs = _sweep_cache_dir(CACHE_DIR / _BML_INNER_CACHE_SUBDIR,
                              500 * 1024 * 1024, 30)
        log.info(
            "startup: bml_inner sweep deleted=%d freed=%d remaining=%d",
            bs["deleted_count"], bs["freed_bytes"], bs["remaining_bytes"],
        )
    except Exception as e:
        log.warning("startup: bml_inner sweep failed: %s", e)
    _LIVE_RELOAD_HUB.start()
    log.info(
        "startup: live-reload watcher active (dirs=%s, poll=%.2fs)",
        list(LIVE_RELOAD_WATCH_DIRS), LIVE_RELOAD_POLL_SECONDS,
    )
    # Background prewarm: load the manifest + lite + health-summary into
    # memory so the first /api/manifest_lite, /api/manifest, and /api/health
    # don't each pay a cold JSON parse on the user's first interaction.
    # Runs in a daemon thread so it never delays first paint / readiness.
    _kick_startup_prewarm()
    yield
    _LIVE_RELOAD_HUB.stop()
    log.info("shutdown: clean exit")


app = FastAPI(title="PSOBB Studio", version=VERSION, lifespan=lifespan)

# Compress JSON-heavy endpoints. The minimum_size guard skips tiny payloads
# (no point gzipping a 200-byte health response) but kicks in well below the
# typical model_bundle / manifest_lite payload sizes.
#
# compresslevel=1 (perf 2026-06-19): Starlette defaults to level 9, the
# SLOWEST setting. For our payloads the marginal size win from 1->9 is
# negligible (manifest_lite 1.65 MB: lvl1 -> 90 KB / lvl9 -> 73 KB, both
# ~95% off) but the CPU cost ~5×es (5.4 ms vs 25.6 ms). For the base64
# tile payloads (poorly-compressible PNG entropy) level 9 is pure waste
# (~30 ms for a 23% reduction). Since the dominant consumer is the
# local editor over loopback, the smaller-but-cheaper level-1 frame is a
# strict win on time-to-byte; remote clients still get ~95% reduction.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)


# ---------------------------------------------------------------------------- API
def _manifest_health_summary() -> dict:
    """Summarize the cached manifest for /api/health.

    Reports entry count + last_built epoch. Delegates to the cheap,
    stat-memoized ``manifest.manifest_summary`` so a warm /api/health
    poll never re-json.load()s the 3.8 MB manifest.json (it only parses
    once per manifest revision, then serves from an in-memory slot).
    Never raises — a missing / unreadable cache reports zeroes so the
    rest of /api/health stays green.
    """
    try:
        return manifest_mod.manifest_summary(DATA_DIR, cache_dir=CACHE_DIR)
    except Exception as e:  # pragma: no cover - defensive net
        log.debug("manifest health summary failed: %s", e)
        cf = manifest_mod.cache_path_for(DATA_DIR, cache_dir=CACHE_DIR)
        return {"entries": 0, "last_built": 0, "path": str(cf)}


@app.get("/api/health")
def api_health():
    """Service health + tool resolution. Used by frontend banner."""
    tools = {
        "puyo": {"path": str(PUYO), "exists": PUYO.exists()},
        "xvr_codec": {"path": str(XVR_CODEC), "exists": XVR_CODEC.exists()},
        "realesrgan": {"path": str(REALESRGAN), "exists": REALESRGAN.exists()},
        "realesrgan_models_dir": {
            "path": str(REALESRGAN_MODELS),
            "exists": REALESRGAN_MODELS.exists(),
        },
        "data_dir": {"path": str(DATA_DIR), "exists": DATA_DIR.exists()},
        "manifest": _manifest_health_summary(),
    }
    # Only gate `ok` on entries that have an `exists` flag — `manifest`
    # is reporting-only, the cache may legitimately be cold on first boot.
    all_ok = all(v.get("exists", True) for v in tools.values())
    return {
        "ok": all_ok,
        "version": VERSION,
        "tools_resolved": tools,
        "cache_dir": str(CACHE_DIR),
        "python": str(PYEXE),
        "locks": {
            "upscale": len(_UPSCALE_LOCKS),
            "repack": len(_REPACK_LOCKS),
            "export_tokens": len(_EXPORT_TOKENS),
            "promote_held": _PROMOTE_LOCK.locked(),
        },
    }


@app.get("/api/manifest")
def api_manifest(request: Request, force: int = 0):
    """Return the cached asset manifest for the active DATA_DIR.

    The manifest is rebuilt on demand whenever any tracked file's mtime
    exceeds the cache file's mtime; otherwise the cached content is
    served verbatim.

    Query params:
      ``force=1``: bypass the in-memory mtime cache (the staleness
        check normally only walks the install tree once per minute).
        Useful after an external rebuild lands files in DATA_DIR.

    ETag: based on the cache file's mtime + entry count, so HTTP
    conditional GETs from the frontend short-circuit the heavy walk on
    repeated calls.

    Body-size guard: refuses to materialize a manifest larger than
    ``MAX_MANIFEST_RESPONSE_BYTES`` bytes.
    """
    try:
        m = manifest_mod.cache_manifest(DATA_DIR, cache_dir=CACHE_DIR,
                                        force=bool(force))
    except OSError as e:
        log.exception("manifest build failed")
        raise HTTPException(500, f"manifest build failed: {e}")

    cf = manifest_mod.cache_path_for(DATA_DIR, cache_dir=CACHE_DIR)
    etag_src = f'{int(cf.stat().st_mtime)}-{len(m.get("entries", []))}' if cf.exists() else "0-0"
    etag = f'W/"{etag_src}"'

    inm = request.headers.get("if-none-match")
    if inm and inm == etag:
        return JSONResponse(status_code=304, content=None, headers={"ETag": etag})

    # Body-size guard. We pre-serialize once for the size check; the JSONResponse
    # path will re-serialize, so for very large payloads we could optimize later
    # but at current scale (~5900 entries / ~1.7 MB) the cost is negligible.
    body = json.dumps(m, sort_keys=True)
    if len(body) > MAX_MANIFEST_RESPONSE_BYTES:
        raise HTTPException(
            413,
            f"manifest too large to serialize ({len(body)} > {MAX_MANIFEST_RESPONSE_BYTES} bytes)",
        )
    return JSONResponse(content=m, headers={"ETag": etag})


# Serialized-JSON-bytes memo for /api/manifest_lite, keyed on the lite
# cache file's (mtime_ns, size). Skips the ~20 ms json.dumps of the
# 1.65 MB lite dict on every warm call.
_LITE_SER_CACHE: dict = {}
_LITE_SER_CACHE_LOCK = threading.Lock()


@app.get("/api/manifest_lite")
def api_manifest_lite(request: Request, force: int = 0):
    """Return a SLIM projection of the asset manifest (Phase 0.5 perf).

    Each entry carries only ``path``, ``category``, ``inferred_category``,
    ``size``, and ``parent_archive``. At ~50 B per entry x 9 k entries
    the wire size is ~470 KB raw / ~110 KB gzipped — ~10× smaller than
    /api/manifest's 3.8 MB payload. The full per-entry shape
    (matched_textures, warnings, format) lazy-loads via
    ``GET /api/asset/<path>`` when a user clicks a tree leaf.

    Query params:
      ``force=1``: bypass the in-memory mtime cache. See /api/manifest.

    ETag: derived from the lite cache file's mtime + entry count.
    """
    # Conditional-GET short-circuit FIRST, off a cheap stat (no dict load
    # needed when the ETag matches). We can build the ETag from the lite
    # file stat alone because cache_manifest_lite's freshness contract
    # guarantees a stale file is rewritten (new mtime) before serving.
    cf = manifest_mod.lite_cache_path_for(DATA_DIR, cache_dir=CACHE_DIR)

    try:
        m = manifest_mod.cache_manifest_lite(
            DATA_DIR, cache_dir=CACHE_DIR, force=bool(force),
        )
    except OSError as e:
        log.exception("manifest_lite build failed")
        raise HTTPException(500, f"manifest_lite build failed: {e}")

    etag_src = (
        f'{int(cf.stat().st_mtime)}-{len(m.get("entries", []))}'
        if cf.exists() else "0-0"
    )
    etag = f'W/"lite-{etag_src}"'

    inm = request.headers.get("if-none-match")
    if inm and inm == etag:
        return JSONResponse(status_code=304, content=None, headers={"ETag": etag})

    # Serialized-bytes memo: json.dumps of the 1.65 MB lite dict costs
    # ~20 ms PER CALL even when the dict itself is cached. Cache the
    # encoded JSON bytes keyed on the lite file's stat so warm calls skip
    # the re-serialize entirely; the GZip middleware still compresses the
    # wire body. Keyed on (mtime_ns, size) like the other lite caches, so
    # an atomic rewrite invalidates it. Bypassed on ?force=1.
    body = None
    if not force:
        try:
            st = cf.stat()
            ser_key = (int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            ser_key = None
        if ser_key is not None:
            with _LITE_SER_CACHE_LOCK:
                slot = _LITE_SER_CACHE.get("slot")
                if slot is not None and slot[0] == ser_key:
                    body = slot[1]
            if body is None:
                body = json.dumps(
                    m, ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")
                with _LITE_SER_CACHE_LOCK:
                    _LITE_SER_CACHE["slot"] = (ser_key, body)
    if body is not None:
        return Response(
            content=body,
            media_type="application/json",
            headers={"ETag": etag},
        )
    return JSONResponse(content=m, headers={"ETag": etag})


@app.get("/api/asset/{path:path}")
def api_asset(path: str):
    """Return the full AssetEntry for ``path`` from the cached manifest.

    Companion to /api/manifest_lite: the lite endpoint omits per-entry
    detail (matched_textures, warnings, format) to keep the cold-load
    payload small. Tree leaves fetch detail through this endpoint when
    the user clicks them.

    Returns 404 when the path is not in the cached manifest. Looking
    up an entry that exists on disk but hasn't been classified yet
    (e.g. a freshly-deployed file before the manifest is rebuilt) will
    return 404 too — the caller can hit /api/manifest?force=1 to
    refresh first.
    """
    if not path:
        raise HTTPException(400, "missing path")
    entry = manifest_mod.lookup_entry(DATA_DIR, path, cache_dir=CACHE_DIR)
    if entry is None:
        raise HTTPException(404, f"no manifest entry for {path!r}")
    return entry


@app.get("/api/manifest/categories")
def api_manifest_categories():
    """Per-category counts for the asset-tree tab strip.

    Returns ``{"categories":[{"name":"texture","count":232},...],"total":2302}``
    with categories ordered to match ``manifest.schema.json``'s enum.

    This endpoint is a thin projection over the cached manifest so the
    frontend doesn't have to scan all entries client-side just to render
    the tab strip / show coverage at a glance.
    """
    try:
        m = manifest_mod.cache_manifest(DATA_DIR, cache_dir=CACHE_DIR)
    except OSError as e:
        log.exception("manifest build failed")
        raise HTTPException(500, f"manifest build failed: {e}")
    counts: dict = {}
    for entry in m.get("entries", []):
        if not entry or entry.get("deprecated"):
            continue
        cat = entry.get("category") or "unknown"
        counts[cat] = counts.get(cat, 0) + 1
    # Stable order matches the schema enum so the UI tab strip is
    # deterministic across runs.
    schema_order = (
        "texture", "model", "container",
        "quest", "map", "audio",
        "ui", "script", "cinematic",
        "metadata", "unknown",
    )
    out = []
    for name in schema_order:
        if name in counts:
            out.append({"name": name, "count": counts[name]})
    # Append any categories present in the manifest but not in our enum
    # (defensive — if the schema grows and we forget to update here).
    for name in sorted(counts):
        if name not in schema_order:
            out.append({"name": name, "count": counts[name]})
    return {"categories": out, "total": sum(counts.values())}


# Content-Type lookup keyed by lowercase extension. Anything not in this
# table falls back to application/octet-stream — that's the right default
# for opaque PSOBB binaries the frontend hex-dumps.
_RAW_CONTENT_TYPES: dict[str, str] = {
    ".prs":  "application/x-prs",
    ".xvm":  "application/x-xvm",
    ".xvr":  "application/x-xvr",
    ".bml":  "application/x-bml",
    ".nj":   "application/x-nj",
    ".njm":  "application/x-njm",
    ".afs":  "application/x-afs",
    ".rel":  "application/x-rel",
    ".dat":  "application/octet-stream",
    ".bin":  "application/octet-stream",
    ".evt":  "application/octet-stream",
    ".pae":  "application/octet-stream",
    ".gsl":  "application/octet-stream",
    ".pr2":  "application/octet-stream",
    ".pr3":  "application/octet-stream",
    ".prc":  "application/octet-stream",
    ".lst":  "text/plain; charset=utf-8",
    ".txt":  "text/plain; charset=utf-8",
    ".png":  "image/png",
    ".ogg":  "audio/ogg",
}


@app.get("/api/raw/{path:path}")
def api_raw(path: str, offset: int = 0, limit: int = 0):
    """Serve any manifest entry's raw bytes for previewing.

    Resolves ``path`` under DATA_DIR first, then LIVE_DATA_DIR (read-only).
    Returns 16 MB max — bigger files would require streaming, which the
    JSON-buffered FastAPI response model doesn't fit.

    Accepts BML-inner ``<base>#<inner>`` paths in addition to plain
    filenames; the inner blob is decompressed and streamed back with the
    Content-Type derived from the inner's own extension. The asset
    router uses this for the download-link fallback so the user can
    grab the inner XVM bytes from a single hop.

    The Content-Type comes from the file extension (see ``_RAW_CONTENT_TYPES``);
    unknown extensions get ``application/octet-stream`` so the browser
    treats the bytes as opaque.

    Used by the asset-tree's audio/hex/text fallbacks. NOT used by the
    tile editor (which has its own decoded PNG pipeline) or the model
    viewer (which uses ``/api/model_mesh``).

    Wave 7 (2026-04-26): added optional ``?offset=&limit=`` for chunked
    hex-view loading. Without these, behaviour is identical to before
    (full file response). With them, the response carries:

      * a slice ``[offset, offset+limit)`` of the file bytes,
      * ``X-Asset-Total`` = full file size,
      * ``X-Asset-Offset`` / ``X-Asset-Limit`` = the served range,
      * the ``Content-Length`` reflects the slice (not the total).

    The JSON-buffer 16 MB cap still applies, but per-slice — so a 50 MB
    asset can be paginated 16 MB at a time.

    Returns
    -------
    200 - file bytes with the right Content-Type
    400 - invalid filename / path traversal / out-of-range offset
    404 - file missing in both data dirs
    413 - file is bigger than ``MAX_RAW_RESPONSE_BYTES`` and no slicing requested
    """
    if offset < 0 or limit < 0:
        raise HTTPException(400, "offset and limit must be non-negative")

    base, inner = _split_inner_path(path)
    if inner is None:
        # Fast path: plain file. Stream from disk via FileResponse so we
        # keep the existing zero-copy behaviour.
        p = _resolve_under_roots(
            base,
            (DATA_DIR, LIVE_DATA_DIR),
            label="path",
            missing_msg=f"asset not found in DATA_DIR or LIVE_DATA_DIR: {base}",
        )
        sz = p.stat().st_size
        ext = p.suffix.lower()
        media_type = _RAW_CONTENT_TYPES.get(ext, "application/octet-stream")
        if offset or limit:
            # Sliced response. Fast for 50 MB files: read a window not
            # the whole thing. Caps the slice at MAX_RAW_RESPONSE_BYTES
            # so a malicious client can't OOM the worker by asking for
            # `?limit=999999999` on a giant file.
            if offset >= sz:
                raise HTTPException(400, f"offset {offset} >= size {sz}")
            cap = limit if limit > 0 else MAX_RAW_RESPONSE_BYTES
            cap = min(cap, MAX_RAW_RESPONSE_BYTES)
            end = min(sz, offset + cap)
            try:
                with p.open("rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read(end - offset)
            except OSError as e:
                raise HTTPException(500, f"read failed: {e}")
            return Response(
                content=chunk,
                media_type=media_type,
                headers={
                    "X-Asset-Size": str(len(chunk)),
                    "X-Asset-Total": str(sz),
                    "X-Asset-Offset": str(offset),
                    "X-Asset-Limit": str(end - offset),
                    "X-Asset-Ext": ext,
                },
            )
        if sz > MAX_RAW_RESPONSE_BYTES:
            raise HTTPException(
                413,
                f"file too large for raw endpoint: {sz} > {MAX_RAW_RESPONSE_BYTES} bytes",
            )
        headers = {
            # Filename hint; browsers display this in the audio player title etc.
            # Don't force download — many callers want inline rendering (audio
            # element, hex dump). Leave Content-Disposition out by default.
            "X-Asset-Size": str(sz),
            "X-Asset-Ext": ext,
        }
        return FileResponse(p, media_type=media_type, headers=headers)

    # BML-inner path. Materialize via the shared resolver so we get the
    # same path-validation + extract pipeline as the tile endpoints.
    blob, logical = resolve_asset_bytes(path)
    inner_ext = Path(logical).suffix.lower()
    media_type = _RAW_CONTENT_TYPES.get(inner_ext, "application/octet-stream")
    if offset or limit:
        sz = len(blob)
        if offset >= sz:
            raise HTTPException(400, f"offset {offset} >= size {sz}")
        cap = limit if limit > 0 else MAX_RAW_RESPONSE_BYTES
        cap = min(cap, MAX_RAW_RESPONSE_BYTES)
        end = min(sz, offset + cap)
        chunk = blob[offset:end]
        return Response(
            content=chunk,
            media_type=media_type,
            headers={
                "X-Asset-Size": str(len(chunk)),
                "X-Asset-Total": str(sz),
                "X-Asset-Offset": str(offset),
                "X-Asset-Limit": str(end - offset),
                "X-Asset-Ext": inner_ext,
            },
        )
    if len(blob) > MAX_RAW_RESPONSE_BYTES:
        raise HTTPException(
            413,
            f"file too large for raw endpoint: {len(blob)} > {MAX_RAW_RESPONSE_BYTES} bytes",
        )
    headers = {
        "X-Asset-Size": str(len(blob)),
        "X-Asset-Ext": inner_ext,
    }
    return Response(content=blob, media_type=media_type, headers=headers)


@app.get("/api/raw_nj/{path:path}")
def api_raw_nj(path: str):
    """Serve the RAW decompressed inner ``.nj`` / ``.njm`` byte buffer.

    Purpose-built for the client-side psov2 Ninja loader
    (``static/psov2_ninja.js``): the frontend fetches these raw bytes and
    parses the NJCM bone tree / chunks in the browser, exactly like the
    psov2 reference does. We only do the container un-wrapping (BML/AFS
    inner extraction + PRS decompress) on the server; NO geometry
    reconstruction happens here.

    Accepts:
      * a plain top-level ``foo.nj`` filename, or
      * a BML/AFS inner path ``bm_npc_momoka.bml#n_momoka_t_body.nj``
        (``#`` separates the container from the inner entry).

    Resolves under DATA_DIR first, then LIVE_DATA_DIR (read-only), via the
    shared ``resolve_asset_bytes`` helper — so it inherits the same
    path-traversal validation and PRS-decompress cache the tile pipeline
    uses. The returned bytes are the inner ``.nj`` payload AFTER PRS
    decompression (ready for ``parseNinjaModel``).

    Returns
    -------
    200 - raw NJ/NJM bytes (``application/octet-stream``), strongly cached
    400 - not a ``.nj``/``.njm`` target, malformed ``#``, or path traversal
    404 - file or inner-entry missing
    413 - file too large to parse in-memory
    502 - PRS decompress subprocess failed
    """
    _, inner = _split_inner_path(path)
    logical = inner if inner is not None else path
    low = logical.lower()
    if not (low.endswith(".nj") or low.endswith(".njm")):
        raise HTTPException(
            400,
            f"/api/raw_nj only serves .nj/.njm targets, got {logical!r}",
        )

    blob, _ = resolve_asset_bytes(path)
    if len(blob) > MAX_RAW_RESPONSE_BYTES:
        raise HTTPException(
            413,
            f"NJ too large for raw endpoint: {len(blob)} > {MAX_RAW_RESPONSE_BYTES}",
        )
    # The inner bytes are content-derived from the (immutable) container;
    # an external rebuild that lands a new BML invalidates via the
    # decompress_prs_cached key (base size+mtime), so a long browser cache
    # is safe within a session. Keep it conservative (1h) + revalidatable.
    return Response(
        content=blob,
        media_type="application/octet-stream",
        headers={
            "X-Asset-Size": str(len(blob)),
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/model/{path:path}/skeleton")
def api_model_skeleton(path: str, inner: Optional[str] = None):
    """Return the bone hierarchy for a `.nj` (or BML inner `.nj`).

    Shape::

        {
          "filename": "<input path>",
          "inner":    "<inner name or null>",
          "bone_count": N,
          "bones": [
            {"index": 0, "parent": -1, "position": [x,y,z], "rotation": [rx,ry,rz]},
            ...
          ]
        }

    Rotations are raw Ninja BAMs (0x10000 = 360°); convert with
    ``r * 2π / 0x10000`` if you need radians.

    Returns
    -------
    200 - bone list (may be empty for static props)
    400 - invalid path / extension / missing required ``inner`` for BML
    404 - file or inner-entry not found
    413 - file too large to parse in-memory
    502 - BML PRS decompress subprocess failed
    """
    p = _resolve_model_mesh_path(path)
    ext = p.suffix.lower()
    if ext == ".bml":
        if not inner:
            raise HTTPException(
                400,
                "BML model requires `?inner=<entry-name>.nj` query parameter",
            )
        _validate_inner_name(inner, msg="invalid inner entry name")
        inner_ext = Path(inner).suffix.lower()
        if inner_ext not in IFF_EXTENSIONS:
            raise HTTPException(
                400,
                f"inner entry must be {IFF_EXTENSIONS!r}, got {inner_ext!r}",
            )
        nj_bytes = _read_inner_nj_from_bml(p, inner)
    elif ext == ".afs":
        if not inner:
            raise HTTPException(
                400,
                "AFS model requires `?inner=NNNN_<basename>` query parameter",
            )
        nj_bytes, _logical = _read_afs_inner_nj(p, inner)
    elif ext in IFF_EXTENSIONS:
        if inner:
            raise HTTPException(
                400,
                f"`inner` query parameter not allowed for {ext} files",
            )
        sz = p.stat().st_size
        if sz > ASSET_PARSE_MAX_BYTES:
            raise HTTPException(
                413,
                f"`.nj` too large: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
            )
        nj_bytes = p.read_bytes()
    else:
        raise HTTPException(
            400,
            f"unsupported model extension {ext!r} (expected .nj, .bml, or .afs)",
        )

    try:
        bones = _parse_cache.parse_skeleton_cached(
            nj_bytes,
            file_key=_build_model_file_key(p, ext, inner),
        )
    except ValueError as e:
        raise HTTPException(400, f"skeleton parse failed: {e}")
    except Exception as e:  # pragma: no cover - defensive
        log.exception("skeleton parse internal error")
        raise HTTPException(500, f"skeleton parse internal error: {e}")

    return {
        "filename": path,
        "inner": inner,
        "bone_count": len(bones),
        "bones": [
            {
                "index": b.index,
                "parent": b.parent,
                "position": list(b.position),
                "rotation": list(b.rotation),
            }
            for b in bones
        ],
    }


@app.get("/api/files")
def api_files():
    """List .prs / .xvm files in data/, with sizes + mtime, sorted by name."""
    items = []
    for ext in ALLOWED_EXT:
        for p in sorted(DATA_DIR.glob(f"*{ext}")):
            if _is_backup_name(p.name):
                continue
            try:
                # Single stat() reused for both size and mtime — was
                # previously stat()ing twice per file.
                st = p.stat()
                items.append({
                    "name": p.name,
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                })
            except OSError as e:
                log.debug("stat failed for %s: %s", p, e)
    items.sort(key=lambda x: x["name"].lower())
    return {"files": items, "data_dir": str(DATA_DIR)}


# --------------------------------------------------------------------------
# /api/tiles response cache (Phase perf 2026-06-19)
#
# The Tile-grid batch endpoint re-read EVERY tile PNG off disk and
# base64-encoded it (png_to_b64) on EVERY warm call — the per-tile
# _TILE_PNG_CACHE LRU was wired only into the single-tile /api/tile_png
# route, never into /api/tiles. We memo the fully-assembled response dict
# keyed on the source file's stat so warm calls skip the per-tile disk
# read + base64 loop entirely (~30 ms -> sub-ms server-side). An ETag
# (md5 of the source file stat) lets the browser revalidate with a cheap
# 304 once the frontend stops cache-busting the request.
_TILES_RESP_CACHE_MAX_ENTRIES = int(
    os.environ.get("PSO_TILES_RESP_CACHE_ENTRIES", "64"),
)
_TILES_RESP_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_TILES_RESP_CACHE_LOCK = threading.Lock()


def _tiles_resp_cache_key(work_path: Path, filename: str):
    """LRU key for one /api/tiles response, on the source file's stat."""
    try:
        st = work_path.stat()
    except OSError:
        return None
    return (str(work_path), int(st.st_mtime_ns), int(st.st_size), filename)


def _tiles_resp_etag(key: tuple) -> str:
    """Strong ETag derived from the cache key (mtime_ns|size|path|name)."""
    return '"' + hashlib.md5(repr(key).encode("utf-8")).hexdigest()[:16] + '"'


@app.get("/api/tiles/{filename}")
def api_tiles(filename: str, request: Request):
    """Extract & return tile metadata + base64 PNGs for one PRS/XVM file.

    Accepts both regular filenames (resolved under DATA_DIR / LIVE_DATA_DIR)
    and the BML-inner ``<base>#<inner>`` form (extracts the inner XVM /
    NJ-texture blob first, then runs the same xvr_codec extract pipeline).

    Warm calls serve the assembled base64 payload from an in-memory LRU
    keyed on the source file's stat, skipping the per-tile disk read +
    base64 loop. Carries an ETag so the browser can revalidate via 304.
    """
    try:
        prs = _materialize_inner_for_extract(filename)
    except HTTPException:
        raise
    if not prs.exists():
        raise HTTPException(404, f"no such file: {filename}")

    cache_key_t = _tiles_resp_cache_key(prs, filename)
    etag = _tiles_resp_etag(cache_key_t) if cache_key_t is not None else None

    # Conditional GET short-circuit — only valid when the live file stat
    # still matches the requested ETag (so a re-deploy auto-invalidates).
    if etag is not None:
        inm = request.headers.get("if-none-match")
        if inm and inm == etag:
            return Response(status_code=304, headers={"ETag": etag})

    # Helper: serve a pre-serialized body. The base64-PNG payload is
    # near-incompressible (gzip of it costs ~34 ms for a 23% reduction —
    # a net loss on loopback), so we tag it `Content-Encoding: identity`
    # which makes Starlette's GZipMiddleware skip compression for this
    # response. The bytes are cached so warm calls also skip json.dumps.
    def _serve_tiles_body(body: bytes):
        return Response(
            content=body,
            media_type="application/json",
            headers={
                "ETag": etag or "",
                "Cache-Control": "private, max-age=300",
                "Content-Encoding": "identity",
            },
        )

    # In-memory response memo (serialized bytes).
    if cache_key_t is not None:
        with _TILES_RESP_CACHE_LOCK:
            cached = _TILES_RESP_CACHE.get(cache_key_t)
            if cached is not None:
                _TILES_RESP_CACHE.move_to_end(cache_key_t)
        if cached is not None:
            return _serve_tiles_body(cached)

    try:
        manifest = extract_tiles(prs)
    except HTTPException:
        raise
    except (OSError, RuntimeError) as e:
        log.exception("extract_tiles failed for %s", filename)
        raise HTTPException(500, f"extract failed: {e}")
    tiles_dir = Path(manifest["tiles_dir"])
    out = []
    for t in manifest["tiles"]:
        png = tiles_dir / t["filename"]
        out.append({**t, "src_png_b64": png_to_b64(png)})
    payload = {
        "filename": filename,
        "tile_count": manifest["tile_count"],
        "is_prs": manifest.get("is_prs", False),
        "tiles": out,
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    if cache_key_t is not None:
        with _TILES_RESP_CACHE_LOCK:
            _TILES_RESP_CACHE[cache_key_t] = body
            _TILES_RESP_CACHE.move_to_end(cache_key_t)
            while len(_TILES_RESP_CACHE) > _TILES_RESP_CACHE_MAX_ENTRIES:
                _TILES_RESP_CACHE.popitem(last=False)
        return _serve_tiles_body(body)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Encoding": "identity"},
    )


@app.get("/api/tile_png/{filename}/{idx}")
def api_tile_png(filename: str, idx: int):
    """Serve a raw PNG of a tile (no base64 wrapping). Useful for large tiles.

    Accepts the same path forms as ``/api/tiles`` — plain filename or
    ``<base>#<inner>``.

    Wraps a 2-tier cache (Phase D Win 5):
      L1: in-memory PNG-bytes LRU keyed on (work_path, mtime, size, idx).
      L2: on-disk PNG cache mirroring the same key (sha2 filename).
    Cold opens of a dragon-class model used to spend ~1.6 s here for the
    16 tile renders; warm opens now serve from the LRU at <5 ms / tile.
    Cache hits return ``Response(content=bytes)`` to avoid the FileResponse
    open() cost; cache misses fall through to the original FileResponse
    path so the file's stat fields (Content-Length etc.) are preserved.
    """
    try:
        prs = _materialize_inner_for_extract(filename)
    except HTTPException:
        raise
    if not prs.exists():
        raise HTTPException(404, f"no such file: {filename}")
    if idx < 0 or idx > MAX_TILE_INDEX:
        raise HTTPException(400, f"tile index out of range (0..{MAX_TILE_INDEX})")

    def _do_extract() -> Path:
        manifest = extract_tiles(prs)
        tile = next((t for t in manifest["tiles"] if t["index"] == idx), None)
        if not tile:
            raise HTTPException(404, "no such tile index")
        p = Path(manifest["tiles_dir"]) / tile["filename"]
        if not p.exists():
            raise HTTPException(500, "tile png missing on disk")
        return p

    # Cache lookup — returns either (bytes, None) on hit, or
    # (None, Path) on miss (the route serves the path via FileResponse).
    try:
        bytes_hit, path_hit = _serve_tile_png_cached(prs, idx, filename, _do_extract)
    except HTTPException:
        raise
    except Exception as e:
        # A bad/undecodable texture (e.g. a bare archive that isn't an XVM/XVR,
        # or an XVM with an unreadable header) must NOT 500 — the viewer treats
        # a clean 4xx as "render untextured" instead of surfacing a crash.
        log.warning("tile_png decode failed for %s/%s: %s: %s",
                    filename, idx, type(e).__name__, e)
        raise HTTPException(415, f"tile not decodable: {type(e).__name__}") from None
    if bytes_hit is not None:
        # In-memory or disk cache hit. Return raw bytes — no extra
        # filesystem open(). Add a long-lived ETag based on the cache
        # key so the browser can short-circuit re-fetches with 304.
        # Single stat() call — was previously stat()ing twice for the
        # same key (mtime_ns + size). On warm cache-hit paths this is
        # the only filesystem call, so halving it matters.
        _et_st = prs.stat()
        etag = hashlib.md5(
            f"{_et_st.st_mtime_ns}|{_et_st.st_size}|{idx}".encode("utf-8"),
        ).hexdigest()[:16]
        return Response(
            content=bytes_hit,
            media_type="image/png",
            headers={
                "ETag": f'"{etag}"',
                # The cache is keyed on the file's mtime so once the
                # underlying file changes, our key changes and the
                # browser will re-fetch. Aggressive caching is safe.
                "Cache-Control": "private, max-age=300",
            },
        )
    return FileResponse(path_hit, media_type="image/png")


@app.get("/api/tile_png_cache/stats")
def api_tile_png_cache_stats():
    """Return tile-PNG cache health (mirrors /api/parse_cache/stats shape)."""
    return _tile_png_cache_stats()


@app.delete("/api/tile_png_cache/clear")
@app.post("/api/tile_png_cache/clear")
def api_tile_png_cache_clear(disk: int = 1):
    """Drop the tile-PNG cache (in-memory + on-disk PNGs unless disk=0)."""
    return _tile_png_cache_clear(drop_disk=bool(disk))


class UpscaleReq(BaseModel):
    filename: str
    tile_index: int = Field(ge=0, le=MAX_TILE_INDEX)
    model: str
    scale: int = 4
    keep_native_dims: bool = True
    # V3 extensions (all optional, backwards compatible):
    tile_size: Optional[int] = None  # 0=auto, 32/64/128/256/512
    tta: bool = False                # test-time augmentation (slow, higher quality)
    gpu_id: Optional[int] = None     # -1 cpu, 0/1/... gpu-id, None=auto


@app.post("/api/upscale")
def api_upscale(req: UpscaleReq, request: Request):
    """Upscale a single tile via realesrgan-ncnn-vulkan.

    Validates model name + lookup, scale, tile_size, gpu_id; serializes
    concurrent requests for the same (file,tile,model,scale,settings) tuple
    so cache writes can't race; honors cascade for scale > native.

    Accepts both bare filenames and the `<bml>#<inner>` / `<afs>#<inner>`
    inner-path syntax so the in-viewport texture panel can drive the
    upscaler against textures embedded inside BML / AFS containers.
    """
    _enforce_body_size(request, MAX_UPSCALE_BODY)
    if "#" in req.filename:
        try:
            prs = _materialize_inner_for_extract(req.filename)
        except HTTPException:
            raise
    else:
        prs = safe_data_path(req.filename)
    if not prs.exists():
        raise HTTPException(404, "missing file")
    if req.scale not in ALLOWED_SCALES:
        raise HTTPException(
            400, f"scale must be one of {ALLOWED_SCALES} (got {req.scale})"
        )
    # Validate model name (alphanum/dash/underscore + matches a real .bin)
    if not MODEL_NAME_RE.match(req.model):
        raise HTTPException(400, "invalid model name")
    bin_path = REALESRGAN_MODELS / f"{req.model}.bin"
    param_path = REALESRGAN_MODELS / f"{req.model}.param"
    if not (bin_path.exists() and param_path.exists()):
        raise HTTPException(400, f"model not found: {req.model}")
    # Validate tile_size
    if req.tile_size is not None and req.tile_size not in ALLOWED_TILE_SIZES:
        raise HTTPException(
            400, f"tile_size must be one of {ALLOWED_TILE_SIZES} (got {req.tile_size})"
        )
    # Validate gpu_id
    if req.gpu_id is not None and not (ALLOWED_GPU_ID_RANGE[0] <= req.gpu_id <= ALLOWED_GPU_ID_RANGE[1]):
        raise HTTPException(400, f"gpu_id must be in {ALLOWED_GPU_ID_RANGE} or null")

    manifest = extract_tiles(prs)
    tile = next((t for t in manifest["tiles"] if t["index"] == req.tile_index), None)
    if not tile:
        raise HTTPException(404, "no such tile")
    tiles_dir = Path(manifest["tiles_dir"])
    src = tiles_dir / tile["filename"]
    cache_subdir = Path(manifest["cache_dir"]) / "upscaled"
    cache_subdir.mkdir(exist_ok=True)

    # Settings tuple is part of the cache key so different settings don't collide.
    settings_tag = (
        f"_t{req.tile_size if req.tile_size is not None else 'A'}"
        f"_tta{1 if req.tta else 0}"
        f"_g{req.gpu_id if req.gpu_id is not None else 'A'}"
    )
    base_name = f"tile{req.tile_index:02d}_{req.model}_x{req.scale}{settings_tag}"
    final_path = cache_subdir / f"{base_name}_native.png"

    lock_key = f"{req.filename}|{req.tile_index}|{req.model}|{req.scale}|{settings_tag}"
    lk = _get_lock(_UPSCALE_LOCKS, lock_key, MAX_UPSCALE_LOCKS)
    with lk:
        # Run cascade — produces an intermediate at the cumulative requested scale.
        casc_out = _cascade_upscale(
            src,
            cache_subdir,
            base_name,
            req.model,
            req.scale,
            tile_size=req.tile_size,
            tta=req.tta,
            gpu_id=req.gpu_id,
        )

        # casc_out is at the "achieved" scale (may slightly differ from requested
        # if the decomposition couldn't hit it exactly — e.g. requesting a prime).
        # We re-resize to exactly src*scale unless the user wants native dims.
        with Image.open(casc_out) as im:
            cw, ch = im.size

        # Bug B (2026-05-01): clamp the requested-scale target dims to
        # the engine's max texture dim. Without this, the "want exactly
        # src*requested_scale pixels" branch below would Lanczos-up the
        # already-clamped cascade output back to e.g. 2048x2048, and
        # the game would silently fail to load it (black tree / missing
        # particle bug). _cascade_upscale already clamps its own output
        # so the target dims here just have to agree.
        req_target_w = tile["width"] * req.scale
        req_target_h = tile["height"] * req.scale
        if req_target_w > PSOBB_MAX_TEXTURE_DIM or req_target_h > PSOBB_MAX_TEXTURE_DIM:
            cap = float(PSOBB_MAX_TEXTURE_DIM)
            ratio = min(cap / req_target_w, cap / req_target_h)
            target_w = min(int(round(req_target_w * ratio)), PSOBB_MAX_TEXTURE_DIM)
            target_h = min(int(round(req_target_h * ratio)), PSOBB_MAX_TEXTURE_DIM)
        else:
            target_w = req_target_w
            target_h = req_target_h
        if req.keep_native_dims:
            # Lanczos-down to native source dims (game requires this).
            if not final_path.exists():
                with Image.open(casc_out) as im:
                    im = _resize_rgba_preserving_binary_alpha(
                        im, (tile["width"], tile["height"])
                    )
                    im.save(final_path)
            out_path = final_path
        else:
            # Want exactly src*requested_scale pixels (capped at PSOBB_MAX_TEXTURE_DIM).
            if (cw, ch) == (target_w, target_h):
                out_path = casc_out
            else:
                resized_path = cache_subdir / f"{base_name}_exact.png"
                if not resized_path.exists():
                    with Image.open(casc_out) as im:
                        im = _resize_rgba_preserving_binary_alpha(
                            im, (target_w, target_h)
                        )
                        im.save(resized_path)
                out_path = resized_path

    with Image.open(out_path) as im:
        ow, oh = im.size
    return {
        "tile_index": req.tile_index,
        "model": req.model,
        "scale": req.scale,
        "tile_size": req.tile_size,
        "tta": req.tta,
        "gpu_id": req.gpu_id,
        "out_b64": png_to_b64(out_path),
        "out_w": ow,
        "out_h": oh,
        "src_w": tile["width"],
        "src_h": tile["height"],
        "cascade_w": cw,
        "cascade_h": ch,
    }


@app.post("/api/import_png/{filename}/{tile_index}")
async def api_import_png(
    filename: str,
    tile_index: int,
    image: UploadFile = File(...),
    keep_native_dims: str = Form("true"),
):
    """Import a user-supplied PNG (e.g. from external Upscayl run) as the
    "upscaled" version of a given tile.

    Validation rules:
      - PNG only.
      - Image must already be at native dim, OR be an exact integer multiple
        of native (in which case we Lanczos-down on the way in).
      - Non-multiple sizes are rejected with a clear error explaining the
        accepted choices.

    Stored alongside the in-editor `_cascade_upscale` outputs in the cache
    `upscaled/` subdir, returned base64 so the frontend can use it
    immediately (registration into `state.tileEdits` is the caller's job
    via the existing repack flow).

    Accepts both bare filenames and the `<bml>#<inner>` / `<afs>#<inner>`
    inner-path syntax (mirrors the upscale endpoint).
    """
    if "#" in filename:
        try:
            prs = _materialize_inner_for_extract(filename)
        except HTTPException:
            raise
    else:
        prs = safe_data_path(filename)
    if not prs.exists():
        raise HTTPException(404, f"no such file: {filename}")
    if tile_index < 0 or tile_index > MAX_TILE_INDEX:
        raise HTTPException(400, f"tile index out of range (0..{MAX_TILE_INDEX})")

    keep_native = str(keep_native_dims).strip().lower() in ("1", "true", "yes", "on")

    # Read upload - cap to MAX_IMPORT_PNG_BYTES so a runaway upload can't OOM us
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_IMPORT_PNG_BYTES:
        raise HTTPException(413, f"upload too large: {len(raw)} bytes (cap {MAX_IMPORT_PNG_BYTES})")
    if raw[:8] != PNG_MAGIC:
        raise HTTPException(400, "not a PNG (magic bytes missing)")

    # Audit C-6: hand off the CPU-heavy section (extract_tiles + Image.open +
    # Lanczos resize + disk writes) to a worker thread so this async handler
    # doesn't head-of-line block the event loop.
    return await asyncio.to_thread(
        _import_png_sync_work,
        prs,
        raw,
        filename,
        tile_index,
        keep_native,
        image.filename or "",
    )


def _import_png_sync_work(
    prs: Path,
    raw: bytes,
    filename: str,
    tile_index: int,
    keep_native: bool,
    upload_filename: str,
) -> dict:
    """CPU-heavy synchronous tail of api_import_png. Runs in a worker thread.

    Does tile-manifest extraction, PNG decode, dimension validation,
    optional Lanczos downsize, and the final write to the cache subdir.
    Returns the JSON-shaped dict the endpoint hands back.
    """
    manifest = extract_tiles(prs)
    tile = next((t for t in manifest["tiles"] if t["index"] == tile_index), None)
    if not tile:
        raise HTTPException(404, "no such tile")
    nat_w = int(tile["width"])
    nat_h = int(tile["height"])

    # Decode PNG
    try:
        im = Image.open(BytesIO(raw))
        im.load()
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"PNG decode failed: {e}")

    src_w, src_h = im.size

    # Dim validation: must equal native, or be an integer multiple of native
    # on BOTH axes (uniform scale).
    if (src_w, src_h) == (nat_w, nat_h):
        scale_factor = 1
    else:
        if src_w % nat_w != 0 or src_h % nat_h != 0:
            raise HTTPException(
                400,
                f"PNG dim {src_w}x{src_h} is not an integer multiple of tile "
                f"native dim {nat_w}x{nat_h}. Resize externally to one of: "
                f"{nat_w}x{nat_h}, {nat_w*2}x{nat_h*2}, {nat_w*3}x{nat_h*3}, "
                f"{nat_w*4}x{nat_h*4}.",
            )
        kx = src_w // nat_w
        ky = src_h // nat_h
        if kx != ky:
            raise HTTPException(
                400,
                f"PNG dim {src_w}x{src_h} has non-uniform scale ({kx}x vs {ky}x). "
                f"Editor requires uniform integer scale of native {nat_w}x{nat_h}.",
            )
        scale_factor = kx

    cache_subdir = Path(manifest["cache_dir"]) / "upscaled"
    cache_subdir.mkdir(exist_ok=True)
    ts = int(time.time())
    safe_orig = IMPORT_FILENAME_SAFE_RE.sub("_", upload_filename or "upload.png")
    base_name = f"tile{tile_index:02d}_imported_x{scale_factor}_{ts}"
    full_path = cache_subdir / f"{base_name}_{safe_orig}"

    im_rgba = im.convert("RGBA")
    if keep_native and (src_w, src_h) != (nat_w, nat_h):
        # Lanczos-down to native dim (game requires this)
        im_rgba_save = im_rgba.resize((nat_w, nat_h), Image.Resampling.LANCZOS)
        out_path = cache_subdir / f"{base_name}_native.png"
        im_rgba_save.save(out_path)
    else:
        out_path = full_path
        im_rgba.save(out_path)

    with Image.open(out_path) as ck:
        ow, oh = ck.size

    return {
        "tile_index": tile_index,
        "filename": filename,
        "imported_filename": upload_filename or "",
        "imported_w": src_w,
        "imported_h": src_h,
        "scale_factor": scale_factor,
        "keep_native_dims": keep_native,
        "out_b64": png_to_b64(out_path),
        "out_w": ow,
        "out_h": oh,
        "src_w": nat_w,
        "src_h": nat_h,
    }


class TileEdit(BaseModel):
    tile_index: int = Field(ge=0, le=MAX_TILE_INDEX)
    png_b64: str


class RepackReq(BaseModel):
    filename: str
    tiles: list[TileEdit]
    deploy: bool = True


class RepackDiffReq(BaseModel):
    filename: str
    edited_indices: list[int]


@app.post("/api/repack_diff")
def api_repack_diff(req: RepackDiffReq, request: Request):
    """Pre-deploy summary: what will be touched by a repack, without doing it.

    Returns:
      - tile_count                total tiles in the file
      - changed_indices           tiles the caller flagged as edited
      - unchanged_indices         everything else
      - backup_name_preview       the .pre_editor_<ts> name we'd use right now
      - file_size_bytes           current on-disk size
    """
    _enforce_body_size(request, MAX_REPACK_DIFF_BODY)
    prs = safe_data_path(req.filename)
    if not prs.exists():
        raise HTTPException(404, "missing file")
    manifest = extract_tiles(prs)
    all_idx = [t["index"] for t in manifest["tiles"]]
    edited_set = set(req.edited_indices)
    valid_changed = sorted(i for i in all_idx if i in edited_set)
    valid_unchanged = sorted(i for i in all_idx if i not in edited_set)
    unknown = sorted(i for i in edited_set if i not in set(all_idx))
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_name = f"{req.filename}.pre_editor_{ts}"
    return {
        "filename": req.filename,
        "tile_count": manifest["tile_count"],
        "is_prs": manifest.get("is_prs", False),
        "changed_indices": valid_changed,
        "unchanged_indices": valid_unchanged,
        "unknown_indices": unknown,
        "backup_name_preview": backup_name,
        "file_size_bytes": prs.stat().st_size,
    }


def _apply_edits_and_rebuild(
    prs_path: Path,
    edits: list["TileEdit"],
    *,
    label: Optional[str] = None,
) -> dict:
    """Apply tile-PNG edits + run xvr_codec rebuild on a single XVMH/PRS source.

    Shared between ``/api/repack`` (top-level files in DATA_DIR) and
    ``/api/repack_afs_inner`` (an inner blob materialised from an AFS).
    The caller has already taken its own lock and validated the source.

    Args:
        prs_path: on-disk path to the source XVM (or PRS-wrapped XVM).
            Must exist and be one of the formats ``extract_tiles``
            understands.
        edits: validated ``TileEdit`` list. May be empty (no-op rebuild —
            useful for PRS round-trip tests).
        label: optional logical filename used for tile-not-found error
            messages and the rebuilt artifact's name. Defaults to
            ``prs_path.name``.

    Returns a dict with the keys callers in this module rely on:
        - manifest: full extract manifest (incl. is_prs, tiles, cache_dir,
          tiles_dir)
        - rebuilt_path: Path to the freshly rebuilt XVM (NOT PRS-wrapped)
        - rebuilt_size: size in bytes
        - verify: ``_verify_rebuilt`` output (ok=True or raised on fail)
        - spliced / reencoded: parsed from xvr_codec stdout (or None)
        - changed_indices: sorted unique tile_index list from ``edits``
        - cache_dir: Path of the cache subdir (== Path(manifest['cache_dir']))

    Raises HTTPException with the same codes as the inline body it
    replaced (400 on invalid PNG / unknown tile, 500 on verify fail).
    The xvr_codec subprocess timeout/RuntimeError is left to bubble.
    """
    name = label or prs_path.name
    manifest = extract_tiles(prs_path)
    tiles_dir = Path(manifest["tiles_dir"])

    changed_indices: list[int] = sorted({e.tile_index for e in edits})

    for edit in edits:
        target = next((t for t in manifest["tiles"] if t["index"] == edit.tile_index), None)
        if not target:
            raise HTTPException(400, f"tile {edit.tile_index} not in {name}")
        target_png = tiles_dir / target["filename"]
        b = edit.png_b64
        if "," in b:
            b = b.split(",", 1)[1]
        try:
            data = base64.b64decode(b)
        except (ValueError, binascii.Error) as e:
            raise HTTPException(400, f"tile {edit.tile_index} png_b64 decode failed: {e}")
        try:
            im = Image.open(BytesIO(data)).convert("RGBA")
        except (OSError, ValueError) as e:
            raise HTTPException(400, f"tile {edit.tile_index} png_b64 not a valid PNG: {e}")
        if im.size != (target["width"], target["height"]):
            im = im.resize((target["width"], target["height"]), Image.Resampling.LANCZOS)
        im.save(target_png)

    # Force-reencode marker (defence-in-depth — see api_repack docs).
    force_marker = tiles_dir / ".force_reencode"
    if changed_indices:
        force_marker.write_text(" ".join(str(i) for i in changed_indices))
    else:
        try:
            force_marker.unlink(missing_ok=True)
        except OSError:
            pass

    cdir = Path(manifest["cache_dir"])
    rebuilt = cdir / (name + ".rebuilt.xvm")
    # In-process rebuild (formats/xvr_decode.rebuild_xvm). Replaces the old
    # subprocess to `xvr_codec.py rebuild`, whose CLI was lost when
    # C:/Tools/re/* was deleted (the shim is import-only, so the subprocess
    # silently produced no output file). Splices untouched tiles verbatim and
    # re-encodes edited tiles in their original pixelFormat.
    _rb = _xvr_decode.rebuild_xvm(tiles_dir, rebuilt)
    spliced = _rb["spliced"]
    reencoded = _rb["reencoded"]

    verify = _verify_rebuilt(rebuilt, manifest)
    if not verify["ok"]:
        raise HTTPException(500, f"rebuild verification failed: {verify['reason']}")

    return {
        "manifest": manifest,
        "rebuilt_path": rebuilt,
        "rebuilt_size": rebuilt.stat().st_size,
        "verify": verify,
        "spliced": spliced,
        "reencoded": reencoded,
        "changed_indices": changed_indices,
        "cache_dir": cdir,
    }


def _verify_rebuilt(rebuilt: Path, manifest: dict) -> dict:
    """Sanity-check the freshly rebuilt XVM body: tile count, dims, fmt.

    Walks the XVMH/XVRT structure of `rebuilt`, compares to `manifest['tiles']`.
    Returns {ok: True, tiles: [...]} on success or {ok: False, reason: ...}.
    """
    blob = rebuilt.read_bytes()
    if blob[:4] != XVM_MAGIC:
        return {"ok": False, "reason": "rebuilt missing XVMH magic"}
    count = int.from_bytes(blob[8:12], "little")
    if count != manifest["tile_count"]:
        return {"ok": False, "reason": f"tile count mismatch {count} != {manifest['tile_count']}"}
    expected = list(manifest["tiles"])
    pos = XVM_HEADER_SIZE
    seen = []
    while pos + XVM_HEADER_SIZE <= len(blob):
        if blob[pos:pos + 4] != XVRT_MAGIC:
            break
        fmt = int.from_bytes(blob[pos + 0x0C:pos + 0x10], "little")
        w = int.from_bytes(blob[pos + 0x14:pos + 0x16], "little")
        h = int.from_bytes(blob[pos + 0x16:pos + 0x18], "little")
        dsz = int.from_bytes(blob[pos + 0x18:pos + 0x1C], "little")
        seen.append({"w": w, "h": h, "fmt": fmt, "dsz": dsz})
        pos += XVM_HEADER_SIZE + dsz
    if len(seen) != len(expected):
        return {"ok": False, "reason": f"walked {len(seen)} XVRT blocks, expected {len(expected)}"}
    for i, (s, e) in enumerate(zip(seen, expected)):
        if s["w"] != e["width"] or s["h"] != e["height"] or s["fmt"] != e["fmt"]:
            return {"ok": False, "reason": f"tile {i} mismatch {s} != {e}"}
    return {"ok": True, "tiles": seen}


def _export_token_index_path(token: str) -> Path:
    """Sidecar JSON for a token. Lives next to the staged artifact so
    cleanup is one rmdir away.
    """
    return EXPORT_DIR / f"{token}.json"


def _persist_export_token(token: str, entry: dict) -> None:
    """Write the token's sidecar JSON. Used by workers=4 to make the
    token visible to peers within the same uvicorn launch.
    """
    sidecar = _export_token_index_path(token)
    try:
        sidecar.write_text(json.dumps(entry), encoding="utf-8")
    except OSError as e:  # pragma: no cover — best-effort
        log.warning("could not persist export sidecar %s: %s", sidecar, e)


def _load_export_token(token: str) -> Optional[dict]:
    """Load a sidecar JSON for `token`. Returns None if missing/corrupt
    or if the staged artifact has been GC'd.
    """
    sidecar = _export_token_index_path(token)
    if not sidecar.exists():
        return None
    try:
        entry = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entry, dict) or "path" not in entry or "expires_at" not in entry:
        return None
    return entry


def _make_export_token(artifact_path: Path, filename: str) -> str:
    """Stage a rebuilt artifact in EXPORT_DIR under a random token; return the
    token. Caller fetches via GET /api/export/<token>.

    Tokens expire after EXPORT_TTL_SECONDS or on server restart (in-memory).
    Wave 7: also persisted via `_persist_export_token` so workers=4 sees them.
    """
    token = secrets.token_urlsafe(24)
    suffix = ".prs" if filename.lower().endswith(".prs") else ".xvm"
    staged = EXPORT_DIR / f"{token}{suffix}"
    shutil.copy(artifact_path, staged)
    entry = {
        "path": str(staged),
        "filename": filename,
        "expires_at": time.time() + EXPORT_TTL_SECONDS,
    }
    with _EXPORT_TOKENS_LOCK:
        _EXPORT_TOKENS[token] = entry
    _persist_export_token(token, entry)
    return token


def _gc_export_tokens() -> None:
    """Drop expired tokens and their artifacts. Idempotent.

    Wave 7: also scans EXPORT_DIR for stale `<token>.json` sidecars so a
    crash that left them orphaned cleans up over time. The combined pass
    keeps memory + disk consistent across workers.
    """
    now = time.time()
    # In-memory pass: tokens we know about. Snapshot under lock so a
    # concurrent mint/pop can't trip "dictionary changed size during
    # iteration" (audit C-2). Pops also under lock.
    with _EXPORT_TOKENS_LOCK:
        dead = [t for t, v in _EXPORT_TOKENS.items() if v["expires_at"] < now]
    for t in dead:
        with _EXPORT_TOKENS_LOCK:
            v = _EXPORT_TOKENS.pop(t, None)
        if v:
            try:
                Path(v["path"]).unlink(missing_ok=True)
            except OSError as e:
                log.debug("could not unlink expired export %s: %s", v.get("path"), e)
            try:
                _export_token_index_path(t).unlink(missing_ok=True)
            except OSError as e:
                log.debug("could not unlink expired sidecar %s: %s", t, e)
    # Disk pass: catch sidecars from peer workers we never saw in-memory.
    try:
        for sidecar in EXPORT_DIR.glob("*.json"):
            try:
                entry = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(entry, dict):
                continue
            exp = entry.get("expires_at")
            if not isinstance(exp, (int, float)) or exp >= now:
                continue
            tk = sidecar.stem
            try:
                Path(entry.get("path", "")).unlink(missing_ok=True)
            except OSError:
                pass
            try:
                sidecar.unlink(missing_ok=True)
            except OSError:
                pass
            with _EXPORT_TOKENS_LOCK:
                _EXPORT_TOKENS.pop(tk, None)
    except OSError:
        pass


@app.post("/api/repack")
def api_repack(req: RepackReq, request: Request):
    """Apply tile edits (PNG b64 -> on-disk PNG) and rebuild the XVM.

    Optionally re-PRS-compresses and copies into DATA_DIR (deploy=True),
    creating a `.pre_editor_<TS>` backup first. Concurrent requests for the
    same filename are rejected with HTTP 409 (one writer at a time). When
    deploy=False, mints an export token instead.
    """
    _enforce_body_size(request, MAX_REPACK_BODY)
    if len(req.tiles) > MAX_TILES_PER_REPACK:
        raise HTTPException(400, f"too many tile edits ({len(req.tiles)} > {MAX_TILES_PER_REPACK})")
    prs = safe_data_path(req.filename)
    if not prs.exists():
        raise HTTPException(404, "missing file")

    lk = _get_lock(_REPACK_LOCKS, req.filename, MAX_REPACK_LOCKS)
    if not lk.acquire(blocking=False):
        raise HTTPException(409, f"repack already in progress for {req.filename}")
    try:
        # Apply edits + run xvr_codec rebuild via the shared helper.
        # Behaviour is identical to the previous inline body — the helper
        # is a pure refactor used by both /api/repack and the AFS-inner
        # endpoint to avoid forking the rebuild logic.
        result = _apply_edits_and_rebuild(prs, req.tiles, label=req.filename)
        manifest = result["manifest"]
        rebuilt = result["rebuilt_path"]
        cdir = result["cache_dir"]
        verify = result["verify"]
        spliced = result["spliced"]
        reencoded = result["reencoded"]
        changed_indices = result["changed_indices"]
        rebuilt_size = result["rebuilt_size"]
        rebuilt_path_str = str(rebuilt)

        deploy_path = None
        backup_path = None
        export_token = None
        export_url = None
        export_filename = None

        if req.deploy:
            ts = time.strftime("%Y%m%d_%H%M%S")
            bak = DATA_DIR / f"{req.filename}.pre_editor_{ts}"
            # Avoid clobber on truly rapid re-deploys
            counter = 0
            while bak.exists():
                counter += 1
                bak = DATA_DIR / f"{req.filename}.pre_editor_{ts}_{counter}"
            shutil.copy(prs, bak)
            backup_path = str(bak)

            if manifest["is_prs"]:
                cpath = cdir / "compress_in"
                cpath.mkdir(exist_ok=True)
                # Clean previous round-trip artifacts in this subdir
                for old in cpath.glob("*"):
                    try:
                        old.unlink()
                    except OSError as e:
                        log.debug("could not unlink %s: %s", old, e)
                stem = req.filename[:-4] if req.filename.endswith(".prs") else req.filename
                shutil.copy(rebuilt, cpath / stem)
                # PuyoToolsCli requires bare filename + cwd
                sh(
                    [PUYO, "compression", "compress", "prs", "--overwrite", "-i", stem],
                    cwd=cpath,
                    timeout=TIMEOUT_PUYO,
                )
                shutil.copy(cpath / stem, prs)
            else:
                shutil.copy(rebuilt, prs)
            deploy_path = str(prs)
            log.info("deployed %s -> %s (%d bytes)", req.filename, prs, rebuilt_size)

            # Invalidate stale cache for this file now that mtime/size changed
            old_cdir = Path(manifest["cache_dir"])
            new_key = cache_key(prs)
            if new_key != old_cdir.name:
                shutil.rmtree(old_cdir, ignore_errors=True)
        else:
            # Export-only: stage the rebuilt artifact, optionally
            # PRS-compress it first, then mint a download token.
            if manifest["is_prs"]:
                cpath = cdir / "compress_in"
                cpath.mkdir(exist_ok=True)
                for old in cpath.glob("*"):
                    try:
                        old.unlink()
                    except OSError as e:
                        log.debug("could not unlink %s: %s", old, e)
                stem = req.filename[:-4] if req.filename.endswith(".prs") else req.filename
                shutil.copy(rebuilt, cpath / stem)
                sh(
                    [PUYO, "compression", "compress", "prs", "--overwrite", "-i", stem],
                    cwd=cpath,
                    timeout=TIMEOUT_PUYO,
                )
                # Now cpath/stem is the compressed PRS
                _gc_export_tokens()
                export_token = _make_export_token(cpath / stem, req.filename)
                export_filename = req.filename  # original .prs name preserved
            else:
                _gc_export_tokens()
                export_token = _make_export_token(rebuilt, req.filename)
                export_filename = req.filename
            export_url = f"/api/export/{export_token}"

        return {
            "filename": req.filename,
            "rebuilt_size": rebuilt_size,
            "rebuilt_path": rebuilt_path_str,
            "verify": verify,
            "deploy_path": deploy_path,
            "backup_path": backup_path,
            "spliced_count": spliced,
            "reencoded_count": reencoded,
            "changed_indices": changed_indices,
            "export_token": export_token,
            "export_url": export_url,
            "export_filename": export_filename,
        }
    finally:
        lk.release()


# ---------------------------------------------------------------------------
# /api/repack_afs_inner (2026-04-30)
#
# Background:
#   /api/repack uses safe_data_path() which only accepts BARE filenames
#   ('foo.xvm'); '<archive>#NNNN' inner-syntax is rejected. AFS-resident
#   inner XVMs (ItemTexture.afs, plOtex.afs, etc.) extract + upscale fine
#   today but couldn't be redeployed.
#
# This endpoint plugs the gap. It materialises the inner via the same
# afs_reader cache the rest of the editor uses, runs the standard tile
# rebuild, and either:
#   - mints an export token (deploy=False), OR
#   - re-PRS-compresses (if the original blob was PRS), splices the new
#     bytes back into the parent AFS, and atomic-replaces both
#     LIVE_DATA_DIR and DATA_DIR copies (deploy=True) with a backup.
#
# Lock: keyed on the parent AFS filename, NOT on the inner index — two
# concurrent inner repacks of the same archive would otherwise race the
# AFS write step and clobber each other's blob splice.
# ---------------------------------------------------------------------------
class RepackAfsInnerReq(BaseModel):
    archive: str
    inner_index: int = Field(ge=0, le=0xFFFF)
    tiles: list[TileEdit] = Field(default_factory=list)
    deploy: bool = False


@app.post("/api/repack_afs_inner")
def api_repack_afs_inner(req: RepackAfsInnerReq, request: Request):
    """Repack one inner XVM of an AFS archive; optional in-place deploy.

    Body:
        archive       bare filename of the AFS (e.g. 'ItemKT.afs')
        inner_index   0-based slot index inside the AFS
        tiles         tile-edit list (same shape as /api/repack); empty =
                      no-op rebuild that round-trips the inner
        deploy        False → mint an export token + URL for the rebuilt
                      inner XVM (caller can download/inspect)
                      True  → splice rebuilt bytes back into the parent
                      AFS, write atomically to LIVE_DATA_DIR and
                      DATA_DIR with a `.pre_promote_<ts>` backup of the
                      live copy.

    Returns the same metric fields /api/repack does (rebuilt_size,
    spliced/reencoded counts, changed_indices, verify), plus AFS-level
    fields when deploy=True (afs_original_size, afs_new_size,
    afs_backup_path, archive_was_prs).
    """
    _enforce_body_size(request, MAX_REPACK_BODY)
    if len(req.tiles) > MAX_TILES_PER_REPACK:
        raise HTTPException(400, f"too many tile edits ({len(req.tiles)} > {MAX_TILES_PER_REPACK})")

    # Validate the archive name + extension up front. Both DEV and LIVE
    # roots are searched (read-only fallback) so this works for users
    # who haven't mirrored an AFS into DATA_DIR yet.
    archive_name = _validate_bare_filename(req.archive, label="archive")
    if not archive_name.lower().endswith(".afs"):
        raise HTTPException(400, f"not an AFS archive: {archive_name}")

    # Materialise the inner via the standard cache path. Re-uses the same
    # decompressed-XVM bytes /api/upscale and /api/tiles already see, so
    # an upscale pass that warmed the cache shares its work.
    try:
        inner_path = _materialize_inner_for_extract(f"{archive_name}#{req.inner_index:04d}")
    except HTTPException:
        raise
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"AFS inner materialise failed: {e}")
    if not inner_path.exists():
        raise HTTPException(404, "materialised inner missing on disk")

    # One writer per archive (NOT per inner) so concurrent inner-repacks
    # of the same AFS don't race the splice + atomic rename of the parent
    # AFS bytes.
    lock_key = archive_name
    lk = _get_lock(_REPACK_LOCKS, lock_key, MAX_REPACK_LOCKS)
    if not lk.acquire(blocking=False):
        raise HTTPException(409, f"repack already in progress for {archive_name}")
    try:
        inner_label = f"{archive_name}#{req.inner_index:04d}"
        result = _apply_edits_and_rebuild(inner_path, req.tiles, label=inner_label)
        rebuilt = result["rebuilt_path"]
        rebuilt_size = result["rebuilt_size"]
        verify = result["verify"]
        spliced = result["spliced"]
        reencoded = result["reencoded"]
        changed_indices = result["changed_indices"]

        deploy_path = None
        afs_backup_path = None
        afs_original_size = None
        afs_new_size = None
        archive_was_prs = None
        export_token = None
        export_url = None
        export_filename = None

        if not req.deploy:
            # Export-only: stage the rebuilt inner XVM (raw, no PRS layer
            # — AFS inners that ARE PRS-compressed are decompressed at
            # materialise time, so the rebuilt bytes are XVMH no matter
            # what). Caller can download via /api/export/<token>.
            _gc_export_tokens()
            export_filename = f"{Path(archive_name).stem}_{req.inner_index:04d}.xvm"
            export_token = _make_export_token(rebuilt, export_filename)
            export_url = f"/api/export/{export_token}"
        else:
            # Splice into the parent AFS and write back to disk.
            # 1. Read the live (or DEV-mirror) AFS bytes.
            live_afs = LIVE_DATA_DIR / archive_name
            dev_afs = DATA_DIR / archive_name
            if live_afs.exists():
                src_afs = live_afs
            elif dev_afs.exists():
                src_afs = dev_afs
            else:
                raise HTTPException(404, f"AFS not found in LIVE or DEV: {archive_name}")
            try:
                buf = src_afs.read_bytes()
            except OSError as e:
                raise HTTPException(500, f"AFS read failed: {e}")
            afs_original_size = len(buf)

            # 2. Parse blobs + recover names so the rewritten archive
            #    keeps any AFS_PSO-style filename table the source had.
            try:
                blobs = list(parse_afs(buf))
            except ValueError as e:
                raise HTTPException(400, f"AFS parse failed: {e}")
            if req.inner_index >= len(blobs):
                raise HTTPException(
                    400,
                    f"inner_index {req.inner_index} out of range (count={len(blobs)})",
                )
            try:
                from formats import afs_reader as _afs_reader
            except ImportError as e:
                raise HTTPException(500, f"AFS reader unavailable: {e}")
            inner_rows = _afs_reader.list_inner_blobs(src_afs)
            # Recover names ONLY when the source had a real filename
            # table. afs_reader synthesises '<stem>_<NNNN>' fallbacks
            # even when the table is absent; passing those to write_afs
            # would invent a name table the original didn't have, which
            # changes byte layout. Detect "real names present" by
            # checking the AFS_PSO descriptor slot directly.
            names = _afs_reader._afs_filename_table(buf)

            # 3. PRS recompression branch. If the ORIGINAL inner blob
            #    was PRS-compressed (sniffed from the source AFS, NOT
            #    the cached materialised path which is always
            #    decompressed), re-PRS-compress the rebuilt XVM before
            #    splicing so the on-disk format matches.
            original_inner = blobs[req.inner_index]
            archive_was_prs = _afs_reader._is_prs_compressed(original_inner)

            new_inner_bytes = rebuilt.read_bytes()
            if archive_was_prs:
                # Reuse the same PuyoToolsCli-via-cwd idiom /api/repack
                # uses for top-level PRS files. Stage in the rebuild's
                # cache subdir to avoid spamming a global tmp.
                cdir = result["cache_dir"]
                cpath = cdir / "compress_in"
                cpath.mkdir(exist_ok=True)
                for old in cpath.glob("*"):
                    try:
                        old.unlink()
                    except OSError as e:
                        log.debug("could not unlink %s: %s", old, e)
                stem_in = f"inner_{req.inner_index:04d}"
                staged = cpath / stem_in
                staged.write_bytes(new_inner_bytes)
                sh(
                    [PUYO, "compression", "compress", "prs", "--overwrite", "-i", stem_in],
                    cwd=cpath,
                    timeout=TIMEOUT_PUYO,
                )
                new_inner_bytes = staged.read_bytes()

            # 4. Splice + serialize. write_afs accepts any bytes-like
            #    list; alignment is recomputed automatically so the
            #    rewritten archive is well-formed even if the new blob
            #    happens to be larger than the original slot.
            blobs[req.inner_index] = new_inner_bytes
            try:
                new_afs_bytes = write_afs(blobs, names=names)
            except ValueError as e:
                raise HTTPException(500, f"AFS rewrite failed: {e}")
            afs_new_size = len(new_afs_bytes)

            # 5. Atomic deploy. Backup the LIVE copy first, then write
            #    via .tmp + os.replace so a crash can't leave a torn
            #    archive on disk. DEV mirror is updated last so a
            #    LIVE-only failure still leaves DEV consistent with
            #    the previous good state.
            ts = time.strftime("%Y%m%d_%H%M%S")
            if live_afs.exists():
                bak = live_afs.parent / f"{archive_name}.pre_promote_{ts}"
                counter = 0
                while bak.exists():
                    counter += 1
                    bak = live_afs.parent / f"{archive_name}.pre_promote_{ts}_{counter}"
                shutil.copy(live_afs, bak)
                afs_backup_path = str(bak)
                tmp_live = live_afs.with_suffix(live_afs.suffix + ".tmp")
                tmp_live.write_bytes(new_afs_bytes)
                os.replace(tmp_live, live_afs)
                deploy_path = str(live_afs)

            # Mirror to DEV so subsequent /api/tiles / extract calls see
            # the new bytes.
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            tmp_dev = dev_afs.with_suffix(dev_afs.suffix + ".tmp")
            tmp_dev.write_bytes(new_afs_bytes)
            os.replace(tmp_dev, dev_afs)
            if deploy_path is None:
                deploy_path = str(dev_afs)

            # Invalidate the inner-blob cache for this archive so the
            # next read of <archive>#NNNN sees the new bytes.
            try:
                inner_cache = _afs_reader.cache_dir_for(dev_afs, CACHE_DIR)
                if inner_cache.exists():
                    shutil.rmtree(inner_cache, ignore_errors=True)
            except OSError as e:
                log.debug("could not invalidate inner cache: %s", e)

            log.info(
                "deployed AFS inner %s#%d (rebuilt=%d new_afs=%d backup=%s)",
                archive_name, req.inner_index, rebuilt_size, afs_new_size, afs_backup_path,
            )

        return {
            "archive": archive_name,
            "inner_index": req.inner_index,
            "rebuilt_size": rebuilt_size,
            "rebuilt_path": str(rebuilt),
            "verify": verify,
            "spliced_count": spliced,
            "reencoded_count": reencoded,
            "changed_indices": changed_indices,
            "deploy_path": deploy_path,
            "afs_backup_path": afs_backup_path,
            "afs_original_size": afs_original_size,
            "afs_new_size": afs_new_size,
            "archive_was_prs": archive_was_prs,
            "export_token": export_token,
            "export_url": export_url,
            "export_filename": export_filename,
        }
    finally:
        lk.release()


# ===========================================================================
# Audio suite (2026-06-20)
#
# Endpoints for the audio perspective: browse container/codec, decode a .pac
# record (pure-Python) or .sfd track (ffmpeg) to WAV, render a downsampled
# waveform, and a DEV-ONLY Replace verb.
#
# CARDINAL SAFETY INVARIANT: the Replace verb writes DEV ONLY. Every output
# path is hard-asserted via `_floor_assert_not_live` to resolve inside
# DEV_DATA_DIR and NEVER under LIVE_DATA_DIR (RuntimeError otherwise) BEFORE
# any byte hits disk. This is the OPPOSITE of /api/repack_afs_inner, which
# deliberately writes LIVE — that pattern is NOT copied here. Atomic write is
# .tmp + os.replace; a `.pre_promote_<ts>` backup is taken on overwrite; a
# per-file lock returns HTTP 409 on a concurrent write.
#
# ffmpeg is OPTIONAL: .ogg/.sfd decode and .ogg re-encode degrade to HTTP 501
# when ffmpeg is absent, never 500. The core .pac codec is pure Python.
# ===========================================================================

MAX_AUDIO_REPLACE_BODY = 64 * 1024 * 1024  # 64 MB multipart cap for replace

# Per-bank write locks for the audio Replace verb (one writer per .pac/.ogg).
_AUDIO_LOCKS: "OrderedDict[str, threading.Lock]" = OrderedDict()
MAX_AUDIO_LOCKS = 64


def _audio_resolve_read(filename: str) -> Tuple[Path, bytes]:
    """Resolve an audio asset under DATA_DIR then LIVE_DATA_DIR (read-only) and
    return ``(path, bytes)``. 404 if missing, 413 if over the raw cap."""
    p = _resolve_under_roots(
        filename,
        (DATA_DIR, LIVE_DATA_DIR),
        label="filename",
        missing_msg=f"audio asset not found in DATA_DIR or LIVE_DATA_DIR: {filename}",
    )
    sz = p.stat().st_size
    if sz > MAX_RAW_RESPONSE_BYTES * 4:  # banks can be ~9 MB; allow generous read
        raise HTTPException(413, f"audio file too large to load: {sz} bytes")
    return p, p.read_bytes()


def _audio_kind_for(filename: str) -> Tuple[str, str, str]:
    """Return ``(container, codec, decode_kind)`` or 400 if not an audio file."""
    info = audio_mod.classify_audio(filename)
    if info is None:
        raise HTTPException(400, f"not an audio container this suite handles: {filename}")
    return info


@app.get("/api/audio/info")
def api_audio_info(path: str):
    """Container/codec/ffmpeg/records[]/replace_supported for an audio asset.

    For a .pac bank, ``records`` is the per-record summary (index, structured,
    bytes, pcm_bytes, sample_rate, duration_s). For .ogg/.sfd/.wav the record
    list is a single logical stream. ``replace_supported`` is True only for
    .pac (and only when the whole bank is structured) and .ogg (ffmpeg-gated).
    """
    container, codec, kind = _audio_kind_for(path)
    p, blob = _audio_resolve_read(path)
    have_ffmpeg = audio_mod.ffmpeg_available()

    records: List[dict] = []
    warnings: List[str] = []
    replace_supported = False

    if kind == "pac":
        bank = audio_mod.parse_pac(blob)
        records = audio_mod.summarize_bank(bank)
        warnings = list(bank.warnings)
        # Only a fully-structured bank is a safe per-record Replace target.
        replace_supported = bank.replace_safe
    elif kind == "ogg":
        records = [{
            "index": 0, "structured": True, "bytes": len(blob),
            "codec": "Ogg Vorbis", "stream": "browser-native",
        }]
        # Whole-file .ogg replace needs ffmpeg only if a non-ogg upload must
        # be transcoded; a like-for-like .ogg upload is a pure byte copy, so
        # replace is always offered for .ogg.
        replace_supported = True
    elif kind == "sfd":
        records = [{
            "index": 0, "structured": have_ffmpeg, "bytes": len(blob),
            "codec": "ASF/WMV (WMV3 + WMAv2)",
            "stream": "ffmpeg" if have_ffmpeg else "ffmpeg-required",
        }]
        replace_supported = False  # A/V movie: never a replace target
    elif kind == "wav":
        records = [{"index": 0, "structured": True, "bytes": len(blob), "codec": "PCM"}]
        replace_supported = False

    return {
        "path": p.name,
        "container": container,
        "codec": codec,
        "decode_kind": kind,
        "ffmpeg": have_ffmpeg,
        "records": records,
        "record_count": len(records),
        "replace_supported": replace_supported,
        "warnings": warnings,
    }


@app.get("/api/audio/decode")
def api_audio_decode(path: str, record: int = 0):
    """Decode one audio record to ``audio/wav``.

    .pac  -> pure-Python per-record PCM -> WAV (under the raw cap).
    .ogg  -> ffmpeg decode to WAV (501 if absent); the frontend normally just
             plays the .ogg natively via /api/raw, but a server-side WAV is
             offered for parity / waveform.
    .sfd  -> ffmpeg audio-track decode to WAV (501 if absent).
    .wav  -> passthrough.
    """
    container, codec, kind = _audio_kind_for(path)
    p, blob = _audio_resolve_read(path)

    if kind == "pac":
        bank = audio_mod.parse_pac(blob)
        if record < 0 or record >= len(bank.records):
            raise HTTPException(404, f"record {record} out of range (count={len(bank.records)})")
        rec = bank.records[record]
        if not rec.structured:
            raise HTTPException(
                422, f"record {record} is an opaque/variant record with no decodable PCM")
        wav = audio_mod.record_to_wav(rec)
        if len(wav) > MAX_RAW_RESPONSE_BYTES:
            raise HTTPException(413, f"decoded WAV too large: {len(wav)} bytes")
        return Response(content=wav, media_type="audio/wav")

    if kind == "wav":
        return Response(content=blob, media_type="audio/wav")

    if kind in ("ogg", "sfd"):
        # The .sfd intro movie's audio track can run minutes -> a full WAV
        # would blow the raw cap. Decode it downmixed to 22050/mono and cap
        # the preview to ~5 min (22050*2*300 = ~12.6 MB < MAX_RAW_RESPONSE_BYTES);
        # ogg is left native (already small).
        if kind == "sfd":
            sfd_sr, sfd_ch, max_secs = 22050, 1, 300.0
        else:
            sfd_sr = sfd_ch = max_secs = None
        try:
            wav = audio_mod.decode_to_wav(
                blob, kind, sample_rate=sfd_sr, channels=sfd_ch, max_seconds=max_secs)
        except audio_mod.FfmpegUnavailable as e:
            raise HTTPException(501, f"ffmpeg required to decode {container}: {e}")
        except audio_mod.FfmpegError as e:
            raise HTTPException(502, f"ffmpeg decode failed: {e}")
        except ValueError as e:
            raise HTTPException(400, str(e))
        if len(wav) > MAX_RAW_RESPONSE_BYTES:
            raise HTTPException(413, f"decoded WAV too large: {len(wav)} bytes")
        return Response(content=wav, media_type="audio/wav")

    raise HTTPException(400, f"cannot decode {container}")


@app.get("/api/audio/waveform")
def api_audio_waveform(path: str, record: int = 0, buckets: int = 600):
    """Downsampled (min,max,rms) peaks for an audio record's overview canvas.

    .pac: pure-Python directly from the record's PCM. .ogg/.sfd/.wav: decoded
    to PCM via ffmpeg (501 if absent) or stdlib (wav), then downsampled.
    """
    container, codec, kind = _audio_kind_for(path)
    p, blob = _audio_resolve_read(path)
    buckets = max(1, min(int(buckets), 4000))

    pcm: bytes
    channels = audio_mod.PCM_CHANNELS
    if kind == "pac":
        bank = audio_mod.parse_pac(blob)
        if record < 0 or record >= len(bank.records):
            raise HTTPException(404, f"record {record} out of range (count={len(bank.records)})")
        rec = bank.records[record]
        if not rec.structured:
            raise HTTPException(422, f"record {record} has no decodable PCM")
        pcm = rec.pcm
    elif kind == "wav":
        pcm, _rate, channels, _bits = audio_mod.wav_to_pcm(blob)
    elif kind in ("ogg", "sfd"):
        try:
            wav = audio_mod.decode_to_wav(blob, kind)
        except audio_mod.FfmpegUnavailable as e:
            raise HTTPException(501, f"ffmpeg required for {container} waveform: {e}")
        except audio_mod.FfmpegError as e:
            raise HTTPException(502, f"ffmpeg decode failed: {e}")
        pcm, _rate, channels, _bits = audio_mod.wav_to_pcm(wav)
    else:
        raise HTTPException(400, f"cannot compute waveform for {container}")

    peaks = audio_mod.waveform_peaks(pcm, buckets=buckets, channels=channels)
    peaks["path"] = p.name
    peaks["record"] = record
    return peaks


@app.post("/api/audio/replace")
async def api_audio_replace(
    request: Request,
    path: str = Form(...),
    record: int = Form(0),
    deploy: bool = Form(False),
    normalize: bool = Form(False),
    trim_start: int = Form(0),
    trim_end: int = Form(-1),
    file: UploadFile = File(...),
):
    """DEV-ONLY Replace: swap a .pac record's PCM (or replace an .ogg) with an
    uploaded .wav/.ogg.

    deploy=false -> mint an export token + preview URL (NOTHING written to the
                    data tree).
    deploy=true  -> write the rebuilt bank/ogg to DEV_DATA_DIR ONLY, atomically
                    (.tmp + os.replace), with a `.pre_promote_<ts>` backup on
                    overwrite. The output path is hard-asserted by
                    `_floor_assert_not_live` to be inside DEV and never LIVE.

    .sfd / .adx are never replace targets (HTTP 400). A concurrent write to the
    same target returns HTTP 409. ffmpeg-needing conversions degrade to 501.
    """
    _enforce_body_size(request, MAX_AUDIO_REPLACE_BODY)
    container, codec, kind = _audio_kind_for(path)
    if not audio_mod.replace_supported(path):
        raise HTTPException(400, f"{container} is not a replace target (only .pac and .ogg)")

    bare = _validate_bare_filename(path, label="path")
    upload = await file.read()
    if not upload:
        raise HTTPException(400, "empty upload")
    if len(upload) > MAX_AUDIO_REPLACE_BODY:
        raise HTTPException(413, f"upload too large: {len(upload)} bytes")
    upload_name = (file.filename or "").lower()

    lk = _get_lock(_AUDIO_LOCKS, bare, MAX_AUDIO_LOCKS)
    if not lk.acquire(blocking=False):
        raise HTTPException(409, f"audio replace already in progress for {bare}")
    try:
        # Resolve & read the source bank/file (DATA_DIR then LIVE_DATA_DIR).
        _src_path, blob = _audio_resolve_read(bare)

        if kind == "pac":
            new_bytes = _audio_build_replaced_pac(
                blob, record, upload, upload_name, normalize, trim_start, trim_end)
        elif kind == "ogg":
            new_bytes = _audio_build_replaced_ogg(upload, upload_name)
        else:  # pragma: no cover - guarded by replace_supported above
            raise HTTPException(400, f"{container} is not a replace target")

        if not deploy:
            # Preview only: stage under a token, write NOTHING into the tree.
            _gc_export_tokens()
            token = _make_audio_export_token(new_bytes, bare)
            return {
                "ok": True,
                "deployed": False,
                "path": bare,
                "record": record if kind == "pac" else None,
                "new_size": len(new_bytes),
                "export_token": token,
                "export_url": f"/api/export/{token}",
                "export_filename": bare,
            }

        # deploy=True: DEV-ONLY atomic write with backup. Hard-assert target.
        target = (DEV_DATA_DIR / bare)
        target = _floor_assert_not_live(target)  # RuntimeError if LIVE/escape
        DEV_DATA_DIR.mkdir(parents=True, exist_ok=True)

        backup_path = None
        if target.exists():
            ts = time.strftime("%Y%m%d_%H%M%S")
            bak = target.with_name(f"{target.name}.pre_promote_{ts}")
            counter = 0
            while bak.exists():
                counter += 1
                bak = target.with_name(f"{target.name}.pre_promote_{ts}_{counter}")
            _floor_assert_not_live(bak)
            shutil.copy(target, bak)
            backup_path = str(bak)

        _floor_atomic_write(target, new_bytes)
        log.info("audio replace DEPLOYED (DEV) %s record=%s new=%d backup=%s",
                 bare, record, len(new_bytes), backup_path)
        return {
            "ok": True,
            "deployed": True,
            "path": str(target),
            "record": record if kind == "pac" else None,
            "new_size": len(new_bytes),
            "backup_path": backup_path,
        }
    finally:
        lk.release()


def _make_audio_export_token(data: bytes, filename: str) -> str:
    """Stage replaced audio bytes under a random export token (preview path).

    Mirrors `_make_export_token` but writes raw bytes (no .prs/.xvm suffix
    coercion) so a .pac/.ogg preview keeps its real extension.
    """
    token = secrets.token_urlsafe(24)
    ext = Path(filename).suffix or ".bin"
    staged = EXPORT_DIR / f"{token}{ext}"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(data)
    entry = {
        "path": str(staged),
        "filename": filename,
        "expires_at": time.time() + EXPORT_TTL_SECONDS,
    }
    with _EXPORT_TOKENS_LOCK:
        _EXPORT_TOKENS[token] = entry
    _persist_export_token(token, entry)
    return token


def _make_bytes_export_token(data: bytes, filename: str) -> str:
    """Stage arbitrary bytes under a random export token (download path).

    Generalises ``_make_audio_export_token`` for any artifact type — model
    exports (.obj/.zip/.glb) stage their bytes here and the existing
    ``GET /api/export/<token>`` route streams them with the real filename.
    The staged file keeps the filename's real extension so the download is
    named/typed correctly.
    """
    token = secrets.token_urlsafe(24)
    ext = Path(filename).suffix or ".bin"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    staged = EXPORT_DIR / f"{token}{ext}"
    staged.write_bytes(data)
    entry = {
        "path": str(staged),
        "filename": filename,
        "expires_at": time.time() + EXPORT_TTL_SECONDS,
    }
    with _EXPORT_TOKENS_LOCK:
        _EXPORT_TOKENS[token] = entry
    _persist_export_token(token, entry)
    return token


def _audio_pcm_from_upload(upload: bytes, upload_name: str) -> bytes:
    """Decode an uploaded .wav/.ogg to PSOBB-native 22050/mono/16 PCM bytes.

    .wav: stdlib `wave`. If it isn't already 22050/mono/16, ffmpeg is used to
          resample/downmix (501 if ffmpeg absent for a non-native WAV).
    .ogg: ffmpeg decode (501 if absent).
    Raises HTTPException on bad input / missing ffmpeg.
    """
    sr_native = audio_mod.PCM_SAMPLE_RATE
    if upload_name.endswith(".wav") or upload[:4] == b"RIFF":
        try:
            pcm, rate, channels, bits = audio_mod.wav_to_pcm(upload)
        except ValueError as e:
            raise HTTPException(400, f"bad WAV upload: {e}")
        if rate == sr_native and channels == 1 and bits == 16:
            return pcm
        # Non-native WAV: ffmpeg resample/downmix to 22050/mono/16
        # (decode_to_wav only knows ogg/sfd demuxers, so run ffmpeg
        # WAV->WAV directly with the target framing).
        try:
            conv = audio_mod.audio_codec._run_ffmpeg(
                ["-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
                 "-f", "wav", "-acodec", "pcm_s16le",
                 "-ar", str(sr_native), "-ac", "1", "pipe:1"],
                upload,
            )
        except audio_mod.FfmpegUnavailable as e:
            raise HTTPException(
                501, f"ffmpeg required to convert WAV to 22050/mono/16: {e}")
        except audio_mod.FfmpegError as e:
            raise HTTPException(400, f"WAV conversion failed: {e}")
        pcm, _r, _c, _b = audio_mod.wav_to_pcm(conv)
        return pcm

    if upload_name.endswith(".ogg") or upload[:4] == b"OggS":
        try:
            wav = audio_mod.decode_to_wav(upload, "ogg", sample_rate=sr_native, channels=1)
        except audio_mod.FfmpegUnavailable as e:
            raise HTTPException(501, f"ffmpeg required to decode Ogg upload: {e}")
        except audio_mod.FfmpegError as e:
            raise HTTPException(400, f"Ogg decode failed: {e}")
        pcm, _r, _c, _b = audio_mod.wav_to_pcm(wav)
        return pcm

    raise HTTPException(400, "upload must be a .wav or .ogg file")


def _audio_build_replaced_pac(blob: bytes, record: int, upload: bytes,
                              upload_name: str, normalize: bool,
                              trim_start: int, trim_end: int) -> bytes:
    """Build new .pac bytes with ``record``'s PCM swapped for the upload."""
    bank = audio_mod.parse_pac(blob)
    if not bank.replace_safe:
        raise HTTPException(
            400, "this .pac bank has opaque/variant records; replace is disabled for it")
    if record < 0 or record >= len(bank.records):
        raise HTTPException(404, f"record {record} out of range (count={len(bank.records)})")

    pcm = _audio_pcm_from_upload(upload, upload_name)
    if trim_start or (trim_end is not None and trim_end >= 0):
        end = None if (trim_end is None or trim_end < 0) else trim_end
        pcm = audio_mod.trim_pcm(pcm, start_frame=trim_start, end_frame=end)
    if normalize:
        pcm = audio_mod.normalize_pcm(pcm)
    if not pcm:
        raise HTTPException(400, "converted PCM is empty")

    try:
        new_bank = audio_mod.replace_record_pcm(bank, record, pcm)
    except (ValueError, IndexError, TypeError) as e:
        raise HTTPException(400, f"record replace failed: {e}")
    return audio_mod.write_pac(new_bank)


def _audio_build_replaced_ogg(upload: bytes, upload_name: str) -> bytes:
    """Build replacement .ogg bytes from the upload.

    A .ogg upload is copied byte-for-byte (like-for-like). A .wav upload is
    encoded to Ogg Vorbis via ffmpeg (501 if absent).
    """
    if upload_name.endswith(".ogg") or upload[:4] == b"OggS":
        return upload
    if upload_name.endswith(".wav") or upload[:4] == b"RIFF":
        try:
            return audio_mod.encode_ogg(upload, in_kind="wav")
        except audio_mod.FfmpegUnavailable as e:
            raise HTTPException(501, f"ffmpeg required to encode WAV to Ogg: {e}")
        except audio_mod.FfmpegError as e:
            raise HTTPException(400, f"Ogg encode failed: {e}")
    raise HTTPException(400, "ogg replace upload must be a .ogg or .wav file")


# ---------------------------------------------------------------------------
# /api/repack_bml_inner (2026-04-30)
#
# Background:
#   /api/repack uses safe_data_path() which only accepts BARE filenames
#   ('foo.xvm'); '<base>#<inner>' inner-syntax is rejected. BML-resident
#   inner XVM textures (the '<entry>.xvm' branch of _extract_bml_inner_bytes
#   — used for the per-entry texture archive that follows each NJ payload
#   inside the BML) extract + upscale fine today but couldn't be redeployed.
#
# This endpoint plugs the gap. It materialises the inner via the same
# _materialize_inner_for_extract cache the rest of the editor uses, runs
# the standard tile rebuild, and either:
#   - mints an export token (deploy=False), OR
#   - re-PRS-compresses the rebuilt XVMH bytes, splices them into the
#     correct entry of the parent BML via parse_bml_for_pack/pack_bml,
#     and atomic-replaces both LIVE_DATA_DIR and DATA_DIR copies
#     (deploy=True) with a `.pre_promote_<TS>` backup.
#
# Lock: keyed on the parent BML filename, NOT on the inner name — two
# concurrent inner repacks of the same BML would otherwise race the
# BML write step and clobber each other's blob splice.
#
# Format note: the user can pass either '<entry>.xvm' (the per-entry
# texture archive — the normal use case for texture editing) OR
# '<entry>.nj' / '<entry>.xj' for the model payload itself. The
# extract-time code in _extract_bml_inner_bytes dispatches on the
# inner_name's '.xvm' suffix; we mirror its decision and only re-pack
# the texture half of the entry. Mesh-payload edits go through the
# import/replace pipeline (api_import_replace) which has different
# semantics (it builds a new BML in BML_EXPORT_DIR; this endpoint
# atomically replaces the live BML in place).
# ---------------------------------------------------------------------------
class RepackBmlInnerReq(BaseModel):
    bml: str
    inner_name: str
    tiles: list[TileEdit] = Field(default_factory=list)
    deploy: bool = False


@app.post("/api/repack_bml_inner")
def api_repack_bml_inner(req: RepackBmlInnerReq, request: Request):
    """Repack one inner XVM of a BML container; optional in-place deploy.

    Body:
        bml           bare BML filename (e.g. 'bm_obj_ep4_boss09_core.bml').
        inner_name    inner entry name. For texture-archive editing, this
                      is '<entry>.xvm' (the per-entry XVM archive that
                      follows each NJ payload inside the BML). The
                      endpoint REQUIRES the '.xvm' suffix because the
                      mesh-payload re-pack path is owned by
                      /api/import/replace.
        tiles         list of TileEdit (may be empty for a no-op rebuild
                      that round-trips the inner). Same shape as the
                      /api/repack body.
        deploy        False ⇒ mint an export token + URL for the rebuilt
                      inner XVM (caller can download/inspect).
                      True  ⇒ splice rebuilt bytes back into the parent
                      BML via formats.bml.pack_bml, write atomically to
                      LIVE_DATA_DIR (with a .pre_promote_<TS> backup of
                      the live copy) and mirror to DATA_DIR.

    Returns the same metric fields /api/repack does (rebuilt_size,
    spliced/reencoded counts, changed_indices, verify), plus BML-level
    fields when deploy=True (bml_original_size, bml_new_size,
    bml_backup_path).
    """
    _enforce_body_size(request, MAX_REPACK_BODY)
    if len(req.tiles) > MAX_TILES_PER_REPACK:
        raise HTTPException(400, f"too many tile edits ({len(req.tiles)} > {MAX_TILES_PER_REPACK})")

    # Validate the BML name + extension up front. Both DEV and LIVE
    # roots are searched (read-only fallback).
    bml_name = _validate_bare_filename(req.bml, label="bml")
    if not bml_name.lower().endswith(".bml"):
        raise HTTPException(400, f"not a BML container: {bml_name}")
    inner = req.inner_name
    _validate_inner_name(inner, msg="invalid inner_name (path components forbidden)", required=True)
    # Only the texture-archive branch is supported here. Mesh-payload
    # repacks live in /api/import/replace.
    if not inner.lower().endswith(".xvm"):
        raise HTTPException(
            400,
            f"inner_name must end in '.xvm' (got {inner!r}); use /api/import/replace for mesh-payload edits",
        )

    # Materialise the inner via the standard cache path. _extract_bml_inner_bytes
    # PRS-decompresses the texture blob; the cached file is raw XVMH bytes.
    composite_path = f"{bml_name}#{inner}"
    try:
        inner_path = _materialize_inner_for_extract(composite_path)
    except HTTPException:
        raise
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"BML inner materialise failed: {e}")
    if not inner_path.exists():
        raise HTTPException(404, "materialised inner missing on disk")

    # One writer per BML (NOT per inner) so concurrent inner-repacks of
    # the same BML don't race the splice + atomic rename of the parent
    # BML bytes.
    lock_key = bml_name
    lk = _get_lock(_REPACK_LOCKS, lock_key, MAX_REPACK_LOCKS)
    if not lk.acquire(blocking=False):
        raise HTTPException(409, f"repack already in progress for {bml_name}")
    try:
        inner_label = composite_path
        result = _apply_edits_and_rebuild(inner_path, req.tiles, label=inner_label)
        rebuilt = result["rebuilt_path"]
        rebuilt_size = result["rebuilt_size"]
        verify = result["verify"]
        spliced = result["spliced"]
        reencoded = result["reencoded"]
        changed_indices = result["changed_indices"]

        deploy_path = None
        bml_backup_path = None
        bml_original_size = None
        bml_new_size = None
        export_token = None
        export_url = None
        export_filename = None

        if not req.deploy:
            # Export-only: stage the rebuilt XVMH (raw, no PRS layer —
            # BML inner textures are decompressed at materialise time,
            # so the rebuilt bytes are raw XVMH). Caller can download
            # via /api/export/<token>.
            _gc_export_tokens()
            # Stem the inner_name's '.xvm' so a download as
            # 'bm_obj_ep4_boss09_core01.nj.xvm' is unambiguous.
            export_filename = inner
            export_token = _make_export_token(rebuilt, export_filename)
            export_url = f"/api/export/{export_token}"
        else:
            # Splice into the parent BML and write back to disk.
            # 1. Read the live (or DEV-mirror) BML bytes.
            live_bml = LIVE_DATA_DIR / bml_name
            dev_bml = DATA_DIR / bml_name
            if live_bml.exists():
                src_bml = live_bml
            elif dev_bml.exists():
                src_bml = dev_bml
            else:
                raise HTTPException(404, f"BML not found in LIVE or DEV: {bml_name}")
            try:
                bml_buf = src_bml.read_bytes()
            except OSError as e:
                raise HTTPException(500, f"BML read failed: {e}")
            bml_original_size = len(bml_buf)

            # 2. parse_bml_for_pack + parse_bml_pack_meta gives byte-exact
            #    round-trip metadata (compression, alignment, has_textures
            #    flag including the lying-flag preservation for player NJ
            #    archives).
            try:
                from formats.bml import parse_bml_for_pack, parse_bml_pack_meta
                pack_entries = parse_bml_for_pack(bml_buf)
                meta = parse_bml_pack_meta(bml_buf)
            except (ValueError, RuntimeError) as e:
                raise HTTPException(400, f"BML parse failed: {e}")

            # 3. Find the target entry. The inner_name is '<entry>.xvm'
            #    (the per-entry texture archive); strip '.xvm' to recover
            #    the BML entry name.
            ent_name = inner[: -len(".xvm")]
            matched = -1
            for i, ent in enumerate(pack_entries):
                if ent.name == ent_name:
                    matched = i
                    break
            if matched < 0:
                names = [e.name for e in pack_entries]
                raise HTTPException(
                    404,
                    f"entry {ent_name!r} not in BML; have: {names[:10]}",
                )
            target = pack_entries[matched]
            if target.texture_data is None or len(target.texture_data) == 0:
                raise HTTPException(
                    400,
                    f"entry {ent_name!r} has no texture archive to repack",
                )

            # 4. PRS-recompress the rebuilt XVMH bytes. BML textures are
            #    ALWAYS PRS-compressed regardless of the container's
            #    compression byte (per pso-blender + the BML format
            #    note in formats/bml.py). Reuse the same PuyoToolsCli-
            #    via-cwd idiom /api/repack uses for top-level PRS files
            #    so byte output is identical to a /api/repack PRS path.
            cdir = result["cache_dir"]
            cpath = cdir / "compress_in"
            cpath.mkdir(exist_ok=True)
            for old in cpath.glob("*"):
                try:
                    old.unlink()
                except OSError as e:
                    log.debug("could not unlink %s: %s", old, e)
            stem_in = f"inner_{ent_name}"
            staged = cpath / stem_in
            shutil.copy(rebuilt, staged)
            sh(
                [PUYO, "compression", "compress", "prs", "--overwrite", "-i", stem_in],
                cwd=cpath,
                timeout=TIMEOUT_PUYO,
            )
            new_tex_prs = staged.read_bytes()
            new_tex_decomp_size = rebuilt_size

            # 5. Splice the new pre-compressed texture into the matched
            #    BmlPackEntry. Preserve every other field (data, unk_a/b/c/d,
            #    is_compressed) so the rest of the archive is byte-exact
            #    with the source modulo this one entry's texture.
            pack_entries[matched] = BmlPackEntry(
                name=target.name,
                data=target.data,
                decompressed_size=target.decompressed_size,
                is_compressed=target.is_compressed,
                texture_data=new_tex_prs,
                texture_decompressed_size=new_tex_decomp_size,
                texture_is_compressed=True,
                unk_a=target.unk_a,
                unk_b=target.unk_b,
                unk_c=target.unk_c,
                unk_d=target.unk_d,
            )

            # 6. Re-pack with the original archive-level metadata so the
            #    lying-flag (has_textures=1, alignment=0x800 for the 23
            #    player NJ archives) is preserved.
            try:
                new_bml_bytes = pack_bml(
                    pack_entries,
                    compression=meta["compression"],
                    file_alignment=meta["file_alignment"],
                    has_textures_override=bool(meta.get("has_textures", False)),
                )
            except (ValueError, RuntimeError) as e:
                raise HTTPException(500, f"BML pack failed: {e}")
            bml_new_size = len(new_bml_bytes)

            # 7. Atomic deploy. Backup the LIVE copy first, then write
            #    via .tmp + os.replace so a crash can't leave a torn
            #    archive on disk. DEV mirror is updated last so a
            #    LIVE-only failure still leaves DEV consistent with
            #    the previous good state.
            ts = time.strftime("%Y%m%d_%H%M%S")
            if live_bml.exists():
                bak = live_bml.parent / f"{bml_name}.pre_promote_{ts}"
                counter = 0
                while bak.exists():
                    counter += 1
                    bak = live_bml.parent / f"{bml_name}.pre_promote_{ts}_{counter}"
                shutil.copy(live_bml, bak)
                bml_backup_path = str(bak)
                tmp_live = live_bml.with_suffix(live_bml.suffix + ".tmp")
                tmp_live.write_bytes(new_bml_bytes)
                os.replace(tmp_live, live_bml)
                deploy_path = str(live_bml)

            # Mirror to DEV so subsequent /api/tiles / extract calls see
            # the new bytes.
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            tmp_dev = dev_bml.with_suffix(dev_bml.suffix + ".tmp")
            tmp_dev.write_bytes(new_bml_bytes)
            os.replace(tmp_dev, dev_bml)
            if deploy_path is None:
                deploy_path = str(dev_bml)

            # Invalidate the cached materialised-inner blob for this BML
            # so the next /api/tiles read picks up the new texture.
            try:
                # _materialize_inner_for_extract keys its scratch dir on
                # (BML name, size, mtime, inner). Both have changed; the
                # cleanest fix is rmtree on the per-BML scratch parent —
                # _BML_INNER_CACHE_SUBDIR / digest is keyed on the OLD
                # mtime so subsequent mtime-based digests will miss it,
                # but the orphan dir lingers on disk until cleanup_cache.
                # We don't have a forward pointer, so just leave it; the
                # next call mints a fresh digest dir for the new mtime.
                pass
            except OSError as e:  # pragma: no cover — best-effort
                log.debug("could not invalidate BML inner cache: %s", e)

            log.info(
                "deployed BML inner %s#%s (rebuilt=%d new_bml=%d backup=%s)",
                bml_name, inner, rebuilt_size, bml_new_size, bml_backup_path,
            )

        return {
            "bml": bml_name,
            "inner_name": inner,
            "rebuilt_size": rebuilt_size,
            "rebuilt_path": str(rebuilt),
            "verify": verify,
            "spliced_count": spliced,
            "reencoded_count": reencoded,
            "changed_indices": changed_indices,
            "deploy_path": deploy_path,
            "bml_backup_path": bml_backup_path,
            "bml_original_size": bml_original_size,
            "bml_new_size": bml_new_size,
            "export_token": export_token,
            "export_url": export_url,
            "export_filename": export_filename,
        }
    finally:
        lk.release()


@app.get("/api/export/{token}")
def api_export(token: str, filename: Optional[str] = None):
    """Stream a previously-built export artifact for download.

    `token` is returned by /api/repack with deploy=false. Tokens expire
    after EXPORT_TTL_SECONDS. Optional `filename` query param overrides
    the suggested download filename (defaults to the original PRS/XVM name).
    """
    # Token format guard (no path traversal possible).
    if not EXPORT_TOKEN_RE.match(token):
        raise HTTPException(400, "invalid token format")
    _gc_export_tokens()
    with _EXPORT_TOKENS_LOCK:
        entry = _EXPORT_TOKENS.get(token)
    if not entry:
        # Wave 7: workers=4 case — token was minted by a peer worker we
        # don't share memory with. Fall back to the on-disk sidecar
        # written by `_persist_export_token`.
        entry = _load_export_token(token)
        if entry:
            # Cache the entry locally so subsequent GETs skip the disk
            # read. The TTL is owned by the sidecar, not memory, so a
            # stale local cache is harmless (gc deletes the sidecar).
            with _EXPORT_TOKENS_LOCK:
                _EXPORT_TOKENS[token] = entry
    if not entry:
        raise HTTPException(404, "export token not found or expired")
    if entry["expires_at"] < time.time():
        raise HTTPException(404, "export token not found or expired")
    p = Path(entry["path"])
    if not p.exists():
        with _EXPORT_TOKENS_LOCK:
            _EXPORT_TOKENS.pop(token, None)
        try:
            _export_token_index_path(token).unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(404, "export artifact missing on disk")
    download_name = filename or entry.get("filename") or p.name
    # Guard the user-provided filename: only basename component
    download_name = Path(download_name).name
    if not download_name:
        download_name = p.name
    return FileResponse(
        p,
        media_type="application/octet-stream",
        filename=download_name,
    )


# --------------------------------------------------------------------------- #
# Model export — OBJ / GLB (Blender-friendly) with textures
# --------------------------------------------------------------------------- #
def _export_default_texture_archive(base: str, inner: Optional[str]) -> Optional[str]:
    """Mirror the frontend ``deriveTextureArchivePath`` for the export route.

    For a BML/AFS inner the texture sibling is ``<base>#<inner>.xvm``; for a
    top-level model it is the same-stem ``<stem>.xvm``.
    """
    if inner:
        return f"{base}#{inner}.xvm"
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", base)
    return f"{stem}.xvm"


def _export_tile_png_bytes(archive_path: str, tile_idx: int) -> Optional[bytes]:
    """Decode one texture tile to PNG bytes, reusing the tile_png pipeline.

    Returns None on any failure (missing archive / tile) so the export can
    degrade to an untextured material rather than aborting.
    """
    try:
        prs = _materialize_inner_for_extract(archive_path)
    except HTTPException:
        return None
    except Exception:  # pragma: no cover - defensive
        return None
    if not prs.exists():
        return None
    if tile_idx < 0 or tile_idx > MAX_TILE_INDEX:
        return None
    try:
        manifest = extract_tiles(prs)
    except Exception:
        return None
    tile = next((t for t in manifest["tiles"] if t["index"] == tile_idx), None)
    if not tile:
        return None
    png_path = Path(manifest["tiles_dir"]) / tile["filename"]
    if not png_path.exists():
        return None
    try:
        return png_path.read_bytes()
    except OSError:
        return None


def _export_resolve_binding_textures(
    binding: list, default_archive: Optional[str],
) -> tuple[dict, list]:
    """Resolve a model's binding rows to ``{material_id: png_bytes}``.

    Mirrors the per-row archive/tile resolution in the frontend's
    ``fetchBoundTextures`` (in_bml / cross_bml / cross_afs). Returns
    ``(textures_by_material_id, warnings)``. Missing tiles are recorded as
    warnings and simply absent from the dict (the exporter emits an
    untextured material for them).
    """
    textures: dict[int, bytes] = {}
    warnings: list[str] = []
    # Cache decoded tiles by (archive, tile) so duplicate refs decode once.
    cache: dict[tuple[str, int], Optional[bytes]] = {}

    for row in binding or []:
        mid = int(row.get("material_id", 0))
        source = row.get("source") or ("missing" if row.get("missing") else "in_bml")
        archive = default_archive
        tile = int(row.get("tile_index", 0) or 0)

        if source == "cross_bml" and row.get("cross_bml"):
            cb = row["cross_bml"]
            if cb.get("bml") and cb.get("inner"):
                archive = f"{cb['bml']}#{cb['inner']}.xvm"
                tile = int(cb.get("xvr_index", 0) or 0)
        elif source == "cross_afs" and row.get("cross_afs"):
            ca = row["cross_afs"]
            arc = ca.get("archive")
            inner_index = int(ca.get("inner_index", -1))
            if arc and inner_index >= 0:
                stem = re.sub(r"\.afs$", "", arc, flags=re.IGNORECASE)
                idx4 = f"{inner_index:04d}"
                archive = f"{arc}#{idx4}_{stem}_{idx4}.xvr"
                tile = max(0, int(ca.get("xvr_index", 0) or 0))
        elif row.get("missing"):
            # Unresolved row with no cross-archive fallback — skip.
            warnings.append(f"material {mid}: no texture resolved")
            continue

        if not archive:
            continue
        key = (archive, tile)
        if key not in cache:
            cache[key] = _export_tile_png_bytes(archive, tile)
        png = cache[key]
        if png:
            textures[mid] = png
        else:
            warnings.append(
                f"material {mid}: texture tile {tile} in {archive} not decodable"
            )
    return textures, warnings


def _export_meshes_to_export_meshes(meshes: list):
    """Convert parsed ``XjMesh`` list to ``model_export.ExportMesh`` list.

    Vertices are already world-baked, so positions are taken verbatim.
    """
    from formats.model_export import ExportMesh

    out = []
    for m in meshes:
        positions = [tuple(v.pos) for v in m.vertices]
        normals = [tuple(v.normal) for v in m.vertices]
        uvs = [tuple(v.uv) for v in m.vertices]
        out.append(ExportMesh(
            positions=positions,
            indices=list(m.indices),
            normals=normals,
            uvs=uvs,
            material_id=int(getattr(m, "material_id", 0)),
        ))
    return out


class ModelExportReq(BaseModel):
    path: str
    format: str = "glb"  # obj | glb | fbx
    inner: Optional[str] = None


@app.get("/api/export_model/capabilities")
def api_export_model_capabilities():
    """Report which model export formats are writable.

    FBX has no pure-Python writer available, so it is advertised as
    unsupported; OBJ and GLB are produced via the model_export module.
    """
    return {"obj": True, "glb": True, "fbx": False}


@app.post("/api/export_model")
def api_export_model(req: ModelExportReq):
    """Rebuild a model's mesh + bound textures and export it for Blender.

    ``format`` is ``obj`` (zip of model.obj + model.mtl + PNGs), ``glb``
    (single self-contained binary glTF), or ``fbx`` (501 — no writer).

    The mesh is rebuilt via the SAME path the 3D viewer uses
    (``/api/model_mesh``) and the textures come from the RESOLVED binding,
    so what you export matches what you see. Returns
    ``{export_url, warnings[]}``; the artifact is staged under an export
    token and streamed by ``GET /api/export/<token>``.
    """
    fmt = (req.format or "glb").strip().lower()
    if fmt == "fbx":
        raise HTTPException(
            501, "no FBX writer available; use GLB or OBJ for Blender import"
        )
    if fmt not in ("obj", "glb"):
        raise HTTPException(400, f"unsupported export format {fmt!r} (obj|glb|fbx)")

    # Resolve the model exactly like /api/model_mesh.
    base, effective_inner = _split_inner_with_query(req.path, req.inner)
    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()
    inner_ext = ""

    if ext == ".bml":
        if not effective_inner:
            raise HTTPException(
                400, "BML model requires '#<inner>' or inner field"
            )
        _validate_inner_name(effective_inner, msg="invalid inner entry name")
        inner_ext = Path(effective_inner).suffix.lower()
        if inner_ext == "" and len(effective_inner) == 32 and effective_inner.endswith("."):
            inner_ext = ".nj"
        if inner_ext not in IFF_EXTENSIONS:
            raise HTTPException(400, f"inner must be {IFF_EXTENSIONS!r}, got {inner_ext!r}")
        nj_bytes = _read_inner_nj_from_bml(p, effective_inner)
    elif ext == ".afs":
        if not effective_inner:
            raise HTTPException(400, "AFS model requires '#NNNN_<basename>' or inner field")
        nj_bytes, logical_inner = _read_afs_inner_nj(p, effective_inner)
        inner_ext = Path(logical_inner).suffix.lower() or ".nj"
    elif ext in IFF_EXTENSIONS:
        if effective_inner:
            raise HTTPException(400, f"'inner' not allowed for {ext} files")
        sz = p.stat().st_size
        if sz > ASSET_PARSE_MAX_BYTES:
            raise HTTPException(413, f"model too large: {sz} bytes")
        nj_bytes = p.read_bytes()
    else:
        raise HTTPException(
            400, f"unsupported model extension {ext!r} (expected .nj/.xj/.bml/.afs)"
        )

    try:
        meshes = _cached_model_parse(nj_bytes, p, ext, inner_ext, effective_inner)
    except ValueError as e:
        raise HTTPException(400, f"model parse failed: {e}")
    if not meshes:
        raise HTTPException(400, "no geometry parsed from model")

    warnings: list[str] = []

    # Resolve textures via the same binding the viewer uses.
    try:
        bd = _build_model_texture_binding_cached(p, ext, effective_inner, nj_bytes, meshes)
        binding = bd.get("binding") or []
    except HTTPException as e:
        binding = []
        warnings.append(f"texture binding unavailable: {getattr(e, 'detail', e)}")
    except Exception as e:  # pragma: no cover - defensive
        binding = []
        warnings.append(f"texture binding failed: {e}")

    default_arch = _export_default_texture_archive(base, effective_inner)
    textures, tex_warnings = _export_resolve_binding_textures(binding, default_arch)
    warnings.extend(tex_warnings)

    # Skinned rest-pose: note bone count if the skinned parser is cheap to
    # query. We don't re-parse; XjMesh carries no bone count, so we leave it
    # 0 unless a future cheap source exists. (Rest pose is what _cached_model_parse
    # returns: world-baked vertices.)
    bone_count = 0

    export_meshes = _export_meshes_to_export_meshes(meshes)
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", Path(base).name) or "model"
    if effective_inner:
        inner_stem = re.sub(r"\.[A-Za-z0-9]+$", "", Path(effective_inner).name)
        if inner_stem:
            stem = f"{stem}_{inner_stem}"

    from formats import model_export

    try:
        if fmt == "obj":
            bundle = model_export.build_obj_bundle(
                export_meshes, textures, model_name=stem, bone_count=bone_count,
            )
            # Zip the OBJ + MTL + PNGs into one downloadable archive.
            import io as _io
            import zipfile
            buf = _io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in bundle.items():
                    zf.writestr(name, data)
            artifact = buf.getvalue()
            out_name = f"{stem}.zip"
        else:  # glb
            artifact = model_export.build_glb_bundle(
                export_meshes, textures, model_name=stem, bone_count=bone_count,
            )
            out_name = f"{stem}.glb"
    except ValueError as e:
        raise HTTPException(400, f"export build failed: {e}")
    except Exception as e:  # pragma: no cover - defensive
        log.exception("model export build error for %s", req.path)
        raise HTTPException(500, f"export build internal error: {e}")

    if not textures:
        warnings.append("no textures resolved — exported model is untextured")

    _gc_export_tokens()
    token = _make_bytes_export_token(artifact, out_name)
    return {
        "export_url": f"/api/export/{token}?filename={out_name}",
        "filename": out_name,
        "format": fmt,
        "mesh_count": len(export_meshes),
        "texture_count": len(textures),
        "warnings": warnings,
    }


class RestoreReq(BaseModel):
    filename: str
    backup_name: Optional[str] = None  # exact backup file; if None, pick newest pre_editor_


@app.post("/api/restore_backup")
def api_restore_backup(req: RestoreReq, request: Request):
    """Roll back a deployed file to a backup.

    If `backup_name` is omitted, restores the most recent
    `<filename>.pre_editor_*` backup. The cache for the file is invalidated.
    """
    _enforce_body_size(request, MAX_RESTORE_BODY)
    prs = safe_data_path(req.filename)
    # Pick a backup
    if req.backup_name:
        bak = safe_data_path(req.backup_name)
        if not bak.exists():
            raise HTTPException(404, f"no such backup: {req.backup_name}")
        if not bak.name.startswith(req.filename + "."):
            raise HTTPException(400, "backup does not belong to this file")
    else:
        candidates = sorted(
            DATA_DIR.glob(f"{req.filename}.pre_editor_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise HTTPException(404, "no backups found")
        bak = candidates[0]
    shutil.copy(bak, prs)
    log.info("restored %s from %s", req.filename, bak.name)
    # Invalidate stale cache
    try:
        new_key = cache_key(prs)
        for cdir in CACHE_DIR.glob(f"{req.filename}_*"):
            if cdir.name != new_key:
                shutil.rmtree(cdir, ignore_errors=True)
    except OSError as e:
        log.warning("cache invalidation after restore failed: %s", e)
    return {"filename": req.filename, "restored_from": str(bak)}


@app.get("/api/verify/{filename}")
def api_verify(filename: str):
    """Quality verification: extract the deployed file and compare each tile's
    decoded PNG to the cached source PNG (if any pre-deploy cache survives).

    Returns:
      tile_count
      tiles: list of {index, width, height, fmt, identical_to_cache, psnr_db}
        where identical_to_cache is true iff the deployed tile decodes to
        the same PNG bytes as the most recent extraction's PNG.

    Useful as a smoke-test after every deploy: ALL unmodified tiles should
    be bit-identical (splice path); modified tiles should have a sane PSNR
    versus their pre-encode PNG.
    """
    prs = safe_data_path(filename)
    if not prs.exists():
        raise HTTPException(404, "missing file")
    # Force a fresh extract from the on-disk file
    manifest = extract_tiles(prs)
    tiles_dir = Path(manifest["tiles_dir"])

    out_tiles = []
    for tile in manifest["tiles"]:
        png = tiles_dir / tile["filename"]
        md5_sidecar = png.with_suffix(".src.md5")
        recorded_md5 = None
        if md5_sidecar.exists():
            try:
                recorded_md5 = md5_sidecar.read_text().strip().lower()
            except OSError:
                pass
        actual_md5 = hashlib.md5(png.read_bytes()).hexdigest()
        identical = (recorded_md5 is not None) and (recorded_md5 == actual_md5)
        out_tiles.append({
            "index": tile["index"],
            "filename": tile["filename"],
            "width": tile["width"],
            "height": tile["height"],
            "fmt": tile["fmt"],
            "recorded_md5": recorded_md5,
            "actual_md5": actual_md5,
            "identical_to_cache": identical,
        })
    return {
        "filename": filename,
        "tile_count": manifest["tile_count"],
        "is_prs": manifest.get("is_prs", False),
        "tiles": out_tiles,
        "all_identical": all(t["identical_to_cache"] for t in out_tiles),
    }


# ============================================================================
# Atlas mode (2026-04-25)
# ----------------------------------------------------------------------------
# Some PRS bundles hold tiles that are spatially tiled — the engine renders
# them edge-to-edge to form a composite splash / poster. atlas_layouts.py
# carries the per-file knowledge needed to:
#   1. Stitch the live tiles into a single editable composite.
#   2. Run the upscaler on that composite (full spatial context — letters
#      that cross tile seams stay continuous across the seam).
#   3. Slice the result back into per-tile crops at native dim and
#      register them in the cache as if the user had upscaled each tile
#      individually, so the existing repack pipeline picks them up.
# ============================================================================


def _build_composite_image(filename: str, manifest: dict, layout: dict) -> Image.Image:
    """Assemble the live source tiles into the composite described by `layout`.

    Returns a fresh RGBA Image of size (composite_w x composite_h). For the
    LogoEP4 layout, no resampling is done — placement_w == tile_native_w.
    For future layouts where placement size differs from tile native, we
    Lanczos-resize the tile crop to the placement rect.
    """
    tiles_dir = Path(manifest["tiles_dir"])
    by_idx = {t["index"]: t for t in manifest["tiles"]}

    cw = int(layout["composite_w"])
    ch = int(layout["composite_h"])
    out = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))

    for p in layout["placements"]:
        idx = p["tile_index"]
        tile = by_idx.get(idx)
        if not tile:
            raise HTTPException(
                400,
                f"atlas layout for {filename} references tile {idx}, but the "
                f"file has tiles {sorted(by_idx)}",
            )
        png = tiles_dir / tile["filename"]
        if not png.exists():
            raise HTTPException(500, f"tile png missing for {idx}: {png}")
        with Image.open(png) as im:
            im_rgba = im.convert("RGBA")
        tw, th = im_rgba.size
        # Apply uv_box (sub-tile crop) before resize. For LogoEP4 the box
        # is full (0,0,1,1) so this is a no-op.
        u0, v0, u1, v1 = p["uv_box"]
        if (u0, v0, u1, v1) != (0.0, 0.0, 1.0, 1.0):
            left = int(round(u0 * tw))
            upper = int(round(v0 * th))
            right = int(round(u1 * tw))
            lower = int(round(v1 * th))
            im_rgba = im_rgba.crop((left, upper, right, lower))
        # Resize to placement rect if needed (Lanczos).
        if im_rgba.size != (p["w"], p["h"]):
            im_rgba = im_rgba.resize((p["w"], p["h"]), Image.Resampling.LANCZOS)
        # Paste opaque (no alpha fade) — tiles already carry full alpha.
        out.paste(im_rgba, (p["x"], p["y"]))
    return out


def _slice_composite_to_tiles(
    composite: Image.Image,
    layout: dict,
    manifest: dict,
    cache_subdir: Path,
    label: str,
    *,
    keep_native_dims: bool,
) -> dict:
    """Crop `composite` according to `layout.placements`, save each crop as a
    per-tile PNG inside `cache_subdir` and return base64-encoded payloads.

    If keep_native_dims is True we Lanczos-down each crop to the tile's
    native dim — required for the game engine to load the rebuilt PRS.
    """
    by_idx = {t["index"]: t for t in manifest["tiles"]}
    cw, ch = composite.size

    expected_cw = int(layout["composite_w"])
    expected_ch = int(layout["composite_h"])
    if cw % expected_cw != 0 or ch % expected_ch != 0:
        raise HTTPException(
            400,
            f"composite {cw}x{ch} not an integer multiple of layout "
            f"{expected_cw}x{expected_ch}",
        )
    sx = cw // expected_cw
    sy = ch // expected_ch
    if sx != sy:
        raise HTTPException(
            400,
            f"composite has non-uniform scale ({sx}x vs {sy}x) — atlas mode "
            f"requires a uniform integer scale of layout dim",
        )
    scale = sx  # 1 = native, 2 = 2x upscaled, etc.

    cache_subdir.mkdir(parents=True, exist_ok=True)

    out_tiles = []
    for p in layout["placements"]:
        idx = p["tile_index"]
        tile = by_idx.get(idx)
        if not tile:
            continue
        nat_w = int(tile["width"])
        nat_h = int(tile["height"])
        # Crop region scaled up by `scale`.
        x0 = p["x"] * scale
        y0 = p["y"] * scale
        x1 = (p["x"] + p["w"]) * scale
        y1 = (p["y"] + p["h"]) * scale
        crop = composite.crop((x0, y0, x1, y1))
        # Lanczos to native dim if requested. For LogoEP4 the placement rect
        # at 1x equals native (1024x1024), so at 1x this is a no-op; at 4x
        # the crop is 4096x4096 and we Lanczos down to 1024x1024.
        if keep_native_dims and crop.size != (nat_w, nat_h):
            crop = crop.resize((nat_w, nat_h), Image.Resampling.LANCZOS)

        out_path = cache_subdir / f"tile{idx:02d}_{label}_native.png"
        crop.save(out_path)

        out_tiles.append({
            "tile_index": idx,
            "out_path": str(out_path),
            "out_w": crop.size[0],
            "out_h": crop.size[1],
            "out_b64": png_to_b64(out_path),
            "src_w": nat_w,
            "src_h": nat_h,
        })

    return {
        "tiles": out_tiles,
        "skipped": list(layout.get("skip_tiles", [])),
        "scale": scale,
    }


@app.get("/api/atlas_layouts")
def api_atlas_layouts():
    """List every filename that has a known atlas layout.

    Returns ``{"filenames": ["LogoEP4.prs", ...]}`` (always 200). The
    frontend fetches this once and only probes ``/api/atlas/<file>`` for
    members, so a normal texture open no longer triggers a 404 on the
    atlas-availability check.
    """
    return {"filenames": atlas_layouts.known_filenames()}


@app.get("/api/atlas/{filename}")
def api_atlas(filename: str):
    """Return the assembled composite image + layout metadata for a file.

    404 if the file has no known atlas layout. The composite is built from
    the live source tiles (just decompressed by extract_tiles) — it is NOT
    the upscaled / edited tiles.
    """
    layout = atlas_layouts.get_layout(filename)
    if not layout:
        raise HTTPException(404, f"no atlas layout known for {filename}")
    prs = safe_data_path(filename)
    if not prs.exists():
        raise HTTPException(404, f"no such file: {filename}")
    manifest = extract_tiles(prs)

    composite = _build_composite_image(filename, manifest, layout)
    buf = BytesIO()
    composite.save(buf, "PNG")
    composite_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "filename": filename,
        "kind": layout.get("kind", "screen_atlas"),
        "composite_w": int(layout["composite_w"]),
        "composite_h": int(layout["composite_h"]),
        "placements": [
            {
                "tile_index": p["tile_index"],
                "x": p["x"],
                "y": p["y"],
                "w": p["w"],
                "h": p["h"],
                "uv_box": list(p["uv_box"]),
            }
            for p in layout["placements"]
        ],
        "skip_tiles": list(layout.get("skip_tiles", [])),
        "source": layout.get("source", ""),
        "composite_b64": composite_b64,
    }


class AtlasUpscaleReq(BaseModel):
    filename: str
    model: str
    scale: int = 4
    keep_native_dims: bool = True
    tile_size: Optional[int] = None
    tta: bool = False
    gpu_id: Optional[int] = None


@app.post("/api/atlas_upscale")
def api_atlas_upscale(req: AtlasUpscaleReq):
    """Upscale the composite (full spatial context), then slice back to per-tile
    PNGs and store them in the cache as if the user had upscaled each tile.

    Returns one entry per placed tile with `out_b64`, dims, etc., suitable
    for direct insertion into the frontend's `state.tileEdits`.
    """
    layout = atlas_layouts.get_layout(req.filename)
    if not layout:
        raise HTTPException(404, f"no atlas layout known for {req.filename}")
    prs = safe_data_path(req.filename)
    if not prs.exists():
        raise HTTPException(404, f"no such file: {req.filename}")
    if req.scale not in ALLOWED_SCALES:
        raise HTTPException(
            400, f"scale must be one of {ALLOWED_SCALES} (got {req.scale})"
        )
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", req.model):
        raise HTTPException(400, "invalid model name")
    bin_path = REALESRGAN_MODELS / f"{req.model}.bin"
    param_path = REALESRGAN_MODELS / f"{req.model}.param"
    if not (bin_path.exists() and param_path.exists()):
        raise HTTPException(400, f"model not found: {req.model}")
    if req.tile_size is not None and req.tile_size not in ALLOWED_TILE_SIZES:
        raise HTTPException(
            400, f"tile_size must be one of {ALLOWED_TILE_SIZES} (got {req.tile_size})"
        )
    if req.gpu_id is not None and not (-1 <= req.gpu_id <= 7):
        raise HTTPException(400, "gpu_id must be -1..7 or null")

    manifest = extract_tiles(prs)

    # Build the composite from the live source tiles
    composite = _build_composite_image(req.filename, manifest, layout)
    cache_subdir = Path(manifest["cache_dir"]) / "upscaled"
    cache_subdir.mkdir(exist_ok=True)

    settings_tag = (
        f"_t{req.tile_size if req.tile_size is not None else 'A'}"
        f"_tta{1 if req.tta else 0}"
        f"_g{req.gpu_id if req.gpu_id is not None else 'A'}"
    )
    base_name = f"atlas_{req.model}_x{req.scale}{settings_tag}"

    # Lock around the entire atlas job: it can take many minutes on a large
    # composite, and concurrent calls would step on the cache.
    lock_key = f"{req.filename}|atlas|{req.model}|{req.scale}|{settings_tag}"
    lk = _get_lock(_UPSCALE_LOCKS, lock_key, MAX_UPSCALE_LOCKS)
    with lk:
        comp_src_path = cache_subdir / f"{base_name}_src.png"
        comp_buf = BytesIO()
        composite.save(comp_buf, "PNG")
        comp_src_path.write_bytes(comp_buf.getvalue())

        # Run the cascade upscaler on the composite as a whole. The cascade
        # function takes care of native-scale-respecting model invocation
        # plus any final Lanczos to hit the requested ratio.
        casc_out = _cascade_upscale(
            comp_src_path,
            cache_subdir,
            base_name,
            req.model,
            req.scale,
            tile_size=req.tile_size,
            tta=req.tta,
            gpu_id=req.gpu_id,
        )

        with Image.open(casc_out) as up_im:
            up = up_im.convert("RGBA").copy()

    # Slice the upscaled composite back to per-tile crops.
    sliced = _slice_composite_to_tiles(
        up,
        layout,
        manifest,
        cache_subdir,
        label=f"{base_name}_slice",
        keep_native_dims=req.keep_native_dims,
    )

    return {
        "filename": req.filename,
        "model": req.model,
        "scale": req.scale,
        "keep_native_dims": req.keep_native_dims,
        "tile_size": req.tile_size,
        "tta": req.tta,
        "gpu_id": req.gpu_id,
        "composite_w": int(layout["composite_w"]),
        "composite_h": int(layout["composite_h"]),
        "upscaled_w": up.size[0],
        "upscaled_h": up.size[1],
        "tiles": sliced["tiles"],
        "skip_tiles": sliced["skipped"],
    }


class AtlasImportReq(BaseModel):
    filename: str
    png_b64: str
    keep_native_dims: bool = True


@app.post("/api/atlas_import")
def api_atlas_import(req: AtlasImportReq):
    """Accept a user-supplied composite PNG (e.g. from an external Upscayl
    run on the composite the editor served via /api/atlas/{filename}).

    The PNG must match composite_w x composite_h, OR be an exact integer
    multiple of it (in which case we treat it as upscaled and the slice
    code Lanczos-downs each crop). Slicing is identical to atlas_upscale.
    """
    layout = atlas_layouts.get_layout(req.filename)
    if not layout:
        raise HTTPException(404, f"no atlas layout known for {req.filename}")
    prs = safe_data_path(req.filename)
    if not prs.exists():
        raise HTTPException(404, f"no such file: {req.filename}")

    # Decode the b64 payload
    b = req.png_b64
    if "," in b:
        b = b.split(",", 1)[1]
    try:
        raw = base64.b64decode(b)
    except Exception:
        raise HTTPException(400, "png_b64 decode failed")
    if len(raw) > 256 * 1024 * 1024:
        raise HTTPException(413, f"composite too large: {len(raw)} bytes (cap 256 MB)")
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise HTTPException(400, "not a PNG (magic bytes missing)")
    try:
        im = Image.open(BytesIO(raw)).convert("RGBA")
        im.load()
    except Exception as e:
        raise HTTPException(400, f"PNG decode failed: {e}")

    cw = int(layout["composite_w"])
    ch = int(layout["composite_h"])
    iw, ih = im.size
    if (iw, ih) == (cw, ch):
        scale = 1
    else:
        if iw % cw != 0 or ih % ch != 0:
            raise HTTPException(
                400,
                f"composite dim {iw}x{ih} is not an integer multiple of "
                f"layout {cw}x{ch}. Acceptable sizes: {cw}x{ch}, "
                f"{cw*2}x{ch*2}, {cw*4}x{ch*4}.",
            )
        kx = iw // cw
        ky = ih // ch
        if kx != ky:
            raise HTTPException(
                400,
                f"composite has non-uniform scale ({kx}x vs {ky}x) — atlas "
                f"mode requires a uniform integer scale of layout dim",
            )
        scale = kx

    manifest = extract_tiles(prs)
    cache_subdir = Path(manifest["cache_dir"]) / "upscaled"
    cache_subdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    label = f"atlas_imported_x{scale}_{ts}"

    sliced = _slice_composite_to_tiles(
        im,
        layout,
        manifest,
        cache_subdir,
        label=label,
        keep_native_dims=req.keep_native_dims,
    )

    return {
        "filename": req.filename,
        "imported_w": iw,
        "imported_h": ih,
        "scale_factor": scale,
        "composite_w": cw,
        "composite_h": ch,
        "keep_native_dims": req.keep_native_dims,
        "tiles": sliced["tiles"],
        "skip_tiles": sliced["skipped"],
    }


# ============================================================================
# Viewport mode (2026-04-25) — "16:9 transform"
# ----------------------------------------------------------------------------
# The game's logical render canvas is 1278x768 (per the layout research). 4:3
# splash assets (LogoEP4, NowLoading, etc.) are designed for the centered
# 1024x768 region; the widescreen ASI fills the 256-px right-edge pillar
# (and any letterbox top/bot rows) with white tiles. HUD/menu textures
# render across the full 16:9 canvas.
#
# This mode lets the user:
#   1. See the source tile(s) laid out as the game would render them on
#      a 16:9 screen, with pillar/letterbox indicators.
#   2. Paint freely across the full 16:9 canvas (including pillar regions).
#   3. Send the painted PNG back to the server, which slices each placement
#      out to native tile dim and registers it in the cache exactly like
#      atlas_import / import_png — so the existing repack flow picks it up.
#
# Endpoints:
#   GET  /api/viewport/{filename}   layout + composite PNG at 1278x768
#   POST /api/viewport_paint        accept painted PNG, slice -> per-tile
#
# For files in atlas_layouts.py the placements come from the ground-truth
# table (scaled to fit the 4:3 inset). For files we have no layout for,
# we emit a single centered placement showing the largest tile or tile 0.
# ============================================================================


def _viewport_inset_43() -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of the 4:3 content inset inside the 16:9 canvas.

    The 4:3 native is 1024x768 inside a 1278x768 canvas; we center it,
    leaving 127px pillars on each side (the right pillar is the one the
    widescreen ASI fills with white tiles in-engine).
    """
    inset_w = VIEWPORT_43_W
    inset_h = VIEWPORT_43_H
    x = (VIEWPORT_W - inset_w) // 2
    y = (VIEWPORT_H - inset_h) // 2
    return x, y, inset_w, inset_h


def _viewport_layout_for_atlas(filename: str, layout: dict, manifest: dict) -> dict:
    """Map atlas_layouts.py placements onto the 1278x768 viewport.

    The atlas layout's composite is some authoring-resolution canvas (e.g.
    2048x2048 for LogoEP4). We scale that into the 4:3 inset so each
    placement lands at the screen rect the game samples.
    """
    cw = int(layout["composite_w"])
    ch = int(layout["composite_h"])
    inset_x, inset_y, inset_w, inset_h = _viewport_inset_43()
    sx = inset_w / float(cw)
    sy = inset_h / float(ch)
    placements = []
    for p in layout["placements"]:
        dx = inset_x + int(round(p["x"] * sx))
        dy = inset_y + int(round(p["y"] * sy))
        dw = int(round(p["w"] * sx))
        dh = int(round(p["h"] * sy))
        placements.append({
            "tile_index": int(p["tile_index"]),
            "dest_x": dx,
            "dest_y": dy,
            "dest_w": dw,
            "dest_h": dh,
            "uv_box": [float(v) for v in p["uv_box"]],
        })
    return {
        "viewport_w": VIEWPORT_W,
        "viewport_h": VIEWPORT_H,
        "layout": "atlas",
        "placements": placements,
        "skip_tiles": [int(i) for i in layout.get("skip_tiles", [])],
        "source": "atlas_layouts.py",
        "atlas_source": layout.get("source", ""),
        "inset": {"x": inset_x, "y": inset_y, "w": inset_w, "h": inset_h},
    }


def _viewport_layout_centered(filename: str, manifest: dict) -> dict:
    """Default placement for files with no atlas_layouts.py entry: pick a
    representative tile and center it inside the 16:9 canvas, preserving
    aspect ratio. The pillar/letterbox regions remain transparent.
    """
    tiles = manifest.get("tiles", [])
    if not tiles:
        # No tiles at all — return an empty layout. The frontend will
        # display the empty 16:9 canvas with letterbox indicators only.
        return {
            "viewport_w": VIEWPORT_W,
            "viewport_h": VIEWPORT_H,
            "layout": "centered",
            "placements": [],
            "skip_tiles": [],
            "source": "guessed",
            "inset": dict(zip(("x", "y", "w", "h"), _viewport_inset_43())),
        }
    # Pick the largest tile (by area) — this is the most likely "main"
    # content for splash / HUD assets that bundle a primary + auxiliary tiles.
    primary = max(tiles, key=lambda t: int(t.get("width", 0)) * int(t.get("height", 0)))
    nat_w = int(primary["width"])
    nat_h = int(primary["height"])
    # Fit the tile inside the 4:3 inset, preserving aspect ratio.
    inset_x, inset_y, inset_w, inset_h = _viewport_inset_43()
    if nat_w <= 0 or nat_h <= 0:
        scale = 1.0
    else:
        scale = min(inset_w / float(nat_w), inset_h / float(nat_h))
    dw = max(1, int(round(nat_w * scale)))
    dh = max(1, int(round(nat_h * scale)))
    dx = inset_x + (inset_w - dw) // 2
    dy = inset_y + (inset_h - dh) // 2
    placements = [{
        "tile_index": int(primary["index"]),
        "dest_x": dx,
        "dest_y": dy,
        "dest_w": dw,
        "dest_h": dh,
        "uv_box": [0.0, 0.0, 1.0, 1.0],
    }]
    skip = [int(t["index"]) for t in tiles if int(t["index"]) != int(primary["index"])]
    return {
        "viewport_w": VIEWPORT_W,
        "viewport_h": VIEWPORT_H,
        "layout": "centered",
        "placements": placements,
        "skip_tiles": skip,
        "source": "guessed",
        "inset": {"x": inset_x, "y": inset_y, "w": inset_w, "h": inset_h},
    }


def _build_viewport_composite(
    filename: str,
    manifest: dict,
    placements: list,
) -> Image.Image:
    """Assemble live source tiles into the 1278x768 viewport canvas.

    The pillar/letterbox regions remain transparent — the frontend draws
    the dotted overlay there. Tiles are Lanczos-resampled to their dest_w
    x dest_h placement rect.
    """
    tiles_dir = Path(manifest["tiles_dir"])
    by_idx = {int(t["index"]): t for t in manifest["tiles"]}
    out = Image.new("RGBA", (VIEWPORT_W, VIEWPORT_H), (0, 0, 0, 0))
    for p in placements:
        idx = int(p["tile_index"])
        tile = by_idx.get(idx)
        if not tile:
            # Stale placement — skip silently rather than failing the request.
            continue
        png = tiles_dir / tile["filename"]
        if not png.exists():
            continue
        with Image.open(png) as im:
            im_rgba = im.convert("RGBA")
        # Apply uv_box (sub-tile crop). Default (0,0,1,1) -> no-op.
        u0, v0, u1, v1 = p["uv_box"]
        if (u0, v0, u1, v1) != (0.0, 0.0, 1.0, 1.0):
            tw, th = im_rgba.size
            im_rgba = im_rgba.crop((
                int(round(u0 * tw)),
                int(round(v0 * th)),
                int(round(u1 * tw)),
                int(round(v1 * th)),
            ))
        dw = int(p["dest_w"])
        dh = int(p["dest_h"])
        if dw <= 0 or dh <= 0:
            continue
        if im_rgba.size != (dw, dh):
            im_rgba = im_rgba.resize((dw, dh), Image.Resampling.LANCZOS)
        out.paste(im_rgba, (int(p["dest_x"]), int(p["dest_y"])))
    return out


@app.get("/api/viewport/{filename}")
def api_viewport(filename: str):
    """Return viewport-laid-out composite + placements for a file.

    For files in atlas_layouts.py the placements are scaled into the 4:3
    inset of the 1278x768 canvas. For unknown files we return a centered
    placement showing the largest tile.

    The frontend (static/viewport.js) renders the composite_b64 on a
    1278x768 HTML canvas, lets the user paint anywhere (including the
    pillar regions), then POSTs the painted PNG to /api/viewport_paint.
    """
    prs = safe_data_path(filename)
    if not prs.exists():
        raise HTTPException(404, f"no such file: {filename}")
    manifest = extract_tiles(prs)

    layout = atlas_layouts.get_layout(filename)
    if layout:
        vp = _viewport_layout_for_atlas(filename, layout, manifest)
    else:
        vp = _viewport_layout_centered(filename, manifest)

    # Sanity: every referenced tile must exist in the manifest.
    valid_idx = {int(t["index"]) for t in manifest.get("tiles", [])}
    for p in vp["placements"]:
        if int(p["tile_index"]) not in valid_idx:
            raise HTTPException(
                500,
                f"viewport layout for {filename} references tile "
                f"{p['tile_index']} but file has tiles {sorted(valid_idx)}",
            )

    composite = _build_viewport_composite(filename, manifest, vp["placements"])
    buf = BytesIO()
    composite.save(buf, "PNG")
    composite_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "filename": filename,
        "viewport_w": vp["viewport_w"],
        "viewport_h": vp["viewport_h"],
        "layout": vp["layout"],
        "placements": vp["placements"],
        "skip_tiles": vp["skip_tiles"],
        "inset": vp["inset"],
        "source": vp["source"],
        "atlas_source": vp.get("atlas_source", ""),
        "composite_b64": composite_b64,
        "tile_count": manifest["tile_count"],
    }


class ViewportPaintReq(BaseModel):
    filename: str
    viewport_png_b64: str
    viewport_w: int = VIEWPORT_W
    viewport_h: int = VIEWPORT_H


@app.post("/api/viewport_paint")
def api_viewport_paint(req: ViewportPaintReq, request: Request):
    """Slice a painted 16:9 viewport PNG back into per-tile edits.

    The PNG must match the canonical 1278x768 viewport size (no upscaling
    accepted on this path — the canvas is for hand-painting at 1:1, not for
    post-process upscaling). Each placement region is cropped, Lanczos-
    downed to native tile dim, saved into the per-file cache `upscaled/`
    subdir, and returned base64 so the frontend registers it in
    state.tileEdits exactly like atlas_import.

    Skipped tiles (pillar fills, etc.) round-trip untouched via the
    existing repack splice — the user paints OVER the pillar in this view
    but pillar tiles aren't part of the placement list, so they retain
    their original bytes.
    """
    _enforce_body_size(request, MAX_VIEWPORT_PAINT_BODY)
    prs = safe_data_path(req.filename)
    if not prs.exists():
        raise HTTPException(404, f"no such file: {req.filename}")
    if req.viewport_w != VIEWPORT_W or req.viewport_h != VIEWPORT_H:
        raise HTTPException(
            400,
            f"viewport must be {VIEWPORT_W}x{VIEWPORT_H} (got "
            f"{req.viewport_w}x{req.viewport_h})",
        )

    # Decode b64 payload
    b = req.viewport_png_b64
    if "," in b:
        b = b.split(",", 1)[1]
    try:
        raw = base64.b64decode(b)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, f"viewport_png_b64 decode failed: {e}")
    if len(raw) > MAX_VIEWPORT_PAINT_BODY:
        raise HTTPException(
            413,
            f"viewport PNG too large: {len(raw)} bytes (cap {MAX_VIEWPORT_PAINT_BODY})",
        )
    if raw[:8] != PNG_MAGIC:
        raise HTTPException(400, "not a PNG (magic bytes missing)")
    try:
        im = Image.open(BytesIO(raw))
        im.load()
        im = im.convert("RGBA")
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"PNG decode failed: {e}")
    if im.size != (VIEWPORT_W, VIEWPORT_H):
        raise HTTPException(
            400,
            f"painted PNG dim {im.size[0]}x{im.size[1]} != viewport "
            f"{VIEWPORT_W}x{VIEWPORT_H}. Resize externally before sending.",
        )

    manifest = extract_tiles(prs)
    layout = atlas_layouts.get_layout(req.filename)
    if layout:
        vp = _viewport_layout_for_atlas(req.filename, layout, manifest)
    else:
        vp = _viewport_layout_centered(req.filename, manifest)

    by_idx = {int(t["index"]): t for t in manifest["tiles"]}
    cache_subdir = Path(manifest["cache_dir"]) / "upscaled"
    cache_subdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    label = f"viewport_paint_{ts}"

    out_tiles = []
    for p in vp["placements"]:
        idx = int(p["tile_index"])
        tile = by_idx.get(idx)
        if not tile:
            continue
        nat_w = int(tile["width"])
        nat_h = int(tile["height"])
        # Crop the painted region.
        x0 = max(0, int(p["dest_x"]))
        y0 = max(0, int(p["dest_y"]))
        x1 = min(VIEWPORT_W, int(p["dest_x"] + p["dest_w"]))
        y1 = min(VIEWPORT_H, int(p["dest_y"] + p["dest_h"]))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = im.crop((x0, y0, x1, y1))
        # Lanczos to native tile dim — required for the game engine to load
        # the rebuilt PRS at its expected pixel ratios.
        if crop.size != (nat_w, nat_h):
            crop = crop.resize((nat_w, nat_h), Image.Resampling.LANCZOS)
        out_path = cache_subdir / f"tile{idx:02d}_{label}_native.png"
        crop.save(out_path)
        out_tiles.append({
            "tile_index": idx,
            "out_path": str(out_path),
            "out_w": crop.size[0],
            "out_h": crop.size[1],
            "out_b64": png_to_b64(out_path),
            "src_w": nat_w,
            "src_h": nat_h,
        })

    return {
        "filename": req.filename,
        "tile_count": manifest["tile_count"],
        "tiles_modified": [t["tile_index"] for t in out_tiles],
        "skipped": list(vp["skip_tiles"]),
        "tiles": out_tiles,
        "viewport_w": VIEWPORT_W,
        "viewport_h": VIEWPORT_H,
        "layout": vp["layout"],
    }


@app.get("/api/models")
def api_models():
    """List ncnn-vulkan models that have both .bin and .param.

    Each entry: {name, default_scale, native_scale, max_scale, supports_tta, description}
    `default_scale` is preserved for backwards compatibility (was the old field).
    """
    models = []
    if REALESRGAN_MODELS.exists():
        bins = {p.stem for p in REALESRGAN_MODELS.glob("*.bin")}
        params = {p.stem for p in REALESRGAN_MODELS.glob("*.param")}
        valid = sorted(bins & params)
        for name in valid:
            meta = model_meta(name)
            models.append({
                "name": name,
                "default_scale": meta["native_scale"],  # legacy alias
                "native_scale": meta["native_scale"],
                "max_scale": meta["max_scale"],
                "supports_tta": meta["supports_tta"],
                "description": meta["description"],
            })
    return {
        "models": models,
        "allowed_scales": list(ALLOWED_SCALES),
        "allowed_tile_sizes": list(ALLOWED_TILE_SIZES),
    }


# ============================================================================
# Deploy to live game install (2026-04-24)
# ----------------------------------------------------------------------------
# The editor operates on a DEV mirror so the user can keep playing PSOBB
# while iterating. When the user wants to promote a finished file to the
# live install, /api/deploy/promote copies named files dev -> live with a
# timestamped backup of the previous live bytes.
#
# Safety rails:
#   - LIVE_DATA_DIR is the user's playable install. We NEVER bulk-overwrite.
#   - Promote takes an explicit list of filenames; no globs, no "all".
#   - Each filename goes through safe_*_path so traversal is rejected.
#   - Each overwrite makes a `.pre_promote_<TS>` backup in LIVE_DATA_DIR
#     (the original game files that came from the user's stock install).
#   - One promote in flight at a time (via _PROMOTE_LOCK).
# ============================================================================


def safe_live_path(name: str) -> Path:
    """Resolve `name` strictly inside LIVE_DATA_DIR. Reject path traversal.

    Mirrors safe_data_path, but for the live game install dir. We keep them
    as separate functions (rather than parameterising) so misuse at the
    call site is loud.
    """
    bare = _validate_bare_filename(name, label="filename")
    p = (LIVE_DATA_DIR / bare).resolve()
    try:
        p.relative_to(LIVE_DATA_DIR)
    except ValueError:
        raise HTTPException(400, "path escapes live data dir")
    return p


def _file_md5(p: Path) -> str:
    """Compute md5 of file bytes. Used by /api/deploy/diff."""
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_deployable_files(d: Path) -> dict[str, Path]:
    """List files in directory `d` that are deploy candidates.

    Skips directories, backups (.pre_*, .suspect_*, etc.), and dotfiles.
    Returns {basename: path}.
    """
    out: dict[str, Path] = {}
    if not d.exists():
        return out
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if _is_backup_name(p.name):
            continue
        out[p.name] = p
    return out


@app.get("/api/deploy/config")
def api_deploy_config():
    """Return the active dev/live data directories.

    The frontend uses this to render source-of-truth labels in the deploy
    modal so the user can confirm where files are flowing.
    """
    return {
        "dev_dir": str(DEV_DATA_DIR),
        "live_dir": str(LIVE_DATA_DIR),
        "active_data_dir": str(DATA_DIR),
        "dev_exists": DEV_DATA_DIR.exists(),
        "live_exists": LIVE_DATA_DIR.exists(),
    }


@app.get("/api/deploy/diff")
def api_deploy_diff():
    """Compute a diff between DEV_DATA_DIR and LIVE_DATA_DIR.

    Returns three buckets:
      - changed:    files present in both with different bytes (md5)
      - dev_only:   files present in dev but missing from live (NEW assets)
      - live_only:  files present in live but missing from dev (purely
                    informational - promote can't push these back, they
                    represent stock files we never imported into dev)

    md5 + size are reported per entry so the frontend can show "X KB
    bigger / smaller" deltas if desired.
    """
    if not DEV_DATA_DIR.exists():
        raise HTTPException(500, f"dev data dir missing: {DEV_DATA_DIR}")
    if not LIVE_DATA_DIR.exists():
        raise HTTPException(500, f"live data dir missing: {LIVE_DATA_DIR}")

    dev_files = _list_deployable_files(DEV_DATA_DIR)
    live_files = _list_deployable_files(LIVE_DATA_DIR)

    dev_names = set(dev_files)
    live_names = set(live_files)

    changed = []
    dev_only = []
    live_only = []

    # Files present in both: compare md5
    for name in sorted(dev_names & live_names):
        dp = dev_files[name]
        lp = live_files[name]
        try:
            d_size = dp.stat().st_size
            l_size = lp.stat().st_size
        except OSError as e:
            log.debug("stat failed during diff: %s", e)
            continue
        # Cheap pre-check: if sizes differ, files are definitely different.
        # md5-equal-but-size-differ is impossible.
        if d_size != l_size:
            try:
                changed.append({
                    "name": name,
                    "dev_size": d_size,
                    "live_size": l_size,
                    "dev_md5": _file_md5(dp),
                    "live_md5": _file_md5(lp),
                })
            except OSError as e:
                log.warning("md5 read failed for %s: %s", name, e)
            continue
        # Same size; hash to confirm.
        try:
            d_md5 = _file_md5(dp)
            l_md5 = _file_md5(lp)
        except OSError as e:
            log.warning("md5 read failed for %s: %s", name, e)
            continue
        if d_md5 != l_md5:
            changed.append({
                "name": name,
                "dev_size": d_size,
                "live_size": l_size,
                "dev_md5": d_md5,
                "live_md5": l_md5,
            })

    for name in sorted(dev_names - live_names):
        try:
            sz = dev_files[name].stat().st_size
        except OSError:
            sz = 0
        dev_only.append({"name": name, "dev_size": sz})

    for name in sorted(live_names - dev_names):
        try:
            sz = live_files[name].stat().st_size
        except OSError:
            sz = 0
        live_only.append({"name": name, "live_size": sz})

    return {
        "dev_dir": str(DEV_DATA_DIR),
        "live_dir": str(LIVE_DATA_DIR),
        "changed": changed,
        "dev_only": dev_only,
        "live_only": live_only,
        "summary": {
            "changed_count": len(changed),
            "dev_only_count": len(dev_only),
            "live_only_count": len(live_only),
        },
    }


class PromoteReq(BaseModel):
    files: list[str]
    create_backup: bool = True


@app.post("/api/deploy/promote")
def api_deploy_promote(req: PromoteReq, request: Request):
    """Copy named files from DEV_DATA_DIR to LIVE_DATA_DIR.

    For each file, when the live target already exists and ``create_backup``
    is True, write a `.pre_promote_<YYYYMMDD_HHMMSS>` copy first inside
    LIVE_DATA_DIR. Skip the copy on per-file errors and report them.

    One promote in flight at a time (global ``_PROMOTE_LOCK``); concurrent
    callers get an HTTP 409.
    """
    _enforce_body_size(request, MAX_PROMOTE_BODY)

    if not isinstance(req.files, list) or len(req.files) == 0:
        raise HTTPException(400, "files: empty list")
    if len(req.files) > MAX_PROMOTE_FILES:
        raise HTTPException(
            400, f"too many files in one promote ({len(req.files)} > {MAX_PROMOTE_FILES})"
        )
    if not DEV_DATA_DIR.exists():
        raise HTTPException(500, f"dev data dir missing: {DEV_DATA_DIR}")
    if not LIVE_DATA_DIR.exists():
        raise HTTPException(500, f"live data dir missing: {LIVE_DATA_DIR}")

    # Validate each name UP-FRONT so we don't half-promote on the first
    # bad input. Both dev (where we read from) and live (where we write to)
    # must accept the name.
    resolved: list[tuple[str, Path, Path]] = []
    for name in req.files:
        # Live-side guard: reject path traversal early.
        live_p = safe_live_path(name)
        # Dev-side resolution mirrors safe_data_path's logic but against
        # DEV_DATA_DIR explicitly, so this works even if DATA_DIR has been
        # pointed elsewhere via PSO_DATA_DIR env var.
        bare = _validate_bare_filename(name, label="filename")
        dev_p = (DEV_DATA_DIR / bare).resolve()
        try:
            dev_p.relative_to(DEV_DATA_DIR)
        except ValueError:
            raise HTTPException(400, "path escapes dev data dir")
        resolved.append((name, dev_p, live_p))

    if not _PROMOTE_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another promote is already running")
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        results: list[dict] = []
        ok_count = 0
        fail_count = 0
        for name, dev_p, live_p in resolved:
            entry: dict = {"name": name, "ok": False}
            if not dev_p.exists():
                entry["error"] = "dev file missing"
                results.append(entry)
                fail_count += 1
                continue

            backup_name: Optional[str] = None
            try:
                if req.create_backup and live_p.exists():
                    bak_name = f"{name}.pre_promote_{ts}"
                    bak = LIVE_DATA_DIR / bak_name
                    counter = 0
                    # Avoid clobber if the user calls promote twice in the same second
                    while bak.exists():
                        counter += 1
                        bak = LIVE_DATA_DIR / f"{bak_name}_{counter}"
                    shutil.copy2(live_p, bak)
                    backup_name = bak.name
                # Copy dev -> live
                shutil.copy2(dev_p, live_p)
            except (OSError, shutil.Error) as e:
                log.warning("promote failed for %s: %s", name, e)
                entry["error"] = f"copy failed: {e}"
                if backup_name:
                    entry["backup_name"] = backup_name
                results.append(entry)
                fail_count += 1
                continue

            try:
                final_size = live_p.stat().st_size
            except OSError:
                final_size = None
            entry.update({
                "ok": True,
                "backup_name": backup_name,
                "live_size": final_size,
            })
            results.append(entry)
            ok_count += 1

        log.info(
            "promote: ok=%d fail=%d (files=%s)",
            ok_count, fail_count, [r["name"] for r in results],
        )
        return {
            "ok_count": ok_count,
            "fail_count": fail_count,
            "results": results,
            "dev_dir": str(DEV_DATA_DIR),
            "live_dir": str(LIVE_DATA_DIR),
        }
    finally:
        _PROMOTE_LOCK.release()


# ============================================================================
# Archive build endpoints (2026-04-25): /api/build_afs, /api/build_bml,
# /api/deploy/<archive>
# ----------------------------------------------------------------------------
# These let the frontend request a fresh AFS or BML built from edited
# entries. The build artifact lands in a per-format export cache and the
# response carries an md5 + size for the caller to sanity-check before
# deploying. Deploy is a separate, explicit step (so a user can preview /
# diff the rebuilt archive before clobbering the live game install).
#
# Body schemas:
#   POST /api/build_afs
#     {
#       "name": "ItemModel.afs",         # output filename (no path)
#       "entries": [
#         {"name": "0001", "b64": "<base64 raw bytes>"} OR
#         {"name": "0001", "path": "DATA_DIR-relative filename"}
#       ],
#       "names_in_archive": false,        # optional; emit name table
#       "first_entry_offset": null        # optional; default 0x80000
#     }
#     -> { ok, path, size, md5 }
#
#   POST /api/build_bml
#     {
#       "name": "biri_ball.bml",
#       "compression": 80,                # default 0x50 (PRS)
#       "file_alignment": null,           # default auto-classify
#       "has_textures": null,             # default auto from entries
#       "entries": [
#         {
#           "name": "biri_ball.nj",
#           "data_b64": "<base64>",       # raw uncompressed inner
#           "is_compressed": false,
#           "decompressed_size": 0,        # required if is_compressed
#           "texture_b64": null,           # optional
#           "texture_is_compressed": false,
#           "texture_decompressed_size": 0,
#           "unk_a": 0, "unk_b": 0,
#           "unk_c": 0, "unk_d": 0
#         }
#       ]
#     }
#     -> { ok, path, size, md5 }
#
#   POST /api/deploy/<archive>
#     Body: empty or { "create_backup": true }
#     Deploys cache/{afs|bml}_export/<archive> -> LIVE_DATA_DIR/<archive>
#     with a timestamped backup of the prior live bytes.
# ============================================================================

AFS_EXPORT_DIR = CACHE_DIR / "afs_export"
BML_EXPORT_DIR = CACHE_DIR / "bml_export"
NJ_EXPORT_DIR = CACHE_DIR / "nj_export"
NJM_EXPORT_DIR = CACHE_DIR / "njm_export"
AFS_EXPORT_DIR.mkdir(exist_ok=True)
BML_EXPORT_DIR.mkdir(exist_ok=True)
NJ_EXPORT_DIR.mkdir(exist_ok=True)
NJM_EXPORT_DIR.mkdir(exist_ok=True)

# Per-build body cap. AFS archives can be large (ItemKT*.afs is ~100 MB),
# but a single base64-encoded build call is unlikely to exceed this.
MAX_BUILD_AFS_BODY = 256 * 1024 * 1024  # 256 MB
MAX_BUILD_BML_BODY = 128 * 1024 * 1024  # 128 MB
MAX_BUILD_DEPLOY_BODY = 4 * 1024


def _decode_b64(s: str, *, ctx: str) -> bytes:
    """Decode a base64 field; raise HTTPException 400 with context on failure."""
    if not isinstance(s, str):
        raise HTTPException(400, f"{ctx}: expected base64 string")
    try:
        return base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(400, f"{ctx}: invalid base64 ({e})")


def _safe_archive_name(name: str) -> str:
    """Validate an output filename: bare basename, no traversal."""
    bare = _validate_bare_filename(name, label="name")
    if not (1 <= len(bare) <= 255):
        raise HTTPException(400, "name: length out of range")
    return bare


def _md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()


class BuildAfsEntry(BaseModel):
    name: Optional[str] = None
    # Exactly one of `b64` or `path` (DATA_DIR-relative) must be provided.
    b64: Optional[str] = None
    path: Optional[str] = None


class BuildAfsReq(BaseModel):
    name: str
    entries: list[BuildAfsEntry]
    names_in_archive: bool = False
    first_entry_offset: Optional[int] = None


@app.post("/api/build_afs")
def api_build_afs(req: BuildAfsReq, request: Request):
    """Build an AFS archive from raw entry bytes (or DATA_DIR file refs).

    Writes the output to ``cache/afs_export/<name>``. Returns size + md5
    so the caller can confirm before deploying. Does NOT touch
    DATA_DIR/LIVE_DATA_DIR — call /api/deploy/<archive> for that.
    """
    _enforce_body_size(request, MAX_BUILD_AFS_BODY)
    name = _safe_archive_name(req.name)
    if not isinstance(req.entries, list) or len(req.entries) == 0:
        raise HTTPException(400, "entries: empty list")
    if len(req.entries) > 0xFFFF:
        raise HTTPException(400, "entries: too many (>65535)")

    # Build per-entry blobs.
    blobs: list[bytes] = []
    names: list[str] = []
    for i, ent in enumerate(req.entries):
        if (ent.b64 is None) == (ent.path is None):
            raise HTTPException(
                400, f"entries[{i}]: must provide exactly one of b64/path"
            )
        if ent.b64 is not None:
            blob = _decode_b64(ent.b64, ctx=f"entries[{i}].b64")
        else:
            ref = safe_data_path(ent.path)  # validates traversal + DATA_DIR
            if not ref.exists():
                raise HTTPException(
                    404, f"entries[{i}].path: not found in DATA_DIR: {ent.path}"
                )
            blob = ref.read_bytes()
        blobs.append(blob)
        names.append(ent.name if isinstance(ent.name, str) else f"entry_{i:04d}")

    try:
        out = write_afs(
            blobs,
            names=names if req.names_in_archive else None,
            first_entry_offset=req.first_entry_offset,
        )
    except ValueError as e:
        raise HTTPException(400, f"write_afs failed: {e}")

    # Write to export cache atomically (write -> rename).
    out_path = AFS_EXPORT_DIR / name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(out)
    os.replace(tmp_path, out_path)

    md5 = _md5_bytes(out)
    log.info("build_afs %s -> %s (%d bytes, md5=%s)", name, out_path, len(out), md5)
    return {
        "ok": True,
        "path": str(out_path),
        "size": len(out),
        "md5": md5,
        "entry_count": len(blobs),
    }


class BuildBmlEntry(BaseModel):
    name: str
    data_b64: Optional[str] = None
    is_compressed: bool = False
    decompressed_size: int = 0
    texture_b64: Optional[str] = None
    texture_is_compressed: bool = False
    texture_decompressed_size: int = 0
    unk_a: int = 0
    unk_b: int = 0
    unk_c: int = 0
    unk_d: int = 0


class BuildBmlReq(BaseModel):
    name: str
    entries: list[BuildBmlEntry]
    compression: int = 0x50
    file_alignment: Optional[int] = None
    has_textures: Optional[bool] = None
    optimal: bool = False


@app.post("/api/build_bml")
def api_build_bml(req: BuildBmlReq, request: Request):
    """Build a BML container from a list of entries.

    Each entry's ``data_b64`` is interpreted per ``is_compressed``:
      - False (default): raw uncompressed bytes; the packer will
        PRS-encode using ``formats.prs.compress`` (or
        ``compress_optimal`` if ``optimal=True``).
      - True: bytes are already PRS-encoded; ``decompressed_size`` is
        required and stored verbatim.

    Same for ``texture_b64`` / ``texture_is_compressed``.

    Output lands in ``cache/bml_export/<name>``. Caller can preview via
    /api/raw/cache/... etc. and deploy via /api/deploy/<name>.
    """
    _enforce_body_size(request, MAX_BUILD_BML_BODY)
    name = _safe_archive_name(req.name)
    if not isinstance(req.entries, list) or len(req.entries) == 0:
        raise HTTPException(400, "entries: empty list")
    if req.compression not in (BML_COMPRESSION_NONE, BML_COMPRESSION_PRS):
        raise HTTPException(
            400, f"compression: must be 0 (none) or 0x50 (PRS), got {req.compression}"
        )
    if req.file_alignment is not None and req.file_alignment not in (
        BML_ALIGN_NO_TEX, BML_ALIGN_HAS_TEX
    ):
        raise HTTPException(
            400,
            f"file_alignment: must be 0x20 or 0x800, got 0x{req.file_alignment:x}",
        )

    pack_entries: list[BmlPackEntry] = []
    for i, ent in enumerate(req.entries):
        if not isinstance(ent.name, str) or not ent.name:
            raise HTTPException(400, f"entries[{i}].name: missing")
        if ent.data_b64 is None:
            raise HTTPException(400, f"entries[{i}].data_b64: required")
        data = _decode_b64(ent.data_b64, ctx=f"entries[{i}].data_b64")
        tex_data: Optional[bytes] = None
        if ent.texture_b64 is not None:
            tex_data = _decode_b64(
                ent.texture_b64, ctx=f"entries[{i}].texture_b64"
            )
        try:
            pack_entries.append(BmlPackEntry(
                name=ent.name,
                data=data,
                decompressed_size=ent.decompressed_size,
                is_compressed=ent.is_compressed,
                texture_data=tex_data,
                texture_decompressed_size=ent.texture_decompressed_size,
                texture_is_compressed=ent.texture_is_compressed,
                unk_a=ent.unk_a,
                unk_b=ent.unk_b,
                unk_c=ent.unk_c,
                unk_d=ent.unk_d,
            ))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"entries[{i}]: {e}")

    try:
        out = pack_bml(
            pack_entries,
            compression=req.compression,
            optimal=req.optimal,
            file_alignment=req.file_alignment,
            has_textures_override=req.has_textures,
        )
    except ValueError as e:
        raise HTTPException(400, f"pack_bml failed: {e}")

    out_path = BML_EXPORT_DIR / name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(out)
    os.replace(tmp_path, out_path)

    md5 = _md5_bytes(out)
    log.info("build_bml %s -> %s (%d bytes, md5=%s)", name, out_path, len(out), md5)
    return {
        "ok": True,
        "path": str(out_path),
        "size": len(out),
        "md5": md5,
        "entry_count": len(pack_entries),
    }


# ============================================================================
# NJ / NJM build endpoints (2026-04-25): /api/build_nj, /api/build_njm,
# /api/sculpt/build_nj
# ----------------------------------------------------------------------------
# These let the frontend (or a sculpt agent) ship freshly-authored NJ
# meshes / NJM motions back to deployable bytes. The output lands in
# cache/nj_export/<name>.nj or cache/njm_export/<name>.njm and the
# response carries an md5 + size for the caller to sanity-check.
#
# JSON body shapes:
#
#   POST /api/build_nj
#     {
#       "name": "biri_ball.nj",
#       "model_json": {
#         "njtl_names": ["b_ball"],            # optional, default []
#         "nodes": [
#           {
#             "eval_flags": 0x17,
#             "position": [0.0, 0.0, 0.0],
#             "rotation_bams": [0, 0, 0],
#             "scale": [1.0, 1.0, 1.0],
#             "mesh_index": 0,                  # -1 if no mesh
#             "child_index": -1,                # -1 if no child
#             "sibling_index": -1               # -1 if no sibling
#           }
#         ],
#         "meshes": [
#           {
#             "bbox": [0.0, 0.0, 0.0, 7.07],
#             "vlist": [
#               {"type_id": 41, "flags": 0, "body_b64": "<...>"}
#             ],
#             "plist": [
#               {"type_id": 64, "flags": 0, "body_b64": "<...>"}
#             ]
#           }
#         ]
#       }
#     }
#     -> { ok, path, size, md5, chunk_count, vert_count }
#
#   POST /api/build_njm
#     {
#       "name": "walk.njm",
#       "motion_json": {
#         "frame_count": 30,
#         "type_flags": 3,
#         "inp_fn": 2,
#         "bones": [
#           {
#             "tracks": [
#               {"kind": 1, "narrow": true, "keyframes": [[0, 0.0, 0.0, 0.0]]},
#               {"kind": 2, "narrow": true, "keyframes": [[0, 0, 100, 0]]}
#             ]
#           }
#         ]
#       }
#     }
#     -> { ok, path, size, md5, frame_count, bone_count }
#
#   POST /api/sculpt/build_nj
#     {
#       "model_path": "biri_ball.bml#biri_ball.nj",
#       "inner_idx": 0,
#       "sculpt_sha": "<32-char hex>"
#     }
#     -> { ok, path, size, md5 }   (chunk_count, vert_count optional)
# ============================================================================

from formats.nj_writer import (  # noqa: E402
    NjChunk as _NjChunk,
    NjMeshChunks as _NjMeshChunks,
    NjModel as _NjModel,
    NjNode as _NjNode,
    encode_nj_model as _encode_nj_model,
    parse_nj_for_writer as _parse_nj_for_writer,
)
from formats.njm_writer import (  # noqa: E402
    NjmBoneTracks as _NjmBoneTracks,
    NjmRawMotion as _NjmRawMotion,
    NjmTrack as _NjmTrack,
    encode_njm as _encode_njm,
)

# Per-build body cap. NJ files top out at ~250 KB; NJM at ~100 KB.
MAX_BUILD_NJ_BODY = 16 * 1024 * 1024
MAX_BUILD_NJM_BODY = 16 * 1024 * 1024


def _build_nj_model_from_json(model_json: dict) -> _NjModel:
    """Decode an NJ model JSON request body into an ``NjModel``."""
    if not isinstance(model_json, dict):
        raise HTTPException(400, "model_json: must be an object")

    njtl_names = model_json.get("njtl_names") or []
    if not isinstance(njtl_names, list) or not all(isinstance(s, str) for s in njtl_names):
        raise HTTPException(400, "model_json.njtl_names: must be list of strings")

    nodes_in = model_json.get("nodes")
    if not isinstance(nodes_in, list) or len(nodes_in) == 0:
        raise HTTPException(400, "model_json.nodes: required, non-empty list")
    meshes_in = model_json.get("meshes") or []
    if not isinstance(meshes_in, list):
        raise HTTPException(400, "model_json.meshes: must be a list")

    nodes: list[_NjNode] = []
    for i, n in enumerate(nodes_in):
        if not isinstance(n, dict):
            raise HTTPException(400, f"nodes[{i}]: must be object")
        try:
            nodes.append(_NjNode(
                eval_flags=int(n.get("eval_flags", 0)),
                position=tuple(float(x) for x in n.get("position", (0.0, 0.0, 0.0))),
                rotation_bams=tuple(int(x) for x in n.get("rotation_bams", (0, 0, 0))),
                scale=tuple(float(x) for x in n.get("scale", (1.0, 1.0, 1.0))),
                mesh_index=int(n.get("mesh_index", -1)),
                child_index=int(n.get("child_index", -1)),
                sibling_index=int(n.get("sibling_index", -1)),
            ))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"nodes[{i}]: {e}")

    meshes: list[_NjMeshChunks] = []
    for i, m in enumerate(meshes_in):
        if not isinstance(m, dict):
            raise HTTPException(400, f"meshes[{i}]: must be object")
        bbox = m.get("bbox", (0.0, 0.0, 0.0, 0.0))
        try:
            bbox_t = tuple(float(x) for x in bbox)
            if len(bbox_t) != 4:
                raise ValueError(f"bbox length {len(bbox_t)} != 4")
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"meshes[{i}].bbox: {e}")
        vlist = []
        for j, c in enumerate(m.get("vlist", []) or []):
            try:
                vlist.append(_NjChunk(
                    type_id=int(c["type_id"]),
                    flags=int(c.get("flags", 0)),
                    body=_decode_b64(c.get("body_b64", ""), ctx=f"meshes[{i}].vlist[{j}].body_b64"),
                ))
            except (KeyError, TypeError) as e:
                raise HTTPException(400, f"meshes[{i}].vlist[{j}]: {e}")
        plist = []
        for j, c in enumerate(m.get("plist", []) or []):
            try:
                plist.append(_NjChunk(
                    type_id=int(c["type_id"]),
                    flags=int(c.get("flags", 0)),
                    body=_decode_b64(c.get("body_b64", ""), ctx=f"meshes[{i}].plist[{j}].body_b64"),
                ))
            except (KeyError, TypeError) as e:
                raise HTTPException(400, f"meshes[{i}].plist[{j}]: {e}")
        meshes.append(_NjMeshChunks(bbox=bbox_t, vlist=vlist, plist=plist))

    # Validate node references.
    for i, n in enumerate(nodes):
        if n.mesh_index >= len(meshes):
            raise HTTPException(400, f"nodes[{i}].mesh_index out of range")
        if n.child_index >= len(nodes):
            raise HTTPException(400, f"nodes[{i}].child_index out of range")
        if n.sibling_index >= len(nodes):
            raise HTTPException(400, f"nodes[{i}].sibling_index out of range")

    return _NjModel(njtl_names=njtl_names, nodes=nodes, meshes=meshes)


class BuildNjReq(BaseModel):
    name: str
    model_json: dict


@app.post("/api/build_nj")
def api_build_nj(req: BuildNjReq, request: Request):
    """Build a .nj file from a model_json description.

    Output lands in cache/nj_export/<name>.nj. Returns size + md5 plus
    derived metrics (chunk_count, vert_count) for the caller's sanity
    check before deploying.
    """
    _enforce_body_size(request, MAX_BUILD_NJ_BODY)
    name = _safe_archive_name(req.name)
    if not name.endswith(".nj"):
        raise HTTPException(400, "name: must end in .nj")

    model = _build_nj_model_from_json(req.model_json)
    try:
        out = _encode_nj_model(model)
    except (ValueError, struct.error) as e:
        raise HTTPException(400, f"encode_nj_model failed: {e}")

    # Derive metrics.
    chunk_count = sum(len(m.vlist) + len(m.plist) for m in model.meshes)
    vert_count = 0
    for mesh in model.meshes:
        for c in mesh.vlist:
            # Vertex chunks (type 32..50) carry their count in the body's
            # second u16 (after the body_words u16).
            if 32 <= c.type_id <= 50 and len(c.body) >= 6:
                vert_count += struct.unpack_from("<H", c.body, 4)[0]

    out_path = NJ_EXPORT_DIR / name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(out)
    os.replace(tmp_path, out_path)
    md5 = _md5_bytes(out)
    log.info(
        "build_nj %s -> %s (%d bytes, md5=%s, chunks=%d, verts=%d)",
        name, out_path, len(out), md5, chunk_count, vert_count,
    )
    return {
        "ok": True,
        "path": str(out_path),
        "size": len(out),
        "md5": md5,
        "chunk_count": chunk_count,
        "vert_count": vert_count,
    }


def _build_njm_motion_from_json(motion_json: dict) -> _NjmRawMotion:
    """Decode an NJM motion JSON body into an ``NjmRawMotion``."""
    if not isinstance(motion_json, dict):
        raise HTTPException(400, "motion_json: must be an object")
    try:
        motion = _NjmRawMotion(
            frame_count=int(motion_json.get("frame_count", 0)),
            type_flags=int(motion_json.get("type_flags", 0)),
            inp_fn=int(motion_json.get("inp_fn", 0)),
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"motion_json header: {e}")
    bones_in = motion_json.get("bones") or []
    if not isinstance(bones_in, list):
        raise HTTPException(400, "motion_json.bones: must be a list")
    for i, b in enumerate(bones_in):
        if not isinstance(b, dict):
            raise HTTPException(400, f"bones[{i}]: must be object")
        bone = _NjmBoneTracks()
        for j, t in enumerate(b.get("tracks") or []):
            try:
                kind = int(t["kind"])
                kfs = list(t.get("keyframes") or [])
                # Coerce each keyframe to a tuple for the encoder.
                kfs_t = [tuple(kf) for kf in kfs]
                bone.tracks_by_kind[kind] = _NjmTrack(
                    kind=kind,
                    keyframes=kfs_t,
                    narrow=bool(t.get("narrow", True)),
                    stored_count=t.get("stored_count"),
                )
            except (KeyError, TypeError, ValueError) as e:
                raise HTTPException(400, f"bones[{i}].tracks[{j}]: {e}")
        motion.bones.append(bone)
    return motion


class BuildNjmReq(BaseModel):
    name: str
    motion_json: dict


@app.post("/api/build_njm")
def api_build_njm(req: BuildNjmReq, request: Request):
    """Build a .njm file from a motion_json description.

    Output lands in cache/njm_export/<name>.njm. Returns size + md5 plus
    bone_count and frame_count for sanity-checking.
    """
    _enforce_body_size(request, MAX_BUILD_NJM_BODY)
    name = _safe_archive_name(req.name)
    if not name.endswith(".njm"):
        raise HTTPException(400, "name: must end in .njm")

    motion = _build_njm_motion_from_json(req.motion_json)
    try:
        out = _encode_njm(motion)
    except (ValueError, struct.error) as e:
        raise HTTPException(400, f"encode_njm failed: {e}")

    out_path = NJM_EXPORT_DIR / name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(out)
    os.replace(tmp_path, out_path)
    md5 = _md5_bytes(out)
    log.info(
        "build_njm %s -> %s (%d bytes, md5=%s, bones=%d, frames=%d)",
        name, out_path, len(out), md5, len(motion.bones), motion.frame_count,
    )
    return {
        "ok": True,
        "path": str(out_path),
        "size": len(out),
        "md5": md5,
        "bone_count": len(motion.bones),
        "frame_count": motion.frame_count,
    }


class SculptBuildNjReq(BaseModel):
    model_path: str
    inner_idx: int = 0
    sculpt_sha: str
    output_name: Optional[str] = None


@app.post("/api/sculpt/build_nj")
def api_sculpt_build_nj(req: SculptBuildNjReq, request: Request):
    """Bridge: read a sculpted mesh JSON sidecar, encode to NJ.

    Reads the C2 sculpt agent's output from
    ``cache/sculpted_meshes/<safe_path>__<sha>.json``, applies the
    sparse displacements to the source NJ's vertex positions, encodes
    the result, and writes ``cache/nj_export/<output_name>.nj``.

    The sculpt JSON format is documented in ``formats/sculpt.py``: each
    submesh records a ``displacement_b64`` (float32[N*3]) plus a
    ``modified_indices_b64`` (uint32[K]) for sparse storage. Dense
    storage uses the displacement directly.

    Returns ``{ok, path, size, md5}``. The C2 agent owns the JSON
    sidecar; this endpoint reads it cold without trusting any
    post-write metadata.
    """
    _enforce_body_size(request, MAX_BUILD_NJ_BODY)

    # Validate sculpt_sha format (32 hex chars).
    sha = req.sculpt_sha.strip().lower()
    if not (8 <= len(sha) <= 64) or any(c not in "0123456789abcdef" for c in sha):
        raise HTTPException(400, "sculpt_sha: must be 8-64 hex chars")

    # Validate model_path. Format: "<bml>#<inner>.nj".
    mp = req.model_path
    if not isinstance(mp, str) or "#" not in mp:
        raise HTTPException(400, "model_path: must be '<bml>#<inner>.nj'")
    bml_name, _, inner_name = mp.partition("#")
    bml_name = _safe_archive_name(bml_name)
    if not inner_name.endswith(".nj"):
        raise HTTPException(400, "model_path inner must end in .nj")

    # Locate the sculpt JSON. The C2 agent's saver uses the same
    # _sculpt_safe_filename helper as /api/sculpt/save so the layouts
    # stay in lock-step.
    sculpt_filename = _sculpt_safe_filename(mp, sha)
    sculpt_path = SCULPT_CACHE_DIR / sculpt_filename
    if not sculpt_path.exists():
        raise HTTPException(
            404,
            f"sculpt JSON not found: {sculpt_path.name} (expected at {sculpt_path})",
        )

    # Locate the source BML/NJ.
    bml_p = LIVE_DATA_DIR / bml_name
    if not bml_p.exists():
        bml_p = DEV_DATA_DIR / bml_name
        if not bml_p.exists():
            raise HTTPException(404, f"BML not found: {bml_name}")
    try:
        all_e = extract_bml(bml_p.read_bytes())
    except Exception as e:
        raise HTTPException(500, f"BML extract failed: {e}")
    if inner_name not in all_e:
        raise HTTPException(404, f"BML {bml_name} has no inner {inner_name}")
    src_bytes = all_e[inner_name]

    # Read the sculpt sidecar.
    try:
        sculpt_data = json.loads(sculpt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(400, f"sculpt JSON parse failed: {e}")

    # Parse the source model. We mutate vertex positions in the chunk
    # bodies directly — the encoder's layout_hint preserves byte
    # positions of unchanged regions, so the diff is minimal and the
    # game-loadable property is preserved.
    try:
        model = _parse_nj_for_writer(src_bytes)
    except Exception as e:
        raise HTTPException(400, f"NJ parse failed: {e}")

    # Apply sculpt displacements.
    submeshes = sculpt_data.get("submeshes") or []
    n_applied = 0
    for sm in submeshes:
        if not isinstance(sm, dict):
            continue
        idx = int(sm.get("submesh_idx", -1))
        if idx < 0 or idx >= len(model.meshes):
            continue
        # Decode displacement (sparse or dense).
        disp_b64 = sm.get("displacement_b64")
        if not disp_b64:
            continue
        try:
            disp_bytes = _decode_b64(disp_b64, ctx="displacement_b64")
        except HTTPException:
            continue
        n_floats = len(disp_bytes) // 4
        if n_floats < 3:
            continue
        # Modified indices (sparse) or all (dense).
        mods_b64 = sm.get("modified_indices_b64")
        if mods_b64:
            mods_bytes = _decode_b64(mods_b64, ctx="modified_indices_b64")
            modified = list(struct.unpack(f"<{len(mods_bytes)//4}I", mods_bytes))
        else:
            modified = list(range(n_floats // 3))

        # Apply to each vertex chunk in this mesh's vlist whose vertex
        # range covers the modified index. The submesh-to-chunk mapping
        # is positional (one vertex chunk per submesh in shipped data).
        mesh = model.meshes[idx]
        for c in mesh.vlist:
            if not (32 <= c.type_id <= 50):
                continue
            if len(c.body) < 6:
                continue
            base_idx, vert_count = struct.unpack_from("<HH", c.body, 2)
            # Compute per-vertex stride.
            stride = _nj_vertex_chunk_stride(c.type_id)
            if stride <= 0:
                continue
            new_body = bytearray(c.body)
            for slot_local, mod_i in enumerate(modified):
                global_idx = mod_i  # sculpt indices are submesh-local;
                # PSOBB vertex chunks share a submesh-aligned base_idx
                # so global_idx == local index within the chunk.
                if global_idx < 0 or global_idx >= vert_count:
                    continue
                vert_off = 6 + global_idx * stride
                if vert_off + 12 > len(new_body):
                    continue
                # Read existing pos.
                ox, oy, oz = struct.unpack_from("<3f", new_body, vert_off)
                # Read displacement at slot_local.
                disp_off = mod_i * 12
                if disp_off + 12 > len(disp_bytes):
                    continue
                dx, dy, dz = struct.unpack_from("<3f", disp_bytes, disp_off)
                struct.pack_into(
                    "<3f", new_body, vert_off,
                    ox + dx, oy + dy, oz + dz,
                )
                n_applied += 1
            c.body = bytes(new_body)

    # Encode.
    try:
        out = _encode_nj_model(model)
    except (ValueError, struct.error) as e:
        raise HTTPException(400, f"encode_nj_model failed: {e}")

    out_name = req.output_name or inner_name
    out_name = _safe_archive_name(out_name)
    if not out_name.endswith(".nj"):
        raise HTTPException(400, "output_name: must end in .nj")
    out_path = NJ_EXPORT_DIR / out_name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(out)
    os.replace(tmp_path, out_path)
    md5 = _md5_bytes(out)
    log.info(
        "sculpt/build_nj %s#%s -> %s (%d bytes, md5=%s, applied=%d)",
        bml_name, inner_name, out_path, len(out), md5, n_applied,
    )
    return {
        "ok": True,
        "path": str(out_path),
        "size": len(out),
        "md5": md5,
        "applied_displacements": n_applied,
    }


# ============================================================================
# External-format import (2026-04-25): /api/import/parse, /api/import/build_nj,
# /api/import/replace, /api/import/templates
# ----------------------------------------------------------------------------
# Lets users drag a .obj/.gltf/.glb file from Blender/Maya/3ds Max into the
# editor and have it converted to a deployable PSOBB .nj. The hand-rolled
# parsers in formats/import_external.py do the heavy lifting; this section
# exposes them over HTTP and chains them with the existing build_nj /
# build_bml / deploy pipelines.
#
# Wire shapes:
#
#   POST /api/import/parse                     multipart file upload
#     -> {ok, model_json, format, warnings, mesh_count, bone_count, ...}
#
#   POST /api/import/build_nj                  application/json
#     {model_json, name, target_class?, axis_flip_z?, scale?}
#     -> {ok, path, size, md5, vert_count, bone_count}
#
#   POST /api/import/replace                   application/json
#     {import_nj_path, target_bml, target_inner, output_name?}
#     -> {ok, archive_path, size, md5, replaced_inner}
#
#   GET  /api/import/templates
#     -> {ok, templates: [{name, bone_count, description, source}, ...]}
# ============================================================================

from formats.import_external import (  # noqa: E402
    imported_from_json as _imp_from_json,
    imported_to_json as _imp_to_json,
    imported_to_nj as _imp_to_nj,
    list_templates as _imp_list_templates,
    parse_external as _imp_parse_external,
)

# Per-import body cap. glTF/glb can be large (textured characters
# routinely 5-15 MB). Cap at 64 MB to keep server memory pressure in
# check while still admitting Blender's "export everything" outputs.
MAX_IMPORT_PARSE_BODY = 64 * 1024 * 1024  # 64 MB
MAX_IMPORT_BUILD_BODY = 64 * 1024 * 1024


@app.post("/api/import/parse")
async def api_import_parse(request: Request, file: UploadFile = File(...)):
    """Parse an uploaded .obj/.gltf/.glb/.fbx file into the editor's
    intermediate ImportedModel JSON shape.

    The response includes a base64-encoded interleaved Float32 vertex
    buffer per mesh (same shape as /api/model_mesh) so the existing
    psoApplyMeshPayload viewer can render the import as a preview without
    a server round-trip.

    Also returns:
      - ``warnings``: parser advisories (no skeleton, no normals, etc.)
      - ``mesh_count`` / ``bone_count`` / ``vert_total`` / ``tri_total``
      - ``format``: "obj" | "gltf" | "glb" | "fbx"

    Raises 413 when the upload exceeds MAX_IMPORT_PARSE_BODY, 400 on
    parse failure (with the parser's error message in detail).
    """
    _enforce_body_size(request, MAX_IMPORT_PARSE_BODY)
    if file is None or not file.filename:
        raise HTTPException(400, "file: missing")
    name = _safe_archive_name(Path(file.filename).name)
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "file: empty")
    if len(data) > MAX_IMPORT_PARSE_BODY:
        raise HTTPException(413, f"file too large ({len(data)} > {MAX_IMPORT_PARSE_BODY})")
    # Audit C-6: hand off CPU-heavy parsing (_imp_parse_external,
    # _imp_to_json, optional glTF/FBX animation summary) to a worker
    # thread so the event loop stays responsive.
    return await asyncio.to_thread(_import_parse_sync_work, data, name)


def _import_parse_sync_work(data: bytes, name: str) -> dict:
    """CPU-heavy synchronous tail of api_import_parse. Runs in a worker thread.

    Parses the model, builds the JSON payload, and on glTF/FBX inputs
    appends a best-effort animation summary.
    """
    try:
        model = _imp_parse_external(data, name)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, f"parse failed: {e}")
    payload = _imp_to_json(model)
    payload["ok"] = True
    payload["filename"] = name
    payload["format"] = model.source_format
    # 2026-04-25 v2 addition: surface animation summary when the source
    # carries one or more animation tracks. Caller uses this to gate
    # the "Convert to NJM" button in the import modal. v2.1 adds FBX
    # alongside glTF.
    payload["animations"] = []
    if model.source_format in ("gltf", "glb"):
        try:
            anim_imp = _imp_parse_gltf_anim(data)
            for a in anim_imp.animations:
                payload["animations"].append({
                    "name": a.name,
                    "duration_seconds": a.duration_seconds,
                    "track_count": len(a.tracks),
                })
        except Exception as e:  # noqa: BLE001 — best-effort summary
            log.debug("import/parse: animation summary failed for %s: %s", name, e)
    elif model.source_format == "fbx":
        try:
            from formats.fbx_reader import parse_fbx_with_animations as _imp_parse_fbx_anim
            anim_imp = _imp_parse_fbx_anim(data)
            for a in anim_imp.animations:
                payload["animations"].append({
                    "name": a.name,
                    "duration_seconds": a.duration_seconds,
                    "track_count": len(a.tracks),
                })
        except Exception as e:  # noqa: BLE001 — best-effort summary
            log.debug("import/parse: FBX animation summary failed for %s: %s", name, e)
    log.info(
        "import/parse %s (format=%s, meshes=%d, bones=%d, verts=%d, warnings=%d, animations=%d)",
        name, model.source_format,
        len(model.meshes), len(model.bones),
        payload.get("vert_total", 0), len(model.warnings),
        len(payload["animations"]),
    )
    return payload


class ImportBuildNjReq(BaseModel):
    name: str
    model_json: dict
    target_class: Optional[str] = None
    axis_flip_z: bool = True
    scale: float = 1.0


@app.post("/api/import/build_nj")
def api_import_build_nj(req: ImportBuildNjReq, request: Request):
    """Convert a model_json (from /api/import/parse) to a deployable .nj.

    Output lands in cache/nj_export/<name>.nj. Returns size + md5 + the
    same convenience metrics as /api/build_nj for the UI's confirmation
    dialog.

    The conversion applies axis flip (right-handed -> left-handed),
    uniform scale, and either uses the source skeleton (when present)
    or substitutes the named ``target_class`` skeleton template
    (player_body / player_head / monster_humanoid / monster_quadruped /
    boss_dragon).
    """
    _enforce_body_size(request, MAX_IMPORT_BUILD_BODY)
    name = _safe_archive_name(req.name)
    if not name.endswith(".nj"):
        raise HTTPException(400, "name: must end in .nj")
    if req.scale <= 0 or req.scale > 10000.0:
        raise HTTPException(400, f"scale: out of range (got {req.scale})")
    try:
        imp = _imp_from_json(req.model_json)
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(400, f"model_json: {e}")
    target = req.target_class if req.target_class else None
    if target and target not in _imp_list_templates():
        raise HTTPException(400, f"target_class {target!r} not found in data/import_templates/")
    try:
        nj_model = _imp_to_nj(
            imp,
            target_class=target,
            axis_flip_z=bool(req.axis_flip_z),
            scale=float(req.scale),
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, f"imported_to_nj failed: {e}")
    try:
        out = _encode_nj_model(nj_model)
    except (ValueError, struct.error) as e:
        raise HTTPException(400, f"encode_nj_model failed: {e}")
    out_path = NJ_EXPORT_DIR / name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(out)
    os.replace(tmp_path, out_path)
    md5 = _md5_bytes(out)
    # Aggregate vert_count from emitted vertex chunks.
    vert_count = 0
    for mesh in nj_model.meshes:
        for c in mesh.vlist:
            if 32 <= c.type_id <= 50 and len(c.body) >= 6:
                vert_count += struct.unpack_from("<H", c.body, 4)[0]
    log.info(
        "import/build_nj %s (target=%s, flip=%s, scale=%.3f) -> %s "
        "(%d bytes, md5=%s, verts=%d, bones=%d)",
        name, target, req.axis_flip_z, req.scale, out_path,
        len(out), md5, vert_count, len(nj_model.nodes),
    )
    return {
        "ok": True,
        "path": str(out_path),
        "size": len(out),
        "md5": md5,
        "vert_count": vert_count,
        "bone_count": len(nj_model.nodes),
        "mesh_count": len(nj_model.meshes),
        "target_class": target,
    }


class ImportReplaceReq(BaseModel):
    # Source: a previously-built .nj in cache/nj_export/.
    import_nj_path: str
    # Target: a BML in DATA_DIR/LIVE_DATA_DIR + an inner entry name.
    target_bml: str
    target_inner: str
    output_name: Optional[str] = None


@app.post("/api/import/replace")
def api_import_replace(req: ImportReplaceReq, request: Request):
    """Take a built .nj and substitute it for the inner of a target BML.

    Reads the imported .nj from cache/nj_export/<import_nj_path> and
    swaps it into the target BML's inner slot, preserving every other
    inner entry verbatim. The resulting BML lands in
    cache/bml_export/<output_name> (default = target_bml's name).

    Caller can deploy via /api/deploy/<output_name>.
    """
    _enforce_body_size(request, MAX_IMPORT_BUILD_BODY)
    nj_name = _safe_archive_name(Path(req.import_nj_path).name)
    bml_name = _safe_archive_name(Path(req.target_bml).name)
    inner = req.target_inner.strip()
    if not inner or "/" in inner or "\\" in inner:
        raise HTTPException(400, "target_inner: must be a bare entry name")
    nj_p = NJ_EXPORT_DIR / nj_name
    if not nj_p.exists():
        raise HTTPException(404, f"import_nj_path not found in cache/nj_export: {nj_name}")
    src_bml = LIVE_DATA_DIR / bml_name
    if not src_bml.exists():
        src_bml = DEV_DATA_DIR / bml_name
        if not src_bml.exists():
            raise HTTPException(404, f"target_bml not found: {bml_name}")
    # Round-trip the BML for re-pack.
    from formats.bml import parse_bml_for_pack, parse_bml_pack_meta
    bml_bytes = src_bml.read_bytes()
    try:
        pack_entries = parse_bml_for_pack(bml_bytes)
        meta = parse_bml_pack_meta(bml_bytes)
    except Exception as e:
        raise HTTPException(500, f"BML parse failed: {e}")
    # Find target inner.
    matched = -1
    for i, ent in enumerate(pack_entries):
        if ent.name == inner:
            matched = i
            break
    if matched < 0:
        names = [e.name for e in pack_entries]
        raise HTTPException(404, f"target_inner {inner!r} not in BML; have: {names[:10]}")
    new_data = nj_p.read_bytes()
    # Preserve textures + unk fields; replace inner data only.
    pack_entries[matched] = BmlPackEntry(
        name=pack_entries[matched].name,
        data=new_data,
        decompressed_size=len(new_data),
        is_compressed=False,  # let the packer PRS-compress fresh
        texture_data=pack_entries[matched].texture_data,
        texture_decompressed_size=pack_entries[matched].texture_decompressed_size,
        texture_is_compressed=pack_entries[matched].texture_is_compressed,
        unk_a=pack_entries[matched].unk_a,
        unk_b=pack_entries[matched].unk_b,
        unk_c=pack_entries[matched].unk_c,
        unk_d=pack_entries[matched].unk_d,
    )
    # Re-pack.
    try:
        out = pack_bml(
            pack_entries,
            compression=meta["compression"],
            has_textures_override=bool(meta.get("has_textures", False)),
            file_alignment=meta["file_alignment"],
        )
    except Exception as e:
        raise HTTPException(500, f"pack_bml failed: {e}")
    out_name = _safe_archive_name(req.output_name or bml_name)
    out_path = BML_EXPORT_DIR / out_name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(out)
    os.replace(tmp_path, out_path)
    md5 = _md5_bytes(out)
    log.info(
        "import/replace %s#%s -> %s (%d bytes, md5=%s, replaced inner #%d)",
        bml_name, inner, out_path, len(out), md5, matched,
    )
    return {
        "ok": True,
        "archive_path": str(out_path),
        "archive_name": out_name,
        "size": len(out),
        "md5": md5,
        "replaced_inner": inner,
        "replaced_index": matched,
    }


@app.get("/api/import/templates")
def api_import_templates():
    """Return the available skeleton templates with bone counts + descriptions.

    The UI uses this to populate the target-class dropdown in the
    import preview pane. Each entry includes the bone count so the user
    can pick a template that matches the source's expected
    rig (e.g. 64 bones for a player body skin from Blender).
    """
    out = []
    from formats.import_external import _TEMPLATES_DIR
    for name in _imp_list_templates():
        p = _TEMPLATES_DIR / f"{name}.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "name": name,
                "bone_count": int(data.get("bone_count", 0)),
                "description": str(data.get("description") or ""),
                "source": str(data.get("source") or ""),
            })
        except (OSError, json.JSONDecodeError) as e:
            log.warning("import/templates: skipping %s: %s", name, e)
            continue
    return {"ok": True, "templates": out}


# ============================================================================
# Blend-shape JSON side-file exporter (v4, 2026-04-25)
# ----------------------------------------------------------------------------
# PSOBB has no morph-target rendering, but FBX BlendShape data is still
# useful for users who want to round-trip through Blender or build a
# separate facial-animation pipeline. The exporter writes a
# self-contained JSON file that ``formats.import_external.blend_shapes_from_json``
# can re-hydrate into ``List[BlendShape]`` byte-identically.
#
# Cache location: ``cache/blend_shape_export/<safe>.json``. The safe
# name is derived from the source filename to avoid clobbering other
# exports.
# ============================================================================

BLEND_SHAPE_EXPORT_DIR = CACHE_DIR / "blend_shape_export"
BLEND_SHAPE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Body cap for the export upload; same as the parse endpoint since the
# user uploads the same file.
MAX_BLEND_SHAPE_EXPORT_BODY = MAX_IMPORT_PARSE_BODY


@app.post("/api/import/blend_shapes/export")
async def api_import_blend_shapes_export(
    request: Request,
    file: UploadFile = File(...),
    model_path: Optional[str] = None,
):
    """Parse the uploaded model + emit its blend-shape data as JSON.

    Accepts the same upload shape as ``/api/import/parse``: a multipart
    file containing an FBX (or any source the importer can parse — the
    endpoint just re-runs the parser and dumps ``model.blend_shapes``).

    The optional ``model_path`` query parameter overrides the output
    filename. Otherwise the response output sits at
    ``cache/blend_shape_export/<source-stem>.json``.

    Response shape:
        { ok, path, size, shape_count, names, md5 }

    A model with zero blend shapes still emits an empty wrapper file
    (``shape_count=0``) so the caller can tell "nothing to export" from
    "endpoint failed".
    """
    _enforce_body_size(request, MAX_BLEND_SHAPE_EXPORT_BODY)
    if file is None or not file.filename:
        raise HTTPException(400, "file: missing")
    src_name = _safe_archive_name(Path(file.filename).name)
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "file: empty")
    if len(data) > MAX_BLEND_SHAPE_EXPORT_BODY:
        raise HTTPException(413, f"file too large ({len(data)} > {MAX_BLEND_SHAPE_EXPORT_BODY})")
    try:
        model = _imp_parse_external(data, src_name)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, f"parse failed: {e}")
    # Output filename: use model_path if supplied (sanitised), else
    # derive from the source filename + ".json".
    if model_path:
        out_stem = _safe_archive_name(Path(model_path).stem) or "blend_shapes"
    else:
        out_stem = Path(src_name).stem or "blend_shapes"
    out_name = f"{out_stem}.json"
    out_path = BLEND_SHAPE_EXPORT_DIR / out_name
    from formats.import_external import export_blend_shapes_json
    payload = export_blend_shapes_json(model)
    text = json.dumps(payload, indent=2, sort_keys=False)
    encoded = text.encode("utf-8")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(encoded)
    os.replace(tmp_path, out_path)
    md5 = _md5_bytes(encoded)
    names = [s["name"] for s in payload["shapes"]]
    log.info(
        "import/blend_shapes/export %s -> %s (%d shapes, %d bytes, md5=%s)",
        src_name, out_path, payload["shape_count"], len(encoded), md5,
    )
    return {
        "ok": True,
        "path": str(out_path),
        "size": len(encoded),
        "shape_count": payload["shape_count"],
        "names": names,
        "md5": md5,
    }


# ============================================================================
# Animation import (2026-04-25): /api/import/animation,
# /api/import/animation/replace
# ----------------------------------------------------------------------------
# Lets users drop a .glb / .gltf with one or more animation tracks and
# retarget the chosen animation onto a PSOBB model's skeleton, emitting
# a deployable .njm. Workflow:
#
#   1. POST a .glb to /api/import/animation along with the target BML
#      path + the chosen target NJM motion-name. The server:
#        - parses the source via parse_gltf_with_animations
#        - reads the target BML's body .nj for the skeleton
#        - retargets the first animation onto the target skeleton
#          using the chosen bone-name map (auto-picked from the BML
#          name or supplied explicitly)
#        - encodes via njm_writer.encode_njm
#        - stages at cache/njm_export/<safe_name>.njm
#      Returns metadata so the UI can confirm the retarget worked
#      (mapped vs dropped bone counts, frame_count, bone_count, md5).
#
#   2. POST to /api/import/animation/replace to splice the staged .njm
#      into a target BML (replacing an existing inner motion or
#      appending a new entry), staging the result at
#      cache/bml_export/<safe>.bml. The user can then deploy via the
#      existing /api/deploy/<bml> path.
#
# The retargeter doesn't currently support "merge into existing motion
# bundle" — each retarget produces a single-motion .njm. NpcApcMot.bml
# carries 1 motion per inner entry, so this is fine for the typing-on
# -lobby-girl workflow.
# ============================================================================

from formats.import_external import parse_gltf_with_animations as _imp_parse_gltf_anim  # noqa: E402
from formats.anim_retarget import (  # noqa: E402
    LOBBY_GIRL_BONE_MAP as _ANIM_LOBBY_GIRL_MAP,
    get_builtin_bone_map as _anim_get_bone_map,
    retarget_animation as _anim_retarget,
    summarize_retarget as _anim_summarize,
)

# Per-import animation body cap. glTF animations are typically smaller
# than full skinned models (just keyframes), but Mixamo bundles can ship
# 30+ MB single files. Stay under 64 MB to match the existing import
# limits.
MAX_ANIM_BODY = 64 * 1024 * 1024


def _pick_bone_map_for_target(target_bml: str) -> tuple[str, dict]:
    """Heuristically pick a bone-name map based on the target BML name.

    Returns ``(map_name, map_dict)``. Defaults to "lobby_girl" for any
    ``bm_npc_*`` BML; the UI can override via an explicit bone_map_name
    parameter.
    """
    base = Path(target_bml).name.lower()
    if base.startswith("bm_npc_") or base.startswith("bm_kenkyuw") or "kenkyu" in base or "momoka" in base or "hosa" in base:
        return ("lobby_girl", dict(_ANIM_LOBBY_GIRL_MAP))
    # Fallback: lobby_girl is our only ground-truth map for now; the
    # NPC humanoid skeleton is shared across most bm_npc_* + many
    # quest NPC files so it's a sensible default until we author more.
    return ("lobby_girl", dict(_ANIM_LOBBY_GIRL_MAP))


def _safe_motion_name(name: str) -> str:
    """Return a safe filename for a .njm export.

    Stripped of path components, kept to ASCII alphanumeric + a few
    punctuation chars. Truncated to 60 chars before the ``.njm``
    extension.
    """
    bare = Path(name).name
    bare = re.sub(r"[^A-Za-z0-9_\-.]", "_", bare)
    if not bare.lower().endswith(".njm"):
        bare = bare[:60] + ".njm"
    return bare[:64]


@app.post("/api/import/animation")
async def api_import_animation(
    request: Request,
    file: UploadFile = File(...),
    target_model_path: str = Form(...),
    motion_name: str = Form(...),
    target_inner: Optional[str] = Form(None),
    include_translation: bool = Form(False),
    target_fps: int = Form(30),
    flip_z: bool = Form(True),
    bone_map_name: Optional[str] = Form(None),
    enable_ik: bool = Form(True),
    enable_ik_rotation: bool = Form(True),
    mirror: bool = Form(False),
):
    """Parse a glTF animation, retarget onto a target skeleton, stage
    the resulting .njm + a preview sidecar JSON.

    DEFAULT (preview-only) flow. Writes:

      - ``cache/njm_export/<safe>.njm``                  staged NJM
      - ``cache/njm_export/<safe>.njm.preview.json``     sidecar so
        ``/api/anim_preview/list`` can find this animation when the user
        opens the target model in the editor.

    Critically, this endpoint **does not** write into ``cache/bml_export/``
    or ``<install>/data/``. The animation appears under the editor's
    "Imported Animations" section in the Motions tab and plays in the
    viewport via ``psoLoadMotion``. The game stays vanilla. Users who
    explicitly want to repack a BML with the new motion call
    ``/api/import/animation/replace`` (append) or
    ``/api/import/animation/swap`` (strict-replace a slot) on top of the
    staged NJM produced here.

    Form fields
    -----------
    file
        The source .glb or .gltf containing the animation.
    target_model_path
        Path to the destination BML (e.g. "bm_npc_kenkyu_w.bml" — the
        BML the animation will eventually be packed into).
    motion_name
        Output filename (e.g. "lobby_girl_typing.njm"). Sanitised by
        the server.
    target_inner
        Optional name of the inner .nj inside ``target_model_path``
        whose skeleton we retarget against. Defaults to the BML's
        first inner.
    include_translation
        When True, emit a POS track on the hip bone (otherwise rotation
        only — the right default for "stand and type").
    target_fps
        Resample rate. PSOBB sim is 30 Hz; rarely useful to change.
    flip_z
        When True (default), apply the glTF -> PSOBB Z-mirror.
    bone_map_name
        Override the auto-picked bone map. Currently supports
        ``"lobby_girl"`` only; pass None to let the server pick.
    enable_ik
        When True (default), run an IK pass over hand/foot end-effector
        chains so the wrist/ankle world position matches the source's.
        Closes the bone-length-mismatch gap that the 1:1 quat copy
        leaves on different-arm-length skeletons (typical Mixamo->
        PSOBB retarget). Disable for simple bone-set retargets where
        no length adjustment is wanted.
    enable_ik_rotation
        When True (default, v3 2026-04-25), the IK pass also rotates
        the end-effector bone (wrist/ankle) so its world rotation
        matches the source's. Without this, a Mixamo wrist twist
        doesn't propagate to the target hand. Disable to reproduce v2
        baseline behaviour (positional IK only).
    mirror
        When True (v3 2026-04-25), apply a left<->right mirror as a
        post-processing pass on the retargeted motion. Swaps every
        Left*/Right* track pair and mirrors per-frame quaternions
        across the YZ plane. Useful for converting one-handed Mixamo
        clips ("right-hand wave") into their mirrored variant without
        re-authoring the source.

    Returns
    -------
    JSON ``{ok, njm_path, frame_count, bone_count, retargeted_bones,
    dropped_bones, md5, size, ...}``. When IK is enabled, ``ik`` key
    in the response surfaces per-chain gap statistics. When mirror is
    enabled, ``mirror`` key surfaces pair-detection diagnostics.
    """
    _enforce_body_size(request, MAX_ANIM_BODY)
    if file is None or not file.filename:
        raise HTTPException(400, "file: missing")
    name = _safe_archive_name(Path(file.filename).name)
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "file: empty")
    if len(data) > MAX_ANIM_BODY:
        raise HTTPException(413, f"file too large ({len(data)} > {MAX_ANIM_BODY})")
    if target_fps <= 0 or target_fps > 240:
        raise HTTPException(400, f"target_fps out of range: {target_fps}")

    # Audit C-6: hand off CPU-heavy work (glTF parse, BML/skeleton load,
    # retarget, NJM encode, sidecar write) to a worker thread so the
    # event loop is never blocked by a long retarget pass.
    return await asyncio.to_thread(
        _import_animation_sync_work,
        data,
        name,
        target_model_path,
        motion_name,
        target_inner,
        bool(include_translation),
        int(target_fps),
        bool(flip_z),
        bone_map_name,
        bool(enable_ik),
        bool(enable_ik_rotation),
        bool(mirror),
    )


def _import_animation_sync_work(
    data: bytes,
    name: str,
    target_model_path: str,
    motion_name: str,
    target_inner: Optional[str],
    include_translation: bool,
    target_fps: int,
    flip_z: bool,
    bone_map_name: Optional[str],
    enable_ik: bool,
    enable_ik_rotation: bool,
    mirror: bool,
) -> dict:
    """CPU-heavy synchronous tail of api_import_animation. Runs in a worker thread.

    Parses the source glTF, resolves the target BML's skeleton, runs the
    retarget pass (with optional IK + mirror), encodes the .njm, and
    stages both the .njm and its preview sidecar.
    """
    # ---- Parse source glTF ----
    try:
        imp = _imp_parse_gltf_anim(data)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, f"glTF parse failed: {e}")
    if not imp.animations:
        raise HTTPException(400, f"{name}: no animations found in glTF")

    # ---- Resolve target skeleton ----
    target_bml_name = _safe_archive_name(Path(target_model_path).name)
    src_bml = LIVE_DATA_DIR / target_bml_name
    if not src_bml.exists():
        src_bml = DATA_DIR / target_bml_name
        if not src_bml.exists():
            src_bml = DEV_DATA_DIR / target_bml_name
            if not src_bml.exists():
                raise HTTPException(404, f"target_model_path not found: {target_bml_name}")
    try:
        bml_bytes = src_bml.read_bytes()
        from formats.bml import parse_bml as _parse_bml, _prs_decompress as _bml_prs_decompress
        from formats.xj import parse_skeleton as _parse_skeleton
        bml_entries = _parse_bml(bml_bytes)
    except Exception as e:
        raise HTTPException(500, f"BML parse failed: {e}")
    if not bml_entries:
        raise HTTPException(400, f"BML {target_bml_name} has no inner entries")
    pick_inner = None
    if target_inner:
        for ent in bml_entries:
            if ent.name == target_inner:
                pick_inner = ent
                break
        if pick_inner is None:
            raise HTTPException(404, f"target_inner not in BML: {target_inner}")
    else:
        # First .nj entry; fall back to first entry of any kind.
        for ent in bml_entries:
            if ent.name.lower().endswith(".nj"):
                pick_inner = ent
                break
        if pick_inner is None:
            pick_inner = bml_entries[0]
    inner_bytes = _bml_prs_decompress(bml_bytes[pick_inner.offset:pick_inner.offset + pick_inner.size_compressed])
    try:
        target_skel = _parse_skeleton(inner_bytes)
    except Exception as e:
        raise HTTPException(500, f"target skeleton parse failed: {e}")
    if not target_skel:
        raise HTTPException(400, f"{target_bml_name}#{pick_inner.name}: no skeleton bones")

    # ---- Resolve bone map ----
    if bone_map_name:
        try:
            bone_map = _anim_get_bone_map(bone_map_name)
        except KeyError:
            raise HTTPException(400, f"unknown bone_map_name: {bone_map_name}")
        chosen_map_name = bone_map_name
    else:
        chosen_map_name, bone_map = _pick_bone_map_for_target(target_bml_name)

    # ---- Retarget ----
    try:
        motion = _anim_retarget(
            imp.animations[0],
            imp.model.bones,
            target_skel,
            bone_map,
            target_fps=int(target_fps),
            include_translation=bool(include_translation),
            flip_z=bool(flip_z),
            enable_ik=bool(enable_ik),
            enable_ik_rotation=bool(enable_ik_rotation),
            mirror=bool(mirror),
        )
    except Exception as e:
        raise HTTPException(500, f"retarget failed: {e}")
    summary = _anim_summarize(motion)

    # ---- Encode + stage ----
    try:
        from formats.njm_writer import encode_njm as _encode_njm
        njm_bytes = _encode_njm(motion)
    except Exception as e:
        raise HTTPException(500, f"encode_njm failed: {e}")
    out_name = _safe_motion_name(motion_name)
    NJM_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NJM_EXPORT_DIR / out_name
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(njm_bytes)
    os.replace(tmp, out_path)
    md5 = _md5_bytes(njm_bytes)

    # Write the preview sidecar so /api/anim_preview/list can find this
    # animation when the user opens its target model. The sidecar is the
    # marker that distinguishes "imported preview" entries from staging
    # NJMs that came from other workflows.
    sidecar = {
        "target_model_path": target_bml_name,
        "target_inner": pick_inner.name,
        "source_glb": name,
        "source_animation": imp.animations[0].name,
        "retargeted_at_ms": int(time.time() * 1000),
        "retargeted_bones": int(summary["mapped_bones"]),
        "dropped_bones": int(summary["dropped_bones"]),
        "frame_count": int(summary["frame_count"]),
        "bone_count": int(summary["bone_count"]),
        "fps": int(target_fps),
        "bone_map": chosen_map_name,
        "njm_md5": md5,
    }
    sidecar_path = NJM_EXPORT_DIR / (out_name + ".preview.json")
    tmp_sc = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp_sc.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    os.replace(tmp_sc, sidecar_path)

    log.info(
        "import/animation %s -> %s (target=%s#%s, mapped=%d, dropped=%d, frames=%d, bones=%d, md5=%s)",
        name, out_path, target_bml_name, pick_inner.name,
        summary["mapped_bones"], summary["dropped_bones"],
        summary["frame_count"], summary["bone_count"], md5,
    )
    return {
        "ok": True,
        "njm_path": str(out_path),
        "njm_name": out_name,
        "size": len(njm_bytes),
        "md5": md5,
        "frame_count": summary["frame_count"],
        "bone_count": summary["bone_count"],
        "retargeted_bones": summary["mapped_bones"],
        "dropped_bones": summary["dropped_bones"],
        "dropped": summary["dropped"],
        "bone_map": chosen_map_name,
        "target_bml": target_bml_name,
        "target_inner": pick_inner.name,
        "source_animation": imp.animations[0].name,
        "source_duration_seconds": imp.animations[0].duration_seconds,
        "ik": summary.get("ik", {}),
        "ik_enabled": bool(enable_ik),
        "ik_rotation_enabled": bool(enable_ik_rotation),
        "mirror": summary.get("mirror", {}),
        "mirror_enabled": bool(mirror),
    }


class ImportAnimationReplaceReq(BaseModel):
    njm_path: str
    target_bml: str
    target_motion_name: str
    output_name: Optional[str] = None
    append_if_missing: bool = True


@app.post("/api/import/animation/replace")
def api_import_animation_replace(req: ImportAnimationReplaceReq, request: Request):
    """Splice a staged .njm into a target BML.

    EXPLICIT-OPT-IN ROUTE. As of 2026-04-25 the **default** animation-import
    flow is editor preview-only (see ``/api/import/animation`` and the
    Motions-tab "Imported Animations" section). This route is reserved for
    users who deliberately want to repack a BML with a new motion entry —
    e.g. for a real mod release. The preview flow does NOT call this
    endpoint and does NOT write into ``cache/bml_export/``.

    Replaces the inner entry matching ``target_motion_name`` (when
    present) or appends a new entry (when ``append_if_missing=True``,
    the default). Stages at ``cache/bml_export/<output_name | bml_name>``.
    """
    _enforce_body_size(request, MAX_ANIM_BODY)
    njm_name = _safe_archive_name(Path(req.njm_path).name)
    bml_name = _safe_archive_name(Path(req.target_bml).name)
    target_motion = req.target_motion_name.strip()
    if not target_motion or "/" in target_motion or "\\" in target_motion:
        raise HTTPException(400, "target_motion_name: must be a bare entry name")
    if not target_motion.lower().endswith(".njm"):
        target_motion = target_motion + ".njm"
    njm_p = NJM_EXPORT_DIR / njm_name
    if not njm_p.exists():
        raise HTTPException(404, f"njm_path not found in cache/njm_export: {njm_name}")
    src_bml = LIVE_DATA_DIR / bml_name
    if not src_bml.exists():
        src_bml = DATA_DIR / bml_name
        if not src_bml.exists():
            src_bml = DEV_DATA_DIR / bml_name
            if not src_bml.exists():
                raise HTTPException(404, f"target_bml not found: {bml_name}")
    bml_bytes = src_bml.read_bytes()
    try:
        from formats.bml import parse_bml_for_pack, parse_bml_pack_meta
        pack_entries = parse_bml_for_pack(bml_bytes)
        meta = parse_bml_pack_meta(bml_bytes)
    except Exception as e:
        raise HTTPException(500, f"BML parse failed: {e}")
    new_data = njm_p.read_bytes()
    matched = -1
    for i, ent in enumerate(pack_entries):
        if ent.name == target_motion:
            matched = i
            break
    if matched < 0 and not req.append_if_missing:
        names = [e.name for e in pack_entries]
        raise HTTPException(
            404, f"target_motion_name {target_motion!r} not in BML (and append_if_missing=False); have: {names[:10]}"
        )
    if matched >= 0:
        pack_entries[matched] = BmlPackEntry(
            name=pack_entries[matched].name,
            data=new_data,
            decompressed_size=len(new_data),
            is_compressed=False,
            texture_data=pack_entries[matched].texture_data,
            texture_decompressed_size=pack_entries[matched].texture_decompressed_size,
            texture_is_compressed=pack_entries[matched].texture_is_compressed,
            unk_a=pack_entries[matched].unk_a,
            unk_b=pack_entries[matched].unk_b,
            unk_c=pack_entries[matched].unk_c,
            unk_d=pack_entries[matched].unk_d,
        )
        op = "replaced"
    else:
        pack_entries.append(BmlPackEntry(
            name=target_motion,
            data=new_data,
            decompressed_size=len(new_data),
            is_compressed=False,
            texture_data=None,
            texture_decompressed_size=0,
            texture_is_compressed=False,
            unk_a=0, unk_b=0, unk_c=0, unk_d=0,
        ))
        matched = len(pack_entries) - 1
        op = "appended"
    try:
        out = pack_bml(
            pack_entries,
            compression=meta["compression"],
            has_textures_override=bool(meta.get("has_textures", False)),
            file_alignment=meta["file_alignment"],
        )
    except Exception as e:
        raise HTTPException(500, f"pack_bml failed: {e}")
    out_name = _safe_archive_name(req.output_name or bml_name)
    BML_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BML_EXPORT_DIR / out_name
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(out)
    os.replace(tmp, out_path)
    md5 = _md5_bytes(out)
    log.info(
        "import/animation/replace %s%s%s (%s) -> %s (%d bytes, md5=%s)",
        bml_name, "+" if op == "appended" else "#", target_motion, op,
        out_path, len(out), md5,
    )
    return {
        "ok": True,
        "archive_path": str(out_path),
        "archive_name": out_name,
        "size": len(out),
        "md5": md5,
        "operation": op,
        "target_motion": target_motion,
        "index": matched,
    }


def _nj_vertex_chunk_stride(type_id: int) -> int:
    """Return per-vertex bytes for a vertex chunk type.

    Matches ``_parse_vertex_chunk`` in ``formats/xj.py``. Returns 0 for
    unsupported types (the sculpt path silently skips them).
    """
    if type_id == 32:
        return 12 + 4
    if type_id == 33:
        return 12 + 4 + 12 + 4
    if type_id == 34:
        return 12
    if 35 <= type_id <= 40:
        return 12 + 4
    if type_id == 41:
        return 12 + 12
    if 42 <= type_id <= 47:
        return 12 + 12 + 4
    if type_id == 48:
        return 12 + 4
    if 49 <= type_id <= 50:
        return 12 + 4 + 4
    return 0


class DeployArchiveReq(BaseModel):
    create_backup: bool = True


@app.post("/api/deploy/{archive}")
def api_deploy_archive(archive: str,
                       request: Request,
                       req: Optional[DeployArchiveReq] = None):
    """Deploy a previously-built archive from cache to LIVE_DATA_DIR.

    Looks up ``archive`` in:
      1. cache/afs_export/<archive>
      2. cache/bml_export/<archive>
      3. DEV_DATA_DIR/<archive>  (for any other DEV-mirror file)

    First match wins. Copies to LIVE_DATA_DIR/<archive> with a
    timestamped backup of the prior live bytes (when ``create_backup``).
    Same lock contract as /api/deploy/promote (one deploy at a time).
    """
    _enforce_body_size(request, MAX_BUILD_DEPLOY_BODY)
    create_backup = req.create_backup if req is not None else True

    name = _safe_archive_name(archive)
    # Resolve source.
    src_candidates = [
        AFS_EXPORT_DIR / name,
        BML_EXPORT_DIR / name,
        NJ_EXPORT_DIR / name,
        NJM_EXPORT_DIR / name,
        DEV_DATA_DIR / name,
    ]
    src: Optional[Path] = None
    for cand in src_candidates:
        if cand.exists() and cand.is_file():
            src = cand
            break
    if src is None:
        raise HTTPException(
            404,
            f"deploy: {name!r} not found in cache/afs_export, "
            f"cache/bml_export, cache/nj_export, cache/njm_export, "
            f"or {DEV_DATA_DIR}",
        )
    if not LIVE_DATA_DIR.exists():
        raise HTTPException(500, f"live data dir missing: {LIVE_DATA_DIR}")
    live_p = safe_live_path(name)

    if not _PROMOTE_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another deploy/promote is already running")
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_name: Optional[str] = None
        if create_backup and live_p.exists():
            bak_name = f"{name}.pre_promote_{ts}"
            bak = LIVE_DATA_DIR / bak_name
            counter = 0
            while bak.exists():
                counter += 1
                bak = LIVE_DATA_DIR / f"{bak_name}_{counter}"
            shutil.copy2(live_p, bak)
            backup_name = bak.name
        shutil.copy2(src, live_p)

        size = live_p.stat().st_size
        log.info(
            "deploy_archive %s: %s -> %s (%d bytes, backup=%s)",
            name, src, live_p, size, backup_name,
        )
        return {
            "ok": True,
            "name": name,
            "source": str(src),
            "destination": str(live_p),
            "live_size": size,
            "backup_name": backup_name,
        }
    except (OSError, shutil.Error) as e:
        log.warning("deploy_archive failed for %s: %s", name, e)
        raise HTTPException(500, f"deploy failed: {e}")
    finally:
        _PROMOTE_LOCK.release()


# ============================================================================
# Model preview (2026-04-25)
# ----------------------------------------------------------------------------
# Phase 1 of "show texture on its 3D model".
# Full XJ-format mesh extraction is a non-trivial RE problem (see
# MODEL_PREVIEW_RESEARCH.md). This endpoint instead provides a *primitive*
# preview: it returns a hint describing which built-in shape (sphere, cube,
# plane, cylinder) best fits the texture, so the frontend can render the
# live texture on a rotating 3D primitive in three.js.
#
# When real XJ parsing lands, we add a `geometry_url` field that points at
# /static/models/<name>.glb and the frontend swaps the primitive for the
# real mesh.
# ============================================================================


def _model_preview_hint(filename: str) -> dict:
    """Return a primitive-shape hint for a texture file based on its name.

    The hint is best-effort: it picks a shape that's *plausible* for the
    texture's role. The frontend defaults to the suggested shape but lets
    the user cycle through all four primitives.
    """
    name = filename.lower()
    # Strip extension
    stem = name
    for ext in (".prs", ".xvm"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    # Known external pair: bm_obj_ep4_boss09_core_tex.xvm <-> bm_obj_ep4_boss09_core.bml
    # Search both DATA_DIR and LIVE_DATA_DIR so the user can preview
    # texture files paired with models that aren't mirrored into the
    # dev tree.
    bml_basename = None
    if stem.endswith("_tex"):
        candidate = stem[:-4] + ".bml"
        for root in (DATA_DIR, LIVE_DATA_DIR):
            if (root / candidate).exists():
                bml_basename = candidate
                break

    # Player textures (plAtex.afs etc.) pair with `pl[A-X]bdy00.nj`. We
    # don't currently expose AFS slot mapping in the preview endpoint
    # (it depends on which tile / slot the user is viewing), but if a
    # paired body `.nj` exists at the predictable name we surface it as
    # `model_archive` so the frontend's real-mesh path can pick it up.
    if not bml_basename:
        # plAtex.afs -> plAbdy00.nj
        m_pl = _PLAYER_TEX_STEM_RE.match(stem)
        if m_pl:
            class_letter = m_pl.group(1).upper()
            candidate_nj = f"pl{class_letter}bdy00.nj"
            for root in (DATA_DIR, LIVE_DATA_DIR):
                if (root / candidate_nj).exists():
                    bml_basename = candidate_nj
                    break

    # Heuristic shape by name pattern
    if stem.startswith("map_"):
        # Maps are usually broad terrain - flat plane is closest preview
        shape = "plane"
        why = "map texture (flat terrain plane)"
    elif "logo" in stem or "title" in stem or stem.startswith("f128") or stem.startswith("f256") or stem.startswith("f512"):
        # UI / logo / font - flat plane / billboard
        shape = "plane"
        why = "UI / logo (flat billboard)"
    elif "boss09_core" in stem or stem.endswith("_core"):
        # The Saint-Million / Olga Flow core is a sphere
        shape = "sphere"
        why = "boss core (sphere)"
    elif "container" in stem or "hako" in stem or "_box" in stem or "iwa" in stem or "kabe" in stem:
        # Container, box, rock, wall = boxy (check before "ball" / "_o_" rules)
        shape = "cube"
        why = "boxy object (cube)"
    elif "ball" in stem or stem.endswith("_o") or "_obj_o_" in stem:
        # "_obj_o_" prefix is "object o-something" naming, often spherical
        shape = "sphere"
        why = "round object (sphere)"
    elif "door" in stem or "billboard" in stem:
        shape = "plane"
        why = "flat surface (plane)"
    elif "warp" in stem or "saka" in stem or "tube" in stem:
        shape = "cylinder"
        why = "cylindrical (cylinder)"
    elif stem.startswith("bm_") or stem.startswith("fe_") or stem.startswith("fs_"):
        # Generic models: cube is most forgiving
        shape = "cube"
        why = "generic 3D object (cube)"
    else:
        shape = "plane"
        why = "default (flat preview)"

    return {
        "shape": shape,
        "why": why,
        "available_shapes": ["sphere", "cube", "plane", "cylinder"],
        "model_archive": bml_basename,
        "model_extraction_status": (
            "paired model found; the frontend will attempt real XJ mesh extraction"
            if bml_basename
            else "no external model archive in data dir (may be embedded inside a BML or in a .rel)"
        ),
    }


@app.get("/api/model_preview/{filename}")
def api_model_preview(filename: str):
    """Return a 3D-preview hint for a texture file.

    Response shape:
      {
        "filename": "<name>",
        "shape": "sphere"|"cube"|"plane"|"cylinder",
        "why": "<one-line explanation>",
        "available_shapes": [...],
        "model_archive": "<name>.bml" | null,
        "model_extraction_status": "<message>",
        "tile_count": <int>,
        "first_tile": {"index": 0, "width": W, "height": H} | null
      }

    The frontend uses `shape` as the default geometry and `tile_count` /
    `first_tile` to pre-select a sensible texture (tile 0 by default). When
    `model_archive` is non-null the UI shows a "real model coming soon"
    affordance so the user knows we know which BML it pairs with.

    Accepts the BML-inner ``<base>#<inner>`` syntax in addition to plain
    filenames; the inner blob is materialized + tile-extracted via the
    standard pipeline so the preview reports a real ``tile_count``.
    """
    try:
        p = _materialize_inner_for_extract(filename)
    except HTTPException:
        raise
    if not p.exists():
        raise HTTPException(404, f"file not found: {filename}")
    # The shape-hint pipeline keys on the BASE filename (it sniffs sibling
    # BMLs in DATA_DIR), so the BML-inner form should resolve to the BML
    # itself for hint purposes — the inner part is only used for tiles.
    base, _inner = _split_inner_path(filename)
    hint = _model_preview_hint(base)
    # Add tile count + first-tile dim for the frontend's default texture pick
    tile_count = 0
    first_tile = None
    try:
        m = extract_tiles(p)
        tile_count = m.get("tile_count", 0)
        if tile_count > 0:
            t0 = m["tiles"][0]
            first_tile = {
                "index": t0["index"],
                "width": t0["width"],
                "height": t0["height"],
            }
    except Exception:
        # Don't fail the preview just because tile extraction stumbles -
        # the frontend can still render the primitive without a texture.
        pass
    hint["filename"] = filename
    hint["tile_count"] = tile_count
    hint["first_tile"] = first_tile
    return hint


# ---------------------------------------------------------------------------- model mesh extraction
#
# `/api/model_mesh/{path}` returns a compact JSON payload of triangulated
# meshes parsed from a PSOBB `.nj` (XJ format) file. The frontend uses
# this to render REAL geometry in three.js (vs the legacy primitive
# fallback) when an `.nj` is available.
#
# Path resolution mirrors the BML endpoints: DATA_DIR first, then
# LIVE_DATA_DIR. For meshes that live INSIDE a `.bml` archive, callers
# pass the BML path with `?inner=<entry_name>` and we extract the inner
# `.nj` via `formats.bml.extract_bml`.
#
# Wire shape (all little-endian Float32 for vertices, Uint32 for indices):
#   {
#     "filename": "<input>",
#     "inner": "<inner-name or null>",
#     "mesh_count": <int>,
#     "vertices_pre_transformed": true,
#     "meshes": [
#       {
#         "vertices_b64": "<base64 of interleaved [px,py,pz, nx,ny,nz, u,v]>",
#         "indices_b64":  "<base64 of [i0,i1,i2, ...]>",
#         "vertex_count": <int>,
#         "triangle_count": <int>,
#         "material_id": <int>,
#         "bounding_sphere": [cx, cy, cz, r],
#         "aabb": [minx, miny, minz, maxx, maxy, maxz],
#         "world_position": [wx, wy, wz],
#         "world_rotation_euler": [rx, ry, rz],
#         "world_scale": [sx, sy, sz],
#         "world_matrix": [m00, m01, ..., m33]
#       },
#       ...
#     ],
#     "totals": {"vertices": <int>, "triangles": <int>}
#   }
#
# vertices_pre_transformed (added 2026-04-24): when true, the strip
# vertices are already in world space (the parser baked the
# MeshTreeNode bone-tree into them). The per-mesh ``world_*`` fields
# are diagnostic only — the frontend MUST NOT compose them into
# ``Object3D.position`` or every submesh will be doubly offset. The
# field exists to fix the "model is exploded shards" bug where naive
# parsers ignored bone transforms.


def _resolve_model_mesh_path(path: str) -> Path:
    """Resolve a model-mesh path under DATA_DIR or LIVE_DATA_DIR.

    Same injection guard as `_resolve_bml_path` but accepts ``.nj``,
    ``.bml`` and ``.afs`` extensions (the latter two forward to the
    inner-extract code path in the caller).

    Also checks ``CACHE_DIR / "subdivided"`` as a third source so the
    "Subdivide model" panel's output can be re-loaded via the same
    /api/model_mesh endpoint without polluting DATA_DIR.
    """
    subdiv_dir = (CACHE_DIR / "subdivided").resolve()
    return _resolve_under_roots(
        path,
        (DATA_DIR, LIVE_DATA_DIR, subdiv_dir),
        label="path",
        missing_msg=f"model not found in DATA_DIR or LIVE_DATA_DIR: {path}",
    )


def _xj_meshes_to_payload(meshes: list) -> dict:
    """Project a list[XjMesh] to the JSON wire shape documented above.

    Vertex data is packed into an interleaved Float32 array with the
    layout (px, py, pz, nx, ny, nz, u, v, r, g, b, a) per vertex — 12
    floats. The trailing RGBA (0..1) is PSOBB's authored per-vertex /
    material-diffuse color, used by the frontend for unlit × vertexColor
    rendering (``has_color: true`` in the returned dict). Indices are
    Uint32 (we widen from u16 strips to allow >65k vertices in future
    files; current files don't need it but the pad is cheap).

    Each mesh entry also carries the per-submesh transform tagging
    added 2026-04-24:

        ``world_position``       3-tuple (cx, cy, cz) — the strip's
                                 world-space AABB centre.
        ``world_rotation_euler`` 3-tuple Euler-XYZ radians (always 0,0,0
                                 — see the comment in
                                 ``formats/xj.py::_emit_strip_mesh``).
        ``world_scale``          3-tuple (always 1,1,1 for the same
                                 reason).
        ``world_matrix``         16-float row-major identity.

    Plus a payload-level ``vertices_pre_transformed: true`` flag that
    tells the frontend the strip vertices are ALREADY in world space
    and the per-mesh transforms above are diagnostic only — applying
    them as ``Object3D.position`` would double-offset every submesh.
    """
    import array

    payload_meshes: list[dict] = []
    total_v = 0
    total_t = 0
    for m in meshes:
        # Interleaved Float32 array. 12 floats per vertex:
        #   [px,py,pz, nx,ny,nz, u,v, r,g,b,a]
        # The trailing RGBA (0..1) carries PSOBB's authored per-vertex /
        # diffuse color so the frontend can render UNLIT × vertexColor
        # (psov2 parity). Always emitted (default white) — the payload
        # advertises ``has_color: true`` so older frontends can branch.
        floats = array.array("f")
        minx = miny = minz = float("inf")
        maxx = maxy = maxz = float("-inf")
        for v in m.vertices:
            px, py, pz = v.pos
            nx, ny, nz = v.normal
            u, vv = v.uv
            cr, cg, cb, ca = v.color
            floats.extend((px, py, pz, nx, ny, nz, u, vv, cr, cg, cb, ca))
            if px < minx:
                minx = px
            if py < miny:
                miny = py
            if pz < minz:
                minz = pz
            if px > maxx:
                maxx = px
            if py > maxy:
                maxy = py
            if pz > maxz:
                maxz = pz
        # Force little-endian on disk regardless of host (the JS side
        # always treats the buffer as LE Float32).
        if sys.byteorder != "little":
            floats.byteswap()
        verts_bytes = floats.tobytes()

        # Uint32 indices
        idx_arr = array.array("I", m.indices)
        if sys.byteorder != "little":
            idx_arr.byteswap()
        idx_bytes = idx_arr.tobytes()

        if not m.vertices:
            aabb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            aabb = [minx, miny, minz, maxx, maxy, maxz]

        vc = len(m.vertices)
        tc = len(m.indices) // 3
        total_v += vc
        total_t += tc
        payload_meshes.append({
            "vertices_b64": base64.b64encode(verts_bytes).decode("ascii"),
            "indices_b64": base64.b64encode(idx_bytes).decode("ascii"),
            "vertex_count": vc,
            "triangle_count": tc,
            "material_id": m.material_id,
            "bounding_sphere": list(m.bounding_sphere),
            "aabb": aabb,
            "world_position": list(m.world_position),
            "world_rotation_euler": list(m.world_rotation_euler),
            "world_scale": list(m.world_scale),
            "world_matrix": list(m.world_matrix),
            # Per-submesh render-state flags (Phase 3, 2026-06-20). The
            # frontend maps blend_mode "additive" -> AdditiveBlending +
            # depthWrite:false; alpha_test -> transparent + alphaTest;
            # two_sided -> DoubleSide. ``getattr`` keeps older pickled
            # XjMesh objects (pre-v2 disk cache) safe.
            "blend_mode": getattr(m, "blend_mode", "none"),
            "two_sided": bool(getattr(m, "two_sided", False)),
            "alpha_test": getattr(m, "alpha_test", None),
            "alpha_blend": getattr(m, "alpha_blend", None),
        })

    return {
        "mesh_count": len(payload_meshes),
        "meshes": payload_meshes,
        "totals": {"vertices": total_v, "triangles": total_t},
        # Tells the frontend that ``meshes[i].vertices_b64`` already
        # holds world-space coordinates (the parser baked the bone
        # tree into them). The per-mesh ``world_position`` etc. fields
        # are diagnostic — DO NOT compose them into Object3D.position
        # or every submesh will be doubly-offset.
        "vertices_pre_transformed": True,
        # Vertex interleave now carries 4 trailing RGBA color floats
        # (12 floats/vertex). ``has_color`` lets the frontend pick the
        # stride safely; ``vertex_format_version`` bumps from the
        # implicit v1 (8 floats) to v2 (12 floats) so any cached/older
        # consumer can detect the shape change.
        "has_color": True,
        "vertex_format_version": 2,
        # Convenience aliases (flat) used by the ad-hoc smoke
        # commands in AGENT_XJ_FAITHFUL_PORT_REPORT.md and other
        # CLI reports. The structured form above is the authoritative
        # one; these mirror it 1:1.
        "vert_total": total_v,
        "tri_total": total_t,
    }


def _xj_meshes_to_skinned_payload(meshes: list, bones: list) -> dict:
    """Project a (list[XjMesh], list[XjBone]) pair to the skinned wire shape.

    Used by ``/api/model_skinned`` — the animation-friendly counterpart
    of ``_xj_meshes_to_payload``. Differs in two ways:

      1. Vertices are in BONE-LOCAL coordinates (not world-baked) and
         carry a per-vertex ``bone_idx`` (Int32) identifying the
         owning bone in the ``bones`` array.
      2. The skeleton is included verbatim — flat DFS order, each bone
         with parent index, bind-pose translation, BAMS rotation, and
         (always 1.0) scale. The frontend uses this to compose
         per-bone bind-pose matrices, then combines them with NJM
         keyframes to produce animated bone matrices.

    Wire-format flags:
      ``vertices_pre_transformed: false`` — opposite of the regular
        payload; tells the frontend to apply bone matrices.
      ``has_bone_indices: true`` — every XjVertex carries a valid
        bone_idx (or -1 if the strip pulled from a slot we didn't
        track; consumers fall back to bone 0 / identity in that case).

    Vertex layout (per the regular payload's interleaved format):
      Float32: [px, py, pz, nx, ny, nz, u, v, r, g, b, a]  (12 floats/vertex)
      Int32:   [bone_idx]                                   (separate buffer)

    Performance (Phase 0.5 perf, 2026-04-25): the inner loop uses
    ``array.array`` (a C-level packed buffer) extended one 8-tuple at
    a time, with a simple per-vertex min/max for the AABB. We tested a
    pure-numpy variant for this path and it ran ~3x SLOWER on dragon —
    PSOBB models have ~1069 submeshes with mean 9 verts each, and
    numpy's per-call overhead dominates that small-submesh shape. The
    array.array path runs at ~24-33 ms for dragon's 9677 verts. The
    on-disk JSON cache further drops repeat opens to <5 ms.
    """
    import array

    payload_meshes: list[dict] = []
    total_v = 0
    total_t = 0
    is_le = (sys.byteorder == "little")
    for m in meshes:
        verts = m.vertices
        vc = len(verts)
        tc = len(m.indices) // 3

        if vc == 0:
            # Empty submesh — emit empty buffers but keep the dict
            # shape so downstream code doesn't have to special-case.
            verts_bytes = b""
            bone_idx_bytes = b""
            aabb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            # array.array.extend on an 8-tuple drops into a C-level
            # buffer copy per vertex — much faster than np.empty +
            # element-wise assignment for the small-submesh shape
            # PSOBB models exhibit (~10 verts/submesh).
            floats = array.array("f")
            bone_idx_arr = array.array("i")
            minx = miny = minz = float("inf")
            maxx = maxy = maxz = float("-inf")
            for v in verts:
                px, py, pz = v.pos
                nx, ny, nz = v.normal
                u, vv = v.uv
                cr, cg, cb, ca = v.color
                floats.extend((px, py, pz, nx, ny, nz, u, vv, cr, cg, cb, ca))
                bone_idx_arr.append(v.bone_idx)
                if px < minx:
                    minx = px
                if py < miny:
                    miny = py
                if pz < minz:
                    minz = pz
                if px > maxx:
                    maxx = px
                if py > maxy:
                    maxy = py
                if pz > maxz:
                    maxz = pz
            aabb = [minx, miny, minz, maxx, maxy, maxz]
            # Force little-endian on disk regardless of host (the JS
            # side always treats the buffer as LE Float32 / Int32).
            if not is_le:
                floats.byteswap()
                bone_idx_arr.byteswap()
            verts_bytes = floats.tobytes()
            bone_idx_bytes = bone_idx_arr.tobytes()

        if m.indices:
            idx_arr = array.array("I", m.indices)
            if not is_le:
                idx_arr.byteswap()
            idx_bytes = idx_arr.tobytes()
        else:
            idx_bytes = b""

        total_v += vc
        total_t += tc
        payload_meshes.append({
            "vertices_b64": base64.b64encode(verts_bytes).decode("ascii"),
            "indices_b64": base64.b64encode(idx_bytes).decode("ascii"),
            "bone_indices_b64": base64.b64encode(bone_idx_bytes).decode("ascii"),
            "vertex_count": vc,
            "triangle_count": tc,
            "material_id": m.material_id,
            "bounding_sphere": list(m.bounding_sphere),
            "aabb": aabb,
            # Per-submesh render-state flags (Phase 3, 2026-06-20) — same
            # shape as the static payload; ``getattr`` keeps older pickled
            # XjMesh objects (pre-v2 skinned disk cache) safe.
            "blend_mode": getattr(m, "blend_mode", "none"),
            "two_sided": bool(getattr(m, "two_sided", False)),
            "alpha_test": getattr(m, "alpha_test", None),
            "alpha_blend": getattr(m, "alpha_blend", None),
        })

    bones_out: list[dict] = [
        {
            "index": b.index,
            "parent": b.parent,
            "position": list(b.position),
            "rotation_bams": list(b.rotation),
            # 2026-04-25: surface eval_flags + scale so the JS bind-pose
            # composition matches the world-baked pipeline. UNIT_POS,
            # UNIT_ANG, UNIT_SCL, SKIP and ZXY_ANG must be honored when
            # composing the bind matrix; without them the skinned path
            # diverges from /api/model_mesh on models like De Rol Le
            # whose head bones use UNIT_POS|UNIT_SCL.
            "scale": list(b.scale),
            "eval_flags": int(b.eval_flags),
        }
        for b in bones
    ]

    return {
        "mesh_count": len(payload_meshes),
        "meshes": payload_meshes,
        "bones": bones_out,
        "bone_count": len(bones_out),
        "totals": {"vertices": total_v, "triangles": total_t},
        "vertices_pre_transformed": False,
        "has_bone_indices": True,
        # Same 12-float interleave (+RGBA) as the static payload.
        "has_color": True,
        "vertex_format_version": 2,
        "vert_total": total_v,
        "tri_total": total_t,
    }


# ---------------------------------------------------------------------------
# Skinned-payload LRU + on-disk cache (Phase 0.5 perf, 2026-04-25)
# ---------------------------------------------------------------------------
# Caches the OUTPUT dict of `_xj_meshes_to_skinned_payload` keyed on the
# same (path, mtime_ns, size, inner) tuple parse_cache uses. Composes
# under both parse_cache and the binding cache:
#
#   parse_cache._PARSE_CACHE        — parsed XjMesh / XjBone lists (256 MB)
#   THIS LAYER (_SKINNED_PAYLOAD)   — skinned wire dict (128 MB / 256)
#   server._BINDING_CACHE           — NJTL→XVMH binding dicts (32 MB / 256)
#
# The disk tier persists payloads as JSON (the dict is already JSON
# shaped — b64 strings + small dicts — so json.dump is faster than
# pickle for this shape and the file is human-readable for debugging).
# Atomic-rename on write so kill-9 leaves either the previous payload
# or no entry, never a half-JSON we'd subsequently fail to decode.
#
# Eviction: LRU by total bytes OR entry count (whichever cap fires
# first), with one-entry-minimum so a single oversize payload doesn't
# trigger an infinite eviction loop.
# Invalidation: implicit via mtime_ns in the key — re-deploys land in a
# newer mtime so the next call computes a fresh entry.


def _skinned_payload_profile_enabled() -> bool:
    """``PSO_PROFILE=1`` env-var gate for the cProfile dump.

    Active only when the environment variable is set; default is OFF so
    production loads pay zero cProfile overhead. When ON, the cold-
    compute branch of ``_xj_meshes_to_skinned_payload_cached`` runs
    inside a cProfile.Profile and dumps the cumulative-time top entries
    to the server log (and a one-line marker to stderr for tail-grep).
    """
    return os.environ.get("PSO_PROFILE", "0") in ("1", "true", "True")


def _profile_skinned_compute(meshes: list, bones: list) -> dict:
    """Run `_xj_meshes_to_skinned_payload` under cProfile and log top-25.

    Used only when PSO_PROFILE=1 — the wrapper picks this path on cold
    miss so we always profile the actually-expensive case (warm hits
    don't enter here). Profile output goes to the standard ``log`` so
    it lands in server.log next to the rest of the run trace.
    """
    import cProfile
    import io
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    try:
        return _xj_meshes_to_skinned_payload(meshes, bones)
    finally:
        pr.disable()
        try:
            buf = io.StringIO()
            pstats.Stats(pr, stream=buf).strip_dirs() \
                .sort_stats("cumulative").print_stats(25)
            log.warning("PSO_PROFILE skinned_payload cold compute:\n%s",
                        buf.getvalue())
        except Exception:  # pragma: no cover — diagnostic-only path
            log.exception("PSO_PROFILE: failed to dump cProfile stats")

_SKINNED_PAYLOAD_CACHE_MAX_ENTRIES = int(
    os.environ.get("PSO_SKINNED_PAYLOAD_CACHE_ENTRIES", "256"),
)
_SKINNED_PAYLOAD_CACHE_MAX_BYTES = int(
    os.environ.get("PSO_SKINNED_PAYLOAD_CACHE_MB", "128"),
) * 1024 * 1024

_SKINNED_PAYLOAD_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_SKINNED_PAYLOAD_CACHE_LOCK = threading.Lock()
_SKINNED_PAYLOAD_CACHE_BYTES = 0
_SKINNED_PAYLOAD_HITS_INMEMORY = 0
_SKINNED_PAYLOAD_HITS_DISK = 0
_SKINNED_PAYLOAD_MISSES = 0


def _skinned_payload_cache_key(
    p: Path,
    inner_name: Optional[str],
) -> Optional[tuple]:
    """Build the LRU+disk key for one skinned-payload request.

    Shape: ``("skinned_payload", abs_path_str, mtime_ns, size, inner_name)``
    matching the parse_cache file_key style — adding parser_id as a
    leading marker so a bug that mixes keys with parse_cache (different
    sha2) gets a clean miss rather than a wrong hit.

    Returns None on stat failure; the caller falls through to the
    uncached compute (correct, just slow on disconnected drives).
    """
    try:
        st = p.stat()
    except OSError:
        return None
    return (
        "skinned_payload",
        str(p),
        int(st.st_mtime_ns),
        int(st.st_size),
        inner_name or "",
    )


def _skinned_payload_disk_path(key: tuple) -> Optional[Path]:
    """Compute on-disk JSON path for a cache key, creating the schema dir.

    Returns None on disk-disable / dir-creation failure (graceful
    degradation — in-memory cache still works).
    """
    if os.environ.get("PSO_DISABLE_DISK_SKINNED_PAYLOAD_CACHE", "0") in ("1", "true", "True"):
        return None
    try:
        base = SKINNED_PAYLOAD_CACHE_DIR / f"v{SKINNED_PAYLOAD_CACHE_SCHEMA}"
        base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("skinned_payload_cache: dir creation failed: %s", e)
        return None
    h = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return base / f"{h}.json"


def _skinned_payload_purge_until_under_caps_locked() -> None:
    """Evict LRU entries until both caps satisfied; always keep one entry."""
    global _SKINNED_PAYLOAD_CACHE_BYTES
    while ((_SKINNED_PAYLOAD_CACHE_BYTES > _SKINNED_PAYLOAD_CACHE_MAX_BYTES
            or len(_SKINNED_PAYLOAD_CACHE) > _SKINNED_PAYLOAD_CACHE_MAX_ENTRIES)
           and len(_SKINNED_PAYLOAD_CACHE) > 1):
        try:
            _evicted_key, value = _SKINNED_PAYLOAD_CACHE.popitem(last=False)
        except KeyError:
            break
        _SKINNED_PAYLOAD_CACHE_BYTES -= int(value[1])


def _skinned_payload_load_from_disk(key: tuple) -> Optional[Tuple[dict, int]]:
    """Read a cached skinned payload from disk; None on miss/corrupt.

    The on-disk file is JSON of the form ``{"key": [...], "payload":
    {...}}`` — we re-verify the embedded key matches before serving so
    a sha-collision (vanishingly unlikely but plausibly possible if the
    schema bumps and we forget to bump the version dir) can't silently
    serve stale bytes.
    """
    p = _skinned_payload_disk_path(key)
    if p is None or not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            raw = f.read()
        obj = json.loads(raw)
    except (OSError, ValueError) as e:
        log.warning("skinned_payload_cache: corrupt JSON %s removed: %s",
                    p.name, e)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    stored_key = obj.get("key") if isinstance(obj, dict) else None
    payload = obj.get("payload") if isinstance(obj, dict) else None
    if stored_key is None or payload is None:
        log.warning("skinned_payload_cache: malformed JSON shape at %s; deleting",
                    p.name)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    # Re-verify the key matches — handles schema drift gracefully.
    # JSON round-trips tuples as lists, so we tuple-ise both sides.
    if tuple(stored_key) != tuple(key):
        log.warning("skinned_payload_cache: key mismatch at %s; deleting",
                    p.name)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    try:
        size = p.stat().st_size
    except OSError:
        size = len(raw)
    return payload, int(size)


def _skinned_payload_write_to_disk(key: tuple, payload: dict, size_hint: int) -> None:
    """Persist the payload to disk via tmp+rename; silent no-op on errors.

    Atomic rename keeps a kill mid-write from leaving a half-JSON we'd
    later fail to decode. We pass ``size_hint`` only to short-circuit
    truly enormous payloads — the 256 MB cap above already filters most
    of them, and a single dragon weighs ~2-4 MB of b64 + JSON overhead.
    """
    p = _skinned_payload_disk_path(key)
    if p is None:
        return
    tmp = p.with_suffix(".json.tmp")
    try:
        # Encode once via json.dumps so we control bytes-on-disk size
        # exactly (separators=',' compresses out 1-2 % whitespace).
        body = json.dumps(
            {"key": list(key), "payload": payload},
            separators=(",", ":"),
        )
        with tmp.open("w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, p)
    except (OSError, ValueError, TypeError) as e:
        log.warning("skinned_payload_cache: write failed for %s: %s",
                    p.name, e)
        try:
            tmp.unlink()
        except OSError:
            pass


def _skinned_payload_estimate_bytes(payload: dict) -> int:
    """Cheap upper bound on payload size for eviction accounting.

    The dominant component is ``meshes[i].vertices_b64`` — base64 of the
    Float32 buffer, ~1.33 × the underlying byte count. Summing those
    gives a tight estimate without re-encoding the whole dict.
    """
    total = 1024  # fixed overhead (meta keys + small dicts)
    for m in payload.get("meshes", []) or []:
        total += len(m.get("vertices_b64") or "")
        total += len(m.get("indices_b64") or "")
        total += len(m.get("bone_indices_b64") or "")
    # bones_out is small — assume 200 bytes per bone is a safe ceiling.
    total += 200 * len(payload.get("bones") or [])
    return total


def _xj_meshes_to_skinned_payload_cached(
    meshes: list,
    bones: list,
    p: Path,
    inner_name: Optional[str],
) -> dict:
    """LRU+disk-cached variant of `_xj_meshes_to_skinned_payload`.

    Lookup order:
      1. In-memory LRU keyed on (parser_id, path, mtime_ns, size, inner).
      2. On-disk JSON under ``cache/skinned_payload/v<schema>/``.
      3. Cold compute via ``_xj_meshes_to_skinned_payload``.

    Cache hit returns the SAME dict object (callers must not mutate).
    The /api/model_skinned path attaches ``filename`` / ``inner`` /
    ``binding_data`` to the returned dict — those are computed AFTER
    this function and would corrupt the cache if they landed in here.
    The wrapper returns the geometry+skeleton portion ONLY; the route
    handler does shallow-copy + add-fields when appropriate.

    Falls through to the uncached compute when the source file's stat()
    fails — keeps responses correct on disconnected drives / live
    deletes at the cost of a ~12-30 ms recompute per call.
    """
    global _SKINNED_PAYLOAD_CACHE_BYTES
    global _SKINNED_PAYLOAD_HITS_INMEMORY, _SKINNED_PAYLOAD_HITS_DISK
    global _SKINNED_PAYLOAD_MISSES

    key = _skinned_payload_cache_key(p, inner_name)
    if key is None:
        return _xj_meshes_to_skinned_payload(meshes, bones)

    # --- L1: in-memory LRU
    with _SKINNED_PAYLOAD_CACHE_LOCK:
        ent = _SKINNED_PAYLOAD_CACHE.get(key)
        if ent is not None:
            _SKINNED_PAYLOAD_CACHE.move_to_end(key)
            ent[2] = ent[2] + 1                    # bump hit count
            _SKINNED_PAYLOAD_HITS_INMEMORY += 1
            return ent[0]

    # --- L2: on-disk JSON
    disk_hit = _skinned_payload_load_from_disk(key)
    if disk_hit is not None:
        payload, byte_estimate = disk_hit
        with _SKINNED_PAYLOAD_CACHE_LOCK:
            ent = _SKINNED_PAYLOAD_CACHE.get(key)
            if ent is None:
                _SKINNED_PAYLOAD_CACHE[key] = [payload, byte_estimate, 1]
                _SKINNED_PAYLOAD_CACHE_BYTES += byte_estimate
                _skinned_payload_purge_until_under_caps_locked()
                _SKINNED_PAYLOAD_HITS_DISK += 1
            else:
                # Race: another caller landed first.
                _SKINNED_PAYLOAD_CACHE.move_to_end(key)
                ent[2] += 1
                payload = ent[0]
                _SKINNED_PAYLOAD_HITS_INMEMORY += 1
        return payload

    # --- Cold compute (optionally profiled when PSO_PROFILE=1)
    if _skinned_payload_profile_enabled():
        payload = _profile_skinned_compute(meshes, bones)
    else:
        payload = _xj_meshes_to_skinned_payload(meshes, bones)
    byte_estimate = _skinned_payload_estimate_bytes(payload)

    with _SKINNED_PAYLOAD_CACHE_LOCK:
        # Race-safe re-check before insert.
        ent = _SKINNED_PAYLOAD_CACHE.get(key)
        if ent is None:
            _SKINNED_PAYLOAD_CACHE[key] = [payload, byte_estimate, 1]
            _SKINNED_PAYLOAD_CACHE_BYTES += byte_estimate
            _skinned_payload_purge_until_under_caps_locked()
            _SKINNED_PAYLOAD_MISSES += 1
        else:
            _SKINNED_PAYLOAD_CACHE.move_to_end(key)
            ent[2] += 1
            payload = ent[0]
            _SKINNED_PAYLOAD_HITS_INMEMORY += 1

    # Disk persist (outside the lock — disk I/O can block other readers
    # otherwise). Even though our payload is a JSON-shaped dict, we
    # serialise here from the in-memory dict to keep the on-disk format
    # readable by `cat` for diagnostic purposes.
    _skinned_payload_write_to_disk(key, payload, byte_estimate)
    return payload


def _skinned_payload_cache_stats() -> dict:
    """Return skinned-payload cache health for /api/skinned_payload_cache/stats.

    Same shape skeleton as parse_cache.cache_stats / binding_cache_stats /
    tile_png cache stats so the frontend can render all four with one
    widget.
    """
    with _SKINNED_PAYLOAD_CACHE_LOCK:
        entries = len(_SKINNED_PAYLOAD_CACHE)
        total = _SKINNED_PAYLOAD_CACHE_BYTES
        hits_mem = _SKINNED_PAYLOAD_HITS_INMEMORY
        hits_disk = _SKINNED_PAYLOAD_HITS_DISK
        misses = _SKINNED_PAYLOAD_MISSES
        # Top-10 by hit count for debug.
        top: list = []
        for k, v in sorted(_SKINNED_PAYLOAD_CACHE.items(),
                           key=lambda kv: kv[1][2], reverse=True)[:10]:
            # k = ("skinned_payload", path, mtime, size, inner)
            path_str = str(k[1]) if len(k) > 1 else ""
            basename = path_str.replace("\\", "/").rsplit("/", 1)[-1]
            inner_str = str(k[4]) if len(k) > 4 else ""
            top.append({
                "key": basename + ((":" + inner_str) if inner_str else ""),
                "hits": int(v[2]),
                "bytes": int(v[1]),
            })

    # Disk usage — outside the lock; just stat-walks.
    disk_entries: Optional[int] = None
    disk_bytes: Optional[int] = None
    try:
        base = SKINNED_PAYLOAD_CACHE_DIR / f"v{SKINNED_PAYLOAD_CACHE_SCHEMA}"
        if base.is_dir():
            disk_entries = 0
            disk_bytes = 0
            for child in base.iterdir():
                if child.is_file() and child.suffix == ".json":
                    disk_entries += 1
                    try:
                        disk_bytes += child.stat().st_size
                    except OSError:
                        pass
    except OSError:
        pass

    total_calls = hits_mem + hits_disk + misses
    hit_rate = (hits_mem + hits_disk) / total_calls if total_calls else 0.0
    return {
        "entries": entries,
        "bytes": total,
        "max_entries": _SKINNED_PAYLOAD_CACHE_MAX_ENTRIES,
        "max_bytes": _SKINNED_PAYLOAD_CACHE_MAX_BYTES,
        "disk_entries": disk_entries,
        "disk_bytes": disk_bytes,
        "hits_inmemory": hits_mem,
        "hits_disk": hits_disk,
        "misses": misses,
        "hit_rate": hit_rate,
        "top_entries": top,
        "schema": SKINNED_PAYLOAD_CACHE_SCHEMA,
    }


def _skinned_payload_cache_clear(*, drop_disk: bool = True) -> dict:
    """Drop the skinned-payload cache (in-memory + on-disk)."""
    global _SKINNED_PAYLOAD_CACHE_BYTES
    global _SKINNED_PAYLOAD_HITS_INMEMORY, _SKINNED_PAYLOAD_HITS_DISK
    global _SKINNED_PAYLOAD_MISSES
    with _SKINNED_PAYLOAD_CACHE_LOCK:
        cleared_entries = len(_SKINNED_PAYLOAD_CACHE)
        cleared_bytes = _SKINNED_PAYLOAD_CACHE_BYTES
        _SKINNED_PAYLOAD_CACHE.clear()
        _SKINNED_PAYLOAD_CACHE_BYTES = 0
        _SKINNED_PAYLOAD_HITS_INMEMORY = 0
        _SKINNED_PAYLOAD_HITS_DISK = 0
        _SKINNED_PAYLOAD_MISSES = 0

    disk_files = 0
    disk_bytes_freed = 0
    if drop_disk:
        try:
            base = SKINNED_PAYLOAD_CACHE_DIR / f"v{SKINNED_PAYLOAD_CACHE_SCHEMA}"
            if base.is_dir():
                for child in base.iterdir():
                    if child.is_file() and child.suffix in (".json", ".tmp"):
                        try:
                            sz = child.stat().st_size
                        except OSError:
                            sz = 0
                        try:
                            child.unlink()
                            disk_files += 1
                            disk_bytes_freed += sz
                        except OSError:
                            pass
        except OSError:
            pass

    return {
        "cleared_entries": cleared_entries,
        "cleared_bytes": cleared_bytes,
        "cleared_disk_files": disk_files,
        "cleared_disk_bytes": disk_bytes_freed,
    }


def _read_inner_nj_from_bml(p: Path, inner_name: str) -> bytes:
    """Read & decompress a single inner `.nj` from a BML archive.

    Uses ``formats.bml.parse_bml`` + ``decompress_prs_cached`` so repeated
    opens of the same BML inner are O(1) (the LRU cache holds 64 MB of
    decompressed inner blobs). PRS itself is decoded in-process by
    default — see ``formats.bml`` for the env-var fallback to PuyoToolsCli.
    """
    # Single stat call — was previously stat()ing twice (once for the
    # size guard, again for st_mtime_ns when building the cache key).
    st = p.stat()
    sz = st.st_size
    if sz > ASSET_PARSE_MAX_BYTES:
        raise HTTPException(
            413,
            f"BML too large to parse in-memory: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
        )
    blob = p.read_bytes()
    try:
        entries = parse_bml(blob)
    except ValueError as e:
        raise HTTPException(400, f"BML parse failed: {e}")
    target = next((ent for ent in entries if ent.name == inner_name), None)
    if target is None:
        raise HTTPException(404, f"no entry named {inner_name!r} in {p.name}")
    try:
        from formats.bml import decompress_prs_cached  # local import keeps server import shape
        slice_start = target.offset
        slice_end = slice_start + target.size_compressed
        return decompress_prs_cached(
            p, st.st_mtime_ns, inner_name,
            lambda: bytes(blob[slice_start:slice_end]),
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(502, f"BML PRS decompress failed: {e}")


def _read_afs_inner_nj(afs_path: Path, inner: str) -> tuple[bytes, str]:
    """Return PRS-decompressed model bytes for an AFS-form path.

    Used by the model_mesh / model_skinned / model_bundle endpoints when
    the resolved base path's extension is ``.afs``. The inner name
    follows the manifest synth shape ``"NNNN_<basename>"`` (or a bare
    digit run); ``_extract_afs_inner_bytes`` already handles both.

    Returns ``(raw_bytes, logical_inner_name)`` — the logical name
    carries the sniffed extension so downstream parsers (which dispatch
    on ``.nj`` / ``.xj``) route correctly. Raises HTTPException(400)
    if the resolved inner is not a chunk-Ninja / descriptor-Xj blob.
    """
    blob, logical = _extract_afs_inner_bytes(afs_path, inner)
    low = logical.lower()
    if not (low.endswith(".nj") or low.endswith(".xj")):
        raise HTTPException(
            400,
            f"AFS inner is not a parsable model "
            f"(expected .nj/.xj, sniffed {logical!r})",
        )
    return blob, logical


def _pick_model_mesh_parser(inner_ext: str, outer_ext: str):
    """Return the parse function appropriate for the model file extension.

    PSOBB.IO ships two distinct mesh formats inside the same NJCM IFF
    wrapper:

        ``.nj``  → chunk-based Ninja-Nj (formats/xj.py).
                   Variable-length chunks linked through a tree.
        ``.xj``  → descriptor-table Xj (formats/xj_descriptor.py).
                   Flat vertex / triangle-strip / material tables.
        ``.njm`` → animation only — same chunk parser as .nj but the
                   tree carries motion data instead of geometry. The
                   parser returns an empty list (no Strip chunks) and
                   the caller renders nothing, which is the right
                   behaviour for "open an animation as a mesh".

    Of the ~656 BML-inner models in PSOBB.IO, ~393 are ``.nj`` and ~263
    are ``.xj``; we dispatch by inner-file extension for BML inners and
    by the outer extension for top-level ``.nj`` / ``.xj`` files.

    Raises HTTPException(400) for unsupported combinations.
    """
    chosen_ext = inner_ext or outer_ext
    if chosen_ext == ".xj":
        return _xj_parse_xj_descriptor_file
    # Fall through: .nj, .njm, or unknown defaults to the chunk parser.
    # The chunk parser returns [] for .njm (animation-only) and for
    # malformed input; both are acceptable since the wire payload's
    # ``mesh_count`` will be 0 and the frontend's "primitive cube" path
    # already handles that gracefully.
    return _xj_parse_nj_file


def _build_model_file_key(
    p: Path, ext: str, inner_name: Optional[str],
) -> Optional[Tuple[Any, ...]]:
    """Build a stable parse-cache key for a resolved model path.

    For a top-level ``.nj`` / ``.xj`` we key on the absolute path,
    mtime_ns, and on-disk size. For a ``.bml`` or ``.afs`` inner we
    additionally include the inner-entry name so different inners of the
    same archive don't collide in the cache.

    Returns None when the file's stat fails — the cache layer falls
    back to a content-hash key in that case (still correct, just slower
    to compute than a stat).
    """
    try:
        st = p.stat()
    except OSError:
        return None
    if ext in (".bml", ".afs") and inner_name:
        return (str(p), int(st.st_mtime_ns), int(st.st_size), inner_name)
    return (str(p), int(st.st_mtime_ns), int(st.st_size))


def _cached_model_parse(
    nj_bytes: bytes,
    p: Path,
    outer_ext: str,
    inner_ext: str,
    inner_name: Optional[str],
) -> list:
    """Parse a model through the parse-cache LRU.

    Replaces the inline ``parser(nj_bytes)`` call site so that warm
    opens of the same model — variant picker, motion preview, paint,
    sculpt — return in <5 ms. Picks the right parser via the same
    extension dispatch as the legacy ``_pick_model_mesh_parser``.

    AFS inners (``ItemModel.afs#NNNN_...``) are sniffed as ``.nj`` by
    the afs_reader because they start with NJTL/NJCM magic, but the
    actual format is descriptor-XJ. We try the chunk-NJ parser first and
    fall back to descriptor-XJ when it fails or returns no meshes.
    """
    chosen_ext = inner_ext or outer_ext
    file_key = _build_model_file_key(p, outer_ext, inner_name)
    if chosen_ext == ".xj":
        return _parse_cache.parse_xj_file_cached(
            nj_bytes, file_key=file_key,
        )
    # Default: chunk-NJ. AFS-resident models are misclassified by the
    # extension sniff, so on failure / 0-mesh result for an AFS inner we
    # retry with the descriptor-XJ parser. Caches each result under a
    # parser-specific key (parse_cache uses parser_id internally), so
    # subsequent opens of the same inner hit the right cache directly.
    try:
        meshes = _parse_cache.parse_nj_file_cached(
            nj_bytes, file_key=file_key,
        )
    except (ValueError, IndexError):
        meshes = []
    if not meshes and outer_ext == ".afs":
        meshes = _parse_cache.parse_xj_file_cached(
            nj_bytes, file_key=file_key,
        )
    return meshes


# --------------------------------------------------------------------------
# /api/model_mesh conditional-GET (Phase perf 2026-06-19)
#
# The parse (_cached_model_parse) and binding (_build_model_texture_binding_
# cached) are each LRU-cached, so the route's heavy work is already cheap on
# a warm call. The remaining warm cost is the network round-trip + JSON
# (re)serialization of the payload. We attach an ETag keyed on the resolved
# source file's stat so the BROWSER can revalidate with a cheap 304 and
# avoid re-downloading + re-decoding a byte-identical payload.
#
# Deliberately NOT a server-side response-dict memo: a higher in-process
# cache tier here would bypass the binding/parse caches' own hit accounting
# (and the model_skinned / model_bundle routes share those tiers). The 304
# path delivers the warm win to the frontend without disturbing that.
def _model_mesh_resp_key(src: Path, path: str, inner):
    """ETag key for one model_mesh response, on the resolved file stat."""
    try:
        st = src.stat()
    except OSError:
        return None
    return (str(src), int(st.st_mtime_ns), int(st.st_size), path, inner or "")


def _model_mesh_resp_etag(key: tuple) -> str:
    """Strong ETag derived from the model_mesh cache key."""
    return '"' + hashlib.md5(repr(key).encode("utf-8")).hexdigest()[:16] + '"'


@app.get("/api/model_mesh/{path:path}")
def api_model_mesh(path: str, request: Request, inner: Optional[str] = None):
    """Parse and return triangulated mesh data for a model file.

    Path forms:
      ``<file>.nj``         - direct chunk-Ninja file
      ``<file>.xj``         - direct descriptor-Xj file
      ``<file>.bml`` + ``?inner=<name>``
                            - legacy form (the model viewer ships this)
      ``<bml>#<inner>.{nj,xj}``
                            - asset-tree BML-inner path; the inner part
                              of the path subsumes the legacy ?inner= query

    The parser is dispatched by inner-file extension (for BML inners)
    or outer extension (for top-level files). See
    ``_pick_model_mesh_parser`` for the dispatch table.

    Responses:
      200 - `{filename, inner, mesh_count, meshes:[...], totals}` per the
             wire-format docs at the top of this section.
      400 - invalid path / unsupported extension / parse failure
      404 - file or inner-entry not found
      413 - file too large to parse in-memory
      502 - BML PRS decompression subprocess failed (only for `.bml`)
    """
    base, effective_inner = _split_inner_with_query(path, inner)

    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()
    inner_ext = ""

    # Conditional-GET short-circuit. The ETag is keyed on the resolved
    # source file's stat so a re-deploy auto-invalidates. A matching
    # If-None-Match means the browser already holds the identical payload
    # — return a bodyless 304 and skip the parse + assembly entirely.
    _mm_key = _model_mesh_resp_key(p, path, effective_inner)
    _mm_etag = _model_mesh_resp_etag(_mm_key) if _mm_key is not None else None
    if _mm_etag is not None:
        _mm_inm = request.headers.get("if-none-match")
        if _mm_inm and _mm_inm == _mm_etag:
            return Response(status_code=304, headers={"ETag": _mm_etag})

    if ext == ".bml":
        if not effective_inner:
            raise HTTPException(
                400,
                "BML model requires `?inner=<entry-name>.{nj,xj}` query parameter or '#<inner>' suffix",
            )
        _validate_inner_name(effective_inner, msg="invalid inner entry name")
        inner_ext = Path(effective_inner).suffix.lower()
        # Truncated 32-character BML name field: a few PSOBB.IO entries
        # have inner names that fill the whole 32-byte name slot, leaving
        # nothing for the file extension (Path.suffix returns "" when
        # the name ends with a bare ".").  These are all .nj files in
        # practice (see ``RESEARCH_REMAINING_GAPS.md`` failure class 3),
        # so dispatch them through the chunk parser.
        if inner_ext == "" and len(effective_inner) == 32 and effective_inner.endswith("."):
            inner_ext = ".nj"
        if inner_ext not in IFF_EXTENSIONS:
            raise HTTPException(
                400,
                f"inner entry must be {IFF_EXTENSIONS!r}, got {inner_ext!r}",
            )
        nj_bytes = _read_inner_nj_from_bml(p, effective_inner)
    elif ext == ".afs":
        if not effective_inner:
            raise HTTPException(
                400,
                "AFS model requires `?inner=NNNN_<basename>` query parameter or '#<inner>' suffix",
            )
        nj_bytes, logical_inner = _read_afs_inner_nj(p, effective_inner)
        inner_ext = Path(logical_inner).suffix.lower() or ".nj"
    elif ext in IFF_EXTENSIONS:
        if effective_inner:
            raise HTTPException(
                400,
                f"`inner` not allowed for {ext} files",
            )
        sz = p.stat().st_size
        if sz > ASSET_PARSE_MAX_BYTES:
            raise HTTPException(
                413,
                f"model too large: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
            )
        nj_bytes = p.read_bytes()
    else:
        raise HTTPException(
            400,
            f"unsupported model extension {ext!r} (expected .nj, .xj, .bml, or .afs)",
        )

    try:
        meshes = _cached_model_parse(
            nj_bytes, p, ext, inner_ext, effective_inner,
        )
    except ValueError as e:
        raise HTTPException(400, f"model parse failed: {e}")
    except Exception as e:  # pragma: no cover - defensive net
        log.exception("model parse internal error")
        raise HTTPException(500, f"model parse internal error: {e}")

    payload = _xj_meshes_to_payload(meshes)
    payload["filename"] = path
    payload["inner"] = effective_inner
    # Per-submesh texture binding (NJTL → XVMH match). When the mesh
    # carries no texture sibling (no `<inner>.xvm`) we still surface
    # whatever NJTL we found so the frontend can degrade gracefully.
    try:
        payload["binding_data"] = _build_model_texture_binding_cached(
            p, ext, effective_inner, nj_bytes, meshes,
        )
    except HTTPException as e:
        # The binding endpoint is non-critical for the geometry payload
        # (the frontend's per-submesh fallback picks tile 0 if binding
        # is missing). Surface the error string but don't block the mesh.
        payload["binding_data"] = {
            "njtl": [],
            "xvmh": [],
            "binding": [],
            "error": e.detail if hasattr(e, "detail") else str(e),
        }
    except Exception as e:  # pragma: no cover — defensive net
        log.exception("texture binding internal error for %s", path)
        payload["binding_data"] = {
            "njtl": [],
            "xvmh": [],
            "binding": [],
            "error": f"binding failed: {e}",
        }
    # Convenience flat alias used by the frontend (which only reads
    # `payload.binding`). The structured `binding_data` has the full
    # NJTL/XVMH name lists for diagnostic display.
    bd = payload.get("binding_data") or {}
    payload["binding"] = bd.get("binding") or []

    # Attach the ETag (+ short max-age) so the browser can revalidate the
    # next open with a cheap 304 instead of re-downloading the payload.
    if _mm_etag is not None:
        return JSONResponse(
            content=payload,
            headers={"ETag": _mm_etag, "Cache-Control": "private, max-age=300"},
        )
    return payload


@app.get("/api/model_skinned/{path:path}")
def api_model_skinned(path: str, inner: Optional[str] = None):
    """Parse a model and return BONE-LOCAL meshes + skeleton for animation.

    Same path forms / errors as /api/model_mesh, but the response shape
    differs:
      * Vertices are in bone-LOCAL coordinates (NOT world-baked).
      * Each vertex carries a ``bone_idx`` (Int32) identifying the
        owning bone in the response's ``bones`` array.
      * The skeleton is returned verbatim — DFS-flattened, with parent
        indices and bind-pose TRS — so the frontend can compose
        per-bone matrices and animate them via NJM keyframes.

    Only `.nj` (chunk-Ninja) is supported. The descriptor-table `.xj`
    format does not carry per-vertex bone tags in a useful form, so we
    emit a 400 for `.xj` inners.

    Wire shape: see ``_xj_meshes_to_skinned_payload``.
    """
    base, effective_inner = _split_inner_with_query(path, inner)

    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()
    inner_ext = ""

    if ext == ".bml":
        if not effective_inner:
            raise HTTPException(
                400,
                "BML model requires `?inner=<entry-name>.nj` query parameter or '#<inner>' suffix",
            )
        _validate_inner_name(effective_inner, msg="invalid inner entry name")
        inner_ext = Path(effective_inner).suffix.lower()
        if inner_ext != ".nj":
            raise HTTPException(
                400,
                f"skinned mesh requires .nj inner (got {inner_ext!r}); "
                f".xj does not carry per-vertex bone tags",
            )
        nj_bytes = _read_inner_nj_from_bml(p, effective_inner)
    elif ext == ".afs":
        if not effective_inner:
            raise HTTPException(
                400,
                "AFS model requires `?inner=NNNN_<basename>` query parameter or '#<inner>' suffix",
            )
        nj_bytes, logical_inner = _read_afs_inner_nj(p, effective_inner)
        inner_ext = Path(logical_inner).suffix.lower() or ".nj"
        if inner_ext != ".nj":
            raise HTTPException(
                400,
                f"skinned mesh requires .nj inner (got {inner_ext!r}); "
                f".xj does not carry per-vertex bone tags",
            )
    elif ext == ".nj":
        if effective_inner:
            raise HTTPException(400, f"`inner` not allowed for {ext} files")
        sz = p.stat().st_size
        if sz > ASSET_PARSE_MAX_BYTES:
            raise HTTPException(
                413,
                f"model too large: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
            )
        nj_bytes = p.read_bytes()
    else:
        raise HTTPException(
            400,
            f"unsupported model extension {ext!r} for skinned path (expected .nj, .bml, or .afs)",
        )

    try:
        meshes, bones = _parse_cache.parse_nj_skinned_cached(
            nj_bytes,
            file_key=_build_model_file_key(p, ext, effective_inner),
        )
    except (ValueError, IndexError) as e:
        # AFS items (ItemModel.afs, ItemModelEp4.afs) ship descriptor-XJ
        # geometry under a `.nj` extension so the chunk-Ninja skinned
        # parser rejects them. Fall back to the world-baked mesh path
        # with a synthesised single-root-bone "skeleton" so the frontend
        # can still render them as static meshes (item models have no
        # bones / animation).
        if ext != ".afs":
            raise HTTPException(400, f"skinned model parse failed: {e}")
        from formats.xj import XjBone as _XjBone
        meshes = _cached_model_parse(
            nj_bytes, p, ext, inner_ext, effective_inner,
        )
        bones = [
            _XjBone(
                index=0, parent=-1,
                position=(0.0, 0.0, 0.0),
                rotation=(0, 0, 0),
                scale=(1.0, 1.0, 1.0),
                eval_flags=0,
            ),
        ]
    except Exception as e:  # pragma: no cover - defensive
        log.exception("skinned model parse internal error")
        raise HTTPException(500, f"skinned model parse internal error: {e}")

    # Geometry+skeleton payload is fully deterministic in (path, mtime,
    # size, inner). Cache hit returns the SAME dict object — we shallow-
    # copy below before adding per-request fields (filename / inner /
    # binding_data) so subsequent hits don't pick up stale cross-request
    # state. The b64 strings inside `meshes[*]` are immutable so the
    # shallow copy is safe (frontend reads them, never mutates).
    cached_payload = _xj_meshes_to_skinned_payload_cached(
        meshes, bones, p, effective_inner,
    )
    payload = dict(cached_payload)
    payload["filename"] = path
    payload["inner"] = effective_inner

    # Per-submesh texture binding — same logic as /api/model_mesh.
    try:
        payload["binding_data"] = _build_model_texture_binding_cached(
            p, ext, effective_inner, nj_bytes, meshes,
        )
    except HTTPException as e:
        payload["binding_data"] = {
            "njtl": [],
            "xvmh": [],
            "binding": [],
            "error": e.detail if hasattr(e, "detail") else str(e),
        }
    except Exception as e:  # pragma: no cover
        log.exception("texture binding internal error for %s", path)
        payload["binding_data"] = {
            "njtl": [],
            "xvmh": [],
            "binding": [],
            "error": f"binding failed: {e}",
        }
    bd = payload.get("binding_data") or {}
    payload["binding"] = bd.get("binding") or []
    return payload


# ---------------------------------------------------------------------------- subdivide
#
# `/api/model/subdivide` runs Loop subdivision on a model's geometry and
# returns a wire payload in the SAME shape as `/api/model_mesh` — the
# frontend's `psoApplyMeshPayload` swaps the rendered geometry to the
# subdivided result without a re-fetch.
#
# Each XjMesh is subdivided independently so material_id boundaries
# (i.e. per-submesh texture binding) survive the operation. UVs are
# linearly interpolated by trimesh.subdivide_loop's barycentric
# averaging, which is correct for Loop-style refinement.
#
# When `smooth_normals=True` the per-vertex normals are recomputed from
# the subdivided face normals (area-weighted average per vertex). This
# is the same behaviour Blender's "Smooth Shading" + Subsurf modifier
# produces.
#
# Cache: a copy of the subdivided mesh's GLB representation is also
# written to ``cache/subdivided/<base>__lvl<N>__sm<0|1>.glb`` so external
# tools can read it; the panel never refetches it (the in-line payload
# already drives the viewer), but having a serialisable artifact on disk
# means the user can "Open in Blender" later if we wire up such a button.
# ----------------------------------------------------------------------------

MAX_SUBDIVIDE_LEVEL = 3
MAX_SUBDIVIDE_TRI_COUNT = 200_000   # post-subdivide cap; ~6.4M verts max


class SubdivideReq(BaseModel):
    path: str = Field(..., description="model path or <bml>#<inner>")
    level: int = Field(1, ge=1, le=MAX_SUBDIVIDE_LEVEL,
                       description="number of Loop subdivision iterations (1..3)")
    smooth_normals: bool = Field(True, description="recompute per-vertex normals after subdivide")


def _subdivide_mesh_payload(meshes: list, level: int, smooth_normals: bool) -> dict:
    """Subdivide each XjMesh in `meshes` and emit a model_mesh-shaped payload.

    Uses trimesh.Trimesh.subdivide_loop (Loop subdivision). Every submesh
    is processed independently so material_id boundaries are preserved.
    UVs and normals follow trimesh's default barycentric interpolation;
    when smooth_normals=True we additionally recompute vertex normals
    from face normals (area-weighted) post-subdivide.
    """
    import trimesh
    import numpy as np
    import array

    payload_meshes: list[dict] = []
    total_v = 0
    total_t = 0
    pre_v = 0
    pre_t = 0
    for m in meshes:
        # Build a numpy vertex/face buffer for trimesh.
        if not m.vertices or not m.indices:
            # Empty submesh — pass through as-is.
            payload_meshes.append(_one_mesh_to_payload_dict(
                m.vertices, m.indices, m, normals_out=None,
            ))
            continue
        verts = np.array([v.pos for v in m.vertices], dtype=np.float64)
        normals = np.array([v.normal for v in m.vertices], dtype=np.float64)
        uvs = np.array([v.uv for v in m.vertices], dtype=np.float64)
        faces = np.array(m.indices, dtype=np.int64).reshape(-1, 3)
        pre_v += len(verts)
        pre_t += len(faces)

        # Loop subdivide the (verts, faces) pair, carrying per-vertex
        # attributes through the same midpoint-averaging path.
        cur_v = verts.copy()
        cur_n = normals.copy()
        cur_uv = uvs.copy()
        cur_f = faces.copy()
        for _ in range(level):
            # trimesh.remesh.subdivide_loop only supports verts+faces;
            # we manually carry attributes by running plain subdivide
            # (linear midpoint, NO Loop smoothing) and getting back the
            # `index` mapping that tells us which midpoint each new
            # vertex came from. Then we apply Loop smoothing to verts
            # only — simple Loop is fine for "Loop subdivision" in our
            # UX context (the user wants more polys + smoother shape;
            # we don't need the exact convergent limit surface).
            new_v, new_f, idx_pairs = _loop_subdivide_with_attr_pairs(cur_v, cur_f)
            # Linear midpoint for normals + UVs using idx_pairs.
            # idx_pairs maps each NEW vertex back to the (a, b) pair of
            # OLD vertex indices it averages between; original vertices
            # come first in `new_v`, so the idx_pairs layout is
            # [(a0,a0), ..., (a_origN, a_origN), (e0_a, e0_b), ...].
            cur_n = _avg_via_pairs(cur_n, idx_pairs)
            cur_uv = _avg_via_pairs(cur_uv, idx_pairs)
            # Vertex positions get the actual Loop-weighted update.
            cur_v = new_v
            cur_f = new_f
            # Bail early if we'd exceed the safety cap.
            if len(cur_v) > MAX_SUBDIVIDE_TRI_COUNT:
                break

        # Normalise + optionally recompute smooth normals from faces.
        if smooth_normals:
            tm = trimesh.Trimesh(vertices=cur_v, faces=cur_f, process=False)
            cur_n = np.asarray(tm.vertex_normals, dtype=np.float64)
        else:
            # Re-normalise the interpolated normals so length is 1.
            lens = np.linalg.norm(cur_n, axis=1, keepdims=True)
            lens[lens < 1e-9] = 1.0
            cur_n = cur_n / lens

        # Triangulated indices, flat list.
        flat_idx = cur_f.reshape(-1).astype(np.int64).tolist()

        # Pack into the same wire shape as _xj_meshes_to_payload.
        floats = array.array("f")
        minx = miny = minz = float("inf")
        maxx = maxy = maxz = float("-inf")
        for i in range(len(cur_v)):
            px, py, pz = float(cur_v[i, 0]), float(cur_v[i, 1]), float(cur_v[i, 2])
            nx, ny, nz = float(cur_n[i, 0]), float(cur_n[i, 1]), float(cur_n[i, 2])
            uu = float(cur_uv[i, 0]) if i < len(cur_uv) else 0.0
            vv = float(cur_uv[i, 1]) if i < len(cur_uv) else 0.0
            floats.extend((px, py, pz, nx, ny, nz, uu, vv))
            if px < minx: minx = px
            if py < miny: miny = py
            if pz < minz: minz = pz
            if px > maxx: maxx = px
            if py > maxy: maxy = py
            if pz > maxz: maxz = pz
        if sys.byteorder != "little":
            floats.byteswap()
        verts_bytes = floats.tobytes()
        idx_arr = array.array("I", flat_idx)
        if sys.byteorder != "little":
            idx_arr.byteswap()
        idx_bytes = idx_arr.tobytes()

        if len(cur_v) == 0:
            aabb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            aabb = [minx, miny, minz, maxx, maxy, maxz]

        vc = int(len(cur_v))
        tc = int(len(cur_f))
        total_v += vc
        total_t += tc
        payload_meshes.append({
            "vertices_b64": base64.b64encode(verts_bytes).decode("ascii"),
            "indices_b64": base64.b64encode(idx_bytes).decode("ascii"),
            "vertex_count": vc,
            "triangle_count": tc,
            "material_id": m.material_id,
            "bounding_sphere": list(m.bounding_sphere),
            "aabb": aabb,
            "world_position": list(m.world_position),
            "world_rotation_euler": list(m.world_rotation_euler),
            "world_scale": list(m.world_scale),
            "world_matrix": list(m.world_matrix),
        })

    return {
        "mesh_count": len(payload_meshes),
        "meshes": payload_meshes,
        "totals": {"vertices": total_v, "triangles": total_t},
        "vertices_pre_transformed": True,
        "vert_total": total_v,
        "tri_total": total_t,
        "before": {"vertices": pre_v, "triangles": pre_t},
        "after": {"vertices": total_v, "triangles": total_t},
    }


def _one_mesh_to_payload_dict(verts, indices, mesh_obj, normals_out=None) -> dict:
    """Pass-through for empty submeshes (skips serialization)."""
    import array
    floats = array.array("f")
    for v in verts:
        nx, ny, nz = v.normal
        floats.extend((v.pos[0], v.pos[1], v.pos[2], nx, ny, nz, v.uv[0], v.uv[1]))
    if sys.byteorder != "little":
        floats.byteswap()
    idx_arr = array.array("I", indices)
    if sys.byteorder != "little":
        idx_arr.byteswap()
    return {
        "vertices_b64": base64.b64encode(floats.tobytes()).decode("ascii"),
        "indices_b64": base64.b64encode(idx_arr.tobytes()).decode("ascii"),
        "vertex_count": len(verts),
        "triangle_count": len(indices) // 3,
        "material_id": mesh_obj.material_id,
        "bounding_sphere": list(mesh_obj.bounding_sphere),
        "aabb": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "world_position": list(mesh_obj.world_position),
        "world_rotation_euler": list(mesh_obj.world_rotation_euler),
        "world_scale": list(mesh_obj.world_scale),
        "world_matrix": list(mesh_obj.world_matrix),
    }


def _loop_subdivide_with_attr_pairs(verts, faces):
    """Loop subdivision step that also returns a midpoint-pair index list.

    Returns (new_verts, new_faces, idx_pairs) where ``idx_pairs[k]`` is
    a tuple ``(a, b)`` of OLD vertex indices that were averaged to
    produce new vertex k. For original (un-subdivided) vertices the
    pair is ``(k, k)`` so caller-side attribute averaging treats them
    as "averaged with self" (i.e. unchanged).

    Implements the Loop scheme:
      * each edge midpoint -> new vertex at (3/8)*(a+b) + (1/8)*(opposite_pair_avg)
      * each old vertex -> updated to (1 - n*beta) * v + beta * sum(neighbours)
        where beta = (1/n)(5/8 - (3/8 + 1/4 cos(2 pi / n))^2) for n != 3
        and beta = 3/16 for n = 3.

    For attribute carriage (UV / normal) we want LINEAR averaging — so
    callers should average via idx_pairs (a + b)/2 ignoring the Loop
    weight. This matches Blender's UV behaviour: Loop affects geometry
    only, UVs subdivide linearly.
    """
    import numpy as np
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    nv = len(verts)
    # Build edge -> midpoint-vertex-id map.
    edge_map = {}
    midpoints = []      # (a, b) pairs in original vertex indices
    edge_opposite = []  # for each midpoint, list of opposite-corner vertex ids

    def get_mid(a, b):
        key = (a, b) if a < b else (b, a)
        if key in edge_map:
            return edge_map[key]
        idx = nv + len(midpoints)
        edge_map[key] = idx
        midpoints.append(key)
        edge_opposite.append([])
        return idx

    # First pass: assign midpoints + collect opposite corners for each edge.
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        m_ab = get_mid(a, b)
        m_bc = get_mid(b, c)
        m_ca = get_mid(c, a)
        edge_opposite[m_ab - nv].append(c)
        edge_opposite[m_bc - nv].append(a)
        edge_opposite[m_ca - nv].append(b)

    # Loop weight midpoint positions.
    new_v_count = nv + len(midpoints)
    new_v = np.zeros((new_v_count, verts.shape[1]), dtype=np.float64)
    # Original-vertex Loop update: v' = (1 - n*beta) * v + beta * sum(neighbours)
    # First compute the neighbour set per vertex.
    neighbours: list[set[int]] = [set() for _ in range(nv)]
    for (a, b) in midpoints:
        neighbours[a].add(b)
        neighbours[b].add(a)
    for i in range(nv):
        nbrs = list(neighbours[i])
        n = len(nbrs)
        if n == 0:
            new_v[i] = verts[i]
            continue
        if n == 3:
            beta = 3.0 / 16.0
        else:
            t = 0.375 + 0.25 * np.cos(2.0 * np.pi / n)
            beta = (1.0 / n) * (0.625 - t * t)
        new_v[i] = (1.0 - n * beta) * verts[i] + beta * np.sum(verts[nbrs], axis=0)

    # Edge-midpoint Loop update: 3/8(a+b) + 1/8(c+d) where c,d are
    # opposite-corner vertices of the two faces sharing edge (a,b).
    # If the edge is on a boundary (only 1 face), use 1/2 (a+b) — the
    # standard boundary mask.
    for k, (a, b) in enumerate(midpoints):
        opp = edge_opposite[k]
        if len(opp) >= 2:
            new_v[nv + k] = (3.0 / 8.0) * (verts[a] + verts[b]) + (1.0 / 8.0) * (verts[opp[0]] + verts[opp[1]])
        else:
            new_v[nv + k] = 0.5 * (verts[a] + verts[b])

    # Second pass: build new face list — each old face splits into 4.
    new_faces = []
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        m_ab = edge_map[(a, b) if a < b else (b, a)]
        m_bc = edge_map[(b, c) if b < c else (c, b)]
        m_ca = edge_map[(c, a) if c < a else (a, c)]
        new_faces.append([a, m_ab, m_ca])
        new_faces.append([m_ab, b, m_bc])
        new_faces.append([m_ca, m_bc, c])
        new_faces.append([m_ab, m_bc, m_ca])
    new_faces = np.array(new_faces, dtype=np.int64)

    # idx_pairs: for original vertices, (k, k); for midpoints, the (a, b) edge.
    idx_pairs = [(k, k) for k in range(nv)] + [tuple(p) for p in midpoints]
    return new_v, new_faces, idx_pairs


def _avg_via_pairs(attr, idx_pairs):
    """Average a per-vertex attribute via midpoint pairs.

    For the first ``len(attr)`` entries we copy verbatim; for the rest
    we emit (attr[a] + attr[b]) / 2.
    """
    import numpy as np
    out = np.zeros((len(idx_pairs), attr.shape[1]), dtype=attr.dtype)
    nv = len(attr)
    for k, (a, b) in enumerate(idx_pairs):
        if k < nv:
            out[k] = attr[k]
        else:
            out[k] = 0.5 * (attr[a] + attr[b])
    return out


@app.post("/api/model/subdivide")
def api_model_subdivide(req: SubdivideReq, request: Request):
    """Run Loop subdivision on a model and return a model_mesh-shape payload.

    Body schema:
        {
          "path": "<bml>#<inner>.nj" | "<file>.nj" | ...,
          "level": 1..3,
          "smooth_normals": true|false  (default true)
        }

    Response:
        {
          ... same shape as /api/model_mesh ...,
          "before": {"vertices": <pre>, "triangles": <pre>},
          "after":  {"vertices": <post>, "triangles": <post>},
          "cache_path": "<rel path inside CACHE_DIR>"  (when written)
        }

    Subdivides each submesh independently so material_id boundaries
    survive. UV/normal interpolation is linear (matches Blender's
    convention for subsurf modifiers + UV maps). Smooth normals
    (default) recompute per-vertex normals from face normals after
    subdivision.

    Errors:
      400 - bad path / unsupported extension / parse failure
      404 - model not found in DATA_DIR or LIVE_DATA_DIR
      413 - model too large to parse in-memory
      502 - PRS subprocess failure (BML inner-extract)
    """
    _enforce_body_size(request, MAX_REPACK_DIFF_BODY)
    path = req.path
    level = max(1, min(MAX_SUBDIVIDE_LEVEL, int(req.level)))
    smooth = bool(req.smooth_normals)

    # Resolve the model bytes through resolve_asset_bytes — handles
    # `<bml>#<inner>` and `<afs>#<NNNN>_<name>` for free.
    blob, logical = resolve_asset_bytes(path)
    # Pick parser by inner extension (logical name carries it).
    ext = Path(logical).suffix.lower()
    if ext not in IFF_EXTENSIONS:
        raise HTTPException(400, f"unsupported model extension {ext!r} (expected {IFF_EXTENSIONS})")
    parser = _pick_model_mesh_parser(ext, ext)
    try:
        meshes = parser(blob)
    except ValueError as e:
        raise HTTPException(400, f"model parse failed: {e}")

    pre_total_v = sum(len(m.vertices) for m in meshes)
    pre_total_t = sum(len(m.indices) // 3 for m in meshes)
    if pre_total_t == 0:
        raise HTTPException(400, "model has no triangles to subdivide")
    # Safety cap: refuse to subdivide if the post-subdivide tri count
    # would exceed our limit (each iteration ~4× tri count).
    projected = pre_total_t * (4 ** level)
    if projected > MAX_SUBDIVIDE_TRI_COUNT:
        raise HTTPException(
            413,
            f"projected triangle count {projected} > cap {MAX_SUBDIVIDE_TRI_COUNT}; "
            f"reduce level (current={level}, ~{pre_total_t} tris × 4^{level})",
        )

    # Run subdivision.
    try:
        payload = _subdivide_mesh_payload(meshes, level, smooth)
    except Exception as e:
        log.exception("subdivide failed")
        raise HTTPException(500, f"subdivide failed: {e}")

    payload["filename"] = path
    payload["level"] = level
    payload["smooth_normals"] = smooth
    # Reuse the binding from the source model. Since we preserve
    # material_id values per submesh, the same binding still applies.
    base, hash_inner = _split_inner_path(path)
    base_path = _resolve_base_path(base)
    outer_ext = base_path.suffix.lower()
    try:
        bd = _build_model_texture_binding_cached(
            base_path, outer_ext, hash_inner, blob, meshes,
        )
        payload["binding_data"] = bd
        payload["binding"] = (bd or {}).get("binding") or []
    except HTTPException as e:
        payload["binding"] = []
        payload["binding_data"] = {"njtl": [], "xvmh": [], "binding": [],
                                   "error": e.detail if hasattr(e, "detail") else str(e)}
    except Exception as e:  # pragma: no cover
        payload["binding"] = []
        payload["binding_data"] = {"njtl": [], "xvmh": [], "binding": [],
                                   "error": f"binding failed: {e}"}

    # Wrap the payload in a top-level result object so we can surface
    # before/after stats independently (the wire payload itself drops
    # them so /api/model_mesh consumers stay clean).
    result = {
        "mesh_payload": payload,
        "totals": payload.get("totals"),
        "before": payload.get("before"),
        "after": payload.get("after"),
        "level": level,
        "smooth_normals": smooth,
    }
    return result


# ---------------------------------------------------------------------------- decimate
#
# `/api/decimate` runs a real Quadric-Error-Metric (QEM) decimator on a
# model's geometry and returns a wire payload in the SAME shape as
# `/api/model_mesh` — the frontend's `psoApplyMeshPayload` swaps the
# rendered geometry to the decimated result without a re-fetch. This is
# the inverse of `/api/model/subdivide`.
#
# The heavy lifting lives in `formats/decimate.py` (decimate_mesh): QEM via
# trimesh's fast-simplification backend, with a hand-rolled NumPy QEM
# fallback when that backend is unavailable. Each XjMesh is decimated
# independently so material_id boundaries (per-submesh texture binding)
# survive. Per-vertex normals are recomputed smooth from the decimated
# faces; UVs are re-sampled by nearest original vertex (a LOD
# approximation — see formats/decimate.py module docstring).
#
# The endpoint accepts either `target_ratio` (fraction of tris to KEEP) or
# `target_tris` (absolute count). When both are given, target_tris wins.
# ----------------------------------------------------------------------------

MIN_DECIMATE_TRIS = 4


class DecimateReq(BaseModel):
    path: str = Field(..., description="model path or <bml>#<inner>")
    target_ratio: Optional[float] = Field(
        None, gt=0.0, le=1.0,
        description="fraction of triangles to KEEP, (0, 1]; 0.5 ~ halve")
    target_tris: Optional[int] = Field(
        None, ge=MIN_DECIMATE_TRIS,
        description="absolute target triangle count (wins over target_ratio)")
    preserve_border: bool = Field(
        True, description="pin boundary edges so open meshes don't shrink at seams")


def _decimate_mesh_payload(meshes: list, *, target_ratio, target_tris,
                           preserve_border: bool) -> dict:
    """Decimate each XjMesh in `meshes` and emit a model_mesh-shaped payload.

    Uses formats.decimate.decimate_mesh (QEM). Every submesh is processed
    independently so material_id boundaries are preserved. The per-submesh
    target is the SAME ratio across submeshes (so a ratio of 0.5 halves the
    whole model proportionally); when an absolute target_tris is given it is
    distributed across submeshes in proportion to each submesh's tri share.
    """
    import array
    import numpy as np
    from formats.decimate import decimate_mesh

    pre_total_t = sum(len(m.indices) // 3 for m in meshes if m.indices)
    # Resolve the global ratio once so we can apply it per-submesh.
    if target_tris is not None and pre_total_t > 0:
        global_ratio = max(0.0, min(1.0, float(target_tris) / float(pre_total_t)))
    else:
        global_ratio = None

    backends: set[str] = set()
    payload_meshes: list[dict] = []
    total_v = 0
    total_t = 0
    pre_v = 0
    pre_t = 0
    for m in meshes:
        if not m.vertices or not m.indices:
            payload_meshes.append(_one_mesh_to_payload_dict(
                m.vertices, m.indices, m, normals_out=None,
            ))
            continue
        verts = np.array([v.pos for v in m.vertices], dtype=np.float64)
        uvs = np.array([v.uv for v in m.vertices], dtype=np.float64)
        faces = np.array(m.indices, dtype=np.int64).reshape(-1, 3)
        pre_v += len(verts)
        pre_t += len(faces)

        # Per-submesh target: distribute the absolute target by tri-share, or
        # apply the global ratio directly.
        if global_ratio is not None:
            sub_target = max(MIN_DECIMATE_TRIS, int(round(len(faces) * global_ratio)))
            sub_ratio = None
        else:
            sub_target = None
            sub_ratio = target_ratio

        out_v, out_f, out_uv, meta = decimate_mesh(
            verts, faces,
            target_ratio=sub_ratio, target_tris=sub_target,
            preserve_border=preserve_border, uvs=uvs, return_meta=True,
        )
        backends.add(meta["backend"])

        # Recompute smooth normals from the decimated faces.
        if len(out_f) > 0:
            import trimesh
            tm = trimesh.Trimesh(vertices=out_v, faces=out_f, process=False)
            out_n = np.asarray(tm.vertex_normals, dtype=np.float64)
        else:
            out_n = np.zeros((len(out_v), 3), dtype=np.float64)
        if out_uv is None:
            out_uv = np.zeros((len(out_v), 2), dtype=np.float64)

        flat_idx = out_f.reshape(-1).astype(np.int64).tolist()
        floats = array.array("f")
        minx = miny = minz = float("inf")
        maxx = maxy = maxz = float("-inf")
        for i in range(len(out_v)):
            px, py, pz = float(out_v[i, 0]), float(out_v[i, 1]), float(out_v[i, 2])
            nx, ny, nz = float(out_n[i, 0]), float(out_n[i, 1]), float(out_n[i, 2])
            uu = float(out_uv[i, 0]) if i < len(out_uv) else 0.0
            vv = float(out_uv[i, 1]) if i < len(out_uv) else 0.0
            floats.extend((px, py, pz, nx, ny, nz, uu, vv))
            if px < minx: minx = px
            if py < miny: miny = py
            if pz < minz: minz = pz
            if px > maxx: maxx = px
            if py > maxy: maxy = py
            if pz > maxz: maxz = pz
        if sys.byteorder != "little":
            floats.byteswap()
        verts_bytes = floats.tobytes()
        idx_arr = array.array("I", flat_idx)
        if sys.byteorder != "little":
            idx_arr.byteswap()
        idx_bytes = idx_arr.tobytes()

        if len(out_v) == 0:
            aabb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            aabb = [minx, miny, minz, maxx, maxy, maxz]

        vc = int(len(out_v))
        tc = int(len(out_f))
        total_v += vc
        total_t += tc
        payload_meshes.append({
            "vertices_b64": base64.b64encode(verts_bytes).decode("ascii"),
            "indices_b64": base64.b64encode(idx_bytes).decode("ascii"),
            "vertex_count": vc,
            "triangle_count": tc,
            "material_id": m.material_id,
            "bounding_sphere": list(m.bounding_sphere),
            "aabb": aabb,
            "world_position": list(m.world_position),
            "world_rotation_euler": list(m.world_rotation_euler),
            "world_scale": list(m.world_scale),
            "world_matrix": list(m.world_matrix),
        })

    return {
        "mesh_count": len(payload_meshes),
        "meshes": payload_meshes,
        "totals": {"vertices": total_v, "triangles": total_t},
        "vertices_pre_transformed": True,
        "vert_total": total_v,
        "tri_total": total_t,
        "before": {"vertices": pre_v, "triangles": pre_t},
        "after": {"vertices": total_v, "triangles": total_t},
        "backend": "+".join(sorted(backends)) if backends else "noop",
    }


@app.post("/api/decimate")
def api_decimate(req: DecimateReq, request: Request):
    """Run QEM decimation on a model and return a model_mesh-shape payload.

    Body schema:
        {
          "path": "<bml>#<inner>.nj" | "<file>.nj" | ...,
          "target_ratio": 0.5,        # fraction of tris to KEEP (mutually
          "target_tris": 2000,        #   exclusive-ish; target_tris wins)
          "preserve_border": true     # default true
        }

    Response (same outer shape as /api/model/subdivide):
        {
          "mesh_payload": { ...same shape as /api/model_mesh... },
          "totals":  {"vertices": .., "triangles": ..},
          "before":  {"vertices": <pre>,  "triangles": <pre>},
          "after":   {"vertices": <post>, "triangles": <post>},
          "backend": "trimesh_fast_simplification" | "numpy_qem_fallback" | ...
        }

    Errors:
      400 - bad path / unsupported extension / parse failure / no triangles
      404 - model not found
      422 - neither target_ratio nor target_tris supplied
    """
    _enforce_body_size(request, MAX_REPACK_DIFF_BODY)
    if req.target_ratio is None and req.target_tris is None:
        raise HTTPException(422, "must supply target_ratio or target_tris")

    path = req.path
    blob, logical = resolve_asset_bytes(path)
    ext = Path(logical).suffix.lower()
    if ext not in IFF_EXTENSIONS:
        raise HTTPException(400, f"unsupported model extension {ext!r} (expected {IFF_EXTENSIONS})")
    parser = _pick_model_mesh_parser(ext, ext)
    try:
        meshes = parser(blob)
    except ValueError as e:
        raise HTTPException(400, f"model parse failed: {e}")

    pre_total_t = sum(len(m.indices) // 3 for m in meshes)
    if pre_total_t == 0:
        raise HTTPException(400, "model has no triangles to decimate")

    try:
        payload = _decimate_mesh_payload(
            meshes,
            target_ratio=req.target_ratio,
            target_tris=req.target_tris,
            preserve_border=bool(req.preserve_border),
        )
    except Exception as e:
        log.exception("decimate failed")
        raise HTTPException(500, f"decimate failed: {e}")

    payload["filename"] = path
    payload["target_ratio"] = req.target_ratio
    payload["target_tris"] = req.target_tris
    # Reuse the source model's texture binding (material_id values survive).
    base, hash_inner = _split_inner_path(path)
    base_path = _resolve_base_path(base)
    outer_ext = base_path.suffix.lower()
    try:
        bd = _build_model_texture_binding_cached(
            base_path, outer_ext, hash_inner, blob, meshes,
        )
        payload["binding_data"] = bd
        payload["binding"] = (bd or {}).get("binding") or []
    except HTTPException as e:
        payload["binding"] = []
        payload["binding_data"] = {"njtl": [], "xvmh": [], "binding": [],
                                   "error": e.detail if hasattr(e, "detail") else str(e)}
    except Exception as e:  # pragma: no cover
        payload["binding"] = []
        payload["binding_data"] = {"njtl": [], "xvmh": [], "binding": [],
                                   "error": f"binding failed: {e}"}

    return {
        "mesh_payload": payload,
        "totals": payload.get("totals"),
        "before": payload.get("before"),
        "after": payload.get("after"),
        "backend": payload.get("backend"),
        "target_ratio": req.target_ratio,
        "target_tris": req.target_tris,
    }


# ---------------------------------------------------------------------------- sculpt
#
# Geometry sculpting persistence (2026-04-25). The frontend's Sculpt tab
# (`static/sculpt_panel.js`) maintains per-vertex displacement deltas in
# THREE.BufferAttribute arrays during a stroke, then POSTs the final
# deltas to /api/sculpt/save when the user clicks "Save sculpt". The
# server stores the JSON sidecar in CACHE_DIR/sculpted_meshes/ and
# returns a 16-char SHA that re-fetches via /api/sculpt/<sha>.
#
# The stored format is the same one ``formats.sculpt.encode_sculpt_payload``
# emits — sparse if <33% of verts are moved, dense otherwise. The client
# decodes via the JS-side mirror (sculpt_panel.js::decodeSculpt) and
# re-applies as a position-attribute mutation on top of the live mesh.
#
# Limitation: sculpt deltas are NOT round-tripped through the BML/AFS
# packer in this v1 — full deploy waits for the NJ encoder (Phase A4).
# /api/sculpt/build_archive surfaces the JSON sidecar for the editor's
# tooling, but does not produce a deployable .bml.
# ----------------------------------------------------------------------------


class SculptSaveReq(BaseModel):
    """Body of POST /api/sculpt/save.

    `model_path` is the source model (e.g. ``<bml>#<inner>.nj``).
    `mesh_payload` is the wire JSON produced by
    ``sculpt_mod.encode_sculpt_payload`` on the JS side — the server
    decodes-then-re-encodes once for validation + canonical SHA.
    """
    model_path: str = Field(..., description="source model path or <bml>#<inner>")
    mesh_payload: dict = Field(..., description="sculpt wire payload (encode_sculpt_payload shape)")
    subdivide_level: int = Field(0, ge=0, le=8, description="subdivide depth applied before sculpt")
    smooth_normals: bool = Field(True, description="were smooth normals on for the source mesh")


def _sculpt_safe_filename(model_path: str, sha: str) -> str:
    """Map (model_path, sha) -> a safe filename inside SCULPT_CACHE_DIR.

    Path separators (`/`, `\\`, `#`) become `__`. Strips trailing
    dot-extensions to keep the leading stem readable in directory
    listings; the suffix is always `.json`.
    """
    safe = _CACHE_PATH_SEPS_RE.sub("__", model_path)
    # Strip a trailing extension so the final filename's suffix
    # is always ``.json``. We keep the rest of the dotted segments
    # to retain the original `<base>__<inner>.nj` shape minus the .nj.
    if "." in safe.rsplit("__", 1)[-1]:
        safe = safe.rsplit(".", 1)[0]
    safe = _CACHE_PATH_SAFE_RE.sub("_", safe)
    return f"{safe}__{sha}.json"


@app.post("/api/sculpt/save")
def api_sculpt_save(req: SculptSaveReq, request: Request):
    """Persist a sculpted mesh (vertex-displacement deltas) for later
    re-use.

    Body schema:
        {
          "model_path":     "<bml>#<inner>.nj" | "<file>.nj" | ...,
          "mesh_payload":   <sculpt JSON in encode_sculpt_payload shape>,
          "subdivide_level": <int>  (default 0),
          "smooth_normals": <bool>  (default true)
        }

    Response:
        {
          "ok": true,
          "sha": "<16 hex chars>",
          "cache_path": "<absolute path>",
          "size": <bytes written>
        }

    Errors:
      400 — payload not a dict / fails decode
      413 — body > 32 MB
    """
    _enforce_body_size(request, MAX_SCULPT_SAVE_BODY)

    payload = req.mesh_payload
    if not isinstance(payload, dict):
        raise HTTPException(400, "mesh_payload must be a JSON object")
    # Validate by decoding once. Any structural problem will surface
    # here as ValueError.
    try:
        subs = sculpt_mod.decode_sculpt_payload(payload)
    except (ValueError, KeyError, TypeError, base64.binascii.Error) as e:
        raise HTTPException(400, f"sculpt payload invalid: {e}")

    # Re-encode through the canonical encoder so the on-disk SHA is
    # stable across client implementations.
    src_sha = str(payload.get("source_sha") or sculpt_mod.compute_source_sha(
        req.model_path.encode("utf-8")
    ))
    canon = sculpt_mod.encode_sculpt_payload(
        source_path=req.model_path,
        source_sha=src_sha,
        submeshes=subs,
        subdivide_level=req.subdivide_level,
        smooth_normals=req.smooth_normals,
    )
    sha = canon["sha"]

    fn = _sculpt_safe_filename(req.model_path, sha)
    out_path = SCULPT_CACHE_DIR / fn
    # Atomic write: tmp + os.replace.
    tmp = out_path.with_suffix(".json.tmp")
    body = json.dumps(canon, separators=(",", ":")).encode("utf-8")
    tmp.write_bytes(body)
    os.replace(tmp, out_path)

    return {
        "ok": True,
        "sha": sha,
        "cache_path": str(out_path),
        "size": len(body),
    }


@app.get("/api/sculpt/{sha}")
def api_sculpt_fetch(sha: str):
    """Re-fetch a previously-saved sculpt by its SHA.

    The SHA is the 16-char hash returned from /api/sculpt/save (or any
    longer SHA where the leading 16 chars match — supports future
    server-side re-hashing).

    Response:
        {
          "ok": true,
          "sha": "<sha>",
          "cache_path": "<abs path>",
          "mesh_payload": { ... full encode_sculpt_payload JSON ... }
        }

    Errors:
      404 — no sculpt with that SHA on disk.
    """
    sha = (sha or "").lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", sha):
        raise HTTPException(400, "sha must be 8-64 hex chars")
    # Glob the cache dir for any file matching `*__<sha>.json`.
    matches = list(SCULPT_CACHE_DIR.glob(f"*__{sha}.json"))
    if not matches:
        # Fallback: any JSON whose internal "sha" matches.
        for cand in SCULPT_CACHE_DIR.glob("*.json"):
            try:
                d = json.loads(cand.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if str(d.get("sha", "")).startswith(sha):
                matches = [cand]
                break
    if not matches:
        raise HTTPException(404, f"no sculpt found for sha {sha!r}")
    out = matches[0]
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(500, f"sculpt read failed: {e}")
    return {
        "ok": True,
        "sha": data.get("sha", sha),
        "cache_path": str(out),
        "mesh_payload": data,
    }


class SculptBuildArchiveReq(BaseModel):
    """Body of POST /api/sculpt/build_archive.

    Surfaces every persisted sculpt JSON for the given host archive
    (BML or AFS) so the editor can present a unified deploy view.
    """
    model_path: str = Field(..., description="host archive path (e.g. <bml>) or <bml>#<inner>")


@app.post("/api/sculpt/build_archive")
def api_sculpt_build_archive(req: SculptBuildArchiveReq):
    """List sculpted meshes that belong to a given host archive.

    Returns metadata only — the editor reads the JSON sidecars itself.

    NOTE: this is a v1 stub. Full archive build (BML/AFS re-pack) waits
    for the NJ encoder (Phase A4). The response includes a
    ``deployable: false`` flag so callers know to surface a "save the
    geometry to disk; deploy when NJ encoder ships" warning.

    Response:
        {
          "ok": true,
          "host": "<bml or afs basename>",
          "deployable": false,
          "sculpts": [
            {"sha": "...", "model_path": "<bml>#<inner>.nj",
             "submesh_count": <int>, "size": <bytes>, "saved_at_ms": <int>},
            ...
          ]
        }
    """
    host = req.model_path.split("#", 1)[0]
    host_lower = host.lower()
    out_list: list[dict] = []
    for cand in SCULPT_CACHE_DIR.glob("*.json"):
        try:
            d = json.loads(cand.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sp = str(d.get("source_path", ""))
        # Match either the exact host (e.g. "foo.bml") or the
        # "<host>#<inner>" prefix.
        sp_host = sp.split("#", 1)[0]
        if sp_host.lower() != host_lower:
            continue
        out_list.append({
            "sha": d.get("sha", ""),
            "model_path": sp,
            "subdivide_level": int(d.get("subdivide_level", 0)),
            "submesh_count": len(d.get("submeshes", [])),
            "size": cand.stat().st_size,
            "saved_at_ms": int(d.get("saved_at_ms", 0)),
        })
    out_list.sort(key=lambda r: r["saved_at_ms"], reverse=True)
    return {
        "ok": True,
        "host": Path(host).name,
        "deployable": False,
        "deploy_pending": "NJ encoder (phase A4)",
        "sculpts": out_list,
    }


# ---------------------------------------------------------------------------- pro-tools edit
#
# Vertex-transform persistence for the Edit tab (`static/edit_panel.js`).
# Same envelope as sculpt sidecars, but the payload carries explicit
# per-vertex INDICES (Edit tab transforms a sparse selection) rather
# than dense displacement arrays. Saved to CACHE_DIR/protools_edits/ so
# the existing sculpt build_archive walker can be extended to list both
# in one call.
#
# Wire format (POST /api/protools/save_vertex_transforms):
#   {
#     "model_path":  "<bml>#<inner>.nj" | "<file>.nj" | ...,
#     "submeshes": [
#       {
#         "submesh_idx":   <int>,
#         "material_id":   <int>,
#         "vertex_count":  <int>,        # total verts in submesh (for validation)
#         "indices":       [<int>, ...], # vertex indices that moved
#         "displacement":  [dx, dy, dz, dx, dy, dz, ...],  # 3*len(indices)
#       },
#       ...
#     ]
#   }
#
# Response:
#   { "ok": true, "sha": "<16 hex>", "cache_path": "<abs>", "size": <bytes> }
#
# Errors: 400 invalid payload; 413 body too big; 422 indices/disp mismatch.

MAX_PROTOOLS_SAVE_BODY = 32 * 1024 * 1024


class ProtoolsVertexEdit(BaseModel):
    """One submesh's worth of vertex deltas in the protools wire format."""
    submesh_idx: int = Field(..., ge=0)
    material_id: int = Field(0, ge=0)
    vertex_count: int = Field(..., ge=0)
    indices: List[int]
    displacement: List[float]


class ProtoolsSaveReq(BaseModel):
    """Body of POST /api/protools/save_vertex_transforms.

    Mirrors :class:`SculptSaveReq` but encodes the SPARSE edit pattern
    of the Edit tab (the user transforms a few selected vertices, not
    a stroke-painted region).
    """
    model_path: str = Field(..., description="<bml>#<inner>.nj or <file>.nj")
    submeshes: List[ProtoolsVertexEdit]
    subdivide_level: int = Field(0, ge=0, le=8)


def _protools_safe_filename(model_path: str, sha: str) -> str:
    """Same naming scheme as `_sculpt_safe_filename` so listings sort cleanly."""
    safe = _CACHE_PATH_SEPS_RE.sub("__", model_path)
    if "." in safe.rsplit("__", 1)[-1]:
        safe = safe.rsplit(".", 1)[0]
    safe = _CACHE_PATH_SAFE_RE.sub("_", safe)
    return f"{safe}__{sha}.json"


def _protools_compute_sha(model_path: str, submeshes: List[ProtoolsVertexEdit]) -> str:
    """Stable 16-char content hash of (model_path, sorted submeshes)."""
    h = hashlib.sha256()
    h.update(model_path.encode("utf-8"))
    h.update(b"\x00")
    for sm in sorted(submeshes, key=lambda x: x.submesh_idx):
        h.update(f"{sm.submesh_idx}|{sm.material_id}|{sm.vertex_count}|".encode("ascii"))
        h.update(struct.pack(f"<{len(sm.indices)}I", *(int(x) & 0xFFFFFFFF for x in sm.indices)))
        h.update(struct.pack(f"<{len(sm.displacement)}f", *sm.displacement))
    return h.hexdigest()[:16]


@app.post("/api/protools/save_vertex_transforms")
def api_protools_save_vertex_transforms(req: ProtoolsSaveReq, request: Request):
    """Persist a vertex-transform edit (Edit tab) for later deploy.

    Returns the on-disk SHA so the editor can re-fetch deterministically.
    """
    _enforce_body_size(request, MAX_PROTOOLS_SAVE_BODY)

    if not req.submeshes:
        raise HTTPException(400, "submeshes must be non-empty")
    for sm in req.submeshes:
        if len(sm.indices) * 3 != len(sm.displacement):
            raise HTTPException(
                422,
                f"submesh {sm.submesh_idx}: displacement length "
                f"{len(sm.displacement)} != 3 * indices {len(sm.indices)}",
            )
        if any(i < 0 or i >= sm.vertex_count for i in sm.indices):
            raise HTTPException(
                422,
                f"submesh {sm.submesh_idx}: index out of range [0, {sm.vertex_count})",
            )

    sha = _protools_compute_sha(req.model_path, req.submeshes)

    canon = {
        "format_version": 1,
        "kind": "protools_vertex_transforms",
        "model_path": req.model_path,
        "subdivide_level": req.subdivide_level,
        "submeshes": [
            {
                "submesh_idx": sm.submesh_idx,
                "material_id": sm.material_id,
                "vertex_count": sm.vertex_count,
                "indices": list(sm.indices),
                "displacement": list(sm.displacement),
            }
            for sm in req.submeshes
        ],
        "sha": sha,
        "saved_at_ms": int(time.time() * 1000),
    }

    fn = _protools_safe_filename(req.model_path, sha)
    out_path = PROTOOLS_EDITS_DIR / fn
    tmp = out_path.with_suffix(".json.tmp")
    body = json.dumps(canon, separators=(",", ":")).encode("utf-8")
    tmp.write_bytes(body)
    os.replace(tmp, out_path)

    return {
        "ok": True,
        "sha": sha,
        "cache_path": str(out_path),
        "size": len(body),
    }


@app.get("/api/protools/{sha}")
def api_protools_fetch(sha: str):
    """Re-fetch a saved protools edit by its SHA (or SHA prefix)."""
    sha = (sha or "").lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", sha):
        raise HTTPException(400, "sha must be 8-64 hex chars")
    matches = list(PROTOOLS_EDITS_DIR.glob(f"*__{sha}.json"))
    if not matches:
        for cand in PROTOOLS_EDITS_DIR.glob("*.json"):
            try:
                d = json.loads(cand.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if str(d.get("sha", "")).startswith(sha):
                matches = [cand]
                break
    if not matches:
        raise HTTPException(404, f"no protools edit found for sha {sha!r}")
    out = matches[0]
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(500, f"protools edit read failed: {e}")
    return {
        "ok": True,
        "sha": data.get("sha", sha),
        "cache_path": str(out),
        "edit_payload": data,
    }


@app.get("/api/protools/list/{path:path}")
def api_protools_list(path: str):
    """List every persisted protools edit for a given host archive.

    Mirrors /api/sculpt/build_archive but is purpose-built for the Edit
    tab so callers don't need to filter sculpt results out client-side.
    """
    host = Path(path).name
    if not host:
        raise HTTPException(400, "path must include a filename")
    out_list = []
    for cand in PROTOOLS_EDITS_DIR.glob("*.json"):
        try:
            d = json.loads(cand.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        mp = str(d.get("model_path", ""))
        if host not in mp:
            continue
        out_list.append({
            "sha": d.get("sha"),
            "model_path": mp,
            "submesh_count": len(d.get("submeshes", [])),
            "saved_at_ms": int(d.get("saved_at_ms", 0)),
            "cache_path": str(cand),
        })
    out_list.sort(key=lambda r: r["saved_at_ms"], reverse=True)
    return {"ok": True, "host": host, "edits": out_list}


# ---------------------------------------------------------------------------- rigging
#
# Skeleton-edit + per-vertex weight + IK target persistence (2026-04-25).
# The Rig tab (`static/rig_panel.js`) drives interactive bone-pose edits,
# weight-paint strokes, and IK target placement in the browser; the
# server only validates + persists JSON sidecars and (for boss-class
# meshes) computes auto-skin weights / FABRIK IK on demand.
#
# Wire format mirrors formats/rigging.py's encode_rig_payload — same
# envelope as sculpt (SHA-keyed sidecar in CACHE_DIR/rigs/) so the
# editor's deploy path can iterate both kinds in one pass.
#
# Limitation (v1): rig data is NOT round-tripped through the BML/AFS
# packer; full deploy waits for the NJ encoder (Phase A4). The
# /api/rig/build_archive endpoint surfaces sidecar metadata for the
# editor's listing UI but reports `deployable: false`.
# ----------------------------------------------------------------------------


class RigSaveReq(BaseModel):
    """Body of POST /api/rig/save.

    `model_path` is the source model (`<bml>#<inner>.nj` or bare `.nj`).
    `rig_payload` is the wire JSON produced by
    ``rigging_mod.encode_rig_payload`` — the server decodes-then-
    re-encodes once for validation + canonical SHA.
    """
    model_path: str = Field(..., description="source model path or <bml>#<inner>")
    rig_payload: dict = Field(..., description="rig wire payload (encode_rig_payload shape)")
    subdivide_level: int = Field(0, ge=0, le=8, description="subdivide depth applied before rigging")


def _rig_safe_filename(model_path: str, sha: str) -> str:
    """Map (model_path, sha) -> a safe filename inside RIG_CACHE_DIR.

    Mirrors `_sculpt_safe_filename` so the two cache dirs use the same
    naming convention.
    """
    safe = _CACHE_PATH_SEPS_RE.sub("__", model_path)
    if "." in safe.rsplit("__", 1)[-1]:
        safe = safe.rsplit(".", 1)[0]
    safe = _CACHE_PATH_SAFE_RE.sub("_", safe)
    return f"{safe}__{sha}.json"


@app.post("/api/rig/save")
def api_rig_save(req: RigSaveReq, request: Request):
    """Persist a rig (skeleton + weights + IK targets) for later re-use.

    Body schema:
        {
          "model_path":     "<bml>#<inner>.nj" | "<file>.nj",
          "rig_payload":    <rig JSON in encode_rig_payload shape>,
          "subdivide_level": <int>  (default 0)
        }

    Response:
        {
          "ok": true,
          "sha": "<16 hex chars>",
          "cache_path": "<absolute path>",
          "size": <bytes written>
        }

    Errors:
      400 — payload not a dict / fails decode
      413 — body > 32 MB
    """
    _enforce_body_size(request, MAX_RIG_SAVE_BODY)

    payload = req.rig_payload
    if not isinstance(payload, dict):
        raise HTTPException(400, "rig_payload must be a JSON object")
    try:
        bones, weights, ik_targets = rigging_mod.decode_rig_payload(payload)
    except (ValueError, KeyError, TypeError, base64.binascii.Error) as e:
        raise HTTPException(400, f"rig payload invalid: {e}")

    src_sha = str(payload.get("source_sha") or rigging_mod.compute_source_sha(
        req.model_path.encode("utf-8")
    ))
    canon = rigging_mod.encode_rig_payload(
        source_path=req.model_path,
        source_sha=src_sha,
        bones=bones,
        weights=weights,
        ik_targets=ik_targets,
        subdivide_level=req.subdivide_level,
    )
    sha = canon["sha"]

    fn = _rig_safe_filename(req.model_path, sha)
    out_path = RIG_CACHE_DIR / fn
    # Use a sibling temp filename rather than ``with_suffix(".json.tmp")``;
    # the latter raised in earlier observations under live uvicorn even
    # though tests pass — sidestep entirely.
    tmp = RIG_CACHE_DIR / (fn + ".tmp")
    body = json.dumps(canon, separators=(",", ":")).encode("utf-8")
    # Defensive: ensure parent dir survived a manual cache wipe between
    # server startup and this call.
    RIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(body)
    os.replace(tmp, out_path)

    return {
        "ok": True,
        "sha": sha,
        "cache_path": str(out_path),
        "size": len(body),
    }


@app.get("/api/rig/{sha}")
def api_rig_fetch(sha: str):
    """Re-fetch a previously-saved rig by its SHA.

    Globs the cache dir for the file, then returns the full payload.
    Mirrors /api/sculpt/<sha> behaviour and error conditions.
    """
    sha = (sha or "").lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", sha):
        raise HTTPException(400, "sha must be 8-64 hex chars")
    matches = list(RIG_CACHE_DIR.glob(f"*__{sha}.json"))
    if not matches:
        for cand in RIG_CACHE_DIR.glob("*.json"):
            try:
                d = json.loads(cand.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if str(d.get("sha", "")).startswith(sha):
                matches = [cand]
                break
    if not matches:
        raise HTTPException(404, f"no rig found for sha {sha!r}")
    out = matches[0]
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(500, f"rig read failed: {e}")
    return {
        "ok": True,
        "sha": data.get("sha", sha),
        "cache_path": str(out),
        "rig_payload": data,
    }


class RigAutoSkinReq(BaseModel):
    """Body of POST /api/rig/auto_skin.

    Computes initial weights for a freshly-loaded model. The frontend
    fetches the skinned mesh once via /api/model_skinned, then if the
    user clicks "Auto-skin" we recompute weights from scratch using the
    bone-pose world matrices.
    """
    model_path: str = Field(..., description="source model path or <bml>#<inner>")
    inner_idx: int = Field(0, ge=0, description="inner index for nested archives")
    algorithm: str = Field("distance", description="distance | heat")
    falloff: float = Field(4.0, gt=0.0, le=16.0, description="distance falloff exponent")
    iterations: int = Field(8, ge=0, le=64, description="heat smoothing iterations")
    max_influences: int = Field(rigging_mod.MAX_INFLUENCES, ge=1, le=8)


@app.post("/api/rig/auto_skin")
def api_rig_auto_skin(req: RigAutoSkinReq):
    """Compute auto-skin weights for the given model.

    Loads the skinned mesh via the existing parse cache, then computes
    weights for every submesh and returns them as a list of base64-
    packed (indices, weights) pairs.

    Response shape:
        {
          "ok": true,
          "algorithm": "distance" | "heat",
          "model_path": "<bml>#<inner>.nj",
          "weights": [
            {"submesh_idx": <int>, "vertex_count": <int>,
             "indices_b64": "<b64>", "weights_b64": "<b64>",
             "max_influences": <int>}, ...
          ]
        }
    """
    if req.algorithm not in rigging_mod.VALID_AUTOSKIN:
        raise HTTPException(
            400,
            f"algorithm must be one of {rigging_mod.VALID_AUTOSKIN}, got {req.algorithm!r}",
        )
    base, hash_inner = _split_inner_path(req.model_path)
    effective_inner = hash_inner
    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()

    if ext == ".bml":
        if not effective_inner:
            raise HTTPException(400, "BML model requires an inner")
        nj_bytes = _read_inner_nj_from_bml(p, effective_inner)
    elif ext == ".afs":
        if not effective_inner:
            raise HTTPException(400, "AFS model requires an inner")
        nj_bytes, _logical = _read_afs_inner_nj(p, effective_inner)
    elif ext == ".nj":
        sz = p.stat().st_size
        if sz > ASSET_PARSE_MAX_BYTES:
            raise HTTPException(413, f"model too large: {sz}")
        nj_bytes = p.read_bytes()
    else:
        raise HTTPException(400, f"unsupported model extension {ext!r}")

    try:
        meshes, bones = _parse_cache.parse_nj_skinned_cached(
            nj_bytes,
            file_key=_build_model_file_key(p, ext, effective_inner),
        )
    except ValueError as e:
        raise HTTPException(400, f"skinned parse failed: {e}")
    except Exception as e:  # pragma: no cover
        log.exception("auto_skin: skinned parse internal error")
        raise HTTPException(500, f"skinned parse internal error: {e}")

    if not bones:
        raise HTTPException(400, "model has no skeleton")

    # Build per-bone WORLD matrices using the source bind pose.
    bone_poses = [
        rigging_mod.BonePose(
            index=b.index,
            parent=b.parent,
            position=tuple(b.position),
            rotation_bams=tuple(b.rotation),
            scale=tuple(b.scale),
            eval_flags=b.eval_flags,
        )
        for b in bones
    ]
    bone_worlds = rigging_mod.compose_world_matrices(bone_poses)

    import numpy as np
    out_weights: list[dict] = []
    for sm_i, m in enumerate(meshes):
        if not m.vertices:
            sw = rigging_mod.empty_weights(sm_i, 0)
        else:
            # Vertices are bone-LOCAL in the skinned payload — to
            # auto-skin we need world-space positions, so transform
            # each by its owning bone's world matrix first.
            pts = np.empty((len(m.vertices), 3), dtype=np.float64)
            for vi, v in enumerate(m.vertices):
                bi = v.bone_idx
                if bi < 0 or bi >= len(bone_worlds):
                    pts[vi] = v.pos
                else:
                    mat = bone_worlds[bi]
                    pts[vi] = rigging_mod.transform_point(mat, v.pos)
            sw = rigging_mod.auto_skin(
                pts, bone_worlds,
                algorithm=req.algorithm,
                falloff=req.falloff,
                iterations=req.iterations,
                max_influences=req.max_influences,
            )
            sw.submesh_idx = sm_i
        out_weights.append({
            "submesh_idx": sm_i,
            "vertex_count": sw.vertex_count,
            "indices_b64": base64.b64encode(
                np.asarray(sw.bone_indices, dtype=np.int32).tobytes()
            ).decode("ascii"),
            "weights_b64": base64.b64encode(
                np.asarray(sw.weights, dtype=np.float32).tobytes()
            ).decode("ascii"),
            "max_influences": rigging_mod.MAX_INFLUENCES,
        })

    return {
        "ok": True,
        "algorithm": req.algorithm,
        "model_path": req.model_path,
        "weights": out_weights,
    }


class RigBuildArchiveReq(BaseModel):
    """Body of POST /api/rig/build_archive."""
    model_path: str = Field(..., description="host archive path or <bml>#<inner>")


@app.post("/api/rig/build_archive")
def api_rig_build_archive(req: RigBuildArchiveReq):
    """List rigs that belong to a given host archive.

    Response shape mirrors /api/sculpt/build_archive — metadata only,
    `deployable: false` until the NJ encoder phase A4 ships.
    """
    host = req.model_path.split("#", 1)[0]
    host_lower = host.lower()
    out_list: list[dict] = []
    for cand in RIG_CACHE_DIR.glob("*.json"):
        try:
            d = json.loads(cand.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sp = str(d.get("source_path", ""))
        sp_host = sp.split("#", 1)[0]
        if sp_host.lower() != host_lower:
            continue
        skel = d.get("skeleton", {}) or {}
        out_list.append({
            "sha": d.get("sha", ""),
            "model_path": sp,
            "subdivide_level": int(d.get("subdivide_level", 0)),
            "bone_count": len(skel.get("bones", [])),
            "weight_submesh_count": len(d.get("weights", [])),
            "ik_count": len(d.get("ik_targets", [])),
            "size": cand.stat().st_size,
            "saved_at_ms": int(d.get("saved_at_ms", 0)),
        })
    out_list.sort(key=lambda r: r["saved_at_ms"], reverse=True)
    return {
        "ok": True,
        "host": Path(host).name,
        "deployable": False,
        "deploy_pending": "NJ encoder (phase A4)",
        "rigs": out_list,
    }


# ---------------------------------------------------------------------------- per-mesh texture binding
#
# `/api/model_textures/{path}` returns the resolved texture-slot binding
# for a model: the list of NJTL entries (texture-name slots), the list
# of XVMH tile records, and the resulting binding array that the
# frontend uses to assign one texture per submesh.
#
# The mapping logic:
#   * Every textured PSOBB model has a `NJTL` chunk listing
#     `slot_index → texture_name`. The slot_index is the same number
#     each submesh stamps into its `material_id` field.
#   * The model's sibling XVM archive (`<inner>.xvm`) holds one XVR
#     record per texture, in the SAME order the writer's TextureManager
#     emitted them — so XVR record `i` corresponds to NJTL slot `i`.
#     XVR records carry only a numeric ID, not a name; the NJTL slot
#     index IS the natural match.
#   * For each unique `material_id` that appears on any submesh, we
#     resolve `material_id → tile_index = material_id` (positional
#     match) and surface both the NJTL name and the XVR record's
#     dimensions for diagnostic display.
#
# When the names DO match by string (some models reference named
# textures from a sibling XVM that has its own ordering), we'd want a
# name-based fall-back. PSOBB.IO's writer guarantees positional match
# in 100% of the BML inners we surveyed (see
# AGENT_TEXTURE_BINDING_REPORT.md), so the positional path is the
# default; we surface a `name_match: true|false` flag for future use.

# Cap on NJTL/XVMH list size — same kind of paranoid bound as elsewhere
# in this module. Real models top out around 16 textures.
_MAX_TEXTURES_PER_MODEL = 256


def _list_xvmh_records(xvm_bytes: bytes) -> list[dict]:
    """Walk an XVMH archive and return per-XVR record metadata.

    Returns a list of dicts ``{tile_index, id, width, height, fmt}``,
    one per XVRT record, in the order they appear in the archive.

    Raises:
        HTTPException(400): not a valid XVMH (no magic / declared count
            disagrees with on-disk count).
    """
    if len(xvm_bytes) < 0x40 or xvm_bytes[:4] != XVM_MAGIC:
        raise HTTPException(400, "XVMH magic not found in texture archive")
    declared_count = struct.unpack_from("<I", xvm_bytes, 0x08)[0]
    out: list[dict] = []
    pos = 0x40
    n = len(xvm_bytes)
    idx = 0
    while pos + 0x40 <= n:
        if xvm_bytes[pos:pos + 4] != XVRT_MAGIC:
            # Some XVMs may pad between records; skip until we find
            # the next XVRT or end of buffer.
            next_pos = xvm_bytes.find(XVRT_MAGIC, pos)
            if next_pos < 0:
                break
            pos = next_pos
            continue
        # XVR header layout (from xvr_codec.py docstring):
        #   +0x08: format1 (u32)
        #   +0x0C: format2 (u32)  — the editor's "fmt" field uses this
        #   +0x10: id/hash (u32)
        #   +0x14: width  (u16)
        #   +0x16: height (u16)
        #   +0x18: data size (u32)
        body_size = struct.unpack_from("<I", xvm_bytes, pos + 0x04)[0]
        fmt = struct.unpack_from("<I", xvm_bytes, pos + 0x0C)[0]
        tex_id = struct.unpack_from("<I", xvm_bytes, pos + 0x10)[0]
        w, h = struct.unpack_from("<HH", xvm_bytes, pos + 0x14)
        data_size = struct.unpack_from("<I", xvm_bytes, pos + 0x18)[0]
        out.append({
            "tile_index": idx,
            "id": int(tex_id),
            "width": int(w),
            "height": int(h),
            "fmt": int(fmt),
            # No texture name stored in XVMH/XVR — the matching name
            # comes from the NJTL slot at the same index.
        })
        # Advance past the entire record. The body_size field at +0x04
        # is "size of rest of record" = 0x38 + data_size, so the full
        # record length is 8 (magic+size header) + body_size.
        pos += 8 + body_size
        idx += 1
        if idx > _MAX_TEXTURES_PER_MODEL:
            raise HTTPException(
                400, f"XVMH record count exceeds sanity cap ({_MAX_TEXTURES_PER_MODEL})"
            )
    if declared_count != len(out):
        # Don't fail — just record the discrepancy. The frontend can
        # still bind on the records we DID find.
        log.warning(
            "XVMH declared %d records, found %d on disk", declared_count, len(out),
        )
    return out


def _build_records_from_sibling_archive(arc) -> list[dict]:
    """Build the same record-dict shape as _list_xvmh_records from a
    magic-sniffed sibling archive (PVM/GVM/PVR/GVR/XVM).

    Tile widths and heights come from a one-shot decode pass on each
    inner record. Records that fail to decode (unknown px_format,
    truncation) get reported with width/height=0 and fmt=-1 so the
    binding loop still produces a row — the frontend draws a magenta
    placeholder for those tiles.
    """
    records: list[dict] = []
    tile_names = arc.list_tiles()
    for idx, _name in enumerate(tile_names):
        try:
            inner = arc.extract_tile(idx)
        except (IndexError, KeyError) as e:
            log.debug("sibling extract_tile failed: %s", e)
            continue
        w = h = 0
        fmt = -1
        if arc.magic in ("PVR", "PVM"):
            pvrt_off = inner.find(b"PVRT")
            if pvrt_off >= 0 and pvrt_off + 0x10 <= len(inner):
                fmt = int(inner[pvrt_off + 0x08])
                try:
                    w, h = struct.unpack_from("<HH", inner, pvrt_off + 0x0C)
                except struct.error:
                    w = h = 0
        elif arc.magic in ("GVR", "GVM"):
            gvrt_off = inner.find(b"GVRT")
            if gvrt_off >= 0 and gvrt_off + 0x10 <= len(inner):
                fmt = int(inner[gvrt_off + 0x09])
                try:
                    w, h = struct.unpack_from(">HH", inner, gvrt_off + 0x0C)
                except struct.error:
                    w = h = 0
        elif arc.magic == "XVM":
            xvrt_off = inner.find(b"XVRT")
            if xvrt_off >= 0 and xvrt_off + 0x18 <= len(inner):
                fmt = int(struct.unpack_from("<I", inner, xvrt_off + 0x0C)[0])
                w, h = struct.unpack_from("<HH", inner, xvrt_off + 0x14)
        records.append({
            "tile_index": idx,
            "id": 0,
            "width": int(w),
            "height": int(h),
            "fmt": int(fmt),
        })
    return records


# ---------------------------------------------------------------------------
# BML-wide per-inner texture offset table (2026-04-26)
# ---------------------------------------------------------------------------
# Phantasmal-style cumulative-shift table: when a BML packs multiple
# .nj/.xj inners and the composite renderer flattens them into one scene
# graph, each inner's tile-id range MUST be disjoint from its siblings'.
# Phantasmal's `CharacterClassAssetLoader.kt:88-98` (`shiftTextureIds`)
# does the same thing for the body+head+hair pipeline — for inner i,
# every mesh.textureId is offset by `sum(tile_count[0..i-1])`.
#
# We compute the per-inner XVMH record count once here so the composite
# client can derive the right shift without N round-trips.

def _compute_bml_inner_tex_offsets(bml_path: Path) -> list[dict]:
    """Walk every .nj/.xj inner of a BML and report its inline tile count.

    Returns a list with one entry per inner (in the order they appear
    in the BML's file table):

        [
            {"name": "boss1_s_nb_dragon.nj",
             "tile_count": 13,
             "cumulative_offset": 0},
            {"name": "boss1_s_sd_dragon.nj",
             "tile_count": 0,
             "cumulative_offset": 13},
            ...
        ]

    Inners with no inline XVM (most common — only ONE inner per BML
    typically carries the textures) report ``tile_count = 0``. The
    cumulative offset is computed left-to-right; the composite renderer
    can use it to shift inner-N's material_id to ``mid + offset_N``
    when binding against a shared XVM pool.

    Quietly returns ``[]`` for a BML that fails to parse — the caller
    should treat that as "no offsets known, fall through to default".
    """
    try:
        blob = bml_path.read_bytes()
        entries = parse_bml(blob)
    except (OSError, ValueError):
        return []

    out: list[dict] = []
    cumulative = 0
    for entry in entries:
        ext = Path(entry.name).suffix.lower()
        if ext not in (".nj", ".xj"):
            continue
        tile_count = 0
        if entry.has_texture:
            try:
                xvm = extract_bml_texture(blob, entry.name, timeout=TIMEOUT_BML_PRS)
                if xvm:
                    records = _list_xvmh_records(xvm)
                    tile_count = len(records)
            except (HTTPException, ValueError, RuntimeError):
                tile_count = 0
        out.append({
            "name": entry.name,
            "tile_count": tile_count,
            "cumulative_offset": cumulative,
        })
        cumulative += tile_count
    return out


# ---------------------------------------------------------------------------
# Texture-binding LRU cache (Phase D follow-up, 2026-04-25)
# ---------------------------------------------------------------------------
# `_build_model_texture_binding` accounts for ~1031 ms of dragon's warm
# /api/model_mesh path (NJTL parse + cross-archive lookup per slot per
# request). The work is fully deterministic in (model_path, inner,
# nj_bytes mtime), so we wrap it in an LRU keyed on
# (path, inner, mtime_ns). Composes with the parse cache layer above.
#
# Cache layering (top-to-bottom — outermost serves first):
#   manifest._NEWEST_MTIME_CACHE   — install-tree mtime, 60 s TTL
#   formats.bml._PRS_INNER_CACHE   — decompressed BML inner blobs (64 MB)
#   parse_cache._PARSE_CACHE        — parsed XjMesh / XjBone lists (256 MB)
#   THIS LAYER (_BINDING_CACHE)     — NJTL→XVMH binding dicts (32 MB / 256)
#                                     + disk tier at cache/binding/v1/ (64 MB)
#   server._NJM_DECOMPRESS_CACHE    — decompressed NJM blobs (32 MB)
#
# Eviction: LRU by total bytes OR entry count (whichever cap fires first).
# Invalidation: implicit via mtime_ns in the key — re-deploys land in a
# newer mtime, so the next call computes a new entry. Cross-archive index
# updates (texture_index re-scan) are NOT in the key, however; see the
# stats/clear endpoints for manual flush.
#
# Disk tier (Item 5 of finishing-line, 2026-04-25): the in-memory LRU
# evaporates on process restart — dragon's first /api/model_mesh after
# restart re-pays the ~1 s binding compute. The disk tier persists each
# binding payload as JSON at cache/binding/v<schema>/<sha>.json with
# atomic tmp+rename writes; cold-after-restart hits a ~30-50 ms json
# load instead. Disable with PSO_DISABLE_DISK_BINDING_CACHE=1.

_BINDING_CACHE_MAX_ENTRIES = int(os.environ.get("PSO_BINDING_CACHE_ENTRIES", "256"))
_BINDING_CACHE_MAX_BYTES = int(os.environ.get("PSO_BINDING_CACHE_MB", "32")) * 1024 * 1024

# Disk persistence caps (Item 5, 2026-04-25):
#   - per-entry: 16 MB (binding payloads are typically <1 MB; cap is a
#     guard against pathological cases. Anything larger stays in-memory
#     only — pickle/JSON cost outweighs the 1 s recompute.)
#   - total disk usage: 64 MB (caps the on-disk footprint at 4× the
#     in-memory cap; cleanup is opportunistic, not pre-emptive).
_BINDING_DISK_PERSIST_MAX_BYTES = int(
    os.environ.get("PSO_BINDING_DISK_PER_ENTRY_MB", "16"),
) * 1024 * 1024
_BINDING_DISK_TOTAL_MAX_BYTES = int(
    os.environ.get("PSO_BINDING_DISK_TOTAL_MB", "64"),
) * 1024 * 1024

_BINDING_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_BINDING_CACHE_LOCK = threading.Lock()
_BINDING_CACHE_BYTES = 0
_BINDING_HITS = 0          # in-memory cache hits
_BINDING_HITS_DISK = 0     # disk-tier hits (warm-restart, pre-mem-fill)
_BINDING_MISSES = 0


def _binding_cache_key(
    bml_path: Path,
    outer_ext: str,
    effective_inner: Optional[str],
) -> Optional[tuple]:
    """Build a stable LRU key for one binding request.

    Returns None when the source file's stat fails (cache bypass — caller
    falls through to a non-cached compute, which is correct but slow).
    The key includes outer_ext so a model that swaps .nj↔.xj on disk via
    rename gets a fresh entry rather than a stale cross-archive lookup.
    """
    try:
        st = bml_path.stat()
    except OSError:
        return None
    return (str(bml_path), int(st.st_mtime_ns), int(st.st_size),
            outer_ext, effective_inner)


def _binding_estimate_bytes(payload: dict) -> int:
    """Estimate bytes for cache eviction accounting.

    We don't pickle (the way parse_cache does) because binding payloads
    are small, JSON-shaped dicts (~1-50 KB even for the most reference-
    heavy models). A cheap recursive sum of string + container overheads
    keeps the estimator at <0.1 ms per insert.
    """
    try:
        # JSON-encoded length is a tight upper bound on RAM for these
        # ASCII-heavy dicts (no float-density penalty since we only have
        # tens of fields per entry, all plain strings/ints).
        return len(json.dumps(payload))
    except (TypeError, ValueError):
        return 4096  # conservative default for unserialisable shapes


def _binding_purge_until_under_caps_locked() -> None:
    """Evict LRU entries until both caps satisfied.

    Called with `_BINDING_CACHE_LOCK` held. We always keep at least one
    entry — same defensive policy as parse_cache so a single oversize
    insert doesn't trigger an infinite eviction loop.
    """
    global _BINDING_CACHE_BYTES
    while ((_BINDING_CACHE_BYTES > _BINDING_CACHE_MAX_BYTES
            or len(_BINDING_CACHE) > _BINDING_CACHE_MAX_ENTRIES)
           and len(_BINDING_CACHE) > 1):
        try:
            _evicted_key, value = _BINDING_CACHE.popitem(last=False)
        except KeyError:
            break
        _BINDING_CACHE_BYTES -= int(value[1])


# ---------------------------------------------------------------------------
# Binding cache disk persistence (Item 5, 2026-04-25)
# ---------------------------------------------------------------------------
# Mirrors the skinned-payload disk layout: JSON file per (path, mtime_ns,
# size, outer_ext, inner) keyed by sha256 of repr(key). Atomic
# tmp+rename writes; corrupt files self-delete on read.
#
# Why JSON not pickle: binding payloads are already JSON-shape (lists of
# small dicts, ASCII strings). JSON keeps the on-disk file `cat`-able
# for diagnostics and avoids pickle's import-eval security surface.
#
# Disable with PSO_DISABLE_DISK_BINDING_CACHE=1 for benchmarking the
# cold path or when disk space is tight.

def _binding_disk_path(key: tuple) -> Optional[Path]:
    """Compute on-disk JSON path for a binding cache key.

    Returns None on disable / dir-creation failure (graceful degradation
    — in-memory cache still works). Schema-versioned subdir lets a
    payload-shape change auto-invalidate stale files without manual
    cleanup (bump BINDING_CACHE_SCHEMA).
    """
    if os.environ.get("PSO_DISABLE_DISK_BINDING_CACHE", "0") in (
        "1", "true", "True",
    ):
        return None
    try:
        base = BINDING_CACHE_DIR / f"v{BINDING_CACHE_SCHEMA}"
        base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("binding_cache: dir creation failed: %s", e)
        return None
    h = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return base / f"{h}.json"


def _binding_disk_load(key: tuple) -> Optional[Tuple[dict, int]]:
    """Read a cached binding payload from disk; None on miss/corrupt.

    The on-disk shape is ``{"key": [...], "payload": {...}}``; we
    re-verify the embedded key matches before serving so a sha
    collision (vanishingly unlikely but worth catching) can't silently
    serve stale bytes. JSON round-trips tuples as lists, so we
    tuple-ise both sides for the comparison.
    """
    p = _binding_disk_path(key)
    if p is None or not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            raw = f.read()
        obj = json.loads(raw)
    except (OSError, ValueError) as e:
        log.warning("binding_cache: corrupt JSON %s removed: %s", p.name, e)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    stored_key = obj.get("key") if isinstance(obj, dict) else None
    payload = obj.get("payload") if isinstance(obj, dict) else None
    if stored_key is None or payload is None:
        log.warning("binding_cache: malformed JSON shape at %s; deleting",
                    p.name)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    if tuple(stored_key) != tuple(key):
        log.warning("binding_cache: key mismatch at %s; deleting", p.name)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    try:
        size = p.stat().st_size
    except OSError:
        size = len(raw)
    return payload, int(size)


def _binding_disk_write(key: tuple, payload: dict, size_hint: int) -> None:
    """Persist a binding payload via atomic tmp+rename; silent on errors.

    Skipped when ``size_hint`` exceeds ``_BINDING_DISK_PERSIST_MAX_BYTES``
    (16 MB by default) — pathological cases recompute faster than they
    pickle. The total-disk cap is enforced opportunistically in
    ``_binding_disk_prune_total_locked``; this function only checks the
    per-entry cap.
    """
    if size_hint > _BINDING_DISK_PERSIST_MAX_BYTES:
        return
    p = _binding_disk_path(key)
    if p is None:
        return
    tmp = p.with_suffix(".json.tmp")
    try:
        body = json.dumps(
            {"key": list(key), "payload": payload},
            separators=(",", ":"),
        )
        with tmp.open("w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, p)
    except (OSError, ValueError, TypeError) as e:
        log.warning("binding_cache: disk write failed for %s: %s",
                    p.name, e)
        try:
            tmp.unlink()
        except OSError:
            pass
        return
    # Opportunistic total-cap prune: when the dir grows past the cap,
    # delete the oldest files (by mtime) until back under. Cheap O(N)
    # scan — N is at most a few hundred files — runs only after writes,
    # not on every read, so no hit on the warm path.
    try:
        base = p.parent
        files = []
        total = 0
        for child in base.iterdir():
            if child.is_file() and child.suffix == ".json":
                try:
                    st = child.stat()
                except OSError:
                    continue
                files.append((st.st_mtime_ns, st.st_size, child))
                total += st.st_size
        if total > _BINDING_DISK_TOTAL_MAX_BYTES:
            files.sort()  # oldest first
            for mt, sz, child in files:
                if total <= _BINDING_DISK_TOTAL_MAX_BYTES:
                    break
                try:
                    child.unlink()
                    total -= sz
                except OSError:
                    pass
    except OSError:
        # Pruning is opportunistic — a stat-fail mid-walk is fine, the
        # next write will retry.
        pass


def _build_model_texture_binding_cached(
    bml_path: Path,
    outer_ext: str,
    effective_inner: Optional[str],
    nj_bytes: bytes,
    meshes: list,
) -> dict:
    """LRU+disk-cached variant of `_build_model_texture_binding`.

    Cache key: (abs_path, mtime_ns, size, outer_ext, inner).
    Lookup order:
      1. In-memory LRU keyed on the above — <5 ms per warm hit.
      2. On-disk JSON under ``cache/binding/v<schema>/`` —
         ~30-50 ms per hit (json.load + dict assembly), populates L1.
      3. Cold compute via ``_build_model_texture_binding`` —
         ~1 s for dragon-class models. Populates both tiers.

    Cache hit returns the SAME dict object — callers must not mutate
    (the /api/model_mesh path only reads).

    Falls through to the uncached builder when the source file's stat()
    fails (key=None). That path costs ~1 s per call but keeps the
    response correct on disconnected drives / live deletes.
    """
    global _BINDING_CACHE_BYTES, _BINDING_HITS, _BINDING_HITS_DISK
    global _BINDING_MISSES
    key = _binding_cache_key(bml_path, outer_ext, effective_inner)
    if key is None:
        return _build_model_texture_binding(
            bml_path, outer_ext, effective_inner, nj_bytes, meshes,
        )

    # --- L1: in-memory LRU
    with _BINDING_CACHE_LOCK:
        ent = _BINDING_CACHE.get(key)
        if ent is not None:
            _BINDING_CACHE.move_to_end(key)
            ent[2] = ent[2] + 1                  # bump hit count
            _BINDING_HITS += 1
            return ent[0]

    # --- L2: on-disk JSON
    disk_hit = _binding_disk_load(key)
    if disk_hit is not None:
        payload, byte_estimate = disk_hit
        with _BINDING_CACHE_LOCK:
            ent = _BINDING_CACHE.get(key)
            if ent is None:
                _BINDING_CACHE[key] = [payload, byte_estimate, 1]
                _BINDING_CACHE_BYTES += byte_estimate
                _binding_purge_until_under_caps_locked()
                _BINDING_HITS_DISK += 1
            else:
                # Race: another caller landed first.
                _BINDING_CACHE.move_to_end(key)
                ent[2] += 1
                payload = ent[0]
                _BINDING_HITS += 1
        return payload

    # --- Cold compute (~1 s for dragon-class models).
    payload = _build_model_texture_binding(
        bml_path, outer_ext, effective_inner, nj_bytes, meshes,
    )
    sz = _binding_estimate_bytes(payload)

    with _BINDING_CACHE_LOCK:
        # Race-safe re-check before insert.
        ent = _BINDING_CACHE.get(key)
        if ent is None:
            _BINDING_CACHE[key] = [payload, sz, 0]   # [obj, bytes, hits]
            _BINDING_CACHE_BYTES += sz
            _binding_purge_until_under_caps_locked()
            _BINDING_MISSES += 1
        else:
            _BINDING_CACHE.move_to_end(key)
            ent[2] += 1
            payload = ent[0]
            _BINDING_HITS += 1

    # Disk persist (outside the lock — disk I/O can block other
    # readers; no mutation of in-memory state below).
    _binding_disk_write(key, payload, sz)
    return payload


def _binding_cache_stats() -> dict:
    """Return a snapshot of binding-cache health for the stats endpoint.

    Same shape skeleton as parse_cache.cache_stats / skinned_payload_cache
    so the frontend can reuse a single dashboard widget. Item 5
    (2026-04-25) added an on-disk tier (cache/binding/v<schema>/) so
    cold-after-restart hits a 30-50 ms disk load instead of 1 s
    recompute. The ``hits_disk`` / ``hits_inmemory`` split + the
    ``disk_*`` keys surface that tier's effectiveness; legacy ``hits``
    remains the in-memory count for backward compatibility with any
    pre-existing scrapers/dashboards.
    """
    with _BINDING_CACHE_LOCK:
        entries = len(_BINDING_CACHE)
        total = _BINDING_CACHE_BYTES
        hits = _BINDING_HITS
        hits_disk = _BINDING_HITS_DISK
        misses = _BINDING_MISSES
        # Top-10 by hit count for debug ("which model is repeat-opening?").
        top: list = []
        for k, v in sorted(_BINDING_CACHE.items(),
                           key=lambda kv: kv[1][2], reverse=True)[:10]:
            path_str = str(k[0]) if k else ""
            basename = path_str.replace("\\", "/").rsplit("/", 1)[-1]
            top.append({
                "key": basename + ((":" + str(k[4])) if (len(k) > 4 and k[4]) else ""),
                "hits": int(v[2]),
                "bytes": int(v[1]),
            })

    # Disk usage — outside the lock; just stat-walks the schema dir.
    disk_entries: Optional[int] = None
    disk_bytes: Optional[int] = None
    try:
        base = BINDING_CACHE_DIR / f"v{BINDING_CACHE_SCHEMA}"
        if base.is_dir():
            disk_entries = 0
            disk_bytes = 0
            for child in base.iterdir():
                if child.is_file() and child.suffix == ".json":
                    disk_entries += 1
                    try:
                        disk_bytes += child.stat().st_size
                    except OSError:
                        pass
    except OSError:
        pass

    total_calls = hits + hits_disk + misses
    hit_rate = (hits + hits_disk) / total_calls if total_calls else 0.0
    return {
        "entries": entries,
        "bytes": total,
        "max_entries": _BINDING_CACHE_MAX_ENTRIES,
        "max_bytes": _BINDING_CACHE_MAX_BYTES,
        # Backward-compat alias (in-memory hits only).
        "hits": hits,
        # Split tiers + total disk view.
        "hits_inmemory": hits,
        "hits_disk": hits_disk,
        "misses": misses,
        "hit_rate": hit_rate,
        "disk_entries": disk_entries,
        "disk_bytes": disk_bytes,
        "schema": BINDING_CACHE_SCHEMA,
        "top_entries": top,
    }


def _binding_cache_clear(*, drop_disk: bool = True) -> dict:
    """Drop the binding cache (in-memory and optionally on-disk).

    Item 5 (2026-04-25): default now drops both tiers so the
    /api/binding_cache/clear endpoint actually clears the cache the user
    sees. Callers that want in-memory-only (e.g. unit tests measuring
    re-warm from disk) pass ``drop_disk=False``.
    """
    global _BINDING_CACHE_BYTES, _BINDING_HITS, _BINDING_HITS_DISK
    global _BINDING_MISSES
    with _BINDING_CACHE_LOCK:
        cleared_entries = len(_BINDING_CACHE)
        cleared_bytes = _BINDING_CACHE_BYTES
        _BINDING_CACHE.clear()
        _BINDING_CACHE_BYTES = 0
        _BINDING_HITS = 0
        _BINDING_HITS_DISK = 0
        _BINDING_MISSES = 0

    disk_files = 0
    disk_bytes_freed = 0
    if drop_disk:
        try:
            base = BINDING_CACHE_DIR / f"v{BINDING_CACHE_SCHEMA}"
            if base.is_dir():
                for child in base.iterdir():
                    if child.is_file() and child.suffix == ".json":
                        try:
                            sz = child.stat().st_size
                            child.unlink()
                            disk_files += 1
                            disk_bytes_freed += sz
                        except OSError:
                            pass
        except OSError:
            pass

    return {
        "cleared_entries": cleared_entries,
        "cleared_bytes": cleared_bytes,
        "disk_files_dropped": disk_files,
        "disk_bytes_freed": disk_bytes_freed,
    }


# ---------------------------------------------------------------------------- tile PNG cache
#
# Phase D Win 5 (2026-04-25). The /api/tile_png route is the dominant
# remaining bottleneck on a dragon-class first-open: 16 tiles × ~50-100 ms
# of XVR→PIL→PNG cost = ~1.6 s of texture wall-time even when every
# higher-level cache (parse / binding / NJM-decompress) hits warm. The
# *outputs* of /api/tile_png are tiny (tens of KB of PNG bytes per tile)
# and depend only on (file mtime+size, tile index) so they're a perfect
# fit for a cheap LRU + on-disk cache.
#
# Layering (full stack, top-down — outermost serves first):
#   manifest._NEWEST_MTIME_CACHE   — install-tree mtime, 60 s TTL
#   formats.bml._PRS_INNER_CACHE   — decompressed BML inner blobs (64 MB)
#   parse_cache._PARSE_CACHE       — parsed XjMesh / XjBone lists (256 MB)
#   server._BINDING_CACHE          — NJTL→XVMH binding dicts (32 MB / 256)
#   THIS LAYER (_TILE_PNG_CACHE)   — PNG bytes per (file, tile) (128 MB / 256)
#   server._NJM_DECOMPRESS_CACHE   — decompressed NJM blobs (32 MB)
#
# Eviction: LRU by total bytes OR entry count.
# Invalidation: implicit via mtime_ns in the key (re-deploys land in a
# new mtime so a stale hit can't survive a write). The disk pickle layer
# uses a sha2 of the key as filename so collision risk is vanishing.

_TILE_PNG_CACHE_MAX_ENTRIES = int(
    os.environ.get("PSO_TILE_PNG_CACHE_ENTRIES", "256"),
)
_TILE_PNG_CACHE_MAX_BYTES = int(
    os.environ.get("PSO_TILE_PNG_CACHE_MB", "128"),
) * 1024 * 1024

_TILE_PNG_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_TILE_PNG_CACHE_LOCK = threading.Lock()
_TILE_PNG_CACHE_BYTES = 0
_TILE_PNG_HITS_INMEMORY = 0
_TILE_PNG_HITS_DISK = 0
_TILE_PNG_MISSES = 0


def _tile_png_cache_key(
    work_path: Path,
    tile_idx: int,
    request_path: str,
) -> Optional[tuple]:
    """Build the LRU key for one tile-PNG request.

    Keys on the stat() of the materialized tile-source file (the
    decompressed BML-inner or AFS-inner sitting in the cache scratch dir
    when called via the `#`-syntax; the raw input path otherwise). That
    means a re-deploy or an inner-rewrite invalidates the entry on the
    next request, while every cross-archive fetch of the same `(bml,
    inner, tile)` triple shares one cache entry.

    `request_path` keeps the URL path verbatim so the same tile served
    via two different URL aliases (e.g. `<bml>#<inner>.xvm` vs. a flat
    XVM under DATA_DIR) still benefits from cache locality without
    accidentally serving the wrong bytes — the materializer points us
    at the same inode either way, and stat() resolves it.

    Returns None on stat failure — the caller falls through to the
    uncached path so a deleted file still 404s correctly.
    """
    try:
        st = work_path.stat()
    except OSError:
        return None
    return (
        str(work_path),
        int(st.st_mtime_ns),
        int(st.st_size),
        int(tile_idx),
        request_path,  # disambiguate aliasing if any
    )


def _tile_png_disk_path(key: tuple) -> Optional[Path]:
    """Compute on-disk PNG path for a cache key, creating the schema dir.

    Returns None on disk-cache disable / dir-creation failure (graceful
    degradation — in-memory cache still works).
    """
    if os.environ.get("PSO_DISABLE_DISK_TILE_PNG_CACHE", "0") in ("1", "true", "True"):
        return None
    try:
        base = TILE_PNG_CACHE_DIR / f"v{TILE_PNG_CACHE_SCHEMA}"
        base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("tile_png_cache: dir creation failed: %s", e)
        return None
    h = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return base / f"{h}.png"


def _tile_png_purge_until_under_caps_locked() -> None:
    """Evict LRU entries until both caps satisfied; always keep one entry.

    Same eviction policy shape as parse_cache and the binding cache:
    keep at least one entry so a single oversize insert can't trigger
    an infinite eviction loop.
    """
    global _TILE_PNG_CACHE_BYTES
    while ((_TILE_PNG_CACHE_BYTES > _TILE_PNG_CACHE_MAX_BYTES
            or len(_TILE_PNG_CACHE) > _TILE_PNG_CACHE_MAX_ENTRIES)
           and len(_TILE_PNG_CACHE) > 1):
        try:
            _evicted_key, value = _TILE_PNG_CACHE.popitem(last=False)
        except KeyError:
            break
        _TILE_PNG_CACHE_BYTES -= int(value[1])


def _tile_png_load_from_disk(key: tuple) -> Optional[bytes]:
    """Read a cached PNG from disk; None on miss / corrupt / disabled.

    Corrupt files are deleted in-place so the next compute can rebuild
    cleanly. A PNG without the signature header is treated as corrupt
    (truncated write).
    """
    p = _tile_png_disk_path(key)
    if p is None or not p.is_file():
        return None
    try:
        data = p.read_bytes()
    except OSError as e:
        log.warning("tile_png_cache: read failed for %s: %s", p.name, e)
        return None
    # PNG magic = 89 50 4E 47 0D 0A 1A 0A — reject anything else.
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        log.warning("tile_png_cache: bad magic at %s; deleting", p.name)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    return data


def _tile_png_write_to_disk(key: tuple, png_bytes: bytes) -> None:
    """Persist the PNG to disk via tmp+rename; silent no-op on any error.

    Atomic-rename means a kill mid-write leaves either the previous
    cache entry or no entry, never a half-PNG that we'd subsequently
    serve and fail to decode in the browser.
    """
    p = _tile_png_disk_path(key)
    if p is None:
        return
    tmp = p.with_suffix(".png.tmp")
    try:
        with tmp.open("wb") as f:
            f.write(png_bytes)
        os.replace(tmp, p)
    except OSError as e:
        log.warning("tile_png_cache: write failed for %s: %s", p.name, e)
        try:
            tmp.unlink()
        except OSError:
            pass


def _serve_tile_png_cached(
    work_path: Path,
    tile_idx: int,
    request_path: str,
    fetch_fn,  # () -> Path: returns the on-disk png from extract_tiles
):
    """Look up tile bytes via the cache; populate on miss.

    Returns either a `bytes` payload (cache hit) or whatever
    ``fetch_fn`` returned on miss — the route handler is responsible
    for converting `Path` → `FileResponse` in the legacy code path.
    On a successful cache MISS we read the PNG bytes off disk and
    populate both the in-memory LRU and the on-disk PNG cache.

    The caller is responsible for input validation (idx range, file
    existence) so this function focuses purely on cache behaviour.
    """
    global _TILE_PNG_CACHE_BYTES, _TILE_PNG_HITS_INMEMORY
    global _TILE_PNG_HITS_DISK, _TILE_PNG_MISSES

    key = _tile_png_cache_key(work_path, tile_idx, request_path)
    if key is None:
        # stat failed; bypass cache (and let the route handler 404 if
        # the file truly doesn't exist).
        return None, fetch_fn()

    # --- L1: in-memory LRU
    with _TILE_PNG_CACHE_LOCK:
        ent = _TILE_PNG_CACHE.get(key)
        if ent is not None:
            _TILE_PNG_CACHE.move_to_end(key)
            ent[2] = ent[2] + 1                  # bump hit count
            _TILE_PNG_HITS_INMEMORY += 1
            return ent[0], None

    # --- L2: on-disk PNG cache
    disk_bytes = _tile_png_load_from_disk(key)
    if disk_bytes is not None:
        with _TILE_PNG_CACHE_LOCK:
            ent = _TILE_PNG_CACHE.get(key)
            if ent is None:
                _TILE_PNG_CACHE[key] = [disk_bytes, len(disk_bytes), 1]
                _TILE_PNG_CACHE_BYTES += len(disk_bytes)
                _tile_png_purge_until_under_caps_locked()
                _TILE_PNG_HITS_DISK += 1
            else:
                # Race: another thread populated us first.
                _TILE_PNG_CACHE.move_to_end(key)
                ent[2] += 1
                disk_bytes = ent[0]
                _TILE_PNG_HITS_INMEMORY += 1
        return disk_bytes, None

    # --- Cold miss: invoke the original fetch path. The legacy code
    # returns a Path object pointing at a tile PNG inside the
    # extract_tiles cache subdir.
    tile_path = fetch_fn()
    if tile_path is None or not isinstance(tile_path, Path) or not tile_path.is_file():
        # fetch_fn already raised an HTTPException (likely 404/500)
        # before returning; nothing to cache.
        return None, tile_path

    try:
        png_bytes = tile_path.read_bytes()
    except OSError as e:
        log.warning("tile_png_cache: failed to read %s: %s", tile_path, e)
        return None, tile_path

    with _TILE_PNG_CACHE_LOCK:
        ent = _TILE_PNG_CACHE.get(key)
        if ent is None:
            _TILE_PNG_CACHE[key] = [png_bytes, len(png_bytes), 1]
            _TILE_PNG_CACHE_BYTES += len(png_bytes)
            _tile_png_purge_until_under_caps_locked()
            _TILE_PNG_MISSES += 1
        else:
            # Race-safe re-check; another caller populated us first.
            _TILE_PNG_CACHE.move_to_end(key)
            ent[2] += 1
            png_bytes = ent[0]
            _TILE_PNG_HITS_INMEMORY += 1

    # Disk persist (outside the lock — disk I/O could block).
    _tile_png_write_to_disk(key, png_bytes)
    return png_bytes, None


def _tile_png_cache_stats() -> dict:
    """Return tile-png cache health for /api/tile_png_cache/stats.

    Same shape skeleton as parse_cache.cache_stats / binding_cache_stats
    so the frontend can render all three with one widget.
    """
    with _TILE_PNG_CACHE_LOCK:
        entries = len(_TILE_PNG_CACHE)
        total = _TILE_PNG_CACHE_BYTES
        hits_mem = _TILE_PNG_HITS_INMEMORY
        hits_disk = _TILE_PNG_HITS_DISK
        misses = _TILE_PNG_MISSES
        # Top-10 by hit count for debug ("which tile is hammering us?").
        top: list = []
        for k, v in sorted(_TILE_PNG_CACHE.items(),
                           key=lambda kv: kv[1][2], reverse=True)[:10]:
            path_str = str(k[0]) if k else ""
            basename = path_str.replace("\\", "/").rsplit("/", 1)[-1]
            top.append({
                "key": f"{basename}#{k[3]}" if len(k) > 3 else basename,
                "hits": int(v[2]),
                "bytes": int(v[1]),
            })

    # Disk usage — outside the lock; just stat-walks.
    disk_entries: Optional[int] = None
    disk_bytes: Optional[int] = None
    try:
        base = TILE_PNG_CACHE_DIR / f"v{TILE_PNG_CACHE_SCHEMA}"
        if base.is_dir():
            disk_entries = 0
            disk_bytes = 0
            for child in base.iterdir():
                if child.is_file() and child.suffix == ".png":
                    disk_entries += 1
                    try:
                        disk_bytes += child.stat().st_size
                    except OSError:
                        pass
    except OSError:
        pass

    total_calls = hits_mem + hits_disk + misses
    hit_rate = (hits_mem + hits_disk) / total_calls if total_calls else 0.0
    return {
        "entries": entries,
        "bytes": total,
        "max_entries": _TILE_PNG_CACHE_MAX_ENTRIES,
        "max_bytes": _TILE_PNG_CACHE_MAX_BYTES,
        "disk_entries": disk_entries,
        "disk_bytes": disk_bytes,
        "hits_inmemory": hits_mem,
        "hits_disk": hits_disk,
        "misses": misses,
        "hit_rate": hit_rate,
        "top_entries": top,
        "schema": TILE_PNG_CACHE_SCHEMA,
    }


def _tile_png_cache_clear(*, drop_disk: bool = True) -> dict:
    """Drop the tile-png cache (in-memory + on-disk)."""
    global _TILE_PNG_CACHE_BYTES, _TILE_PNG_HITS_INMEMORY
    global _TILE_PNG_HITS_DISK, _TILE_PNG_MISSES
    with _TILE_PNG_CACHE_LOCK:
        cleared_entries = len(_TILE_PNG_CACHE)
        cleared_bytes = _TILE_PNG_CACHE_BYTES
        _TILE_PNG_CACHE.clear()
        _TILE_PNG_CACHE_BYTES = 0
        _TILE_PNG_HITS_INMEMORY = 0
        _TILE_PNG_HITS_DISK = 0
        _TILE_PNG_MISSES = 0

    disk_files = 0
    disk_bytes_freed = 0
    if drop_disk:
        try:
            base = TILE_PNG_CACHE_DIR / f"v{TILE_PNG_CACHE_SCHEMA}"
            if base.is_dir():
                for child in base.iterdir():
                    if child.is_file() and child.suffix in (".png", ".tmp"):
                        try:
                            sz = child.stat().st_size
                        except OSError:
                            sz = 0
                        try:
                            child.unlink()
                            disk_files += 1
                            disk_bytes_freed += sz
                        except OSError:
                            pass
        except OSError:
            pass

    return {
        "cleared_entries": cleared_entries,
        "cleared_bytes": cleared_bytes,
        "cleared_disk_files": disk_files,
        "cleared_disk_bytes": disk_bytes_freed,
    }


def _bml_obvious_textured_inner(bml_path: Path) -> Optional[str]:
    """Return the unambiguous inner model name of a BML, or None.

    Mirrors psov2's per-asset scripts, which always name a concrete
    inner + its paired texture instead of asking a generic resolver
    "what is THIS container's texture?". A BML is unambiguous when it
    carries exactly one entry that has an inline texture appendix; if no
    entry is textured but there is exactly one model entry, fall back to
    that. Returns None when the BML is genuinely ambiguous (0 or >1
    candidate) so the caller still produces a clean diagnostic rather
    than guessing.
    """
    try:
        entries = parse_bml(bml_path.read_bytes())
    except Exception:  # pragma: no cover - defensive; resolver bails cleanly
        return None
    textured = [e for e in entries if getattr(e, "has_texture", False)]
    if len(textured) == 1:
        return textured[0].name
    models = [
        e for e in entries
        if Path(e.name).suffix.lower() in (".nj", ".xj")
    ]
    if len(models) == 1:
        return models[0].name
    return None


def _build_model_texture_binding(
    bml_path: Path,
    outer_ext: str,
    effective_inner: Optional[str],
    nj_bytes: bytes,
    meshes: list,
) -> dict:
    """Compute the NJTL→XVMH→material_id binding for one model.

    Returns a dict with keys:
        njtl:      [{"slot": i, "name": str}, ...]
        xvmh:      [{"tile_index": i, "id": int, "width": w, "height": h,
                     "fmt": f, "name": str (from NJTL[i] if present)},
                    ...]
        binding:   [{"material_id": m, "tile_index": t, "missing": bool,
                     "name": str}, ...] for each unique material_id
        name_match: bool — True if every NJTL slot's name appears in
                    the XVMH side. Today we have no way to verify this
                    (XVR records carry no name field), so this stays
                    True if the counts match.

    Raises HTTPException on irrecoverable input (invalid path / archive
    not extractable). A model with no NJTL chunk is NOT an error — we
    return empty lists with a synthetic "no NJTL" hint.
    """
    # Parse the NJTL chunk from the model bytes.
    try:
        njtl_entries = find_and_parse_njtl(nj_bytes) or []
    except ValueError as e:
        raise HTTPException(400, f"NJTL parse failed: {e}")

    # Discover the texture archive. For top-level .nj / .xj we look for
    # a sibling .xvm in the same directory (rare in PSOBB.IO — most
    # models live inside a BML); for BML inners we extract the inner's
    # paired XVM via extract_bml_texture.
    xvm_records: list[dict] = []
    xvm_error: Optional[str] = None
    if outer_ext == ".bml":
        # psov2 "the obvious inner" contract: a BML opened with no inner is
        # ambiguous in general, but when it carries exactly ONE textured
        # entry (or just one .nj/.xj model) the inner is unambiguous —
        # auto-select it instead of bailing with "no inner specified". The
        # model-mesh / model-textures endpoints already require an inner
        # (they raise 400 first), so this branch is a robustness net for
        # any path that reaches the resolver with effective_inner is None
        # (and the momoka-class single-inner BML is exactly this case).
        if not effective_inner:
            auto = _bml_obvious_textured_inner(bml_path)
            if auto is not None:
                effective_inner = auto
        if effective_inner:
            try:
                blob = bml_path.read_bytes()
                xvm = extract_bml_texture(blob, effective_inner, timeout=TIMEOUT_BML_PRS)
            except ValueError as e:
                xvm_error = f"BML texture extract failed: {e}"
                xvm = None
            except RuntimeError as e:
                xvm_error = f"BML texture extract failed: {e}"
                xvm = None
            except Exception as e:  # pragma: no cover - defensive net
                xvm_error = f"BML texture extract internal error: {e}"
                xvm = None
            if xvm is not None:
                try:
                    xvm_records = _list_xvmh_records(xvm)
                except HTTPException as e:
                    xvm_error = e.detail if hasattr(e, "detail") else str(e)
            elif xvm_error is None:
                xvm_error = "no texture sibling for this BML inner"
        else:
            xvm_error = "no inner specified — cannot resolve texture sibling"
    elif outer_ext == ".afs":
        # AFS-resident model: the inner blob has no sibling .xvm — its
        # textures live in the paired ``Item*Texture*.afs`` (or another
        # cross-archive). The per-mid loop below handles the lookup via
        # ``texture_index.lookup`` + ``cross_afs`` rows. We still record a
        # diagnostic so the response shows why ``xvmh`` is empty.
        xvm_error = "AFS-resident model — textures resolved via cross_afs lookup"
    else:
        # Top-level .nj/.xj — look for a same-stem .xvm next to it.
        sibling = bml_path.with_suffix(".xvm")
        if sibling.exists():
            try:
                xvm_records = _list_xvmh_records(sibling.read_bytes())
            except HTTPException as e:
                xvm_error = e.detail if hasattr(e, "detail") else str(e)
        else:
            # No same-stem .xvm; try magic-sniffed sibling discovery
            # (PVM/GVM/PRS-wrapped variants). The first sibling whose
            # record list is non-empty wins.
            try:
                discovered = _sibling_archives.discover_sibling_textures(bml_path)
            except Exception as e:  # pragma: no cover - defensive
                log.warning(
                    "discover_sibling_textures failed for %s: %s",
                    bml_path, e,
                )
                discovered = []
            chosen = None
            for arc in discovered:
                if arc.list_tiles():
                    chosen = arc
                    break
            if chosen is not None:
                xvm_error = None
                xvm_records = _build_records_from_sibling_archive(chosen)
            else:
                xvm_error = "no sibling .xvm found"

    # Annotate XVMH records with the NJTL-derived name (positional).
    for rec in xvm_records:
        ti = rec["tile_index"]
        rec["name"] = njtl_entries[ti].name if ti < len(njtl_entries) else ""

    # Compute the per-material binding. Each unique material_id seen on
    # any submesh becomes one binding row. The default mapping is
    # positional: tile_index = material_id. If the NJTL/XVMH lists
    # disagree in length, slots beyond min(|NJTL|,|XVMH|) get a
    # cross-BML lookup pass via formats/texture_index.py before
    # falling back to tile 0 on the frontend.
    seen_mids: set[int] = set()
    for m in meshes:
        seen_mids.add(int(getattr(m, "material_id", 0)))
    if not seen_mids:
        seen_mids.add(0)
    sorted_mids = sorted(seen_mids)

    n_njtl = len(njtl_entries)
    n_xvmh = len(xvm_records)
    binding: list[dict] = []
    cross_bml_uses: dict[str, list[dict]] = {}  # name -> [{bml, inner, xvr}]
    cross_afs_uses: dict[str, list[dict]] = {}  # name -> [{archive, inner_index, xvr_index}]
    host_basename = bml_path.name if outer_ext == ".bml" else None

    # Player-class positional fallback: ``plAbdy00.nj``, ``plKhai00.nj``,
    # etc. ship WITHOUT an NJTL chunk; their material slots bind
    # positionally against the matching ``pl<class>tex.afs``. Resolve
    # this once up-front so the per-mid loop can index into it cheaply.
    # Triggers for two cases:
    #   1. Top-level standalone ``plAbdy00.nj`` (etc.) — no BML host.
    #   2. BML-inner versions packed in ``pl[A-Z]nj.bml`` — the BML is
    #      a thin wrapper, the underlying model still binds against
    #      ``pl[A-Z]tex.afs``. Without this branch every player-class
    #      BML inner renders untextured (~140 inner models across
    #      plAnj.bml..plYnj.bml).
    pl_class_locs: list = []
    pl_class_lookup_name: Optional[str] = None
    if LIVE_DATA_DIR.exists():
        if outer_ext != ".bml":
            pl_class_lookup_name = bml_path.name
        elif effective_inner is not None:
            # Only attempt the player-class lookup when this BML inner's
            # filename ITSELF matches the pl[A-Z]<part>NN convention. Other
            # inners packed in the same BML (e.g. accessory caps, hair
            # variants) are handled by the same-BML cross-inner fallback
            # below.
            stem = Path(effective_inner).stem
            if _texture_index.player_class_for(effective_inner):
                pl_class_lookup_name = effective_inner
        if pl_class_lookup_name is not None:
            try:
                pl_class_locs = _texture_index.lookup_player_class_textures(
                    LIVE_DATA_DIR, pl_class_lookup_name,
                )
            except Exception as e:  # pragma: no cover
                log.warning(
                    "lookup_player_class_textures failed for %s: %s",
                    pl_class_lookup_name, e,
                )
                pl_class_locs = []

    # ItemModel positional fallback: ItemModel.afs#NNNN's K-th NJTL slot
    # binds to the K-th XVR record of ItemTexture.afs#NNNN (and the same
    # for ItemModelEp4 ↔ ItemTextureEp4). PSOBB items don't have a
    # paired-XVMH sibling inside the AFS — their textures live in a
    # SIBLING archive, indexed by (inner_index, xvr_index). Without this
    # branch every weapon / mag / unit / shield model in PSOBB.IO renders
    # untextured (354 ItemModel + ItemModelEp4 inners flagged
    # ok_no_textures by render_coverage_audit pre-fix).
    item_tex_locs: list = []
    if (
        outer_ext == ".afs"
        and LIVE_DATA_DIR.exists()
        and effective_inner is not None
    ):
        item_tex_archive = _texture_index.item_archive_for(bml_path.name)
        if item_tex_archive:
            inner_idx, _basename = _parse_afs_inner_name(effective_inner)
            if inner_idx is not None:
                try:
                    item_tex_locs = _texture_index.lookup_item_textures(
                        LIVE_DATA_DIR, bml_path.name, inner_idx,
                    )
                except Exception as e:  # pragma: no cover
                    log.warning(
                        "lookup_item_textures failed for %s#%d: %s",
                        bml_path.name, inner_idx, e,
                    )
                    item_tex_locs = []

    # Same-BML cross-inner positional fallback: many PSOBB BMLs pack
    # multiple inner models, of which only the "main" inner carries the
    # paired XVM. Accessory inners (Vol Opt monitor bezels, De Rol Le
    # helm/shell shards, item boxes, dragon "sd" subform, gryphon LODs)
    # have no inline texture but reference the same texture pool by
    # positional material_id. The runtime resolves these via shared NJTL
    # state in PSOBB's texture allocator; we mirror that here by picking
    # the BML's XVMH-richest sibling inner and pointing every otherwise-
    # unresolved material_id at it via the `cross_bml` source kind.
    #
    # Only run when the binding code has any chance of needing the
    # fallback (i.e. THIS inner's XVMH coverage is incomplete and we have
    # a BML host to inspect).
    # Per-inner texture-id shift table (Phantasmal-style). Used by the
    # sibling-XVMH fallback below to remap THIS inner's mid into the
    # shared sibling's tile-index space. ``_inner_shift_for_self`` is
    # the cumulative offset of this inner (0 for the first inner;
    # sum(tile_count[0..i-1]) for the i-th).
    _all_inner_offsets: list[dict] = []
    _inner_shift_for_self: int = 0
    if outer_ext == ".bml" and effective_inner:
        try:
            _all_inner_offsets = _compute_bml_inner_tex_offsets(bml_path)
        except Exception as e:  # pragma: no cover
            log.warning("inner offsets compute failed for %s: %s",
                        bml_path.name, e)
            _all_inner_offsets = []
        for entry in _all_inner_offsets:
            if entry.get("name") == effective_inner:
                _inner_shift_for_self = int(entry.get("cumulative_offset", 0))
                break

    sibling_xvmh: Optional[tuple] = None
    if outer_ext == ".bml" and LIVE_DATA_DIR.exists():
        # Skip if the host already covers every plausible mid via in_bml.
        max_mid_seen = max(seen_mids) if seen_mids else 0
        if n_xvmh <= max_mid_seen:
            try:
                sibling_xvmh = _texture_index.best_sibling_xvmh_for(
                    bml_path,
                    exclude_inner=effective_inner,
                    min_records=max_mid_seen + 1,
                )
            except Exception as e:  # pragma: no cover
                log.warning(
                    "best_sibling_xvmh_for failed for %s: %s", bml_path.name, e,
                )
                sibling_xvmh = None
            # Fall back to the largest sibling regardless of record count
            # if no sibling cleared the strict bound — even partial
            # coverage is better than every mid showing tile 0.
            if sibling_xvmh is None:
                try:
                    sibling_xvmh = _texture_index.best_sibling_xvmh_for(
                        bml_path,
                        exclude_inner=effective_inner,
                        min_records=1,
                    )
                except Exception:
                    sibling_xvmh = None

    # Cross-BML stem-family lookup (2026-04-26) — runs BEFORE the
    # same-BML positional fallback so that when our inner has 0 inline
    # tiles AND 0 NJTL names, we first try a sibling BML whose XVMH-
    # bearing inner shares a meaningful stem token (e.g. "gawa",
    # "monitor", "dragon"). Catches the warp-gate-frame bug where
    # ``bm_obj_warpboss_ancient.bml#fe_obj_df_warp_gawa.xj`` has 6 mids
    # and no NJTL — its textures live in
    # ``bm_o_warp_ancient.bml#fd_obj1_swarp_gawa.xj``, not in the
    # in-BML sibling ``de_obj_df_warp_sbeam.xj`` (which is a beam
    # effect, totally different texture pool).
    #
    # We only fire this for inners with no NJTL AND no inline XVMH —
    # the usual cross-archive name lookup at line ~9477 already covers
    # the NJTL-present case, and an inner WITH inline tiles already
    # has its own answer.
    cross_bml_stem: Optional[tuple] = None
    if (
        outer_ext == ".bml"
        and effective_inner is not None
        and n_njtl == 0
        and n_xvmh == 0
        and seen_mids
        and bml_path.parent.is_dir()
    ):
        try:
            cross_bml_stem = _texture_index.find_sibling_bml_by_inner_stem(
                bml_path,
                effective_inner,
                min_xvr_count=max(seen_mids) + 1,
            )
        except Exception as e:  # pragma: no cover - defensive
            log.warning(
                "find_sibling_bml_by_inner_stem failed for %s#%s: %s",
                bml_path.name, effective_inner, e,
            )
            cross_bml_stem = None
        # Relax to "any tile count" if the strict-coverage probe missed.
        if cross_bml_stem is None:
            try:
                cross_bml_stem = _texture_index.find_sibling_bml_by_inner_stem(
                    bml_path,
                    effective_inner,
                    min_xvr_count=1,
                )
            except Exception:
                cross_bml_stem = None

    for mid in sorted_mids:
        if 0 <= mid < n_xvmh:
            tile_idx = mid
            missing = False
            source = "in_bml"
            cross_bml = None
            cross_afs = None
        else:
            tile_idx = 0  # frontend fallback (in-BML default)
            missing = True
            source = "missing"
            cross_bml = None
            cross_afs = None
            # Try cross-BML lookup if we know the texture name. PSOBB.IO
            # uses ~60 NJTL refs (Vol Opt monitor parts, jungle props,
            # decoration containers) whose texture lives in a sibling
            # BML. Without this fallback those submeshes render with
            # tile 0 instead of the right tile.
            if 0 <= mid < n_njtl:
                tex_name = njtl_entries[mid].name or ""
                if tex_name and LIVE_DATA_DIR.exists():
                    try:
                        locs = _texture_index.lookup(LIVE_DATA_DIR, tex_name)
                    except Exception as e:  # pragma: no cover
                        log.warning(
                            "texture_index lookup failed for %s: %s", tex_name, e,
                        )
                        locs = []
                    # Skip locations that point at THIS bml+inner — they
                    # should have resolved in_bml above. Keep the rest.
                    locs = [
                        loc for loc in locs
                        if not (
                            host_basename is not None
                            and loc.bml_name == host_basename
                            and loc.inner_name == effective_inner
                        )
                    ]
                    # Partition: BML hits beat AFS hits because the
                    # frontend has cheaper machinery for BML tiles. AFS
                    # rows are still surfaced as `cross_afs` for the
                    # fallback path.
                    bml_hits = [loc for loc in locs if loc.kind == "bml"]
                    afs_hits = [loc for loc in locs if loc.kind == "afs"]
                    if bml_hits:
                        first = bml_hits[0]
                        source = "cross_bml"
                        missing = False
                        cross_bml = {
                            "bml": first.bml_name,
                            "inner": first.inner_name,
                            "xvr_index": int(first.xvr_index),
                            "candidates": len(bml_hits),
                        }
                        cross_bml_uses[tex_name] = [
                            {
                                "bml": loc.bml_name,
                                "inner": loc.inner_name,
                                "xvr_index": int(loc.xvr_index),
                            }
                            for loc in bml_hits
                        ]
                    elif afs_hits:
                        first = afs_hits[0]
                        source = "cross_afs"
                        missing = False
                        cross_afs = {
                            "archive": first.archive,
                            "inner_index": int(first.inner_index),
                            "xvr_index": int(first.xvr_index),
                            "candidates": len(afs_hits),
                        }
                        cross_afs_uses[tex_name] = [
                            {
                                "archive": loc.archive,
                                "inner_index": int(loc.inner_index),
                                "xvr_index": int(loc.xvr_index),
                            }
                            for loc in afs_hits
                        ]
            # Player-class positional fallback: when the model is a
            # top-level player NJ with no NJTL coverage for this slot,
            # resolve `material_id N` -> `pl<class>tex.afs` blob N via
            # the player-class table. plAbdy00 + plAhai00 + every other
            # ``pl[A-Z]<part>NN.nj`` lands here.
            if missing and pl_class_locs and 0 <= mid < len(pl_class_locs):
                loc = pl_class_locs[mid]
                source = "cross_afs"
                missing = False
                cross_afs = {
                    "archive": loc.archive,
                    "inner_index": int(loc.inner_index),
                    "xvr_index": int(loc.xvr_index),
                    "candidates": 1,
                }
                # Synthesise a stable diagnostic name so the binding
                # row carries something visible for player NJ models.
                synth_name = f"{Path(loc.archive).stem}_{loc.inner_index:04d}"
                cross_afs_uses[synth_name] = [
                    {
                        "archive": loc.archive,
                        "inner_index": int(loc.inner_index),
                        "xvr_index": int(loc.xvr_index),
                    }
                ]
            # ItemModel positional fallback: ItemModel.afs inners declare
            # an NJTL chunk but ship no inline XVMH — their textures live
            # in ItemTexture.afs (Ep1-3) or ItemTextureEp4.afs (Ep4) at
            # the SAME inner_index and the K-th NJTL slot resolves to
            # the K-th XVR record there. Resolves all 354 weapon / mag /
            # unit / shield models flagged ok_no_textures by the audit.
            if missing and item_tex_locs and 0 <= mid < len(item_tex_locs):
                loc = item_tex_locs[mid]
                source = "cross_afs"
                missing = False
                cross_afs = {
                    "archive": loc.archive,
                    "inner_index": int(loc.inner_index),
                    "xvr_index": int(loc.xvr_index),
                    "candidates": 1,
                }
                # Use the real NJTL slot name when present so the binding
                # diagnostic row carries the model's declared texture
                # name (e.g. "wxmS02e_z_w_huda"); fall back to a
                # synth otherwise.
                if 0 <= mid < n_njtl and njtl_entries[mid].name:
                    diag_name = njtl_entries[mid].name
                else:
                    diag_name = (
                        f"{Path(loc.archive).stem}_"
                        f"{loc.inner_index:04d}_{loc.xvr_index:04d}"
                    )
                cross_afs_uses[diag_name] = [
                    {
                        "archive": loc.archive,
                        "inner_index": int(loc.inner_index),
                        "xvr_index": int(loc.xvr_index),
                    }
                ]
            # Cross-BML stem-family fallback (2026-04-26) — runs BEFORE
            # the same-BML positional pick so an inner with no NJTL +
            # no inline XVMH (warp-gate frame, etc.) can find its
            # textures in a sibling BML whose inner shares a meaningful
            # stem token. Without this, gawa.xj (gate-frame) wrongly
            # cross-bound to its same-BML neighbour sbeam.xj (a beam
            # effect), painting the gate with sbeam's pink-stripe atlas.
            if missing and cross_bml_stem is not None:
                sib_bml_name, sib_inner_name, sib_xvr_count = cross_bml_stem
                if 0 <= mid < sib_xvr_count:
                    source = "cross_bml"
                    missing = False
                    cross_bml = {
                        "bml": sib_bml_name,
                        "inner": sib_inner_name,
                        "xvr_index": int(mid),
                        "candidates": 1,
                        "via": "stem_family",
                    }
                    synth_name = (
                        f"{Path(sib_inner_name).stem}_xvr{mid:02d}"
                    )
                    cross_bml_uses.setdefault(synth_name, []).append({
                        "bml": sib_bml_name,
                        "inner": sib_inner_name,
                        "xvr_index": int(mid),
                    })
            # Same-BML cross-inner positional fallback: the BML packs a
            # sibling inner whose XVMH is large enough to host this mid.
            # Triggers for accessory inners (Vol Opt monitor bezels, De Rol
            # Le helm/shell shards, item boxes, dragon "sd" subform,
            # gryphon LODs) that share a texture pool with the BML's
            # main inner. We surface this as a `cross_bml` row pointing
            # at THIS BML + the sibling inner — the existing frontend
            # cross_bml fetcher handles the URL synthesis.
            #
            # Suppressed when n_njtl == 0 — those inners (warp gate frame,
            # ancient boss subparts) are most likely shadow / stencil
            # geometry that the runtime never textures, and the same-
            # BML positional pick is almost certainly wrong (it would
            # paint a gate-frame with beam-effect atlas tiles). Inners
            # WITH NJTL still flow through the same-BML path because
            # their NJTL slot count gives the binding meaningful order.
            #
            # Suppression covers BOTH cases:
            #   - stem-family sibling-BML found a hit (those mids beyond
            #     the sibling's tile pool stay missing, which is honest
            #     than painting them with the wrong texture);
            #   - no stem-family sibling found at all (no plausible
            #     cross-BML answer; same-BML positional is a guess).
            _suppress_same_bml = (
                n_njtl == 0 and effective_inner is not None
            )
            if missing and sibling_xvmh is not None and _suppress_same_bml:
                log.warning(
                    "cross_bml: suppressing same-BML positional fallback for "
                    "%s#%s mid=%d (no NJTL + no stem-family sibling found)",
                    bml_path.name, effective_inner, mid,
                )
            if missing and sibling_xvmh is not None and not _suppress_same_bml:
                sib_inner_name, sib_count = sibling_xvmh
                # Phantasmal shift: when this inner is not the first one
                # in the BML and the sibling has enough tiles, our
                # mid-th texture lives at sibling[mid + self_offset].
                # Falls back to the unshifted index if the shifted lookup
                # would overflow the sibling's tile count.
                shifted_idx = mid + _inner_shift_for_self
                if 0 <= shifted_idx < sib_count:
                    chosen_idx = shifted_idx
                elif 0 <= mid < sib_count:
                    chosen_idx = mid
                else:
                    chosen_idx = -1
                if chosen_idx >= 0:
                    source = "cross_bml"
                    missing = False
                    cross_bml = {
                        "bml": host_basename or bml_path.name,
                        "inner": sib_inner_name,
                        "xvr_index": int(chosen_idx),
                        "candidates": 1,
                    }
                    # Synthesise a name rooted in the sibling inner so the
                    # diagnostic row is informative.
                    synth_name = (
                        f"{Path(sib_inner_name).stem}_xvr{chosen_idx:02d}"
                    )
                    cross_bml_uses.setdefault(synth_name, []).append({
                        "bml": host_basename or bml_path.name,
                        "inner": sib_inner_name,
                        "xvr_index": int(chosen_idx),
                    })
        if 0 <= mid < n_njtl:
            name = njtl_entries[mid].name
        elif source == "cross_afs" and cross_afs is not None:
            # Player-class fallback: surface a synthetic name so the UI
            # has something to show ('plAtex_0000') in place of the
            # real NJTL slot string.
            name = f"{Path(cross_afs['archive']).stem}_{cross_afs['inner_index']:04d}"
        elif source == "cross_bml" and cross_bml is not None:
            # Same-BML cross-inner fallback: surface a synthetic name
            # rooted in the sibling inner. The `xvr_index` is the
            # positional tile selected.
            name = (
                f"{Path(cross_bml['inner']).stem}"
                f"_xvr{int(cross_bml.get('xvr_index', 0)):02d}"
            )
        else:
            name = ""
        row = {
            "material_id": mid,
            "tile_index": int(tile_idx),
            "missing": bool(missing),
            "name": name,
            "source": source,
        }
        if cross_bml is not None:
            row["cross_bml"] = cross_bml
        if cross_afs is not None:
            row["cross_afs"] = cross_afs
        binding.append(row)

    name_match = (n_njtl > 0 and n_njtl == n_xvmh)
    out: dict = {
        "njtl": [{"slot": e.slot, "name": e.name} for e in njtl_entries],
        "xvmh": xvm_records,
        "binding": binding,
        "name_match": name_match,
    }
    if cross_bml_uses:
        out["cross_bml"] = cross_bml_uses
    if cross_afs_uses:
        out["cross_afs"] = cross_afs_uses
    if xvm_error:
        out["xvm_error"] = xvm_error
    if not njtl_entries:
        out["njtl_missing"] = True

    # Phantasmal-style per-inner texture-id shift table (2026-04-26).
    # Only emit for BML inners — top-level .nj / .afs models don't have
    # the multi-inner sibling concept. The frontend uses this to keep
    # inner-N's texture-id range disjoint from inner-0..N-1 when
    # compositing them into one scene graph.
    if outer_ext == ".bml":
        offsets = _all_inner_offsets
        if not offsets:
            try:
                offsets = _compute_bml_inner_tex_offsets(bml_path)
            except Exception as e:  # pragma: no cover — defensive
                log.warning("inner_tex_offsets compute failed for %s: %s",
                            bml_path.name, e)
                offsets = []
        if offsets:
            out["inner_tex_offsets"] = offsets

    return out


@app.get("/api/model_textures/{path:path}")
def api_model_textures(path: str, inner: Optional[str] = None):
    """Return the NJTL / XVMH / per-material binding for a model.

    Response:
        {
            "njtl":  [{"slot": int, "name": str}, ...],
            "xvmh":  [{"tile_index": int, "id": int, "width": int,
                        "height": int, "fmt": int, "name": str}, ...],
            "binding": [{"material_id": int, "tile_index": int,
                          "missing": bool, "name": str}, ...],
            "name_match": bool,
            "filename": "<input>",
            "inner": "<inner-name or null>"
        }

    Path forms (same as /api/model_mesh):
      <file>.nj                        - direct chunk-Ninja file
      <file>.xj                        - direct descriptor-Xj file
      <file>.bml + ?inner=<name>       - BML inner via query
      <bml>#<inner>.{nj,xj}            - BML inner via path fragment

    Errors mirror /api/model_mesh: 400 / 404 / 413 / 502.
    """
    base, effective_inner = _split_inner_with_query(path, inner)

    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()
    inner_ext = ""

    if ext == ".bml":
        if not effective_inner:
            raise HTTPException(
                400,
                "BML model requires `?inner=<entry-name>.{nj,xj}` query parameter or '#<inner>' suffix",
            )
        _validate_inner_name(effective_inner, msg="invalid inner entry name")
        # Defensive strip of trailing .xvm — the asset router can hand
        # us an entry name that ends in .nj.xvm if the user clicked on
        # the texture entry inside the BML; the binding is computed
        # from the MESH side, so use the matching .nj/.xj.
        if effective_inner.lower().endswith(".xvm"):
            effective_inner = effective_inner[: -len(".xvm")]
        inner_ext = Path(effective_inner).suffix.lower()
        if inner_ext not in IFF_EXTENSIONS:
            raise HTTPException(
                400,
                f"inner entry must be {IFF_EXTENSIONS!r}, got {inner_ext!r}",
            )
        nj_bytes = _read_inner_nj_from_bml(p, effective_inner)
    elif ext == ".afs":
        if not effective_inner:
            raise HTTPException(
                400,
                "AFS model requires `?inner=NNNN_<basename>` query parameter or '#<inner>' suffix",
            )
        nj_bytes, logical_inner = _read_afs_inner_nj(p, effective_inner)
        inner_ext = Path(logical_inner).suffix.lower() or ".nj"
    elif ext in IFF_EXTENSIONS:
        if effective_inner:
            raise HTTPException(400, f"`inner` not allowed for {ext} files")
        sz = p.stat().st_size
        if sz > ASSET_PARSE_MAX_BYTES:
            raise HTTPException(
                413,
                f"model too large: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
            )
        nj_bytes = p.read_bytes()
    else:
        raise HTTPException(
            400,
            f"unsupported model extension {ext!r} (expected .nj, .xj, .bml, or .afs)",
        )

    # Parse the mesh side once so we know which material_ids the model
    # actually references. (We could skip this and just emit one binding
    # row per NJTL slot — but only enumerating material_ids that show
    # up on real submeshes makes the binding compact and prevents the
    # frontend from issuing a tile_png fetch for textures the model
    # never uses.) Goes through the parse-cache LRU so repeat opens are
    # served in <5 ms.
    try:
        meshes = _cached_model_parse(
            nj_bytes, p, ext, inner_ext, effective_inner,
        )
    except ValueError as e:
        raise HTTPException(400, f"model parse failed: {e}")
    except Exception as e:  # pragma: no cover - defensive net
        log.exception("model parse internal error")
        raise HTTPException(500, f"model parse internal error: {e}")

    binding_data = _build_model_texture_binding_cached(p, ext, effective_inner, nj_bytes, meshes)
    return {
        "filename": path,
        "inner": effective_inner,
        **binding_data,
    }


# ---------------------------------------------------------------------------- material inspector
#
# /api/material/<path>?inner=<inner>           -> per-submesh material breakdown
# POST /api/material/<path>                    -> save edits, stage BML for repack
# GET /api/material_presets                    -> shipping value catalog (Task 4)
#
# These endpoints power the Material Inspector tab. Decoded chunk
# semantics live in ``formats/material.py`` — this layer just wires
# the model-path resolver, BML extract / repack, and the preset list.


@app.get("/api/material/{path:path}")
def api_material_get(path: str, inner: Optional[str] = None):
    """Return per-submesh material breakdown for a model.

    Response shape:
        {
            "filename": "<input>",
            "inner": "<inner name or null>",
            "submesh_count": int,
            "submeshes": [
                {
                    "idx": int,
                    "material_id": int,
                    "diffuse_rgba": [r, g, b, a],
                    "ambient_rgba": [r, g, b, a],
                    "specular_rgb": [r, g, b],
                    "specular_exponent": int,
                    "alpha_test": null | {"enabled": bool, "threshold": int},
                    "alpha_blend": null | {"src": str, "dst": str},
                    "blend_mode": str,
                    "two_sided": bool,
                    "depth_test": bool,
                    "depth_write": bool,
                    "flat_shaded": bool,
                    "env_mapped": bool
                }, ...
            ]
        }

    Path forms (same as /api/model_textures):
      <file>.nj                        - direct chunk-Ninja file
      <file>.bml + ?inner=<name>       - BML inner via query
      <bml>#<inner>.nj                 - BML inner via path fragment
    """
    base, effective_inner = _split_inner_with_query(path, inner)

    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()

    if ext == ".bml":
        if not effective_inner:
            raise HTTPException(400, "BML model requires `?inner=<entry-name>.nj`")
        if not effective_inner.lower().endswith(".nj"):
            raise HTTPException(400, "Material Inspector currently supports `.nj` inners only")
        _validate_inner_name(effective_inner, msg="invalid inner entry name")
        nj_bytes = _read_inner_nj_from_bml(p, effective_inner)
    elif ext == ".nj":
        if effective_inner:
            raise HTTPException(400, "`inner` not allowed for top-level .nj files")
        sz = p.stat().st_size
        if sz > ASSET_PARSE_MAX_BYTES:
            raise HTTPException(413, f"file too large: {sz}")
        nj_bytes = p.read_bytes()
    else:
        raise HTTPException(
            400,
            f"unsupported model extension {ext!r} for materials (expected .nj or .bml)",
        )

    # Parse the round-trip-preserving NJ model so we have NjChunk objects
    # to walk per submesh. (The xj.py parser bakes vertices into world
    # space and drops chunk metadata — wrong for inspecting materials.)
    from formats.nj_writer import parse_nj_for_writer
    from formats.material import aggregate_submesh_state
    try:
        model = parse_nj_for_writer(nj_bytes)
    except Exception as e:  # pragma: no cover — defensive
        raise HTTPException(400, f"NJ parse failed: {e}")

    submeshes_out: list[dict] = []
    global_idx = 0
    for mesh_idx, mesh in enumerate(model.meshes):
        rows = aggregate_submesh_state(mesh.plist)
        for r in rows:
            submeshes_out.append({
                "idx": global_idx,
                "mesh_idx": mesh_idx,
                "submesh_idx_in_mesh": r.submesh_idx,
                "material_id": r.material_id,
                "diffuse_rgba": list(r.diffuse_rgba),
                "ambient_rgba": list(r.ambient_rgba),
                "specular_rgb": list(r.specular_rgb),
                "specular_exponent": r.specular_exponent,
                "alpha_test": r.alpha_test,
                "alpha_blend": r.alpha_blend,
                "blend_mode": r.blend_mode,
                "two_sided": r.two_sided,
                "depth_test": r.depth_test,
                "depth_write": r.depth_write,
                "flat_shaded": r.flat_shaded,
                "env_mapped": r.env_mapped,
            })
            global_idx += 1

    return {
        "filename": path,
        "inner": effective_inner,
        "submesh_count": len(submeshes_out),
        "submeshes": submeshes_out,
    }


class MaterialEditReq(BaseModel):
    """One submesh's worth of material edits.

    Every field is optional — fields that are ``None`` (or absent) are
    NOT changed. ``submesh_idx`` is the global index across all meshes
    (matches the ``idx`` field in the GET response). ``mesh_idx`` and
    ``submesh_idx_in_mesh`` resolve any ambiguity for models with many
    meshes.
    """
    submesh_idx: int = 0
    mesh_idx: Optional[int] = None
    submesh_idx_in_mesh: Optional[int] = None
    diffuse_rgba: Optional[list] = None
    alpha_test: Optional[dict] = None
    alpha_blend: Optional[dict] = None
    two_sided: Optional[bool] = None
    depth_test: Optional[bool] = None
    depth_write: Optional[bool] = None


class MaterialSaveReq(BaseModel):
    inner: Optional[str] = None
    submeshes: list = []


@app.post("/api/material/{path:path}")
def api_material_save(path: str, req: MaterialSaveReq, request: Request):
    """Apply per-submesh material edits and stage a rebuilt BML.

    Body shape:
        {
          "inner": "<inner.nj>",
          "submeshes": [
            {"submesh_idx": int, "diffuse_rgba": [r,g,b,a], ...},
            ...
          ]
        }

    For BML hosts, the rebuilt BML is staged at
    ``cache/bml_export/<safe>.bml``. For top-level `.nj` files we
    stage at ``cache/nj_export/<name>.nj`` so the same deploy path
    works. Returns ``{ok, archive_path, size, md5, edits_applied}``.
    """
    _enforce_body_size(request, MAX_BUILD_NJ_BODY)
    base, hash_inner = _split_inner_path(path)
    effective_inner = hash_inner or req.inner

    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()

    if ext not in (".bml", ".nj"):
        raise HTTPException(
            400,
            f"unsupported archive type for material save: {ext!r} (.bml or .nj)",
        )

    # Coerce + validate the edits list — the wire format is just a list
    # of plain dicts so the apply_submesh_edits helper accepts them as-is.
    if not isinstance(req.submeshes, list):
        raise HTTPException(400, "`submeshes` must be a list")
    edits = []
    for e in req.submeshes:
        if isinstance(e, dict):
            edits.append(e)
        elif hasattr(e, "model_dump"):
            edits.append(e.model_dump())
        elif hasattr(e, "dict"):
            edits.append(e.dict())
        else:
            raise HTTPException(400, "submeshes entries must be dicts")
    if not edits:
        raise HTTPException(400, "no edits supplied")

    # Resolve source NJ bytes.
    from formats.nj_writer import parse_nj_for_writer, encode_nj_model as _encode_nj_model
    from formats.material import apply_submesh_edits

    if ext == ".bml":
        if not effective_inner:
            raise HTTPException(400, "BML save requires `inner` in body or '#' suffix")
        if not effective_inner.lower().endswith(".nj"):
            raise HTTPException(400, "Material save supports `.nj` inners only")
        try:
            nj_bytes = _read_inner_nj_from_bml(p, effective_inner)
        except HTTPException:
            raise
    else:
        nj_bytes = p.read_bytes()

    try:
        model = parse_nj_for_writer(nj_bytes)
    except Exception as e:
        raise HTTPException(400, f"NJ parse failed: {e}")

    # Apply edits across all meshes — submesh indexing is global so we
    # need to translate (global_idx) -> (mesh_idx, local_idx_in_mesh).
    # The simplest way to get this right is to walk the meshes the same
    # way aggregate_submesh_state does and remap each edit to the mesh
    # whose strip contributes the global_idx-th submesh.
    edits_by_global: dict[int, dict] = {}
    for e in edits:
        gidx = int(e.get("submesh_idx", -1))
        if gidx < 0:
            continue
        edits_by_global[gidx] = e

    edits_applied = 0
    global_idx = 0
    for mesh in model.meshes:
        # Count how many strip chunks this mesh has.
        strip_count_this_mesh = sum(
            1 for c in mesh.plist if 64 <= c.type_id <= 75
        )
        if strip_count_this_mesh == 0:
            continue
        # Build a mesh-local edits list for chunks that fall in our range.
        local_edits = []
        for local_i in range(strip_count_this_mesh):
            ge = edits_by_global.get(global_idx + local_i)
            if ge is not None:
                # Translate global submesh_idx to mesh-local for the helper.
                me = dict(ge)
                me["submesh_idx"] = local_i
                local_edits.append(me)
        if local_edits:
            mesh.plist = apply_submesh_edits(list(mesh.plist), local_edits)
            edits_applied += len(local_edits)
        global_idx += strip_count_this_mesh

    if edits_applied == 0:
        raise HTTPException(
            400,
            f"no edits matched any submesh — "
            f"requested indices {sorted(edits_by_global.keys())[:8]} "
            f"but model has {global_idx} submeshes",
        )

    # Re-encode NJ.
    try:
        new_nj = _encode_nj_model(model)
    except (ValueError, struct.error) as e:
        raise HTTPException(400, f"encode_nj_model failed: {e}")

    if ext == ".bml":
        # Re-pack the BML with the edited inner.
        from formats.bml import parse_bml_for_pack, parse_bml_pack_meta
        try:
            blob = p.read_bytes()
            pack_entries = parse_bml_for_pack(blob)
            meta = parse_bml_pack_meta(blob)
        except (OSError, ValueError) as e:
            raise HTTPException(500, f"BML parse failed: {e}")

        replaced = False
        for pe in pack_entries:
            if pe.name == effective_inner:
                # Store as raw + tell packer to PRS-compress on emit.
                pe.data = new_nj
                pe.is_compressed = False
                pe.decompressed_size = len(new_nj)
                replaced = True
                break
        if not replaced:
            raise HTTPException(404, f"BML {p.name} has no inner {effective_inner!r}")

        try:
            out = pack_bml(
                pack_entries,
                compression=meta["compression"],
                file_alignment=meta["file_alignment"],
                has_textures_override=meta["has_textures"],
            )
        except ValueError as e:
            raise HTTPException(500, f"pack_bml failed: {e}")

        BML_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_name = _safe_archive_name(p.name)
        out_path = BML_EXPORT_DIR / out_name
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_bytes(out)
        os.replace(tmp_path, out_path)
        md5 = _md5_bytes(out)
        log.info(
            "material_save BML %s#%s -> %s (%d bytes, %d edits, md5=%s)",
            p.name, effective_inner, out_path, len(out), edits_applied, md5,
        )
        return {
            "ok": True,
            "archive_path": str(out_path),
            "archive_name": out_name,
            "size": len(out),
            "md5": md5,
            "edits_applied": edits_applied,
            "kind": "bml",
        }

    # Top-level .nj.
    NJ_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = _safe_archive_name(p.name)
    out_path = NJ_EXPORT_DIR / out_name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(new_nj)
    os.replace(tmp_path, out_path)
    md5 = _md5_bytes(new_nj)
    log.info(
        "material_save NJ %s -> %s (%d bytes, %d edits, md5=%s)",
        p.name, out_path, len(new_nj), edits_applied, md5,
    )
    return {
        "ok": True,
        "archive_path": str(out_path),
        "archive_name": out_name,
        "size": len(new_nj),
        "md5": md5,
        "edits_applied": edits_applied,
        "kind": "nj",
    }


@app.get("/api/material_presets")
def api_material_presets():
    """Return the PSOBB shipping-value catalog (Task 4).

    Used by the Material Inspector to populate "preset" buttons.
    Mapped from ``formats.material.PSOBB_MATERIAL_PRESETS``.
    """
    from formats.material import list_presets
    return {"ok": True, "presets": list_presets()}


# ---------------------------------------------------------------------------- skeletal animation
#
# `/api/animations/{path}` lists NJM motions associated with a model.
# `/api/animation_data/{path}?motion=<name|index>` returns the parsed
# keyframes (per-bone tracks) for one motion, ready for the frontend to
# interpolate at render time.
#
# Motion source resolution rules (delegated to ``formats.motion_pairing.
# resolve_motions_for_model`` — see that module's docstring for the
# canonical four-tier taxonomy):
#   * Tier 1 — engine-table override (reserved; no rows declare a hint
#     today, but the resolver leaves room for cross-archive hints when
#     modded data needs them).
#   * Tier 2 — same-BML siblings whose post-verb stem matches the loaded
#     inner-model stem. ``walk_boss1_s_nb_dragon.njm`` pairs with
#     ``boss1_s_nb_dragon.nj`` because the post-verb tail matches.
#   * Tier 3 — same-BML siblings with any other stem. Multi-form BMLs
#     (Pan Arms, De Rol Le, Vol Opt, Pouilly Slime) still expose every
#     motion, but at lower priority so the auto-play picker doesn't
#     land on a track for the wrong inner-rig.
#   * Tier 4 — ``NpcApcMot.bml`` fallback for ``pl*`` / ``bm_n_*`` /
#     ``bm_npc_*`` host BMLs that ship without inline motions. Only
#     fires when no Tier-2 hit is found.
#
# Within each tier, candidates are sorted by action priority
# (walk > idle > attack > die > hit > spawn > fly > despawn > unknown)
# so ``default_index`` falls out as 0 — the existing auto-play in
# ``static/model_viewer.js`` doesn't need to know about tiers.
#
# Empirical inventory backing the tier weights lives in
# ``_reports/motion_inventory.md``.
#
# Wire shape (animations endpoint):
#     {
#       "filename": "<input>",
#       "inner":    "<inner-name or null>",
#       "motions":  [
#         {
#           "index": <int>,
#           "name": "<entry-name>",
#           "frame_count": <int>,
#           "fps": <float>,
#           "bone_count": <int>,
#           "type_flags": <int>,
#           "interpolation": <int>,
#           "source_path": "<bml#inner or top-level path>"
#         },
#         ...
#       ],
#       "default_index": <int|null>   // motion the frontend should
#                                      //   auto-play; null when none
#       "skeleton_bone_count": <int>   // bones in the source model;
#                                      //   helps the frontend clamp
#                                      //   if a motion has more
#     }
#
# Wire shape (animation_data endpoint):
#     {
#       "filename": "<input>",
#       "motion": "<motion-name>",
#       "motion_index": <int>,
#       "frame_count": <int>,
#       "fps": <float>,
#       "bone_count": <int>,
#       "type_flags": <int>,
#       "interpolation": <int>,
#       "bones": [
#         {
#           "idx": <int>,
#           "kf": [
#             {
#               "t": <int>,
#               "tx": <float>, "ty": <float>, "tz": <float>,
#               "rx": <int>,   // BAMS — multiply by 2π/65536 for radians
#               "ry": <int>,
#               "rz": <int>,
#               "sx": <float>, "sy": <float>, "sz": <float>
#               // optional: "qw","qx","qy","qz" when type bit 13 set
#             }, ...
#           ]
#         },
#         ...
#       ]
#     }


def _resolve_njm_inner_bytes(p: Path, inner_name: str) -> bytes:
    """Read & PRS-decompress one named NJM entry from a BML container.

    Mirrors `_read_inner_nj_from_bml` but explicit about the inner
    naming. Returns the raw NMDM-IFF bytes ready for `parse_njm`.

    Cached in-memory by ``(bml_path, mtime, inner_name)`` so repeated
    listing calls on the same BML don't re-spawn PuyoToolsCli for the
    same entry. This is critical for the NpcApcMot fallback path:
    NpcApcMot.bml has 120 inner motions, each requiring a ~100 ms
    subprocess call; without the cache, every player-class model open
    takes 12+ seconds to populate the animation panel.

    Raises HTTPException on missing entry / decompress failure.
    """
    cache_key = (str(p), p.stat().st_mtime_ns, inner_name)
    with _NJM_DECOMPRESS_CACHE_LOCK:
        cached = _NJM_DECOMPRESS_CACHE.get(cache_key)
        if cached is not None:
            # LRU bookkeeping: move to end so we don't evict it next.
            _NJM_DECOMPRESS_CACHE.move_to_end(cache_key)
            return cached

    sz = p.stat().st_size
    if sz > ASSET_PARSE_MAX_BYTES:
        raise HTTPException(
            413,
            f"BML too large to parse in-memory: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
        )
    blob = p.read_bytes()
    try:
        entries = parse_bml(blob)
    except ValueError as e:
        raise HTTPException(400, f"BML parse failed: {e}")
    target = next((ent for ent in entries if ent.name == inner_name), None)
    if target is None:
        raise HTTPException(404, f"no entry named {inner_name!r} in {p.name}")
    try:
        from formats.bml import decompress_prs_cached
        slice_start = target.offset
        slice_end = slice_start + target.size_compressed
        # Reuse the shared LRU — that cache already serves the manifest
        # walker and the model viewer, so the same inner-blob can hit on
        # either path. The NjmDecompressCache is now a redundant L2,
        # but keeping it ensures repeated /api/animations calls don't
        # hit even the LRU lookup. Both layers key on (path, mtime,
        # name) so they invalidate together.
        out = decompress_prs_cached(
            p, p.stat().st_mtime_ns, inner_name,
            lambda: bytes(blob[slice_start:slice_end]),
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(502, f"BML PRS decompress failed: {e}")

    # L2 cache. Cap by total bytes — once the cache exceeds NJM_CACHE_MAX
    # bytes, evict LRU entries until we fit. Keeps memory usage bounded
    # while still giving a 100% hit rate for the NpcApcMot fallback
    # (NpcApcMot's 120 entries total ~700 KB decompressed — well below
    # the cap).
    #
    # Perf 2026-04-30: track running byte total in a module global so the
    # eviction loop is O(evictions) instead of O(N) per iteration. The
    # previous ``sum(len(v) for v in _NJM_DECOMPRESS_CACHE.values())``
    # inside the while-condition was O(N²) when the cache was full and
    # under repeated insertion — re-summed every loop iteration.
    global _NJM_DECOMPRESS_CACHE_BYTES
    with _NJM_DECOMPRESS_CACHE_LOCK:
        _NJM_DECOMPRESS_CACHE[cache_key] = out
        _NJM_DECOMPRESS_CACHE_BYTES += len(out)
        while _NJM_DECOMPRESS_CACHE_BYTES > NJM_CACHE_MAX_BYTES and len(_NJM_DECOMPRESS_CACHE) > 1:
            try:
                _evicted_key, _evicted_val = _NJM_DECOMPRESS_CACHE.popitem(last=False)
            except KeyError:
                break
            _NJM_DECOMPRESS_CACHE_BYTES -= len(_evicted_val)
    return out


# In-memory cache for decompressed NJM blobs. Keyed by (path, mtime_ns,
# inner_name) tuples so the cache invalidates cleanly when a BML's
# mtime changes (e.g. /api/repack writes a new copy of the file).
# Bounded by total cached bytes; LRU eviction.
_NJM_DECOMPRESS_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
# Live byte counter — refreshed on insert/evict so we don't sum the
# whole dict on every cap check. See the eviction loop above.
_NJM_DECOMPRESS_CACHE_BYTES = 0
# Audit C-1 (2026-05-01): touched by request handlers AND the tile-prewarm
# ThreadPoolExecutor. Mirror formats.bml._PRS_INNER_CACHE_LOCK pattern —
# single Lock guarding reads (.get + move_to_end), writes, and the
# eviction loop so the check-then-act sequence at insert can't interleave.
_NJM_DECOMPRESS_CACHE_LOCK = threading.Lock()
NJM_CACHE_MAX_BYTES = 32 * 1024 * 1024  # 32 MB — ~5x the largest realistic working set


def _resolve_motion_sources(
    base_path: Path,
    base_ext: str,
    inner_name: Optional[str],
) -> list[tuple[Path, str, str]]:
    """Find NJM motion source candidates for a model.

    Returns a list of ``(bml_path, inner_njm_name, source_label)``
    tuples ranked by ``formats.motion_pairing.resolve_motions_for_model``
    (see that module's header for the four-tier taxonomy).

    The tuple shape is preserved for backwards compatibility — older
    callers (``api_animation_data``, the motion-editor save endpoints)
    only care about the first two fields. The new ranker just changes
    the *order*, putting Tier-2 stem-matched motions first so the
    auto-play pick (index 0 in the wire response) lands on a track
    authored for the right inner-rig.
    """
    out: list[tuple[Path, str, str]] = []
    if base_ext == ".bml" or base_ext in IFF_EXTENSIONS:
        # Pass the new resolver the same context the old code used —
        # base path + inner-name. Resolver handles BML inner enumeration,
        # NpcApcMot fallback, and standalone-NJ sibling discovery
        # under one roof.
        refs = _resolve_motion_pairing(
            base_path,
            inner_name=inner_name,
            npc_motion_pack_search_roots=(DATA_DIR, LIVE_DATA_DIR),
        )
        for ref in refs:
            label = ref.source_label
            out.append((ref.archive, ref.inner_name, label))
    return out


def _read_njm_for_source(bml_path: Path, inner_name: str) -> bytes:
    """Read raw NJM bytes for a motion-source tuple.

    When `inner_name` is empty the source is a top-level .njm file
    (read directly); otherwise it's a BML inner (extracted via
    `_resolve_njm_inner_bytes`).
    """
    if inner_name:
        return _resolve_njm_inner_bytes(bml_path, inner_name)
    sz = bml_path.stat().st_size
    if sz > ASSET_PARSE_MAX_BYTES:
        raise HTTPException(
            413,
            f"NJM too large: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
        )
    return bml_path.read_bytes()


@app.get("/api/animations/{path:path}")
def api_animations(path: str, inner: Optional[str] = None):
    """List NJM motions associated with a model.

    Path forms (same as /api/model_mesh):
      ``<file>.nj``
      ``<file>.xj``
      ``<file>.bml`` + ``?inner=<name>``
      ``<bml>#<inner>.{nj,xj}``

    Per-motion metadata is surfaced WITHOUT parsing the keyframes (we
    do read each NJM's header to get frame_count + bone_count, which
    is cheap — a few u32 reads after PRS-decompress). The default
    motion is the first match in priority order: walk > run > move >
    swim > fly > idle > stand > whatever's first.

    Returns 200 with an empty `motions` array when no NJMs are found
    (rather than 404) so the frontend can degrade gracefully.
    """
    base, effective_inner = _split_inner_with_query(path, inner)

    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()

    sources = _resolve_motion_sources(p, ext, effective_inner)

    # Re-run the pairing resolver in parallel to surface tier / action
    # metadata in the wire payload (backward-compat: existing fields
    # are unchanged, just augmented). The resolver runs the same
    # discovery as ``_resolve_motion_sources`` and returns
    # ``MotionRef`` objects keyed by ``(archive, inner_name)`` — we use
    # those keys to attach tier/action to the per-motion dicts.
    pairing_refs = _resolve_motion_pairing(
        p,
        inner_name=effective_inner,
        npc_motion_pack_search_roots=(DATA_DIR, LIVE_DATA_DIR),
    ) if ext in (".bml",) or ext in IFF_EXTENSIONS else []
    pairing_index: dict[tuple[str, str], object] = {
        (str(r.archive.resolve()), r.inner_name): r
        for r in pairing_refs
    }

    motions_out: list[dict] = []
    for i, (bml_path, inner_name, label) in enumerate(sources):
        try:
            njm_bytes = _read_njm_for_source(bml_path, inner_name)
            # Listing endpoint: header-only parse is ~50× cheaper than
            # full keyframe decode (the latter touches every BAMS u16
            # in the file). Frontend fetches full data on demand via
            # /api/animation_data when the user picks a motion.
            header = _njm_parse_header(njm_bytes)
        except (HTTPException, ValueError, RuntimeError) as e:
            log.warning("animation list: skip %s: %s", label, e)
            continue
        if header is None:
            continue
        # Display name: strip the `.njm` suffix and any leading path.
        name_disp = inner_name if inner_name else bml_path.name
        if name_disp.lower().endswith(".njm"):
            name_disp = name_disp[:-4]

        # Attach tier / action / confidence from the pairing resolver
        # (additive — older clients ignore the new fields). Falls back
        # to verb-prefix heuristic when the source-list and pairing
        # disagree on resolved paths (rare; happens when a sub-call
        # resolves a symlink one way and the other doesn't).
        ref = pairing_index.get((str(bml_path.resolve()), inner_name))
        if ref is not None:
            action = ref.action          # type: ignore[attr-defined]
            tier = ref.tier              # type: ignore[attr-defined]
            confidence = ref.confidence  # type: ignore[attr-defined]
        else:
            action = _motion_action_hint(name_disp)
            tier = 3
            confidence = 0.5

        motions_out.append({
            "index": i,
            "name": name_disp,
            "frame_count": header.frame_count,
            "fps": _njm_guess_fps(name_disp),
            "bone_count": header.bone_count,
            "type_flags": header.type_flags,
            "interpolation": header.interpolation,
            "source_path": label,
            "action": action,
            "tier": tier,
            "confidence": confidence,
        })

    # Pick a default motion. The pairing resolver has already sorted
    # ``sources`` by ``(tier, action_priority)``, so index 0 is the
    # right default ~99 % of the time. We keep the legacy
    # ``pick_default_motion`` (verb-keyword tiered match) as a tie-
    # breaker for the rare case where index 0 is an "unknown" action
    # but a downstream entry has a real walk/idle keyword — matches
    # the pre-2026-04-26 contract for any test asserting walk-first.
    if motions_out:
        if motions_out[0].get("action", "unknown") == "unknown":
            motion_names = [m["name"] for m in motions_out]
            default_index = _njm_pick_default(motion_names)
        else:
            default_index = 0
    else:
        default_index = None

    # Skeleton bone count for the model (so the frontend can clamp NJM
    # tracks if the motion has more bones than the mesh tree).
    skel_bones = 0
    if effective_inner:
        try:
            if ext == ".bml":
                skel_inner_ext = Path(effective_inner).suffix.lower()
                if skel_inner_ext == ".nj":
                    nj_bytes = _read_inner_nj_from_bml(p, effective_inner)
                    skel_bones = len(_parse_cache.parse_skeleton_cached(
                        nj_bytes,
                        file_key=_build_model_file_key(p, ext, effective_inner),
                    ))
            elif ext == ".afs":
                nj_bytes, logical_inner = _read_afs_inner_nj(p, effective_inner)
                if logical_inner.lower().endswith(".nj"):
                    skel_bones = len(_parse_cache.parse_skeleton_cached(
                        nj_bytes,
                        file_key=_build_model_file_key(p, ext, effective_inner),
                    ))
        except HTTPException:
            pass
        except (ValueError, RuntimeError):
            pass
    elif ext == ".nj":
        try:
            nj_bytes = p.read_bytes()
            skel_bones = len(_parse_cache.parse_skeleton_cached(
                nj_bytes,
                file_key=_build_model_file_key(p, ext, None),
            ))
        except (OSError, ValueError, RuntimeError):
            pass

    return {
        "filename": path,
        "inner": effective_inner,
        "motion_count": len(motions_out),
        "motions": motions_out,
        "default_index": default_index,
        "skeleton_bone_count": skel_bones,
    }


@app.get("/api/animation_data/{path:path}")
def api_animation_data(path: str, motion: str = "", inner: Optional[str] = None):
    """Return parsed keyframes for one motion.

    Query parameters:
      - ``motion``: motion name (matches the entry name minus `.njm`,
        case-insensitive) OR an integer index into the listing returned
        by /api/animations. Required.

    Response shape: see the comment at the top of this section.

    Returns 400 on missing/invalid `motion`, 404 when the motion is not
    found, and propagates parse errors as 400. The keyframe payload
    is sent as plain JSON (no base64) — typical NJMs have a few
    thousand keyframes, well within JSON's overhead budget. (Switching
    to typed-array b64 would halve the wire size; we keep JSON for
    debuggability and rely on gzip for compression.)
    """
    base, effective_inner = _split_inner_with_query(path, inner)

    if not motion:
        raise HTTPException(400, "missing ?motion=<name|index>")

    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()
    sources = _resolve_motion_sources(p, ext, effective_inner)
    if not sources:
        raise HTTPException(404, f"no NJM motions found for {path}")

    # Resolve the motion identifier.
    try:
        motion_idx = int(motion)
        # Numeric index path.
        if motion_idx < 0 or motion_idx >= len(sources):
            raise HTTPException(404, f"motion index {motion_idx} out of range (0..{len(sources)})")
        chosen = sources[motion_idx]
        chosen_idx = motion_idx
    except ValueError:
        # Name-based lookup. We try in priority order:
        #   1. Exact match against the inner_name minus `.njm`
        #      (case-insensitive). The frontend's animation listing uses
        #      this form as its canonical motion key.
        #   2. Exact match against the inner_name with `.njm`.
        #   3. Substring match — first motion whose stem CONTAINS the
        #      query. This lets the auto-detect "walk" string find
        #      "walk_boss1_s_nb_dragon" without the frontend having to
        #      know the model-specific suffix. Substring match is
        #      ordered by source-list index, so the first matching
        #      motion in DFS order wins.
        m_lower = motion.lower()
        m_lower_njm = m_lower if m_lower.endswith(".njm") else f"{m_lower}.njm"
        m_stem = m_lower[:-4] if m_lower.endswith(".njm") else m_lower

        chosen = None
        chosen_idx = -1
        # Pass 1: exact match (with and without .njm suffix).
        for i, (bml_path, inner_name, label) in enumerate(sources):
            cand = inner_name.lower() if inner_name else bml_path.name.lower()
            cand_stem = cand[:-4] if cand.endswith(".njm") else cand
            if cand_stem == m_stem or cand == m_lower or cand == m_lower_njm:
                chosen = sources[i]
                chosen_idx = i
                break
        # Pass 2: substring match (only if pass 1 didn't hit).
        if chosen is None:
            for i, (bml_path, inner_name, label) in enumerate(sources):
                cand = inner_name.lower() if inner_name else bml_path.name.lower()
                cand_stem = cand[:-4] if cand.endswith(".njm") else cand
                if m_stem and m_stem in cand_stem:
                    chosen = sources[i]
                    chosen_idx = i
                    break
        if chosen is None:
            raise HTTPException(404, f"no motion named {motion!r}")

    bml_path, inner_name, label = chosen
    try:
        njm_bytes = _read_njm_for_source(bml_path, inner_name)
    except HTTPException:
        raise
    try:
        parsed = _njm_parse(njm_bytes)
    except ValueError as e:
        raise HTTPException(400, f"NJM parse failed: {e}")
    if not parsed:
        raise HTTPException(400, "NJM had no motion data")
    m = parsed[0]

    # Project to wire format. We keep BAMS rotations as raw integers —
    # the frontend converts to radians at apply time. Position/scale
    # are floats. Quaternion fields are emitted only when present.
    #
    # Per-bone ``present`` bitfield (from ``NjmMotion.bone_present_tracks``)
    # tells the frontend which TRS channels were ACTUALLY authored on
    # this bone. Bit 0 = POS, bit 1 = ANG, bit 2 = SCL, bit 13 = QUAT.
    # When a bit is unset, the consumer (see
    # ``static/model_viewer.js::_sampleBoneTrack``) MUST fall back to
    # the bone's bind-pose TRS for that channel — treating an unset bit
    # as "this bone never had translation keyframes; do not yank to 0".
    # Without this signal the parser's default (0,0,0)/(0,0,0)/(1,1,1)
    # values look identical to "intentional zero translation" and
    # collapse rotation-only bones to the world origin during playback.
    present_per_bone = m.bone_present_tracks or []
    bones_out: list[dict] = []
    for b_idx, track in enumerate(m.tracks):
        present = present_per_bone[b_idx] if b_idx < len(present_per_bone) else 0
        if not track:
            # Empty track — emit a placeholder so the index alignment
            # with the mesh-tree's bone DFS order is preserved.
            bones_out.append({"idx": b_idx, "kf": [], "present": present})
            continue
        kf_out: list[dict] = []
        for kf in track:
            entry: dict = {
                "t": kf.time,
                "tx": kf.tx, "ty": kf.ty, "tz": kf.tz,
                "rx": kf.rx_bams, "ry": kf.ry_bams, "rz": kf.rz_bams,
                "sx": kf.sx, "sy": kf.sy, "sz": kf.sz,
            }
            if kf.qw is not None:
                entry["qw"] = kf.qw
                entry["qx"] = kf.qx
                entry["qy"] = kf.qy
                entry["qz"] = kf.qz
            kf_out.append(entry)
        bones_out.append({"idx": b_idx, "kf": kf_out, "present": present})

    name_disp = inner_name if inner_name else bml_path.name
    if name_disp.lower().endswith(".njm"):
        name_disp = name_disp[:-4]

    return {
        "filename": path,
        "motion": name_disp,
        "motion_index": chosen_idx,
        "source_path": label,
        "frame_count": m.frame_count,
        "fps": _njm_guess_fps(name_disp),
        "bone_count": m.bone_count,
        "type_flags": m.type_flags,
        "interpolation": m.interpolation,
        "bones": bones_out,
    }


# ---------------------------------------------------------------------------- model bundle
#
# `/api/model_bundle/{path}` consolidates the 4-7 round-trips a cold model
# open requires today (model_skinned + binding_data + animations + optional
# animation_data) into a single JSON response. The frontend's
# asset_router falls back to the per-endpoint flow when this 404s, so older
# servers still work.
#
# Bundling cuts ~250 ms on a typical localhost run (5x 50 ms RTT each), and
# makes the editor feel "instant" on a remote dev box where each fetch is
# 100+ ms. The body is gzipped by GZipMiddleware (configured at the app
# root); typical compressed sizes are 30-60% of raw JSON.
#
# ``include_motion`` may be:
#   - omitted          → bundle WITHOUT animation_data (fastest cold open;
#                        viewer fetches motions lazily on user pick)
#   - "default"        → bundle the model's auto-detected default motion
#   - "<motion_name>"  → bundle the named motion (substring match, same
#                        rules as /api/animation_data)
#
# Tile PNGs are NOT bundled — they're binary PNGs served verbatim by
# /api/tile_png and load via THREE.TextureLoader. Including them as base64
# would inflate the JSON 35%+ for no win (THREE.TextureLoader can't
# consume them inline anyway without a Blob URL conversion).


@app.get("/api/model_bundle/{path:path}")
def api_model_bundle(path: str, inner: Optional[str] = None,
                     include_motion: str = ""):
    """Return mesh + skinned mesh + binding + animations [+ animation_data]
    in a single JSON response.

    Wire shape:
      {
        "filename":    "<path>",
        "inner":       "<inner|null>",
        "skinned":     <api_model_skinned shape>,
        "animations":  <api_animations shape>,
        "motion_data": <api_animation_data shape | null>,
        "errors":      {"skinned": "...", ...}   // present only on partial
      }

    Errors are surfaced PER-COMPONENT in the bundle's ``errors`` map so a
    single failed sub-call (e.g. a model with no skinned data) doesn't
    sink the whole bundle. The frontend can decide which components are
    mandatory.

    The endpoint is GET so it can ride the FastAPI middleware GZip layer
    and benefit from HTTP cache headers. Path forms match all other model
    endpoints (BML+inner, top-level .nj, hash form).
    """
    # Validate path / split inner — same logic as the constituent endpoints.
    base, effective_inner = _split_inner_with_query(path, inner)

    # Resolve the path once so a missing file 404s here, not inside three
    # sub-calls.
    p = _resolve_model_mesh_path(base)

    bundle: dict = {
        "filename": path,
        "inner":    effective_inner,
        "skinned":      None,
        "animations":   None,
        "motion_data":  None,
    }
    errors: dict[str, str] = {}

    # Component 1: skinned mesh + binding (the JSON shape the model
    # viewer's tryLoadSkinnedMesh consumes). Reuse the endpoint function
    # directly; it raises HTTPException on hard failure, otherwise returns
    # the same dict shape.
    try:
        bundle["skinned"] = api_model_skinned(path, inner=inner)
    except HTTPException as e:
        errors["skinned"] = e.detail if hasattr(e, "detail") else str(e)
    except Exception as e:  # pragma: no cover — defensive
        log.exception("bundle: skinned sub-call failed for %s", path)
        errors["skinned"] = f"internal: {e}"

    # Component 2: motion list. Fail-soft: if motions can't be enumerated
    # (e.g. .nj with no NJM siblings), surface an empty motions array
    # rather than a hard error so the viewer can still display the mesh.
    try:
        bundle["animations"] = api_animations(path, inner=inner)
    except HTTPException as e:
        # api_animations is itself fail-soft (returns empty motions on
        # missing siblings), so a 4xx here is unusual and worth recording.
        errors["animations"] = e.detail if hasattr(e, "detail") else str(e)
    except Exception as e:  # pragma: no cover — defensive
        log.exception("bundle: animations sub-call failed for %s", path)
        errors["animations"] = f"internal: {e}"

    # Component 3: optional default-motion keyframe data. Only fetched if
    # include_motion is set; the frontend can lazily request motion data
    # later via /api/animation_data when the user picks a different motion.
    if include_motion:
        anim_block = bundle.get("animations") or {}
        motions = anim_block.get("motions") or []
        chosen: Optional[str] = None
        if include_motion == "default":
            di = anim_block.get("default_index")
            if isinstance(di, int) and 0 <= di < len(motions):
                chosen = motions[di].get("name")
        else:
            # substring match to mirror api_animation_data
            chosen = include_motion
        if chosen and motions:
            try:
                bundle["motion_data"] = api_animation_data(
                    path, motion=chosen, inner=inner,
                )
            except HTTPException as e:
                errors["motion_data"] = e.detail if hasattr(e, "detail") else str(e)
            except Exception as e:  # pragma: no cover
                log.exception("bundle: motion_data sub-call failed for %s", path)
                errors["motion_data"] = f"internal: {e}"
        elif include_motion != "default":
            # User asked for a specific motion but none matched — record
            # for diagnostics. Default-not-found is silent (no motions
            # exist for this model).
            errors["motion_data"] = f"motion {include_motion!r} not in animation list"

    if errors:
        bundle["errors"] = errors

    # Wave 7 (2026-04-26): tile pre-warm.
    #
    # Once the binding has resolved how many tiles this model references,
    # kick off a background ThreadPoolExecutor that pre-decodes each tile
    # PNG into the tile_png cache. The bundle response returns immediately
    # so the frontend doesn't block; by the time the browser fires its
    # /api/tile_png GETs the cache is already warm and each call serves
    # in <5 ms instead of paying 30-100 ms of cold XVR decode.
    #
    # We DON'T await the pool — fire-and-forget. If the user moves on,
    # the wasted decode is small (<1 s on dragon-class), and any pre-warm
    # pages that DO land help future opens of the same asset.
    try:
        skinned = bundle.get("skinned") or {}
        binding_data = skinned.get("binding_data") or {}
        xvmh_rows = binding_data.get("xvmh") or []
        if xvmh_rows:
            tile_indices = sorted({
                row.get("tile_index") for row in xvmh_rows
                if isinstance(row.get("tile_index"), int)
            })
            tile_indices = [t for t in tile_indices if t is not None and t >= 0]
            if tile_indices:
                # Derive the texture archive filename the frontend will
                # request via /api/tile_png/{filename}/{idx}. For BML-
                # inner models it's `<base>#<inner>.xvm`; for top-level
                # `.nj`/`.xj` it's the model's `<stem>.xvm` sibling.
                tex_filename: Optional[str] = None
                if effective_inner:
                    tex_filename = f"{base}#{effective_inner}.xvm"
                else:
                    # Top-level model — sibling .xvm. The frontend
                    # resolves this via fetchPreviewHint, but pre-warming
                    # the most likely candidate is cheap (a missing file
                    # just yields a 404 in the worker which we swallow).
                    pth = Path(base)
                    tex_filename = str(pth.with_suffix(".xvm"))
                _kick_tile_prewarm(tex_filename, tile_indices)
    except Exception:  # pragma: no cover — pre-warm is best-effort
        log.exception("bundle pre-warm scheduling failed for %s", path)

    return bundle


# ---------------------------------------------------------------------------
# /api/composite_bundle — multi-inner BML assembly (2026-04-30)
# ---------------------------------------------------------------------------
#
# Built to fix the "boss parts stacked at origin" rendering complaint:
# multi-part bosses (De Rol Le, Vol Opt, Dragon, ...) ship as a single
# .bml with several .nj inners, each rooted at its OWN origin (verified
# by probing every boss BML — root MeshTreeNode TRS is identity in
# every primary inner). The actual inter-part offsets live in PSOBB.exe
# entity-init code, NOT in the asset files.
#
# This endpoint returns a curated per-part TRS table (from
# ``formats/composite_assembly.py``) wrapped around the existing
# /api/model_skinned per-inner payloads. The frontend can then position
# each inner via the supplied (pos, rot_euler, scale, parent_inner)
# instead of dropping every part at world origin.
#
# Wire shape mirrors /api/model_bundle but expands ``skinned`` into a
# ``parts`` array (one entry per inner). For unknown BMLs we fall back
# to identity placement of every primary inner so the endpoint is
# usable for non-composite assets too — caller decides how to render.
@app.get("/api/composite_bundle/{path:path}")
def api_composite_bundle(path: str):
    """Return the composite multi-inner assembly for a BML.

    Wire shape::

      {
        "filename": "bm_boss2_de_rol_le.bml",
        "parts": [
          {
            "inner": "boss2_b_derorure_body.nj",
            "pos": [x, y, z],
            "rot_euler": [rx, ry, rz],   # ZYX intrinsic, radians
            "scale": [sx, sy, sz],
            "parent_inner": null | "<other inner name>",
            "notes": "<provenance / caveat string>",
            "skinned": <api_model_skinned wire shape>,
            "binding": <per-inner binding (passthrough from skinned)>
          },
          ...
        ],
        "source": "hand-curated" | "identity-fallback" | "...",
        "errors": { "<inner>": "<error message>" }   # only on partial
      }

    Behaviour:
      * Curated table miss -> falls back to identity placement for
        every primary ``.nj`` inner discovered in the BML directory.
      * Per-inner skinned-mesh failure -> recorded in ``errors`` and
        the part's ``skinned`` is null (the caller can still position
        the empty slot or skip it).
      * Non-BML inputs are rejected with HTTP 400 — composite assembly
        only makes sense for BML containers.

    The endpoint reuses ``api_model_skinned`` for each part so the
    per-inner mesh / bone / binding logic stays in one place. Only
    the assembly metadata is new.
    """
    base, hash_inner = _split_inner_path(path)
    if hash_inner:
        # The composite endpoint always returns ALL parts; specifying
        # an inner here is a misuse.
        raise HTTPException(
            400,
            "composite_bundle does not accept an inner suffix; "
            "the endpoint returns every part of the BML",
        )

    p = _resolve_model_mesh_path(base)
    if p.suffix.lower() != ".bml":
        raise HTTPException(
            400,
            f"composite_bundle requires a .bml input (got {p.suffix!r}); "
            f"use /api/model_bundle for single-inner models",
        )

    bml_basename = p.name
    assembly = _lookup_composite_assembly(bml_basename)

    # Identity fallback: enumerate the BML's NJ inners and synthesise a
    # CompositeAssembly with every primary inner placed at the origin.
    # This keeps the endpoint useful even for BMLs we have no curated
    # data for — the frontend can still draw something.
    fallback_used = False
    if assembly is None:
        try:
            blob = p.read_bytes()
            entries = parse_bml(blob)
        except (OSError, ValueError) as e:
            raise HTTPException(400, f"BML parse failed: {e}")
        nj_entries = [
            ent for ent in entries
            if ent.name.lower().endswith(".nj")
        ]
        if not nj_entries:
            raise HTTPException(
                404,
                f"BML {bml_basename!r} contains no .nj inners; nothing to assemble",
            )
        assembly = CompositeAssembly(
            bml_path=bml_basename.lower(),
            parts=[
                CompositePart(
                    inner_nj=ent.name,
                    pos=(0.0, 0.0, 0.0),
                    rot_euler=(0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                    parent_inner=None,
                    notes="identity fallback (no curated placement data)",
                )
                for ent in nj_entries
            ],
            source="identity-fallback",
        )
        fallback_used = True

    parts_payload: list[dict] = []
    errors: dict[str, str] = {}
    for part in assembly.parts:
        entry: dict = {
            "inner":        part.inner_nj,
            "pos":          list(part.pos),
            "rot_euler":    list(part.rot_euler),
            "scale":        list(part.scale),
            "parent_inner": part.parent_inner,
            "notes":        part.notes,
            "skinned":      None,
            "binding":      None,
        }
        try:
            skinned = api_model_skinned(base, inner=part.inner_nj)
            entry["skinned"] = skinned
            # Hoist the binding so a frontend that only consumes
            # composite_bundle doesn't need to re-walk the skinned
            # payload to find textures.
            if isinstance(skinned, dict):
                entry["binding"] = skinned.get("binding_data")
        except HTTPException as e:
            detail = e.detail if hasattr(e, "detail") else str(e)
            errors[part.inner_nj] = str(detail)
        except Exception as e:  # pragma: no cover — defensive
            log.exception(
                "composite_bundle: skinned sub-call failed for %s#%s",
                base, part.inner_nj,
            )
            errors[part.inner_nj] = f"internal: {e}"
        parts_payload.append(entry)

    response: dict = {
        "filename": bml_basename,
        "parts":    parts_payload,
        "source":   assembly.source,
    }
    if fallback_used:
        # Surface a hint so the frontend can show a "composite layout
        # unknown" badge instead of pretending the identity placement
        # is meaningful.
        response["fallback"] = True
    if errors:
        response["errors"] = errors
    return response


# Wave 7 — bundle pre-warm executor.
#
# Single shared ThreadPoolExecutor (4 workers, lazily created) that runs
# tile-PNG decode jobs in the background after a bundle response goes
# out. Threading is sufficient since the XVR decode work releases the
# GIL inside Pillow's C codec. Capping at 4 keeps the pool small enough
# that a rapid-click flurry of 10 different bundle requests doesn't
# saturate worker bandwidth.
#
# The pool is intentionally NOT shut down at process exit — daemon
# threads are killed cleanly by the interpreter, and a graceful
# shutdown adds complexity for no user-visible benefit.

_TILE_PREWARM_EXECUTOR: Optional["concurrent.futures.ThreadPoolExecutor"] = None
_TILE_PREWARM_LOCK = threading.Lock()
# Cap on outstanding pre-warm jobs so a pathological burst (1000-asset
# script bombarding the bundle endpoint) doesn't queue thousands of
# decode tasks behind whatever the user is actually viewing.
_TILE_PREWARM_MAX_QUEUED = 64
_TILE_PREWARM_QUEUE_SIZE = 0

# Wave 7 follow-up: rapid-click prewarm kill switch.
#
# Each bundle response stamps a fresh "current asset" key on the prewarm
# pool. When the user moves on to a different asset, jobs scheduled for
# the PRIOR asset see a stale key and silently no-op when the worker
# picks them up. This keeps the worker pool free for the asset the user
# actually cares about — without that, a 10-rapid-click flurry queues
# 100+ stale decode jobs ahead of the live one.
_TILE_PREWARM_CURRENT: Optional[str] = None


def _ensure_tile_prewarm_executor():
    """Lazy-init the prewarm pool. Imports concurrent.futures locally so
    test environments that monkeypatch the module aren't surprised at
    import time.
    """
    global _TILE_PREWARM_EXECUTOR
    if _TILE_PREWARM_EXECUTOR is not None:
        return _TILE_PREWARM_EXECUTOR
    import concurrent.futures
    with _TILE_PREWARM_LOCK:
        if _TILE_PREWARM_EXECUTOR is None:
            _TILE_PREWARM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="tile-prewarm",
            )
    return _TILE_PREWARM_EXECUTOR


def _prewarm_one_tile(tex_filename: str, tile_idx: int, asset_key: str) -> None:
    """Worker body — decode one tile and seed the tile_png cache.

    `asset_key` is the prewarm-current key at the time the job was
    scheduled. If the user has since clicked a different asset (which
    bumps `_TILE_PREWARM_CURRENT`), this job no-ops to keep worker
    bandwidth available for the live asset.
    """
    global _TILE_PREWARM_QUEUE_SIZE
    try:
        # Stale-asset gate.
        if _TILE_PREWARM_CURRENT is not None and asset_key != _TILE_PREWARM_CURRENT:
            return
        try:
            prs = _materialize_inner_for_extract(tex_filename)
        except HTTPException:
            return
        if not prs.exists():
            return
        if tile_idx < 0 or tile_idx > MAX_TILE_INDEX:
            return
        # Re-check the staleness flag right before the heavy work — the
        # user could have moved on while we were materialising.
        if _TILE_PREWARM_CURRENT is not None and asset_key != _TILE_PREWARM_CURRENT:
            return

        def _do_extract() -> Path:
            manifest = extract_tiles(prs)
            tile = next(
                (t for t in manifest["tiles"] if t["index"] == tile_idx), None,
            )
            if not tile:
                raise HTTPException(404, "no such tile index")
            p = Path(manifest["tiles_dir"]) / tile["filename"]
            if not p.exists():
                raise HTTPException(500, "tile png missing on disk")
            return p

        # Same code path /api/tile_png uses — populates both LRUs.
        try:
            _serve_tile_png_cached(prs, tile_idx, tex_filename, _do_extract)
        except HTTPException:
            return
        except Exception:
            log.exception("prewarm: tile %s/%d decode failed",
                          tex_filename, tile_idx)
    finally:
        with _TILE_PREWARM_LOCK:
            _TILE_PREWARM_QUEUE_SIZE = max(0, _TILE_PREWARM_QUEUE_SIZE - 1)


def _kick_tile_prewarm(tex_filename: str, tile_indices: list) -> None:
    """Submit decode jobs for `tile_indices` against `tex_filename` to
    the prewarm pool. No-op if the queue is already saturated (back-
    pressure in lieu of unbounded growth).

    Wave 7 follow-up: stamps the prewarm-current key with this archive
    so jobs queued by EARLIER (now-stale) bundle responses no-op when
    workers pick them up. Reduces wasted CPU during rapid-click flurries.
    """
    global _TILE_PREWARM_QUEUE_SIZE, _TILE_PREWARM_CURRENT
    if not tex_filename or not tile_indices:
        return
    pool = _ensure_tile_prewarm_executor()
    with _TILE_PREWARM_LOCK:
        # The new asset becomes "current". Jobs already in flight for
        # the prior asset will see this on their next staleness check.
        _TILE_PREWARM_CURRENT = tex_filename
        if _TILE_PREWARM_QUEUE_SIZE >= _TILE_PREWARM_MAX_QUEUED:
            return
        budget = _TILE_PREWARM_MAX_QUEUED - _TILE_PREWARM_QUEUE_SIZE
        scheduled = list(tile_indices)[:budget]
        _TILE_PREWARM_QUEUE_SIZE += len(scheduled)
    for idx in scheduled:
        pool.submit(_prewarm_one_tile, tex_filename, idx, tex_filename)


@app.get("/api/tile_prewarm_stats")
def api_tile_prewarm_stats():
    """Diagnostic — current pre-warm queue depth + tile cache stats.

    Used by tests/test_bundle_prewarm.py to verify the pool drained
    after a bundle GET.
    """
    return {
        "queue_size": _TILE_PREWARM_QUEUE_SIZE,
        "max_queued": _TILE_PREWARM_MAX_QUEUED,
        "tile_png_cache": _tile_png_cache_stats(),
    }


# ---------------------------------------------------------------------------- color/state variants
#
# `/api/variants/<bml_path>` returns the list of detected variant pairs
# for one model — sibling BMLs (Booma/Gobooma, Lappy/PalRappy) and
# intra-BML NJTL slot groups (Mericarol/Mericus/Merikle). The frontend
# uses this to render a variant-picker strip above the 3D viewport.
#
# See ``formats/variant_detector.py`` for the heuristic + family table.
# Wire shape:
#   {
#     "filename": "<input>",
#     "variants": [
#       {
#         "path":         "<bml or bml#inner?slot_group=N>",
#         "label":        "Mericus",
#         "variant_kind": "color"|"lod"|"damaged",
#         "icon_color":   "#hex",
#         "slot_group":   <int|null>,   // intra-BML slot offset
#         "slot_count":   <int|null>,   // intra-BML slot stride
#         "is_self":      bool          // is this the model the user opened
#       },
#       ...
#     ]
#   }


# Dominant-color cache for variant icons. Keyed by (bml_path, mtime_ns,
# slot_group) so changes to the BML invalidate. Bounded by entry count.
_VARIANT_COLOR_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_VARIANT_COLOR_CACHE_MAX = 1024

# 8-swatch palette for dominant-color snap. Each entry is a (R,G,B) tuple
# in linear-ish 0..255 space; the palette covers the rough hues that
# appear in PSOBB monster textures.
_DOMINANT_SWATCH_RGB = [
    (0x7f, 0xaf, 0x42),  # green
    (0x5b, 0x8d, 0xf0),  # blue
    (0xe5, 0x48, 0x48),  # red
    (0xe8, 0xc5, 0x42),  # yellow
    (0xa9, 0x60, 0xe8),  # purple
    (0x56, 0xc8, 0xc8),  # cyan
    (0xe8, 0x89, 0x3a),  # orange
    (0xd9, 0x6b, 0xb8),  # pink
]
_DOMINANT_SWATCH_HEX = [
    "#7faf42", "#5b8df0", "#e54848", "#e8c542",
    "#a960e8", "#56c8c8", "#e8893a", "#d96bb8",
]


def _snap_rgb_to_swatch(rgb: tuple[int, int, int]) -> str:
    """Find the closest swatch in ``_DOMINANT_SWATCH_RGB`` to ``rgb``.

    Uses squared-Euclidean in RGB space — adequate for the ~8 widely
    spaced hues we have. (Lab would be more perceptually uniform but
    needs a colorspace conversion that costs more than the win.)

    Returns the HEX string from ``_DOMINANT_SWATCH_HEX`` at the same
    index as the closest swatch.
    """
    best_idx = 0
    best_d = 1 << 30
    for i, (r, g, b) in enumerate(_DOMINANT_SWATCH_RGB):
        dr = r - rgb[0]
        dg = g - rgb[1]
        db = b - rgb[2]
        d = dr * dr + dg * dg + db * db
        if d < best_d:
            best_d = d
            best_idx = i
    return _DOMINANT_SWATCH_HEX[best_idx]


def _sample_dominant_color_for_variant(
    bml_filename: str,
    *,
    inner_nj_name: Optional[str] = None,
    slot_group_idx: Optional[int] = None,
    slot_count: Optional[int] = None,
) -> Optional[str]:
    """Compute (or fetch from cache) a dominant-color HEX for one variant.

    The variant is identified by its ``bml_filename``. For intra-BML
    variants we additionally take ``inner_nj_name`` + ``slot_group_idx``
    + ``slot_count`` so we sample the FIRST tile of THAT slot group.

    Returns ``None`` on any error (missing XVM, decode failure, etc.) so
    the caller can fall back to the synthetic palette.
    """
    try:
        scoped_p = (DATA_DIR / bml_filename).resolve()
        scoped_p.relative_to(DATA_DIR)
        if not scoped_p.exists():
            scoped_p = (LIVE_DATA_DIR / bml_filename).resolve()
            scoped_p.relative_to(LIVE_DATA_DIR)
            if not scoped_p.exists():
                return None
    except (OSError, ValueError):
        return None

    try:
        mtime_ns = scoped_p.stat().st_mtime_ns
    except OSError:
        return None
    cache_key = (str(scoped_p), mtime_ns, inner_nj_name or "", slot_group_idx or 0)
    cached = _VARIANT_COLOR_CACHE.get(cache_key)
    if cached is not None:
        _VARIANT_COLOR_CACHE.move_to_end(cache_key)
        return cached

    try:
        # Resolve the inner NJ name. For cross-BML variants we walk the
        # BML to find its first .nj entry.
        nj_name = inner_nj_name
        if nj_name is None:
            try:
                blob = scoped_p.read_bytes()
                entries = parse_bml(blob)
            except (OSError, ValueError):
                return None
            nj_entry = next((e for e in entries if e.name.lower().endswith(".nj")), None)
            if nj_entry is None:
                return None
            nj_name = nj_entry.name

        # Build the inner XVM path that _materialize_inner_for_extract uses.
        inner_xvm_path = f"{bml_filename}#{nj_name}.xvm"
        try:
            prs = _materialize_inner_for_extract(inner_xvm_path)
        except HTTPException:
            return None
        if not prs.exists():
            return None
        manifest = extract_tiles(prs)

        # Pick the tile index to sample: for intra-BML variants the
        # variant's slot range is [slot_group_idx * slot_count,
        # slot_group_idx * slot_count + slot_count). Within that range we
        # choose the SMALLEST tile (typically a 512×512 face/accent
        # texture that carries the color identity — the larger 1024×1024
        # tiles are body skin, often nearly identical across variants).
        # For cross-BML variants we just sample tile 0 — but we still
        # apply the smallest-tile heuristic across all available tiles
        # because the "main" texture is usually the smallest dimension
        # (PSOBB packs the body in 1024×, accents in 512×).
        if slot_group_idx is not None and slot_count is not None:
            slot_lo = slot_group_idx * slot_count
            slot_hi = slot_lo + slot_count
        else:
            slot_lo = 0
            slot_hi = max(1, len(manifest["tiles"]))

        candidates = [
            t for t in manifest["tiles"]
            if slot_lo <= t["index"] < slot_hi
        ]
        if not candidates:
            return None

        # Pick the smallest tile (by area).
        def _tile_area(t):
            try:
                from PIL import Image
                with Image.open(Path(manifest["tiles_dir"]) / t["filename"]) as im:
                    return im.size[0] * im.size[1]
            except Exception:
                return 1 << 30
        chosen_tile = min(candidates, key=_tile_area)
        png_path = Path(manifest["tiles_dir"]) / chosen_tile["filename"]
        if not png_path.exists():
            return None

        # Sample mean RGB. Resize to a 16×16 thumb for speed.
        from PIL import Image
        with Image.open(png_path) as im:
            im = im.convert("RGB").resize((16, 16))
            px = list(im.getdata())
        if not px:
            return None
        # Skip near-black pixels (background) when computing the mean.
        # PSOBB textures are full-bleed so a hard alpha cutoff isn't
        # available — we exclude pixels with all channels < 32.
        r_sum = g_sum = b_sum = 0
        n_kept = 0
        for r, g, b in px:
            if r < 32 and g < 32 and b < 32:
                continue
            r_sum += r
            g_sum += g
            b_sum += b
            n_kept += 1
        if n_kept == 0:
            # All near-black — average everything.
            n_kept = len(px)
            for r, g, b in px:
                r_sum += r
                g_sum += g
                b_sum += b
        mean = (r_sum // n_kept, g_sum // n_kept, b_sum // n_kept)
        snapped = _snap_rgb_to_swatch(mean)
    except Exception as e:  # pragma: no cover - defensive
        log.warning("variant color sample failed for %s: %s", bml_filename, e)
        return None

    _VARIANT_COLOR_CACHE[cache_key] = snapped
    while len(_VARIANT_COLOR_CACHE) > _VARIANT_COLOR_CACHE_MAX:
        try:
            _VARIANT_COLOR_CACHE.popitem(last=False)
        except KeyError:
            break
    return snapped


@app.get("/api/variants/{path:path}")
def api_variants(path: str):
    """Return color/state variants for a model.

    Path is just the BML's filename (or with a ``#inner`` fragment, which
    we ignore — variants are container-level). Returns a 200 with an
    empty `variants` array if no siblings are detected.
    """
    base, _hash_inner = _split_inner_path(path)
    # Variant detection works at the BML level; if the caller passed an
    # .nj or .xj it has no concept of "sibling variants" so we just return
    # empty.
    p = safe_data_path(base)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"asset not found: {base}")
    if p.suffix.lower() != ".bml":
        return {"filename": path, "variants": []}

    try:
        # Search both the dev mirror and the live install so the user
        # sees variants even when only one of the two has the asset
        # extracted into PSOBB.IO/data.
        candidates: list[VariantInfo] = []
        seen_paths: set[str] = set()
        for root in (DATA_DIR, LIVE_DATA_DIR):
            scoped_p = (root / p.name).resolve()
            if not scoped_p.exists():
                continue
            try:
                scoped_p.relative_to(root)
            except ValueError:
                continue
            for v in _detect_variants(scoped_p, data_dir=root):
                key = v.path + (f"?slot={v.slot_group}" if v.slot_group is not None else "")
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                candidates.append(v)
    except (OSError, ValueError) as e:
        log.warning("variants: detector failed for %s: %s", path, e)
        candidates = []

    # When BOTH a cross-BML self-entry AND an intra-BML group-0 entry
    # exist, drop the cross-BML self-entry — the intra-BML one is the
    # richer record (it knows about the slot group). Without this we'd
    # show "Mericarol / Mericarol (low-poly) / Mericarol / Mericus / Merikle"
    # which is confusing.
    has_intra_self = any(
        v.is_self and v.slot_group is not None for v in candidates
    )
    if has_intra_self:
        candidates = [
            v for v in candidates
            if not (v.is_self and v.slot_group is None)
        ]

    out = []
    for v in candidates:
        # Try to refine the icon color by sampling the first XVR tile of
        # the variant's texture set. Falls back to the synthetic swatch
        # if the sample fails (missing XVM, decode error, etc.).
        sampled = None
        try:
            if v.slot_group is not None:
                # Intra-BML variant — extract the inner NJ name from the
                # path's "#" fragment.
                hash_idx = v.path.find("#")
                if hash_idx > 0:
                    bml_part = v.path[:hash_idx]
                    inner_part = v.path[hash_idx + 1:]
                    sampled = _sample_dominant_color_for_variant(
                        bml_part,
                        inner_nj_name=inner_part,
                        slot_group_idx=v.slot_group,
                        slot_count=v.slot_count,
                    )
            else:
                # Cross-BML variant — sample the first XVR of the BML's
                # first NJ.
                sampled = _sample_dominant_color_for_variant(v.path)
        except Exception:  # pragma: no cover - defensive
            sampled = None
        out.append({
            "path": v.path,
            "label": v.label,
            "variant_kind": v.variant_kind,
            "icon_color": sampled or v.icon_color,
            "slot_group": v.slot_group,
            "slot_count": v.slot_count,
            "is_self": v.is_self,
        })
    return {"filename": path, "variants": out}


# ---------------------------------------------------------------------------- asset format readers
#
# These endpoints expose the read-only `formats/iff.py` and `formats/afs.py`
# parsers so the JS frontend can introspect raw data files without
# spawning a subprocess. They are GET-only; no DATA_DIR mutation.
#
# Cap: refuse to load files larger than this into RAM to parse. PSOBB
# data files are well under 64 MB; map_aancient03.xvm at ~47 MB is the
# biggest legitimate file. AFS archives in the install max out around
# 2 MB. This bound exists purely to make a malicious large rename a
# clean 413 instead of an OOM.
ASSET_PARSE_MAX_BYTES = 64 * 1024 * 1024

# Extensions whose content we recognise as PSO IFF.
#
# Both ``.nj`` and ``.xj`` use the little-endian IFF chunked container
# (NJCM/NJTL/POF0); the difference between them is the inner format
# of the NJCM chunk (chunk-streamed Ninja-Nj vs descriptor-table Xj).
# /api/model_mesh dispatches between the two parsers on inner-file
# extension; /api/iff and /api/model/.../skeleton accept both because
# the IFF wrapper is identical.
IFF_EXTENSIONS = (".nj", ".njm", ".njs", ".xj")
# .njs files use the NSSM IFF chunk (Ninja State-machine Sequence Motion)
# — animation-only, like .njm. The chunk parser walks both correctly
# (NSSM lacks the NJCM chunk so parse_nj_file returns []), so dispatch
# routes through the same code path as .njm.


def _read_asset_for_parse(filename: str) -> bytes:
    """Resolve `filename` under DATA_DIR and read it for in-memory parsing.

    Reuses safe_data_path() so the same path-traversal guard applies as
    every other endpoint. Refuses oversized files up front so a bogus
    rename (e.g. swapping a 2 GB blob to .nj) can't OOM the server.
    """
    p = safe_data_path(filename)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"asset not found: {filename}")
    sz = p.stat().st_size
    if sz > ASSET_PARSE_MAX_BYTES:
        raise HTTPException(
            413,
            f"asset too large to parse in-memory: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
        )
    return p.read_bytes()


@app.get("/api/asset/{filename}")
def api_asset_meta(filename: str, meta: int = 0):
    """Return container-level metadata for a recognised asset.

    Currently understands:
      .afs  -> {format: "afs", count, sizes:[size_per_entry], names:[]}.
               Names are an empty list because PSOBB AFS files do not
               carry a name table; callers can reconcile by sibling
               index files (e.g. ItemPMT bin) which is a separate layer.

    Other extensions return 400 - they are either reachable via existing
    typed endpoints (.prs/.xvm via /api/tiles) or not yet supported.

    The `?meta=1` query parameter is required to disambiguate from a
    future "fetch raw bytes" verb on the same path; calling without it
    is a 400 so misuse is loud.
    """
    if not meta:
        raise HTTPException(
            400,
            "missing or zero `?meta=1` - this endpoint only returns metadata; "
            "use the tiles / iff variants for content",
        )
    ext = Path(filename).suffix.lower()
    if ext != ".afs":
        raise HTTPException(
            400,
            f"asset meta not supported for extension {ext!r} "
            f"(only .afs is recognised by this endpoint)",
        )
    blob = _read_asset_for_parse(filename)
    try:
        files = parse_afs(blob)
    except ValueError as e:
        raise HTTPException(400, f"AFS parse failed: {e}")
    return {
        "filename": filename,
        "format": "afs",
        "count": len(files),
        "sizes": [len(f) for f in files],
        # PSOBB AFS archives have no embedded name table. We surface an
        # empty list rather than omitting the key so the JS shape is
        # stable.
        "names": [],
    }


@app.get("/api/asset/{filename}/iff")
def api_asset_iff(filename: str):
    """Return a chunk listing for an IFF-flavoured asset.

    Currently accepts `.nj` and `.njm` files (top-level NJCM/NMDM/POF0
    chunks). Returns:

        {
          "filename": "<name>",
          "format": "iff",
          "chunks": [{"type": "NJCM", "size": 0x52b0}, ...],
          "total_chunks": 2,
        }

    On malformed input returns 400 with a clean error message; the
    parser raises ValueError which we translate, never leaking a 500.
    """
    ext = Path(filename).suffix.lower()
    if ext not in IFF_EXTENSIONS:
        raise HTTPException(
            400,
            f"IFF parse not supported for extension {ext!r} "
            f"(supported: {', '.join(IFF_EXTENSIONS)})",
        )
    blob = _read_asset_for_parse(filename)
    try:
        chunks = parse_iff(blob)
    except ValueError as e:
        raise HTTPException(400, f"IFF parse failed: {e}")
    return {
        "filename": filename,
        "format": "iff",
        "chunks": [{"type": c.type, "size": len(c.data)} for c in chunks],
        "total_chunks": len(chunks),
    }


# ---------------------------------------------------------------------------- BML container reader
#
# BML (Binary Model Library) holds the bulk of PSOBB's models and their
# inline texture archives - 365 BMLs in the install. These endpoints
# expose the read-only `formats/bml.py` parser. They are GET-only and
# accept paths in two locations:
#
#   1. DATA_DIR    (the dev mirror used for editing) - any file in here
#   2. LIVE_DATA_DIR (`~/PSOBB.IO/data/`) - read-only
#      ingestion so the editor can manifest the clean install without
#      copying every BML into the dev mirror first.
#
# The {path:path} parameter is a single path component (no slashes);
# nested directories are not supported because the install is flat.
# Path-traversal characters / separators are rejected up front.

# Inner-PRS decompress timeout per file (subprocess via PuyoToolsCli).
TIMEOUT_BML_PRS = 60


def _resolve_bml_path(path: str) -> Path:
    """Resolve a BML path under DATA_DIR or LIVE_DATA_DIR (read-only).

    Mirrors safe_data_path()'s injection guard but extends the search
    to LIVE_DATA_DIR for read-only ingestion. The search order is:
    DATA_DIR first (so an in-progress edit shadows the live install),
    then LIVE_DATA_DIR.
    """
    return _resolve_under_roots(
        path,
        (DATA_DIR, LIVE_DATA_DIR),
        label="path",
        missing_msg=f"BML not found in DATA_DIR or LIVE_DATA_DIR: {path}",
    )


def _read_bml(path: str) -> bytes:
    """Resolve a BML and read it for in-memory parse.

    Refuses oversized files up front. The biggest BML in the install
    is bm_boss3_volopt.bml at ~11 MB, well under the cap.
    """
    p = _resolve_bml_path(path)
    if p.suffix.lower() != ".bml":
        raise HTTPException(400, f"not a .bml file: {path}")
    sz = p.stat().st_size
    if sz > ASSET_PARSE_MAX_BYTES:
        raise HTTPException(
            413,
            f"BML too large to parse in-memory: {sz} > {ASSET_PARSE_MAX_BYTES} bytes",
        )
    return p.read_bytes()


@app.get("/api/bml/{path:path}/list")
def api_bml_list(path: str):
    """List the inner files of a BML container.

    Response shape:
        {
          "path": "<input path>",
          "count": N,
          "entries": [
            {name, size_compressed, size_decompressed, has_texture, tex_size_compressed},
            ...
          ]
        }

    Returns 400 on malformed BML / non-.bml extension, 404 if the BML
    is not found in DATA_DIR or LIVE_DATA_DIR, 413 if the file is
    larger than ASSET_PARSE_MAX_BYTES.

    For BMLs in ``EXPECTED_BML_INNER_COUNTS`` (Wave 2 ground-truth audit),
    an inner-count check runs after parsing: when the discovered
    ``.nj`` + ``.xj`` count is below the audited expected count, a WARN
    is logged so a future walker regression surfaces as an observable.
    The response payload itself is unchanged — we never fail-closed on a
    mismatch.
    """
    blob = _read_bml(path)
    try:
        entries = parse_bml(blob)
    except ValueError as e:
        raise HTTPException(400, f"BML parse failed: {e}")

    # Inner-discovery validation (audit-ground-truth comparison). Uses
    # only the BML basename so requests with a leading directory still
    # match the audit's flat keying scheme. The asset tree never sends
    # a directory component (manifest paths are flat) but stay defensive.
    bml_name = Path(path).name
    expected = expected_bml_inner_count(bml_name)
    if expected is not None:
        actual_model_count = sum(
            1 for ent in entries
            if ent.name.lower().endswith((".nj", ".xj"))
        )
        if actual_model_count < expected:
            log.warning(
                "inner-discovery: %s reports %d .nj/.xj inners, expected >= %d "
                "(walker may be filtering entries; see "
                "_reports/inner_discovery_audit.md)",
                bml_name, actual_model_count, expected,
            )

    return {
        "path": path,
        "count": len(entries),
        "entries": [
            {
                "name": e.name,
                "size_compressed": e.size_compressed,
                "size_decompressed": e.size_decompressed,
                "has_texture": e.has_texture,
                "tex_size_compressed": e.tex_size_compressed,
            }
            for e in entries
        ],
    }


@app.get("/api/bml/{path:path}/extract/{name}")
def api_bml_extract(path: str, name: str):
    """Return one inner file from a BML, PRS-decompressed.

    Streams the raw bytes back as application/octet-stream with a
    Content-Disposition naming the inner file. The frontend can save
    or pipe the buffer through other parsers (IFF, NJCM, NMDM).

    Returns 400 if the BML is malformed or the named entry is missing,
    404 if the BML itself is missing, 413 if oversized, 504 if the
    PuyoToolsCli subprocess times out.
    """
    _validate_inner_name(name, msg="invalid entry name", required=True)
    blob = _read_bml(path)
    try:
        entries = parse_bml(blob)
    except ValueError as e:
        raise HTTPException(400, f"BML parse failed: {e}")
    target = next((e for e in entries if e.name == name), None)
    if target is None:
        raise HTTPException(404, f"no entry named {name!r} in {path}")
    try:
        # Re-extract just the one we want — cheaper than extract_bml()
        # which would walk every entry. PRS is now in-process (see
        # ``formats.bml._prs_decompress``); we pipe through the shared
        # LRU so a "list → extract" round trip on the same inner is O(1).
        from formats.bml import decompress_prs_cached
        # Re-resolve the path through _resolve_bml_path so we have the
        # filesystem location for the cache key.
        resolved_p = _resolve_bml_path(path)
        slice_start = target.offset
        slice_end = slice_start + target.size_compressed
        out = decompress_prs_cached(
            resolved_p, resolved_p.stat().st_mtime_ns, name,
            lambda: bytes(blob[slice_start:slice_end]),
        )
    except (RuntimeError, ValueError) as e:
        # Subprocess failure / missing tool / timeout / malformed PRS
        raise HTTPException(502, f"BML extract failed: {e}")
    headers = {
        "Content-Disposition": f'attachment; filename="{name}"',
    }
    return Response(content=out, media_type="application/octet-stream", headers=headers)


@app.get("/api/bml/{path:path}/texture/{name}")
def api_bml_texture(path: str, name: str):
    """Return the texture archive (XVM) for one BML entry, decompressed.

    The XVM is PRS-compressed inside the BML; we decompress and stream
    back the raw XVMH bytes. The Content-Type is application/x-xvm so
    the frontend can pipe it through xvr_codec.py.

    Returns 404 if the entry has no texture, OR if the BML / entry
    itself is missing. 400 on bad path / parse, 502 on subprocess
    failure.
    """
    _validate_inner_name(name, msg="invalid entry name", required=True)
    blob = _read_bml(path)
    try:
        tex = extract_bml_texture(blob, name)
    except ValueError as e:
        # Distinguish "no such entry" from a parse error to give callers
        # a useful 404. extract_bml_texture raises ValueError for both;
        # we re-route by checking whether the entry list contains the
        # name (cheap, parse_bml is pure Python).
        try:
            entries = parse_bml(blob)
        except ValueError:
            raise HTTPException(400, f"BML parse failed: {e}")
        if not any(ent.name == name for ent in entries):
            raise HTTPException(404, f"no entry named {name!r} in {path}")
        raise HTTPException(400, f"BML texture extract failed: {e}")
    except RuntimeError as e:
        raise HTTPException(502, f"BML texture extract failed: {e}")
    if tex is None:
        raise HTTPException(404, f"entry {name!r} has no texture")
    headers = {
        "Content-Disposition": f'attachment; filename="{name}.xvm"',
    }
    return Response(content=tex, media_type="application/x-xvm", headers=headers)


# ============================================================================
# AFS archive endpoints (2026-04-25)
# ----------------------------------------------------------------------------
# Sega AFS containers (ItemModel.afs / ItemTexture.afs / pl?tex.afs / ...)
# are inflated lazily into manifest entries by manifest._synthesize_afs_entries
# (path form ``<archive>#<NNNN>_<name>``). The endpoint here covers the
# direct list path:
#     GET /api/afs/{archive}/list           list inner-blob metadata
# Per-inner-blob fetch goes through the standard /api/asset/{path} route
# using the `<archive>#<NNNN>_<name>` inner-syntax instead.
# ============================================================================

@app.get("/api/afs/{archive}/list")
def api_afs_list(archive: str):
    """List the inner blobs of an AFS container.

    Response shape:
        {
          "archive": "<input archive>",
          "count": N,
          "entries": [
            {index, name, size, magic_hex, inner_format, inner_category,
             inner_ext, compressed},
            ...
          ]
        }

    Returns 400 on a malformed archive / non-.afs extension, 404 if the
    archive is missing, 413 if the file is larger than ASSET_PARSE_MAX_BYTES.
    """
    p = _resolve_base_path(archive)
    if p.suffix.lower() != ".afs":
        raise HTTPException(400, f"not an AFS archive: {archive}")
    # Single stat() reused for guard + diagnostic — was previously
    # stat()ing twice when over the cap.
    sz = p.stat().st_size
    if sz > ASSET_PARSE_MAX_BYTES:
        raise HTTPException(413, f"AFS too large to parse in-memory: {sz}")
    try:
        from formats import afs_reader as _afs_reader
    except ImportError as e:
        raise HTTPException(500, f"AFS reader unavailable: {e}")
    try:
        rows = _afs_reader.list_inner_blobs(p)
    except (ValueError, OSError) as e:
        raise HTTPException(400, f"AFS list failed: {e}")
    return {"archive": archive, "count": len(rows), "entries": rows}


# ============================================================================
# AI image generation (2026-04-24, additive)
# ----------------------------------------------------------------------------
# Provides img2img / inpaint / text2img / controlnet routing to whichever
# of the three providers is reachable: A1111-compatible WebUI on :7860,
# ComfyUI on :8188, or in-process HuggingFace Diffusers (only if the user
# installed the heavy deps themselves).
#
# Privacy: all v1 providers are hard-coded to localhost. The is_local
# flag is surfaced to the UI so future remote backends can show a clear
# "sends data to <host>" warning before sending tile data.
#
# Cache key contract: AI gen results pass through the same flow as
# /api/upscale on the frontend (state.tileEdits) — same registration,
# same A/B slider, same repack splice path. We do NOT write to the
# realesrgan cache_subdir, so AI gen and Upscayl runs never collide.
# ============================================================================

# Body limits — input PNG/mask is base64; tile sizes top out at ~1024² so
# 32 MB gives ~16 MB of binary which is plenty for a 1024² PNG + 1024² mask
# + prompt strings + generation knobs.
MAX_AIGEN_BODY = 32 * 1024 * 1024
# Fast-path liveness probe — frontend hits this every modal-open + every few
# seconds while the AI tab is active, so cache results in the providers
# themselves (the Provider implementations memoize ``is_available()``).
ALLOWED_AIGEN_MODES = ("img2img", "inpaint", "text2img", "controlnet")

# Lock map for AI gen requests. Keyed on (provider, filename, tile_index)
# so the user can't accidentally fire 5 concurrent generations on the same
# tile, but different tiles + different providers can still run in parallel.
_AIGEN_LOCKS: "OrderedDict[str, threading.Lock]" = OrderedDict()
MAX_AIGEN_LOCKS = 256


@app.get("/api/aigen/providers")
def api_aigen_providers():
    """Liveness probe for all known AI-gen providers.

    Returns a list of {name, label, status, available, supported_modes,
    base_url, is_local, hint}. The frontend uses this to populate the
    provider dropdown; only ``available: true`` providers are clickable.

    Cached per-provider for ~5s so rapid UI refreshes don't spam the
    underlying WebUIs.
    """
    try:
        legacy = aigen_mod.list_providers_status()
    except Exception as e:  # noqa: BLE001 — defensive; never let the UI go dark
        log.exception("aigen providers probe failed")
        raise HTTPException(500, f"providers probe failed: {e}")
    # Also surface the MVP provider abstraction (local upscale + budget-gated
    # paid providers) with its cost model, plus the current budget state.
    try:
        mvp = _aigen_registry().describe_all()
        budget = _aigen_budget().snapshot()
    except Exception:  # noqa: BLE001 — never let the MVP layer take the UI dark
        log.exception("aigen MVP providers probe failed")
        mvp, budget = [], None
    return {"providers": legacy, "mvp_providers": mvp, "budget": budget}


@app.get("/api/aigen/models/{provider}")
def api_aigen_models(provider: str):
    """List models the given provider has available.

    A1111 returns its loaded sd-models list; ComfyUI returns the
    ``CheckpointLoaderSimple`` choices; Diffusers returns whatever the
    user has used so far in this process plus the curated default IDs.

    Returns ``{provider, models: [{name, label?}, ...]}``.
    """
    if provider not in aigen_mod.all_provider_names():
        raise HTTPException(400, f"unknown provider {provider!r}")
    p = aigen_mod.get_provider(provider)
    if p is None:
        return {"provider": provider, "models": [], "available": False}
    try:
        models = p.list_models()
    except (RuntimeError, OSError) as e:
        log.warning("aigen list_models failed for %s: %s", provider, e)
        return {"provider": provider, "models": [], "available": False, "error": str(e)}
    return {"provider": provider, "models": models, "available": p.is_available()}


class AigenRequest(BaseModel):
    provider: str
    mode: str
    filename: Optional[str] = None  # for "associate the result with a tile" UX
    tile_index: Optional[int] = Field(default=None, ge=0, le=MAX_TILE_INDEX)
    prompt: str = ""
    neg_prompt: str = ""
    denoise: float = 0.6
    steps: int = 30
    cfg: float = 7.0
    seed: int = -1
    src_b64: Optional[str] = None  # source PNG (if not provided we extract from filename+tile_index)
    mask_b64: Optional[str] = None
    controlnet: Optional[dict] = None
    model: Optional[str] = None
    target_w: Optional[int] = None
    target_h: Optional[int] = None
    work_w: Optional[int] = None
    work_h: Optional[int] = None


def _resolve_aigen_source(req: AigenRequest) -> tuple[Optional[str], int, int]:
    """Pull the source PNG either from req.src_b64 directly, or from the
    extracted-tile cache when filename+tile_index were provided.

    Returns (src_b64_or_None, src_w, src_h). Raises HTTPException on bad
    inputs (e.g. tile_index out of range for the named file).
    """
    if req.src_b64:
        # Caller passed bytes directly. We don't know the dim until we
        # decode; let the provider compute it via _imageutil.
        return req.src_b64, 0, 0
    if not req.filename or req.tile_index is None:
        return None, 0, 0
    prs = safe_data_path(req.filename)
    if not prs.exists():
        raise HTTPException(404, f"no such file: {req.filename}")
    manifest = extract_tiles(prs)
    tile = next((t for t in manifest["tiles"] if t["index"] == req.tile_index), None)
    if not tile:
        raise HTTPException(404, "no such tile")
    p = Path(manifest["tiles_dir"]) / tile["filename"]
    if not p.exists():
        raise HTTPException(500, "tile png missing on disk")
    return png_to_b64(p), int(tile["width"]), int(tile["height"])


@app.post("/api/aigen/generate")
def api_aigen_generate(req: AigenRequest, request: Request):
    """Run a generation through the named provider.

    Body shape: see ``AigenRequest``. Returns
    ``{out_b64, out_w, out_h, seed, generation_time_s, model, provider, mode, info}``.

    The frontend treats this exactly like a /api/upscale response —
    register in state.tileEdits, A/B-compare in the existing modal.
    """
    _enforce_body_size(request, MAX_AIGEN_BODY)
    if req.mode not in ALLOWED_AIGEN_MODES:
        raise HTTPException(400, f"mode must be one of {ALLOWED_AIGEN_MODES}")
    if req.provider not in aigen_mod.all_provider_names():
        raise HTTPException(400, f"unknown provider {req.provider!r}")
    p = aigen_mod.get_provider(req.provider)
    if p is None:
        raise HTTPException(503, f"provider {req.provider!r} unavailable (import failed)")
    if not p.is_available():
        raise HTTPException(
            503,
            f"provider {req.provider!r} is not running. Start it and try again "
            f"(hint: {aigen_mod.hint_for(req.provider)})",
        )
    if req.mode not in p.supported_modes:
        raise HTTPException(
            400, f"provider {req.provider!r} does not support mode {req.mode!r}",
        )
    src_b64, src_w, src_h = _resolve_aigen_source(req)
    needs_src = req.mode in ("img2img", "inpaint", "controlnet")
    if needs_src and not src_b64:
        raise HTTPException(400, f"mode {req.mode} requires src_b64 or filename+tile_index")
    if req.mode == "inpaint" and not req.mask_b64:
        raise HTTPException(400, "inpaint requires mask_b64 (white=repaint, black=preserve)")

    # Per-(provider, file, tile) lock so concurrent same-tile clicks don't race.
    lock_key = f"{req.provider}|{req.filename or ''}|{req.tile_index if req.tile_index is not None else ''}"
    lk = _get_lock(_AIGEN_LOCKS, lock_key, MAX_AIGEN_LOCKS)

    # Build the normalized provider request.
    gen_req = aigen_mod.GenRequest(
        mode=req.mode,
        src_b64=src_b64,
        src_w=src_w,
        src_h=src_h,
        prompt=req.prompt,
        neg_prompt=req.neg_prompt,
        denoise=req.denoise,
        steps=req.steps,
        cfg=req.cfg,
        seed=req.seed,
        mask_b64=req.mask_b64,
        controlnet=req.controlnet,
        model=req.model,
        target_w=req.target_w if req.target_w else (src_w or None),
        target_h=req.target_h if req.target_h else (src_h or None),
        work_w=req.work_w,
        work_h=req.work_h,
    )

    with lk:
        try:
            result = p.generate(gen_req)
        except (RuntimeError, ValueError, OSError) as e:
            log.warning("aigen generate failed: provider=%s mode=%s err=%s", req.provider, req.mode, e)
            raise HTTPException(502, f"generation failed: {e}")

    return {
        "out_b64": result.out_b64,
        "out_w": result.out_w,
        "out_h": result.out_h,
        "src_w": src_w,
        "src_h": src_h,
        "seed": result.seed,
        "generation_time_s": result.generation_time_s,
        "model": result.model,
        "provider": result.provider,
        "mode": result.mode,
        "info": result.info,
    }


# ============================================================================
# AI-gen MVP — provider abstraction + budget-guarded upscale (P5)
# ----------------------------------------------------------------------------
# A clean Provider/cost-model layer on top of the legacy aigen v1 WebUI
# providers above. The local-upscale provider needs NO keys and runs out of
# the box; paid providers (e.g. Stability) are gated behind the BudgetGuard
# which defaults to a ZERO budget — so a fresh install can never spend.
# ============================================================================

# Process-lifetime singletons. The registry is stateless (probes env each
# call); the budget guard holds session spend + a daily ledger on disk.
_AIGEN_REGISTRY: "Optional[object]" = None
_AIGEN_BUDGET: "Optional[object]" = None
_AIGEN_MVP_LOCK = threading.Lock()


def _aigen_registry():
    """Lazily build the MVP provider registry (idempotent, thread-safe)."""
    global _AIGEN_REGISTRY
    if _AIGEN_REGISTRY is None:
        with _AIGEN_MVP_LOCK:
            if _AIGEN_REGISTRY is None:
                _AIGEN_REGISTRY = _aigen_default_registry()
    return _AIGEN_REGISTRY


def _aigen_budget():
    """Lazily build the budget guard (reads AIGEN_*_BUDGET_USD env at first use)."""
    global _AIGEN_BUDGET
    if _AIGEN_BUDGET is None:
        with _AIGEN_MVP_LOCK:
            if _AIGEN_BUDGET is None:
                _AIGEN_BUDGET = AigenBudgetGuard()
    return _AIGEN_BUDGET


@app.post("/api/aigen/upscale")
async def api_aigen_mvp_upscale(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    provider: str = Form(default="local_upscale"),
    scale: int = Form(default=2),
    filename: Optional[str] = Form(default=None),
    tile_index: Optional[int] = Form(default=None),
):
    """Budget-guarded upscale via the MVP provider layer.

    Accepts EITHER a multipart ``file`` (an image) OR an asset reference
    (``filename`` + ``tile_index``) resolved from the extracted-tile cache.
    Returns the upscaled image as a ``image/png`` response.

    Budget: the provider's ``estimate_cost_usd`` is checked against the
    BudgetGuard BEFORE any work. A request whose provider cost > 0 under a
    zero budget returns HTTP 402 (clear message), never silently spending.
    The default ``local_upscale`` provider has cost 0 and always works.
    """
    _enforce_body_size(request, MAX_AIGEN_BODY)

    reg = _aigen_registry()
    p = reg.get(provider)
    if p is None:
        raise HTTPException(400, f"unknown provider {provider!r}")
    if not p.available():
        raise HTTPException(
            503, f"provider {provider!r} is not available (missing key/URL/host)"
        )

    # Resolve source PNG bytes from the upload or the asset reference.
    src_png: Optional[bytes] = None
    if file is not None:
        src_png = await file.read()
        if not src_png:
            raise HTTPException(400, "uploaded file is empty")
    elif filename is not None and tile_index is not None:
        if tile_index < 0 or tile_index > MAX_TILE_INDEX:
            raise HTTPException(400, "tile_index out of range")
        prs = safe_data_path(filename)
        if not prs.exists():
            raise HTTPException(404, f"no such file: {filename}")
        manifest = extract_tiles(prs)
        tile = next((t for t in manifest["tiles"] if t["index"] == tile_index), None)
        if not tile:
            raise HTTPException(404, "no such tile")
        tpath = Path(manifest["tiles_dir"]) / tile["filename"]
        if not tpath.exists():
            raise HTTPException(500, "tile png missing on disk")
        src_png = tpath.read_bytes()
    else:
        raise HTTPException(400, "provide a multipart 'file' or filename+tile_index")

    req = AigenImageRequest(image_png=src_png, scale=int(scale))

    # Budget pre-flight: estimate cost, then check. cost>0 under budget 0
    # raises BudgetExceeded -> 402. cost==0 (local) always passes.
    try:
        cost = float(p.estimate_cost_usd(req))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"cost estimation failed: {e}")
    try:
        _aigen_budget().check(cost)
    except AigenBudgetExceeded as e:
        # 402 Payment Required is the precise semantic: the request would
        # cost money the operator hasn't budgeted for.
        raise HTTPException(
            402,
            f"{e} (provider={provider}, cost=${cost:.4f}, "
            f"{e.scope} remaining=${e.remaining:.4f})",
        )

    # Run the upscale.
    try:
        result = p.upscale(req)
    except (ValueError, RuntimeError, OSError) as e:
        raise HTTPException(502, f"upscale failed: {e}")
    except NotImplementedError as e:
        raise HTTPException(501, str(e))

    # Record actual spend only after success (no-op for cost==0).
    if cost > 0:
        _aigen_budget().record(cost)

    return Response(
        content=result.image_png,
        media_type="image/png",
        headers={
            "X-Aigen-Provider": result.provider,
            "X-Aigen-Cost-Usd": f"{result.cost_usd:.4f}",
            "X-Aigen-Out-Width": str(result.width),
            "X-Aigen-Out-Height": str(result.height),
            "X-Aigen-Model": result.model,
        },
    )


# ---------------------------------------------------------------------------- battle params
# Endpoints for editing newserv BattleParamEntry*.dat files. Read flow:
# the UI calls GET /api/battle_param/<variant> which loads the .dat
# from the configured newserv path (or local Booma.Server fixture),
# parses it, and returns JSON. Write flow: POST writes the edited JSON
# back to a stage directory; /api/battle_param/<variant>/deploy then
# copies the staged file to the newserv install.

_BP_DEPLOY_LOCK = threading.Lock()


@app.get("/api/battle_param/config")
def api_battle_param_config():
    """Return the resolved newserv path + variant filenames for the UI.

    Used by the Battle Params perspective to show the user where data is
    being read from and where staged exports will land. The frontend
    falls back to a "configure newserv path" stub when ``newserv_dir``
    is None.
    """
    nsdir = _resolve_newserv_battleparam_dir()
    return {
        "newserv_dir": str(nsdir) if nsdir else None,
        "configured_path": str(NEWSERV_PATH),
        "stage_dir": str(BATTLE_PARAM_STAGE_DIR),
        "variants": list(bp_mod.VALID_VARIANTS),
        "variant_to_filename": dict(bp_mod.VARIANT_TO_FILENAME),
        "file_size": bp_mod.FILE_SIZE,
        "candidates_probed": [
            str(c) for c in NEWSERV_BLUEBURST_CANDIDATES if c is not None
        ],
    }


@app.get("/api/battle_param/slots")
def api_battle_param_slots():
    """Return the slot table sidecar (mob slot index -> human name)."""
    return JSONResponse(content={
        "slots": {f"0x{slot:02X}": name for slot, name in sorted(bp_mod.SLOT_NAMES.items())},
    })


def _battle_param_path(variant: str, *, prefer_stage: bool = False) -> Path:
    """Resolve the path to a BattleParamEntry .dat for a given variant.

    Args:
        variant: one of bp_mod.VALID_VARIANTS.
        prefer_stage: if True, look in the stage dir first (so callers
            see their own edits). The default (False) reads from the
            newserv install — this is what the GET endpoint uses, so
            successive POST -> GET cycles always show the *server's*
            current truth, not the staged copy.
    """
    if variant not in bp_mod.VALID_VARIANTS:
        raise HTTPException(400, f"unknown variant {variant!r}")
    fname = bp_mod.VARIANT_TO_FILENAME[variant]
    if prefer_stage:
        stage = BATTLE_PARAM_STAGE_DIR / fname
        if stage.is_file():
            return stage
    nsdir = _resolve_newserv_battleparam_dir()
    if nsdir is None:
        raise HTTPException(
            404,
            "newserv install not found; set NEWSERV_PATH or "
            "NEWSERV_BLUEBURST_DIR (probed: "
            + ", ".join(str(c) for c in NEWSERV_BLUEBURST_CANDIDATES if c)
            + ")",
        )
    p = nsdir / fname
    if not p.is_file():
        raise HTTPException(404, f"variant file not found at {p}")
    return p


@app.get("/api/battle_param/{variant}")
def api_battle_param_get(variant: str, source: str = "newserv"):
    """Load a BattleParamEntry variant and return its parsed JSON.

    Query param ``source`` selects between:
        - "newserv" (default) — read from the configured newserv install
        - "stage"             — read the staged (edited) copy if any
    """
    prefer_stage = (source == "stage")
    path = _battle_param_path(variant, prefer_stage=prefer_stage)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise HTTPException(500, f"could not read {path}: {e}")
    try:
        bpf = bp_mod.parse(raw, variant=variant)
    except Exception as e:
        raise HTTPException(500, f"parse failed: {e}")
    return JSONResponse(content={
        "source_path": str(path),
        "source_type": source,
        "data": bpf.to_json(),
    })


class BattleParamPostReq(BaseModel):
    data: dict
    # If set, use this filename instead of variant_to_filename[variant].
    # Lets future callers stage non-canonical names if needed; the deploy
    # step still validates the filename.
    output_filename: Optional[str] = None


@app.post("/api/battle_param/{variant}")
def api_battle_param_post(variant: str, req: BattleParamPostReq, request: Request):
    """Serialize edited JSON and write to the stage directory.

    The staged file is written to ``cache/battle_param_export/<filename>``
    where <filename> defaults to the variant's canonical filename (e.g.
    ``BattleParamEntry_on.dat``). Returns the staged path + size + md5
    so the frontend can confirm the bytes that will be deployed.
    """
    if variant not in bp_mod.VALID_VARIANTS:
        raise HTTPException(400, f"unknown variant {variant!r}")
    # Body size sanity (a full JSON dump of one variant is ~4 MB).
    _enforce_body_size(request, 8 * 1024 * 1024)

    try:
        bpf = bp_mod.BattleParamFile.from_json(req.data)
    except Exception as e:
        raise HTTPException(400, f"could not parse JSON body: {e}")
    if not bpf.variant:
        bpf.variant = variant
    if bpf.variant != variant:
        raise HTTPException(
            400,
            f"variant mismatch: URL says {variant}, body says {bpf.variant}",
        )

    try:
        out = bp_mod.serialize(bpf)
    except Exception as e:
        raise HTTPException(400, f"serialize failed: {e}")
    if len(out) != bp_mod.FILE_SIZE:
        raise HTTPException(500, f"serialized {len(out)} bytes, expected {bp_mod.FILE_SIZE}")

    fname = req.output_filename or bp_mod.VARIANT_TO_FILENAME[variant]
    # Refuse path-traversal in the filename. We accept only the bare
    # filename (no directory components) — the stage dir is fixed.
    bare = _validate_bare_filename(fname, label="filename")

    target = BATTLE_PARAM_STAGE_DIR / bare
    try:
        BATTLE_PARAM_STAGE_DIR.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out)
    except OSError as e:
        raise HTTPException(500, f"could not stage to {target}: {e}")

    md5 = hashlib.md5(out).hexdigest()
    return {
        "ok": True,
        "variant": variant,
        "stage_path": str(target),
        "size": len(out),
        "md5": md5,
    }


@app.post("/api/battle_param/{variant}/deploy")
def api_battle_param_deploy(variant: str):
    """Copy the staged variant file into the configured newserv install.

    Creates a ``.pre_deploy_<timestamp>`` backup of the existing file
    if one is present. Refuses to deploy if the staged file is missing
    or the wrong size; refuses if a deploy is already in flight.
    """
    if variant not in bp_mod.VALID_VARIANTS:
        raise HTTPException(400, f"unknown variant {variant!r}")
    fname = bp_mod.VARIANT_TO_FILENAME[variant]
    staged = BATTLE_PARAM_STAGE_DIR / fname
    if not staged.is_file():
        raise HTTPException(404, f"no staged file at {staged}")
    raw = staged.read_bytes()
    if len(raw) != bp_mod.FILE_SIZE:
        raise HTTPException(
            500, f"staged file is {len(raw)} bytes, expected {bp_mod.FILE_SIZE}"
        )

    nsdir = _resolve_newserv_battleparam_dir()
    if nsdir is None:
        raise HTTPException(404, "newserv install not configured (set NEWSERV_PATH)")
    target = nsdir / fname

    if not _BP_DEPLOY_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another battle-param deploy is already running")
    try:
        backup_path: Optional[Path] = None
        if target.is_file():
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = nsdir / f"{fname}.pre_deploy_{ts}"
            try:
                shutil.copy2(target, backup_path)
            except OSError as e:
                raise HTTPException(500, f"could not back up {target}: {e}")
        try:
            # Audit C-7 (2026-05-01): atomic write — tmp + os.replace so a
            # crash mid-write doesn't leave a half-written live-game file.
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, target)
        except OSError as e:
            raise HTTPException(500, f"could not write {target}: {e}")
        return {
            "ok": True,
            "deployed_to": str(target),
            "backup": str(backup_path) if backup_path else None,
            "size": len(raw),
            "md5": hashlib.md5(raw).hexdigest(),
        }
    finally:
        _BP_DEPLOY_LOCK.release()


# ---------------------------------------------------------------------------- mob AI DSL (Tier 1)
# Higher-level authoring layer over BattleParamEntry. The user picks a
# mob (slot + name), the schema gives them named DSL fields with kind-
# aware editors (degrees, seconds, percentages); compile turns those
# named fields back into raw BattleParam bytes.
#
# Contracts:
#   GET  /api/mob_dsl/schemas               full schema table
#   GET  /api/mob_dsl/{mob}                 one mob's schema
#   GET  /api/mob_dsl/presets               list shipped presets
#   GET  /api/mob_dsl/{mob}/preset/{p}      one preset's payload (for prefill)
#   POST /api/mob_dsl/compile               apply patches → BattleParam JSON
#
# /compile takes a stock variant (loaded from disk) + a list of patches
# and returns the resulting BattleParamFile JSON. The caller then
# pipes that through the existing /api/battle_param/{variant} POST →
# /api/battle_param/{variant}/deploy chain to push to newserv.

from formats import mob_dsl as mob_dsl_mod  # noqa: E402


@app.get("/api/mob_dsl/schemas")
def api_mob_dsl_schemas():
    """Return every mob's DSL schema with named field labels + groups."""
    return JSONResponse(content=mob_dsl_mod.all_schemas_json())


@app.get("/api/mob_dsl/presets")
def api_mob_dsl_presets():
    """List shipped presets in data/mob_presets/."""
    return JSONResponse(content={"presets": mob_dsl_mod.list_presets()})


@app.get("/api/mob_dsl/{mob}")
def api_mob_dsl_one_schema(mob: str):
    """Return one mob's schema. ``mob`` may be slot name or ``0xNN``."""
    try:
        return JSONResponse(content=mob_dsl_mod.schema_json(mob))
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/api/mob_dsl/{mob}/preset/{preset}")
def api_mob_dsl_preset_for_mob(mob: str, preset: str):
    """Return a preset filtered to one mob's patches.

    Useful for the UI's "Apply preset" dropdown — given the active mob
    and a chosen preset, pull just the patches that touch this mob.
    """
    try:
        slot = mob_dsl_mod.resolve_mob(mob)
    except ValueError as e:
        raise HTTPException(404, str(e))
    try:
        payload = mob_dsl_mod.load_preset(preset)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError:
        raise HTTPException(404, f"preset {preset!r} not found")
    matching = []
    for p in payload.get("mobs", []):
        try:
            if mob_dsl_mod.resolve_mob(p.get("mob")) == slot:
                matching.append(p)
        except ValueError:
            continue
    return JSONResponse(content={
        "preset": preset,
        "title": payload.get("title", preset),
        "description": payload.get("description", ""),
        "mob": bp_mod.SLOT_NAMES.get(slot, f"slot_{slot:02X}"),
        "patches": matching,
    })


class MobDslCompileReq(BaseModel):
    """POST /api/mob_dsl/compile request body."""
    variant: str  # which BattleParamEntry variant to base on
    mobs: list    # list of mob-patch dicts (see formats/mob_dsl.parse_patches)
    # If true, return the compiled payload AND POST it to the staging
    # dir so the existing /api/battle_param/{variant}/deploy flow can
    # pick it up. Default false — caller composes endpoints explicitly.
    stage: bool = False


@app.post("/api/mob_dsl/compile")
def api_mob_dsl_compile(req: MobDslCompileReq, request: Request):
    """Apply DSL patches to a stock BattleParamEntry; return compiled JSON.

    Pipes through:
        load(variant) → parse → apply_patches(patches) → to_json()

    If ``stage=True`` we also serialize and write to the battle-param
    stage dir so the user can immediately deploy via the existing
    /api/battle_param/{variant}/deploy.
    """
    _enforce_body_size(request, 8 * 1024 * 1024)

    variant = req.variant
    if variant not in bp_mod.VALID_VARIANTS:
        raise HTTPException(400, f"unknown variant {variant!r}")

    # Load the stock baseline.
    path = _battle_param_path(variant, prefer_stage=False)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise HTTPException(500, f"could not read {path}: {e}")
    try:
        base = bp_mod.parse(raw, variant=variant)
    except Exception as e:
        raise HTTPException(500, f"parse failed: {e}")

    # Parse patches and apply.
    try:
        patches = mob_dsl_mod.parse_patches({"mobs": req.mobs})
    except ValueError as e:
        raise HTTPException(400, f"bad patch payload: {e}")
    try:
        compiled = mob_dsl_mod.compile_to_battle_param(base, patches)
    except ValueError as e:
        raise HTTPException(400, f"compile failed: {e}")

    out = {
        "variant": variant,
        "patches_applied": len(patches),
        "data": compiled.to_json(),
    }

    if req.stage:
        try:
            blob = bp_mod.serialize(compiled)
        except Exception as e:
            raise HTTPException(500, f"serialize failed: {e}")
        if len(blob) != bp_mod.FILE_SIZE:
            raise HTTPException(
                500, f"serialized {len(blob)} bytes, expected {bp_mod.FILE_SIZE}"
            )
        fname = bp_mod.VARIANT_TO_FILENAME[variant]
        target = BATTLE_PARAM_STAGE_DIR / fname
        try:
            BATTLE_PARAM_STAGE_DIR.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        except OSError as e:
            raise HTTPException(500, f"could not stage to {target}: {e}")
        out["stage_path"] = str(target)
        out["size"] = len(blob)
        out["md5"] = hashlib.md5(blob).hexdigest()

    return JSONResponse(content=out)


# ============================================================================
# ItemPMT endpoints (BB V4)
# ============================================================================
# The Item Parameter Table is a PRS-compressed binary holding stats for
# every weapon, armor, shield, unit, mag and tool the BB client knows.
# The editor exposes this as a single-document edit surface (no per-slot
# variant like battle_param has). GET /api/itempmt loads and parses;
# POST /api/itempmt stages an edit by re-serializing + PRS-compressing;
# /api/itempmt/deploy copies the staged file to the configured newserv
# install (with a timestamped backup).

_IPMT_DEPLOY_LOCK = threading.Lock()


@app.get("/api/itempmt/config")
def api_itempmt_config():
    """Return the resolved ItemPMT.prs path + staging dir for the UI."""
    src = _resolve_newserv_itempmt()
    return {
        "newserv_dir": str(src.parent) if src else None,
        "configured_path": str(src) if src else None,
        "stage_dir": str(ITEMPMT_STAGE_DIR),
        "candidates_probed": [
            str(c) for c in ITEMPMT_CANDIDATES if c is not None
        ],
        "filename_bb_v4": ITEMPMT_STAGE_FILENAME_BB_V4,
        "filename_legacy": ITEMPMT_STAGE_FILENAME_LEGACY,
    }


@app.get("/api/itempmt")
def api_itempmt_get(source: str = "newserv"):
    """Load the ItemPMT.prs and return its parsed JSON.

    Query param ``source`` selects between:
        - "newserv" (default) — read from the configured newserv install
        - "stage"             — read the staged (edited) copy if any
    """
    if source not in ("newserv", "stage"):
        raise HTTPException(400, f"unknown source {source!r}")
    if source == "stage":
        # Try staged BB-V4 name first, then legacy.
        for fname in (ITEMPMT_STAGE_FILENAME_BB_V4,
                      ITEMPMT_STAGE_FILENAME_LEGACY):
            p = ITEMPMT_STAGE_DIR / fname
            if p.is_file():
                src = p
                break
        else:
            raise HTTPException(404, f"no staged ItemPMT in {ITEMPMT_STAGE_DIR}")
    else:
        src = _resolve_newserv_itempmt()
        if src is None:
            raise HTTPException(
                404,
                "ItemPMT.prs not found; set NEWSERV_PATH or NEWSERV_ITEMPMT_DIR "
                "(probed: "
                + ", ".join(str(c) for c in ITEMPMT_CANDIDATES if c)
                + ")",
            )
    try:
        prs_blob = src.read_bytes()
    except OSError as e:
        raise HTTPException(500, f"could not read {src}: {e}")
    try:
        raw = prs_mod.decompress(prs_blob)
    except Exception as e:
        raise HTTPException(500, f"PRS decompress failed for {src}: {e}")
    try:
        pmt = ipmt_mod.parse_with_meta(raw)
    except Exception as e:
        raise HTTPException(500, f"ItemPMT parse failed: {e}")
    return JSONResponse(content={
        "source_path": str(src),
        "source_type": source,
        "raw_size": len(raw),
        "prs_size": len(prs_blob),
        "data": pmt.to_json(),
    })


class ItemPmtPostReq(BaseModel):
    data: dict
    output_filename: Optional[str] = None  # default: same filename as source


@app.post("/api/itempmt")
def api_itempmt_post(req: ItemPmtPostReq, request: Request):
    """Serialize edited JSON, PRS-compress, write to the staging directory.

    Returns staged path + sizes + md5s so the frontend can confirm the
    bytes that will be deployed.
    """
    # ItemPMT JSON can be ~1.2 MB pretty-printed; allow generous body cap.
    _enforce_body_size(request, 32 * 1024 * 1024)
    try:
        pmt = ipmt_mod.ItemPMTFile.from_json(req.data)
    except Exception as e:
        raise HTTPException(400, f"could not parse JSON body: {e}")
    try:
        raw = ipmt_mod.serialize(pmt)
    except Exception as e:
        raise HTTPException(400, f"serialize failed: {e}")
    try:
        prs_blob = prs_mod.compress(raw)
    except Exception as e:
        raise HTTPException(500, f"PRS compress failed: {e}")

    # Pick filename: caller may override; otherwise mirror what's
    # currently on disk (BB-V4 if newserv canonical, else legacy).
    if req.output_filename:
        fname = _validate_bare_filename(req.output_filename, label="filename")
    else:
        cur_src = _resolve_newserv_itempmt()
        fname = cur_src.name if cur_src else ITEMPMT_STAGE_FILENAME_BB_V4

    target = ITEMPMT_STAGE_DIR / fname
    try:
        ITEMPMT_STAGE_DIR.mkdir(parents=True, exist_ok=True)
        target.write_bytes(prs_blob)
    except OSError as e:
        raise HTTPException(500, f"could not stage to {target}: {e}")

    return {
        "ok": True,
        "stage_path": str(target),
        "raw_size": len(raw),
        "raw_md5": hashlib.md5(raw).hexdigest(),
        "prs_size": len(prs_blob),
        "prs_md5": hashlib.md5(prs_blob).hexdigest(),
    }


@app.post("/api/itempmt/deploy")
def api_itempmt_deploy():
    """Copy the staged ItemPMT.prs into the configured newserv install.

    Picks the staged filename whose name matches the resolved live file;
    if neither matches, prefers BB-V4 canonical. Creates a
    ``.pre_deploy_<timestamp>`` backup of the existing file.
    """
    cur_src = _resolve_newserv_itempmt()
    if cur_src is None:
        raise HTTPException(404, "newserv ItemPMT.prs install not found")
    nsdir = cur_src.parent
    target_name = cur_src.name

    # Locate a matching staged file.
    staged = ITEMPMT_STAGE_DIR / target_name
    if not staged.is_file():
        # Try BB-V4 / legacy fallback.
        for alt in (ITEMPMT_STAGE_FILENAME_BB_V4, ITEMPMT_STAGE_FILENAME_LEGACY):
            cand = ITEMPMT_STAGE_DIR / alt
            if cand.is_file():
                staged = cand
                break
        else:
            raise HTTPException(404, f"no staged file in {ITEMPMT_STAGE_DIR}")

    raw_prs = staged.read_bytes()
    # Sanity: must decompress without error and parse without error.
    try:
        raw = prs_mod.decompress(raw_prs)
        ipmt_mod.parse_with_meta(raw)  # validates structure
    except Exception as e:
        raise HTTPException(400, f"refusing to deploy: staged file invalid: {e}")

    target = nsdir / target_name
    if not _IPMT_DEPLOY_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another ItemPMT deploy is already running")
    try:
        backup_path: Optional[Path] = None
        if target.is_file():
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = nsdir / f"{target_name}.pre_deploy_{ts}"
            try:
                shutil.copy2(target, backup_path)
            except OSError as e:
                raise HTTPException(500, f"could not back up {target}: {e}")
        try:
            # Audit C-7 (2026-05-01): atomic write — tmp + os.replace so a
            # crash mid-write doesn't leave a half-written live-game file.
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(raw_prs)
            os.replace(tmp, target)
        except OSError as e:
            raise HTTPException(500, f"could not write {target}: {e}")
        return {
            "ok": True,
            "deployed_to": str(target),
            "backup": str(backup_path) if backup_path else None,
            "prs_size": len(raw_prs),
            "prs_md5": hashlib.md5(raw_prs).hexdigest(),
        }
    finally:
        _IPMT_DEPLOY_LOCK.release()


# ============================================================================
# Live Mod-Test (2026-04-25)
# ----------------------------------------------------------------------------
# Push an in-progress edit into the running game / running newserv without
# requiring the user to close + relaunch. Three categories of target:
#
#   1. Server-side data (BattleParam / ItemPMT / Mob DSL):
#         - Copy staged file to the configured newserv install (re-uses
#           the existing /api/battle_param/{variant}/deploy + /api/itempmt/deploy
#           plumbing).
#         - Optionally invoke `newserv reload patch-indexes` over the local
#           console pipe (probed at NEWSERV_RELOAD_URL; HTTP control plane
#           is NOT shipped by upstream newserv as of this writing — the
#           reload-detect logic gracefully degrades to "manual reload
#           required" when no auto-reload sidecar is present).
#         - The on-the-wire effect is: next mob spawn picks up the new
#           BattleParam stats; next session login picks up the new
#           ItemPMT.
#
#   2. Client-side textures (Phase 2 — sketched here, not shipped in v1):
#         - Drop a PNG into ``<install>/cache/live_overrides/`` keyed by the
#           BML/inner asset path; the combo ASI's `mod_live_replace` module
#           (NOT in this commit; see ROADMAP) hot-reloads it via D3D9
#           SetTexture redirection.
#         - For v1 we expose the staging side only; the ASI side is a
#           future agent's job.
#
#   3. Geometry (Phase 3 — punted):
#         - Vertex-buffer rewriting; documented as future work.
#
# Frontend integration:
#   static/live_test.js exports `triggerLiveTest(kind, opts)` which calls
#   one of the endpoints below and surfaces a status pip into the calling
#   panel. Each panel adds a "Live Test" button that invokes this.
#
# Endpoints:
#   POST /api/live_test                     — kind="battle_param"|"itempmt"|"mob_dsl"|"texture"
#                                             routes to the right deploy chain.
#   POST /api/live_test/newserv_reload      — probe + reload newserv if a control
#                                             plane is available, else 503 with
#                                             a manual-reload message.
#   GET  /api/live_test/config              — newserv path + reload availability
#                                             surface for the UI status pip.
#   GET  /api/live_test/log                 — recent action log (for status pips).
# ============================================================================

# Cache dir for client-side texture overrides. The combo ASI's
# `mod_live_replace` module watches this dir and applies SetTexture
# redirects when its INI bit is enabled. The ASI also touches
# `_consumer_heartbeat` here every ~5 s while installed, which is how
# /api/live_test/config decides whether to set
# texture_override_consumer_active=true.
LIVE_OVERRIDES_DIR = CACHE_DIR / "live_overrides"
LIVE_OVERRIDES_DIR.mkdir(exist_ok=True)

# Heartbeat staleness threshold. If the ASI's _consumer_heartbeat file
# was written within this many seconds, we consider the consumer alive.
# The ASI touches every ~5 s; 10 s gives one missed-tick of slack.
LIVE_OVERRIDES_HEARTBEAT_STALE_S = 10.0


def _live_overrides_consumer_active() -> bool:
    """Return True iff the ASI consumer's heartbeat is fresh.

    The consumer (mod_live_replace) writes
    ``<live_overrides_dir>/_consumer_heartbeat`` every ~5 seconds while
    installed. We check the file's mtime against
    ``LIVE_OVERRIDES_HEARTBEAT_STALE_S``; any error (file missing, OS
    error, time arithmetic edge case) is treated as "not alive".

    The ``_consumer_alive`` filename is also accepted as an alternate
    sentinel — the spec calls out either name as acceptable, so we
    check both. Whichever was written most recently wins.
    """
    try:
        for name in ("_consumer_heartbeat", "_consumer_alive"):
            sentinel = LIVE_OVERRIDES_DIR / name
            if not sentinel.is_file():
                continue
            try:
                mtime = sentinel.stat().st_mtime
            except OSError:
                continue
            age = time.time() - mtime
            # Negative age (clock skew, file written in the future)
            # also counts as fresh — the ASI is clearly active.
            if age <= LIVE_OVERRIDES_HEARTBEAT_STALE_S:
                return True
        return False
    except Exception:
        return False

# In-memory log of recent live-test actions. UI fetches the tail to show
# the per-panel "last 3 actions" status. Cap keeps it bounded across long
# editor sessions.
_LIVE_TEST_LOG_CAP = 64
_LIVE_TEST_LOG: "list[dict]" = []
_LIVE_TEST_LOG_LOCK = threading.Lock()


def _live_test_log(action: dict) -> None:
    """Append an entry to the live-test action log (thread-safe, bounded)."""
    with _LIVE_TEST_LOG_LOCK:
        _LIVE_TEST_LOG.append(dict(action))
        if len(_LIVE_TEST_LOG) > _LIVE_TEST_LOG_CAP:
            del _LIVE_TEST_LOG[:-_LIVE_TEST_LOG_CAP]


def _live_test_log_tail(limit: int = 16, panel: Optional[str] = None) -> list[dict]:
    """Return the most recent ``limit`` entries (newest last)."""
    with _LIVE_TEST_LOG_LOCK:
        items = list(_LIVE_TEST_LOG)
    if panel:
        items = [it for it in items if it.get("panel") == panel]
    return items[-limit:]


# Optional: HTTP control-plane URL for newserv. Stock newserv has no such
# endpoint — but a thin sidecar (e.g. a 30-line FastAPI shim that runs
# `newserv reload patch-indexes` on POST) is feasible. Set NEWSERV_RELOAD_URL
# to point at it; otherwise reload returns 503 with an actionable message.
NEWSERV_RELOAD_URL = os.environ.get("NEWSERV_RELOAD_URL") or ""
# Short timeout — the reload should be near-instant, and we don't want to
# block the editor's request thread on a misconfigured sidecar.
NEWSERV_RELOAD_TIMEOUT = 5.0


def _newserv_reload_available() -> bool:
    """Return True iff NEWSERV_RELOAD_URL is set.

    No network probe here — we don't want to spam the sidecar on every
    config GET. The probe happens lazily on /api/live_test/newserv_reload.
    """
    return bool(NEWSERV_RELOAD_URL)


def _try_newserv_reload() -> tuple[bool, str]:
    """POST to NEWSERV_RELOAD_URL to fire `reload patch-indexes`.

    Returns ``(ok, message)``. ``ok=False`` on any error — the live-test
    flow then surfaces "deployed; manual reload required".
    """
    if not NEWSERV_RELOAD_URL:
        return False, "NEWSERV_RELOAD_URL not configured"
    import urllib.request
    import urllib.error
    req = urllib.request.Request(
        NEWSERV_RELOAD_URL,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=NEWSERV_RELOAD_TIMEOUT) as resp:
            body = resp.read(2048).decode("utf-8", errors="replace")
            if 200 <= resp.status < 300:
                return True, f"reload OK ({resp.status}): {body[:120]}"
            return False, f"reload HTTP {resp.status}: {body[:120]}"
    except urllib.error.URLError as e:
        return False, f"reload unreachable: {e.reason}"
    except Exception as e:  # pragma: no cover — generic safety net
        return False, f"reload failed: {e}"


# Body schemas
class LiveTestReq(BaseModel):
    kind: str                                # battle_param | itempmt | mob_dsl | texture
    # Server-side targets (battle_param + mob_dsl share variant; itempmt
    # has no variant). Both are validated per-kind below.
    variant: Optional[str] = None
    # Optional explicit path override; default uses the staging dir.
    staged_path: Optional[str] = None
    # Caller hint for action-log grouping (panel id), e.g. "battle-param"
    # or "mob-dsl". Ignored if missing.
    panel: Optional[str] = None
    # Texture-mode fields. ``asset_path`` is the BML#inner reference,
    # ``png_b64`` is the OVERRIDE PNG (the user's painted result).
    asset_path: Optional[str] = None         # e.g. "bm_ene_bm9_s_mericarol.bml#mericarol_body"
    png_b64: Optional[str] = None            # raw PNG bytes (override)
    # Optional original-texture PNG bytes. When supplied, the server
    # decodes it once, hashes the raw RGBA pixels, and writes the digest
    # as ``match.src_rgba_md5`` into the .replace sidecar. The ASI
    # consumer (mod_live_replace) keys SetTexture redirects on this
    # fingerprint — without it, the override file is staged but inert.
    # The editor's texture-paint panel has the original PNG in hand
    # already (it loaded the surface to paint on), so adding this is
    # a one-liner on the client side.
    src_png_b64: Optional[str] = None
    # Whether to attempt newserv reload after deploy. Default true; UI can
    # uncheck to stage-only.
    attempt_reload: bool = True


# Request body cap for live-test (32 MB so a 4K texture override fits with
# room for JSON inflation).
MAX_LIVE_TEST_BODY = 32 * 1024 * 1024


@app.get("/api/live_test/config")
def api_live_test_config():
    """Return the live-test configuration the UI needs to render status."""
    nsdir_bp = _resolve_newserv_battleparam_dir()
    ipmt_path = _resolve_newserv_itempmt()
    consumer_alive = _live_overrides_consumer_active()
    return {
        "newserv_battleparam_dir": str(nsdir_bp) if nsdir_bp else None,
        "newserv_itempmt_path": str(ipmt_path) if ipmt_path else None,
        "newserv_reload_url": NEWSERV_RELOAD_URL or None,
        "newserv_reload_available": _newserv_reload_available(),
        "live_overrides_dir": str(LIVE_OVERRIDES_DIR),
        "kinds_supported": ["battle_param", "itempmt", "mob_dsl", "texture"],
        # True iff the ASI consumer (mod_live_replace) has touched its
        # heartbeat file within LIVE_OVERRIDES_HEARTBEAT_STALE_S seconds.
        # When False, the texture endpoint is staging-only — files are
        # written but the running game won't pick them up until the ASI
        # is loaded (and its [live_replace] enabled=1 in the INI).
        "texture_override_consumer_active": consumer_alive,
    }


@app.get("/api/live_test/log")
def api_live_test_log(limit: int = 16, panel: Optional[str] = None):
    """Return the tail of the live-test action log (for status pips)."""
    if limit <= 0 or limit > _LIVE_TEST_LOG_CAP:
        limit = 16
    return {"entries": _live_test_log_tail(limit, panel)}


def _live_test_battle_param(req: "LiveTestReq") -> dict:
    """Live-test a staged BattleParamEntry*.dat: deploy."""
    if not req.variant or req.variant not in bp_mod.VALID_VARIANTS:
        raise HTTPException(400, f"battle_param requires valid variant; got {req.variant!r}")
    fname = bp_mod.VARIANT_TO_FILENAME[req.variant]
    if req.staged_path:
        staged = Path(req.staged_path).resolve()
        try:
            staged.relative_to(BATTLE_PARAM_STAGE_DIR.resolve())
        except ValueError:
            raise HTTPException(400, "staged_path escapes battle_param_export dir")
    else:
        staged = BATTLE_PARAM_STAGE_DIR / fname
    if not staged.is_file():
        raise HTTPException(404, f"no staged file at {staged}")
    raw = staged.read_bytes()
    if len(raw) != bp_mod.FILE_SIZE:
        raise HTTPException(500, f"staged file is {len(raw)} bytes, expected {bp_mod.FILE_SIZE}")
    nsdir = _resolve_newserv_battleparam_dir()
    if nsdir is None:
        raise HTTPException(404, "newserv install not configured (set NEWSERV_PATH)")
    target = nsdir / fname

    if not _BP_DEPLOY_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another battle-param deploy is already running")
    try:
        backup_path: Optional[Path] = None
        if target.is_file():
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = nsdir / f"{fname}.pre_livetest_{ts}"
            try:
                shutil.copy2(target, backup_path)
            except OSError as e:
                raise HTTPException(500, f"could not back up {target}: {e}")
        try:
            # Audit C-7 (2026-05-01): atomic write — tmp + os.replace so a
            # crash mid-write doesn't leave a half-written live-game file.
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, target)
        except OSError as e:
            raise HTTPException(500, f"could not write {target}: {e}")
    finally:
        _BP_DEPLOY_LOCK.release()
    return {
        "deployed_to": str(target),
        "backup": str(backup_path) if backup_path else None,
        "size": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
    }


def _live_test_itempmt(req: "LiveTestReq") -> dict:
    """Live-test a staged ItemPMT.prs: deploy."""
    cur_src = _resolve_newserv_itempmt()
    if cur_src is None:
        raise HTTPException(404, "newserv ItemPMT.prs install not found")
    nsdir = cur_src.parent
    target_name = cur_src.name
    if req.staged_path:
        staged = Path(req.staged_path).resolve()
        try:
            staged.relative_to(ITEMPMT_STAGE_DIR.resolve())
        except ValueError:
            raise HTTPException(400, "staged_path escapes itempmt_export dir")
    else:
        staged = ITEMPMT_STAGE_DIR / target_name
        if not staged.is_file():
            for alt in (ITEMPMT_STAGE_FILENAME_BB_V4, ITEMPMT_STAGE_FILENAME_LEGACY):
                cand = ITEMPMT_STAGE_DIR / alt
                if cand.is_file():
                    staged = cand
                    break
    if not staged.is_file():
        raise HTTPException(404, f"no staged file in {ITEMPMT_STAGE_DIR}")
    raw_prs = staged.read_bytes()
    # Sanity: must decompress without error.
    try:
        raw = prs_mod.decompress(raw_prs)
        ipmt_mod.parse_with_meta(raw)
    except Exception as e:
        raise HTTPException(400, f"refusing to deploy: staged file invalid: {e}")
    target = nsdir / target_name

    if not _IPMT_DEPLOY_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another ItemPMT deploy is already running")
    try:
        backup_path: Optional[Path] = None
        if target.is_file():
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = nsdir / f"{target_name}.pre_livetest_{ts}"
            try:
                shutil.copy2(target, backup_path)
            except OSError as e:
                raise HTTPException(500, f"could not back up {target}: {e}")
        try:
            # Audit C-7 (2026-05-01): atomic write — tmp + os.replace so a
            # crash mid-write doesn't leave a half-written live-game file.
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(raw_prs)
            os.replace(tmp, target)
        except OSError as e:
            raise HTTPException(500, f"could not write {target}: {e}")
    finally:
        _IPMT_DEPLOY_LOCK.release()
    return {
        "deployed_to": str(target),
        "backup": str(backup_path) if backup_path else None,
        "prs_size": len(raw_prs),
        "prs_md5": hashlib.md5(raw_prs).hexdigest(),
    }


def _live_test_texture(req: "LiveTestReq") -> dict:
    """Phase 2: drop a PNG override into the live_overrides cache.

    The combo ASI's ``mod_live_replace`` module reads this directory,
    hooks IDirect3DDevice9::SetTexture, and swaps the override texture
    in place of any IDirect3DTexture9* whose decoded RGBA matches the
    ``match.src_rgba_md5`` in the .replace sidecar.

    For the swap to fire, the request MUST include ``src_png_b64`` —
    the original asset's PNG (which the editor has on hand from the
    paint-load step). Without it, the override is staged but inert
    (the ASI logs a warning per missing-fingerprint .replace and
    skips it).

    The ``texture_override_consumer_active`` flag from
    /api/live_test/config tells the UI whether the ASI is alive.
    """
    if not req.asset_path:
        raise HTTPException(400, "texture live-test requires asset_path")
    if not req.png_b64:
        raise HTTPException(400, "texture live-test requires png_b64")
    raw = _decode_b64(req.png_b64, ctx="png_b64")
    if not raw.startswith(PNG_MAGIC):
        raise HTTPException(400, "png_b64 is not a valid PNG")
    # Optional source-PNG fingerprint. If supplied, the ASI will use it
    # to identify the running game's original IDirect3DTexture9* and
    # swap to our override at SetTexture time. The fingerprint is the
    # MD5 of the decoded RGBA bytes (top-down, R8 G8 B8 A8 byte order).
    match_block: Optional[dict] = None
    if req.src_png_b64:
        src_png = _decode_b64(req.src_png_b64, ctx="src_png_b64")
        if not src_png.startswith(PNG_MAGIC):
            raise HTTPException(400, "src_png_b64 is not a valid PNG")
        try:
            from PIL import Image  # type: ignore
            import io as _io
            with Image.open(_io.BytesIO(src_png)) as im:
                rgba_im = im.convert("RGBA")
                w, h = rgba_im.size
                rgba_bytes = rgba_im.tobytes()  # top-down, RGBA byte order
        except Exception as e:
            raise HTTPException(400, f"could not decode src_png_b64: {e}")
        match_block = {
            "width": int(w),
            "height": int(h),
            "src_rgba_md5": hashlib.md5(rgba_bytes).hexdigest(),
        }
    # Sanitize: replace any path separator with __ so the override key is
    # filename-safe on Windows. Preserve the asset_path verbatim in the
    # written sidecar JSON so the ASI consumer can route correctly.
    safe_basename = re.sub(r"[^A-Za-z0-9._-]+", "__", req.asset_path)
    if not safe_basename:
        raise HTTPException(400, "asset_path resolves to empty key")
    out_png = LIVE_OVERRIDES_DIR / f"{safe_basename}.png"
    out_meta = LIVE_OVERRIDES_DIR / f"{safe_basename}.replace"
    try:
        out_png.write_bytes(raw)
        # The .replace sidecar tells the ASI which asset to redirect.
        # When match_block is present the ASI's SetTexture hook actively
        # fires; absent, the .replace lands on disk but stays inert.
        meta: dict = {
            "asset_path": req.asset_path,
            "png_basename": out_png.name,
            "png_path": str(out_png),
            "ts": time.time(),
            "md5": hashlib.md5(raw).hexdigest(),
        }
        if match_block:
            meta["match"] = match_block
        out_meta.write_text(json.dumps(meta, indent=2))
    except OSError as e:
        raise HTTPException(500, f"could not write override: {e}")
    consumer_alive = _live_overrides_consumer_active()
    out: dict = {
        "override_png": str(out_png),
        "override_meta": str(out_meta),
        "asset_path": req.asset_path,
        "size": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
        "consumer_active": consumer_alive,
        "has_fingerprint": match_block is not None,
    }
    if not consumer_alive:
        # ASI consumer isn't running (or hasn't checked in within the
        # heartbeat stale window). The .replace + .png are on disk; the
        # next time the ASI polls, it'll pick them up. Surface that to
        # the UI so the live-test pip can show "staged" instead of "live".
        out["warning"] = ("texture override staged; mod_live_replace "
                          "consumer heartbeat is stale — start PSOBB "
                          "with [live_replace] enabled=1 to apply")
    elif match_block is None:
        # Consumer alive but no match block — the override won't fire.
        # Surface this so the UI doesn't pretend the swap is live.
        out["warning"] = ("texture override on disk but no src_png_b64 "
                          "supplied — ASI will skip swap. Include the "
                          "original PNG bytes to enable matching.")
    return out


@app.post("/api/live_test")
def api_live_test(req: LiveTestReq, request: Request):
    """Push a staged edit into the running game / newserv.

    Routes to the right deploy chain based on ``kind``. Returns
    ``{ok, kind, deployed: {...}, reload: {ok, message}, requires_manual_reload}``.
    Server-side kinds (battle_param/itempmt/mob_dsl) ALWAYS deploy first;
    the optional newserv reload is best-effort and never fails the call.

    For ``mob_dsl`` we only support the "compile already happened, now push"
    flow — caller is expected to invoke /api/mob_dsl/compile with stage=true
    first, which produces a battle_param staged file. We then deploy that
    via the same battle_param flow.
    """
    _enforce_body_size(request, MAX_LIVE_TEST_BODY)
    started_ts = time.time()
    kind = (req.kind or "").strip()
    deployed: dict = {}
    if kind == "battle_param" or kind == "mob_dsl":
        # mob_dsl rides the same plumbing — caller staged a compiled
        # BattleParam; we just deploy + (try) reload.
        deployed = _live_test_battle_param(req)
        category = "server"
    elif kind == "itempmt":
        deployed = _live_test_itempmt(req)
        category = "server"
    elif kind == "texture":
        deployed = _live_test_texture(req)
        category = "client"
    else:
        raise HTTPException(400, f"unknown kind {kind!r}")

    reload_result: dict = {"attempted": False, "ok": False, "message": ""}
    requires_manual = False
    if category == "server":
        if req.attempt_reload and _newserv_reload_available():
            ok, msg = _try_newserv_reload()
            reload_result = {"attempted": True, "ok": ok, "message": msg}
            requires_manual = not ok
        else:
            requires_manual = True
            reload_result = {
                "attempted": False,
                "ok": False,
                "message": ("NEWSERV_RELOAD_URL not configured; run "
                            "`reload patch-indexes` in the newserv console."),
            }

    elapsed_ms = int((time.time() - started_ts) * 1000)
    log_entry = {
        "ts": started_ts,
        "kind": kind,
        "panel": req.panel or kind,
        "category": category,
        "elapsed_ms": elapsed_ms,
        "deployed": deployed,
        "reload": reload_result,
        "requires_manual_reload": requires_manual,
        "ok": True,
    }
    _live_test_log(log_entry)
    return {
        "ok": True,
        "kind": kind,
        "category": category,
        "deployed": deployed,
        "reload": reload_result,
        "requires_manual_reload": requires_manual,
        "elapsed_ms": elapsed_ms,
    }


@app.post("/api/live_test/newserv_reload")
def api_live_test_newserv_reload():
    """Probe + invoke the newserv reload sidecar.

    Returns ``{ok, message}`` on success, or 503 if the sidecar isn't
    configured / reachable so the UI can render "manual reload required".
    """
    if not _newserv_reload_available():
        raise HTTPException(503, ("NEWSERV_RELOAD_URL not configured; "
                                  "manual `reload patch-indexes` required"))
    ok, msg = _try_newserv_reload()
    if not ok:
        raise HTTPException(503, f"newserv reload failed: {msg}")
    return {"ok": True, "message": msg}


# ============================================================================
# Texture Paint MVP (2026-04-25)
# ----------------------------------------------------------------------------
# Persists painted PNGs to ``cache/painted_textures/`` and re-packs the host
# BML/AFS through the existing /api/build_bml + /api/build_afs writers.
#
# The frontend (static/paint_panel.js) does the actual UV-aware brush work
# directly in WebGL — these endpoints only persist the resulting PNG and
# rebuild archives. Strictly server-driven re-painting (e.g. headless
# regression tests) can use the helpers in formats/paint.py.
#
# Endpoints:
#   POST /api/paint/save           — write a painted PNG to the cache.
#   POST /api/paint/build_archive  — rebuild the host BML/AFS with painted entries.
#   POST /api/paint/deploy         — wrap /api/deploy/<archive>.
#   GET  /api/paint/active         — list every painted texture currently cached.
#   v5 (2026-04-25): layer endpoints
#   POST /api/paint/layer/save     — write one layer (or its mask) into the stack
#   POST /api/paint/layer/delete   — drop a layer (or its mask) from the stack
#   POST /api/paint/manifest       — replace the manifest after reorder/merge
#   GET  /api/paint/load           — return the layer stack for a texture
# ============================================================================

PAINTED_TEX_DIR = CACHE_DIR / "painted_textures"
PAINT_EXPORT_DIR = CACHE_DIR / "exports"
PAINTED_TEX_DIR.mkdir(exist_ok=True)
PAINT_EXPORT_DIR.mkdir(exist_ok=True)

MAX_PAINT_SAVE_BODY = 32 * 1024 * 1024  # 32 MB — single 4096x4096 RGBA PNG fits

from formats import paint as _paint_mod  # noqa: E402  (kept low to group with paint endpoints)


class PaintSaveReq(BaseModel):
    model_path: str          # e.g. "bm_ene_bm9_s_mericarol.bml"
    inner: str               # e.g. "bm_ene_bm9_s_mericarol.nj.xvm"
    png_b64: str
    source_md5: Optional[str] = None  # optional concurrency check


@app.post("/api/paint/save")
def api_paint_save(req: PaintSaveReq, request: Request):
    """Persist a painted PNG to ``cache/painted_textures/<safe>.png``.

    The frontend supplies the live-edited RGBA buffer as a base64-encoded
    PNG. We validate the magic, normalize to 8-bit RGBA via PIL, and
    write atomically (tmp + os.replace).

    Returns ``{ok, cache_path, md5, width, height}``. Does NOT touch the
    BML/AFS — that's the build-archive step.
    """
    _enforce_body_size(request, MAX_PAINT_SAVE_BODY)
    if not req.model_path or not req.inner:
        raise HTTPException(400, "model_path and inner are required")
    safe_name = _paint_mod.safe_painted_basename(req.model_path, req.inner)
    out_path = PAINTED_TEX_DIR / safe_name
    # Re-validate that the resolved path stays inside PAINTED_TEX_DIR even
    # after :func:`safe_painted_basename` strips separators.
    try:
        out_path.resolve().relative_to(PAINTED_TEX_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "path escapes painted_textures dir")

    raw = _decode_b64(req.png_b64, ctx="png_b64")
    if len(raw) < 8 or raw[:8] != PNG_MAGIC:
        raise HTTPException(400, "png_b64: not a PNG (magic bytes missing)")
    # Round-trip via PIL to normalize and to validate decodability.
    try:
        im = Image.open(BytesIO(raw))
        im.load()
        if im.mode != "RGBA":
            im = im.convert("RGBA")
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"PNG decode failed: {e}")
    w, h = im.size
    out_buf = BytesIO()
    im.save(out_buf, format="PNG")
    final_bytes = out_buf.getvalue()

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(final_bytes)
    os.replace(tmp_path, out_path)
    md5 = hashlib.md5(final_bytes).hexdigest()
    log.info(
        "paint_save %s -> %s (%dx%d, %d bytes, md5=%s)",
        f"{req.model_path}#{req.inner}", out_path, w, h, len(final_bytes), md5,
    )
    return {
        "ok": True,
        "cache_path": str(out_path),
        "basename": safe_name,
        "md5": md5,
        "width": w,
        "height": h,
        "size": len(final_bytes),
    }


class PaintBuildArchiveReq(BaseModel):
    model_path: str  # e.g. "bm_ene_bm9_s_mericarol.bml"


def _list_painted_for_model(model_path: str) -> list[Path]:
    """Return every painted PNG in the cache that targets ``model_path``.

    Matches by ``<safe(model_path)>__`` prefix on the cache filename.
    """
    if not PAINTED_TEX_DIR.exists():
        return []
    base_safe = (
        model_path.replace("/", "_").replace("\\", "_").replace("#", "_")
    )
    prefix = f"{base_safe}__"
    return sorted(p for p in PAINTED_TEX_DIR.iterdir()
                  if p.is_file() and p.name.startswith(prefix))


def _png_to_xvm_bytes(png_path: Path, source_xvm: bytes) -> bytes:
    """Replace tile 0 of ``source_xvm`` with the PNG at ``png_path``.

    Uses xvr_codec to extract the source's tiles, swaps tile-0's PNG for
    the painted version, then rebuilds. The painted PNG MUST be at the
    same dimensions as the source's tile 0 (otherwise xvr_codec rejects).

    Returns the rebuilt XVM bytes. Raises HTTPException on any failure.
    """
    if not XVR_CODEC.exists():
        raise HTTPException(500, f"xvr_codec missing: {XVR_CODEC}")
    # Materialize the source XVM into a temp dir and extract its tiles.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="paint_xvm_") as td:
        td_path = Path(td)
        src_xvm = td_path / "src.xvm"
        src_xvm.write_bytes(source_xvm)
        extract_dir = td_path / "tiles"
        extract_dir.mkdir()
        rc = subprocess.run(
            [sys.executable, str(XVR_CODEC), "extract",
             str(src_xvm), str(extract_dir)],
            capture_output=True, timeout=TIMEOUT_BML_PRS, text=True,
        )
        if rc.returncode != 0:
            raise HTTPException(500, f"xvr_codec extract failed: {rc.stderr.strip()[:200]}")
        # Replace tile 0 with our painted PNG. xvr_codec's extract names
        # tiles by their texture index — we copy the painted PNG over the
        # first tile's PNG.
        png_files = sorted(extract_dir.glob("*.png"))
        if not png_files:
            raise HTTPException(500, "xvr_codec extract produced no PNGs")
        # Use the painted PNG bytes verbatim; xvr_codec re-encodes from PNG.
        target = png_files[0]
        target.write_bytes(png_path.read_bytes())
        out_xvm = td_path / "out.xvm"
        rc = subprocess.run(
            [sys.executable, str(XVR_CODEC), "rebuild",
             str(extract_dir), str(out_xvm)],
            capture_output=True, timeout=TIMEOUT_BML_PRS, text=True,
        )
        if rc.returncode != 0:
            raise HTTPException(500, f"xvr_codec rebuild failed: {rc.stderr.strip()[:200]}")
        return out_xvm.read_bytes()


@app.post("/api/paint/build_archive")
def api_paint_build_archive(req: PaintBuildArchiveReq, request: Request):
    """Rebuild the host archive with painted textures spliced in.

    For BML hosts:
        1. Read every painted PNG that targets ``model_path``.
        2. For each, locate the matching inner XVM and splice tile 0
           with the painted PNG (via xvr_codec).
        3. Repack the BML with all unmodified inners + the painted XVMs.

    For AFS hosts: same idea, but inner ``NNNN_*.xvm`` entries are the
    unit of replacement.

    Returns ``{ok, archive_path, size, md5, painted_count}``.
    """
    _enforce_body_size(request, MAX_BUILD_DEPLOY_BODY)
    name = _safe_archive_name(req.model_path)
    is_bml = name.lower().endswith(".bml")
    is_afs = name.lower().endswith(".afs")
    if not (is_bml or is_afs):
        raise HTTPException(400, f"unsupported archive type: {name}")

    painted = _list_painted_for_model(name)
    if not painted:
        raise HTTPException(404, f"no painted textures cached for {name!r}")

    # Resolve the source archive in DATA_DIR / LIVE_DATA_DIR.
    src_archive = _resolve_base_path(name)

    if is_bml:
        # parse_bml_for_pack + parse_bml_pack_meta gives byte-exact
        # round-trip for the entries we don't touch (PRS payloads carried
        # forward verbatim — no decompress + recompress cycle).
        from formats.bml import parse_bml_for_pack, parse_bml_pack_meta
        try:
            blob = src_archive.read_bytes()
            pack_entries = parse_bml_for_pack(blob)
            meta = parse_bml_pack_meta(blob)
        except (OSError, ValueError) as e:
            raise HTTPException(500, f"BML parse failed: {e}")

        # Painted basename pattern: <safe(model)>__<inner>.png. Inner
        # names for textures look like "<entry>.nj.xvm".
        prefix = (
            name.replace("/", "_").replace("\\", "_").replace("#", "_") + "__"
        )
        painted_by_inner: dict[str, Path] = {}
        for p in painted:
            tail = p.name[len(prefix):]
            inner_name = tail
            if inner_name.lower().endswith(".png"):
                inner_name = inner_name[: -len(".png")]
            painted_by_inner[inner_name] = p

        painted_count = 0
        for pe in pack_entries:
            inner_data_xvm_name = pe.name + ".xvm"
            if inner_data_xvm_name in painted_by_inner and pe.texture_data:
                # Splice the painted PNG into the existing XVM. We need
                # raw (uncompressed) XVM bytes to feed xvr_codec, so
                # decompress the stored PRS first when needed.
                if pe.texture_is_compressed:
                    try:
                        raw_xvm = prs_mod.decompress(pe.texture_data)
                    except Exception as e:
                        raise HTTPException(500, f"texture PRS decompress failed for {pe.name!r}: {e}")
                else:
                    raw_xvm = pe.texture_data
                new_xvm = _png_to_xvm_bytes(painted_by_inner[inner_data_xvm_name], raw_xvm)
                # Store back as raw + flag for the packer to PRS it on write.
                pe.texture_data = new_xvm
                pe.texture_is_compressed = False
                pe.texture_decompressed_size = len(new_xvm)
                painted_count += 1

        try:
            out = pack_bml(
                pack_entries,
                compression=meta["compression"],
                file_alignment=meta["file_alignment"],
                has_textures_override=meta["has_textures"],
            )
        except ValueError as e:
            raise HTTPException(500, f"pack_bml failed: {e}")

        out_path = BML_EXPORT_DIR / name
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_bytes(out)
        os.replace(tmp_path, out_path)
        md5 = hashlib.md5(out).hexdigest()
        log.info("paint_build_archive BML %s -> %s (%d bytes, %d painted, md5=%s)",
                 name, out_path, len(out), painted_count, md5)
        return {
            "ok": True,
            "archive_path": str(out_path),
            "archive_name": name,
            "size": len(out),
            "md5": md5,
            "painted_count": painted_count,
            "kind": "bml",
        }

    # AFS host.
    try:
        blob = src_archive.read_bytes()
        afs_blobs = parse_afs(blob)
    except (OSError, ValueError) as e:
        raise HTTPException(500, f"AFS parse failed: {e}")

    # Painted basename pattern: <safe(model)>__<inner>.png. For AFS,
    # the inner is "NNNN_<basename>.xvm" or "NNNN.xvm".
    prefix = (
        name.replace("/", "_").replace("\\", "_").replace("#", "_") + "__"
    )
    painted_by_inner: dict[str, Path] = {}
    for p in painted:
        tail = p.name[len(prefix):]
        inner_name = tail
        if inner_name.lower().endswith(".png"):
            inner_name = inner_name[: -len(".png")]
        painted_by_inner[inner_name] = p

    painted_count = 0
    out_blobs: list[bytes] = []
    for i, raw in enumerate(afs_blobs):
        # Match inner name by index — the painter saved with whichever
        # inner name was passed; we accept "NNNN_..." and bare "NNNN".
        candidate_keys = [k for k in painted_by_inner
                          if k.startswith(f"{i:04d}") or k == str(i)]
        if candidate_keys:
            # First-match wins — there should only ever be one per inner.
            key = candidate_keys[0]
            new_raw = _png_to_xvm_bytes(painted_by_inner[key], raw)
            out_blobs.append(new_raw)
            painted_count += 1
        else:
            out_blobs.append(raw)

    try:
        out = write_afs(out_blobs)
    except ValueError as e:
        raise HTTPException(500, f"write_afs failed: {e}")

    out_path = AFS_EXPORT_DIR / name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_bytes(out)
    os.replace(tmp_path, out_path)
    md5 = hashlib.md5(out).hexdigest()
    log.info("paint_build_archive AFS %s -> %s (%d bytes, %d painted, md5=%s)",
             name, out_path, len(out), painted_count, md5)
    return {
        "ok": True,
        "archive_path": str(out_path),
        "archive_name": name,
        "size": len(out),
        "md5": md5,
        "painted_count": painted_count,
        "kind": "afs",
    }


class PaintDeployReq(BaseModel):
    archive_name: str
    create_backup: bool = True


@app.post("/api/paint/deploy")
def api_paint_deploy(req: PaintDeployReq, request: Request):
    """Wrap /api/deploy/<archive_name> after a build_archive call.

    Re-uses the existing deploy code path — same backup contract,
    same lock. Frontend can equivalently call /api/deploy/<archive>
    directly; this endpoint exists so the paint panel can keep its
    URL surface contained.
    """
    _enforce_body_size(request, MAX_BUILD_DEPLOY_BODY)
    return api_deploy_archive(
        req.archive_name, request, DeployArchiveReq(create_backup=req.create_backup)
    )


# ============================================================================
# v5 layer-stack endpoints (2026-04-25).
#
# The MVP saved a single composited PNG per painted texture. v5 adds a layer
# system: each painted texture now lives in its OWN subdirectory holding
# a manifest.json + per-layer <idx>.png + optional <idx>_mask.png. The
# composited PNG is still served (legacy archive-build path uses it) but
# is recomputed from the layer stack at save time.
#
# Backward compatibility: when the layer dir doesn't exist for a given
# (model, inner), we fall back to the flat ``<safe>.png`` from the MVP.
# The first layer-save call auto-converts: read the flat PNG -> seed
# layer 0 -> write the manifest.
# ============================================================================
PAINTED_LAYER_ROOT = PAINTED_TEX_DIR  # subdirs: PAINTED_LAYER_ROOT / <safe>


def _layer_dir_for(model_path: str, inner: str) -> Path:
    """Resolve the per-texture layer directory.

    Returns the path even if it doesn't exist yet — the caller decides
    whether to create + populate it. The directory name uses the same
    safe-basename rules as the flat-PNG cache (no separators, no #, no ..).
    """
    name = _paint_mod.safe_painted_dirname(model_path, inner)
    out = PAINTED_LAYER_ROOT / name
    # Re-validate post-resolve so any traversal is rejected.
    try:
        out.resolve().relative_to(PAINTED_LAYER_ROOT.resolve())
    except ValueError:
        raise HTTPException(400, "path escapes painted_textures dir")
    return out


def _read_manifest(layer_dir: Path) -> Optional[dict]:
    """Load + validate a manifest.json from a layer directory.

    Returns None if the file is missing or malformed (caller reseeds
    from scratch). Validation errors raise HTTPException to surface in
    the API response.
    """
    mf = layer_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
        return _paint_mod.validate_manifest(data)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"manifest read failed: {e}")


def _write_manifest(layer_dir: Path, manifest: dict) -> None:
    """Atomically write a manifest.json into a layer directory."""
    layer_dir.mkdir(parents=True, exist_ok=True)
    norm = _paint_mod.validate_manifest(manifest)
    mf = layer_dir / "manifest.json"
    tmp = mf.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(norm, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, mf)


def _migrate_flat_png_to_layer_dir(model_path: str, inner: str) -> Optional[dict]:
    """One-shot: convert a legacy flat ``<safe>.png`` to a layer stack.

    If a flat PNG exists for this (model, inner) but no layer
    directory does, copy the PNG to ``<dir>/0.png`` and write a
    minimal manifest. Returns the new manifest dict (so the caller
    can use it) or None if no flat PNG exists.
    """
    flat_name = _paint_mod.safe_painted_basename(model_path, inner)
    flat_path = PAINTED_TEX_DIR / flat_name
    if not flat_path.exists():
        return None
    layer_dir = _layer_dir_for(model_path, inner)
    if layer_dir.exists() and (layer_dir / "manifest.json").exists():
        return None  # already migrated
    # Read width/height from the flat PNG.
    try:
        im = Image.open(flat_path)
        w, h = im.size
    except (OSError, ValueError):
        return None
    layer_dir.mkdir(parents=True, exist_ok=True)
    # Copy as layer 0.
    layer0 = layer_dir / "0.png"
    layer0.write_bytes(flat_path.read_bytes())
    manifest = _paint_mod.make_default_manifest(
        model_path=model_path, inner=inner, width=w, height=h,
    )
    _write_manifest(layer_dir, manifest)
    log.info("paint migrate flat -> layers: %s -> %s", flat_path.name, layer_dir)
    return manifest


def _composite_and_write_flat(model_path: str, inner: str, manifest: dict) -> Optional[Path]:
    """Recompute the flat ``<safe>.png`` from the layer stack.

    Used after every layer-save / layer-delete / layer-reorder so the
    archive-build path (which still consumes flat PNGs) sees the
    latest composite. Returns the flat-png path on success, None if
    the manifest is missing layer files.
    """
    layer_dir = _layer_dir_for(model_path, inner)
    width = int(manifest["width"])
    height = int(manifest["height"])
    layer_inputs: list[tuple[bytes, dict, Optional[bytes]]] = []
    for L in manifest["layers"]:
        idx = int(L["idx"])
        png_path = layer_dir / f"{idx}.png"
        if not png_path.exists():
            return None
        try:
            rgba_bytes, lw, lh = _paint_mod.png_bytes_to_rgba(png_path.read_bytes())
        except (OSError, ValueError) as e:
            raise HTTPException(500, f"layer {idx} decode failed: {e}")
        if lw != width or lh != height:
            raise HTTPException(500, f"layer {idx} dim mismatch: {lw}x{lh} vs {width}x{height}")
        mask_bytes: Optional[bytes] = None
        if L.get("has_mask"):
            mp = layer_dir / f"{idx}_mask.png"
            if mp.exists():
                try:
                    mask_buf, mw, mh = _paint_mod.png_bytes_to_rgba(mp.read_bytes())
                except (OSError, ValueError):
                    mask_buf = None
                if mask_buf is not None and mw == width and mh == height:
                    mask_bytes = bytes(mask_buf)
        layer_inputs.append((bytes(rgba_bytes), L, mask_bytes))
    composite = _paint_mod.composite_layers(layer_inputs, width, height)
    flat_name = _paint_mod.safe_painted_basename(model_path, inner)
    flat_path = PAINTED_TEX_DIR / flat_name
    out_bytes = _paint_mod.rgba_to_png_bytes(bytes(composite), width, height)
    tmp = flat_path.with_suffix(flat_path.suffix + ".tmp")
    tmp.write_bytes(out_bytes)
    os.replace(tmp, flat_path)
    return flat_path


class PaintLayerSaveReq(BaseModel):
    model_path: str
    inner: str
    layer_idx: int
    png_b64: str
    is_mask: bool = False              # True writes <idx>_mask.png instead
    name: Optional[str] = None         # layer display name
    visible: Optional[bool] = None
    opacity: Optional[float] = None
    blend_mode: Optional[str] = None
    locked: Optional[bool] = None
    has_mask: Optional[bool] = None    # explicit toggle (Add/Remove mask UI)


@app.post("/api/paint/layer/save")
def api_paint_layer_save(req: PaintLayerSaveReq, request: Request):
    """Persist a single layer (or its mask) into the layer stack.

    Creates the layer directory + manifest on first call. If a flat
    legacy PNG existed at ``<safe>.png`` we silently migrate it to
    layer 0 before processing the new save. Always recomputes the
    flat PNG composite at the end so the archive-build path stays
    consistent.

    Returns ``{ok, manifest, layer_idx, width, height, composite_md5}``.
    """
    _enforce_body_size(request, MAX_PAINT_SAVE_BODY)
    if not req.model_path or not req.inner:
        raise HTTPException(400, "model_path and inner are required")
    if req.layer_idx < 0 or req.layer_idx > 31:
        raise HTTPException(400, "layer_idx out of range")

    layer_dir = _layer_dir_for(req.model_path, req.inner)
    # Try to migrate a flat-PNG cache entry (no-op if already migrated).
    _migrate_flat_png_to_layer_dir(req.model_path, req.inner)
    layer_dir.mkdir(parents=True, exist_ok=True)

    raw = _decode_b64(req.png_b64, ctx="png_b64")
    if len(raw) < 8 or raw[:8] != PNG_MAGIC:
        raise HTTPException(400, "png_b64: not a PNG (magic bytes missing)")
    try:
        im = Image.open(BytesIO(raw))
        im.load()
        if im.mode != "RGBA":
            im = im.convert("RGBA")
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"PNG decode failed: {e}")
    w, h = im.size

    # Read or create the manifest.
    manifest = _read_manifest(layer_dir)
    if manifest is None:
        manifest = _paint_mod.make_default_manifest(
            model_path=req.model_path, inner=req.inner, width=w, height=h,
        )
    else:
        if manifest["width"] != w or manifest["height"] != h:
            # Dimensions for new layers must match the stack.
            raise HTTPException(
                400,
                f"layer size {w}x{h} mismatch with manifest "
                f"{manifest['width']}x{manifest['height']}",
            )

    # Re-encode through PIL for canonicalisation.
    out_buf = BytesIO()
    im.save(out_buf, format="PNG")
    final_bytes = out_buf.getvalue()
    suffix = "_mask.png" if req.is_mask else ".png"
    fp = layer_dir / f"{req.layer_idx}{suffix}"
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_bytes(final_bytes)
    os.replace(tmp, fp)

    # Make sure this layer is in the manifest. New-layer flow inserts at
    # the top of the stack (highest idx).
    layers = manifest["layers"]
    by_idx = {L["idx"]: L for L in layers}
    if req.layer_idx not in by_idx:
        new_layer = {
            "idx": req.layer_idx,
            "name": req.name or f"Layer {req.layer_idx}",
            "visible": True if req.visible is None else bool(req.visible),
            "opacity": 1.0 if req.opacity is None else float(req.opacity),
            "blend_mode": req.blend_mode or "normal",
            "locked": bool(req.locked) if req.locked is not None else False,
            "has_mask": bool(req.has_mask) if req.has_mask is not None else False,
        }
        layers.append(new_layer)
    else:
        L = by_idx[req.layer_idx]
        if req.name is not None:
            L["name"] = req.name
        if req.visible is not None:
            L["visible"] = bool(req.visible)
        if req.opacity is not None:
            L["opacity"] = float(req.opacity)
        if req.blend_mode is not None:
            L["blend_mode"] = req.blend_mode
        if req.locked is not None:
            L["locked"] = bool(req.locked)
        if req.has_mask is not None:
            L["has_mask"] = bool(req.has_mask)
        # Saving a mask implies has_mask = True.
        if req.is_mask:
            L["has_mask"] = True

    _write_manifest(layer_dir, manifest)
    composite_path = _composite_and_write_flat(req.model_path, req.inner, manifest)
    composite_md5 = ""
    if composite_path:
        composite_md5 = hashlib.md5(composite_path.read_bytes()).hexdigest()
    return {
        "ok": True,
        "manifest": _paint_mod.validate_manifest(manifest),
        "layer_idx": req.layer_idx,
        "width": w,
        "height": h,
        "composite_md5": composite_md5,
    }


class PaintManifestUpdateReq(BaseModel):
    model_path: str
    inner: str
    manifest: dict           # caller writes the FULL replacement manifest


@app.post("/api/paint/manifest")
def api_paint_manifest_update(req: PaintManifestUpdateReq):
    """Replace the manifest (after reorder / delete / merge / duplicate).

    The caller is the source of truth for layer order + metadata. We
    validate, normalize, and write atomically. Layers referenced by
    the manifest but not present on disk are flagged in the response
    so the frontend can re-upload them.
    """
    if not req.model_path or not req.inner:
        raise HTTPException(400, "model_path and inner are required")
    layer_dir = _layer_dir_for(req.model_path, req.inner)
    if not layer_dir.exists():
        raise HTTPException(404, "no layer dir for this texture")
    norm = _paint_mod.validate_manifest(req.manifest)
    _write_manifest(layer_dir, norm)
    composite_path = _composite_and_write_flat(req.model_path, req.inner, norm)
    composite_md5 = ""
    if composite_path:
        composite_md5 = hashlib.md5(composite_path.read_bytes()).hexdigest()
    missing: list[int] = []
    for L in norm["layers"]:
        idx = int(L["idx"])
        if not (layer_dir / f"{idx}.png").exists():
            missing.append(idx)
    return {
        "ok": True,
        "manifest": norm,
        "composite_md5": composite_md5,
        "missing_layer_indices": missing,
    }


class PaintLayerDeleteReq(BaseModel):
    model_path: str
    inner: str
    layer_idx: int
    is_mask: bool = False


@app.post("/api/paint/layer/delete")
def api_paint_layer_delete(req: PaintLayerDeleteReq):
    """Delete a single layer's PNG (or its mask) and prune the manifest.

    When ``is_mask=True`` only the mask file is removed; the layer
    persists with ``has_mask = False``. Otherwise the entire layer
    + its mask + its manifest entry are removed.
    """
    layer_dir = _layer_dir_for(req.model_path, req.inner)
    if not layer_dir.exists():
        raise HTTPException(404, "no layer dir for this texture")
    manifest = _read_manifest(layer_dir)
    if not manifest:
        raise HTTPException(404, "no manifest for this texture")
    by_idx = {L["idx"]: L for L in manifest["layers"]}
    if req.layer_idx not in by_idx:
        raise HTTPException(404, f"layer_idx {req.layer_idx} not in manifest")
    if req.is_mask:
        mp = layer_dir / f"{req.layer_idx}_mask.png"
        if mp.exists():
            mp.unlink()
        by_idx[req.layer_idx]["has_mask"] = False
    else:
        if len(manifest["layers"]) <= 1:
            raise HTTPException(400, "cannot delete the only layer in the stack")
        # Remove layer png + mask png.
        for suffix in (".png", "_mask.png"):
            fp = layer_dir / f"{req.layer_idx}{suffix}"
            if fp.exists():
                fp.unlink()
        manifest["layers"] = [L for L in manifest["layers"] if L["idx"] != req.layer_idx]
        if manifest["active"] == req.layer_idx:
            manifest["active"] = manifest["layers"][-1]["idx"]
    _write_manifest(layer_dir, manifest)
    composite_path = _composite_and_write_flat(req.model_path, req.inner, manifest)
    composite_md5 = ""
    if composite_path:
        composite_md5 = hashlib.md5(composite_path.read_bytes()).hexdigest()
    return {"ok": True, "manifest": manifest, "composite_md5": composite_md5}


@app.get("/api/paint/load")
def api_paint_load(model_path: str, inner: str):
    """Return the layer stack for a (model_path, inner) pair.

    Response shape::

        {
          "ok": true,
          "manifest": {...},
          "layers": [
            {"idx": 0, "png_b64": "...", "mask_b64": null},
            ...
          ]
        }

    If the layer directory doesn't exist but a legacy flat PNG does,
    it is auto-migrated and returned as a single layer. If neither
    exists, returns ``{ok: true, manifest: null, layers: []}``.
    """
    if not model_path or not inner:
        raise HTTPException(400, "model_path and inner are required")
    layer_dir = _layer_dir_for(model_path, inner)
    manifest = _read_manifest(layer_dir) if layer_dir.exists() else None
    if manifest is None:
        manifest = _migrate_flat_png_to_layer_dir(model_path, inner)
    if manifest is None:
        return {"ok": True, "manifest": None, "layers": []}
    layers_out: list[dict] = []
    for L in manifest["layers"]:
        idx = int(L["idx"])
        png_b64 = ""
        mask_b64 = None
        png_path = layer_dir / f"{idx}.png"
        if png_path.exists():
            png_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
        if L.get("has_mask"):
            mp = layer_dir / f"{idx}_mask.png"
            if mp.exists():
                mask_b64 = base64.b64encode(mp.read_bytes()).decode("ascii")
        layers_out.append({"idx": idx, "png_b64": png_b64, "mask_b64": mask_b64})
    return {"ok": True, "manifest": manifest, "layers": layers_out}


@app.get("/api/paint/active")
def api_paint_active():
    """List every painted texture currently cached.

    Returns ``{ok, painted: [{model_path, inner, basename, size, mtime,
    md5, layer_count}, ...]}``. Frontend can render a "you've painted these"
    panel without hitting the filesystem itself.

    v5: also reports the layer count when a layer directory exists.
    """
    out: list[dict] = []
    if not PAINTED_TEX_DIR.exists():
        return {"ok": True, "painted": out, "dir": str(PAINTED_TEX_DIR)}
    for p in sorted(PAINTED_TEX_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        try:
            sz = p.stat().st_size
            mtime = p.stat().st_mtime
        except OSError:
            continue
        # Reverse-engineer model_path / inner from the basename.
        stem = p.name
        if stem.lower().endswith(".png"):
            stem = stem[: -len(".png")]
        if "__" not in stem:
            # Shouldn't occur — but skip gracefully if it does.
            continue
        model_part, _, inner_part = stem.partition("__")
        # PNG md5 (cheap — server-side cache, called rarely).
        try:
            data = p.read_bytes()
            md5 = hashlib.md5(data).hexdigest()
        except OSError:
            md5 = ""
        # v5: if a layer dir exists alongside this composite, record count.
        layer_count = 1
        layer_dir = PAINTED_TEX_DIR / stem
        if layer_dir.is_dir():
            mf = layer_dir / "manifest.json"
            if mf.exists():
                try:
                    manifest_data = json.loads(mf.read_text(encoding="utf-8"))
                    layer_count = len(manifest_data.get("layers", []) or [])
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        out.append({
            "model_path": model_part,
            "inner": inner_part,
            "basename": p.name,
            "size": sz,
            "mtime": mtime,
            "md5": md5,
            "layer_count": layer_count,
        })
    return {"ok": True, "painted": out, "dir": str(PAINTED_TEX_DIR)}


# ---------------------------------------------------------------------------- parse cache
#
# Phase D Win 4 LRU for parsed XjMesh / XjBone trees. The /api/model_mesh,
# /api/model_skinned, /api/model_skeleton, /api/animations, and
# /api/model_textures endpoints all route through formats.parse_cache so
# repeat opens of the same model — variant picker, motion preview, paint,
# sculpt, model_textures fetch — return parsed data in <5 ms instead of
# re-walking the NJ chunk stream (~1.1 s for dragon-class models).
#
# Endpoints below are dev-facing diagnostics; the cache itself is on for
# every model load.

@app.get("/api/parse_cache/stats")
def api_parse_cache_stats():
    """Return parsed-mesh cache health (see formats/parse_cache.cache_stats for shape)."""
    return _parse_cache.cache_stats()


@app.delete("/api/parse_cache/clear")
@app.post("/api/parse_cache/clear")
def api_parse_cache_clear(disk: int = 1):
    """Drop the parsed-mesh cache (in-memory + on-disk pickles unless disk=0)."""
    return _parse_cache.cache_clear(drop_disk=bool(disk))


# ---------------------------------------------------------------------------- disk-tier sweep
#
# Audit M-3 (2026-05-01). Hand-trigger for `_sweep_cache_dir` so the user
# can reclaim disk without restarting the server. Body fields are all
# optional — defaults match the startup pass on the parse_cache dir.
#
# Allowed `dir` values map to whitelisted subdirs of CACHE_DIR; arbitrary
# paths are rejected to keep this from doubling as a path-traversal foot-
# gun. The 'all' value sweeps both managed dirs in one call.
class CacheSweepReq(BaseModel):
    dir: str = Field(
        "parse_cache",
        description="cache subdir name: 'parse_cache' | 'bml_inner' | 'all'",
    )
    max_bytes: Optional[int] = Field(
        None, ge=0, description="aggregate size cap; default per dir if omitted",
    )
    max_age_days: Optional[int] = Field(
        None, ge=0, description="max file age in days; default 30 if omitted",
    )


_SWEEP_DEFAULTS = {
    # name -> (subdir-from-CACHE_DIR, default_max_bytes, default_max_age_days)
    "parse_cache": ("parse_cache", 1024 * 1024 * 1024, 30),
    "bml_inner":   (_BML_INNER_CACHE_SUBDIR, 500 * 1024 * 1024, 30),
}


@app.post("/api/cache/sweep")
def api_cache_sweep(req: CacheSweepReq, request: Request):
    """Trigger disk-tier sweep on a managed cache dir.

    Returns the per-dir `_sweep_cache_dir` stats dict, or `{dirs: {...}}`
    when sweeping all managed dirs at once.
    """
    _enforce_body_size(request, MAX_REPACK_DIFF_BODY)
    name = (req.dir or "").strip().lower()
    if name == "all":
        out: dict[str, dict] = {}
        for k, (sub, dbytes, dage) in _SWEEP_DEFAULTS.items():
            mbytes = req.max_bytes if req.max_bytes is not None else dbytes
            mage = req.max_age_days if req.max_age_days is not None else dage
            out[k] = _sweep_cache_dir(CACHE_DIR / sub, mbytes, mage)
        return {"dirs": out}
    if name not in _SWEEP_DEFAULTS:
        raise HTTPException(
            400,
            f"unknown dir {req.dir!r}; allowed: "
            f"{list(_SWEEP_DEFAULTS) + ['all']}",
        )
    sub, dbytes, dage = _SWEEP_DEFAULTS[name]
    mbytes = req.max_bytes if req.max_bytes is not None else dbytes
    mage = req.max_age_days if req.max_age_days is not None else dage
    return _sweep_cache_dir(CACHE_DIR / sub, mbytes, mage)


# ---------------------------------------------------------------------------- binding cache
#
# Phase D Win 4 follow-up LRU for the per-model NJTL→XVMH binding payload.
# The /api/model_mesh, /api/model_skinned, /api/model_bundle, and
# /api/model_textures endpoints all route binding computation through
# `_build_model_texture_binding_cached`, which resolves cross-archive
# references once per (path, inner, mtime) and reuses the result across
# repeat opens. Composes with the parse_cache layer (parse → binding).

@app.get("/api/binding_cache/stats")
def api_binding_cache_stats():
    """Return texture-binding cache health (mirrors /api/parse_cache/stats shape)."""
    return _binding_cache_stats()


@app.delete("/api/binding_cache/clear")
@app.post("/api/binding_cache/clear")
def api_binding_cache_clear():
    """Drop the in-memory binding cache."""
    return _binding_cache_clear()


# ---------------------------------------------------------------------------- skinned-payload cache
#
# Phase 0.5 perf (2026-04-25). LRU + on-disk cache for the OUTPUT dict of
# `_xj_meshes_to_skinned_payload`. Keyed on (path, mtime_ns, size, inner)
# so a re-deploy invalidates implicitly. Disk tier is JSON for human-
# readable cat-debug; in-memory holds the assembled dict so warm hits
# return in <1 ms.

@app.get("/api/skinned_payload_cache/stats")
def api_skinned_payload_cache_stats():
    """Return skinned-payload cache health (mirrors /api/parse_cache/stats shape)."""
    return _skinned_payload_cache_stats()


@app.delete("/api/skinned_payload_cache/clear")
@app.post("/api/skinned_payload_cache/clear")
def api_skinned_payload_cache_clear(disk: int = 1):
    """Drop the skinned-payload cache (in-memory + on-disk JSON unless disk=0)."""
    return _skinned_payload_cache_clear(drop_disk=bool(disk))


# ----------------------------------------------------------------------------
# Coverage status (2026-04-25): surfaces the latest parser-side and
# render-side audit results so the editor can show an "X / Y models
# render correctly" pip in the deploy panel without re-running the audits.
# Pure-read; both CSVs are produced by scripts/coverage_audit.py and
# scripts/render_coverage_audit.py.
# ----------------------------------------------------------------------------


@app.get("/api/coverage_status")
def api_coverage_status():
    """Return parser-side + render-side coverage stats.

    Reads MODEL_COVERAGE.csv (parser audit) and
    _reports/render_coverage.csv (render audit) and produces a small
    summary dict shaped:

        {
          "parser":    {"total": N, "ok": N, "by_failure": {...}, "csv_mtime": int|None},
          "render":    {"total": N, "by_status": {...}, "csv_mtime": int|None},
          "delta":     {"parser_ok_render_not": [...],  // up to 16 sample rows
                        "parser_failed_render_ok": [...]},
        }

    Returns 404 if neither CSV exists (rather than a hard 500). The
    editor surfaces "coverage unknown — run scripts/coverage_audit.py"
    when this 404s, so the endpoint is fail-soft by design.
    """
    parser_csv = Path(__file__).resolve().parent / "MODEL_COVERAGE.csv"
    render_csv = Path(__file__).resolve().parent / "_reports" / "render_coverage.csv"

    out: dict = {"parser": None, "render": None, "delta": None}

    # Parser side ---------------------------------------------------------
    if parser_csv.is_file():
        import csv as _csv
        try:
            with parser_csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(_csv.DictReader(f))
        except (OSError, _csv.Error) as e:
            log.warning("coverage_status: parser csv read failed: %s", e)
            rows = []
        by_fail: dict = {}
        ok = 0
        for r in rows:
            cls = r.get("failure_class") or ""
            if cls:
                by_fail[cls] = by_fail.get(cls, 0) + 1
            else:
                ok += 1
        out["parser"] = {
            "total": len(rows),
            "ok": ok,
            "by_failure": by_fail,
            "csv_mtime": int(parser_csv.stat().st_mtime),
        }

    # Render side ---------------------------------------------------------
    if render_csv.is_file():
        import csv as _csv
        try:
            with render_csv.open("r", encoding="utf-8", newline="") as f:
                rrows = list(_csv.DictReader(f))
        except (OSError, _csv.Error) as e:
            log.warning("coverage_status: render csv read failed: %s", e)
            rrows = []
        by_status: dict = {}
        for r in rrows:
            s = r.get("status") or "unknown"
            by_status[s] = by_status.get(s, 0) + 1
        out["render"] = {
            "total": len(rrows),
            "by_status": by_status,
            "csv_mtime": int(render_csv.stat().st_mtime),
        }

    # If neither side has any data, treat as 404 so the UI can show
    # "coverage unknown" rather than a misleading "0 / 0 ok".
    if out["parser"] is None and out["render"] is None:
        raise HTTPException(
            404,
            "no coverage data; run "
            "scripts/coverage_audit.py and scripts/render_coverage_audit.py",
        )
    return out


# ----------------------------------------------------------------------------
# UX maturity layer (2026-04-25): workspaces + cross-asset batch.
# These endpoints back the new client-side modules in static/workspace.js and
# static/multi_select.js. Both are additive - disabling either of them on the
# frontend leaves the rest of the editor working.
# ----------------------------------------------------------------------------

# Named-workspace JSON snapshots live here. localStorage holds the
# unnamed automatic snapshot; this directory holds anything the user
# clicked "Save Workspace as ..." on. JSON-only payloads, capped at 1 MB
# each - workspaces are layout state, not asset data.
WORKSPACE_DIR = CACHE_DIR / "workspaces"
WORKSPACE_DIR.mkdir(exist_ok=True)
MAX_WORKSPACE_BODY = 1 * 1024 * 1024  # 1 MB - pure JSON layout.
MAX_BATCH_BODY = 4 * 1024 * 1024      # 4 MB - paths only.
MAX_WORKSPACE_NAME_LEN = 80
_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-. ]{1,80}$")


def _safe_workspace_path(name: str) -> Path:
    """Resolve a workspace name safely under WORKSPACE_DIR.

    Rejects empty / dotted / oversize names + anything that round-trips
    out of WORKSPACE_DIR. Returns the .json path (may not exist).
    """
    if not isinstance(name, str) or not name:
        raise HTTPException(400, "missing workspace name")
    if len(name) > MAX_WORKSPACE_NAME_LEN:
        raise HTTPException(400, "workspace name too long")
    if not _WORKSPACE_NAME_RE.match(name):
        raise HTTPException(400, "workspace name contains forbidden characters")
    p = (WORKSPACE_DIR / (name + ".json")).resolve()
    try:
        p.relative_to(WORKSPACE_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "workspace name escapes workspace dir")
    return p


class WorkspaceSaveReq(BaseModel):
    name: str
    blob: dict


@app.post("/api/workspace/save")
def api_workspace_save(req: WorkspaceSaveReq, request: Request):
    """Persist a named workspace JSON snapshot under cache/workspaces/."""
    _enforce_body_size(request, MAX_WORKSPACE_BODY)
    p = _safe_workspace_path(req.name)
    # Atomic write: tmp -> replace, so a crash mid-save can't corrupt
    # an existing workspace file.
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(req.blob, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        raise HTTPException(500, f"save failed: {e}")
    return {"ok": True, "name": req.name, "path": str(p), "size": p.stat().st_size}


@app.get("/api/workspace/load")
def api_workspace_load(name: str):
    """Load a previously saved workspace by name."""
    p = _safe_workspace_path(name)
    if not p.exists():
        raise HTTPException(404, f"workspace '{name}' not found")
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"load failed: {e}")
    return {"ok": True, "name": name, "blob": blob}


@app.get("/api/workspace/list")
def api_workspace_list():
    """List every saved workspace with size + mtime."""
    out = []
    for p in sorted(WORKSPACE_DIR.glob("*.json")):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({
            "name": p.stem,
            "size": st.st_size,
            "mtime": int(st.st_mtime),
        })
    return {"workspaces": out, "dir": str(WORKSPACE_DIR)}


@app.post("/api/workspace/delete")
def api_workspace_delete(name: str):
    """Delete a named workspace."""
    p = _safe_workspace_path(name)
    if not p.exists():
        raise HTTPException(404, f"workspace '{name}' not found")
    try:
        p.unlink()
    except OSError as e:
        raise HTTPException(500, f"delete failed: {e}")
    return {"ok": True, "name": name}


class BatchReq(BaseModel):
    op: str = Field(..., min_length=1, max_length=64)
    paths: list[str] = Field(default_factory=list)
    payload: dict = Field(default_factory=dict)


@app.post("/api/batch")
def api_batch(req: BatchReq, request: Request):
    """Cross-asset batch operations.

    Currently supported ops:
      * ``upscale`` - payload {scale, model?, keep_native_dims?}
                      runs the existing upscaler against each path
                      interpreted as a `<archive>` (or `<archive>#<inner>`)
                      and operates on every tile in the file. Returns a
                      per-path result list with {path, tiles, ok, error}.
      * ``noop``    - payload ignored. Echoes input. Useful for tests.

    Designed to be additive: panels can introduce new ops here without
    bleeding into the single-asset endpoints.
    """
    _enforce_body_size(request, MAX_BATCH_BODY)
    if not req.paths:
        raise HTTPException(400, "paths must not be empty")
    if len(req.paths) > 500:
        raise HTTPException(400, "batch capped at 500 paths")
    op = req.op.lower()
    results = []
    n_ok = 0
    n_failed = 0
    if op == "noop":
        for p in req.paths:
            results.append({"path": p, "ok": True})
            n_ok += 1
        return {"ok": n_ok, "failed": n_failed, "results": results}
    if op == "upscale":
        scale = int(req.payload.get("scale", 4))
        if scale not in ALLOWED_SCALES:
            raise HTTPException(400, f"scale must be one of {ALLOWED_SCALES}")
        model = str(req.payload.get("model") or "realesrgan-x4plus")
        if not MODEL_NAME_RE.match(model):
            raise HTTPException(400, "invalid model name")
        keep_native = bool(req.payload.get("keep_native_dims", True))
        for raw in req.paths:
            try:
                # Each path may be "<file>" or "<file>#<inner>". The
                # single-asset upscale endpoint accepts both via
                # _materialize_inner_for_extract.
                tiles_attempted = 0
                tiles_ok = 0
                if "#" in raw:
                    bare = raw
                    prs = _materialize_inner_for_extract(bare)
                else:
                    bare = Path(raw).name
                    prs = safe_data_path(bare)
                if not prs.exists():
                    results.append({"path": raw, "ok": False, "error": "file missing"})
                    n_failed += 1
                    continue
                manifest = extract_tiles(prs)
                for tile in manifest.get("tiles", []):
                    tiles_attempted += 1
                    try:
                        api_upscale(  # call the validated single-tile path
                            UpscaleReq(
                                filename=bare,
                                tile_index=int(tile["index"]),
                                model=model,
                                scale=scale,
                                keep_native_dims=keep_native,
                            ),
                            request,
                        )
                        tiles_ok += 1
                    except HTTPException as e:
                        log.warning(
                            "batch upscale: tile %s of %s failed: %s",
                            tile.get("index"), bare, e.detail,
                        )
                results.append({
                    "path": raw,
                    "ok": tiles_ok > 0,
                    "tiles": tiles_attempted,
                    "tiles_ok": tiles_ok,
                })
                if tiles_ok > 0:
                    n_ok += 1
                else:
                    n_failed += 1
            except HTTPException as e:
                results.append({"path": raw, "ok": False, "error": e.detail})
                n_failed += 1
            except Exception as e:  # noqa: BLE001 - defence in depth for batch
                log.exception("batch upscale: %s threw", raw)
                results.append({"path": raw, "ok": False, "error": str(e)})
                n_failed += 1
        return {"ok": n_ok, "failed": n_failed, "results": results}
    raise HTTPException(400, f"unknown batch op: {req.op!r}")


# ============================================================================
# Map Editor endpoints (2026-04-25).
#
# 4 routes:
#   GET  /api/map/list                  picker payload
#   GET  /api/map/<map_id>?floor=N       per-floor asset bundle
#   POST /api/map/edits                  save spawns/waypoints to sidecar JSON
#   GET  /api/map/edits/<map_id>         load sidecar JSON
#
# All four use the cached manifest as source of truth; ``scene_loader``
# does the heavy lifting (parsing names, classifying areas, validating
# spawn payloads). Sidecar writes are atomic (tmp + replace).
# ============================================================================

# Module-level cache: rebuild only when the underlying manifest mtime
# changes. The catalogue is cheap (~3 ms over 5900 entries) but the
# picker is on the page-load critical path so we may as well cache.
_MAP_CATALOGUE_CACHE: dict = {"mtime": None, "maps": None}


def _build_map_catalogue() -> list:
    """Return a cached list[MapInfo] from the live manifest."""
    cf = manifest_mod.cache_path_for(DATA_DIR, cache_dir=CACHE_DIR)
    mtime = int(cf.stat().st_mtime) if cf.exists() else 0
    if _MAP_CATALOGUE_CACHE["mtime"] == mtime and _MAP_CATALOGUE_CACHE["maps"] is not None:
        return _MAP_CATALOGUE_CACHE["maps"]
    m = manifest_mod.cache_manifest(DATA_DIR, cache_dir=CACHE_DIR)
    maps = _scene_loader.catalogue(m.get("entries", []))
    _MAP_CATALOGUE_CACHE["mtime"] = mtime
    _MAP_CATALOGUE_CACHE["maps"] = maps
    return maps


_MAP_ID_RE = re.compile(r"^[a-z]+\d+$")


def _safe_map_id(map_id: str) -> str:
    """Path-injection guard for map_ids used as filenames."""
    if not map_id or not isinstance(map_id, str):
        raise HTTPException(400, "map_id required")
    if not _MAP_ID_RE.match(map_id):
        raise HTTPException(400, f"invalid map_id {map_id!r}")
    return map_id


@app.get("/api/map/list")
def api_map_list():
    """Return picker payload (mirrors scene_loader.make_picker_payload shape)."""
    maps = _build_map_catalogue()
    return _scene_loader.make_picker_payload(maps)


@app.get("/api/map/edits/{map_id}")
def api_map_edits_load(map_id: str):
    """Return the sidecar JSON for ``map_id``, or an empty doc if absent.

    NOTE: defined BEFORE /api/map/{map_id} so FastAPI's path matcher
    routes ``/api/map/edits/aancient01`` to this handler instead of the
    bundle endpoint with map_id="edits".
    """
    map_id = _safe_map_id(map_id)
    p = MAP_EDITS_DIR / f"{map_id}.json"
    if not p.exists():
        return {
            "ok": True,
            "map_id": map_id,
            "exists": False,
            "version": _scene_loader.SPAWN_FILE_VERSION,
            "spawns": [],
            "waypoints": [],
        }
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"sidecar corrupt: {e}")
    return {
        "ok": True,
        "map_id": map_id,
        "exists": True,
        **data,
    }


class MapEditsReq(BaseModel):
    map_id: str
    spawns: list = Field(default_factory=list)
    waypoints: list = Field(default_factory=list)


# Sidecar writes are bounded; a max of 4096 spawns + waypoints is
# already wildly more than any quest ships.
MAX_MAP_EDITS_BODY = 256 * 1024


@app.post("/api/map/edits")
def api_map_edits_save(req: MapEditsReq, request: Request):
    """Persist spawn + waypoint placements to ``cache/map_edits/<map_id>.json``.

    Validation runs through :func:`scene_loader.validate_edits_payload`.
    Atomic write (tmp + replace). Returns the normalized payload back so
    the frontend can re-anchor after server-side coercion.
    """
    _enforce_body_size(request, MAX_MAP_EDITS_BODY)
    payload = req.model_dump()
    ok, err = _scene_loader.validate_edits_payload(payload)
    if not ok:
        raise HTTPException(400, err)
    map_id = _safe_map_id(payload["map_id"])
    norm = _scene_loader.normalize_edits_payload(payload)
    out_path = MAP_EDITS_DIR / f"{map_id}.json"
    body = json.dumps(norm, indent=2)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, out_path)
    return {
        "ok": True,
        "map_id": map_id,
        "path": str(out_path),
        "spawn_count": len(norm["spawns"]),
        "waypoint_count": len(norm["waypoints"]),
        "edits": norm,
    }


# Path resolver for files under data/scene/. The standard
# _resolve_model_mesh_path is locked to flat filenames; the Map Editor
# needs to load scene/map_*.nj which lives one level deeper. We accept
# only paths that begin with "scene/" + a single filename component;
# anything else (.., absolute, multiple dirs) rejects.
_SCENE_PATH_RE = re.compile(r"^scene/[A-Za-z0-9_.-]+\.[A-Za-z0-9]+$")


def _resolve_scene_asset_path(path: str) -> Path:
    if not path or not isinstance(path, str):
        raise HTTPException(400, "missing scene path")
    if not _SCENE_PATH_RE.match(path):
        raise HTTPException(400, f"invalid scene path: {path!r}")
    bare = path.split("/", 1)[1]
    if bare in _INVALID_FILENAMES:
        raise HTTPException(400, "invalid scene filename")
    for root in (DATA_DIR, LIVE_DATA_DIR):
        cand = (root / "scene" / bare).resolve()
        try:
            cand.relative_to((root / "scene").resolve())
        except ValueError:
            continue
        if cand.exists() and cand.is_file():
            return cand
    raise HTTPException(404, f"scene asset not found: {path}")


@app.get("/api/map/asset/{path:path}")
def api_map_asset(path: str):
    """Parse and return triangulated mesh data for a scene/map_* file.

    Mirror of /api/model_mesh but accepts ``scene/<file>.nj`` /
    ``scene/<file>.xj`` paths. The standard model_mesh endpoint refuses
    paths with directory components (path-injection guard); for scene
    files we know the layout is exactly ``scene/<flat_name>.<ext>`` so
    we can validate strictly and pipe through to the same parser.

    Wire shape matches /api/model_mesh exactly (since both call
    ``_xj_meshes_to_payload`` after parsing). Texture binding is
    skipped — terrain meshes share one .xvm sibling and the Map Editor
    binds them itself if needed (v2).

    Special-case for ``.rel`` files (n.rel only): extract the embedded
    XJ buffer-descriptor mesh (Pioneer 2 / city / lab maps that ship no
    raw .nj). All other ``.rel`` flavours (c.rel, r.rel) are rejected
    with 400 since they carry collision / runtime hints rather than
    drawable terrain.
    """
    p = _resolve_scene_asset_path(path)
    ext = p.suffix.lower()
    if ext == ".rel":
        return _api_map_asset_rel(p, path)
    if ext not in (".nj", ".xj"):
        raise HTTPException(400, f"unsupported scene mesh extension: {ext}")
    sz = p.stat().st_size
    if sz > ASSET_PARSE_MAX_BYTES:
        raise HTTPException(413, f"scene mesh too large: {sz} bytes")
    data = p.read_bytes()
    try:
        meshes = _cached_model_parse(data, p, ext, "", None)
    except ValueError as e:
        raise HTTPException(400, f"scene parse failed: {e}")
    except Exception as e:  # pragma: no cover
        log.exception("scene parse internal error for %s", path)
        raise HTTPException(500, f"scene parse internal error: {e}")
    payload = _xj_meshes_to_payload(meshes)
    payload["filename"] = path
    payload["binding"] = []
    payload["binding_data"] = {"njtl": [], "xvmh": [], "binding": []}
    return payload


def _api_map_asset_rel(p: Path, path: str) -> dict:
    """Extract terrain meshes from an n.rel and return /api/map/asset shape.

    Only n.rel files (PSOBB visibility-chunk archives whose payload
    starts with the ``"fmt2"`` magic) carry drawable terrain. We
    forward to the same wire shape used by /api/model_mesh so the
    frontend's scene assembler can consume both paths uniformly.
    """
    from formats import rel as _rel  # lazy import — keeps cold start small

    sz = p.stat().st_size
    if sz > ASSET_PARSE_MAX_BYTES:
        raise HTTPException(413, f"scene rel too large: {sz} bytes")
    try:
        buf = p.read_bytes()
        rel = _rel.parse_rel(buf)
    except _rel.RelParseError as e:
        raise HTTPException(400, f"rel parse failed: {e}")
    except Exception as e:  # pragma: no cover
        log.exception("rel parse internal error for %s", path)
        raise HTTPException(500, f"rel parse internal error: {e}")
    if not _rel.is_n_rel(rel):
        raise HTTPException(
            400, f"{path}: not an n.rel (only n.rel carries terrain)")
    try:
        meshes = _rel.extract_nrel_meshes(rel)
    except _rel.RelParseError as e:
        raise HTTPException(400, f"rel mesh extract failed: {e}")
    payload = _xj_meshes_to_payload(meshes)
    payload["filename"] = path
    payload["binding"] = []
    payload["binding_data"] = {"njtl": [], "xvmh": [], "binding": []}
    # Surface the texture names the n.rel references so future versions
    # can route to the per-map .xvm sibling.
    try:
        payload["rel_texture_names"] = _rel.read_texture_names(rel)
    except Exception:  # pragma: no cover — defensive
        payload["rel_texture_names"] = []
    return payload


@app.get("/api/map/{map_id}")
def api_map_get(map_id: str, floor: int = 0):
    """Return the asset bundle for one (map, floor) tuple.

    Args:
      map_id: PSOBB map id like ``aancient01``, ``machine02``, ``boss09``.
      floor: 0-based floor index. Most maps have 0..4; bosses 0..2.

    Returns:
      JSON shaped per :func:`formats.scene_loader.floor_bundle`, plus
      v3 additions:
        * ``rrel_render_hints`` — the parsed r.rel anchor list + bbox
          hints when the floor ships an ``*_NN r.rel``, else None.
        * ``nrel_texture_names`` — the embedded TextureList from the
          n.rel sibling (positional names that map 1:1 to the
          ``map_<area>.xvm`` XVR records), else None.

      Empty ``renderable`` is OK (Pioneer 2 / city maps ship terrain
      via .rel relocation tables, not raw .nj).
    """
    map_id = _safe_map_id(map_id)
    maps = _build_map_catalogue()
    info = next((m for m in maps if m.map_id == map_id), None)
    if info is None:
        raise HTTPException(404, f"unknown map_id {map_id!r}")
    if floor not in info.floors:
        # Surface available floors so the frontend can re-pick gracefully.
        avail = sorted(info.floors.keys())
        if not avail:
            raise HTTPException(404, f"map {map_id!r} has no floors")
        # Default to lowest available
        floor = avail[0]
    bundle = _scene_loader.floor_bundle(info, floor)

    # v3: pull r.rel render-hints + n.rel texture-name list into the
    # bundle.  Both are best-effort — failures degrade to None so the
    # frontend can fall back to its hardcoded category table.
    bundle["rrel_render_hints"] = _load_rrel_hints(bundle.get("rrel_path"))
    bundle["nrel_texture_names"] = _load_nrel_texture_names(bundle.get("nrel_path"))
    return bundle


def _load_rrel_hints(rel_path: Optional[str]) -> Optional[dict]:
    """Best-effort fetch of r.rel render-hints for a bundle path.

    ``rel_path`` is the manifest-relative ``scene/map_*r.rel`` path
    surfaced by ``floor_bundle``.  Returns None when the path is None,
    the file is absent, or parsing fails.

    The cost is small (~1-2ms per file: 156 files cap at ~250 anchors
    each) and the bundle endpoint is hit on map-pick only, so we don't
    bother caching here.
    """
    if not rel_path:
        return None
    try:
        from formats import rel as _rel  # lazy
        p = _resolve_scene_asset_path(rel_path)
    except HTTPException:
        return None
    except Exception:  # pragma: no cover — defensive
        return None
    try:
        buf = p.read_bytes()
    except OSError:
        return None
    if len(buf) > ASSET_PARSE_MAX_BYTES:
        return None
    try:
        result = _rel.parse_rrel_render_hints(buf)
    except Exception:  # pragma: no cover — defensive
        log.exception("r.rel parse failed for %s", rel_path)
        return None
    return result if result.get("ok") else None


def _load_nrel_texture_names(rel_path: Optional[str]) -> Optional[list]:
    """Best-effort fetch of the n.rel embedded TextureList.

    The names are positional and map 1:1 to the XVR records inside
    ``map_<area>.xvm`` (verified at v3 RE time across 4 maps).
    Returns None when the n.rel is absent or has no embedded list.
    """
    if not rel_path:
        return None
    try:
        from formats import rel as _rel  # lazy
        p = _resolve_scene_asset_path(rel_path)
    except HTTPException:
        return None
    except Exception:  # pragma: no cover — defensive
        return None
    try:
        buf = p.read_bytes()
    except OSError:
        return None
    if len(buf) > ASSET_PARSE_MAX_BYTES:
        return None
    try:
        rel = _rel.parse_rel(buf)
        if not _rel.is_n_rel(rel):
            return None
        names = _rel.read_texture_names(rel)
    except Exception:  # pragma: no cover — defensive
        log.exception("n.rel name read failed for %s", rel_path)
        return None
    return names if names else None


# ===========================================================================
# Floor copy/create editor (2026-06-20)
# ===========================================================================
#   GET  /api/floors                      floor browser payload
#   POST /api/floors/copy                 duplicate a floor into a DEV slot
#   POST /api/floors/create               author a floor from a GLB / asset
#   GET  /api/floors/{floor_id}           per-floor bundle (shape == /api/map)
#   DELETE /api/floors/{floor_id}         delete a DEV slot (copies/glb only)
#
# SAFETY INVARIANT (the whole reason this block exists):
#   This editor has NO live-write verb. EVERY copy/create output resolves
#   to DEV_DATA_DIR and is HARD-ASSERTED to be neither LIVE_DATA_DIR nor a
#   child of it BEFORE any bytes hit disk -> else RuntimeError. Writes are
#   atomic (.tmp -> fsync -> os.replace, unique .tmp per writer). A verify
#   failure (re-parse + simulate_rel_relocation + budget) aborts BEFORE any
#   write. This module deliberately does NOT import / reference
#   safe_live_path or LIVE_DATA_DIR for any write path.
#
# The LITERAL routes (/api/floors, /api/floors/copy, /api/floors/create) are
# registered BEFORE the parameterized /api/floors/{floor_id} so FastAPI's
# in-order path matcher does not swallow "copy"/"create" as a floor_id.
# ===========================================================================

# Reserved stem prefix so DEV floor slots are visually distinct and the
# slot enumerator can find them. A slot named "myfloor" lands on disk as
# ``map_devmyfloor_00{n,c}.rel`` + ``map_devmyfloor_00s.xvm``.
FLOOR_SLOT_PREFIX = "map_dev"
# Stock floor ids are prefixed so the {floor_id} resolver can tell a
# read-only catalogue floor apart from an editable dev slot without
# guessing. Format: ``stk__<map_id>__<floor>`` (e.g. stk__aancient01__0).
_FLOOR_STOCK_PREFIX = "stk__"
_FLOOR_STOCK_ID_RE = re.compile(r"^stk__([a-z]+\d+)__(\d+)$")
# A dev-slot floor_id IS the bare stem (post _safe_archive_name).
_FLOOR_SLOT_STEM_RE = re.compile(r"^map_dev[A-Za-z0-9_\-]+_\d{2}$")
# Single global lock around floor copy/create — one slot-write in flight at
# a time. Non-blocking acquire -> HTTP 409 on contention (mirrors the
# _PROMOTE_LOCK pattern). Keeps the concurrency model dead simple.
_FLOOR_BUILD_LOCK = threading.Lock()


def _floor_resolve_out_dir() -> Path:
    """Return the DEV dir all floor writes are confined to, hard-asserting
    it is NOT the live install (nor a child of it).

    This is THE write boundary. It resolves to server.py's module-level
    DEV_DATA_DIR (NOT a fresh env read) so the slot list, the writer, and
    safe_data_path all agree on one directory. If that ever resolved
    inside LIVE_DATA_DIR, we raise BEFORE any caller can write a byte.
    """
    out = DEV_DATA_DIR.resolve()
    live = LIVE_DATA_DIR.resolve()
    if out == live:
        raise RuntimeError("floor editor must never write LIVE_DATA_DIR")
    try:
        out.relative_to(live)
    except ValueError:
        pass  # good — out is NOT inside live
    else:
        raise RuntimeError("floor editor must never write under LIVE_DATA_DIR")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _floor_assert_not_live(target: Path) -> Path:
    """Assert ``target`` resolves inside DEV and never inside LIVE.

    Called at the write boundary for every individual output file. Raises
    RuntimeError (NOT HTTPException) so a programming error that points a
    write at the live install fails loud and uncaught rather than being
    silently turned into a client error.
    """
    t = Path(target).resolve()
    live = LIVE_DATA_DIR.resolve()
    if t == live:
        raise RuntimeError("floor editor must never write LIVE_DATA_DIR")
    try:
        t.relative_to(live)
    except ValueError:
        pass
    else:
        raise RuntimeError(f"floor editor refusing to write under LIVE_DATA_DIR: {t}")
    # And it MUST be inside the resolved DEV dir.
    out = DEV_DATA_DIR.resolve()
    try:
        t.relative_to(out)
    except ValueError:
        raise RuntimeError(f"floor write target escapes DEV dir: {t}")
    return t


def _floor_atomic_write(target: Path, data: bytes) -> None:
    """Atomic write confined to DEV: unique .tmp -> fsync -> os.replace.

    A unique .tmp per writer avoids the Windows os.replace PermissionError
    when two writers share a .tmp name. The .tmp is unlinked in a finally
    block if the replace never happens (torn-write cleanup).
    """
    target = _floor_assert_not_live(target)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _floor_slot_stem(name: str) -> str:
    """Derive a DEV slot stem from a user name. Guards traversal + charset.

    ``name`` -> _safe_archive_name (bare basename, 1..255) -> MODEL_NAME_RE
    (^[A-Za-z0-9_-]+$) so the derived ``<stem>n.rel`` can't inject a suffix
    or extension. Returns ``map_dev<name>_00`` (the stock-shaped stem).
    """
    bare = _safe_archive_name(name)
    if not MODEL_NAME_RE.match(bare):
        raise HTTPException(400, "name: only letters, digits, '_' and '-' allowed")
    return f"{FLOOR_SLOT_PREFIX}{bare}_00"


def _floor_slot_files(out_dir: Path, stem: str) -> dict:
    """The three on-disk paths for a dev slot stem (n.rel, c.rel, xvm)."""
    return {
        "nrel": out_dir / f"{stem}n.rel",
        "crel": out_dir / f"{stem}c.rel",
        "xvm":  out_dir / f"{stem}s.xvm",
    }


def _floor_scan_slots(out_dir: Path) -> list:
    """Enumerate DEV floor slots: every ``<stem>n.rel`` under the dev dir.

    Skips dotfiles, backups (_is_backup_name), and *.tmp scratch. Returns
    a list of {stem, nrel_path, has_crel, has_xvm, source}.
    """
    out: list = []
    if not out_dir.exists():
        return out
    for p in sorted(out_dir.iterdir()):
        if not p.is_file():
            continue
        nm = p.name
        if nm.startswith(".") or nm.endswith(".tmp") or _is_backup_name(nm):
            continue
        if not nm.endswith("n.rel"):
            continue
        stem = nm[:-len("n.rel")]
        if not _FLOOR_SLOT_STEM_RE.match(stem):
            continue
        files = _floor_slot_files(out_dir, stem)
        out.append({
            "stem": stem,
            "nrel_path": files["nrel"],
            "has_crel": files["crel"].exists(),
            "has_xvm": files["xvm"].exists(),
        })
    return out


def _floor_slot_label(stem: str) -> str:
    """Human label for a dev slot stem: strip the prefix + trailing _NN."""
    core = stem
    if core.startswith(FLOOR_SLOT_PREFIX):
        core = core[len(FLOOR_SLOT_PREFIX):]
    core = re.sub(r"_\d{2}$", "", core)
    return core or stem


def _floor_part_count(nrel_path: Path) -> int:
    """Best-effort root-node count for a slot's n.rel (0 on any failure)."""
    try:
        from formats import rel as _rel  # lazy
        buf = nrel_path.read_bytes()
        if len(buf) > ASSET_PARSE_MAX_BYTES:
            return 0
        rel = _rel.parse_rel(buf)
        if not _rel.is_n_rel(rel):
            return 0
        return sum(1 for _ in _rel.iter_nrel_mesh_root_offsets(rel))
    except Exception:  # pragma: no cover — defensive
        return 0


def _floor_dev_record(out_dir: Path, slot: dict) -> dict:
    """Build a GET /api/floors record for one dev slot."""
    stem = slot["stem"]
    label = _floor_slot_label(stem)
    files = []
    for kind, key in (("n.rel", "nrel_path"),):
        files.append(slot[key].name)
    if slot["has_crel"]:
        files.append(_floor_slot_files(out_dir, stem)["crel"].name)
    if slot["has_xvm"]:
        files.append(_floor_slot_files(out_dir, stem)["xvm"].name)
    return {
        "floor_id": stem,
        "label": label,
        "area": "dev",
        "area_num": 0,
        "source": "copy",   # editable dev slot (copy or glb)
        "base_map_id": None,
        "floor_index": 0,
        "part_count": _floor_part_count(slot["nrel_path"]),
        "renderable_files": files,
        "thumb_url": None,
    }


def _floor_stock_records() -> list:
    """Enumerate read-only stock floors from the map catalogue.

    One record per (map_id, floor) tuple. These are Copy SOURCES only —
    never an overwrite/Create target.
    """
    out: list = []
    try:
        maps = _build_map_catalogue()
    except Exception:  # pragma: no cover — manifest may be absent in CI
        return out
    for mi in maps:
        for floor in sorted(mi.floors.keys()):
            out.append({
                "floor_id": f"{_FLOOR_STOCK_PREFIX}{mi.map_id}__{floor}",
                "label": f"{mi.label} — floor {floor}",
                "area": mi.area,
                "area_num": mi.area_num,
                "source": "stock",
                "base_map_id": mi.map_id,
                "floor_index": floor,
                "part_count": mi.renderable_files,
                "renderable_files": [a.path for a in mi.floors.get(floor, [])],
                "thumb_url": None,
            })
    return out


@app.get("/api/floors")
def api_floors_list():
    """Floor-browser payload: editable DEV slots + read-only stock floors.

    Mirrors the /api/map/list category list so the frontend can reuse
    map_panel's optgroup-by-category grouping. Degrades to an empty list
    (never 500) when neither the dev dir nor the manifest is present.
    """
    out_dir = _floor_resolve_out_dir()
    dev = [_floor_dev_record(out_dir, s) for s in _floor_scan_slots(out_dir)]
    stock = _floor_stock_records()
    return {
        "ok": True,
        "categories": [
            {"id": "dev",        "label": "Dev slots (editable)"},
            {"id": "city",       "label": "City / Pioneer 2"},
            {"id": "forest",     "label": "Forest"},
            {"id": "cave",       "label": "Cave"},
            {"id": "mine",       "label": "Mine"},
            {"id": "ruins",      "label": "Ruins"},
            {"id": "battle",     "label": "Battle (Versus)"},
            {"id": "corruption", "label": "Corruption / EP IV"},
            {"id": "boss",       "label": "Boss arena"},
            {"id": "other",      "label": "Other"},
        ],
        "floors": dev + stock,
    }


def _floor_dev_bundle(out_dir: Path, stem: str) -> dict:
    """Build a per-floor bundle for a DEV slot, shape-identical to
    /api/map/{id}?floor=N (formats.scene_loader.floor_bundle) so Preview
    can reuse psoSceneLoadMapWithEnvironment unchanged.

    Carries the fidelity-warning flags root_only_preview (R7) +
    single_texture_slot (R8) so the UI can banner them.
    """
    files = _floor_slot_files(out_dir, stem)
    nrel_path = files["nrel"]
    if not nrel_path.exists():
        raise HTTPException(404, f"floor slot not found: {stem}")
    nrel_rel = f"scene/{nrel_path.name}"  # synthetic manifest-relative path
    renderable = [{
        "path": nrel_rel,
        "kind": "rel_terrain",
        "ext": "rel",
        "suffix": "n",
        "size": nrel_path.stat().st_size,
    }]
    textures = []
    if files["xvm"].exists():
        textures.append({"path": f"scene/{files['xvm'].name}",
                         "size": files["xvm"].stat().st_size})
    scripts = []
    if files["crel"].exists():
        scripts.append({"path": f"scene/{files['crel'].name}", "kind": "collision",
                        "ext": "rel", "suffix": "c", "size": files["crel"].stat().st_size})
    # Best-effort: tell the user when the slot's n.rel has child sub-trees
    # the preview can't render (root-only read).
    root_only = False
    try:
        from formats import rel as _rel  # lazy
        rel = _rel.parse_rel(nrel_path.read_bytes())
        if _rel.is_n_rel(rel):
            roots = list(_rel.iter_nrel_mesh_root_offsets(rel))
            root_only = len(roots) > 0
    except Exception:  # pragma: no cover — defensive
        root_only = False
    return {
        "map_id": stem,
        "area": "dev",
        "area_num": 0,
        "floor": 0,
        "category": "other",
        "label": _floor_slot_label(stem),
        "rrel_path": None,
        "nrel_path": nrel_rel,
        "renderable": renderable,
        "textures": textures,
        "scripts": scripts,
        "animations": [],
        "other": [],
        "rrel_render_hints": None,
        "nrel_texture_names": _load_nrel_texture_names(nrel_rel),
        # Fidelity banners.
        "root_only_preview": root_only,
        "single_texture_slot": True,
    }


def _floor_resolve_dev_slot_for_read(out_dir: Path, stem: str) -> str:
    """Validate a dev-slot floor_id stem for a READ path (no traversal)."""
    bare = _validate_bare_filename(stem, label="floor_id")
    if not _FLOOR_SLOT_STEM_RE.match(bare):
        raise HTTPException(400, f"invalid floor_id {stem!r}")
    return bare


class FloorCopyReq(BaseModel):
    floor_id: Optional[str] = None      # source: stock id or dev stem
    src_name: Optional[str] = None      # alt source: DATA_DIR-relative file
    dest_name: Optional[str] = None     # new slot name (else derived)
    overwrite: bool = False
    mode: str = "passthrough"


@app.post("/api/floors/copy")
def api_floors_copy(req: FloorCopyReq):
    """Duplicate a floor into an editable DEV slot.

    Default mode 'passthrough' BYTE-COPIES the source n.rel/c.rel/.xvm so
    multi-node mesh trees are preserved on disk (avoids the R7 root-only
    loss). Atomic per file; .pre_edit_<TS> backup on overwrite; 409 on a
    no-overwrite clobber or a concurrent-build race. Writes confined to
    DEV, asserted not LIVE.
    """
    if not _FLOOR_BUILD_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another floor build is in progress; retry shortly")
    try:
        out_dir = _floor_resolve_out_dir()
        dest_stem = _floor_slot_stem(req.dest_name or req.floor_id or "")
        mode = (req.mode or "passthrough").lower()
        if mode not in ("passthrough", "reauthor"):
            raise HTTPException(400, "mode must be 'passthrough' or 'reauthor'")

        # Resolve the SOURCE triple (n.rel + optional c.rel/.xvm bytes).
        src_nrel, src_crel, src_xvm = _floor_copy_source_bytes(out_dir, req)

        dest = _floor_slot_files(out_dir, dest_stem)
        # Overwrite guard + backup.
        _floor_overwrite_guard(dest, req.overwrite)

        written = []
        # n.rel (required).
        if mode == "reauthor":
            nrel_bytes = _floor_reauthor_nrel(src_nrel)
        else:
            nrel_bytes = src_nrel
        _floor_atomic_write(dest["nrel"], nrel_bytes)
        written.append({"name": dest["nrel"].name, "size": len(nrel_bytes)})
        # c.rel + xvm passthrough (only meaningful in passthrough mode; a
        # re-author would need geometry we don't reconstruct here).
        if src_crel is not None:
            _floor_atomic_write(dest["crel"], src_crel)
            written.append({"name": dest["crel"].name, "size": len(src_crel)})
        if src_xvm is not None:
            _floor_atomic_write(dest["xvm"], src_xvm)
            written.append({"name": dest["xvm"].name, "size": len(src_xvm)})

        # Count dropped child nodes for the banner (passthrough preserves
        # them on disk; reauthor flattens to root-only).
        preview_root_only = True
        return {
            "ok": True,
            "new_floor_id": dest_stem,
            "label": _floor_slot_label(dest_stem),
            "mode": mode,
            "preview_root_only": preview_root_only,
            "files": written,
        }
    finally:
        _FLOOR_BUILD_LOCK.release()


def _floor_copy_source_bytes(out_dir: Path, req: "FloorCopyReq"):
    """Resolve the source floor's (n.rel, c.rel?, xvm?) bytes for a copy.

    The source is either a stock floor (floor_id 'stk__<map>__<n>', read
    via _resolve_scene_asset_path over the catalogue) or a dev slot
    (floor_id == a slot stem), or an explicit DATA_DIR-relative src_name
    (via safe_data_path). The n.rel MUST exist and parse as n.rel.
    """
    from formats import rel as _rel  # lazy

    nrel_bytes = crel_bytes = xvm_bytes = None
    fid = req.floor_id or ""

    if req.src_name:
        # Explicit DATA_DIR-relative source filename.
        p = safe_data_path(req.src_name)
        if not p.exists():
            raise HTTPException(404, f"source not found: {req.src_name}")
        nrel_bytes = p.read_bytes()
    elif fid.startswith(_FLOOR_STOCK_PREFIX):
        m = _FLOOR_STOCK_ID_RE.match(fid)
        if not m:
            raise HTTPException(400, f"invalid stock floor_id {fid!r}")
        map_id, floor = m.group(1), int(m.group(2))
        bundle = _floor_stock_bundle(map_id, floor)
        nrel_path = bundle.get("nrel_path")
        if not nrel_path:
            raise HTTPException(400, "source floor has no n.rel to copy")
        nrel_bytes = _resolve_scene_asset_path(nrel_path).read_bytes()
        # Best-effort siblings.
        crel_bytes = _floor_sibling_bytes(nrel_path, "c.rel")
        xvm_bytes = _floor_sibling_bytes_xvm(nrel_path)
    else:
        # Dev slot stem.
        stem = _floor_resolve_dev_slot_for_read(out_dir, fid)
        files = _floor_slot_files(out_dir, stem)
        if not files["nrel"].exists():
            raise HTTPException(404, f"floor slot not found: {stem}")
        nrel_bytes = files["nrel"].read_bytes()
        if files["crel"].exists():
            crel_bytes = files["crel"].read_bytes()
        if files["xvm"].exists():
            xvm_bytes = files["xvm"].read_bytes()

    # Validate the n.rel parses (don't copy garbage).
    try:
        rf = _rel.parse_rel(nrel_bytes)
        if not _rel.is_n_rel(rf):
            raise HTTPException(400, "source is not a valid n.rel")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"source n.rel failed to parse: {e}")
    return nrel_bytes, crel_bytes, xvm_bytes


def _floor_stock_bundle(map_id: str, floor: int) -> dict:
    """Return the scene_loader floor_bundle for a stock (map, floor)."""
    map_id = _safe_map_id(map_id)
    maps = _build_map_catalogue()
    info = next((m for m in maps if m.map_id == map_id), None)
    if info is None:
        raise HTTPException(404, f"unknown map_id {map_id!r}")
    if floor not in info.floors:
        raise HTTPException(404, f"map {map_id!r} has no floor {floor}")
    return _scene_loader.floor_bundle(info, floor)


def _floor_sibling_bytes(nrel_scene_path: str, suffix_ext: str):
    """Best-effort read of a sibling rel (e.g. swap n.rel -> c.rel)."""
    try:
        if not nrel_scene_path.endswith("n.rel"):
            return None
        sib = nrel_scene_path[:-len("n.rel")] + suffix_ext
        return _resolve_scene_asset_path(sib).read_bytes()
    except Exception:
        return None


def _floor_sibling_bytes_xvm(nrel_scene_path: str):
    """Best-effort read of the sibling .xvm for an n.rel scene path."""
    try:
        if not nrel_scene_path.endswith("n.rel"):
            return None
        base = nrel_scene_path[:-len("n.rel")]
        for cand in (base + "s.xvm", base + ".xvm"):
            try:
                return _resolve_scene_asset_path(cand).read_bytes()
            except Exception:
                continue
        return None
    except Exception:
        return None


def _floor_reauthor_nrel(src_nrel: bytes) -> bytes:
    """Re-author an n.rel from its ROOT-node meshes (R7: root-only).

    Used by copy mode='reauthor'. Loses child sub-trees by design — the
    default 'passthrough' mode avoids this. Runs the same verify gate.
    """
    from formats import rel as _rel  # lazy
    from formats import lobby_pipeline as _lp

    rf = _rel.parse_rel(src_nrel)
    meshes = _rel.extract_nrel_meshes(rf)
    names = _rel.read_texture_names(rf) or ["lobby"]
    out = _rw.build_nrel_from_meshes(
        _rw.nrel_nodes_from_meshes(meshes), names[:1] or ["lobby"],
        enforce_budget=True,
    )
    ok, msg = _lp.verify_nrel(out)
    if not ok:
        raise HTTPException(422, f"re-authored n.rel failed verification: {msg}")
    return out


def _floor_overwrite_guard(dest_files: dict, overwrite: bool) -> None:
    """Refuse to clobber an existing slot unless overwrite=True.

    On overwrite=True, keep a .pre_edit_<TS> backup of each existing dest
    file (so a prior dev build isn't lost). 409 on a no-overwrite clobber.
    """
    existing = [p for p in dest_files.values() if p.exists()]
    if existing and not overwrite:
        names = ", ".join(p.name for p in existing)
        raise HTTPException(409, f"slot exists ({names}); pass overwrite=true")
    if existing and overwrite:
        ts = time.strftime("%Y%m%d_%H%M%S")
        for p in existing:
            backup = p.with_name(f"{p.name}.pre_edit_{ts}")
            try:
                _floor_assert_not_live(backup)
                backup.write_bytes(p.read_bytes())
            except OSError:
                pass  # best-effort backup; the write still proceeds


@app.post("/api/floors/create")
async def api_floors_create(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    source_path: Optional[str] = Form(default=None),
    name: str = Form(...),
    area_template: str = Form(default="other"),
    tex: Optional[str] = Form(default=None),
    overwrite: bool = Form(default=False),
):
    """Author a floor from a GLB upload OR an imported-asset reference.

    GLB is magic-sniffed (b"glTF" + version 2 + length-field == len) BEFORE
    parse; body size capped at MAX_IMPORT_PARSE_BODY (64MB) up front. The
    build runs off the event loop. Over-budget n.rel/c.rel RAISE -> HTTP
    422 carrying the byte-count + budget string. simulate_rel_relocation
    failure -> 422. Atomic write confined to DEV (asserted not LIVE);
    .pre_edit_<TS> backup on overwrite. Returns a verify report.
    """
    _enforce_body_size(request, MAX_IMPORT_PARSE_BODY)

    # Resolve GLB bytes from EITHER an upload or a manifest asset ref.
    glb_bytes: bytes
    if file is not None and file.filename:
        glb_bytes = await file.read()
    elif source_path:
        glb_bytes = await asyncio.to_thread(_floor_read_source_asset, source_path)
    else:
        raise HTTPException(400, "provide either an uploaded file or source_path")

    if len(glb_bytes) == 0:
        raise HTTPException(400, "uploaded GLB is empty")
    if len(glb_bytes) > MAX_IMPORT_PARSE_BODY:
        raise HTTPException(413, f"GLB too large ({len(glb_bytes)} > {MAX_IMPORT_PARSE_BODY})")
    _floor_sniff_glb(glb_bytes)

    texname = _safe_archive_name(tex) if tex else "lobby"

    if not _FLOOR_BUILD_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another floor build is in progress; retry shortly")
    try:
        out_dir = _floor_resolve_out_dir()
        dest_stem = _floor_slot_stem(name)
        dest = _floor_slot_files(out_dir, dest_stem)
        _floor_overwrite_guard(dest, overwrite)

        # Author off the event loop. Maps RelWriteError -> 422, SystemExit
        # (decimate exhaustion) -> 422, ValueError (no geometry) -> 400.
        result = await asyncio.to_thread(_floor_build_sync, glb_bytes, texname)

        # Atomic writes (confined to DEV, asserted not LIVE).
        written = []
        _floor_atomic_write(dest["nrel"], result.nrel)
        written.append({"name": dest["nrel"].name, "size": len(result.nrel),
                        "budget_ok": len(result.nrel) <= _rw.NREL_SIZE_BUDGET})
        if result.crel is not None:
            _floor_atomic_write(dest["crel"], result.crel)
            written.append({"name": dest["crel"].name, "size": len(result.crel),
                            "budget_ok": len(result.crel) <= _rw.CREL_SIZE_BUDGET})
        if result.xvm is not None:
            _floor_atomic_write(dest["xvm"], result.xvm)
            written.append({"name": dest["xvm"].name, "size": len(result.xvm),
                            "budget_ok": True})

        rep = result.report
        return {
            "ok": True,
            "floor_id": dest_stem,
            "label": _floor_slot_label(dest_stem),
            "area_template": area_template,
            "report": {
                "part_count": rep.get("submesh_count", 0),
                "vertex_count": rep.get("tri_out", 0) * 3,
                "texture_count": rep.get("texture_count", 0),
                "bbox": [],
                "tri_in": rep.get("tri_in", 0),
                "tri_out": rep.get("tri_out", 0),
                "warnings": rep.get("warnings", []),
                "errors": rep.get("errors", []),
                "scale_applied": 1.0,
                "axis_flip": False,
                "dropped_child_nodes": rep.get("dropped_child_nodes", 0),
                "single_texture_slot": rep.get("single_texture_slot", True),
                "files": written,
            },
        }
    finally:
        _FLOOR_BUILD_LOCK.release()


def _floor_build_sync(glb_bytes: bytes, texname: str):
    """Worker-thread tail of create: run build_floor, map errors to HTTP."""
    from formats import lobby_pipeline as _lp
    try:
        return _lp.build_floor(glb_bytes, texname=texname)
    except _rw.RelWriteError as e:
        raise HTTPException(422, str(e))
    except SystemExit as e:
        raise HTTPException(422, f"could not fit floor under budget: {e}")
    except ValueError as e:
        raise HTTPException(400, f"GLB has no usable geometry: {e}")
    except RuntimeError as e:
        raise HTTPException(422, f"authored floor failed verification: {e}")


def _floor_sniff_glb(data: bytes) -> None:
    """Validate the binary-GLB magic + header. 400 on any mismatch.

    A binary glTF starts with a 12-byte header: magic 'glTF' (0x46546C67
    LE), uint32 version (==2), uint32 total length (== len(data)). We do
    NOT trust the client Content-Type or filename extension.
    """
    if len(data) < 12 or data[:4] != b"glTF":
        raise HTTPException(400, "not a binary GLB (missing 'glTF' magic)")
    version, total_len = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise HTTPException(400, f"unsupported GLB version {version} (need 2)")
    if total_len != len(data):
        raise HTTPException(400, f"GLB length field {total_len} != actual {len(data)}")


def _floor_read_source_asset(source_path: str) -> bytes:
    """Read a manifest model-asset by its DATA_DIR-relative path (no traversal)."""
    # Accept either a flat filename (safe_data_path) or a scene/ path.
    try:
        if source_path.startswith("scene/"):
            return _resolve_scene_asset_path(source_path).read_bytes()
        return safe_data_path(source_path).read_bytes()
    except HTTPException:
        raise
    except OSError as e:
        raise HTTPException(404, f"source asset not found: {e}")


@app.get("/api/floors/{floor_id}")
def api_floors_get(floor_id: str):
    """Per-floor asset bundle, shape-identical to /api/map/{id}?floor=N.

    For a stock floor_id ('stk__<map>__<n>') this delegates to the map
    bundle path; for a dev slot stem it builds a synthetic bundle over the
    slot files in DEV_DATA_DIR. Both carry the root_only_preview /
    single_texture_slot fidelity flags. Resolved via guards that reject
    traversal.
    """
    if floor_id.startswith(_FLOOR_STOCK_PREFIX):
        m = _FLOOR_STOCK_ID_RE.match(floor_id)
        if not m:
            raise HTTPException(400, f"invalid stock floor_id {floor_id!r}")
        bundle = _floor_stock_bundle(m.group(1), int(m.group(2)))
        bundle["rrel_render_hints"] = _load_rrel_hints(bundle.get("rrel_path"))
        bundle["nrel_texture_names"] = _load_nrel_texture_names(bundle.get("nrel_path"))
        bundle.setdefault("root_only_preview", False)
        bundle.setdefault("single_texture_slot", False)
        return bundle
    out_dir = _floor_resolve_out_dir()
    stem = _floor_resolve_dev_slot_for_read(out_dir, floor_id)
    return _floor_dev_bundle(out_dir, stem)


@app.delete("/api/floors/{floor_id}")
def api_floors_delete(floor_id: str):
    """Delete a DEV slot (copies/glb only). Stock floors are not deletable."""
    if floor_id.startswith(_FLOOR_STOCK_PREFIX):
        raise HTTPException(400, "stock floors cannot be deleted")
    out_dir = _floor_resolve_out_dir()
    stem = _floor_resolve_dev_slot_for_read(out_dir, floor_id)
    files = _floor_slot_files(out_dir, stem)
    removed = []
    for p in files.values():
        q = _floor_assert_not_live(p)
        if q.exists():
            try:
                q.unlink()
                removed.append(q.name)
            except OSError as e:
                raise HTTPException(500, f"could not delete {q.name}: {e}")
    if not removed:
        raise HTTPException(404, f"floor slot not found: {stem}")
    return {"ok": True, "deleted": removed}


# ===========================================================================
# Archive entry editor (2026-06-20) — duplicate / create / delete / rename
# an inner entry inside a container archive (AFS or BML).
#
# Write model: edits write BACK to the path the archive was OPENED from.
# We resolve the archive with the SAME dual-root resolver the readers use
# (DATA_DIR first, then LIVE_DATA_DIR) via _resolve_under_roots, and write
# the rewritten archive to THAT resolved location — DATA_DIR or LIVE,
# wherever it was found. This follows the legacy /api/repack_afs_inner +
# /api/repack_bml_inner write-target behaviour (they too write the
# resolved, possibly-live, path), NOT the floor editor's DEV-only boundary.
#
# Safety kept: (1) atomic write (<target>.tmp -> fsync -> os.replace so a
# crash can't leave a half-written archive), and (2) a .pre_edit_<ts>
# backup of the target before overwriting. Per-archive lock (_REPACK_LOCKS)
# serialises concurrent edits of the same archive; the inner cache is
# invalidated (afs_reader.cache_dir_for rmtree) so subsequent
# "<archive>#NNNN" reads see the new bytes; the frontend refreshes via
# GET /api/manifest?force=1.
# ===========================================================================
from formats import archive_entry as _archive_entry


def _archive_resolve_for_write(archive: str) -> Tuple[Path, str]:
    """Resolve ``archive`` (bare filename) under DATA_DIR then LIVE_DATA_DIR.

    Returns ``(resolved_path, kind)`` where ``kind`` is ``"afs"`` / ``"bml"``.
    The resolved path is the actual file location the edit writes back to.

    Raises HTTPException: 400 (bad name / unsupported container) or 404
    (archive not found in either root).
    """
    bare = _validate_bare_filename(archive, label="archive")
    kind = _archive_entry.archive_kind(bare)
    if kind is None:
        raise HTTPException(
            400,
            f"duplicate/create/delete/rename not supported for this container: {bare}",
        )
    target = _resolve_under_roots(
        bare,
        (DATA_DIR, LIVE_DATA_DIR),
        label="archive",
        missing_msg=f"archive not found in DATA_DIR or LIVE_DATA_DIR: {bare}",
    )
    return target, kind


def _archive_backup_and_write(target: Path, new_bytes: bytes) -> Optional[str]:
    """Back up ``target`` (.pre_edit_<ts>) then atomically overwrite it.

    Returns the backup path string (or ``None`` if the target didn't exist
    yet, which shouldn't happen since we only edit existing archives).
    The write is <target>.tmp -> fsync -> os.replace so a crash can never
    leave a torn archive on disk.
    """
    backup_path: Optional[str] = None
    if target.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        bak = target.with_name(f"{target.name}.pre_edit_{ts}")
        counter = 0
        while bak.exists():
            counter += 1
            bak = target.with_name(f"{target.name}.pre_edit_{ts}_{counter}")
        try:
            shutil.copy(target, bak)
            backup_path = str(bak)
        except OSError as e:
            raise HTTPException(500, f"backup failed: {e}")
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(new_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except OSError as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise HTTPException(500, f"archive write failed: {e}")
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return backup_path


def _archive_invalidate_inner_cache(target: Path) -> None:
    """Drop the cached materialised inner blobs for an edited AFS archive."""
    if target.suffix.lower() != ".afs":
        return
    try:
        from formats import afs_reader as _afs_reader
        inner_cache = _afs_reader.cache_dir_for(target, CACHE_DIR)
        if inner_cache.exists():
            shutil.rmtree(inner_cache, ignore_errors=True)
    except OSError as e:
        log.debug("could not invalidate inner cache for %s: %s", target.name, e)


def _archive_new_path_key(archive: str, kind: str, *,
                          new_index: Optional[int] = None,
                          new_entry_name: Optional[str] = None,
                          inner_name: Optional[str] = None) -> Optional[str]:
    """Build the manifest addressing key for a new/renamed entry.

    AFS inner entries are positional: ``<archive>#<NNNN>_<inner_name>``
    (matches manifest._synthesize_afs_entries). BML entries are
    name-addressed: ``<archive>#<entry_name>``.
    """
    if kind == "afs" and new_index is not None:
        suffix = f"_{inner_name}" if inner_name else ""
        return f"{archive}#{new_index:04d}{suffix}"
    if kind == "bml" and new_entry_name is not None:
        return f"{archive}#{new_entry_name}"
    return None


def _archive_afs_inner_name(target: Path, index: int) -> Optional[str]:
    """Best-effort display name for AFS slot ``index`` (for new_path)."""
    try:
        from formats import afs_reader as _afs_reader
        rows = _afs_reader.list_inner_blobs(target)
        for r in rows:
            if r.get("index") == index:
                return r.get("name")
    except (OSError, ValueError) as e:
        log.debug("could not derive inner name for %s#%d: %s", target.name, index, e)
    return None


def _archive_map_pure_error(e: Exception) -> "HTTPException":
    """Map a pure-layer exception to the HTTP status the spec mandates.

    ValueError -> 422; KeyError / IndexError -> 404. ValueErrors that read
    as a name collision / "no name table" rename are surfaced as 409.
    """
    if isinstance(e, (KeyError, IndexError)):
        return HTTPException(404, str(e).strip("'\""))
    if isinstance(e, ValueError):
        msg = str(e)
        low = msg.lower()
        if "already exists" in low or "no filename table" in low:
            return HTTPException(409, msg)
        return HTTPException(422, msg)
    return HTTPException(500, str(e))


class ArchiveDuplicateReq(BaseModel):
    archive: str
    index: Optional[int] = Field(default=None, ge=0, le=0xFFFF)  # AFS
    entry_name: Optional[str] = None                              # BML
    new_name: Optional[str] = None


class ArchiveDeleteReq(BaseModel):
    archive: str
    index: Optional[int] = Field(default=None, ge=0, le=0xFFFF)  # AFS
    entry_name: Optional[str] = None                              # BML


class ArchiveRenameReq(BaseModel):
    archive: str
    index: Optional[int] = Field(default=None, ge=0, le=0xFFFF)  # AFS
    entry_name: Optional[str] = None                              # BML
    new_name: str


class ArchiveCreateJsonReq(BaseModel):
    """JSON body for AFS template create (no blob upload)."""
    archive: str
    new_name: Optional[str] = None
    template: str = "empty"  # "empty" | "copy_first"


def _archive_edit_locked(target: Path, kind: str, mutate):
    """Run ``mutate(buf) -> (new_bytes, result_dict)`` under the archive lock.

    Reads the resolved target, applies the pure mutation, atomically writes
    back with a backup, invalidates the inner cache, and merges the result.
    Maps pure-layer errors to HTTP status codes.
    """
    lock_key = target.name
    lk = _get_lock(_REPACK_LOCKS, lock_key, MAX_REPACK_LOCKS)
    if not lk.acquire(blocking=False):
        raise HTTPException(409, f"archive edit already in progress for {target.name}")
    try:
        try:
            buf = target.read_bytes()
        except OSError as e:
            raise HTTPException(500, f"archive read failed: {e}")
        original_size = len(buf)
        try:
            new_bytes, extra = mutate(buf)
        except HTTPException:
            raise
        except (ValueError, KeyError, IndexError) as e:
            raise _archive_map_pure_error(e)
        backup_path = _archive_backup_and_write(target, new_bytes)
        _archive_invalidate_inner_cache(target)
        out = {
            "ok": True,
            "archive": target.name,
            "kind": kind,
            "target_path": str(target),
            "original_size": original_size,
            "new_size": len(new_bytes),
            "backup_path": backup_path,
        }
        out.update(extra)
        return out
    finally:
        lk.release()


@app.post("/api/archive/duplicate_entry")
def api_archive_duplicate_entry(req: ArchiveDuplicateReq, request: Request):
    """Duplicate one inner entry inside an AFS / BML archive (writes back).

    AFS: pass ``index`` (positional). BML: pass ``entry_name`` +
    ``new_name``. The rewritten archive is written to the resolved source
    path (DATA_DIR or LIVE) with a .pre_edit backup. Returns the new
    addressing key so the frontend can open the duplicate immediately.
    """
    _enforce_body_size(request, MAX_REPACK_BODY)
    target, kind = _archive_resolve_for_write(req.archive)

    if kind == "afs":
        if req.index is None:
            raise HTTPException(400, "AFS duplicate requires 'index'")
        idx = req.index

        def mutate(buf):
            new_buf, new_index = _archive_entry.afs_duplicate(buf, idx)
            return new_buf, {"new_index": new_index}

        result = _archive_edit_locked(target, kind, mutate)
        inner_name = _archive_afs_inner_name(target, result["new_index"])
        result["new_path"] = _archive_new_path_key(
            target.name, kind, new_index=result["new_index"], inner_name=inner_name)
        return result

    # BML
    if not req.entry_name:
        raise HTTPException(400, "BML duplicate requires 'entry_name'")
    _validate_inner_name(req.entry_name, msg="invalid entry_name", required=True)
    new_name = req.new_name or f"{req.entry_name}_copy"
    _validate_inner_name(new_name, msg="invalid new_name", required=True)

    def mutate(buf):
        new_buf = _archive_entry.bml_duplicate(buf, req.entry_name, new_name)
        return new_buf, {"new_entry_name": new_name}

    result = _archive_edit_locked(target, kind, mutate)
    result["new_path"] = _archive_new_path_key(
        target.name, kind, new_entry_name=new_name)
    return result


@app.post("/api/archive/create_entry")
async def api_archive_create_entry(
    request: Request,
    archive: str = Form(...),
    new_name: Optional[str] = Form(default=None),
    template: Optional[str] = Form(default=None),
    is_compressed: bool = Form(default=False),
    source_path: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    """Create a new inner entry from an uploaded blob, a DATA_DIR ref, or a
    template ("empty" / "copy_first"). Multipart form.

    AFS: blob OR template; ``new_name`` only used if the archive carries a
    name table. BML: requires ``new_name`` and a blob (upload or
    source_path); ``is_compressed`` marks an already-PRS blob.
    """
    _enforce_body_size(request, MAX_REPACK_BODY)
    target, kind = _archive_resolve_for_write(archive)

    blob: Optional[bytes] = None
    if file is not None and file.filename:
        blob = await file.read()
    elif source_path:
        blob = await asyncio.to_thread(_archive_read_source_blob, source_path)
    if blob is not None and len(blob) > MAX_REPACK_BODY:
        raise HTTPException(413, f"blob too large ({len(blob)} > {MAX_REPACK_BODY})")

    if kind == "afs":
        def mutate(buf):
            new_buf, new_index = _archive_entry.afs_create(
                buf, blob, template, new_name=new_name)
            return new_buf, {"new_index": new_index}

        result = _archive_edit_locked(target, kind, mutate)
        inner_name = _archive_afs_inner_name(target, result["new_index"])
        result["new_path"] = _archive_new_path_key(
            target.name, kind, new_index=result["new_index"], inner_name=inner_name)
        return result

    # BML: a name + blob is required (no meaningful empty BML entry).
    if not new_name:
        raise HTTPException(400, "BML create requires 'new_name'")
    _validate_inner_name(new_name, msg="invalid new_name", required=True)
    if blob is None:
        raise HTTPException(400, "BML create requires an uploaded file or source_path")
    blob_bytes = blob

    def mutate(buf):
        new_buf = _archive_entry.bml_create(
            buf, new_name, blob_bytes, is_compressed=is_compressed)
        return new_buf, {"new_entry_name": new_name}

    result = _archive_edit_locked(target, kind, mutate)
    result["new_path"] = _archive_new_path_key(
        target.name, kind, new_entry_name=new_name)
    return result


def _archive_read_source_blob(source_path: str) -> bytes:
    """Read a DATA_DIR / LIVE blob referenced by ``source_path`` for create."""
    try:
        p = _resolve_under_roots(
            source_path, (DATA_DIR, LIVE_DATA_DIR),
            label="source_path",
            missing_msg=f"source blob not found: {source_path}",
        )
        return p.read_bytes()
    except HTTPException:
        raise
    except OSError as e:
        raise HTTPException(404, f"source blob not found: {e}")


@app.delete("/api/archive/entry")
def api_archive_delete_entry(req: ArchiveDeleteReq, request: Request):
    """Delete one inner entry (AFS ``index`` or BML ``entry_name``).

    AFS deletes renumber every later slot; the frontend must refresh the
    manifest after. Writes back to the resolved source path with a backup.
    """
    _enforce_body_size(request, MAX_REPACK_BODY)
    target, kind = _archive_resolve_for_write(req.archive)

    if kind == "afs":
        if req.index is None:
            raise HTTPException(400, "AFS delete requires 'index'")
        idx = req.index

        def mutate(buf):
            return _archive_entry.afs_delete(buf, idx), {"deleted": f"#{idx:04d}"}

        return _archive_edit_locked(target, kind, mutate)

    if not req.entry_name:
        raise HTTPException(400, "BML delete requires 'entry_name'")
    _validate_inner_name(req.entry_name, msg="invalid entry_name", required=True)

    def mutate(buf):
        return _archive_entry.bml_delete(buf, req.entry_name), {"deleted": req.entry_name}

    return _archive_edit_locked(target, kind, mutate)


@app.post("/api/archive/rename_entry")
def api_archive_rename_entry(req: ArchiveRenameReq, request: Request):
    """Rename one inner entry.

    AFS rename is allowed ONLY when the archive carries a real filename
    table (else 409 — renaming a no-table AFS would change byte layout).
    BML entries are always name-addressed and renamable.
    """
    _enforce_body_size(request, MAX_REPACK_BODY)
    target, kind = _archive_resolve_for_write(req.archive)
    if not req.new_name:
        raise HTTPException(400, "rename requires 'new_name'")

    if kind == "afs":
        if req.index is None:
            raise HTTPException(400, "AFS rename requires 'index'")
        idx = req.index

        def mutate(buf):
            new_buf = _archive_entry.afs_rename(buf, idx, req.new_name)
            return new_buf, {"new_entry_name": req.new_name}

        result = _archive_edit_locked(target, kind, mutate)
        inner_name = _archive_afs_inner_name(target, idx)
        result["new_path"] = _archive_new_path_key(
            target.name, kind, new_index=idx, inner_name=inner_name)
        return result

    if not req.entry_name:
        raise HTTPException(400, "BML rename requires 'entry_name'")
    _validate_inner_name(req.entry_name, msg="invalid entry_name", required=True)
    _validate_inner_name(req.new_name, msg="invalid new_name", required=True)

    def mutate(buf):
        new_buf = _archive_entry.bml_rename(buf, req.entry_name, req.new_name)
        return new_buf, {"new_entry_name": req.new_name}

    result = _archive_edit_locked(target, kind, mutate)
    result["new_path"] = _archive_new_path_key(
        target.name, kind, new_entry_name=req.new_name)
    return result


@app.get("/api/archive/{name}/entries")
def api_archive_entries(name: str):
    """Read-only list of an archive's inner entries (for the entry editor).

    Returns ``{ok, archive, kind, supported, entries:[...]}``. GSL /
    unknown containers return ``supported=false`` with an empty list so
    the frontend can disable the controls with a clear note.
    """
    bare = _validate_bare_filename(name, label="archive")
    kind = _archive_entry.archive_kind(bare)
    if kind is None:
        return {
            "ok": True, "archive": bare, "kind": None, "supported": False,
            "note": "entry editing is not supported for this container",
            "entries": [],
        }
    target = _resolve_under_roots(
        bare, (DATA_DIR, LIVE_DATA_DIR), label="archive",
        missing_msg=f"archive not found in DATA_DIR or LIVE_DATA_DIR: {bare}",
    )
    entries: list = []
    if kind == "afs":
        from formats import afs_reader as _afs_reader
        try:
            rows = _afs_reader.list_inner_blobs(target)
        except (OSError, ValueError) as e:
            raise HTTPException(400, f"AFS parse failed: {e}")
        for r in rows:
            entries.append({
                "index": r.get("index"),
                "name": r.get("name"),
                "size": r.get("size"),
                "inner_format": r.get("inner_format"),
                "compressed": r.get("compressed"),
                "path": f"{bare}#{r.get('index'):04d}_{r.get('name')}",
            })
    else:  # bml
        try:
            buf = target.read_bytes()
            pack_entries = _archive_entry.parse_bml_for_pack(buf)
        except (OSError, ValueError) as e:
            raise HTTPException(400, f"BML parse failed: {e}")
        for i, ent in enumerate(pack_entries):
            entries.append({
                "index": i,
                "name": ent.name,
                "size": len(ent.data),
                "inner_format": "NJ_IFF",
                "compressed": ent.is_compressed,
                "has_texture": ent.texture_data is not None and len(ent.texture_data) > 0,
                "path": f"{bare}#{ent.name}",
            })
    return {
        "ok": True, "archive": bare, "kind": kind, "supported": True,
        "target_path": str(target), "entries": entries,
    }


# ===========================================================================
# Endpoints added 2026-04-25 (finishing-line polish batch).
# Kept in a single block at the bottom of server.py to minimize merge
# surface with parallel agents that may also be editing earlier sections.
#
# Currently:
#   POST /api/import/animation/swap   — Item 1 (motion-slot override)
#   GET  /api/binding_cache/disk_stats — Item 5 (disk-cache visibility)
# ===========================================================================


class ImportAnimationSwapReq(BaseModel):
    njm_path: str
    target_bml: str
    target_inner_to_replace: str
    output_name: Optional[str] = None


@app.post("/api/import/animation/swap")
def api_import_animation_swap(req: ImportAnimationSwapReq, request: Request):
    """Strict-replace splice: overwrite an existing inner motion slot.

    EXPLICIT-OPT-IN ROUTE. As of 2026-04-25 the **default** animation-import
    flow is editor preview-only — see ``/api/import/animation`` (which
    stages an .njm + .preview.json sidecar) and the Motions-tab
    "Imported Animations" section in ``static/texture_panel.js`` (which
    plays the staged animation in the viewport via ``psoLoadMotion``
    without ever writing into ``<install>/data/``).

    This endpoint is for users who **explicitly want** to overwrite a
    real game motion slot. Use cases: shipping a true mod that ships
    `<install>/data/NpcApcMot.bml` with a custom motion. The output is
    staged in ``cache/bml_export/`` and only goes live when the user
    deliberately calls ``/api/deploy/<bml>``.

    Mirrors ``/api/import/animation/replace`` but with two key differences:
      - no append fallback; ``target_inner_to_replace`` MUST exist in the
        target BML (404 otherwise). Typos silently appending a fresh
        entry would never be referenced by the game's NPC controller —
        which loads motions by inner name — so we fail loudly.
      - the new entry preserves the original entry's name AND its
        ``unk_a/b/c/d`` fields, so a parsed-BML round-trip continues to
        match. This is critical for the lobby-girl motion-override
        workflow where we keep the slot name exactly as the game expects
        but redirect the underlying bytes to a synthesized animation.

    Body
    ----
    njm_path
        Filename in ``cache/njm_export/`` (e.g. ``"lobby_girl_typing.njm"``).
    target_bml
        Filename of the BML to overwrite (looked up in LIVE_DATA_DIR →
        DATA_DIR → DEV_DATA_DIR, in that order).
    target_inner_to_replace
        Existing inner-entry name to overwrite (e.g.
        ``"pxuG01_A06_W_body.njm"``).
    output_name
        Optional output filename in ``cache/bml_export/``. Defaults to
        ``target_bml`` (overwriting the prior staged build).
    """
    _enforce_body_size(request, MAX_ANIM_BODY)
    njm_name = _safe_archive_name(Path(req.njm_path).name)
    bml_name = _safe_archive_name(Path(req.target_bml).name)
    target_inner = req.target_inner_to_replace.strip()
    if not target_inner or "/" in target_inner or "\\" in target_inner:
        raise HTTPException(
            400, "target_inner_to_replace: must be a bare entry name",
        )
    njm_p = NJM_EXPORT_DIR / njm_name
    if not njm_p.exists():
        raise HTTPException(
            404, f"njm_path not found in cache/njm_export: {njm_name}",
        )
    src_bml = LIVE_DATA_DIR / bml_name
    if not src_bml.exists():
        src_bml = DATA_DIR / bml_name
        if not src_bml.exists():
            src_bml = DEV_DATA_DIR / bml_name
            if not src_bml.exists():
                raise HTTPException(404, f"target_bml not found: {bml_name}")
    bml_bytes = src_bml.read_bytes()
    try:
        from formats.bml import parse_bml_for_pack, parse_bml_pack_meta
        pack_entries = parse_bml_for_pack(bml_bytes)
        meta = parse_bml_pack_meta(bml_bytes)
    except Exception as e:
        raise HTTPException(500, f"BML parse failed: {e}")

    new_data = njm_p.read_bytes()
    matched = -1
    for i, ent in enumerate(pack_entries):
        if ent.name == target_inner:
            matched = i
            break
    if matched < 0:
        names = [e.name for e in pack_entries]
        # Truncate the names list in the error so a 1000-entry BML
        # doesn't blow up the error body — first 20 names is plenty for
        # the user to spot a typo.
        raise HTTPException(
            404,
            f"target_inner_to_replace {target_inner!r} not in BML; "
            f"have ({len(names)} entries; first 20): {names[:20]}",
        )
    original = pack_entries[matched]
    pack_entries[matched] = BmlPackEntry(
        name=original.name,                     # preserve exact slot name
        data=new_data,
        decompressed_size=len(new_data),
        is_compressed=False,
        texture_data=original.texture_data,
        texture_decompressed_size=original.texture_decompressed_size,
        texture_is_compressed=original.texture_is_compressed,
        unk_a=original.unk_a,
        unk_b=original.unk_b,
        unk_c=original.unk_c,
        unk_d=original.unk_d,
    )
    try:
        out = pack_bml(
            pack_entries,
            compression=meta["compression"],
            has_textures_override=bool(meta.get("has_textures", False)),
            file_alignment=meta["file_alignment"],
        )
    except Exception as e:
        raise HTTPException(500, f"pack_bml failed: {e}")
    out_name = _safe_archive_name(req.output_name or bml_name)
    BML_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BML_EXPORT_DIR / out_name
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(out)
    os.replace(tmp, out_path)
    md5 = _md5_bytes(out)
    log.info(
        "import/animation/swap %s#%s <- %s -> %s (%d bytes, md5=%s)",
        bml_name, target_inner, njm_name, out_path, len(out), md5,
    )
    return {
        "ok": True,
        "archive_path": str(out_path),
        "archive_name": out_name,
        "size": len(out),
        "md5": md5,
        "operation": "replaced",
        "target_inner": target_inner,
        "index": matched,
        "original_size_compressed": len(original.data),
        "new_size_compressed": len(new_data),
    }


# ============================================================================
# Animation preview endpoints (2026-04-25)
# ----------------------------------------------------------------------------
# Editor-only animation playback for imported glTF/FBX motions. The flow:
#
#   1. User drops a .glb/.gltf via /api/import/animation. Server retargets
#      onto the target BML's skeleton, stages an .njm in cache/njm_export/,
#      AND writes a .preview.json sidecar tagging the .njm with its target
#      model_path.
#   2. Editor's Motions tab calls /api/anim_preview/list?model_path=<bml>
#      to enumerate every staged .njm whose sidecar's target_model_path
#      matches. Each entry is rendered in a separate "Imported Animations"
#      section with a "(imported)" badge, distinct from the Movement /
#      Combat / Damage / Death / Idle groups (which come from the model's
#      own /api/animations listing).
#   3. Click → /api/anim_preview/data?njm_path=<filename> returns the
#      parsed motion JSON in the same wire shape as /api/animation_data,
#      so the frontend can hand it directly to the existing playback
#      pipeline (psoLoadMotion).
#   4. NO BML repacking, NO write into <install>/data/, NO interaction
#      with the model's own motion list. The game stays vanilla; the
#      imported animation is purely a viewport preview.
#
# Sidecar shape (cache/njm_export/<safe>.njm.preview.json):
#     {
#       "target_model_path":  "bm_npc_kenkyu_w.bml",
#       "target_inner":       "kenkyu_w_hone_body.nj",
#       "source_glb":         "standing_typing.glb",
#       "source_animation":   "StandingTyping",
#       "retargeted_at_ms":   1729900000000,
#       "retargeted_bones":   22,
#       "dropped_bones":      0,
#       "frame_count":        90,
#       "bone_count":         64,
#       "fps":                30,
#       "bone_map":           "lobby_girl",
#       "njm_md5":            "<hex>"
#     }
# ============================================================================


def _safe_njm_export_name(name: str) -> str:
    """Validate a filename to lookup in NJM_EXPORT_DIR.

    Bare basename only — no traversal, no separators. Matches the
    sanitisation done by ``_safe_motion_name`` on import so any name
    that came back from ``/api/import/animation`` is accepted as-is.
    """
    if not isinstance(name, str) or not name:
        raise HTTPException(400, "njm_path: missing or non-string")
    bare = Path(name).name
    if bare != name:
        raise HTTPException(400, "njm_path: must be a bare filename")
    if not re.match(r"^[A-Za-z0-9_\-.]+$", bare):
        raise HTTPException(400, "njm_path: contains forbidden characters")
    if not bare.lower().endswith(".njm"):
        raise HTTPException(400, "njm_path: must end with .njm")
    return bare


def _load_preview_sidecar(njm_path: Path) -> Optional[dict]:
    """Read the .preview.json sidecar adjacent to a staged .njm.

    Returns None if no sidecar exists (the .njm came from a non-preview
    workflow such as /api/anim_keyframe/save or a manual drop). The
    list endpoint silently skips njms without sidecars.
    """
    sc = njm_path.parent / (njm_path.name + ".preview.json")
    if not sc.is_file():
        return None
    try:
        return json.loads(sc.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("anim_preview: bad sidecar %s: %s", sc, e)
        return None


@app.get("/api/anim_preview/list")
def api_anim_preview_list(model_path: str):
    """List preview-only imported animations targeting a given model.

    Returns one entry per staged .njm in ``cache/njm_export/`` whose
    .preview.json sidecar names ``model_path`` as its target. The
    ``model_path`` query param is matched case-insensitively against the
    sidecar's ``target_model_path`` (basename comparison — full paths
    with directory components are stripped first).

    Wire shape::

        {
          "model_path": "<input>",
          "count": <int>,
          "items": [
            {
              "name":             "<njm filename>",   // primary key
              "display_name":     "<njm minus .njm>",
              "frame_count":      <int>,
              "bone_count":       <int>,
              "fps":              <float>,
              "source_glb":       "<filename>",
              "source_animation": "<name in glb>",
              "retargeted_at_ms": <epoch ms>,
              "retargeted_bones": <int>,
              "dropped_bones":    <int>,
              "njm_md5":          "<hex>",
              "size":             <bytes>
            }, ...
          ]
        }

    Returns 200 with an empty list when nothing matches (rather than
    404) so the UI can degrade gracefully.
    """
    if not isinstance(model_path, str) or not model_path:
        raise HTTPException(400, "model_path: missing or non-string")
    target_basename = Path(model_path).name.lower()
    if not target_basename:
        raise HTTPException(400, "model_path: empty basename")

    items: list[dict] = []
    if NJM_EXPORT_DIR.is_dir():
        for njm in sorted(NJM_EXPORT_DIR.glob("*.njm")):
            sidecar = _load_preview_sidecar(njm)
            if sidecar is None:
                continue
            sc_target = (sidecar.get("target_model_path") or "").strip()
            if not sc_target:
                continue
            if Path(sc_target).name.lower() != target_basename:
                continue
            try:
                size = njm.stat().st_size
            except OSError:
                continue
            display = njm.name
            if display.lower().endswith(".njm"):
                display = display[:-4]
            items.append({
                "name": njm.name,
                "display_name": display,
                "frame_count": int(sidecar.get("frame_count") or 0),
                "bone_count": int(sidecar.get("bone_count") or 0),
                "fps": float(sidecar.get("fps") or 30.0),
                "source_glb": sidecar.get("source_glb") or "",
                "source_animation": sidecar.get("source_animation") or "",
                "retargeted_at_ms": int(sidecar.get("retargeted_at_ms") or 0),
                "retargeted_bones": int(sidecar.get("retargeted_bones") or 0),
                "dropped_bones": int(sidecar.get("dropped_bones") or 0),
                "njm_md5": sidecar.get("njm_md5") or "",
                "size": size,
            })
    # Most-recent first — matches the user's mental model of "I just
    # dropped this thing in, where is it?"
    items.sort(key=lambda it: it["retargeted_at_ms"], reverse=True)
    return {
        "model_path": model_path,
        "count": len(items),
        "items": items,
    }


@app.get("/api/anim_preview/data")
def api_anim_preview_data(njm_path: str):
    """Return parsed keyframes for one preview-only imported animation.

    The wire shape mirrors ``/api/animation_data`` so the frontend can
    pass the response directly to the existing playback pipeline
    (psoLoadMotion). Notably we DON'T require a ``?motion=`` query: the
    .njm is single-motion (one retarget = one motion), so the caller
    only needs to supply the staged filename.

    Returns 404 if ``njm_path`` is not in ``cache/njm_export/`` or the
    file's not parseable as NJM.
    """
    name = _safe_njm_export_name(njm_path)
    p = NJM_EXPORT_DIR / name
    if not p.is_file():
        raise HTTPException(404, f"njm_path not found in cache/njm_export: {name}")
    try:
        njm_bytes = p.read_bytes()
    except OSError as e:
        raise HTTPException(500, f"read failed: {e}")
    try:
        parsed = _njm_parse(njm_bytes)
    except ValueError as e:
        raise HTTPException(400, f"NJM parse failed: {e}")
    if not parsed:
        raise HTTPException(400, "NJM had no motion data")
    m = parsed[0]

    sidecar = _load_preview_sidecar(p)
    target_path = (sidecar or {}).get("target_model_path") or ""

    # Project to the same wire shape as /api/animation_data.
    present_per_bone = m.bone_present_tracks or []
    bones_out: list[dict] = []
    for b_idx, track in enumerate(m.tracks):
        present = present_per_bone[b_idx] if b_idx < len(present_per_bone) else 0
        if not track:
            bones_out.append({"idx": b_idx, "kf": [], "present": present})
            continue
        kf_out: list[dict] = []
        for kf in track:
            entry: dict = {
                "t": kf.time,
                "tx": kf.tx, "ty": kf.ty, "tz": kf.tz,
                "rx": kf.rx_bams, "ry": kf.ry_bams, "rz": kf.rz_bams,
                "sx": kf.sx, "sy": kf.sy, "sz": kf.sz,
            }
            if kf.qw is not None:
                entry["qw"] = kf.qw
                entry["qx"] = kf.qx
                entry["qy"] = kf.qy
                entry["qz"] = kf.qz
            kf_out.append(entry)
        bones_out.append({"idx": b_idx, "kf": kf_out, "present": present})

    name_disp = name[:-4] if name.lower().endswith(".njm") else name
    fps_hint = float((sidecar or {}).get("fps") or _njm_guess_fps(name_disp))
    return {
        "filename": name,
        "motion": name_disp,
        "motion_index": 0,
        "source_path": f"cache/njm_export/{name}",
        "target_model_path": target_path,
        "frame_count": m.frame_count,
        "fps": fps_hint,
        "bone_count": m.bone_count,
        "type_flags": m.type_flags,
        "interpolation": m.interpolation,
        "bones": bones_out,
        "imported": True,
    }


class AnimPreviewDeleteReq(BaseModel):
    njm_path: str


@app.post("/api/anim_preview/delete")
def api_anim_preview_delete(req: AnimPreviewDeleteReq):
    """Remove a preview-only imported animation (njm + sidecar).

    Used by the "Remove from preview" button in the Motions tab. Only
    operates on files inside ``cache/njm_export/`` — refuses anything
    with directory components or a name that doesn't end ``.njm``.
    Returns 200 with ``{ok, removed: [<paths>]}`` on success, including
    when the file was already gone (idempotent).
    """
    name = _safe_njm_export_name(req.njm_path)
    njm_p = NJM_EXPORT_DIR / name
    sidecar_p = NJM_EXPORT_DIR / (name + ".preview.json")
    removed: list[str] = []
    for p in (njm_p, sidecar_p):
        if p.is_file():
            try:
                p.unlink()
                removed.append(str(p))
            except OSError as e:
                raise HTTPException(500, f"unlink {p}: {e}")
    log.info("anim_preview/delete %s -> removed %s", name, removed)
    return {"ok": True, "njm_path": name, "removed": removed}


# ============================================================================
# Anim Keyframe Editor endpoints (2026-04-25)
# ----------------------------------------------------------------------------
# Round-trippable motion-edit surface for the new Anim Editor panel. The four
# endpoints below let the frontend:
#
#   POST /api/anim_keyframe/load    fetch a parsed motion as JSON. Wire shape
#                                   matches /api/animation_data (per-bone
#                                   keyframe lists + present bitmask) PLUS
#                                   round-trip metadata (narrow/wide ANG
#                                   choice per bone, source POF0 + raw body
#                                   for byte-exact re-encode, source layout
#                                   hints).
#   POST /api/anim_keyframe/insert  insert/upsert a keyframe at a frame on a
#                                   bone. Returns the updated motion JSON.
#                                   This is a server-side helper — the panel
#                                   could do the same mutation in JS — but
#                                   keeping the canonical mutation in Python
#                                   means tests cover both paths and the JS
#                                   stays simple.
#   POST /api/anim_keyframe/delete  inverse of insert: remove the keyframe at
#                                   the given (frame, bone) coordinate.
#   POST /api/anim_keyframe/save    encode via njm_writer, stage to
#                                   cache/njm_export/<safe>.njm.
#
# Wire format (one of the load response / save body / mutation request):
#
#     {
#       "name":           "<motion_name>",          # for save: output filename stem
#       "frame_count":    int,
#       "type_flags":     int,                       # NJD_MTYPE_* bitfield
#       "interpolation":  int,                       # 0=linear, ...
#       "inp_fn":         int,                       # raw u16 from header
#                                                   # (high 2 bits = interp,
#                                                   #  low 4 bits = element_count).
#                                                   # Re-derived from
#                                                   # type_flags+interpolation
#                                                   # if absent.
#       "fps":            float,                     # display-only, default 30
#       "bones": [
#         {
#           "idx":     int,
#           "present": int,                          # NJD_MTYPE_* bitfield —
#                                                   # WHICH channels this bone
#                                                   # actually has keyframes for.
#           "kf": [
#             { "t": int,
#               "tx": float, "ty": float, "tz": float,
#               "rx": int,   "ry": int,   "rz": int,    # BAMS
#               "sx": float, "sy": float, "sz": float,
#               "qw"?: float, "qx"?: float, "qy"?: float, "qz"?: float,
#             }, ...
#           ],
#           "narrow_ang": bool,                      # round-trip hint
#         }, ...
#       ],
#       "round_trip"?: {                             # opaque to clients;
#         "pof0_b64":  str,                          # round-trip metadata
#         "source_body_b64": str,                    # surfaced by /load,
#         "track_offset_hints": [{                   # echoed back by /save.
#           "bone": int, "kind": int, "offset": int
#         }],
#         "trailing_size": int,
#         "m_data_table_offset": int,
#       },
#     }
#
# The frontend (anim_editor_panel.js) treats `round_trip` as a black box: it
# loads it, edits keyframes, and posts the same blob back. Mutating the
# `kf` lists invalidates the byte-exact round-trip — that's expected. The
# server's /save path tolerates a missing or stale `round_trip` field
# (re-derives a packed layout via NjmRawMotion's default-packed branch).
#
# All 4 endpoints are bounded by MAX_BUILD_NJM_BODY (16 MB) — the largest
# shipping NJM is ~100 KB, so a 16 MB cap is generous even with edits.
# ============================================================================

# Pull the round-trip parser separately from the writer-only helpers above
# (which were already imported at the build_njm region).
from formats.njm_writer import parse_njm_for_writer as _njm_parse_for_writer  # noqa: E402
from formats.njm import (  # noqa: E402
    NJD_MTYPE_POS as _NJD_POS,
    NJD_MTYPE_ANG as _NJD_ANG,
    NJD_MTYPE_SCL as _NJD_SCL,
    NJD_MTYPE_VEC as _NJD_VEC,
    NJD_MTYPE_QUAT as _NJD_QUAT,
)


def _ake_kinds_in_order(type_flags: int) -> list[int]:
    """Mirror NJM track-emission order: POS, ANG, SCL, VEC, QUAT."""
    out: list[int] = []
    if type_flags & _NJD_POS:
        out.append(_NJD_POS)
    if type_flags & _NJD_ANG:
        out.append(_NJD_ANG)
    if type_flags & _NJD_SCL:
        out.append(_NJD_SCL)
    if type_flags & _NJD_VEC:
        out.append(_NJD_VEC)
    if type_flags & _NJD_QUAT:
        out.append(_NJD_QUAT)
    return out


def _ake_motion_to_json(raw: _NjmRawMotion, *, fps: float = 30.0,
                       motion_name: str = "") -> dict:
    """Project an ``NjmRawMotion`` to the editor wire format.

    Per-bone tracks are merged back into a single keyframe list (indexed
    by frame number) so the UI's "select a keyframe to edit all its
    channels" loop has a single sortable list per bone. **Each keyframe
    carries a ``chan`` bitmask** identifying the channels actually
    authored at THAT specific frame — without this, a keyframe that
    appears in the merged list only because a SIBLING channel had a
    keyframe there would be re-emitted as if it had been authored on
    every channel during ``_ake_motion_from_json``, which expands track
    counts and breaks byte-exact round-trip.

    The bone-wide ``present`` mask is the OR of every kf.chan bitfield
    on that bone; consumers (the UI) use it to decide whether to expose
    POS / ANG / SCL inspector rows at all. ``narrow_ang`` preserves the
    source's narrow-vs-wide euler-encoding choice (Phantasmal-style
    fallback rule).
    """
    bones_out: list[dict] = []
    for bi, bone in enumerate(raw.bones):
        merged: dict[int, dict] = {}
        present = 0
        narrow_ang = True
        # POS / SCL / VEC tracks — merge into per-frame dicts.
        pos_tk = bone.tracks_by_kind.get(_NJD_POS)
        if pos_tk and pos_tk.keyframes:
            present |= _NJD_POS
            for (frame, x, y, z) in pos_tk.keyframes:
                kf = merged.setdefault(int(frame), {"t": int(frame), "chan": 0})
                kf["tx"], kf["ty"], kf["tz"] = float(x), float(y), float(z)
                kf["chan"] |= _NJD_POS
        ang_tk = bone.tracks_by_kind.get(_NJD_ANG)
        if ang_tk and ang_tk.keyframes:
            present |= _NJD_ANG
            narrow_ang = bool(ang_tk.narrow)
            for (frame, rx, ry, rz) in ang_tk.keyframes:
                kf = merged.setdefault(int(frame), {"t": int(frame), "chan": 0})
                kf["rx"], kf["ry"], kf["rz"] = int(rx), int(ry), int(rz)
                kf["chan"] |= _NJD_ANG
        scl_tk = bone.tracks_by_kind.get(_NJD_SCL)
        if scl_tk and scl_tk.keyframes:
            present |= _NJD_SCL
            for (frame, sx, sy, sz) in scl_tk.keyframes:
                kf = merged.setdefault(int(frame), {"t": int(frame), "chan": 0})
                kf["sx"], kf["sy"], kf["sz"] = float(sx), float(sy), float(sz)
                kf["chan"] |= _NJD_SCL
        quat_tk = bone.tracks_by_kind.get(_NJD_QUAT)
        if quat_tk and quat_tk.keyframes:
            present |= _NJD_QUAT
            for (frame, qw, qx, qy, qz) in quat_tk.keyframes:
                kf = merged.setdefault(int(frame), {"t": int(frame), "chan": 0})
                kf["qw"] = float(qw); kf["qx"] = float(qx)
                kf["qy"] = float(qy); kf["qz"] = float(qz)
                kf["chan"] |= _NJD_QUAT
        # Sort by frame, fill identity defaults for missing channels.
        kf_list: list[dict] = []
        for t in sorted(merged.keys()):
            kf = merged[t]
            kf.setdefault("tx", 0.0); kf.setdefault("ty", 0.0); kf.setdefault("tz", 0.0)
            kf.setdefault("rx", 0); kf.setdefault("ry", 0); kf.setdefault("rz", 0)
            kf.setdefault("sx", 1.0); kf.setdefault("sy", 1.0); kf.setdefault("sz", 1.0)
            kf_list.append(kf)
        bones_out.append({
            "idx": bi,
            "present": present,
            "narrow_ang": narrow_ang,
            "kf": kf_list,
        })

    rt: dict = {
        "pof0_b64": base64.b64encode(raw.pof0_bytes or b"").decode("ascii"),
        "m_data_table_offset": int(raw.m_data_table_offset),
        "trailing_size": int(raw.trailing_size_hint or 0),
    }
    if raw.source_body is not None:
        rt["source_body_b64"] = base64.b64encode(raw.source_body).decode("ascii")
    if raw.track_offset_hint:
        rt["track_offset_hints"] = [
            {"bone": int(b), "kind": int(k), "offset": int(o)}
            for ((b, k), o) in sorted(raw.track_offset_hint.items())
        ]
    return {
        "name": motion_name,
        "frame_count": int(raw.frame_count),
        "type_flags": int(raw.type_flags),
        "interpolation": int((raw.inp_fn >> 6) & 0b11),
        "inp_fn": int(raw.inp_fn),
        "fps": float(fps),
        "bones": bones_out,
        "round_trip": rt,
    }


def _ake_motion_from_json(motion_json: dict) -> _NjmRawMotion:
    """Decode the editor wire format back into an ``NjmRawMotion``.

    Inverse of ``_ake_motion_to_json``. Splits the merged per-frame
    keyframe list back into per-track lists keyed off the per-bone
    ``present`` bitfield. Preserves narrow-vs-wide ANG choice and the
    optional ``round_trip`` block (which the writer uses for byte-exact
    re-encode when nothing was edited).
    """
    if not isinstance(motion_json, dict):
        raise HTTPException(400, "motion_json: must be an object")
    try:
        type_flags = int(motion_json.get("type_flags", 0))
        frame_count = int(motion_json.get("frame_count", 0))
        if "inp_fn" in motion_json:
            inp_fn = int(motion_json["inp_fn"])
        else:
            # Re-derive: high 2 bits = interpolation; low 4 bits = element_count.
            interp = int(motion_json.get("interpolation", 0)) & 0b11
            element_count = bin(type_flags & 0x200F).count("1")
            inp_fn = (interp << 6) | element_count
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"motion_json header: {e}")
    rt = motion_json.get("round_trip") or {}
    raw = _NjmRawMotion(
        frame_count=frame_count,
        type_flags=type_flags,
        inp_fn=inp_fn,
        m_data_table_offset=int(rt.get("m_data_table_offset", 0xC) or 0xC),
    )
    pof0_b64 = rt.get("pof0_b64")
    if pof0_b64:
        try:
            raw.pof0_bytes = base64.b64decode(pof0_b64, validate=False)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(400, f"round_trip.pof0_b64: {e}")
    sb_b64 = rt.get("source_body_b64")
    if sb_b64:
        try:
            raw.source_body = base64.b64decode(sb_b64, validate=False)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(400, f"round_trip.source_body_b64: {e}")
    if rt.get("trailing_size"):
        raw.trailing_size_hint = int(rt["trailing_size"])
    if rt.get("track_offset_hints"):
        raw.track_offset_hint = {}
        for h in rt["track_offset_hints"]:
            try:
                raw.track_offset_hint[(int(h["bone"]), int(h["kind"]))] = int(h["offset"])
            except (KeyError, TypeError, ValueError) as e:
                raise HTTPException(400, f"round_trip.track_offset_hints: {e}")

    bones_in = motion_json.get("bones") or []
    if not isinstance(bones_in, list):
        raise HTTPException(400, "motion_json.bones: must be a list")
    kinds_in_order = _ake_kinds_in_order(type_flags)
    for i, b in enumerate(bones_in):
        if not isinstance(b, dict):
            raise HTTPException(400, f"bones[{i}]: must be object")
        try:
            present = int(b.get("present", 0))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"bones[{i}].present: {e}")
        narrow_ang = bool(b.get("narrow_ang", True))
        bone = _NjmBoneTracks()
        kf_list = b.get("kf") or []
        if not isinstance(kf_list, list):
            raise HTTPException(400, f"bones[{i}].kf: must be a list")
        # Stable sort by frame.
        try:
            kf_sorted = sorted(kf_list, key=lambda k: int(k.get("t", 0)))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"bones[{i}].kf sort: {e}")
        pos_kfs: list[tuple] = []
        ang_kfs: list[tuple] = []
        scl_kfs: list[tuple] = []
        quat_kfs: list[tuple] = []
        for j, kf in enumerate(kf_sorted):
            if not isinstance(kf, dict):
                raise HTTPException(400, f"bones[{i}].kf[{j}]: must be object")
            try:
                t = int(kf.get("t", 0))
            except (TypeError, ValueError) as e:
                raise HTTPException(400, f"bones[{i}].kf[{j}].t: {e}")
            # Per-keyframe channel mask: which channels were authored at
            # THIS frame. When absent (e.g. UI-inserted keyframes that
            # didn't propagate `chan`) we fall back to the bone-wide
            # `present` mask — this is the lossy case where any channel
            # the bone has is treated as authored at every keyframe (the
            # editor signals this by stripping `chan` on insertion). A
            # /load → /save round-trip with no edits keeps `chan` so byte-
            # exact round-trip survives.
            chan = kf.get("chan")
            if chan is None:
                chan = present
            else:
                try:
                    chan = int(chan)
                except (TypeError, ValueError) as e:
                    raise HTTPException(400, f"bones[{i}].kf[{j}].chan: {e}")
            if (chan & _NJD_POS) and (present & _NJD_POS):
                pos_kfs.append((
                    t, float(kf.get("tx", 0.0)),
                    float(kf.get("ty", 0.0)), float(kf.get("tz", 0.0)),
                ))
            if (chan & _NJD_ANG) and (present & _NJD_ANG):
                ang_kfs.append((
                    t, int(kf.get("rx", 0)) & 0xFFFF,
                    int(kf.get("ry", 0)) & 0xFFFF,
                    int(kf.get("rz", 0)) & 0xFFFF,
                ) if narrow_ang else (
                    t, int(kf.get("rx", 0)),
                    int(kf.get("ry", 0)), int(kf.get("rz", 0)),
                ))
            if (chan & _NJD_SCL) and (present & _NJD_SCL):
                scl_kfs.append((
                    t, float(kf.get("sx", 1.0)),
                    float(kf.get("sy", 1.0)), float(kf.get("sz", 1.0)),
                ))
            if (chan & _NJD_QUAT) and (present & _NJD_QUAT) and "qw" in kf:
                quat_kfs.append((
                    t, float(kf["qw"]), float(kf["qx"]),
                    float(kf["qy"]), float(kf["qz"]),
                ))
        # Always emit a slot for each enabled kind (offset/count = 0 when empty).
        for kind in kinds_in_order:
            if kind == _NJD_POS:
                bone.tracks_by_kind[kind] = _NjmTrack(kind, pos_kfs, narrow=True)
            elif kind == _NJD_ANG:
                bone.tracks_by_kind[kind] = _NjmTrack(kind, ang_kfs, narrow=narrow_ang)
            elif kind == _NJD_SCL:
                bone.tracks_by_kind[kind] = _NjmTrack(kind, scl_kfs, narrow=True)
            elif kind == _NJD_QUAT:
                bone.tracks_by_kind[kind] = _NjmTrack(kind, quat_kfs, narrow=True)
            else:  # VEC — rare; round-trip as POS-shaped track if any.
                bone.tracks_by_kind[kind] = _NjmTrack(kind, [], narrow=True)
        raw.bones.append(bone)
    return raw


def _ake_count_keyframes(motion_json: dict) -> int:
    """Sum kf-count across bones (used for log lines + audit)."""
    n = 0
    for b in motion_json.get("bones") or []:
        kf = b.get("kf") if isinstance(b, dict) else None
        if isinstance(kf, list):
            n += len(kf)
    return n


def _ake_invalidate_round_trip(motion_json: dict) -> None:
    """Drop ``round_trip`` source-body + track_offset_hints after a mutation.

    The byte-exact-round-trip data is only valid for an UNCHANGED motion.
    Once a keyframe is edited / inserted / deleted we must drop the
    source_body and track_offset_hints so the encoder falls into the
    packed-layout branch — keeping them would either get partly clobbered
    by the mutated keyframes (if offset hint still applies) or leave
    stale uninitialised bytes between tracks.

    POF0 + m_data_table_offset stay — they're independent of keyframe
    layout.
    """
    rt = motion_json.get("round_trip")
    if not isinstance(rt, dict):
        return
    rt.pop("source_body_b64", None)
    rt.pop("track_offset_hints", None)
    rt.pop("trailing_size", None)


class AnimKeyframeLoadReq(BaseModel):
    model_path: str
    motion_name: str


@app.post("/api/anim_keyframe/load")
def api_anim_keyframe_load(req: AnimKeyframeLoadReq, request: Request):
    """Load a motion as round-trippable JSON for the keyframe editor.

    Body:
      ``model_path`` follows the same form as /api/animations:
        ``<file>.nj``, ``<file>.xj``, ``<bml>``, or ``<bml>#<inner>.{nj,xj}``.
      ``motion_name`` matches an entry returned by /api/animations.

    Returns a JSON envelope per the wire format above (load shape).
    """
    _enforce_body_size(request, MAX_BUILD_NJM_BODY)
    base, hash_inner = _split_inner_path(req.model_path)
    p = _resolve_model_mesh_path(base)
    ext = p.suffix.lower()
    sources = _resolve_motion_sources(p, ext, hash_inner)
    if not sources:
        raise HTTPException(404, f"no NJM motions for {req.model_path!r}")

    target = (req.motion_name or "").strip()
    if not target:
        raise HTTPException(400, "motion_name: required")
    target_lower = target.lower()
    target_stem = target_lower[:-4] if target_lower.endswith(".njm") else target_lower

    chosen = None
    chosen_idx = -1
    for i, (bml_path, inner_name, label) in enumerate(sources):
        cand = inner_name.lower() if inner_name else bml_path.name.lower()
        cand_stem = cand[:-4] if cand.endswith(".njm") else cand
        if cand_stem == target_stem or cand == target_lower:
            chosen = sources[i]
            chosen_idx = i
            break
    if chosen is None:
        # Substring fallback (mirrors /api/animation_data).
        for i, (bml_path, inner_name, label) in enumerate(sources):
            cand = inner_name.lower() if inner_name else bml_path.name.lower()
            cand_stem = cand[:-4] if cand.endswith(".njm") else cand
            if target_stem and target_stem in cand_stem:
                chosen = sources[i]
                chosen_idx = i
                break
    if chosen is None:
        raise HTTPException(404, f"no motion named {target!r}")

    bml_path, inner_name, label = chosen
    try:
        njm_bytes = _read_njm_for_source(bml_path, inner_name)
    except HTTPException:
        raise
    try:
        raw = _njm_parse_for_writer(njm_bytes)
    except (ValueError, struct.error) as e:
        raise HTTPException(400, f"NJM parse failed: {e}")
    name_disp = inner_name if inner_name else bml_path.name
    if name_disp.lower().endswith(".njm"):
        name_disp = name_disp[:-4]
    payload = _ake_motion_to_json(
        raw, fps=_njm_guess_fps(name_disp), motion_name=name_disp,
    )
    payload.update({
        "model_path": req.model_path,
        "motion_index": chosen_idx,
        "source_path": label,
        "bone_count": len(raw.bones),
    })
    # v4 / Task 2 — probe for a prior save's .preview.json sidecar
    # adjacent to the cached export. Keying matches what the panel uses
    # by default (name = motion_name + ".njm"). If multiple sidecars
    # could match (e.g. user did a "save as new"), we pick the
    # exact-name match first; otherwise we leave bezier_handles unset.
    sidecar_handles: dict | None = None
    candidate_names: list[str] = []
    base_stem = name_disp.lower()
    if base_stem.endswith(".njm"):
        base_stem = base_stem[:-4]
    candidate_names.append(base_stem + ".njm")
    # Also accept the BML-inner spelling (some round-trips strip ".njm"
    # when the inner already includes it).
    if not base_stem.endswith(".njm"):
        candidate_names.append(base_stem)
    for cand in candidate_names:
        try:
            sc_path = NJM_EXPORT_DIR / (cand + ".preview.json")
            if not sc_path.is_file():
                # Try once more with stripped/normalised case via dir scan.
                cand_lower = cand.lower()
                for entry in NJM_EXPORT_DIR.iterdir():
                    if not entry.name.lower().endswith(".njm.preview.json"):
                        continue
                    stem = entry.name[: -len(".preview.json")]
                    if stem.lower() == cand_lower:
                        sc_path = entry
                        break
                else:
                    continue
            sc_data = json.loads(sc_path.read_text(encoding="utf-8"))
            if isinstance(sc_data, dict):
                handles = sc_data.get("bezier_handles")
                if isinstance(handles, dict):
                    sidecar_handles = handles
                    break
        except (OSError, ValueError, FileNotFoundError):
            continue
    if sidecar_handles is not None:
        payload["bezier_handles"] = sidecar_handles
    log.info(
        "anim_keyframe/load %s#%s -> %d bones / %d frames / %d keyframes, handles=%s",
        req.model_path, target, len(raw.bones), raw.frame_count,
        _ake_count_keyframes(payload),
        len(sidecar_handles) if sidecar_handles else 0,
    )
    return payload


class AnimKeyframeSaveReq(BaseModel):
    motion_json: dict
    name: str
    # v4 / Task 2 — optional bezier handle map, persisted into the
    # adjacent .preview.json sidecar so the panel can restore them on
    # the next load. Shape: { "<boneIdx>:<kfIdx>:<channelKey>":
    # {"inDx": float, "inDy": float, "outDx": float, "outDy": float} }
    # The runtime ignores this — it only round-trips through the panel.
    bezier_handles: Optional[dict] = None


@app.post("/api/anim_keyframe/save")
def api_anim_keyframe_save(req: AnimKeyframeSaveReq, request: Request):
    """Encode the edited motion via njm_writer + stage to cache/njm_export.

    ``name`` is the output filename (must end in ``.njm``); the same
    safe-name rules as /api/build_njm apply.

    v4 / Task 2 — when ``bezier_handles`` is provided, also write/merge
    a ``.preview.json`` sidecar adjacent to the output .njm so the
    motion editor can restore handle state on the next load.
    """
    _enforce_body_size(request, MAX_BUILD_NJM_BODY)
    name = _safe_archive_name(req.name)
    if not name.lower().endswith(".njm"):
        raise HTTPException(400, "name: must end in .njm")
    raw = _ake_motion_from_json(req.motion_json)
    try:
        out = _encode_njm(raw)
    except (ValueError, struct.error) as e:
        raise HTTPException(400, f"encode_njm failed: {e}")
    out_path = NJM_EXPORT_DIR / name
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(out)
    os.replace(tmp, out_path)
    md5 = _md5_bytes(out)

    # v4 / Task 2 — write bezier handles into the sidecar (additive).
    # Merges with an existing sidecar so we don't drop other fields
    # (target_model_path, bone_map, etc.) that other workflows wrote.
    sidecar_written = False
    if req.bezier_handles is not None:
        if not isinstance(req.bezier_handles, dict):
            raise HTTPException(400, "bezier_handles: must be an object")
        # Validate each entry's shape — keep the wire contract narrow.
        validated: dict = {}
        for k, v in req.bezier_handles.items():
            if not isinstance(k, str):
                raise HTTPException(400, "bezier_handles: keys must be strings")
            if not isinstance(v, dict):
                raise HTTPException(400, f"bezier_handles[{k!r}]: must be an object")
            try:
                validated[k] = {
                    "inDx": float(v.get("inDx", 0.0)),
                    "inDy": float(v.get("inDy", 0.0)),
                    "outDx": float(v.get("outDx", 0.0)),
                    "outDy": float(v.get("outDy", 0.0)),
                }
            except (TypeError, ValueError) as e:
                raise HTTPException(400, f"bezier_handles[{k!r}]: bad scalar: {e}")
        sidecar_path = NJM_EXPORT_DIR / (name + ".preview.json")
        existing: dict = {}
        if sidecar_path.is_file():
            try:
                existing = json.loads(sidecar_path.read_text(encoding="utf-8")) or {}
                if not isinstance(existing, dict):
                    existing = {}
            except (OSError, ValueError):
                existing = {}
        existing["bezier_handles"] = validated
        # Stamp some basic metadata so the sidecar is self-describing
        # even when we wrote it fresh (rather than merging into an
        # import-side sidecar).
        existing.setdefault("source", "anim_keyframe_save")
        existing.setdefault("frame_count", int(raw.frame_count))
        existing.setdefault("bone_count", int(len(raw.bones)))
        existing["njm_md5"] = md5
        tmp_sc = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
        tmp_sc.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        os.replace(tmp_sc, sidecar_path)
        sidecar_written = True

    log.info(
        "anim_keyframe/save %s -> %s (%d bytes, md5=%s, bones=%d, frames=%d, handles=%s)",
        name, out_path, len(out), md5, len(raw.bones), raw.frame_count,
        len(req.bezier_handles) if req.bezier_handles else 0,
    )
    return {
        "ok": True,
        "path": str(out_path),
        "name": name,
        "size": len(out),
        "md5": md5,
        "bone_count": len(raw.bones),
        "frame_count": raw.frame_count,
        "sidecar_written": sidecar_written,
    }


class AnimKeyframeInsertReq(BaseModel):
    motion_json: dict
    bone_idx: int
    frame_idx: int
    pos: Optional[list] = None       # [tx, ty, tz] — float or None for "keep present"
    ang: Optional[list] = None       # [rx, ry, rz] — int BAMS
    scl: Optional[list] = None       # [sx, sy, sz] — float
    quat: Optional[list] = None      # [qw, qx, qy, qz] — float


@app.post("/api/anim_keyframe/insert")
def api_anim_keyframe_insert(req: AnimKeyframeInsertReq, request: Request):
    """Insert OR upsert a keyframe at (bone_idx, frame_idx).

    If a keyframe at the same frame already exists for the bone, its
    fields are merged (the request's pos/ang/scl/quat overwrite the
    matching channels; other channels are preserved). Otherwise a new
    keyframe is appended in frame order.

    Always returns the updated full motion_json so the caller can swap
    its in-memory copy without recomputing.
    """
    _enforce_body_size(request, MAX_BUILD_NJM_BODY)
    motion = req.motion_json
    if not isinstance(motion, dict):
        raise HTTPException(400, "motion_json: must be an object")
    bones = motion.get("bones") or []
    if not isinstance(bones, list):
        raise HTTPException(400, "motion_json.bones: must be a list")
    if not (0 <= req.bone_idx < len(bones)):
        raise HTTPException(400, f"bone_idx out of range: {req.bone_idx}")
    if req.frame_idx < 0:
        raise HTTPException(400, "frame_idx: must be >= 0")
    bone = bones[req.bone_idx]
    if not isinstance(bone, dict):
        raise HTTPException(400, f"bones[{req.bone_idx}]: must be object")
    kf_list = bone.get("kf") or []
    if not isinstance(kf_list, list):
        raise HTTPException(400, f"bones[{req.bone_idx}].kf: must be a list")
    # Find existing keyframe at this frame.
    existing = next((k for k in kf_list if int(k.get("t", -1)) == req.frame_idx), None)
    if existing is None:
        # Default channel values — identity unless the request supplies otherwise.
        kf = {
            "t": int(req.frame_idx),
            "tx": 0.0, "ty": 0.0, "tz": 0.0,
            "rx": 0,   "ry": 0,   "rz": 0,
            "sx": 1.0, "sy": 1.0, "sz": 1.0,
            "chan": 0,
        }
        kf_list.append(kf)
        existing = kf
    chan = int(existing.get("chan", 0))
    if req.pos is not None and len(req.pos) == 3:
        existing["tx"], existing["ty"], existing["tz"] = (
            float(req.pos[0]), float(req.pos[1]), float(req.pos[2])
        )
        chan |= _NJD_POS
    if req.ang is not None and len(req.ang) == 3:
        existing["rx"], existing["ry"], existing["rz"] = (
            int(req.ang[0]), int(req.ang[1]), int(req.ang[2])
        )
        chan |= _NJD_ANG
    if req.scl is not None and len(req.scl) == 3:
        existing["sx"], existing["sy"], existing["sz"] = (
            float(req.scl[0]), float(req.scl[1]), float(req.scl[2])
        )
        chan |= _NJD_SCL
    if req.quat is not None and len(req.quat) == 4:
        existing["qw"], existing["qx"], existing["qy"], existing["qz"] = (
            float(req.quat[0]), float(req.quat[1]),
            float(req.quat[2]), float(req.quat[3]),
        )
        chan |= _NJD_QUAT
    existing["chan"] = chan
    # Update bone's `present` mask if the inserted channel was new.
    present = int(bone.get("present", 0))
    if req.pos is not None: present |= _NJD_POS
    if req.ang is not None: present |= _NJD_ANG
    if req.scl is not None: present |= _NJD_SCL
    if req.quat is not None: present |= _NJD_QUAT
    bone["present"] = present
    # Re-sort by frame.
    kf_list.sort(key=lambda k: int(k.get("t", 0)))
    bone["kf"] = kf_list
    # Bump frame_count if needed.
    if req.frame_idx + 1 > int(motion.get("frame_count", 0)):
        motion["frame_count"] = int(req.frame_idx + 1)
    _ake_invalidate_round_trip(motion)
    return motion


class AnimKeyframeDeleteReq(BaseModel):
    motion_json: dict
    bone_idx: int
    frame_idx: int


@app.post("/api/anim_keyframe/delete")
def api_anim_keyframe_delete(req: AnimKeyframeDeleteReq, request: Request):
    """Remove the keyframe at (bone_idx, frame_idx). No-op if absent."""
    _enforce_body_size(request, MAX_BUILD_NJM_BODY)
    motion = req.motion_json
    if not isinstance(motion, dict):
        raise HTTPException(400, "motion_json: must be an object")
    bones = motion.get("bones") or []
    if not (0 <= req.bone_idx < len(bones)):
        raise HTTPException(400, f"bone_idx out of range: {req.bone_idx}")
    bone = bones[req.bone_idx]
    if not isinstance(bone, dict):
        raise HTTPException(400, f"bones[{req.bone_idx}]: must be object")
    kf_list = bone.get("kf") or []
    if not isinstance(kf_list, list):
        raise HTTPException(400, f"bones[{req.bone_idx}].kf: must be a list")
    n_before = len(kf_list)
    kf_list[:] = [k for k in kf_list if int(k.get("t", -1)) != req.frame_idx]
    bone["kf"] = kf_list
    _ake_invalidate_round_trip(motion)
    return {
        "ok": True,
        "removed": n_before - len(kf_list),
        "motion_json": motion,
    }


# ============================================================================
# Animation blend-spaces (Task B / 2026-04-25)
# ----------------------------------------------------------------------------
# /api/anim/blend body:
#   {
#     "model_path":          "<bml or bml#inner>",     # to resolve motions
#     "source_motion_names": ["walk.njm", "run.njm"],  # inner names in BML
#     "weights":             [0.5, 0.5],
#     "output_name":         "walkrun_blend.njm",
#     "frame_count":         <int|null>,               # null = max source
#     "transition_curve":    "linear" | "smooth" | ...
#   }
#
# The server resolves each source motion via the same logic as
# /api/animations, parses each via parse_njm_for_writer (round-trip
# friendly), passes them through formats.anim_blend.blend_motions, and
# stages the result at cache/njm_export/<output_name>.
#
# Coordinates with the motion-editing agent (anim_editor_panel.js): we
# expose only the server endpoint + format module here. UI integration
# happens in their tab; this sits behind the same /api surface.
# ============================================================================

from formats.anim_blend import (  # noqa: E402
    VALID_TRANSITIONS as _BLEND_VALID_CURVES,
    blend_motions as _blend_motions,
    summarize_blend as _summarize_blend,
    TRANSITION_LINEAR as _BLEND_LINEAR,
)
from formats.njm_writer import parse_njm_for_writer as _parse_njm_for_writer  # noqa: E402


class AnimBlendReq(BaseModel):
    """Request body for ``POST /api/anim/blend``."""
    model_path: str
    source_motion_names: list[str]
    weights: list[float]
    output_name: str
    frame_count: Optional[int] = None
    transition_curve: str = _BLEND_LINEAR


@app.post("/api/anim/blend")
def api_anim_blend(req: AnimBlendReq, request: Request):
    """Blend N source motions into one .njm.

    Source motions are resolved relative to ``model_path`` using the
    same lookup as ``/api/animations`` — they may live as inner BML
    entries (``foo.bml#bar.njm``) or as top-level files. Output lands
    in ``cache/njm_export/<output_name>``.

    Returns
    -------
    JSON with keys: ok, njm_path, njm_name, size, md5, frame_count,
    bone_count, weights, curve, source_count.
    """
    _enforce_body_size(request, MAX_ANIM_BODY)
    if not req.source_motion_names:
        raise HTTPException(400, "source_motion_names: must be non-empty")
    if len(req.source_motion_names) != len(req.weights):
        raise HTTPException(
            400,
            f"source_motion_names ({len(req.source_motion_names)}) and "
            f"weights ({len(req.weights)}) length mismatch",
        )
    if req.transition_curve not in _BLEND_VALID_CURVES:
        raise HTTPException(
            400,
            f"transition_curve {req.transition_curve!r} not in {_BLEND_VALID_CURVES}",
        )
    if req.frame_count is not None and req.frame_count <= 0:
        raise HTTPException(400, f"frame_count must be positive, got {req.frame_count}")

    # ---- Resolve source motions ----
    base_path, hash_inner = _split_inner_path(req.model_path)
    try:
        p = _resolve_model_mesh_path(base_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"model_path resolve failed: {e}")
    ext = p.suffix.lower()
    sources = _resolve_motion_sources(p, ext, hash_inner)
    by_name: dict[str, tuple] = {}
    for (bml_path, inner_name, label) in sources:
        # Use both the inner name AND the full label as keys so callers
        # can pass either "foo.njm" or "NpcApcMot#foo.njm".
        if inner_name:
            by_name.setdefault(inner_name, (bml_path, inner_name, label))
        by_name.setdefault(label, (bml_path, inner_name, label))

    motions = []
    for nm in req.source_motion_names:
        entry = by_name.get(nm) or by_name.get(nm + ".njm")
        if entry is None:
            available = sorted(set(k for k in by_name.keys() if not k.endswith(("#",))))
            raise HTTPException(
                404,
                f"source motion {nm!r} not found for {req.model_path!r}; "
                f"available (first 10): {available[:10]}",
            )
        bml_path, inner_name, _label = entry
        try:
            njm_bytes = _read_njm_for_source(bml_path, inner_name)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"read source NJM {nm!r} failed: {e}")
        try:
            motion = _parse_njm_for_writer(njm_bytes)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(400, f"parse source NJM {nm!r} failed: {e}")
        motions.append(motion)

    # ---- Blend ----
    try:
        blended = _blend_motions(
            motions, req.weights,
            frame_count=req.frame_count,
            transition_curve=req.transition_curve,
        )
    except ValueError as e:
        raise HTTPException(400, f"blend failed: {e}")
    summary = _summarize_blend(blended)

    # ---- Encode + stage ----
    try:
        out = _encode_njm(blended)
    except (ValueError, struct.error) as e:
        raise HTTPException(500, f"encode_njm failed: {e}")
    out_name = _safe_motion_name(req.output_name)
    NJM_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NJM_EXPORT_DIR / out_name
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(out)
    os.replace(tmp, out_path)
    md5 = _md5_bytes(out)
    log.info(
        "anim/blend %s sources=%s weights=%s -> %s (%d bytes, md5=%s, frames=%d, bones=%d)",
        req.model_path, req.source_motion_names, req.weights,
        out_path, len(out), md5,
        summary["frame_count"], summary["bone_count"],
    )
    return {
        "ok": True,
        "njm_path": str(out_path),
        "njm_name": out_name,
        "size": len(out),
        "md5": md5,
        "frame_count": summary["frame_count"],
        "bone_count": summary["bone_count"],
        "weights": summary["weights"],
        "curve": summary["curve"],
        "source_count": summary["source_count"],
    }


# ---------------------------------------------------------------------------- root
#
# Static-asset cache busting (v4 visual polish bundle, 2026-04-25).
#
# Browsers cache responses for /static/<file> aggressively (Chrome heuristic
# can hold them for hours even with no Cache-Control set), so editing
# `static/app.js` server-side then reloading the editor would still serve
# the stale bytes — users had to hard-reload to pick up new code. We
# fingerprint every <script src="/static/..."> and <link href="/static/...">
# in index.html with `?v=<sha8>` of the referenced file's bytes; the
# browser sees a brand-new URL whenever a file changes and bypasses cache.
#
# Design choices:
#   - Rewrite at serve time, not build time. Zero build-step requirement,
#     and the dev workflow (edit, refresh) just works.
#   - SHA computed lazily and cached by (mtime_ns, size); a re-stat is
#     ~microseconds, the SHA itself only re-runs when the file actually
#     changes. Cache is unbounded but capped by the static dir's file
#     count (~25 files), so unbounded-equals-bounded in practice.
#   - First 8 hex chars of SHA-256 — enough to dodge accidental collisions
#     across the ~25-file static surface.
#   - Plain regex over the HTML. The index template is hand-authored (not
#     generated) and the script/link tags use a small set of stable
#     forms, so a regex is more robust than asking us to depend on a
#     parser like lxml. Safety: we only match attributes on /static/...
#     paths; unrelated href= attributes (e.g. <a href="...">) are
#     untouched.

_STATIC_VERSION_CACHE: "dict[str, tuple[int, int, str]]" = {}
_STATIC_VERSION_LOCK = threading.Lock()


def _static_asset_version(rel_path: str) -> Optional[str]:
    """Return an 8-char SHA-256 of /static/<rel_path>, cached by mtime+size.

    Returns None on stat / read failure — caller must skip the rewrite
    in that case (no querystring is better than a stale fingerprint).
    """
    # Strip a leading slash if present (the regex path may include it).
    rel = rel_path.lstrip("/")
    if rel.startswith("static/"):
        rel = rel[len("static/"):]
    full = STATIC_DIR / rel
    try:
        st = full.stat()
    except OSError:
        return None
    key = str(full.resolve())
    mtime_ns = int(st.st_mtime_ns)
    size = int(st.st_size)
    with _STATIC_VERSION_LOCK:
        cached = _STATIC_VERSION_CACHE.get(key)
        if cached is not None and cached[0] == mtime_ns and cached[1] == size:
            return cached[2]
    try:
        data = full.read_bytes()
    except OSError:
        return None
    sha8 = hashlib.sha256(data).hexdigest()[:8]
    with _STATIC_VERSION_LOCK:
        _STATIC_VERSION_CACHE[key] = (mtime_ns, size, sha8)
    return sha8


# Match script src= or link href= pointing at /static/... — capture the
# full attribute up to a (maybe-existing) querystring or fragment so we
# can splice ?v=... in cleanly. The pattern accepts single OR double
# quotes (HTML5 allows both) and tolerates extra whitespace around the
# = sign. Anchored on `<script` or `<link` so we don't accidentally
# rewrite text content that happens to contain `/static/` (e.g. inline
# string literals in a JS-heredoc would be safe).
#
# Group numbering (re.VERBOSE doesn't change capture indices, comments
# do):
#   1: full prefix through the opening quote: `<script src="`
#   2: opening quote char (back-referenced as group 5 to match closer)
#   3: clean /static/<path> with no querystring or fragment
#   4: any pre-existing querystring + fragment (discarded by replacer)
#   5: closing quote, asserted equal to opener via backreference
_STATIC_REWRITE_RE = re.compile(
    r"""
    (<(?:script|link)\b[^>]*?\s        # 1: opening tag of script or link
        (?:src|href)\s*=\s*             #    attribute we care about
        (['"]))                         # 2: opening quote (captured to re-emit)
    (/static/[^'"\#?]+)                 # 3: clean path (no existing query / fragment)
    ((?:\?[^'"\#]*)?(?:\#[^'"]*)?)      # 4: any pre-existing query+fragment
    (\2)                                # 5: closing quote (must match opener)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _rewrite_static_asset_urls(html: str) -> str:
    """Append ``?v=<sha>`` to every /static/... script + link tag in HTML.

    Idempotent over the served bytes — running it twice would just
    overwrite v= with the same value. Safe over already-versioned URLs
    (e.g. if the template was hand-edited to include `?v=foo`); we
    replace the whole querystring so a developer-pinned override gets
    blown away. That's intentional: cache-busting beats authoring
    intent, and there's no current use case for hand-written ?v=.
    """
    def _replace(m: re.Match) -> str:
        prefix = m.group(1)         # `<script src="` (incl. opening quote)
        path = m.group(3)           # `/static/foo.js`
        # m.group(4) is the existing querystring/fragment — discarded
        # so our ?v= wins. If the developer needs a fragment they can
        # re-add it on the path side, not via this rewriter.
        closer = m.group(5)         # closing quote
        ver = _static_asset_version(path)
        if not ver:
            return m.group(0)
        return f"{prefix}{path}?v={ver}{closer}"

    return _STATIC_REWRITE_RE.sub(_replace, html)


# ============================================================================
# Live-reload SSE endpoint + Anim Library endpoints (v5 polish, 2026-04-25)
# ----------------------------------------------------------------------------
# /api/events             -> Server-Sent Events stream (cache.changed events)
# /api/events/status      -> JSON snapshot of watcher state (debug + frontend
#                            badge can poll on cold start)
# /api/anim_library/list  -> All staged .njm + sidecar metadata
# /api/anim_library/delete-> Bulk delete N animations (njm + sidecar)
# /api/anim_library/zip   -> Stream a .zip of selected NJMs (batch deploy)
# /api/anim_library/rename-> Bulk rename N animations
# ============================================================================


@app.get("/api/events")
async def api_events(request: Request):
    """SSE stream of cache.changed events.

    Wire format::

        event: cache.changed
        data: {"path": "cache/njm_export/foo.njm", "kind": "create"}

    Plus a periodic ``event: heartbeat`` (every 25 s) so middlewares /
    proxies don't time out an idle connection. Connection lifetime is
    bound to the request — when the client disconnects, the subscriber
    queue is freed.
    """
    loop, q = _LIVE_RELOAD_HUB.subscribe()
    HEARTBEAT_SECONDS = 25.0

    async def gen():
        try:
            # Initial sync event so a freshly-connected frontend learns
            # the watcher is alive and gets a snapshot of subscriber id.
            yield (
                "event: ready\n"
                "data: " + json.dumps({"ok": True, "subscribers": _LIVE_RELOAD_HUB.subscriber_count()}) + "\n\n"
            )
            while True:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
                    continue
                yield "event: cache.changed\ndata: " + json.dumps(ev) + "\n\n"
        finally:
            _LIVE_RELOAD_HUB.unsubscribe(loop, q)

    headers = {
        "Cache-Control": "no-cache, no-store, no-transform, must-revalidate",
        "X-Accel-Buffering": "no",
        # Force chunked: text/event-stream + Connection close-on-disconnect.
        "Connection": "keep-alive",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@app.get("/api/events/status")
def api_events_status():
    """Watcher state snapshot. Used by the live-badge UI on cold start."""
    return _LIVE_RELOAD_HUB.snapshot_state()


@app.post("/api/events/rescan")
def api_events_rescan():
    """Force a full re-scan of watched dirs; dispatch any deltas now.

    Useful for the "Refresh from disk" button in the Anim Library tab and
    for tests that need deterministic delivery without waiting for the
    poll interval.
    """
    n = _LIVE_RELOAD_HUB.force_rescan()
    return {"ok": True, "events_fired": n}


# ----------------------------------------------------------------------------
# Anim Library endpoints — global view of cache/njm_export/.
# ----------------------------------------------------------------------------
def _safe_njm_lib_name(name: str) -> str:
    """Validate ``name`` is a bare .njm filename (no path components).

    Mirrors _safe_njm_export_name + _safe_njm_export_name's checks.
    Reject everything outside the staging dir.
    """
    bare = Path(name).name
    if bare != name:
        raise HTTPException(400, "name: must be a bare filename")
    if not re.match(r"^[A-Za-z0-9_\-.]+$", bare):
        raise HTTPException(400, "name: contains forbidden characters")
    if not bare.lower().endswith(".njm"):
        raise HTTPException(400, "name: must end with .njm")
    return bare


def _anim_library_entry(njm: Path) -> Optional[dict]:
    """Project one staged .njm + sidecar into the library wire shape.

    Returns None if the .njm is unreadable. Sidecar absence is tolerated
    (the entry will have empty source/target fields).
    """
    try:
        st = njm.stat()
    except OSError:
        return None
    sidecar = _load_preview_sidecar(njm)
    sidecar = sidecar or {}
    md5 = sidecar.get("njm_md5")
    if not md5:
        # Compute on demand for non-preview njms (e.g. from anim_keyframe/save).
        try:
            md5 = hashlib.md5(njm.read_bytes()).hexdigest()
        except OSError:
            md5 = ""
    display = njm.name
    if display.lower().endswith(".njm"):
        display = display[:-4]
    target_path = sidecar.get("target_model_path") or ""
    if target_path:
        # target_model_path may carry an inner-asset suffix like
        # "bm_boss1_dragon.bml#dragon_body". The basename for the UI
        # filter is the BML file alone — the inner part identifies
        # which mesh got the swap, not the file the user thinks of.
        target_basename = Path(target_path.split("#", 1)[0]).name
    else:
        target_basename = ""
    return {
        "name": njm.name,
        "display_name": display,
        "size": int(st.st_size),
        "mtime_ms": int(st.st_mtime * 1000),
        "md5": md5,
        # Sidecar-derived metadata (may be empty for legacy .njms).
        "frame_count": int(sidecar.get("frame_count") or 0),
        "bone_count": int(sidecar.get("bone_count") or 0),
        "fps": float(sidecar.get("fps") or 30.0),
        "source_glb": sidecar.get("source_glb") or "",
        "source_animation": sidecar.get("source_animation") or "",
        "target_model_path": target_path,
        "target_model_name": target_basename,
        "retargeted_at_ms": int(sidecar.get("retargeted_at_ms") or 0),
        "retargeted_bones": int(sidecar.get("retargeted_bones") or 0),
        "dropped_bones": int(sidecar.get("dropped_bones") or 0),
        "bone_map": sidecar.get("bone_map") or "",
        "has_sidecar": bool(sidecar),
    }


@app.get("/api/anim_library/list")
def api_anim_library_list():
    """Return every staged animation in cache/njm_export/.

    Wire shape::

        {
          "count": <int>,
          "items": [<entry>, ...],
          "totals": {"size": <bytes>, "with_sidecar": <int>}
        }

    ``items`` are sorted most-recent-first by mtime so the UI doesn't have
    to re-sort. The frontend handles search/filter client-side because
    the typical population is well under 10k entries.
    """
    items: list[dict] = []
    total_size = 0
    with_sidecar = 0
    if NJM_EXPORT_DIR.is_dir():
        for njm in sorted(NJM_EXPORT_DIR.glob("*.njm")):
            entry = _anim_library_entry(njm)
            if entry is None:
                continue
            items.append(entry)
            total_size += entry["size"]
            if entry["has_sidecar"]:
                with_sidecar += 1
    items.sort(key=lambda e: max(e["mtime_ms"], e["retargeted_at_ms"]), reverse=True)
    return {
        "count": len(items),
        "items": items,
        "totals": {
            "size": total_size,
            "with_sidecar": with_sidecar,
        },
    }


class AnimLibraryDeleteReq(BaseModel):
    names: list[str] = Field(default_factory=list)


@app.post("/api/anim_library/delete")
def api_anim_library_delete(req: AnimLibraryDeleteReq):
    """Bulk-delete N animations + their .preview.json sidecars.

    Each name must be a bare filename ending in ``.njm`` (rejects path
    components / traversal). Idempotent — already-missing files are not
    an error and report as ``removed: false``.
    """
    if not isinstance(req.names, list):
        raise HTTPException(400, "names: must be a list")
    if len(req.names) > 1000:
        raise HTTPException(400, "names: too many (max 1000)")
    results: list[dict] = []
    for raw in req.names:
        try:
            name = _safe_njm_lib_name(raw)
        except HTTPException as e:
            results.append({"name": str(raw), "removed": False, "error": e.detail})
            continue
        njm_p = NJM_EXPORT_DIR / name
        sidecar_p = NJM_EXPORT_DIR / (name + ".preview.json")
        removed_paths: list[str] = []
        err: Optional[str] = None
        for p in (njm_p, sidecar_p):
            if p.is_file():
                try:
                    p.unlink()
                    removed_paths.append(str(p))
                except OSError as e:
                    err = f"unlink {p}: {e}"
                    break
        results.append({
            "name": name,
            "removed": bool(removed_paths) and err is None,
            "removed_paths": removed_paths,
            "error": err,
        })
    log.info("anim_library/delete: %d requested, %d removed",
             len(req.names), sum(1 for r in results if r["removed"]))
    return {"ok": True, "results": results}


class AnimLibraryRenameItem(BaseModel):
    old_name: str
    new_name: str


class AnimLibraryRenameReq(BaseModel):
    renames: list[AnimLibraryRenameItem] = Field(default_factory=list)


@app.post("/api/anim_library/rename")
def api_anim_library_rename(req: AnimLibraryRenameReq):
    """Bulk-rename N animations + their sidecars.

    Each rename validates both old and new names. Skips when the target
    name already exists (so the operation is safely re-runnable). Each
    result entry reports ``renamed: bool`` so the UI can show partial
    success.
    """
    if not isinstance(req.renames, list):
        raise HTTPException(400, "renames: must be a list")
    if len(req.renames) > 1000:
        raise HTTPException(400, "renames: too many (max 1000)")
    results: list[dict] = []
    for item in req.renames:
        try:
            old_name = _safe_njm_lib_name(item.old_name)
            new_name = _safe_njm_lib_name(item.new_name)
        except HTTPException as e:
            results.append({
                "old_name": item.old_name, "new_name": item.new_name,
                "renamed": False, "error": e.detail,
            })
            continue
        if old_name == new_name:
            results.append({
                "old_name": old_name, "new_name": new_name,
                "renamed": False, "error": "same name",
            })
            continue
        old_p = NJM_EXPORT_DIR / old_name
        new_p = NJM_EXPORT_DIR / new_name
        if not old_p.is_file():
            results.append({
                "old_name": old_name, "new_name": new_name,
                "renamed": False, "error": "source missing",
            })
            continue
        if new_p.exists():
            results.append({
                "old_name": old_name, "new_name": new_name,
                "renamed": False, "error": "target exists",
            })
            continue
        try:
            os.replace(old_p, new_p)
        except OSError as e:
            results.append({
                "old_name": old_name, "new_name": new_name,
                "renamed": False, "error": f"rename failed: {e}",
            })
            continue
        # Move sidecar if present. Don't fail the whole rename on sidecar
        # weirdness; just report.
        old_sc = NJM_EXPORT_DIR / (old_name + ".preview.json")
        new_sc = NJM_EXPORT_DIR / (new_name + ".preview.json")
        sc_moved = False
        if old_sc.is_file():
            try:
                os.replace(old_sc, new_sc)
                sc_moved = True
            except OSError as e:
                log.warning("rename sidecar %s -> %s: %s", old_sc, new_sc, e)
        results.append({
            "old_name": old_name, "new_name": new_name,
            "renamed": True, "sidecar_moved": sc_moved,
        })
    log.info("anim_library/rename: %d requested, %d renamed",
             len(req.renames), sum(1 for r in results if r["renamed"]))
    return {"ok": True, "results": results}


class AnimLibraryZipReq(BaseModel):
    names: list[str] = Field(default_factory=list)


@app.post("/api/anim_library/zip")
def api_anim_library_zip(req: AnimLibraryZipReq):
    """Build a .zip of N animations (njm + sidecar) and stream it back.

    Output is a single zip with the structure::

        njm/<name>.njm
        sidecar/<name>.njm.preview.json   (only if sidecar exists)

    Uses ZIP_STORED rather than ZIP_DEFLATED — NJMs are already tightly
    packed binary; CPU spent compressing pays back ~2 % at most.
    """
    import zipfile
    if not isinstance(req.names, list):
        raise HTTPException(400, "names: must be a list")
    if not req.names:
        raise HTTPException(400, "names: empty")
    if len(req.names) > 1000:
        raise HTTPException(400, "names: too many (max 1000)")
    safe_names: list[str] = []
    for raw in req.names:
        safe_names.append(_safe_njm_lib_name(raw))
    buf = BytesIO()
    found = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in safe_names:
            njm_p = NJM_EXPORT_DIR / name
            if not njm_p.is_file():
                continue
            try:
                zf.writestr(f"njm/{name}", njm_p.read_bytes())
            except OSError as e:
                log.warning("zip read %s: %s", njm_p, e)
                continue
            found += 1
            sc_p = NJM_EXPORT_DIR / (name + ".preview.json")
            if sc_p.is_file():
                try:
                    zf.writestr(f"sidecar/{name}.preview.json", sc_p.read_bytes())
                except OSError:
                    pass
    if found == 0:
        raise HTTPException(404, "no requested animations exist on disk")
    payload = buf.getvalue()
    log.info("anim_library/zip: packed %d/%d into %d bytes", found, len(safe_names), len(payload))
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="anim_library_{found}.zip"',
            "X-Anim-Count": str(found),
        },
    )


@app.get("/")
def index():
    """Serve the SPA index with cache-busted /static/* URLs."""
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return JSONResponse({"error": "index.html missing"}, status_code=500)
    try:
        html = idx.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("index: failed to read %s: %s", idx, e)
        return FileResponse(idx)
    rewritten = _rewrite_static_asset_urls(html)
    # no-store on the HTML itself — we want every page-load to re-read
    # the (small) HTML, see the latest fingerprints, and only THEN make
    # the (cacheable) /static/* requests. Without this header, the
    # browser would happily cache the rewritten HTML for hours and still
    # serve old script-tag URLs after a server-side edit.
    return Response(
        content=rewritten,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    # Wave 7 (2026-04-26): worker count is configurable via
    # PSO_UVICORN_WORKERS. Default = 1 (single-process, all caches in
    # one address space — historic behaviour). 4 has been validated:
    # export tokens persist via sidecar JSON (so a token minted by
    # worker A is fetchable from worker B), and every other cache
    # (parse_cache / binding_cache / skinned_payload / tile_png) is
    # content-keyed by (path, mtime, size) so each worker maintains
    # its own LRU with no cross-worker invalidation needed.
    #
    # The live-reload watcher polls per-worker (4× CPU on a tiny
    # poll loop, sub-percent-load); SSE clients connect to whichever
    # worker the load balancer hands them, and each worker's poll
    # detects file changes independently — events are delivered as
    # long as ANY worker sees the change.
    workers_env = os.environ.get("PSO_UVICORN_WORKERS")
    workers = int(workers_env) if workers_env else 1
    # Startup banner intentionally on stdout (not via the logger) so it
    # appears even if logging is reconfigured by uvicorn before lifespan runs.
    print(f"PSOBB Studio v{VERSION} - http://127.0.0.1:8765")
    print(f"  data:  {DATA_DIR}")
    print(f"  cache: {CACHE_DIR}")
    print(f"  models from: {REALESRGAN_MODELS}")
    print(f"  workers: {workers}")
    if workers > 1:
        # uvicorn's multi-worker mode requires an importable factory
        # string, NOT a live `app` instance. Convert to the string
        # form. The current module is `server` (assumed to be in PYTHONPATH).
        uvicorn.run(
            "server:app",
            host="127.0.0.1",
            port=8765,
            log_level="info",
            workers=workers,
        )
    else:
        uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
