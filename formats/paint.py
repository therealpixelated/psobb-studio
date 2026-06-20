"""UV-space brush stamp + flood-fill helpers for the Paint MVP.

This module is intentionally side-effect-free — every function takes raw
RGBA byte buffers (or PIL Images) and returns transformed bytes. The
server-side use is the round-trip "load source PNG -> paint -> save PNG"
flow; the front-end uses an exact JS port (in ``static/paint_panel.js``)
so the math has to be deterministic.

UV convention
=============
Three.js feeds raycaster hits as ``(u, v)`` in ``[0, 1]^2`` with origin
at the BOTTOM-LEFT of the texture (the WebGL convention). Internal pixel
coordinates here use ``(px, py)`` with origin at the TOP-LEFT (the PIL /
PNG convention). The conversion is::

    px = round(u * (w - 1))
    py = round((1.0 - v) * (h - 1))

Brush stamp
-----------
Square stamp with a Gaussian alpha falloff — closed-form for cheap
recompute. ``stamp_circle`` returns an ``H x W x 4`` byte buffer ready
for ``alpha_over``.

Edge handling
-------------
The stamp can land near a UV seam (edge of the texture). We DON'T wrap
to the opposite side (would bleed a stripe across unrelated meshes that
share the texture); instead we clip the stamp to the texture bounds.
The frontend follows the same rule.
"""
from __future__ import annotations

import math
import struct
from typing import Iterable, Optional

# Stdlib only when possible; PIL is only used in the optional convenience
# wrappers since it's already a server dep (see server.py top imports).
try:  # pragma: no cover - import guard
    from PIL import Image
    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False


# ---------------------------------------------------------------------------
# UV math
# ---------------------------------------------------------------------------
def uv_to_pixel(u: float, v: float, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized UV to top-left integer pixel coordinates.

    The Three.js / WebGL UV origin is bottom-left; PNG / PIL origin is
    top-left. We flip ``v`` here so callers can treat the result as a
    PIL `(x, y)` pair directly.

    UVs outside [0,1] are clamped (no wrap). Out-of-range hits in the
    front-end's raycaster shouldn't occur for valid meshes, but if they
    do we'd rather paint at the edge than crash.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive (got {width}, {height})")
    # Clamp into [0, 1].
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    px = int(round(u * (width - 1)))
    py = int(round((1.0 - v) * (height - 1)))
    return px, py


def pixel_to_uv(px: int, py: int, width: int, height: int) -> tuple[float, float]:
    """Inverse of :func:`uv_to_pixel`. Used by the colour-picker tool."""
    if width <= 1 or height <= 1:
        return 0.0, 0.0
    u = px / (width - 1)
    v = 1.0 - (py / (height - 1))
    return u, v


# ---------------------------------------------------------------------------
# Brush stamp
# ---------------------------------------------------------------------------
def stamp_circle(
    radius: float,
    rgba: tuple[int, int, int, int],
    *,
    hardness: float = 0.5,
) -> bytes:
    """Generate a square Gaussian-falloff stamp.

    Returns an ``H x W x 4`` raw RGBA byte string where ``H == W ==
    int(ceil(radius * 2)) + 1``. Centre pixel is alpha 255 * input_a;
    edges fall off following ``exp(-(d/r)^2 / (1 - hardness))`` so
    ``hardness=0`` gives a soft brush, ``hardness=1`` gives a hard disc.

    The colour is stored verbatim per pixel; the alpha is multiplied by
    the falloff (so the call site can do straight alpha-over without
    pre-multiplication).
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive (got {radius})")
    if not (0.0 <= hardness <= 1.0):
        raise ValueError(f"hardness must be in [0, 1] (got {hardness})")
    r = int(math.ceil(radius))
    w = h = r * 2 + 1
    cx = cy = r
    cr, cg, cb, ca = rgba
    # Falloff parameter — at hardness=1 we want a near-step function;
    # at hardness=0 we want a soft Gaussian. ``inv_sigma2`` controls
    # how fast the curve falls.
    soft = 1.0 - hardness
    # Avoid divide-by-zero: clamp ``soft`` to a minimum, fall back to
    # a hard step when hardness >= 0.999.
    out = bytearray(w * h * 4)
    r_squared = radius * radius
    soft_squared = max(soft * soft, 1e-6)
    for y in range(h):
        dy = y - cy
        for x in range(w):
            dx = x - cx
            d2 = dx * dx + dy * dy
            if d2 > r_squared:
                # Outside the bounding circle — fully transparent.
                a = 0.0
            else:
                # Gaussian: a = exp(-(d^2 / r^2) / soft^2). At d=r,
                # a = exp(-1/soft^2). Hardness=1 -> a~0 at the edge;
                # hardness=0 -> a~exp(-1) ≈ 0.37 — still soft, fades
                # smoothly past the radius.
                if hardness >= 0.999:
                    a = 1.0
                else:
                    norm = (d2 / r_squared) / soft_squared
                    a = math.exp(-norm)
            i = (y * w + x) * 4
            out[i + 0] = cr
            out[i + 1] = cg
            out[i + 2] = cb
            out[i + 3] = int(round(a * ca))
    return bytes(out)


# ---------------------------------------------------------------------------
# Alpha-over compositing
# ---------------------------------------------------------------------------
def alpha_over(
    dst: bytearray,
    dst_w: int,
    dst_h: int,
    stamp: bytes,
    stamp_w: int,
    stamp_h: int,
    cx: int,
    cy: int,
    *,
    opacity: float = 1.0,
    erase: bool = False,
) -> tuple[int, int, int, int]:
    """Alpha-blend a stamp onto a destination RGBA buffer in-place.

    ``dst`` is mutated. Coordinates ``(cx, cy)`` are the centre of the
    stamp on the destination (top-left origin). ``opacity`` is a scalar
    multiplier on the stamp's alpha (0..1).

    When ``erase`` is True, the stamp is interpreted as an inverse mask:
    each destination pixel's alpha is reduced by the stamp's alpha.

    Returns the bounding rect on the DESTINATION that was actually
    touched, as ``(x0, y0, x1, y1)`` half-open. ``(x0, y0)`` is inclusive,
    ``(x1, y1)`` exclusive. Caller can use this to upload only a sub-rect
    of the texture to the GPU.
    """
    if len(dst) != dst_w * dst_h * 4:
        raise ValueError(f"dst size mismatch: {len(dst)} != {dst_w}*{dst_h}*4")
    if len(stamp) != stamp_w * stamp_h * 4:
        raise ValueError(f"stamp size mismatch: {len(stamp)} != {stamp_w}*{stamp_h}*4")
    if not (0.0 <= opacity <= 1.0):
        raise ValueError(f"opacity must be in [0, 1] (got {opacity})")
    rx = stamp_w // 2
    ry = stamp_h // 2
    # Clip the stamp rect to the destination's bounds.
    x0 = max(0, cx - rx)
    y0 = max(0, cy - ry)
    x1 = min(dst_w, cx - rx + stamp_w)
    y1 = min(dst_h, cy - ry + stamp_h)
    if x0 >= x1 or y0 >= y1:
        # Stamp landed entirely outside the texture.
        return (0, 0, 0, 0)
    sx0 = x0 - (cx - rx)
    sy0 = y0 - (cy - ry)
    op = opacity
    for y in range(y1 - y0):
        dy = y0 + y
        sy = sy0 + y
        for x in range(x1 - x0):
            dx = x0 + x
            sx = sx0 + x
            si = (sy * stamp_w + sx) * 4
            di = (dy * dst_w + dx) * 4
            sa = stamp[si + 3] * op / 255.0
            if sa <= 0.0:
                continue
            if erase:
                # Multiply destination alpha by (1 - stamp_alpha) — the
                # canonical "soft eraser" rule.
                da_new = dst[di + 3] * (1.0 - sa)
                dst[di + 3] = int(round(max(0.0, min(255.0, da_new))))
                continue
            sr = stamp[si + 0]
            sg = stamp[si + 1]
            sb = stamp[si + 2]
            dr = dst[di + 0]
            dg = dst[di + 1]
            db = dst[di + 2]
            da = dst[di + 3]
            # Straight alpha-over (Porter-Duff "over"):
            #   out_a   = sa + da * (1 - sa)
            #   out_rgb = (sr*sa + dr*da*(1-sa)) / out_a
            inv = 1.0 - sa
            out_a = sa + (da / 255.0) * inv
            if out_a <= 0.0:
                # Fully-transparent result.
                dst[di + 0] = 0
                dst[di + 1] = 0
                dst[di + 2] = 0
                dst[di + 3] = 0
                continue
            # Composite RGB.
            out_r = (sr * sa + dr * (da / 255.0) * inv) / out_a
            out_g = (sg * sa + dg * (da / 255.0) * inv) / out_a
            out_b = (sb * sa + db * (da / 255.0) * inv) / out_a
            dst[di + 0] = int(round(max(0.0, min(255.0, out_r))))
            dst[di + 1] = int(round(max(0.0, min(255.0, out_g))))
            dst[di + 2] = int(round(max(0.0, min(255.0, out_b))))
            dst[di + 3] = int(round(max(0.0, min(255.0, out_a * 255.0))))
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------
def flood_fill(
    buf: bytearray,
    width: int,
    height: int,
    sx: int,
    sy: int,
    fill_rgba: tuple[int, int, int, int],
    *,
    tolerance: int = 8,
) -> int:
    """4-connected flood-fill in RGBA buffer space.

    Replaces every pixel reachable from ``(sx, sy)`` whose ``max
    (|dr|, |dg|, |db|, |da|)`` from the seed colour is <= ``tolerance``
    with ``fill_rgba``. Returns the number of pixels filled.

    Iterative scan-line implementation so we don't blow Python's
    recursion limit on large fills (1024x1024 textures = up to ~1M cells).
    """
    if not (0 <= sx < width and 0 <= sy < height):
        return 0
    if len(buf) != width * height * 4:
        raise ValueError(f"buf size mismatch: {len(buf)} != {width}*{height}*4")
    seed_i = (sy * width + sx) * 4
    sr, sg, sb, sa = buf[seed_i], buf[seed_i + 1], buf[seed_i + 2], buf[seed_i + 3]
    fr, fg, fb, fa = fill_rgba
    if (sr, sg, sb, sa) == (fr, fg, fb, fa):
        return 0  # already that colour — no-op, prevents infinite loop too

    def matches(idx: int) -> bool:
        return (
            abs(buf[idx] - sr) <= tolerance
            and abs(buf[idx + 1] - sg) <= tolerance
            and abs(buf[idx + 2] - sb) <= tolerance
            and abs(buf[idx + 3] - sa) <= tolerance
        )

    filled = 0
    # Stack of seed pixels to scan from. Scan-line fill walks left/right
    # then queues seeds for the row above + below.
    stack: list[tuple[int, int]] = [(sx, sy)]
    while stack:
        x, y = stack.pop()
        idx = (y * width + x) * 4
        if not matches(idx):
            continue
        # Walk left.
        lx = x
        while lx >= 0:
            li = (y * width + lx) * 4
            if not matches(li):
                break
            lx -= 1
        lx += 1
        # Walk right.
        rx = x
        while rx < width:
            ri = (y * width + rx) * 4
            if not matches(ri):
                break
            rx += 1
        rx -= 1
        # Fill the [lx, rx] span on row y, queuing seeds for adjacent rows.
        prev_above = False
        prev_below = False
        for cx in range(lx, rx + 1):
            ci = (y * width + cx) * 4
            buf[ci + 0] = fr
            buf[ci + 1] = fg
            buf[ci + 2] = fb
            buf[ci + 3] = fa
            filled += 1
            if y > 0:
                ai = ((y - 1) * width + cx) * 4
                if matches(ai):
                    if not prev_above:
                        stack.append((cx, y - 1))
                        prev_above = True
                else:
                    prev_above = False
            if y + 1 < height:
                bi = ((y + 1) * width + cx) * 4
                if matches(bi):
                    if not prev_below:
                        stack.append((cx, y + 1))
                        prev_below = True
                else:
                    prev_below = False
    return filled


# ---------------------------------------------------------------------------
# Smear (drag-direction average)
# ---------------------------------------------------------------------------
def smear_stamp(
    dst: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    radius: int,
    dx: int,
    dy: int,
    *,
    strength: float = 0.5,
) -> tuple[int, int, int, int]:
    """In-place smear: pull pixels at (cx-dx, cy-dy) toward (cx, cy).

    Implements a simple finger-smudge by sampling the destination at
    ``(cx-dx, cy-dy)`` for each destination pixel within the radius
    around ``(cx, cy)``, then alpha-blending that sample over the
    destination at strength ``strength``.

    Returns the bounding rect actually touched (same shape as
    :func:`alpha_over`). When the drag has zero length this is a no-op.
    """
    if radius <= 0:
        return (0, 0, 0, 0)
    if dx == 0 and dy == 0:
        return (0, 0, 0, 0)
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be in [0, 1] (got {strength})")
    if len(dst) != width * height * 4:
        raise ValueError(f"dst size mismatch: {len(dst)} != {width}*{height}*4")
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(width, cx + radius + 1)
    y1 = min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return (0, 0, 0, 0)
    # Snapshot the source rect so iteration doesn't see its own writes.
    snap = bytes(dst)
    r_sq = radius * radius
    for py in range(y0, y1):
        for px in range(x0, x1):
            ddx = px - cx
            ddy = py - cy
            if ddx * ddx + ddy * ddy > r_sq:
                continue
            sx = px - dx
            sy = py - dy
            if sx < 0 or sx >= width or sy < 0 or sy >= height:
                continue
            si = (sy * width + sx) * 4
            di = (py * width + px) * 4
            for c in range(4):
                src_v = snap[si + c]
                dst_v = dst[di + c]
                # Linear blend src_v over dst_v at `strength`.
                dst[di + c] = int(round(dst_v + (src_v - dst_v) * strength))
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# PIL convenience wrappers (server-side helpers)
# ---------------------------------------------------------------------------
def png_bytes_to_rgba(buf: bytes) -> tuple[bytearray, int, int]:
    """Decode a PNG buffer to (rgba_bytes, width, height) using PIL."""
    if not HAS_PIL:
        raise RuntimeError("PIL not available; cannot decode PNG")
    from io import BytesIO
    im = Image.open(BytesIO(buf)).convert("RGBA")
    w, h = im.size
    return bytearray(im.tobytes()), w, h


def rgba_to_png_bytes(buf: bytes, width: int, height: int) -> bytes:
    """Encode raw RGBA bytes back into a PNG."""
    if not HAS_PIL:
        raise RuntimeError("PIL not available; cannot encode PNG")
    from io import BytesIO
    im = Image.frombuffer("RGBA", (width, height), bytes(buf), "raw", "RGBA", 0, 1)
    out = BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def safe_painted_basename(model_path: str, inner: str) -> str:
    """Build the disk basename for a painted texture cache entry.

    Format: ``<base>__<inner>.png``. The ``base`` is the host archive
    name (e.g. ``bm_ene_bm9_s_mericarol.bml``), the ``inner`` is the
    in-archive entry name (e.g. ``bm_ene_bm9_s_mericarol.nj.xvm``).
    Path separators and ``#`` are stripped to keep the result
    filesystem-safe; the ``__`` joiner is reserved as the unique
    base/inner separator (PSOBB asset names never contain ``__``).
    """
    base_safe = (
        model_path.replace("/", "_")
        .replace("\\", "_")
        .replace("#", "_")
    )
    inner_safe = (
        inner.replace("/", "_")
        .replace("\\", "_")
        .replace("#", "_")
    )
    if not inner_safe.lower().endswith(".png"):
        inner_safe = inner_safe + ".png"
    return f"{base_safe}__{inner_safe}"


def safe_painted_dirname(model_path: str, inner: str) -> str:
    """Build the disk directory name for a multi-layer painted texture.

    v5 introduced layer stacks. Each painted texture now lives in its
    own subdirectory under ``cache/painted_textures/`` named
    ``<safe>``. The directory holds:

      * ``manifest.json`` — layer order, blend modes, opacity, etc.
      * ``<idx>.png`` — RGBA bytes for layer ``<idx>``.
      * ``<idx>_mask.png`` — optional alpha mask for layer ``<idx>``.

    The legacy single-PNG format (``<safe>.png``) still works — when
    the server sees the file but no directory it auto-converts on
    first save (see ``server.api_paint_save``).
    """
    return safe_painted_basename(model_path, inner)[: -len(".png")]


# ---------------------------------------------------------------------------
# Layer manifest
# ---------------------------------------------------------------------------
SUPPORTED_BLEND_MODES = ("normal", "multiply", "screen", "overlay")
MANIFEST_VERSION = 1


def make_default_manifest(
    *, model_path: str, inner: str, width: int, height: int,
) -> dict:
    """Build a fresh manifest dict for a brand-new layer stack.

    A v5 manifest has shape::

        {
          "version": 1,
          "model_path": "...",
          "inner": "...",
          "width": int, "height": int,
          "active": 0,
          "layers": [
            {"idx": 0, "name": "Background",
             "visible": True, "opacity": 1.0,
             "blend_mode": "normal", "locked": False,
             "has_mask": False}
          ]
        }
    """
    return {
        "version": MANIFEST_VERSION,
        "model_path": model_path,
        "inner": inner,
        "width": int(width),
        "height": int(height),
        "active": 0,
        "layers": [
            {
                "idx": 0,
                "name": "Background",
                "visible": True,
                "opacity": 1.0,
                "blend_mode": "normal",
                "locked": False,
                "has_mask": False,
            },
        ],
    }


def validate_manifest(m: dict) -> dict:
    """Normalize + validate a layer manifest dict (mutates a copy).

    Rejects unknown blend modes, clamps opacity to [0, 1], makes sure
    every layer has a unique ``idx``, and re-numbers ``active`` if it
    points outside the layer list. Returns the validated dict.
    """
    if not isinstance(m, dict):
        raise ValueError("manifest must be a dict")
    out = dict(m)
    out["version"] = int(out.get("version", MANIFEST_VERSION))
    out["width"] = int(out.get("width", 0))
    out["height"] = int(out.get("height", 0))
    layers_in = out.get("layers")
    if not isinstance(layers_in, list) or not layers_in:
        raise ValueError("manifest.layers must be a non-empty list")
    seen: set[int] = set()
    norm_layers = []
    for L in layers_in:
        if not isinstance(L, dict):
            raise ValueError("layer must be a dict")
        idx = int(L.get("idx", -1))
        if idx < 0:
            raise ValueError("layer.idx must be >= 0")
        if idx in seen:
            raise ValueError(f"duplicate layer idx {idx}")
        seen.add(idx)
        bm = str(L.get("blend_mode", "normal")).lower()
        if bm not in SUPPORTED_BLEND_MODES:
            raise ValueError(f"unsupported blend_mode {bm!r}")
        op = float(L.get("opacity", 1.0))
        op = 0.0 if op < 0.0 else (1.0 if op > 1.0 else op)
        norm_layers.append({
            "idx": idx,
            "name": str(L.get("name", f"Layer {idx}"))[:64],
            "visible": bool(L.get("visible", True)),
            "opacity": op,
            "blend_mode": bm,
            "locked": bool(L.get("locked", False)),
            "has_mask": bool(L.get("has_mask", False)),
        })
    out["layers"] = norm_layers
    active = int(out.get("active", 0))
    valid_indices = [L["idx"] for L in norm_layers]
    if active not in valid_indices:
        active = norm_layers[0]["idx"]
    out["active"] = active
    return out


# ---------------------------------------------------------------------------
# Blend mode math
# ---------------------------------------------------------------------------
def _blend_pixel(
    sr: float, sg: float, sb: float, sa: float,
    dr: float, dg: float, db: float, da: float,
    mode: str, opacity: float,
) -> tuple[float, float, float, float]:
    """Apply a single Photoshop-style blend mode pixel.

    All inputs are normalized [0, 1] floats. Returns straight RGBA
    (not premultiplied). The blend math operates on RGB only;
    final alpha = source_alpha * opacity composited over destination
    alpha (Porter-Duff "over" semantics across all modes — only the
    RGB combiner changes).

    Reference formulae (matches Photoshop / GIMP):

      * normal     : C = S
      * multiply   : C = S * D
      * screen     : C = 1 - (1 - S) * (1 - D)
      * overlay    : C = D < 0.5 ? 2*S*D : 1 - 2*(1-S)*(1-D)
    """
    eff_a = sa * opacity
    if eff_a <= 0.0:
        return dr, dg, db, da
    if mode == "multiply":
        br, bg, bb = sr * dr, sg * dg, sb * db
    elif mode == "screen":
        br = 1.0 - (1.0 - sr) * (1.0 - dr)
        bg = 1.0 - (1.0 - sg) * (1.0 - dg)
        bb = 1.0 - (1.0 - sb) * (1.0 - db)
    elif mode == "overlay":
        br = (2.0 * sr * dr) if dr < 0.5 else (1.0 - 2.0 * (1.0 - sr) * (1.0 - dr))
        bg = (2.0 * sg * dg) if dg < 0.5 else (1.0 - 2.0 * (1.0 - sg) * (1.0 - dg))
        bb = (2.0 * sb * db) if db < 0.5 else (1.0 - 2.0 * (1.0 - sb) * (1.0 - db))
    else:  # normal (default)
        br, bg, bb = sr, sg, sb
    # Porter-Duff "over": dst is the existing accumulated RGBA, B is the
    # blended source RGB at full strength of S, alpha-overed with eff_a.
    inv = 1.0 - eff_a
    out_a = eff_a + da * inv
    if out_a <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    out_r = (br * eff_a + dr * da * inv) / out_a
    out_g = (bg * eff_a + dg * da * inv) / out_a
    out_b = (bb * eff_a + db * da * inv) / out_a
    return out_r, out_g, out_b, out_a


def composite_layers(
    layers: Iterable[tuple[bytes, dict, Optional[bytes]]],
    width: int,
    height: int,
) -> bytearray:
    """Composite a stack of layers (bottom -> top) into a single RGBA buf.

    Each entry is ``(rgba_bytes, layer_meta, mask_bytes_or_None)``.
    ``layer_meta`` is one of the dicts from a validated manifest's
    ``layers`` list. The mask, when present, is a single-channel
    alpha buffer (``width * height`` bytes; white = visible,
    black = hidden) — internally we treat the R channel of an RGBA
    encoding as the mask value too, for ease of round-trip.

    Layers with ``visible == False`` are skipped. Layers with
    ``opacity <= 0`` are skipped. The very first visible layer is
    painted onto a fully-transparent buffer; subsequent visible
    layers stack with their blend mode on top.

    The result is a fresh ``bytearray`` of ``width * height * 4``
    bytes. Caller can pass that to :func:`rgba_to_png_bytes` for
    serialization, or hand it to ``THREE.CanvasTexture`` from JS via
    the API.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive (got {width}, {height})")
    n = width * height
    out = bytearray(n * 4)  # transparent black
    any_drawn = False
    for rgba, meta, mask in layers:
        if len(rgba) != n * 4:
            raise ValueError(f"layer rgba size mismatch: {len(rgba)} != {n}*4")
        if not meta.get("visible", True):
            continue
        opacity = float(meta.get("opacity", 1.0))
        if opacity <= 0.0:
            continue
        mode = str(meta.get("blend_mode", "normal")).lower()
        if mode not in SUPPORTED_BLEND_MODES:
            mode = "normal"
        # Mask: per-pixel alpha multiplier from the R channel of an RGBA mask.
        has_mask = mask is not None and len(mask) == n * 4
        if not any_drawn:
            # First visible layer: copy in (with opacity + mask + alpha
            # contribution). No need to invoke the full blend pipeline.
            for i in range(n):
                bi = i * 4
                m = (mask[bi] / 255.0) if has_mask else 1.0
                a = (rgba[bi + 3] / 255.0) * opacity * m
                if a <= 0.0:
                    continue
                out[bi + 0] = rgba[bi + 0]
                out[bi + 1] = rgba[bi + 1]
                out[bi + 2] = rgba[bi + 2]
                out[bi + 3] = int(round(max(0.0, min(255.0, a * 255.0))))
            any_drawn = True
            continue
        for i in range(n):
            bi = i * 4
            sr = rgba[bi + 0] / 255.0
            sg = rgba[bi + 1] / 255.0
            sb = rgba[bi + 2] / 255.0
            sa = rgba[bi + 3] / 255.0
            if has_mask:
                sa *= mask[bi] / 255.0
            if sa <= 0.0:
                continue
            dr = out[bi + 0] / 255.0
            dg = out[bi + 1] / 255.0
            db = out[bi + 2] / 255.0
            da = out[bi + 3] / 255.0
            nr, ng, nb, na = _blend_pixel(sr, sg, sb, sa, dr, dg, db, da, mode, opacity)
            out[bi + 0] = int(round(max(0.0, min(255.0, nr * 255.0))))
            out[bi + 1] = int(round(max(0.0, min(255.0, ng * 255.0))))
            out[bi + 2] = int(round(max(0.0, min(255.0, nb * 255.0))))
            out[bi + 3] = int(round(max(0.0, min(255.0, na * 255.0))))
    return out


# ---------------------------------------------------------------------------
# Clone stamp (Alt+click source -> drag dest)
# ---------------------------------------------------------------------------
def clone_stamp(
    dst: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    radius: int,
    *,
    src_buf: bytes,
    src_w: int,
    src_h: int,
    src_offset_x: int,
    src_offset_y: int,
    opacity: float = 1.0,
    hardness: float = 0.5,
) -> tuple[int, int, int, int]:
    """Stamp a circle of pixels from a source buffer at an offset.

    The user picks a SOURCE point (Alt+click) and a DEST point (first
    drag click). We carry the offset ``(src_offset_x, src_offset_y)``
    such that ``dest_pixel := source_pixel + offset``. For each
    destination pixel inside the brush radius:

        src_x = cx + dx - src_offset_x
        src_y = cy + dy - src_offset_y

    The stamped pixel is alpha-overed onto ``dst`` with a Gaussian
    falloff (matches :func:`stamp_circle`). When the source is the
    SAME layer as the destination, callers should snapshot ``dst``
    BEFORE calling so the read doesn't see in-progress writes
    (matches :func:`smear_stamp`'s self-snapshot rule).

    ``src_buf`` may equal ``dst`` (in bytes) when source/dest are the
    same layer. Otherwise it's a different layer's RGBA buffer.

    Returns the touched-rect bounds on ``dst`` (half-open).
    """
    if radius <= 0:
        return (0, 0, 0, 0)
    if not (0.0 <= opacity <= 1.0):
        raise ValueError(f"opacity must be in [0, 1] (got {opacity})")
    if not (0.0 <= hardness <= 1.0):
        raise ValueError(f"hardness must be in [0, 1] (got {hardness})")
    if len(dst) != width * height * 4:
        raise ValueError(f"dst size mismatch: {len(dst)} != {width}*{height}*4")
    if len(src_buf) != src_w * src_h * 4:
        raise ValueError(f"src_buf size mismatch: {len(src_buf)} != {src_w}*{src_h}*4")
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(width, cx + radius + 1)
    y1 = min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return (0, 0, 0, 0)
    r2 = radius * radius
    soft = 1.0 - hardness
    soft_sq = max(soft * soft, 1e-6)
    for py in range(y0, y1):
        for px in range(x0, x1):
            ddx = px - cx
            ddy = py - cy
            d2 = ddx * ddx + ddy * ddy
            if d2 > r2:
                continue
            sx = px - src_offset_x
            sy = py - src_offset_y
            if sx < 0 or sx >= src_w or sy < 0 or sy >= src_h:
                continue
            # Falloff alpha for this pixel.
            if hardness >= 0.999:
                fa = 1.0
            else:
                fa = math.exp(-((d2 / r2) / soft_sq))
            sa = fa * opacity * (src_buf[(sy * src_w + sx) * 4 + 3] / 255.0)
            if sa <= 0.0:
                continue
            si = (sy * src_w + sx) * 4
            di = (py * width + px) * 4
            sr = src_buf[si + 0]
            sg = src_buf[si + 1]
            sb = src_buf[si + 2]
            dr = dst[di + 0]
            dg = dst[di + 1]
            db = dst[di + 2]
            da = dst[di + 3]
            inv = 1.0 - sa
            out_a = sa + (da / 255.0) * inv
            if out_a <= 0.0:
                dst[di + 0] = 0
                dst[di + 1] = 0
                dst[di + 2] = 0
                dst[di + 3] = 0
                continue
            out_r = (sr * sa + dr * (da / 255.0) * inv) / out_a
            out_g = (sg * sa + dg * (da / 255.0) * inv) / out_a
            out_b = (sb * sa + db * (da / 255.0) * inv) / out_a
            dst[di + 0] = int(round(max(0.0, min(255.0, out_r))))
            dst[di + 1] = int(round(max(0.0, min(255.0, out_g))))
            dst[di + 2] = int(round(max(0.0, min(255.0, out_b))))
            dst[di + 3] = int(round(max(0.0, min(255.0, out_a * 255.0))))
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Gradient (linear / radial / angular)
# ---------------------------------------------------------------------------
def _interp_stops(t: float, stops: list[tuple[float, tuple[int, int, int, int]]]) -> tuple[int, int, int, int]:
    """Sample a sorted list of ``(pos, rgba)`` stops at parameter ``t``.

    ``t`` is clamped to ``[stops[0].pos, stops[-1].pos]``. Linear
    interpolation between adjacent stops.
    """
    if not stops:
        return (0, 0, 0, 0)
    if t <= stops[0][0]:
        return stops[0][1]
    if t >= stops[-1][0]:
        return stops[-1][1]
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        if p0 <= t <= p1:
            if p1 == p0:
                return c0
            f = (t - p0) / (p1 - p0)
            return (
                int(round(c0[0] + (c1[0] - c0[0]) * f)),
                int(round(c0[1] + (c1[1] - c0[1]) * f)),
                int(round(c0[2] + (c1[2] - c0[2]) * f)),
                int(round(c0[3] + (c1[3] - c0[3]) * f)),
            )
    return stops[-1][1]


def gradient_fill(
    dst: bytearray,
    width: int,
    height: int,
    *,
    x0: float, y0: float,
    x1: float, y1: float,
    stops: list[tuple[float, tuple[int, int, int, int]]],
    kind: str = "linear",
    opacity: float = 1.0,
) -> None:
    """Render a gradient between ``(x0, y0)`` and ``(x1, y1)`` over ``dst``.

    ``kind`` is one of:
      * ``linear``  — solid plane perpendicular to the start->end vector.
      * ``radial``  — concentric rings centered at start, reaching end.
      * ``angular`` — sweep around start, ``angle = 0`` aligned to end.

    ``stops`` is a sorted list of ``(t, (r, g, b, a))`` where ``t`` in
    ``[0, 1]`` is the position along the gradient axis. The stops are
    sampled per-pixel; inside the destination buffer they are alpha-overed
    onto whatever is already there at the supplied ``opacity``. Out-of-axis
    pixels (``t < 0`` or ``t > 1``) clamp to the nearest stop, matching
    Photoshop's "no extension" / clamp-to-edge default.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    if not stops:
        raise ValueError("stops list must be non-empty")
    if not (0.0 <= opacity <= 1.0):
        raise ValueError(f"opacity must be in [0, 1] (got {opacity})")
    if len(dst) != width * height * 4:
        raise ValueError(f"dst size mismatch: {len(dst)} != {width}*{height}*4")
    sorted_stops = sorted(stops, key=lambda s: s[0])
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-3 and kind != "angular":
        # Zero-length linear/radial — fill with the first stop.
        first = sorted_stops[0][1]
        for i in range(width * height):
            bi = i * 4
            sa = (first[3] / 255.0) * opacity
            if sa <= 0.0:
                continue
            inv = 1.0 - sa
            da = dst[bi + 3] / 255.0
            out_a = sa + da * inv
            if out_a <= 0.0:
                dst[bi:bi + 4] = bytes(4)
                continue
            for c in range(3):
                dst[bi + c] = int(round((first[c] * sa + dst[bi + c] * da * inv) / out_a))
            dst[bi + 3] = int(round(out_a * 255.0))
        return
    inv_len = (1.0 / length) if length > 0 else 0.0
    nx = dx * inv_len
    ny = dy * inv_len
    for py in range(height):
        for px in range(width):
            if kind == "linear":
                # Project (px - x0, py - y0) onto the unit direction.
                t = ((px - x0) * nx + (py - y0) * ny) * inv_len
            elif kind == "radial":
                rx = px - x0
                ry = py - y0
                t = math.sqrt(rx * rx + ry * ry) * inv_len
            elif kind == "angular":
                # Angle relative to the start->end vector. 0 along the
                # axis, 1 after a full turn (we wrap -pi..pi to [0, 1)).
                rx = px - x0
                ry = py - y0
                a = math.atan2(ry, rx) - math.atan2(dy, dx)
                a = a / (2.0 * math.pi)
                t = a - math.floor(a)
            else:
                raise ValueError(f"unsupported gradient kind {kind!r}")
            r, g, b, a = _interp_stops(t, sorted_stops)
            sa = (a / 255.0) * opacity
            if sa <= 0.0:
                continue
            bi = (py * width + px) * 4
            inv = 1.0 - sa
            dr = dst[bi + 0]
            dg = dst[bi + 1]
            db = dst[bi + 2]
            da = dst[bi + 3] / 255.0
            out_a = sa + da * inv
            if out_a <= 0.0:
                dst[bi:bi + 4] = bytes(4)
                continue
            dst[bi + 0] = int(round((r * sa + dr * da * inv) / out_a))
            dst[bi + 1] = int(round((g * sa + dg * da * inv) / out_a))
            dst[bi + 2] = int(round((b * sa + db * da * inv) / out_a))
            dst[bi + 3] = int(round(max(0.0, min(255.0, out_a * 255.0))))


# ---------------------------------------------------------------------------
# Alpha mask helpers
# ---------------------------------------------------------------------------
def make_blank_mask(width: int, height: int, *, white: bool = True) -> bytearray:
    """Allocate a fresh RGBA mask buffer.

    A mask is stored as RGBA so it round-trips through the same PNG
    pipeline as a regular layer (the JS mask canvas paints into an
    RGBA <canvas>, the server saves it via :func:`rgba_to_png_bytes`).
    Only the R channel is consulted by :func:`composite_layers`; the
    G / B / A channels are filled to match for visual debugging.

    ``white=True`` (default) means "fully visible" — black means
    "fully hidden" (the canonical Photoshop convention).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive (got {width}, {height})")
    val = 255 if white else 0
    n = width * height
    buf = bytearray(n * 4)
    for i in range(n):
        bi = i * 4
        buf[bi + 0] = val
        buf[bi + 1] = val
        buf[bi + 2] = val
        buf[bi + 3] = 255
    return buf


def apply_mask_to_layer(
    rgba: bytearray,
    mask: bytes,
    width: int,
    height: int,
) -> None:
    """Bake a mask into a layer's alpha channel (in-place).

    After this call the layer's alpha is multiplied by the mask's R
    channel, and the mask itself can be discarded (set
    ``layer.has_mask = False`` and delete ``<idx>_mask.png``).

    Frontend "Apply mask" right-click action calls this server-side
    via the layer-save endpoint.
    """
    n = width * height
    if len(rgba) != n * 4:
        raise ValueError(f"rgba size mismatch: {len(rgba)} != {n}*4")
    if len(mask) != n * 4:
        raise ValueError(f"mask size mismatch: {len(mask)} != {n}*4")
    for i in range(n):
        bi = i * 4
        m = mask[bi] / 255.0  # R channel only
        new_a = rgba[bi + 3] * m
        rgba[bi + 3] = int(round(max(0.0, min(255.0, new_a))))
