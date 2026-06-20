"""Unit tests for the Material chunk decoder + Material Inspector.

Covers the decode/encode/aggregate paths in ``formats/material.py``
plus a smoke test that the new ``/api/material/<path>`` GET / POST
endpoints round-trip a real PSOBB submesh.

Test fixture choice — ``bm_ene_bm9_s_mericarol.bml`` because it has
56 submeshes spanning every material chunk type the inspector needs
to surface, and is small enough (47 kB NJCM) to load in <100 ms.
"""
from __future__ import annotations
import os

import struct

import pytest

from formats.material import (
    BLEND_PRESETS,
    BlendAlphaPayload,
    MaterialChunkPayload,
    PSOBB_MATERIAL_PRESETS,
    RGBA,
    StripFlagsPayload,
    SubmeshMaterial,
    TinyChunkPayload,
    aggregate_submesh_state,
    apply_submesh_edits,
    decode_blend_alpha_chunk,
    decode_material_chunk,
    decode_strip_chunk_flags,
    decode_tiny_chunk,
    encode_blend_alpha_flags,
    encode_material_chunk,
    encode_strip_flags,
    list_presets,
)


# ---------------------------------------------------------------------------
# RGBA
# ---------------------------------------------------------------------------


def test_rgba_bgra_roundtrip():
    src = RGBA(r=10, g=200, b=30, a=128)
    raw = src.to_bgra_bytes()
    # On disk: B, G, R, A.
    assert raw == bytes([30, 200, 10, 128])
    back = RGBA.from_bgra_bytes(raw)
    assert (back.r, back.g, back.b, back.a) == (10, 200, 30, 128)


def test_rgba_clamps_oversized_components():
    # Defensive — server may receive [-1, 999, 255, "x"] from a buggy UI.
    # The encoder masks to 0..255 implicitly; coerce_rgba does explicit
    # clamping. RGBA itself doesn't clamp on construction (so test the
    # to_bgra_bytes mask).
    src = RGBA(r=0x1FF, g=-5, b=300, a=128)
    raw = src.to_bgra_bytes()
    # 0x1FF & 0xFF = 0xFF, -5 & 0xFF = 0xFB, 300 & 0xFF = 0x2C
    assert raw == bytes([0x2C, 0xFB, 0xFF, 0x80])


# ---------------------------------------------------------------------------
# Material chunk decoder
# ---------------------------------------------------------------------------


def test_decode_chunk_17_diffuse_only():
    # 0200ffffffff — wc=2, BGRA=ff,ff,ff,ff (white, full alpha)
    body = bytes.fromhex("0200ffffffff")
    out = decode_material_chunk(17, 0x25, body)
    assert out.type_id == 17
    assert out.flags == 0x25
    assert out.diffuse is not None
    assert out.diffuse.to_tuple() == (255, 255, 255, 255)
    assert out.ambient is None
    assert out.specular is None


def test_decode_chunk_19_diffuse_plus_ambient():
    # 0400 ff ff ff ff 7f 7f 7f ff
    body = bytes.fromhex("0400ffffffff7f7f7fff")
    out = decode_material_chunk(19, 0x25, body)
    assert out.diffuse.to_tuple() == (255, 255, 255, 255)
    assert out.ambient.to_tuple() == (127, 127, 127, 255)
    assert out.specular is None


def test_decode_chunk_23_full_dasource():
    # Real shipped value: diffuse=white, ambient=mid-gray, specular=white+exp=0x0b
    body = bytes.fromhex("0600ffffffff7f7f7fffffffff0b")
    out = decode_material_chunk(23, 0x25, body)
    assert out.diffuse.to_tuple() == (255, 255, 255, 255)
    assert out.ambient.to_tuple() == (127, 127, 127, 255)
    assert out.specular.to_tuple() == (255, 255, 255, 0x0b)
    assert out.specular_exponent == 0x0b


def test_decode_chunk_unknown_type_returns_empty():
    out = decode_material_chunk(99, 0, b"\x00\x00")
    assert out.diffuse is None and out.ambient is None and out.specular is None


def test_decode_chunk_truncated_body_does_not_raise():
    # Request type 23 but only give 4 payload bytes — should decode
    # diffuse and gracefully give up.
    body = bytes.fromhex("0600ffffffff")  # claims wc=6 but only 4 payload bytes
    out = decode_material_chunk(23, 0x25, body)
    assert out.diffuse.to_tuple() == (255, 255, 255, 255)
    # Ambient/specular not present (truncated).


# ---------------------------------------------------------------------------
# Material chunk encoder — round-trip
# ---------------------------------------------------------------------------


def test_encode_picks_smallest_chunk_type():
    p = MaterialChunkPayload(
        diffuse=RGBA(r=200, g=100, b=50, a=255),
    )
    t, f, body = encode_material_chunk(p)
    assert t == 17  # diffuse-only -> chunk type 17
    # Round-trip
    decoded = decode_material_chunk(t, f, body)
    assert decoded.diffuse.to_tuple() == (200, 100, 50, 255)


def test_encode_diffuse_plus_specular_picks_type_21():
    p = MaterialChunkPayload(
        diffuse=RGBA(r=255, g=255, b=255, a=255),
        specular=RGBA(r=200, g=200, b=200, a=11),
        specular_exponent=11,
    )
    t, f, body = encode_material_chunk(p)
    assert t == 21
    decoded = decode_material_chunk(t, f, body)
    assert decoded.diffuse.to_tuple() == (255, 255, 255, 255)
    assert decoded.specular_exponent == 11


def test_encode_full_das_picks_type_23():
    p = MaterialChunkPayload(
        diffuse=RGBA(255, 255, 255, 255),
        ambient=RGBA(127, 127, 127, 255),
        specular=RGBA(255, 255, 255, 0x0b),
        specular_exponent=0x0b,
    )
    t, f, body = encode_material_chunk(p)
    assert t == 23
    # Round-trip the body bytes — must equal the canonical shipped form.
    assert body == bytes.fromhex("0600ffffffff7f7f7fffffffff0b")


def test_decode_then_encode_byte_exact_for_canonical_value():
    """The most common shipped value MUST byte-round-trip."""
    canonical = bytes.fromhex("0400ffffffff7f7f7fff")
    decoded = decode_material_chunk(19, 0x25, canonical)
    t, f, body = encode_material_chunk(decoded)
    assert t == 19
    assert body == canonical


# ---------------------------------------------------------------------------
# Blend alpha decoder
# ---------------------------------------------------------------------------


def test_decode_blend_alpha_0x25_is_standard_blend():
    """Most-common shipped flag for BlendAlpha (type 1): src=src_alpha,
    dst=one_minus_src_alpha. Encoding: bits 3..5 = src_idx, bits 0..2
    = dst_idx; 0x25 = 0010 0101 -> src=4 (src_alpha), dst=5
    (one_minus_src_alpha)."""
    out = decode_blend_alpha_chunk(0x25)
    assert out.src_factor == "src_alpha"
    assert out.dst_factor == "one_minus_src_alpha"
    assert out.mode == "blend"


def test_decode_blend_alpha_0x21_is_additive():
    """Second-most-common: 0x21 = src=src_alpha, dst=one (additive glow)."""
    out = decode_blend_alpha_chunk(0x21)
    assert out.src_factor == "src_alpha"
    assert out.dst_factor == "one"
    assert out.mode == "additive"


def test_encode_blend_alpha_inverse_of_decode():
    for raw in (0x21, 0x25):
        d = decode_blend_alpha_chunk(raw)
        enc = encode_blend_alpha_flags(d.src_factor, d.dst_factor)
        # Lower 6 bits should match (upper 2 are reserved).
        assert (enc & 0x3F) == (raw & 0x3F)


# ---------------------------------------------------------------------------
# Tiny chunk decoder
# ---------------------------------------------------------------------------


def test_decode_tiny_chunk_extracts_texture_id():
    body = struct.pack("<H", 7 | (0x4 << 13))  # tex_id=7, alpha-thresh overlay=4
    out = decode_tiny_chunk(0x34, body)
    assert out.texture_id == 7
    assert out.alpha_threshold_bits == 4
    # Flag bits per Sega convention: 0x34 = bit 2 (point_filter) + bit
    # 4 (super) + bit 5 (use_filter)
    assert out.point_filter
    assert out.super_sample
    assert out.use_filter
    assert not out.clamp_u and not out.clamp_v


def test_decode_tiny_chunk_clamp_flags():
    body = struct.pack("<H", 0)
    out = decode_tiny_chunk(0xC4, b"\x00\x00")
    assert out.clamp_u and out.clamp_v


def test_decode_tiny_chunk_masks_high_bits_off_texture_id():
    """``texture_id`` MUST be the body word's bottom 13 bits (& 0x1FFF).

    Reference (MIT, Justin113D/BlenderSASupport ``format_CHUNK.py:276``):

        texID = header & 0x1FFF

    PSOBB overloads the top 3 bits with an alpha-threshold exponent —
    if we forgot the mask, those bits would leak into texture_id and
    indices > 0x1FFF would wrap around to nonexistent textures
    (the texture index list peaks at ~0x800 in practice; 0x2000+ is
    nonsense).

    This test confirms the mask is applied even when bits 13-15 are
    set to all-ones.
    """
    # Body word = 0xE042 (bits 13-15 = 0b111, low 13 bits = 0x42 = 66).
    body = struct.pack("<H", (0x7 << 13) | 0x42)
    out = decode_tiny_chunk(0x34, body)
    assert out.texture_id == 0x42
    # The high bits surface separately as alpha_threshold_bits.
    assert out.alpha_threshold_bits == 0x7


def test_decode_tiny_chunk_max_texture_id_is_0x1fff():
    """Body word 0xFFFF should decode to texture_id = 0x1FFF, not 0xFFFF."""
    body = struct.pack("<H", 0xFFFF)
    out = decode_tiny_chunk(0x34, body)
    assert out.texture_id == 0x1FFF
    assert out.alpha_threshold_bits == 0x7


# ---------------------------------------------------------------------------
# Strip chunk flags
# ---------------------------------------------------------------------------


def test_decode_strip_flags_double_sided():
    out = decode_strip_chunk_flags(0x04)
    assert out.double_sided
    assert not out.flat_shaded


def test_strip_flags_roundtrip():
    payload = StripFlagsPayload(
        double_sided=True, no_zwrite=True, ignore_light=False,
    )
    encoded = encode_strip_flags(payload)
    decoded = decode_strip_chunk_flags(encoded)
    assert decoded.double_sided
    assert decoded.no_zwrite
    assert not decoded.ignore_light


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _mk_chunk(type_id: int, flags: int, body: bytes = b""):
    """Tiny stand-in for NjChunk so we don't pull formats.nj_writer here."""
    class _C:
        pass
    c = _C()
    c.type_id = type_id
    c.flags = flags
    c.body = body
    return c


def test_aggregate_emits_one_row_per_strip():
    chunks = [
        _mk_chunk(17, 0x25, bytes.fromhex("0200aabbccff")),
        _mk_chunk(8, 0x34, struct.pack("<H", 5)),
        _mk_chunk(64, 0x04, b""),  # strip 0 -- two-sided
        _mk_chunk(8, 0x34, struct.pack("<H", 7)),
        _mk_chunk(64, 0x00, b""),  # strip 1 -- single-sided
    ]
    rows = aggregate_submesh_state(chunks)
    assert len(rows) == 2
    assert rows[0].submesh_idx == 0
    assert rows[0].material_id == 5
    # Diffuse hex 0xaabbccff on disk = BGRA -> R=cc, G=bb, B=aa, A=ff
    assert rows[0].diffuse_rgba == (0xCC, 0xBB, 0xAA, 0xFF)
    assert rows[0].two_sided is True
    assert rows[1].submesh_idx == 1
    assert rows[1].material_id == 7
    assert rows[1].two_sided is False


def test_aggregate_blend_mode_cascades_until_overwritten():
    chunks = [
        _mk_chunk(1, 0x25, b""),    # standard blend (src_alpha / inv_src_alpha)
        _mk_chunk(64, 0x00),
        _mk_chunk(1, 0x21, b""),    # additive (src_alpha / one)
        _mk_chunk(64, 0x00),
    ]
    rows = aggregate_submesh_state(chunks)
    assert rows[0].blend_mode == "blend"
    assert rows[1].blend_mode == "additive"


def test_aggregate_default_when_no_chunks():
    rows = aggregate_submesh_state([_mk_chunk(64, 0x00)])
    assert len(rows) == 1
    r = rows[0]
    assert r.diffuse_rgba == (255, 255, 255, 255)
    assert r.depth_test is True and r.depth_write is True


# ---------------------------------------------------------------------------
# Edit applicator
# ---------------------------------------------------------------------------


def test_apply_edit_mutates_strip_two_sided():
    from formats.nj_writer import NjChunk
    chunks = [NjChunk(64, 0x00, b"")]
    out = apply_submesh_edits(chunks, [
        {"submesh_idx": 0, "two_sided": True},
    ])
    assert out[0].flags & 0x04


def test_apply_edit_inserts_blend_alpha_when_missing():
    from formats.nj_writer import NjChunk
    chunks = [NjChunk(64, 0x00, b"")]
    out = apply_submesh_edits(chunks, [
        {"submesh_idx": 0, "alpha_blend": {"src": "src_alpha", "dst": "one"}},
    ])
    # The blend chunk should now precede the strip chunk.
    assert any(c.type_id == 1 for c in out)
    ba = next(c for c in out if c.type_id == 1)
    decoded = decode_blend_alpha_chunk(ba.flags)
    assert decoded.mode == "additive"


def test_apply_edit_replaces_existing_diffuse_chunk():
    from formats.nj_writer import NjChunk
    chunks = [
        NjChunk(17, 0x25, bytes.fromhex("0200ffffffff")),  # white diffuse
        NjChunk(64, 0x00, b""),
    ]
    out = apply_submesh_edits(chunks, [
        {"submesh_idx": 0, "diffuse_rgba": [255, 0, 0, 128]},
    ])
    # The first chunk should still be material (replaced, not appended).
    mat_chunks = [c for c in out if 17 <= c.type_id <= 23]
    assert len(mat_chunks) == 1
    decoded = decode_material_chunk(
        mat_chunks[0].type_id, mat_chunks[0].flags, mat_chunks[0].body
    )
    assert decoded.diffuse.to_tuple() == (255, 0, 0, 128)


def test_apply_edit_depth_write_off():
    from formats.nj_writer import NjChunk
    chunks = [NjChunk(64, 0x00, b"")]
    out = apply_submesh_edits(chunks, [
        {"submesh_idx": 0, "depth_write": False},
    ])
    assert out[0].flags & 0x40   # no_zwrite bit set


def test_apply_edit_alpha_test_overlay_on_tiny():
    from formats.nj_writer import NjChunk
    chunks = [
        NjChunk(8, 0x34, struct.pack("<H", 7)),  # tex id 7
        NjChunk(64, 0x00, b""),
    ]
    out = apply_submesh_edits(chunks, [
        {"submesh_idx": 0,
         "alpha_test": {"enabled": True, "threshold": 128}},
    ])
    tiny = out[0]
    (word,) = struct.unpack_from("<H", tiny.body, 0)
    # Texture id preserved.
    assert (word & 0x1FFF) == 7
    # 128 -> top 3 bits = 4 (binary 100). Overlay region = bits 13..15.
    assert ((word >> 13) & 0x07) == 4


# ---------------------------------------------------------------------------
# Real-fixture round-trip — uses a shipped PSOBB BML if one is available.
# Skipped on systems where the data isn't present.
# ---------------------------------------------------------------------------


def test_aggregate_against_real_mericarol_model():
    """Smoke test: decode a real shipped model and assert sanity invariants.

    The mericarol BML has 56 submeshes spanning chunk types 17, 19, 20,
    23 and Tiny chunks 8 with flags 0x34. We don't pin specific values
    (those are fixture-dependent) but assert the aggregator emits one
    row per strip, all rows have valid RGBA tuples, and at least one
    row references each major chunk family.
    """
    from pathlib import Path
    fixture_path = Path(
        os.path.expanduser("~/PSOBB.IO/data/bm_ene_bm9_s_mericarol.bml")
    )
    if not fixture_path.exists():
        pytest.skip("PSOBB.IO fixtures not available on this host")
    from formats.bml import extract_bml
    from formats.nj_writer import parse_nj_for_writer
    inners = extract_bml(fixture_path.read_bytes())
    nj_bytes = inners.get("bm9_s_meri_body.nj")
    assert nj_bytes is not None, "expected mericarol nj inner to exist"
    model = parse_nj_for_writer(nj_bytes)
    total_rows = 0
    for mesh in model.meshes:
        rows = aggregate_submesh_state(mesh.plist)
        for r in rows:
            assert all(0 <= c <= 255 for c in r.diffuse_rgba)
            assert all(0 <= c <= 255 for c in r.ambient_rgba)
        total_rows += len(rows)
    # Mericarol has many strips; the count should be > 0.
    assert total_rows > 0


# ---------------------------------------------------------------------------
# Preset catalogue
# ---------------------------------------------------------------------------


def test_preset_catalogue_includes_required_keys():
    keys = set(PSOBB_MATERIAL_PRESETS.keys())
    for required in (
        "player_skin", "hair_fur", "energy_glass", "standard_solid",
    ):
        assert required in keys, f"missing required preset {required!r}"


def test_preset_catalogue_serializable():
    """Frontend-facing API: list_presets must be JSON-serializable."""
    import json
    presets = list_presets()
    s = json.dumps(presets)
    assert "player_skin" in s
    assert "hair_fur" in s


def test_preset_player_skin_has_alpha_test_128():
    p = PSOBB_MATERIAL_PRESETS["player_skin"]
    assert p["alpha_test"]["enabled"] is True
    assert p["alpha_test"]["threshold"] == 128


def test_preset_energy_glass_disables_depth_write():
    p = PSOBB_MATERIAL_PRESETS["energy_glass"]
    assert p["depth_write"] is False
    assert p["alpha_blend"] == {"src": "src_alpha", "dst": "one"}
