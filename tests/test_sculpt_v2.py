"""Tests for the v5 polish layer of the geometry sculpt module.

Coverage:
  - Three new brushes: smudge, twist, layer (server-side primitives).
  - Mirror axis reflection helper (X/Y/Z/Off).
  - Retopo region operator: positional Laplacian relax.
  - Sculpt-panel JS source-level guards: state.mirrorAxis dropdown
    plumbing, bus closure capture, brush list inclusion.
  - Bus integration smoke: simulate a sculpt push -> bus.push() ->
    panel-undo, verify the closure successfully drives the mesh
    revert without depending on which tab is mounted.

The JS sanity checks read static/sculpt_panel.js as text and grep
for sentinel patterns (since we don't run a JS interpreter in the
test harness). This catches "did the agent forget to wire X" while
the heavier pure-numeric tests cover the math.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from formats import sculpt as sc


REPO_ROOT = Path(__file__).resolve().parent.parent
SCULPT_JS = REPO_ROOT / "static" / "sculpt_panel.js"


# ---------------------------------------------------------------------------
# v5 brush primitives — server-side numeric tests.
# ---------------------------------------------------------------------------
def _grid_mesh(n=5, spacing=0.2):
    """Helper: build an N×N planar grid centred at origin in the XY
    plane, triangulated into 2*(N-1)^2 tris."""
    verts = []
    for j in range(n):
        for i in range(n):
            x = (i - (n - 1) / 2) * spacing
            y = (j - (n - 1) / 2) * spacing
            verts.extend([x, y, 0.0])
    normals = []
    for _ in range(n * n):
        normals.extend([0.0, 0.0, 1.0])
    indices = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + (i + 1)
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1)
            indices.extend([a, b, c, b, d, c])
    return (
        np.array(verts, dtype=np.float64),
        np.array(normals, dtype=np.float64),
        np.array(indices, dtype=np.int64),
    )


def test_brushes_v5_in_valid_set():
    """The four new brushes must be discoverable via the public
    VALID_BRUSHES tuple so server-side validation accepts them."""
    for b in (sc.BRUSH_SMUDGE, sc.BRUSH_TWIST, sc.BRUSH_LAYER, sc.BRUSH_RETOPO):
        assert b in sc.VALID_BRUSHES


def test_brush_smudge_translates_in_radius():
    pos, nrm, ind = _grid_mesh(5, 0.2)
    centre = (0.0, 0.0, 0.0)
    affected = sc.GridIndex.build(pos.reshape(-1, 3), 0.2).query_radius(centre, 0.5)
    drag = (0.1, 0.0, 0.0)
    out, moved = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_SMUDGE,
        brush_centre=centre,
        brush_direction=(0, 0, 1),
        radius=0.5,
        strength=1.0,
        falloff_curve=sc.FALLOFF_LINEAR,
        drag_vector=drag,
    )
    out = out.reshape(-1, 3)
    pre = pos.reshape(-1, 3)
    # The centre vertex should have moved by the full drag*falloff(0)*strength=1.
    centre_idx = (5 // 2) * 5 + (5 // 2)
    assert out[centre_idx, 0] > pre[centre_idx, 0]
    # Edge verts (still inside radius=0.5) should have moved less than centre.
    delta_centre = out[centre_idx, 0] - pre[centre_idx, 0]
    edge_idx = 0  # corner at (-0.4, -0.4, 0)
    if edge_idx in moved:
        delta_edge = out[edge_idx, 0] - pre[edge_idx, 0]
        assert delta_edge < delta_centre


def test_brush_smudge_no_drag_no_op():
    """drag_vector=None should produce no movement."""
    pos, nrm, ind = _grid_mesh(3, 0.2)
    out, moved = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=list(range(9)),
        brush=sc.BRUSH_SMUDGE,
        brush_centre=(0, 0, 0),
        brush_direction=(0, 0, 1),
        radius=0.5,
        strength=1.0,
        drag_vector=None,
    )
    assert moved == []
    np.testing.assert_array_equal(out, pos)


def test_brush_twist_rotates_around_axis():
    pos, nrm, ind = _grid_mesh(5, 0.2)
    centre = (0.0, 0.0, 0.0)
    affected = sc.GridIndex.build(pos.reshape(-1, 3), 0.2).query_radius(centre, 1.0)
    pre = pos.reshape(-1, 3).copy()
    out, moved = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_TWIST,
        brush_centre=centre,
        brush_direction=(0, 0, 1),  # rotate around Z
        radius=1.0,
        strength=1.0,
        falloff_curve=sc.FALLOFF_LINEAR,
        drag_distance=1.0,
        twist_rate=1.0,  # ~57 degrees of twist at the centre with falloff=1
    )
    out = out.reshape(-1, 3)
    # Z component should be unchanged for any vertex (rotation around Z).
    np.testing.assert_allclose(out[:, 2], pre[:, 2], atol=1e-6)
    # At least one off-centre vertex should have rotated (changed XY).
    centre_idx = 12  # (2,2) of 5x5
    moved_xy = False
    for vi in moved:
        if vi == centre_idx:
            continue
        if abs(out[vi, 0] - pre[vi, 0]) > 1e-6 or abs(out[vi, 1] - pre[vi, 1]) > 1e-6:
            moved_xy = True
            break
    assert moved_xy, "twist must rotate at least one off-centre vertex"


def test_brush_twist_negative_rate_flips_direction():
    """Negative twist_rate rotates counter-clockwise (opposite of
    positive). Verified by checking that at least one vertex's XY
    delta has the opposite sign of the same vertex's delta with
    positive rate."""
    pos, nrm, ind = _grid_mesh(3, 0.5)
    affected = list(range(9))
    pre = pos.reshape(-1, 3).copy()
    out_pos, _ = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_TWIST, brush_centre=(0, 0, 0),
        brush_direction=(0, 0, 1),
        radius=2.0, strength=1.0,
        falloff_curve=sc.FALLOFF_LINEAR,
        drag_distance=0.5, twist_rate=1.0,
    )
    out_neg, _ = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_TWIST, brush_centre=(0, 0, 0),
        brush_direction=(0, 0, 1),
        radius=2.0, strength=1.0,
        falloff_curve=sc.FALLOFF_LINEAR,
        drag_distance=0.5, twist_rate=-1.0,
    )
    pos_xy = out_pos.reshape(-1, 3)[:, 0:2] - pre[:, 0:2]
    neg_xy = out_neg.reshape(-1, 3)[:, 0:2] - pre[:, 0:2]
    # Sum of sign-products should be negative-leaning (opposite directions).
    sign_sum = float(np.sign(pos_xy).flatten() @ np.sign(neg_xy).flatten())
    assert sign_sum <= 0


def test_brush_twist_zero_drag_no_op():
    pos, nrm, ind = _grid_mesh(3, 0.5)
    out, moved = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=list(range(9)),
        brush=sc.BRUSH_TWIST,
        brush_centre=(0, 0, 0),
        brush_direction=(0, 0, 1),
        radius=2.0,
        strength=1.0,
        drag_distance=0.0,
        twist_rate=1.0,
    )
    assert moved == []
    np.testing.assert_array_equal(out, pos)


def test_brush_layer_offsets_along_normal():
    pos, nrm, ind = _grid_mesh(5, 0.2)
    affected = list(range(pos.shape[0] // 3))
    centre = (0.0, 0.0, 0.0)
    pre = pos.reshape(-1, 3).copy()
    out, moved = sc.apply_brush(
        positions=pos, normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_LAYER,
        brush_centre=centre,
        brush_direction=(0, 0, 1),
        radius=0.6,
        strength=0.8,
        falloff_curve=sc.FALLOFF_SMOOTH,
    )
    out = out.reshape(-1, 3)
    # Every moved vertex should have +Z offset (normals point +Z).
    for vi in moved:
        assert out[vi, 2] > pre[vi, 2]


def test_brush_layer_compounds_on_reapply():
    """Each layer pass should add MORE thickness, not converge to a
    cap (Blender's "non-anchored" layer brush behaviour)."""
    pos, nrm, ind = _grid_mesh(3, 0.2)
    affected = list(range(9))
    centre = (0.0, 0.0, 0.0)
    args = dict(
        normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_LAYER,
        brush_centre=centre,
        brush_direction=(0, 0, 1),
        radius=0.6, strength=0.5,
    )
    pos1, _ = sc.apply_brush(positions=pos, **args)
    pos2, _ = sc.apply_brush(positions=pos1, **args)
    pre = pos.reshape(-1, 3)
    p1 = pos1.reshape(-1, 3)
    p2 = pos2.reshape(-1, 3)
    # The centre should have moved further on the second pass.
    centre_idx = 4
    z1 = p1[centre_idx, 2] - pre[centre_idx, 2]
    z2 = p2[centre_idx, 2] - pre[centre_idx, 2]
    assert z2 > z1


def test_brush_retopo_relaxes_irregular_grid():
    """Retopo brush is a Laplacian relax: take a mesh with one vertex
    pulled off the plane and verify it gets pulled back toward the
    neighbour mean."""
    pos, nrm, ind = _grid_mesh(5, 0.2)
    p = pos.reshape(-1, 3).copy()
    spike = 12  # centre
    p[spike, 0] += 0.3  # pull X off
    affected = list(range(p.shape[0]))
    out = sc._retopo_region(p, ind, affected, strength=1.0)
    out = out.reshape(-1, 3)
    # The spike should have moved back toward 0 (its neighbours' mean).
    assert abs(out[spike, 0]) < abs(p[spike, 0])


def test_brush_retopo_strength_zero_noop():
    pos, _, ind = _grid_mesh(3, 0.2)
    out = sc._retopo_region(pos.reshape(-1, 3), ind, list(range(9)), strength=0.0)
    np.testing.assert_array_equal(out.reshape(-1), pos.reshape(-1))


def test_brush_retopo_via_apply_brush():
    """End-to-end: route through apply_brush(BRUSH_RETOPO) so the
    public surface is exercised the same way the JS path will be."""
    pos, nrm, ind = _grid_mesh(5, 0.2)
    p = pos.reshape(-1, 3).copy()
    p[12, 1] += 0.4  # pull Y off
    affected = list(range(p.shape[0]))
    out, moved = sc.apply_brush(
        positions=p.reshape(-1), normals=nrm, indices=ind,
        affected_indices=affected,
        brush=sc.BRUSH_RETOPO,
        brush_centre=(0, 0, 0),
        brush_direction=(0, 0, 1),
        radius=1.0,
        strength=0.8,
    )
    out = out.reshape(-1, 3)
    # The off-centre vertex should be closer to 0 in Y now.
    assert abs(out[12, 1]) < abs(p[12, 1])


# ---------------------------------------------------------------------------
# Mirror axis helper.
# ---------------------------------------------------------------------------
def test_reflect_axis_x():
    assert sc.reflect_axis([1.0, 2.0, 3.0], sc.MIRROR_X) == [-1.0, 2.0, 3.0]


def test_reflect_axis_y():
    assert sc.reflect_axis([1.0, 2.0, 3.0], sc.MIRROR_Y) == [1.0, -2.0, 3.0]


def test_reflect_axis_z():
    assert sc.reflect_axis([1.0, 2.0, 3.0], sc.MIRROR_Z) == [1.0, 2.0, -3.0]


def test_reflect_axis_off_passes_through():
    assert sc.reflect_axis([1.0, 2.0, 3.0], sc.MIRROR_OFF) == [1.0, 2.0, 3.0]


def test_valid_mirrors_complete():
    for m in (sc.MIRROR_OFF, sc.MIRROR_X, sc.MIRROR_Y, sc.MIRROR_Z):
        assert m in sc.VALID_MIRRORS


# ---------------------------------------------------------------------------
# JS source-level guards (catch missing wiring without spinning up a
# headless browser). We rely on the source being readable rather than
# parsing it — this keeps the test stable across formatting changes
# while still pinning the key sentinels.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sculpt_js_text() -> str:
    if not SCULPT_JS.exists():
        pytest.skip(f"sculpt_panel.js not found at {SCULPT_JS}")
    return SCULPT_JS.read_text(encoding="utf-8")


def test_js_has_v5_brushes_in_list(sculpt_js_text):
    """The BRUSHES array must include all four v5 brushes."""
    # Capture the array literal that lists the brush strings. We allow
    # newlines + whitespace inside it.
    m = re.search(r"const\s+BRUSHES\s*=\s*\[([^\]]*)\]", sculpt_js_text)
    assert m, "BRUSHES const not found"
    brushes_text = m.group(1)
    for b in ("smudge", "twist", "layer", "retopo"):
        assert f'"{b}"' in brushes_text, f"BRUSHES missing {b!r}"


def test_js_has_mirror_axis_dropdown(sculpt_js_text):
    """state.mirrorAxis must exist alongside the legacy state.mirrorX
    so the dropdown works AND old test harnesses don't crash."""
    assert "mirrorAxis" in sculpt_js_text
    assert "MIRROR_AXES" in sculpt_js_text
    # Dropdown UI fragment.
    assert 'data-knob="mirrorAxis"' in sculpt_js_text


def test_js_has_brush_apply_for_each_v5_brush(sculpt_js_text):
    """Each new brush must have its own `case "<name>":` arm in the
    primary brush switch. Otherwise the brush button would no-op.
    """
    for b in ("smudge", "twist", "layer", "retopo"):
        pat = rf'case\s+"{b}"\s*:'
        assert re.search(pat, sculpt_js_text), f"missing case for brush {b!r}"


def test_js_has_undo_bus_capture(sculpt_js_text):
    """Closure capture must reference the `entry` local — guarding
    against a regression where the bus push closes over a mutable
    outer variable."""
    # Look for the captured-entry idiom near the bus push.
    assert "const captured = entry" in sculpt_js_text
    # The bus push itself should set panelId: "sculpt".
    assert 'panelId: "sculpt"' in sculpt_js_text


def test_js_drag_tracking_for_smudge_twist(sculpt_js_text):
    """Smudge + twist need lastLocal / dragDistance tracking on the
    stroke object."""
    assert "lastLocal" in sculpt_js_text
    assert "dragDistance" in sculpt_js_text
    # Twist axis usage — Rodrigues rotation reference.
    assert "twistRate" in sculpt_js_text


# ---------------------------------------------------------------------------
# Bus closure smoke: simulate the JS closure semantics in Python so we
# can verify "stroke records are looked up at undo time, not at push
# time" — i.e. tab switches don't invalidate the closure.
# ---------------------------------------------------------------------------
class _FakeRecord:
    """Mimic the sculpt_panel.js record { meshRef, originalPos,
    accumDisp, modifiedSet }. We don't need full THREE.Mesh fidelity;
    just enough that applyEntry() can run."""
    def __init__(self, vertex_count):
        self.vc = vertex_count
        self.position = np.zeros(vertex_count * 3, dtype=np.float32)
        self.original = self.position.copy()
        self.accumDisp = np.zeros_like(self.position)
        self.modifiedSet: set[int] = set()


def _apply_entry(rec: _FakeRecord, entry: dict, from_pos: str):
    """Pure-Python mirror of applyEntry() in sculpt_panel.js — same
    behaviour, no THREE.Mesh dependency."""
    indices = entry["indices"]
    data = entry[from_pos]
    for vi in indices:
        sx = vi * 3
        rec.position[sx + 0] = data[sx + 0]
        rec.position[sx + 1] = data[sx + 1]
        rec.position[sx + 2] = data[sx + 2]
        rec.accumDisp[sx + 0] = rec.position[sx + 0] - rec.original[sx + 0]
        rec.accumDisp[sx + 1] = rec.position[sx + 1] - rec.original[sx + 1]
        rec.accumDisp[sx + 2] = rec.position[sx + 2] - rec.original[sx + 2]
        is_mod = (
            rec.accumDisp[sx + 0] != 0 or
            rec.accumDisp[sx + 1] != 0 or
            rec.accumDisp[sx + 2] != 0
        )
        if is_mod:
            rec.modifiedSet.add(vi)
        else:
            rec.modifiedSet.discard(vi)


def test_undo_bus_closure_captures_entry_across_tab_switch():
    """Simulate: the user pushes a stroke, we register a bus undo
    closure, then we "switch tabs" (which in the real app is just a
    DOM swap; the panel's module-state stays alive).

    The closure must:
      1. Capture the entry by value (so we don't lose data when the
         panel's local stack evicts under UNDO_LIMIT).
      2. Look up the record at undo TIME (records persist across tab
         switches because they live in the module's `sculptRecords`
         Map, not in any DOM-scoped state).
    """
    sculpt_records: dict[int, _FakeRecord] = {0: _FakeRecord(vertex_count=10)}
    # Push 3 strokes (each touching one vertex) and register bus
    # closures.
    bus: list[dict] = []

    for i in range(3):
        rec = sculpt_records[0]
        # Apply the stroke to the live "mesh".
        before = rec.position.copy()
        rec.position[i * 3] = float(i + 1) * 0.1
        rec.accumDisp[i * 3] = rec.position[i * 3] - rec.original[i * 3]
        rec.modifiedSet.add(i)
        after = rec.position.copy()
        sub_entry = {
            "submeshIdx": 0,
            "indices": [i],
            "before": before.copy(),
            "after": after.copy(),
        }
        entry = {"subs": [sub_entry]}

        # Same closure idiom as sculpt_panel.js's pushUndoEntry.
        captured = entry

        def make_closures(captured_entry):
            def apply_dir(from_pos):
                for sub in captured_entry["subs"]:
                    rec_ref = sculpt_records.get(sub["submeshIdx"])
                    if not rec_ref:
                        continue
                    _apply_entry(rec_ref, sub, from_pos)
            return apply_dir

        applier = make_closures(captured)
        bus.append({"undo": lambda a=applier: a("before"),
                    "redo": lambda a=applier: a("after")})

    # All 3 verts should be displaced.
    assert all(sculpt_records[0].position[i * 3] > 0 for i in range(3))

    # "Switch to mob_dsl tab and back": in the real app this is just
    # a DOM swap; we simulate it by NO-OP'ing here. The records map
    # is unchanged.
    # ... (no state change) ...

    # Now Ctrl+Z three times (undo from the tip of the deque).
    for _ in range(3):
        bus.pop()["undo"]()

    # Mesh should be back to original.
    np.testing.assert_array_equal(sculpt_records[0].position, sculpt_records[0].original)
    assert not sculpt_records[0].modifiedSet


def test_undo_bus_closure_survives_record_eviction_then_restore():
    """If the panel's local UNDO_LIMIT evicts an entry, the bus
    closure should still revert the mesh — the closure holds the
    `before`/`after` payload independently of the panel's stack.
    """
    sculpt_records = {0: _FakeRecord(vertex_count=5)}
    rec = sculpt_records[0]
    before = rec.position.copy()
    rec.position[0] = 0.5
    after = rec.position.copy()
    sub = {"submeshIdx": 0, "indices": [0], "before": before, "after": after}
    captured_entry = {"subs": [sub]}

    def apply_dir(from_pos):
        for s in captured_entry["subs"]:
            r = sculpt_records.get(s["submeshIdx"])
            if r:
                _apply_entry(r, s, from_pos)

    # Simulate the panel evicting its local stack — the closure
    # doesn't reference it.
    panel_stack = [captured_entry]
    panel_stack.clear()

    # Bus closure should still be valid.
    apply_dir("before")
    np.testing.assert_array_equal(rec.position, rec.original)
    apply_dir("after")
    assert rec.position[0] == 0.5
