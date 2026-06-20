"""Tests for the live-reload SSE bridge.

Covers the cache watcher hub, the /api/events SSE stream, /api/events/status,
and /api/events/rescan. The watcher is exercised via its public _LiveReloadHub
methods so we don't have to wait for the 1.0 s default poll tick.

Strategy:
  * Build a TestClient against server.app.
  * Take a snapshot of the watched dirs (cache/njm_export, etc.).
  * Drop a file in cache/njm_export/, call /api/events/rescan, assert the
    /api/events/status counters bumped + rescan returned events_fired > 0.
  * For the SSE stream, use TestClient.stream() with a generator client and
    yank the first event off the socket within a generous timeout.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def srv():
    """Import server module — picks up the env-var poll override below.

    Drop the poll seconds way down so even tests that fall back to the
    timer-driven path complete fast. We can't change the value AFTER
    import since the constant is read once at module-load.
    """
    os.environ.setdefault("PSO_LIVE_RELOAD_POLL_SECONDS", "0.1")
    if "server" in sys.modules:
        # Force a re-import so the new env var sticks. pytest may have
        # already imported via conftest indirectly.
        del sys.modules["server"]
    import server  # noqa: F401  imported for side-effect
    return sys.modules["server"]


@pytest.fixture
def client(srv):
    return TestClient(srv.app)


@pytest.fixture
def staging_dir(srv):
    """Make sure cache/njm_export exists and is empty for each test."""
    d = srv.NJM_EXPORT_DIR
    d.mkdir(parents=True, exist_ok=True)
    # Wipe pre-existing test files (preserve any user files outside this test).
    for p in d.glob("__live_reload_test_*.njm*"):
        try:
            p.unlink()
        except OSError:
            pass
    yield d
    for p in d.glob("__live_reload_test_*.njm*"):
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Hub-level tests (watcher logic without the network).
# ---------------------------------------------------------------------------


def test_hub_starts_with_seed(srv):
    hub = srv._LIVE_RELOAD_HUB
    snap = hub.snapshot_state()
    assert "watched_dirs" in snap
    # All watched dirs from the spec are tracked.
    assert "njm_export" in snap["watched_dirs"]
    assert "painted_textures" in snap["watched_dirs"]
    assert "sculpted_meshes" in snap["watched_dirs"]
    assert "bml_export" in snap["watched_dirs"]
    assert "nj_export" in snap["watched_dirs"]
    assert "itempmt_export" in snap["watched_dirs"]
    assert "battle_param_export" in snap["watched_dirs"]


def test_force_rescan_detects_create(srv, staging_dir):
    """Adding a file to njm_export bumps mtime state on rescan."""
    hub = srv._LIVE_RELOAD_HUB
    # Seed pre-state (rescan once to absorb anything pre-existing).
    hub.force_rescan()
    pre = hub.snapshot_state()["tracked_files"]

    target = staging_dir / "__live_reload_test_create.njm"
    target.write_bytes(b"\x00" * 32)
    n = hub.force_rescan()
    assert n >= 1, f"expected at least 1 event, got {n}"
    post = hub.snapshot_state()["tracked_files"]
    assert post == pre + 1


def test_force_rescan_detects_modify(srv, staging_dir):
    """Modifying an existing file fires a 'modify' event."""
    hub = srv._LIVE_RELOAD_HUB
    target = staging_dir / "__live_reload_test_modify.njm"
    target.write_bytes(b"\x01" * 16)
    hub.force_rescan()  # absorb the create

    captured: list[dict] = []
    saw = threading.Event()

    class _StubLoop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)

    class _StubQ:
        def __init__(self):
            self._items: list[dict] = []
        def full(self):
            return False
        def put_nowait(self, item):
            self._items.append(item)
            captured.append(item)
            saw.set()
        def get_nowait(self):
            return self._items.pop(0)

    loop = _StubLoop()
    q = _StubQ()
    hub._subs.append((loop, q))
    try:
        # Sleep 1 ms ensures mtime_ns moves on filesystems with coarser
        # resolution (Windows NTFS has 100-ns ticks, but truly identical
        # ns happens often when writes are back-to-back).
        time.sleep(0.01)
        target.write_bytes(b"\x02" * 32)
        hub.force_rescan()
    finally:
        hub.unsubscribe(loop, q)

    kinds = {ev.get("kind") for ev in captured}
    assert "modify" in kinds, f"no modify in {kinds!r} (captured={captured!r})"


def test_force_rescan_detects_delete(srv, staging_dir):
    """Deleting a tracked file produces a 'delete' event."""
    hub = srv._LIVE_RELOAD_HUB
    target = staging_dir / "__live_reload_test_delete.njm"
    target.write_bytes(b"\x03" * 8)
    hub.force_rescan()
    target.unlink()
    n = hub.force_rescan()
    assert n >= 1


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


def test_events_status_returns_state(client):
    r = client.get("/api/events/status")
    assert r.status_code == 200
    data = r.json()
    assert "watched_dirs" in data
    assert "subscribers" in data
    assert "poll_seconds" in data


def test_events_rescan_endpoint(client, srv, staging_dir):
    target = staging_dir / "__live_reload_test_endpoint.njm"
    target.write_bytes(b"endpoint test")
    # Force one rescan first (so the create gets absorbed into a known
    # state). Then drop one more file and assert events_fired bumps.
    client.post("/api/events/rescan")
    target2 = staging_dir / "__live_reload_test_endpoint2.njm"
    target2.write_bytes(b"endpoint test 2")
    r = client.post("/api/events/rescan")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["events_fired"] >= 1


def test_events_route_is_registered(srv):
    """The /api/events route is registered on the FastAPI app.

    We don't open the actual stream in pytest — TestClient blocks on
    long-lived SSE because the generator awaits is_disconnected() and
    the test thread can't async-cancel it cleanly. The full
    publish→subscribe flow is exercised at the hub level in the tests
    below, which is the same code path /api/events uses.
    """
    paths = {r.path for r in srv.app.routes}
    assert "/api/events" in paths
    assert "/api/events/status" in paths
    assert "/api/events/rescan" in paths


def test_publish_delivers_to_subscriber(srv):
    """A direct hub publish reaches an attached subscriber's queue.

    This is the core delivery contract that /api/events relies on. We
    test it via the hub's public surface rather than through TestClient
    because the SSE stream's await-loop fights synchronous iteration.
    """
    import asyncio
    hub = srv._LIVE_RELOAD_HUB
    delivered: list[dict] = []

    async def driver():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        hub._subs.append((loop, q))
        try:
            # Publish from THIS thread — call_soon_threadsafe still
            # works because the loop is running.
            hub.publish({"path": "cache/njm_export/foo.njm", "kind": "create"})
            hub.publish({"path": "cache/njm_export/bar.njm", "kind": "modify"})
            # Drain.
            for _ in range(2):
                ev = await asyncio.wait_for(q.get(), timeout=2.0)
                delivered.append(ev)
        finally:
            hub.unsubscribe(loop, q)

    asyncio.run(driver())
    assert len(delivered) == 2
    paths = {d["path"] for d in delivered}
    assert "cache/njm_export/foo.njm" in paths
    assert "cache/njm_export/bar.njm" in paths


def test_force_rescan_publishes_to_subscribers(srv, staging_dir):
    """force_rescan() pushes events through the same path /api/events uses."""
    import asyncio
    hub = srv._LIVE_RELOAD_HUB
    target = staging_dir / "__live_reload_test_pubsub.njm"
    delivered: list[dict] = []

    async def driver():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        hub._subs.append((loop, q))
        try:
            # Drain anything already in the queue from setup leftovers.
            try:
                while True:
                    q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            # Now seed an event by writing on disk + forcing rescan.
            # Run rescan in an executor so it doesn't block the loop.
            target.write_bytes(b"new file")
            await loop.run_in_executor(None, hub.force_rescan)
            # Pull at least one event.
            ev = await asyncio.wait_for(q.get(), timeout=3.0)
            delivered.append(ev)
        finally:
            hub.unsubscribe(loop, q)

    asyncio.run(driver())
    assert len(delivered) >= 1
    assert delivered[0]["path"].startswith("cache/njm_export/")
    assert delivered[0]["kind"] in ("create", "modify")


def test_subscriber_count_increments_and_decrements(srv):
    """Subscribing via the hub bumps the count; unsubscribing clears it."""
    hub = srv._LIVE_RELOAD_HUB
    base = hub.subscriber_count()

    class _StubLoop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)

    class _StubQ:
        def full(self):
            return False
        def put_nowait(self, item):
            pass
        def get_nowait(self):
            return None

    loop = _StubLoop()
    q = _StubQ()
    hub._subs.append((loop, q))
    assert hub.subscriber_count() == base + 1
    hub.unsubscribe(loop, q)
    assert hub.subscriber_count() == base


def test_publish_drops_oldest_when_queue_full(srv):
    """A stalled subscriber doesn't grow memory unbounded."""
    import asyncio
    hub = srv._LIVE_RELOAD_HUB

    # We create a FakeLoop that immediately runs callbacks to mimic
    # call_soon_threadsafe; that exercises the put-with-drop path.
    delivered: list[dict] = []

    class _StubLoop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)

    # Real asyncio.Queue with a tiny maxsize so we hit the drop branch.
    # Need an event loop to construct the Queue.
    loop = asyncio.new_event_loop()
    try:
        q = asyncio.Queue(maxsize=2, loop=None) if False else None  # placeholder
        # Construct via the running loop trick:
        async def make_q():
            return asyncio.Queue(maxsize=2)
        q = loop.run_until_complete(make_q())
        # Wrap in our patched delivery helper.
        original = hub._enqueue
        def _spy(qq, ev):
            original(qq, ev)
            try:
                # peek by draining one item
                while True:
                    item = qq.get_nowait()
                    delivered.append(item)
            except asyncio.QueueEmpty:
                pass

        # Push 5 events; queue maxsize=2 -> drops oldest as it fills.
        sub_loop = _StubLoop()
        hub._subs.append((sub_loop, q))
        for i in range(5):
            hub.publish({"path": f"x/{i}.njm", "kind": "create"})
        # Drain everything that landed; we should see only the LAST 2-3
        # because earlier pushes were drop-oldest'd.
        try:
            while True:
                delivered.append(q.get_nowait())
        except asyncio.QueueEmpty:
            pass
        hub.unsubscribe(sub_loop, q)
    finally:
        loop.close()

    paths = [d["path"] for d in delivered]
    assert any(p.endswith("4.njm") for p in paths), f"latest event missing in {paths}"
    # We should have AT MOST 2 surviving (queue maxsize=2).
    assert len(delivered) <= 2, f"queue overflowed: {paths}"
