"""Tests for the archive entry editor — formats/archive_entry.py + the
``/api/archive/*`` endpoints (duplicate / create / delete / rename).

Two layers:

  PURE (no disk): exercises afs_* / bml_* over in-memory bytes, using
  ``write_afs`` / ``pack_bml`` as the construction oracle and re-parsing as
  the verification oracle. Covers the two parity traps:
    * an UNTOUCHED AFS round-trips byte-exact through the NO-name-table path
      (synthesising a table would change the byte layout);
    * a BML duplicate preserves ``is_compressed`` PRS bytes verbatim and the
      "lying header" (has_textures=1 but 0x800-aligned) round-trips.

  ENDPOINT (TestClient, monkeypatched DATA_DIR/LIVE_DATA_DIR -> tmp dirs):
  proves the WRITE goes BACK to the resolved source path with a .pre_edit
  backup, that GSL -> 400, and that the multipart create endpoint works.
  No real PSOBB install required — the AFS/BML fixtures are synthesised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import server  # noqa: E402
from formats import archive_entry as ae  # noqa: E402
from formats.afs import parse_afs, write_afs  # noqa: E402
from formats.afs_reader import _afs_filename_table  # noqa: E402
from formats.bml import (  # noqa: E402
    BmlPackEntry,
    pack_bml,
    parse_bml_for_pack,
    parse_bml_pack_meta,
)


# ---------------------------------------------------------------------------
# fixtures: synthetic archives (CI-safe, no game assets)
# ---------------------------------------------------------------------------
def _afs_no_names() -> bytes:
    """Shipped-style AFS: no filename table (the parity trap subject)."""
    return write_afs([b"AAA", b"BBBB", b"CCCCC"])


def _afs_with_names() -> bytes:
    return write_afs([b"AAA", b"BBBB"], names=["one", "two"])


def _bml_simple() -> bytes:
    """A 2-entry uncompressed BML (compression byte 0x50 in the header is
    written by pack_bml; entries stored raw via is_compressed=False)."""
    ents = [
        BmlPackEntry(name="alpha.nj", data=b"hello-alpha-bytes"),
        BmlPackEntry(name="beta.nj", data=b"world-beta-data-xx"),
    ]
    return pack_bml(ents)


def _bml_lying_header() -> bytes:
    """A BML whose header says has_textures=1 but is 0x800-aligned with NO
    real texture (the 23 player-NJ archives). Built by forcing the override
    so the round-trip path must preserve it."""
    ents = [BmlPackEntry(name="plike.nj", data=b"player-nj-bytes-aaa")]
    return pack_bml(ents, has_textures_override=True, file_alignment=0x800)


# ===========================================================================
# PURE — AFS
# ===========================================================================
def test_archive_kind():
    assert ae.archive_kind("x.afs") == "afs"
    assert ae.archive_kind("X.AFS") == "afs"
    assert ae.archive_kind("y.bml") == "bml"
    assert ae.archive_kind("z.gsl") is None
    assert ae.archive_kind("z.prs") is None
    assert ae.archive_kind("") is None
    assert ae.archive_kind(None) is None  # type: ignore[arg-type]


def test_afs_untouched_rewrite_byte_exact():
    """The NO-name-table parity trap: re-serialising an untouched no-name
    AFS via the same path archive_entry uses must be byte-identical."""
    buf = _afs_no_names()
    assert _afs_filename_table(buf) is None
    rewrote = write_afs(list(parse_afs(buf)), names=None)
    assert rewrote == buf


def test_afs_duplicate_appends_identical_bytes():
    buf = _afs_no_names()
    blobs = parse_afs(buf)
    new_buf, new_index = ae.afs_duplicate(buf, 0)
    nb = parse_afs(new_buf)
    assert len(nb) == len(blobs) + 1
    assert new_index == len(blobs)        # appended at the end
    assert nb[new_index] == blobs[0]      # dup bytes byte-identical to source
    assert _afs_filename_table(new_buf) is None  # still no table (no synthesis)


def test_afs_duplicate_offsets_aligned():
    buf = _afs_no_names()
    new_buf, _ = ae.afs_duplicate(buf, 1)
    # Re-parse is the oracle: no ValueError means every (offset,size) is in
    # range. Header file_count must equal the new entry count.
    import struct
    count = struct.unpack_from("<H", new_buf, 4)[0]
    assert count == len(parse_afs(new_buf))


def test_afs_create_empty_and_copy_first():
    buf = _afs_no_names()
    n0 = len(parse_afs(buf))
    eb, ei = ae.afs_create(buf, None, "empty")
    assert len(parse_afs(eb)) == n0 + 1
    assert parse_afs(eb)[ei] == b""
    cb, ci = ae.afs_create(buf, None, "copy_first")
    assert parse_afs(cb)[ci] == parse_afs(buf)[0]
    # explicit blob
    bb, bi = ae.afs_create(buf, b"NEWBLOB", None)
    assert parse_afs(bb)[bi] == b"NEWBLOB"


def test_afs_create_bad_template_raises():
    with pytest.raises(ValueError):
        ae.afs_create(_afs_no_names(), None, "nonsense")


def test_afs_delete_renumbers():
    buf = _afs_no_names()
    blobs = parse_afs(buf)
    db = ae.afs_delete(buf, 1)
    nb = parse_afs(db)
    assert len(nb) == len(blobs) - 1
    assert nb[0] == blobs[0]
    assert nb[1] == blobs[2]   # the slot after the deleted one renumbered down


def test_afs_delete_out_of_range_raises():
    with pytest.raises(IndexError):
        ae.afs_delete(_afs_no_names(), 99)


def test_afs_rename_requires_name_table():
    # no table -> ValueError (API maps to 409)
    with pytest.raises(ValueError):
        ae.afs_rename(_afs_no_names(), 0, "newname")
    # with a table -> works and the name changes
    nb = _afs_with_names()
    ren = ae.afs_rename(nb, 1, "renamed")
    assert _afs_filename_table(ren) == ["one", "renamed"]


def test_afs_rename_duplicate_name_raises():
    nb = _afs_with_names()
    with pytest.raises(ValueError):
        ae.afs_rename(nb, 1, "one")  # collides with slot 0


def test_afs_with_names_duplicate_dedups():
    nb = _afs_with_names()
    new_buf, _ = ae.afs_duplicate(nb, 0)
    names = _afs_filename_table(new_buf)
    assert names is not None
    assert len(names) == 3
    assert names[2] != names[0]  # de-duplicated


# ===========================================================================
# PURE — BML
# ===========================================================================
def test_bml_untouched_repack_byte_exact():
    buf = _bml_simple()
    rt = ae._bml_repack(buf, parse_bml_for_pack(buf))
    assert rt == buf


def test_bml_duplicate_preserves_compressed_and_data():
    buf = _bml_simple()
    ents = parse_bml_for_pack(buf)
    src = ents[0]
    nb = ae.bml_duplicate(buf, src.name, "alpha_copy.nj")
    ne = parse_bml_for_pack(nb)
    assert len(ne) == len(ents) + 1
    dup = ne[-1]
    assert dup.name == "alpha_copy.nj"
    assert dup.data == src.data                      # PRS bytes preserved verbatim
    assert dup.is_compressed == src.is_compressed


def test_bml_duplicate_lying_header_round_trips():
    buf = _bml_lying_header()
    meta = parse_bml_pack_meta(buf)
    assert meta["has_textures"] is True
    assert meta["file_alignment"] == 0x800           # lying: tex flag set, 0x800 aligned
    # repack untouched stays byte-exact
    assert ae._bml_repack(buf, parse_bml_for_pack(buf)) == buf
    # duplicate still produces a parseable archive that keeps the meta
    nb = ae.bml_duplicate(buf, "plike.nj", "plike_dup.nj")
    assert parse_bml_pack_meta(nb)["has_textures"] is True
    assert len(parse_bml_for_pack(nb)) == 2


def test_bml_duplicate_bad_name_raises():
    buf = _bml_simple()
    with pytest.raises(ValueError):
        ae.bml_duplicate(buf, "alpha.nj", "beta.nj")          # collision
    with pytest.raises(ValueError):
        ae.bml_duplicate(buf, "alpha.nj", "x" * 40)            # >32 bytes
    with pytest.raises(KeyError):
        ae.bml_duplicate(buf, "missing.nj", "ok.nj")           # src not found


def test_bml_create_and_delete_round_trip():
    buf = _bml_simple()
    payload = b"gamma-payload-bytes"
    nb = ae.bml_create(buf, "gamma.nj", payload)
    ne = parse_bml_for_pack(nb)
    # The created entry is PRS-compressed by the packer (is_compressed=False
    # input), so its on-disk decompressed_size must equal the raw length and
    # decompressing it must recover the original bytes.
    from formats.bml import _prs_decompress
    gamma = next(e for e in ne if e.name == "gamma.nj")
    assert gamma.decompressed_size == len(payload)
    assert _prs_decompress(gamma.data) == payload
    # delete it again -> back to original count
    db = ae.bml_delete(nb, "gamma.nj")
    assert len(parse_bml_for_pack(db)) == len(parse_bml_for_pack(buf))


def test_bml_create_precompressed_verbatim():
    """is_compressed=True stores the PRS bytes verbatim (no re-compression)."""
    buf = _bml_simple()
    # take an already-compressed entry's bytes from the source archive
    src = parse_bml_for_pack(buf)[0]
    assert src.is_compressed
    nb = ae.bml_create(buf, "verbatim.nj", src.data,
                       is_compressed=True)
    got = next(e for e in parse_bml_for_pack(nb) if e.name == "verbatim.nj")
    assert got.data == src.data


def test_bml_delete_missing_raises():
    with pytest.raises(KeyError):
        ae.bml_delete(_bml_simple(), "nope.nj")


def test_bml_rename():
    buf = _bml_simple()
    nb = ae.bml_rename(buf, "alpha.nj", "alpha2.nj")
    names = [e.name for e in parse_bml_for_pack(nb)]
    assert "alpha2.nj" in names and "alpha.nj" not in names
    with pytest.raises(ValueError):
        ae.bml_rename(buf, "alpha.nj", "beta.nj")  # collision


# ===========================================================================
# ENDPOINT — monkeypatched tmp data dirs + TestClient
# ===========================================================================
@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point server.DATA_DIR + LIVE_DATA_DIR at tmp dirs and seed archives.

    DATA_DIR (= DEV) holds the editable copies; LIVE is a separate dir. The
    resolver searches DATA_DIR first, so a seeded DATA_DIR archive is the
    write target.
    """
    data = (tmp_path / "data").resolve()
    live = (tmp_path / "live").resolve()
    data.mkdir(parents=True)
    live.mkdir(parents=True)
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "LIVE_DATA_DIR", live)
    # seed
    (data / "Item.afs").write_bytes(_afs_no_names())
    (data / "model.bml").write_bytes(_bml_simple())
    (data / "junk.gsl").write_bytes(b"GSL stuff not a real container")
    return {"data": data, "live": live}


@pytest.fixture
def client(env):
    return TestClient(server.app)


def test_endpoint_duplicate_afs_writes_resolved_path_with_backup(env, client):
    """Duplicate writes BACK to the resolved DATA_DIR archive + .pre_edit
    backup, and the on-disk archive grows by one entry."""
    target = env["data"] / "Item.afs"
    before = target.read_bytes()
    n_before = len(parse_afs(before))

    r = client.post("/api/archive/duplicate_entry",
                    json={"archive": "Item.afs", "index": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "afs"
    assert body["new_index"] == n_before
    assert body["target_path"] == str(target)

    # the actual opened file grew + reparsed has N+1
    after = target.read_bytes()
    assert len(after) >= len(before)
    assert len(parse_afs(after)) == n_before + 1

    # a .pre_edit backup of the target exists and equals the pre-edit bytes
    assert body["backup_path"]
    bak = Path(body["backup_path"])
    assert bak.exists()
    assert bak.read_bytes() == before
    assert ".pre_edit_" in bak.name


def test_endpoint_duplicate_does_not_touch_live(env, client):
    """When the archive resolves in DATA_DIR, LIVE is never written."""
    # seed a same-named file in LIVE; it must stay byte-identical.
    live_copy = env["live"] / "Item.afs"
    live_copy.write_bytes(b"LIVE-SENTINEL-DO-NOT-TOUCH")
    r = client.post("/api/archive/duplicate_entry",
                    json={"archive": "Item.afs", "index": 0})
    assert r.status_code == 200, r.text
    assert live_copy.read_bytes() == b"LIVE-SENTINEL-DO-NOT-TOUCH"


def test_endpoint_writes_to_live_when_only_in_live(env, client):
    """Owner write model: if the archive lives ONLY in LIVE, the edit
    writes back to LIVE (the path the user opened it from)."""
    # a fresh AFS present ONLY in LIVE
    live_only = env["live"] / "LiveOnly.afs"
    live_only.write_bytes(_afs_no_names())
    n_before = len(parse_afs(live_only.read_bytes()))
    r = client.post("/api/archive/duplicate_entry",
                    json={"archive": "LiveOnly.afs", "index": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_path"] == str(live_only)
    assert len(parse_afs(live_only.read_bytes())) == n_before + 1


def test_endpoint_create_multipart_afs_empty(env, client):
    target = env["data"] / "Item.afs"
    n_before = len(parse_afs(target.read_bytes()))
    r = client.post("/api/archive/create_entry",
                    data={"archive": "Item.afs", "template": "empty"})
    assert r.status_code == 200, r.text
    assert r.json()["new_index"] == n_before
    assert len(parse_afs(target.read_bytes())) == n_before + 1


def test_endpoint_create_multipart_afs_with_file(env, client):
    target = env["data"] / "Item.afs"
    n_before = len(parse_afs(target.read_bytes()))
    r = client.post(
        "/api/archive/create_entry",
        data={"archive": "Item.afs"},
        files={"file": ("blob.bin", b"UPLOADED-BLOB-BYTES", "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    after = parse_afs(target.read_bytes())
    assert len(after) == n_before + 1
    assert after[r.json()["new_index"]] == b"UPLOADED-BLOB-BYTES"


def test_endpoint_create_bml_multipart(env, client):
    target = env["data"] / "model.bml"
    r = client.post(
        "/api/archive/create_entry",
        data={"archive": "model.bml", "new_name": "newentry.nj"},
        files={"file": ("e.nj", b"fresh-bml-entry-bytes", "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["new_entry_name"] == "newentry.nj"
    names = [e.name for e in parse_bml_for_pack(target.read_bytes())]
    assert "newentry.nj" in names


def test_endpoint_delete_afs_renumbers(env, client):
    target = env["data"] / "Item.afs"
    n_before = len(parse_afs(target.read_bytes()))
    r = client.request("DELETE", "/api/archive/entry",
                       json={"archive": "Item.afs", "index": 1})
    assert r.status_code == 200, r.text
    assert len(parse_afs(target.read_bytes())) == n_before - 1


def test_endpoint_rename_bml(env, client):
    r = client.post("/api/archive/rename_entry",
                    json={"archive": "model.bml",
                          "entry_name": "alpha.nj", "new_name": "alpha9.nj"})
    assert r.status_code == 200, r.text
    assert r.json()["new_path"] == "model.bml#alpha9.nj"


def test_endpoint_rename_afs_no_table_409(env, client):
    r = client.post("/api/archive/rename_entry",
                    json={"archive": "Item.afs", "index": 0, "new_name": "x"})
    assert r.status_code == 409, r.text


def test_endpoint_gsl_400(env, client):
    r = client.post("/api/archive/duplicate_entry",
                    json={"archive": "junk.gsl", "index": 0})
    assert r.status_code == 400
    assert "not supported" in r.text.lower()


def test_endpoint_unknown_archive_404(env, client):
    r = client.post("/api/archive/duplicate_entry",
                    json={"archive": "missing.afs", "index": 0})
    assert r.status_code == 404


def test_endpoint_afs_bad_index_404(env, client):
    r = client.post("/api/archive/duplicate_entry",
                    json={"archive": "Item.afs", "index": 999})
    assert r.status_code == 404


def test_endpoint_entries_list_afs(env, client):
    r = client.get("/api/archive/Item.afs/entries")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["supported"] is True and body["kind"] == "afs"
    assert len(body["entries"]) == len(parse_afs((env["data"] / "Item.afs").read_bytes()))


def test_endpoint_entries_list_bml(env, client):
    r = client.get("/api/archive/model.bml/entries")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "bml"
    names = [e["name"] for e in body["entries"]]
    assert "alpha.nj" in names


def test_endpoint_entries_gsl_unsupported(env, client):
    r = client.get("/api/archive/junk.gsl/entries")
    assert r.status_code == 200, r.text
    assert r.json()["supported"] is False


def test_endpoint_concurrent_lock_409(env, client):
    """Holding the per-archive lock makes a concurrent edit 409."""
    lk = server._get_lock(server._REPACK_LOCKS, "Item.afs", server.MAX_REPACK_LOCKS)
    assert lk.acquire(blocking=False)
    try:
        r = client.post("/api/archive/duplicate_entry",
                        json={"archive": "Item.afs", "index": 0})
        assert r.status_code == 409, r.text
    finally:
        lk.release()
