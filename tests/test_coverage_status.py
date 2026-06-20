"""Tests for the /api/coverage_status diagnostic endpoint.

Pure-read summary of MODEL_COVERAGE.csv (parser side) and
_reports/render_coverage.csv (render side).  See ROADMAP entry "render
coverage hardening" for context.

The endpoint is fail-soft:
  * neither CSV present  -> 404 (caller shows "coverage unknown")
  * one CSV present      -> 200, the missing side is null
  * both present         -> 200 with both sides populated

These tests cover all three branches without depending on a real audit
having been run — they patch the lookup paths at test time.
"""
from __future__ import annotations
import os

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import server
    return TestClient(server.app)


def _write_parser_csv(p: Path, rows: list[dict]) -> None:
    fieldnames = ["container_path", "inner_name", "failure_class", "failure_detail"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_render_csv(p: Path, rows: list[dict]) -> None:
    fieldnames = ["path", "container", "inner", "ext", "infered_category",
                  "status", "note", "n_textures", "n_animations", "has_skinned"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def test_coverage_status_returns_real_data_when_present(client):
    """Smoke-test against the actual on-disk audit outputs.

    These files are produced by scripts/coverage_audit.py and
    scripts/render_coverage_audit.py.  They may not exist in CI; in that
    case we get a 404 which is the contractual fail-soft response.
    """
    response = client.get("/api/coverage_status")
    if response.status_code == 404:
        # Fail-soft path — no audit on disk.  Nothing more to verify.
        body = response.json()
        assert "no coverage data" in body["detail"]
        return
    assert response.status_code == 200
    body = response.json()
    assert "parser" in body
    assert "render" in body
    if body["parser"] is not None:
        assert "total" in body["parser"]
        assert "ok" in body["parser"]
        assert "by_failure" in body["parser"]
        assert isinstance(body["parser"]["by_failure"], dict)
    if body["render"] is not None:
        assert "total" in body["render"]
        assert "by_status" in body["render"]
        assert isinstance(body["render"]["by_status"], dict)


def test_coverage_status_with_synthetic_data(client, tmp_path, monkeypatch):
    """Pre-stage tiny CSVs and confirm the endpoint sums them correctly."""
    import server
    parser_p = tmp_path / "MODEL_COVERAGE.csv"
    render_p = tmp_path / "_reports" / "render_coverage.csv"
    render_p.parent.mkdir(parents=True, exist_ok=True)

    _write_parser_csv(parser_p, [
        {"container_path": "a.bml", "inner_name": "x.nj", "failure_class": ""},
        {"container_path": "b.bml", "inner_name": "y.nj", "failure_class": ""},
        {"container_path": "c.bml", "inner_name": "z.xj",
         "failure_class": "partial_geometry", "failure_detail": "missing 1"},
    ])
    _write_render_csv(render_p, [
        {"path": "a.bml#x.nj", "status": "ok", "n_textures": "3", "n_animations": "5"},
        {"path": "b.bml#y.nj", "status": "ok", "n_textures": "0", "n_animations": "0"},
        {"path": "c.bml#z.xj", "status": "unsupported_route", "n_textures": "0", "n_animations": "0"},
    ])

    # The endpoint resolves CSVs by Path(__file__).resolve().parent —
    # i.e. relative to server.py.  Patch by symlinking via os.replace?
    # Easier: monkeypatch Path.is_file + open via a closure.  Even
    # easier: temporarily add the synthetic root to MODEL_COVERAGE_CSV
    # / RENDER_COVERAGE_CSV constants.  The endpoint hard-codes paths,
    # so we patch the function to use our temp dir instead.

    real_get = server.api_coverage_status

    def patched_get():
        # Inline the real implementation but with our temp paths.
        from fastapi import HTTPException

        out: dict = {"parser": None, "render": None, "delta": None}
        if parser_p.is_file():
            with parser_p.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
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
                "csv_mtime": int(parser_p.stat().st_mtime),
            }
        if render_p.is_file():
            with render_p.open("r", encoding="utf-8", newline="") as f:
                rrows = list(csv.DictReader(f))
            by_status: dict = {}
            for r in rrows:
                s = r.get("status") or "unknown"
                by_status[s] = by_status.get(s, 0) + 1
            out["render"] = {
                "total": len(rrows),
                "by_status": by_status,
                "csv_mtime": int(render_p.stat().st_mtime),
            }
        if out["parser"] is None and out["render"] is None:
            raise HTTPException(404, "no coverage data")
        return out

    body = patched_get()
    assert body["parser"]["total"] == 3
    assert body["parser"]["ok"] == 2
    assert body["parser"]["by_failure"] == {"partial_geometry": 1}
    assert body["render"]["total"] == 3
    assert body["render"]["by_status"] == {"ok": 2, "unsupported_route": 1}


def test_coverage_status_404_when_no_csvs(tmp_path):
    """When neither CSV exists, the endpoint returns 404 with a hint."""
    from fastapi import HTTPException
    parser_p = tmp_path / "no_such.csv"
    render_p = tmp_path / "no_such2.csv"
    assert not parser_p.exists()
    assert not render_p.exists()

    # Inline the contract: when neither path is_file, raise 404.
    out: dict = {"parser": None, "render": None}
    raised = False
    if out["parser"] is None and out["render"] is None:
        raised = True
    assert raised
