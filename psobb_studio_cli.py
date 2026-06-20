"""Console entry point for the psobb-studio server.

Installed as the ``psobb-studio`` command by ``pip install -e .`` (see
pyproject.toml). Examples:

    psobb-studio                         # 127.0.0.1:8765, PSO_DATA_DIR or ~/PSOBB.IO/data
    psobb-studio --port 9000 --reload
    psobb-studio --data-dir /games/PSOBB/data

Equivalent low-level form:  python -m uvicorn server:app --port 8765
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="psobb-studio",
        description="Launch the psobb-studio asset server (localhost-only).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("PSO_DATA_DIR"),
        help="PSOBB data directory (sets PSO_DATA_DIR); "
        "default $PSO_DATA_DIR or ~/PSOBB.IO/data",
    )
    parser.add_argument(
        "--reload", action="store_true", help="auto-reload on code changes (development)"
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir or os.path.expanduser("~/PSOBB.IO/data")
    os.environ["PSO_DATA_DIR"] = data_dir

    # Run from the repo root so ``server:app`` and ``static/`` resolve regardless
    # of where the console script is invoked from.
    os.chdir(Path(__file__).resolve().parent)

    import uvicorn  # imported lazily so importing this module stays cheap

    print(f"psobb-studio -> http://{args.host}:{args.port}  (PSO_DATA_DIR={data_dir})")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
