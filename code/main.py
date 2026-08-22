"""Command-line entry point for the Message Notification Router."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

from router.pipeline import PipelineError, run_pipeline_with_report


CODE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODE_DIR.parent


def _route_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="message-router route",
        description="Route every incoming message and write a validated output CSV.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "dataset",
        help="Directory containing messages.csv and participant context files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "dataset" / "output.csv",
        help="Destination for the exact six-column predictions CSV",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "auto", "hybrid", "api"),
        default="offline",
        help=(
            "offline is deterministic; auto enriches all content when configured; "
            "hybrid selects uncertain text plus all media; api requires full Gateway coverage"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional content-fact cache (contains no credentials or personalized decisions)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the successful JSON summary",
    )
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        help="Optional content-free API/performance diagnostics JSON",
    )
    return parser


def _write_json_atomic(path: Path, value: object) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "route":
        return values[1:]
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = _route_parser()
    args = parser.parse_args(_normalize_argv(argv))
    try:
        run = run_pipeline_with_report(
            args.dataset,
            args.output,
            mode=args.mode,
            cache_dir=args.cache_dir,
        )
    except (OSError, ValueError, PipelineError) as exc:
        print(f"routing error: {exc}", file=sys.stderr)
        return 2

    predictions = run.predictions
    if args.diagnostics_json is not None:
        _write_json_atomic(args.diagnostics_json, run.report)
    if not args.quiet:
        counts = Counter(prediction.action.value for prediction in predictions)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "rows": len(predictions),
                    "actions": dict(sorted(counts.items())),
                    "mode": args.mode,
                    "api": run.report["api"],
                    "elapsed_seconds": run.report["elapsed_seconds"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
