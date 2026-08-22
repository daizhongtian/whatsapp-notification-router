"""Build a deterministic, credential-free submission archive for ``code/``."""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import Sequence


CODE_ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = frozenset(
    {
        "main.py",
        "README.md",
        "LICENSE",
        "ARCHITECTURE.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-media.lock",
        "requirements-dev.lock",
        ".env.example",
        "router/pipeline.py",
        "evaluation/main.py",
        "evaluation/audit_dataset.py",
        "prompts/content_facts.md",
    }
)
EXCLUDED_PARTS = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        ".venv",
        "venv",
        "build",
        "dist",
        "node_modules",
    }
)
SECRET_PATTERNS = (
    re.compile(rb"ccc_live_[A-Za-z0-9_-]{16,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
MAX_SOURCE_FILE_BYTES = 5_000_000


class PackageError(RuntimeError):
    """Raised when source cannot be packaged without violating the contract."""


def _eligible_files() -> list[Path]:
    files: list[Path] = []
    for path in CODE_ROOT.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(CODE_ROOT)
        parts = set(relative.parts)
        if parts & EXCLUDED_PARTS:
            continue
        # Test fixtures use UUID-named hidden directories.  A killed test can
        # leave one behind, so never allow hidden directory contents into the
        # submission.  The root-level .env.example remains the only permitted
        # dotfile and contains placeholders only.
        if any(part.startswith(".") for part in relative.parts[:-1]):
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix.casefold() in {
            ".pyc",
            ".pyo",
            ".zip",
            ".tmp",
            ".log",
            ".db",
            ".sqlite",
        }:
            continue
        if path.name.startswith(".") and path.name != ".env.example":
            continue
        if path.stat().st_size > MAX_SOURCE_FILE_BYTES:
            raise PackageError(f"unexpected oversized source file: {relative.as_posix()}")
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(CODE_ROOT).as_posix())


def build_archive(output_path: str | Path) -> Path:
    """Create and read back a deterministic ZIP containing only safe code files."""

    destination = Path(output_path).expanduser().resolve()
    if destination.is_relative_to(CODE_ROOT):
        raise PackageError("write the submission archive outside code/")
    files = _eligible_files()
    names = {path.relative_to(CODE_ROOT).as_posix() for path in files}
    missing = sorted(REQUIRED_FILES - names)
    if missing:
        raise PackageError(f"required package files are missing: {', '.join(missing)}")

    for path in files:
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                raise PackageError(
                    f"credential-like value found in {path.relative_to(CODE_ROOT).as_posix()}"
                )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in files:
                name = path.relative_to(CODE_ROOT).as_posix()
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=9)
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    with zipfile.ZipFile(destination, "r") as archive:
        archived_names = set(archive.namelist())
        if archived_names != names or archive.testzip() is not None:
            raise PackageError("archive readback verification failed")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = build_archive(args.output)
    except (OSError, PackageError, zipfile.BadZipFile) as exc:
        print(f"package error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
