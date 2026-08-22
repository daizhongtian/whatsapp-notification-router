from __future__ import annotations

import contextlib
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import package as package_module  # noqa: E402


@contextlib.contextmanager
def package_tempdir():
    path = CODE_ROOT / (".package-test-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class PackageSelectionTests(unittest.TestCase):
    def test_hidden_test_residue_and_dotfiles_are_excluded(self) -> None:
        with package_tempdir() as root:
            (root / "main.py").write_text("pass\n", encoding="utf-8")
            (root / ".env.example").write_text("AI_API_KEY=\n", encoding="utf-8")
            (root / ".secret").write_text("placeholder\n", encoding="utf-8")
            residue = root / ".offline-core-test-residue"
            residue.mkdir()
            (residue / "messages.csv").write_text("message_id\n", encoding="utf-8")

            with patch.object(package_module, "CODE_ROOT", root):
                names = {
                    path.relative_to(root).as_posix()
                    for path in package_module._eligible_files()
                }

        self.assertEqual(names, {".env.example", "main.py"})


if __name__ == "__main__":
    unittest.main()
