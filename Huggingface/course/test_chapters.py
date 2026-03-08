"""Validate syntax and documentation quality for chapter scripts."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).resolve().parent


def test_chapter(num: int) -> Tuple[str, str, str]:
    """Run structural checks for one chapter file.

    Args:
        num: Chapter number in the range [1, 9].

    Returns:
        A tuple of (filename, status, details).
    """
    filename = f"chapter{num}.py"
    filepath = BASE_DIR / filename

    if not filepath.is_file():
        return filename, "FAIL", "File not found"

    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()
    line_count = len(lines)

    try:
        ast.parse(content)
    except SyntaxError as exc:
        return filename, "FAIL", f"Syntax error line {exc.lineno}: {exc.msg}"

    first_line = lines[0] if lines else ""
    has_docstring = '"""' in first_line or "'''" in first_line
    has_bilingual_header = any("Chapter" in line and "#" in line for line in lines[:100])
    has_architecture_hint = any("Architecture" in line or "Diagram" in line for line in lines[:100])
    has_pipeline = "pipeline" in content

    code_lines = [
        line for line in lines
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith('"""')
        and not line.strip().startswith("'''")
    ]
    comment_lines = [line for line in lines if line.strip().startswith("#")]

    details = (
        f"{line_count} lines, "
        f"{len(code_lines)} code, "
        f"{len(comment_lines)} comments, "
        f"docstring={'Y' if has_docstring else 'N'}, "
        f"bilingual={'Y' if has_bilingual_header else 'N'}, "
        f"diagram={'Y' if has_architecture_hint else 'N'}, "
        f"pipeline={'Y' if has_pipeline else 'N'}"
    )
    return filename, "PASS", details


def main() -> int:
    """Execute quality checks for all chapter scripts."""
    print("=" * 70)
    print("Huggingface Chapter Test Suite")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for chapter_num in range(1, 10):
        name, status, details = test_chapter(chapter_num)
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name}: {details}")
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print()
    print("-" * 70)
    print(f"  Results: {passed} passed, {failed} failed out of 9")
    print("-" * 70)
    print("  All tests passed!" if failed == 0 else f"  {failed} test(s) failed!")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
