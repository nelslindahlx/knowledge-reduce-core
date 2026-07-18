#!/usr/bin/env python3
"""Install/verify the repo-local Hermes skill from the canonical manifest."""
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / ".agents" / "skills" / "knowledge-reduce-core" / "SKILL.md"
DEST = Path.home() / ".hermes" / "skills" / "software-development" / "knowledge-reduce-core" / "SKILL.md"


def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def main() -> int:
    if not CANONICAL.exists():
        print(f"error: canonical skill not found: {CANONICAL}", file=sys.stderr)
        return 2

    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CANONICAL, DEST)

    diff = run(["diff", "-q", str(CANONICAL), str(DEST)])
    if diff.returncode != 0:
        print("error: installed skill diverged from canonical source", file=sys.stderr)
        print(diff.stdout, file=sys.stderr)
        print(diff.stderr, file=sys.stderr)
        return 3

    print(f"installed: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
