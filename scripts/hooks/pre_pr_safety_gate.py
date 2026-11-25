#!/usr/bin/env python3
"""Antigravity PreToolUse Safety & Anti-Oscillation Gate (.agents/hooks.json).

Implements the Antigravity hooks.json PreToolUse stdin/stdout JSON contract:
1. Hard-denies direct pushes to ``main`` (``git push origin main`` / ``HEAD:main``).
2. Runs ``pytest tests/meta/ -q`` automatically before allowing ``gh pr create|merge``
   or ``scripts/github_pr_helper.py --create|--merge``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

DIRECT_MAIN_PUSH_RE = re.compile(
    r"(?:^|[;&|]\s*)git\s+push(?:\s+[-\w=]+)*\s+origin\s+(?:HEAD:)?main(?:\s|$|[;&|])"
)
GH_PR_GATE_RE = re.compile(
    r"(?:(?:^|[;&|]\s*)gh\s+pr\s+(?:create|merge)\b"
    r"|github_pr_helper\.py\b[^\n;&|]*--(?:create|merge)\b)"
)


def _resolve_repo_python(repo_root: Path) -> str:
    """Resolve the repository virtualenv interpreter across Windows and POSIX before falling back."""
    win_venv_py = repo_root / ".venv" / "Scripts" / "python.exe"
    if win_venv_py.exists():
        return str(win_venv_py)
    posix_venv_py = repo_root / ".venv" / "bin" / "python"
    if posix_venv_py.exists():
        return str(posix_venv_py)
    return sys.executable


def evaluate_command(
    cmd: str,
    repo_root: Path,
    *,
    run_meta_suite: bool = True,
) -> dict[str, str]:
    """Evaluate a shell command string and return a PreToolUse decision dict."""
    if not cmd:
        return {"decision": "allow"}

    if DIRECT_MAIN_PUSH_RE.search(cmd):
        return {
            "decision": "deny",
            "reason": (
                "Direct pushes to `main` are prohibited by AGENTS.md Section 2. "
                "Push to a feature branch and use `gh pr create` + `gh pr merge`."
            ),
        }

    if run_meta_suite and GH_PR_GATE_RE.search(cmd):
        py_bin = _resolve_repo_python(repo_root)
        res = subprocess.run(  # noqa: S603
            [py_bin, "-m", "pytest", "tests/meta/", "-q"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        if res.returncode != 0:
            tail = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()[-600:]
            return {
                "decision": "deny",
                "reason": (f"Blocked by Pre-PR Meta Gate (`tests/meta/` failed):\n{tail}"),
            }

    return {"decision": "allow"}


def main() -> int:
    """Read PreToolUse JSON payload on stdin and print allow/deny decision."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        sys.stdout.write(json.dumps({"decision": "allow"}) + "\n")
        return 0

    tool_call = payload.get("toolCall") or {}
    args = tool_call.get("args") or {}
    cmd = str(args.get("CommandLine") or "")
    out = evaluate_command(cmd, repo_root=repo_root, run_meta_suite=True)
    sys.stdout.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
