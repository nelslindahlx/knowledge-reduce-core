"""GitHub API PR Automation Helper.

Enables Maker/Checker agents to create, review, merge, close, and audit GitHub
Pull Requests cleanly without consuming GitHub Actions minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: S110
        pass

logger = logging.getLogger(__name__)

REPO_OWNER = "nelslindahlx"
REPO_NAME = "knowledge-reduce-core"
_API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"


def _get_auth_headers() -> dict[str, str]:
    token: str | None = None
    git_bin = shutil.which("git")
    gh_bin = shutil.which("gh")
    if git_bin:
        try:
            res = subprocess.run(  # noqa: S603
                [git_bin, "remote", "-v"],
                capture_output=True,
                text=True,
                check=False,
            )
            match = re.search(r"https://x-access-token:([a-zA-Z0-9_]+)@", res.stdout)
            if match:
                token = match.group(1)
        except Exception:  # noqa: S110
            pass
    if not token and gh_bin:
        try:
            res = subprocess.run(  # noqa: S603
                [gh_bin, "auth", "token"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                token = res.stdout.strip()
        except Exception:  # noqa: S110
            pass
    if not token:
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GitHub token not found via gh auth, git remote, or GH_TOKEN env var.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "KnowledgeReduceCore-Agent-PR-Helper",
    }


def _api_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    if not url.startswith("https://api.github.com/"):
        raise ValueError(f"Refusing non-GitHub-API URL: {url!r}")
    body_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(  # noqa: S310
        url,
        data=body_bytes,
        headers=_get_auth_headers(),
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw.strip() else {}


def create_github_pr(
    head_branch: str,
    title: str,
    body: str,
    base_branch: str = "main",
) -> dict[str, Any]:
    """Create a GitHub Pull Request via the GitHub REST API."""
    result: dict[str, Any] = _api_request(
        f"{_API_BASE}/pulls",
        method="POST",
        payload={
            "title": title,
            "head": head_branch,
            "base": base_branch,
            "body": body,
        },
    )
    return result


def merge_github_pr(
    pr_number: int,
    commit_title: str = "",
    commit_message: str = "",
    merge_method: str = "squash",
) -> dict[str, Any]:
    """Merge a GitHub Pull Request via the GitHub REST API."""
    payload: dict[str, Any] = {
        "merge_method": merge_method,
    }
    if commit_title:
        payload["commit_title"] = commit_title
    if commit_message:
        payload["commit_message"] = commit_message
    result: dict[str, Any] = _api_request(
        f"{_API_BASE}/pulls/{pr_number}/merge",
        method="PUT",
        payload=payload,
    )
    return result


def get_github_pr_details(pr_number: int) -> dict[str, Any]:
    """Retrieve metadata for a GitHub Pull Request."""
    result: dict[str, Any] = _api_request(f"{_API_BASE}/pulls/{pr_number}", method="GET")
    return result


def close_github_pr(pr_number: int, comment: str | None = None) -> dict[str, Any]:
    """Close a GitHub Pull Request with an optional explanatory comment."""
    if comment:
        _api_request(
            f"{_API_BASE}/issues/{pr_number}/comments",
            method="POST",
            payload={"body": comment},
        )
    result: dict[str, Any] = _api_request(
        f"{_API_BASE}/pulls/{pr_number}",
        method="PATCH",
        payload={"state": "closed"},
    )
    return result


def delete_github_branch(branch_name: str) -> bool:
    """Delete a remote branch on GitHub."""
    url = f"{_API_BASE}/git/refs/heads/{branch_name}"
    if not url.startswith("https://api.github.com/"):
        raise ValueError(f"Refusing non-GitHub-API URL: {url!r}")
    req = urllib.request.Request(  # noqa: S310
        url, headers=_get_auth_headers(), method="DELETE"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return bool(resp.status in (200, 204))
    except urllib.error.HTTPError as exc:
        logger.warning("Failed to delete remote branch %s: %s", branch_name, exc)
        return False


def list_open_prs() -> list[dict[str, Any]]:
    """List open Pull Requests targeting this repository."""
    result: list[dict[str, Any]] = _api_request(f"{_API_BASE}/pulls?state=open", method="GET")
    return result


def list_remote_branches() -> list[dict[str, Any]]:
    """List remote branches in this repository."""
    result: list[dict[str, Any]] = _api_request(f"{_API_BASE}/branches", method="GET")
    return result


def audit_zero_lingering_prs() -> bool:
    """Return True only when zero open PRs and zero non-main remote branches remain."""
    prs = list_open_prs()
    branches = list_remote_branches()
    feature_branches = [b["name"] for b in branches if b["name"] != "main"]

    clean = True
    if prs:
        clean = False
        print(f"FOUND {len(prs)} OPEN PULL REQUEST(S):")
        for pr in prs:
            print(
                f"  - PR #{pr['number']}: {pr['title']!r} ({pr['head']['ref']}) -> {pr['html_url']}"
            )
    else:
        print("OK: Zero open pull requests.")

    if feature_branches:
        clean = False
        print(
            f"FOUND {len(feature_branches)} NON-MAIN REMOTE BRANCH(ES): {', '.join(feature_branches)}"
        )
    else:
        print("OK: Zero lingering remote feature branches.")

    return clean


def main() -> int:
    """CLI entrypoint for agent Maker/Checker PR automation."""
    parser = argparse.ArgumentParser(description="KnowledgeReduce Core GitHub PR Helper")
    parser.add_argument(
        "--audit",
        "--check-lingering",
        dest="audit",
        action="store_true",
        help="Audit zero open PRs and zero lingering remote feature branches",
    )
    parser.add_argument("--list-prs", dest="list_prs", action="store_true", help="List open PRs")
    parser.add_argument("--create", dest="create", action="store_true", help="Create a PR")
    parser.add_argument("--title", dest="title", type=str, default="", help="PR title")
    parser.add_argument("--head", dest="head", type=str, default="", help="PR head branch")
    parser.add_argument("--base", dest="base", type=str, default="main", help="PR base branch")
    parser.add_argument("--body", dest="body", type=str, default="", help="PR body")
    parser.add_argument("--merge", dest="merge", action="store_true", help="Merge a PR")
    parser.add_argument("--close", dest="close", action="store_true", help="Close an open PR")
    parser.add_argument(
        "--comment", dest="comment", type=str, default="", help="Comment when closing"
    )
    parser.add_argument("--pr", dest="pr_num", type=int, default=0, help="PR number")
    parser.add_argument(
        "--commit-title", dest="commit_title", type=str, default="", help="Merge commit title"
    )
    parser.add_argument(
        "--merge-method",
        dest="merge_method",
        choices=["squash", "merge", "rebase"],
        default="squash",
        help="GitHub merge method",
    )
    parser.add_argument(
        "--delete-branch",
        dest="delete_branch",
        action="store_true",
        help="Delete remote branch after merge or close",
    )
    args = parser.parse_args()

    if args.audit:
        return 0 if audit_zero_lingering_prs() else 1
    if args.list_prs:
        print(json.dumps(list_open_prs(), indent=2))
        return 0
    if args.create:
        git_bin = shutil.which("git") or "git"
        head_branch = args.head or (
            subprocess.run(  # noqa: S603
                [git_bin, "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        pr = create_github_pr(
            head_branch=head_branch,
            title=args.title,
            body=args.body,
            base_branch=args.base,
        )
        print(f"Created PR #{pr['number']}: {pr['html_url']}")
        print(json.dumps(pr, indent=2))
        return 0
    if args.merge:
        head_branch = args.head
        if args.delete_branch and not head_branch:
            head_branch = str(get_github_pr_details(args.pr_num).get("head", {}).get("ref", ""))
        res = merge_github_pr(
            pr_number=args.pr_num,
            commit_title=args.commit_title,
            merge_method=args.merge_method,
        )
        print(f"Merged PR #{args.pr_num}")
        if (
            args.delete_branch
            and head_branch
            and head_branch != "main"
            and delete_github_branch(head_branch)
        ):
            print(f"Deleted remote branch {head_branch}")
        print(json.dumps(res, indent=2))
        return 0
    if args.close:
        pr_info = get_github_pr_details(args.pr_num)
        head_branch = args.head or str(pr_info.get("head", {}).get("ref", ""))
        res = close_github_pr(pr_number=args.pr_num, comment=args.comment or None)
        print(f"Closed PR #{args.pr_num}")
        if args.delete_branch and head_branch and head_branch != "main":
            delete_github_branch(head_branch)
        print(json.dumps({"number": res.get("number"), "state": res.get("state")}, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
