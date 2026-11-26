"""Meta-tests enforcing repository governance, zero-GitHub-minutes CI, and code invariants."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

import knowledge_graph_pkg
from knowledge_graph_pkg.critique import FactCritic
from knowledge_graph_pkg.graph_tool import TOOL_SCHEMAS, GraphTools
from scripts import github_pr_helper
from scripts.hooks.pre_pr_safety_gate import evaluate_command

REPO_ROOT = Path(__file__).resolve().parents[2]
_MAX_AGENTS_BYTES = 24_000
_BACKTICK_PATH_RE = re.compile(
    r"`((?:knowledge_graph_pkg|tests|scripts|docs|deploy|\.github|\.agents)/[A-Za-z0-9_./-]+\.[A-Za-z0-9]+|"
    r"Makefile|pyproject\.toml|AGENTS\.md|CHANGELOG\.md|CONTRIBUTING\.md|lessons_learned\.md|CITATION\.cff|LICENSE)`"
)


def _read_pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib

        data = tomllib.loads(text)
        return str(data["project"]["version"])
    except ModuleNotFoundError:
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
        assert match is not None, "Could not locate [project].version in pyproject.toml"
        return match.group(1)


def test_agents_md_parity_and_size_budget() -> None:
    root_agents = REPO_ROOT / "AGENTS.md"
    scoped_agents = REPO_ROOT / ".agents" / "AGENTS.md"
    assert root_agents.is_file()
    assert scoped_agents.is_file()
    root_bytes = root_agents.read_bytes()
    assert root_bytes == scoped_agents.read_bytes(), (
        "AGENTS.md and .agents/AGENTS.md must remain byte-identical."
    )
    assert len(root_bytes) < _MAX_AGENTS_BYTES


def test_semver_manifest_parity() -> None:
    pyproject_ver = _read_pyproject_version()
    assert knowledge_graph_pkg.__version__ == pyproject_ver

    init_text = (REPO_ROOT / "knowledge_graph_pkg" / "__init__.py").read_text(encoding="utf-8")
    fallback_versions = re.findall(r'__version__\s*=\s*"([^"]+)"', init_text)
    assert fallback_versions, "Expected fallback __version__ literals in __init__.py"
    assert all(v == pyproject_ver for v in fallback_versions), (
        f"Fallback __version__ mismatch in __init__.py: {fallback_versions!r} vs {pyproject_ver!r}"
    )

    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert str(citation["version"]) == pyproject_ver

    skill_text = (
        REPO_ROOT / ".agents" / "skills" / "knowledge-reduce-core" / "SKILL.md"
    ).read_text(encoding="utf-8")
    parts = skill_text.split("---")
    assert len(parts) >= 3
    skill_meta = yaml.safe_load(parts[1])
    assert str(skill_meta["version"]) == pyproject_ver

    smoke_text = (REPO_ROOT / "scripts" / "smoke_skill.py").read_text(encoding="utf-8")
    smoke_match = re.search(r'expected_version\s*=\s*"([^"]+)"', smoke_text)
    assert smoke_match is not None
    assert smoke_match.group(1) == pyproject_ver


def test_mit_license_complete() -> None:
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.is_file()
    text = license_path.read_text(encoding="utf-8").strip()
    assert len(text.encode("utf-8")) >= 900
    assert "Permission is hereby granted, free of charge" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in text
    assert not text.endswith("...")


def test_github_workflows_are_workflow_dispatch_only() -> None:
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    workflow_files = sorted(workflows_dir.glob("*.yml"))
    workflow_names = {p.name for p in workflow_files}
    expected_workflows = {"ci.yml", "pr-governance.yml", "release.yml", "skill-validation.yml"}
    assert expected_workflows.issubset(workflow_names), (
        f"Missing expected workflows: {expected_workflows - workflow_names!r}"
    )
    for wf_path in workflow_files:
        parsed = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        # PyYAML may parse bare `on:` as boolean True key in YAML 1.1
        triggers = parsed.get("on") if "on" in parsed else parsed.get(True)
        if isinstance(triggers, str):
            trigger_keys = {triggers}
        elif isinstance(triggers, (list, dict)):
            trigger_keys = {str(k) for k in triggers}
        else:
            trigger_keys = set()
        assert trigger_keys == {"workflow_dispatch"}, (
            f"{wf_path.name} must trigger ONLY on workflow_dispatch to save GitHub Actions minutes "
            f"(found {trigger_keys!r})."
        )


def test_no_tracked_pyc_or_local_paths() -> None:
    git_bin = shutil.which("git")
    assert git_bin is not None
    pyc_out = subprocess.check_output(  # noqa: S603
        [git_bin, "ls-files", "*.pyc"],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()
    assert pyc_out == "", f"Tracked .pyc files found:\n{pyc_out}"

    forbidden_needles = (
        "/" + "Users/",
        "file:///" + "Users/",
        "C:" + "\\Users\\",
    )
    checked_paths: list[Path] = [
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "Makefile",
        REPO_ROOT / "README.md",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "lessons_learned.md",
        REPO_ROOT / "ARCHITECTURE.md",
        REPO_ROOT / "MODELREDUCE_MASTER_PLAN.md",
        REPO_ROOT / "MODELREDUCE_SESSIONS.md",
        REPO_ROOT / "improvement_master_plan_phase_2.md",
        REPO_ROOT / "task.md",
        REPO_ROOT / "walkthrough.md",
        *sorted((REPO_ROOT / ".github").rglob("*")),
        *sorted((REPO_ROOT / ".agents").rglob("*")),
        *sorted((REPO_ROOT / "deploy").rglob("*")),
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
        *sorted((REPO_ROOT / "scripts").rglob("*.py")),
        *sorted((REPO_ROOT / "tests" / "meta").glob("*.py")),
    ]
    for path in checked_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden_needles:
            assert needle not in text, f"Machine-local path {needle!r} leaked into {path}"


def test_lessons_learned_relevance_and_integrity() -> None:
    lessons_path = REPO_ROOT / "lessons_learned.md"
    assert lessons_path.is_file(), "lessons_learned.md must exist at repository root."
    content = lessons_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 200

    headings = re.findall(r"^##\s+(?:Lesson\s+)?\d+[.:]", content, flags=re.MULTILINE)
    assert len(headings) >= 5, (
        f"Expected >=5 numbered lessons in lessons_learned.md (got {len(headings)})."
    )

    referenced_paths = set(_BACKTICK_PATH_RE.findall(content))
    assert len(referenced_paths) >= 8, (
        "lessons_learned.md should reference concrete repository files."
    )
    for rel_path in sorted(referenced_paths):
        target = REPO_ROOT / rel_path
        assert target.exists(), f"lessons_learned.md references nonexistent repo path: {rel_path}"

    required_domain_anchors = (
        "knowledge_graph_pkg",
        "KùzuDB",
        "SVOExtractor",
        "ModelReduce",
        "Watcher",
        "Graph-RAG",
    )
    for anchor in required_domain_anchors:
        assert anchor.lower() in content.lower(), (
            f"lessons_learned.md is missing domain context {anchor!r}."
        )

    foreign_needles = (
        "two-adic" + "-zipper",
        "two_adic" + "_zipper",
        "DuckDB" + "ParquetStore",
        "AsyncWeb" + "Crawler",
        "DeterministicNLP" + "Mapper",
        "RSA" + "-8192",
        "Coppersmith" + "Solver",
    )
    governance_files: list[Path] = [
        lessons_path,
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / ".agents" / "AGENTS.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "Makefile",
        REPO_ROOT / "CHANGELOG.md",
        *sorted((REPO_ROOT / ".agents" / "rules").glob("*.md")),
        *sorted((REPO_ROOT / "docs" / "plans").glob("*.md")),
        REPO_ROOT / "scripts" / "github_pr_helper.py",
        *sorted((REPO_ROOT / "scripts" / "hooks").glob("*.py")),
    ]
    for path in governance_files:
        if not path.is_file():
            continue
        file_text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in foreign_needles:
            assert needle.lower() not in file_text.lower(), (
                f"Foreign template term {needle!r} found in {path}; "
                "keep governance rules repository-specific."
            )


def test_pre_pr_safety_gate_blocks_direct_main_push_and_allows_feature_push() -> None:
    deny_main = evaluate_command(
        "git push origin main",
        REPO_ROOT,
        run_meta_suite=False,
    )
    assert deny_main["decision"] == "deny"

    deny_head = evaluate_command(
        "git push --force origin HEAD:main",
        REPO_ROOT,
        run_meta_suite=False,
    )
    assert deny_head["decision"] == "deny"

    allow_feat = evaluate_command(
        "git push -u origin chore/repo-hardening",
        REPO_ROOT,
        run_meta_suite=False,
    )
    assert allow_feat["decision"] == "allow"


def test_github_pr_helper_targets_this_repo() -> None:
    assert github_pr_helper.REPO_OWNER == "nelslindahlx"
    assert github_pr_helper.REPO_NAME == "knowledge-reduce-core"


def test_runtime_domain_invariants_and_import_safety() -> None:
    tools = GraphTools(store=None)
    for keyword in ("CREATE", "MERGE", "SET", "DELETE", "DROP"):
        try:
            tools.graph_query(f"{keyword} (n:Fact)")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for mutating Cypher keyword {keyword!r}")

    for schema in TOOL_SCHEMAS:
        method = getattr(tools, schema["name"], None)
        assert callable(method), f"Missing callable GraphTools method for schema {schema['name']!r}"

    critic = FactCritic(backend_name="none")
    assert (
        critic.critique_fact(
            {
                "subject": "it",
                "predicate": "divides",
                "object": "cells",
                "statement": "it divides cells.",
            }
        )["is_factual"]
        is False
    )
    assert (
        critic.critique_fact(
            {
                "subject": "Mitosis",
                "predicate": "divides",
                "object": "cells",
                "statement": "Mitosis divides cells.",
            }
        )["is_factual"]
        is True
    )

    proc = subprocess.run(
        [sys.executable, "-W", "error::Warning", "-c", "import knowledge_graph_pkg"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_coverage_gate_parity() -> None:
    gate_files = (
        REPO_ROOT / "Makefile",
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / ".agents" / "AGENTS.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / ".github" / "pull_request_template.md",
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
    )
    for path in gate_files:
        text = path.read_text(encoding="utf-8")
        assert "--cov-fail-under=75" in text, f"Expected 75% coverage gate in {path.name}"
        assert "--cov-fail-under=58" not in text, f"Stale 58% coverage gate in {path.name}"
        assert "--cov-fail-under=60" not in text, f"Stale 60% coverage gate in {path.name}"


def test_golden_benchmark_artifacts_and_makefile_target() -> None:
    required_paths = (
        REPO_ROOT / "benchmarks" / "civichonors_book_benchmark.py",
        REPO_ROOT / "benchmarks" / "multisource_factory_benchmark.py",
        REPO_ROOT / "benchmarks" / "datasets" / "civichonors_book" / "gold_annotations.json",
        REPO_ROOT / "benchmarks" / "datasets" / "civichonors_book" / "baseline_v0_4_0.json",
        REPO_ROOT / "tests" / "golden" / "test_civichonors_book_golden.py",
        REPO_ROOT / "tests" / "golden" / "test_multisource_factory_golden.py",
    )
    for path in required_paths:
        assert path.is_file(), f"Missing golden benchmark artifact: {path}"
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "golden:" in makefile_text, "Makefile must define a 'golden:' target"
