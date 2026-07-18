"""
Tests for batch automation (Session 12).

factory.batch_drop ingests many sources into a store in one pass, skipping
already-ingested (unchanged) sources, and returns a run report. This is
the building block for scheduled / watch-folder automation.
"""
import pytest

from knowledge_graph_pkg.store import KnowledgeStore
from knowledge_graph_pkg.factory import batch_drop, scan_folder


def _make_sources(tmp_path):
    (tmp_path / "a.txt").write_text("Robert Putnam wrote Bowling Alone.")
    (tmp_path / "b.txt").write_text("Marie Curie was born in Warsaw.")
    return [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]


def test_batch_drop_ingests_all(tmp_path):
    sources = _make_sources(tmp_path)
    store_dir = str(tmp_path / "store")
    report = batch_drop(sources, store_dir)
    assert report["dropped"] == 2
    assert report["skipped"] == 0
    assert report["total_facts"] >= 2
    assert KnowledgeStore(store_dir).stats()["total_drops"] == 2


def test_batch_drop_is_idempotent(tmp_path):
    sources = _make_sources(tmp_path)
    store_dir = str(tmp_path / "store")
    batch_drop(sources, store_dir)
    report = batch_drop(sources, store_dir)  # second pass
    assert report["dropped"] == 0
    assert report["skipped"] == 2
    assert KnowledgeStore(store_dir).stats()["total_drops"] == 2


def test_batch_drop_reports_per_source(tmp_path):
    sources = _make_sources(tmp_path)
    report = batch_drop(sources, str(tmp_path / "store"))
    assert len(report["items"]) == 2
    assert all("source" in it and "status" in it for it in report["items"])


def test_batch_drop_handles_missing_file(tmp_path):
    sources = [str(tmp_path / "ghost.txt")]
    report = batch_drop(sources, str(tmp_path / "store"))
    assert report["errors"] == 1
    assert report["items"][0]["status"] == "error"


def test_scan_folder_finds_supported_files(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.md").write_text("y")
    (tmp_path / "c.html").write_text("z")
    (tmp_path / "ignore.png").write_text("binary-ish")
    found = scan_folder(str(tmp_path))
    names = sorted(f.rsplit("/", 1)[-1] for f in found)
    assert names == ["a.txt", "b.md", "c.html"]
