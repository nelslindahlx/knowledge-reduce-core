"""Tests for BlockchainKnowledgeGraph verification, mining, and chain import/export."""
import json

import pytest

from knowledge_graph_pkg.blockchain import BlockchainKnowledgeGraph
from knowledge_graph_pkg.core import KnowledgeGraph, ReliabilityRating


def test_add_update_verify_and_mine_block():
    kg = KnowledgeGraph()
    bkg = BlockchainKnowledgeGraph(kg, difficulty=2)

    tx1 = bkg.add_fact(
        "f1",
        "Gravity attracts mass.",
        "Physics",
        ["gravity", "mass"],
        ReliabilityRating.LIKELY_TRUE,
        "src1",
    )
    assert len(tx1) == 64

    verify1 = bkg.verify_fact("f1")
    assert verify1["verified"] is True

    caller_updates = {"fact_statement": "Gravity curves spacetime.", "reliability_rating": ReliabilityRating.VERIFIED}
    bkg.update_fact("f1", **caller_updates)
    # Caller dict must remain untouched (Enum intact).
    assert caller_updates["reliability_rating"] == ReliabilityRating.VERIFIED

    verify2 = bkg.verify_fact("f1")
    assert verify2["verified"] is True
    assert verify2["transactions"] == 2
    assert len(bkg.get_fact_history("f1")) == 2

    block = bkg.mine_block()
    assert block is not None
    assert block["hash"].startswith("00")
    assert bkg.mine_block() is None
    assert bkg.verify_chain()["valid"] is True


def test_verify_fact_detects_tampering_and_failed_updates_do_not_pollute_ledger():
    kg = KnowledgeGraph()
    bkg = BlockchainKnowledgeGraph(kg)
    bkg.add_fact("f1", "Original fact.", "General", ["a"], ReliabilityRating.UNVERIFIED, "s1")

    with pytest.raises(ValueError, match="Invalid attribute"):
        bkg.update_fact("f1", nonexistent_field="bad")
    assert len(bkg.get_fact_history("f1")) == 1

    # Direct tampering outside blockchain is flagged by verify_fact.
    kg.graph.nodes["f1"]["fact_statement"] = "Tampered fact."
    result = bkg.verify_fact("f1")
    assert result["verified"] is False
    assert "fact_statement" in result["mismatches"]

    missing = bkg.verify_fact("does_not_exist")
    assert missing["verified"] is False


def test_verify_chain_and_import_rollback_on_tampered_chain(tmp_path):
    kg = KnowledgeGraph()
    bkg = BlockchainKnowledgeGraph(kg, difficulty=2)
    bkg.add_fact("f1", "Fact one.", "General", ["x"], ReliabilityRating.VERIFIED, "s1")
    bkg.mine_block()

    valid_file = tmp_path / "chain.json"
    bkg.export_chain(str(valid_file))

    kg2 = KnowledgeGraph()
    bkg2 = BlockchainKnowledgeGraph(kg2, difficulty=2)
    assert bkg2.import_chain(str(valid_file)) == 2
    assert kg2.get_fact("f1")["fact_statement"] == "Fact one."

    # Tampered difficulty / hash must fail import and preserve valid chain in memory.
    bad_chain = json.loads(valid_file.read_text())
    bad_chain[1]["hash"] = "ff" + bad_chain[1]["hash"][2:]
    bad_file = tmp_path / "bad_chain.json"
    bad_file.write_text(json.dumps(bad_chain))

    blocks_before = len(bkg2.chain)
    with pytest.raises(ValueError, match="Invalid blockchain"):
        bkg2.import_chain(str(bad_file))
    assert len(bkg2.chain) == blocks_before
    assert bkg2.verify_chain()["valid"] is True
