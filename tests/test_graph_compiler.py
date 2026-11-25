import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import pytest

from knowledge_graph_pkg.entity_resolution import resolve_and_merge_entities
from knowledge_graph_pkg.graph_compiler import (
    compile_contradiction_instructions,
    compile_subgraph_instructions,
    save_compiled_instructions,
)
from knowledge_graph_pkg.graph_store_factory import get_graph_store

KUZU_AVAILABLE = importlib.util.find_spec("kuzu") is not None


class TestGraphCompilerPure(unittest.TestCase):

    def test_compile_subgraph_excludes_self_cycles_and_missing_reliability(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "stmt1": "A causes B",
                "subj1": "A",
                "obj1": "B",
                "stmt2": "B causes a",
                "subj2": "B",
                "obj2": "a",
                "rel1": "VERIFIED",
                "rel2": "VERIFIED",
            },
            {
                "stmt1": "X causes Y",
                "subj1": "X",
                "obj1": "Y",
                "stmt2": "Y causes Z",
                "subj2": "Y",
                "obj2": "Z",
                "rel1": None,
                "rel2": "VERIFIED",
            },
            {
                "stmt1": "Sun warms Earth",
                "subj1": "Sun",
                "obj1": "Earth",
                "stmt2": "Earth supports Life",
                "subj2": "Earth",
                "obj2": "Life",
                "rel1": "verified",
                "rel2": "LIKELY_TRUE",
            },
        ]
        instructions = compile_subgraph_instructions(mock_store, auto_link=False)
        mock_store.auto_link_relations.assert_not_called()
        self.assertEqual(len(instructions), 3)
        self.assertIn("Sun to Life", instructions[0]["instruction"])

    def test_compile_subgraph_max_chains_zero(self):
        mock_store = MagicMock()
        self.assertEqual(compile_subgraph_instructions(mock_store, max_chains=0), [])
        mock_store.query.assert_not_called()

    def test_compile_contradictions_ignores_non_contradictory_and_case_ranks(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "stmt1": "Mitochondria produce ATP.",
                "pred1": "produce",
                "rel1": "VERIFIED",
                "agree1": 3,
                "stmt2": "Mitochondria store ATP.",
                "pred2": "store",
                "rel2": "VERIFIED",
                "agree2": 3,
            },
            {
                "stmt1": "Mitochondria produce ATP.",
                "pred1": "produce",
                "rel1": "likely_true",
                "agree1": 2,
                "stmt2": "Mitochondria do not produce ATP.",
                "pred2": "do not produce",
                "rel2": "verified",
                "agree2": 3,
            },
        ]
        instructions = compile_contradiction_instructions(mock_store)
        self.assertEqual(len(instructions), 1)
        self.assertIn("Claim 2", instructions[0]["response"])
        self.assertIn("VERIFIED", instructions[0]["response"])

    def test_save_compiled_instructions_atomic_nested(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "nested", "dir", "train.jsonl")
            save_compiled_instructions([{"instruction": "i", "response": "r"}], out_path)
            self.assertTrue(os.path.isfile(out_path))
            self.assertFalse(os.path.exists(f"{out_path}.tmp"))
            with open(out_path, "r", encoding="utf-8") as fh:
                row = json.loads(fh.readline())
            self.assertEqual(row["instruction"], "i")


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="kuzu not installed")
class TestGraphCompilerAndResolution(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_graph_compiler_")
        self.store = get_graph_store(os.path.join(self.temp_dir, "kdb"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp_dir)

    def test_entity_resolution_and_deduplication(self):
        facts = [
            {
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "fact_statement": "Mitochondria produce ATP.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
            {
                "subject": "Adenosine Triphosphate (ATP)",
                "predicate": "fuels",
                "object": "cellular activities",
                "fact_statement": "Adenosine Triphosphate (ATP) fuels cellular activities.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
        ]
        self.store.ingest_facts(facts)
        self.assertEqual(self.store.count(), 2)

        res = resolve_and_merge_entities(self.store, threshold=0.85)
        self.assertEqual(res["resolved_clusters"], 1)
        self.assertIn("ATP", res["clusters"][0])
        self.assertIn("Adenosine Triphosphate (ATP)", res["clusters"][0])

        updated_facts = self.store.query(
            "MATCH (f:Fact) RETURN f.subject AS subject, f.object AS object"
        )
        subjects = [f["subject"] for f in updated_facts]
        objects = [f["object"] for f in updated_facts]
        self.assertIn("Adenosine Triphosphate (ATP)", subjects)
        self.assertIn("Adenosine Triphosphate (ATP)", objects)

    def test_subgraph_instruction_compiler(self):
        facts = [
            {
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "fact_statement": "Mitochondria produce ATP.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
            {
                "subject": "ATP",
                "predicate": "fuels",
                "object": "cells",
                "fact_statement": "ATP fuels cells.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
        ]
        self.store.ingest_facts(facts)
        self.store.auto_link_relations()

        instructions = compile_subgraph_instructions(self.store)
        self.assertEqual(len(instructions), 3)

        self.assertEqual(
            instructions[0]["instruction"],
            "Trace the relationship connection starting from Mitochondria to cells.",
        )
        self.assertIn("intermediate concept ATP", instructions[0]["response"])
        self.assertEqual(
            instructions[1]["instruction"],
            "Explain the sequence of connections linking Mitochondria to cells.",
        )
        self.assertEqual(
            instructions[2]["instruction"],
            "If Mitochondria produce ATP. and we know that ATP connects to cells, what is the second step in the path?",
        )

        temp_out = os.path.join(self.temp_dir, "sft_train.jsonl")
        save_compiled_instructions(instructions, temp_out)
        self.assertTrue(os.path.isfile(temp_out))

        with open(temp_out, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        self.assertEqual(len(lines), 3)
        data = json.loads(lines[0])
        self.assertIn("instruction", data)
        self.assertIn("response", data)

    def test_unverified_filtering_in_subgraph_compilation(self):
        facts = [
            {
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "fact_statement": "Mitochondria produce ATP.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
            {
                "subject": "ATP",
                "predicate": "fuels",
                "object": "cells",
                "fact_statement": "ATP fuels cells.",
                "domain": "biochemistry",
                "reliability_rating": "UNVERIFIED",
                "cross_model_agreement": 1,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
        ]
        self.store.ingest_facts(facts)
        self.store.auto_link_relations()

        instructions = compile_subgraph_instructions(self.store)
        self.assertEqual(len(instructions), 0)

    def test_contradiction_instruction_compiler(self):
        facts = [
            {
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "fact_statement": "Mitochondria produce ATP.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
            {
                "subject": "Mitochondria",
                "predicate": "stores",
                "object": "ATP",
                "fact_statement": "Mitochondria stores ATP.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
            {
                "subject": "Mitochondria",
                "predicate": "do not produce",
                "object": "ATP",
                "fact_statement": "Mitochondria do not produce ATP.",
                "domain": "biochemistry",
                "reliability_rating": "UNVERIFIED",
                "cross_model_agreement": 1,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
        ]
        self.store.ingest_facts(facts)

        instructions = compile_contradiction_instructions(self.store)
        self.assertEqual(len(instructions), 1)
        self.assertIn(
            "Evaluate the following two conflicting claims",
            instructions[0]["instruction"],
        )
        self.assertIn("produce ATP", instructions[0]["instruction"])
        self.assertIn("do not produce ATP", instructions[0]["instruction"])
        self.assertIn(
            "Claim 1 ('Mitochondria produce ATP.') is more reliable",
            instructions[0]["response"],
        )
