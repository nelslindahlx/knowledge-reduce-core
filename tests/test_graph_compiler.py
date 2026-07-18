import os
import unittest
import tempfile
import shutil
import pytest
from knowledge_graph_pkg.graph_store_factory import get_graph_store
from knowledge_graph_pkg.entity_resolution import resolve_and_merge_entities
from knowledge_graph_pkg.graph_compiler import compile_subgraph_instructions, save_compiled_instructions

pytest.importorskip("kuzu")

class TestGraphCompilerAndResolution(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_graph_compiler_")
        self.store = get_graph_store(os.path.join(self.temp_dir, "kdb"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp_dir)

    def test_entity_resolution_and_deduplication(self):
        # 1. Ingest facts with synonyms
        # Note: "ATP" is a clean substring of "Adenosine Triphosphate (ATP)" -> similarity threshold will merge them
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
                "source_models": "gpt-4"
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
                "source_models": "gpt-4"
            }
        ]
        self.store.ingest_facts(facts)
        self.assertEqual(self.store.count(), 2)

        # 2. Run entity resolution
        res = resolve_and_merge_entities(self.store, threshold=0.85)
        
        # Verify that synonym "ATP" and "Adenosine Triphosphate (ATP)" were matched/clustered
        self.assertEqual(res["resolved_clusters"], 1)
        self.assertIn("ATP", res["clusters"][0])
        self.assertIn("Adenosine Triphosphate (ATP)", res["clusters"][0])

        # 3. Retrieve the updated facts
        updated_facts = self.store.query("MATCH (f:Fact) RETURN f.subject AS subject, f.object AS object")
        subjects = [f["subject"] for f in updated_facts]
        objects = [f["object"] for f in updated_facts]
        
        # Verify they now share the exact same entity name
        self.assertIn("Adenosine Triphosphate (ATP)", subjects)
        self.assertIn("Adenosine Triphosphate (ATP)", objects)

    def test_subgraph_instruction_compiler(self):
        # 1. Ingest overlapping facts
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
                "source_models": "gpt-4"
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
                "source_models": "gpt-4"
            }
        ]
        self.store.ingest_facts(facts)
        
        # Auto-linking connects the nodes (matching object "ATP" to subject "ATP")
        self.store.auto_link_relations()
        
        # 2. Compile instructions
        instructions = compile_subgraph_instructions(self.store)
        self.assertEqual(len(instructions), 3) # 3 templates generated per chain
        
        # Verify templates
        self.assertEqual(instructions[0]["instruction"], "Trace the relationship connection starting from Mitochondria to cells.")
        self.assertIn("intermediate concept ATP", instructions[0]["response"])
        self.assertEqual(instructions[1]["instruction"], "Explain the sequence of connections linking Mitochondria to cells.")
        self.assertEqual(instructions[2]["instruction"], "If Mitochondria produce ATP. and we know that ATP connects to cells, what is the second step in the path?")
        
        # 3. Test saving instructions
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
        # 1. Ingest overlapping facts where one is UNVERIFIED
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
                "source_models": "gpt-4"
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
                "source_models": "gpt-4"
            }
        ]
        self.store.ingest_facts(facts)
        self.store.auto_link_relations()
        
        # 2. Compile instructions
        instructions = compile_subgraph_instructions(self.store)
        
        # Since one of the path nodes is UNVERIFIED, zero instructions should be compiled
        self.assertEqual(len(instructions), 0)

    def test_contradiction_instruction_compiler(self):
        from knowledge_graph_pkg.graph_compiler import compile_contradiction_instructions
        
        # Ingest two contradicting facts
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
                "source_models": "gpt-4"
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
                "source_models": "gpt-4"
            }
        ]
        self.store.ingest_facts(facts)
        
        # Compile contradiction instructions
        instructions = compile_contradiction_instructions(self.store)
        self.assertEqual(len(instructions), 1)
        
        # Verify the instruction is synthesized and prefers the VERIFIED one
        self.assertIn("Evaluate the following two conflicting claims", instructions[0]["instruction"])
        self.assertIn("produce ATP", instructions[0]["instruction"])
        self.assertIn("do not produce ATP", instructions[0]["instruction"])
        
        # Since VERIFIED > UNVERIFIED, response should choose the first one
        self.assertIn("Claim 1 ('Mitochondria produce ATP.') is more reliable", instructions[0]["response"])

