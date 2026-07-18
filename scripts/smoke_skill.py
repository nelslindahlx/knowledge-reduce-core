#!/usr/bin/env python3
"""
Lightweight smoke test script to verify that the knowledge-reduce-core package
and the Hermes skill environment are properly installed and functional.
"""

import sys
import tempfile

def run_smoke_tests():
    print("[Smoke] Starting Hermes skill validation sequence...")
    
    # 1. Package Importability
    try:
        import knowledge_graph_pkg
        print(f"[Smoke] Successfully imported knowledge_graph_pkg (version: {knowledge_graph_pkg.__version__})")
    except ImportError as exc:
        print(f"[Smoke] ERROR: Failed to import knowledge_graph_pkg: {exc}", file=sys.stderr)
        return 1

    # 2. Version Alignment Check
    expected_version = "0.3.0"
    if knowledge_graph_pkg.__version__ != expected_version:
        print(f"[Smoke] ERROR: Package version mismatch! Expected {expected_version}, got {knowledge_graph_pkg.__version__}", file=sys.stderr)
        return 2

    # 3. Offline Heuristic Critique Fallback
    try:
        from knowledge_graph_pkg.critique import FactCritic
        critic = FactCritic(backend_name="none")
        
        # Test 3a: Valid non-pronoun SVO fact
        fact_ok = {
            "subject": "Mitosis",
            "predicate": "divides",
            "object": "cells",
            "statement": "Mitosis divides cells."
        }
        res_ok = critic.critique_fact(fact_ok)
        if not res_ok["is_factual"]:
            print(f"[Smoke] ERROR: Valid fact was incorrectly flagged: {res_ok}", file=sys.stderr)
            return 3
            
        # Test 3b: Invalid pronoun subject fact
        fact_pronoun = {
            "subject": "it",
            "predicate": "divides",
            "object": "cells",
            "statement": "it divides cells."
        }
        res_pronoun = critic.critique_fact(fact_pronoun)
        if res_pronoun["is_factual"]:
            print(f"[Smoke] ERROR: Pronoun subject fact was not rejected: {res_pronoun}", file=sys.stderr)
            return 4
            
        print("[Smoke] Offline Heuristic Critique Fallback validated successfully.")
    except Exception as exc:
        print(f"[Smoke] ERROR: Failed while testing FactCritic heuristics: {exc}", file=sys.stderr)
        return 5

    # 4. KnowledgeStore Audit Diagnostics
    try:
        from knowledge_graph_pkg.store import KnowledgeStore, Drop
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            
            # Write a mock drop
            drop = Drop(
                drop_id="smoke_drop_1",
                source="smoke.txt",
                source_hash="smoke_hash_1",
                facts=[fact_ok]
            )
            store.write_drop(drop)
            
            # Run audit summary
            report = store.audit_summary()
            if report["total_drops"] != 1 or report["total_facts"] != 1:
                print(f"[Smoke] ERROR: Store audit summary reports incorrect stats: {report}", file=sys.stderr)
                return 6
                
            print("[Smoke] KnowledgeStore Drop & Audit capabilities validated successfully.")
    except Exception as exc:
        print(f"[Smoke] ERROR: Failed while validating KnowledgeStore: {exc}", file=sys.stderr)
        return 7

    print("[Smoke] SUCCESS: All smoke tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(run_smoke_tests())
