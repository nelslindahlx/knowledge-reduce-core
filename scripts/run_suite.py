#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def get_test_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(project_root, "tests")
    
    all_files = []
    for root, _, filenames in os.walk(test_dir):
        for f in filenames:
            if f.startswith("test_") and f.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, f), project_root)
                all_files.append((f, rel_path))
    return all_files

def classify_stages(all_files):
    stage1 = []  # Fast Unit / Core Graph & Schemas
    stage2 = []  # Semantic & Ingestion/Extraction
    stage3 = []  # Lifecycle/Store, Database & Graph Integration
    stage4 = []  # Model Probing, Embeddings, Cross-Model & Evaluation

    stage1_files = {
        "test_core.py", "test_schemas.py", "test_store.py", "test_factory.py",
        "test_qa.py", "test_quality.py", "test_distillation_io.py"
    }
    
    stage2_files = {
        "test_ingest.py", "test_extraction.py", "test_extractor_base.py",
        "test_coref.py", "test_semantic.py", "test_backends.py"
    }
    
    stage3_files = {
        "test_catalog.py", "test_kuzu_store.py", "test_graph_tool.py",
        "test_lifecycle.py", "test_training_prep.py", "test_export.py",
        "test_mcp_server.py", "test_reasoning.py", "test_watcher.py",
        "test_rag.py", "test_consensus.py", "test_train_wrapper.py",
        "test_crawler.py", "test_compile_sft.py"
    }
    
    stage4_files = {
        "test_model_probe.py", "test_model_drop.py", "test_model_distill.py",
        "test_model_eval.py", "test_cross_model.py", "test_embeddings.py",
        "test_cli.py", "test_graveyard.py", "test_critique.py"
    }

    for filename, rel_path in all_files:
        if filename in stage1_files:
            stage1.append(rel_path)
        elif filename in stage2_files:
            stage2.append(rel_path)
        elif filename in stage3_files:
            stage3.append(rel_path)
        elif filename in stage4_files:
            stage4.append(rel_path)
        else:
            # Fallback to Stage 2 for newly added files
            stage2.append(rel_path)

    return {
        "1": sorted(stage1),
        "2": sorted(stage2),
        "3": sorted(stage3),
        "4": sorted(stage4)
    }

def main():
    parser = argparse.ArgumentParser(description="KnowledgeReduce Test Suite Stage Manager")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "4", "all"],
        default="1",
        help="Select execution stage: 1 (Fast Unit/Core), 2 (Semantic/Extract), 3 (DB/Lifecycle/Integration), 4 (LLM Probes/Eval), all (Run everything)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display pytest verbose logging output"
    )

    args = parser.parse_args()
    all_files = get_test_files()
    stages = classify_stages(all_files)

    files_to_run = []
    if args.stage == "all":
        for s in ["1", "2", "3", "4"]:
            files_to_run.extend(stages[s])
    else:
        files_to_run = stages[args.stage]

    if not files_to_run:
        print(f"No test files found for stage {args.stage}!")
        sys.exit(0)

    print("\n" + "=" * 60)
    print(f"🏃 RUNNING TEST SUITE - STAGE {args.stage.upper()}")
    print(f"   Target Files: {len(files_to_run)} files matched")
    print("=" * 60 + "\n")

    cmd = [sys.executable, "-m", "pytest"]
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
        
    cmd.extend(files_to_run)

    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
