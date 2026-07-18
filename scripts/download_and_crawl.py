import os
import sys
import shutil
import urllib.request
import subprocess

def main():
    print("=== STARTING MODEL DOWNLOAD & CRAWL TEST ===")
    
    # 1. Download Qwen2.5-0.5B-Instruct-GGUF from Hugging Face
    model_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    model_path = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    if not os.path.exists(model_path):
        print(f"\n[1/3] Downloading Qwen2.5-0.5B-Instruct GGUF model (390MB) from Hugging Face...")
        try:
            def report_progress(block_num, block_size, total_size):
                percent = int(block_num * block_size * 100 / total_size)
                # Cap percentage display to 100%
                percent = min(100, percent)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()
                
            urllib.request.urlretrieve(model_url, model_path, reporthook=report_progress)
            print("\n  Download complete!")
        except Exception as exc:
            print(f"\nError downloading model: {exc}")
            return
    else:
        print(f"\n[1/3] Model file '{model_path}' already exists. Skipping download.")
        
    # 2. Run the crawler using the llama-cpp GGUF backend
    seed_topic = "Mitochondria"
    store_dir = "qwen_crawled_store"
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
        
    print(f"\n[2/3] Recursively crawling weights of Qwen2.5 for seed '{seed_topic}'...")
    print(f"  Command: knowledgereduce crawl --seed \"{seed_topic}\" --backend llama-cpp --model {model_path} --store {store_dir} --max-depth 1")
    
    cmd = [
        ".venv/bin/python3",
        "-m", "knowledge_graph_pkg.cli",
        "crawl",
        "--seed", seed_topic,
        "--backend", "llama-cpp",
        "--model", model_path,
        "--store", store_dir,
        "--max-depth", "1",
        "--concepts-per-level", "2"
    ]
    
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(res.stdout)
    except subprocess.CalledProcessError as exc:
        print(f"Crawler failed with exit code {exc.returncode}")
        print("Error output:")
        print(exc.stderr)
        return
        
    # 3. Check the store for extracted facts
    print("\n[3/3] Verifying store contents...")
    if os.path.exists(store_dir):
        files = os.listdir(store_dir)
        print(f"  Extracted drops in '{store_dir}': {files}")
    else:
        print("  Error: Crawled store directory was not created.")
        
    print("\n=== MODEL DOWNLOAD & CRAWL TEST COMPLETE ===")

if __name__ == "__main__":
    main()
