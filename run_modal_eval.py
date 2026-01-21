import modal
import subprocess
import os
import sys
from pathlib import Path

# Define the app name
app = modal.App("rlaif-v-eval")

# 1. Define the Environment
image = (
    modal.Image.micromamba(python_version="3.10")
    .apt_install("git", "wget", "build-essential")
    .run_commands(
        # Clone your fork of the repository
        "git clone https://github.com/Ahmed-5/RLAIF-V /root/RLAIF-V",
        
        # Install python dependencies
        "cd /root/RLAIF-V && pip install -e .",
        
        # Install spaCy model
        "wget https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3.tar.gz",
        "pip install en_core_web_trf-3.7.3.tar.gz",

        # Install PyTorch (CUDA 12.1)
        "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121",
        
        # Install Flash Attention (Optional but recommended)
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",

        # Install Transformers, Accelerate, and evaluation utilities
        # Added huggingface_hub for dataset download and openai for evaluation
        "pip install transformers==4.37.2 accelerate==0.28.0 huggingface_hub openai==0.28",
    )
)

# 2. Define Storage
results_volume = modal.Volume.from_name("rlaif-v-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("rlaif-v-dataset", create_if_missing=True)
logps_volume = modal.Volume.from_name("RLAIF-V-Dataset-logps", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# Mount this to store the downloaded evaluation data
# eval_data_volume = modal.Volume.from_name("rlaif-v-eval-dataset", create_if_missing=True)

# 3. Define the Evaluation Function
@app.function(
    image=image,
    gpu="A100-80GB:1",  # Using 1x A100 as requested
    timeout=86400,      # 24 hours
    volumes={
        "/root/RLAIF-V/.ckpt": results_volume, 
        "/root/RLAIF-V/RLAIF-V-Dataset": data_volume,
        "/root/RLAIF-V/RLAIF-V-Dataset_logps": logps_volume,
        # "/root/RLAIF-V/eval/data": eval_data_volume, # Mount eval data here
        "/root/.cache/huggingface": hf_cache
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"), 
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("openai-secret") # Ensure this secret exists for GPT evaluation
    ], 
)
def run_evaluation(checkpoints_path_str: str, save_dir: str):
    """
    checkpoints_path_str: Space-separated list of checkpoint paths (e.g. "/root/RLAIF-V/.ckpt/checkpoints/epoch_1 ...")
    save_dir: Directory to save results (e.g. "/root/RLAIF-V/eval_results")
    """
    import shutil
    from huggingface_hub import snapshot_download

    # --- Setup Environment Variables ---
    # Mimic the bash script exports
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:/root/RLAIF-V"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    base_path = Path("/root/RLAIF-V")
    os.chdir(base_path)
    print(f"Working directory: {os.getcwd()}")

    # git pull to ensure latest code
    subprocess.run(["git", "pull"], check=True)

    # --- 1. Download Evaluation Dataset ---
    print("========> Downloading HuggingFace Dataset <========")
    try:
        # Downloads contents of the repo to the mounted volume at /root/RLAIF-V/eval/data
        snapshot_download(
            repo_id="Ahmed5/mm_hal_jsonl",
            repo_type="dataset",
            local_dir="/root/RLAIF-V/eval/data",
            ignore_patterns=["*.git*"] # Clean download
        )
        print("Dataset downloaded successfully to /root/RLAIF-V/eval/data")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

    # --- 2. Prepare Paths ---
    q_file = base_path / "eval/data/mmhal-bench_with_image.jsonl"
    # template_file = base_path / "eval/data/mmhal-bench_answer_template.json" # (Unused variable in bash script, kept for reference)
    answer_file_name = "mmhal-bench_answer.jsonl"
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Parse checkpoints
    checkpoints = [c.strip() for c in checkpoints_path_str.split() if c.strip()]
    
    # Filter checkpoints (skip if output already exists, similar to bash script)
    to_process = []
    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}, skipping.")
            continue
            
        output_file = Path(save_dir) / answer_file_name
        if output_file.exists():
            print(f"Output {output_file} already exists. Skipping generation for this run (logic from bash script).")
            # The original script skips adding to list if file exists. 
            # Note: If you want to re-eval, delete the file manually or modify this logic.
        else:
            to_process.append(ckpt)

    print(f"Process these checkpoints: {to_process}")

    # --- 3. Run Generation (Muffin VQA) ---
    # The bash script used a loop with & and wait for parallel execution on 8 GPUs.
    # Since we are on 1 GPU, we run sequentially.
    
    for ckpt_path in to_process:
        output_file = Path(save_dir) / answer_file_name
        print(f"Generating answers for checkpoint: {ckpt_path}")
        
        cmd_gen = [
            "python", "./muffin/eval/muffin_vqa.py",
            "--model-path", ckpt_path,
            "--question-file", str(q_file),
            "--answers-file", str(output_file),
            "--temperature", "0",
            "--num_beam", "3"
        ]
        
        subprocess.run(cmd_gen, check=True)
    
    print("========> Done generating answers <========")

    # --- 4. Run Evaluation (GPT-4) ---
    print("========> Start evaluating answers <========")
    
    final_answer_file = Path(save_dir) / answer_file_name
    gpt_model = "gpt-4-1106-preview"
    
    if not final_answer_file.exists():
        print(f"No answer file found at {final_answer_file}. Skipping evaluation.")
        return

    # Retrieve API Key from Modal Secret
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return

    eval_output_file = str(final_answer_file) + ".mmhal_test_eval.json"
    log_file = str(final_answer_file) + ".eval_log.log"

    cmd_eval = [
        "python", "./eval/eval_gpt_mmhal.py",
        "--response", str(final_answer_file),
        "--evaluation", eval_output_file,
        "--gpt-model", gpt_model,
        "--api-key", api_key,
        "--is_jsonl"
    ]

    print(f"Running GPT Eval for {final_answer_file}...")
    with open(log_file, "w") as outfile:
        # i want to print both stdout and stderr to the log file and also in terminal
        subprocess.run(cmd_eval, stdout=outfile, stderr=outfile, check=True)

    # --- 5. Summarize Results ---
    print("Summarizing scores...")
    cmd_summary = ["python", "./eval/summarize_gpt_mmhal_review.py", save_dir]
    
    summary_file = Path(save_dir) / "mmhal_scores.txt"
    with open(summary_file, "w") as outfile:
        # print error to terminal not in file
        subprocess.run(cmd_summary, stdout=outfile, stderr=subprocess.STDOUT, check=True)

    print("Scores are:")
    print(summary_file.read_text())
    print("done")

# Entrypoint for local testing via `modal run file.py`
@app.local_entrypoint()
def main():
    # Example usage

    exps = [
        # "llava15_7b_DPO-llava15_rlaifv",
        # "llava15_7b_KTO-llava15_rlaifv_kto",
        # "llava15_7b_ORPO-llava15_rlaifv_orpo",
        "llava15_7b_SimPO-llava15_rlaifv_simpo",
    ]


    for exp in exps:
        chkpt = f"/root/RLAIF-V/.ckpt/{exp}/checkpoints" # Update this to your actual path in the volume
        save = f"/root/RLAIF-V/.ckpt/results_eval/{exp}_3"
        run_evaluation.remote(chkpt, save)
