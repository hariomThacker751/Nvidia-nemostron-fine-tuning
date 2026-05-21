import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
original_run = subprocess.run

def patched_run(*args, **kwargs):
    cmd = args[0] if args else kwargs.get("args", "")
    if "habana-torch-plugin" in str(cmd):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr=""
        )
    return original_run(*args, **kwargs)

subprocess.run = patched_run

import sys
import argparse
from pathlib import Path

# Insert project root to path for relative python imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.utils import logger
from src.data_prep import run_data_prep


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Nemotron Reasoning Challenge Pipeline CLI"
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["prep", "prepare", "train", "infer", "all"],
        help="Pipeline phase to trigger (prep: data preparation, train: SFT/GRPO, infer: test-inference, all: sequential run)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit to restrict record count for fast dry run checks."
    )
    
    args = parser.parse_args()
    
    # Standardize phase choices
    phase = args.phase.lower()
    if phase == "prepare":
        phase = "prep"
        
    limit = args.limit
    
    logger.info("=====================================================================")
    logger.info("🚀 ORCHESTRATION LAYER: STARTING PIPELINE CLI DRIVER")
    logger.info(f"   Target Phase: {phase.upper()}")
    logger.info(f"   Debug Limit:  {limit if limit else 'DISABLED'}")
    logger.info("=====================================================================")
    
    # ── RAW FILES RESOLUTIONS ──
    # Check if raw files exist. If not, check if they exist in project root and copy/link them.
    train_csv = config.DATA_RAW_DIR / "train.csv"
    test_csv = config.DATA_RAW_DIR / "test.csv"
    
    workspace_train = config.PROJECT_ROOT.parent / "train.csv"
    workspace_test = config.PROJECT_ROOT.parent / "test.csv"
    
    import shutil
    if not train_csv.exists() and workspace_train.exists():
        logger.info(f"Copying workspace train.csv to config path {train_csv}")
        shutil.copy(workspace_train, train_csv)
    if not test_csv.exists() and workspace_test.exists():
        logger.info(f"Copying workspace test.csv to config path {test_csv}")
        shutil.copy(workspace_test, test_csv)
        
    # If still missing and not mocked, alert
    if not train_csv.exists() and phase in ["prep", "all"]:
        logger.warning(f"Could not find train.csv under {train_csv.parent} or project workspace root.")
        # Create a dummy train.csv for quick unit testing/CI purposes
        logger.info("Creating micro dummy train.csv for execution.")
        dummy_train = pd_dummy_train()
        dummy_train.to_csv(train_csv, index=False)
        
    if not test_csv.exists() and phase in ["prep", "infer", "all"]:
        logger.info("Creating micro dummy test.csv for execution.")
        dummy_test = pd_dummy_test()
        dummy_test.to_csv(test_csv, index=False)

    try:
        if phase == "prep":
            run_data_prep(train_csv, test_csv, config.DATA_PROCESSED_DIR, limit=limit)
            
        elif phase == "train":
            from src.train import run_training_pipeline
            run_training_pipeline(limit=bool(limit))
            
        elif phase == "infer":
            from src.inference import run_inference_pipeline
            run_inference_pipeline(limit=limit)
            
        elif phase == "all":
            logger.info("=== STEP 1: Starting Data Preparation ===")
            run_data_prep(train_csv, test_csv, config.DATA_PROCESSED_DIR, limit=limit)
            
            logger.info("=== STEP 2: Starting Model Training ===")
            from src.train import run_training_pipeline
            run_training_pipeline(limit=bool(limit))
            
            logger.info("=== STEP 3: Starting Inference Submissions ===")
            from src.inference import run_inference_pipeline
            run_inference_pipeline(limit=limit)
            
        logger.info("=====================================================================")
        logger.info("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info("=====================================================================")
        
    except Exception as e:
        logger.error(f"❌ PIPELINE EXECUTION FAILED: {e}", exc_info=True)
        sys.exit(1)


def pd_dummy_train():
    import pandas as pd
    return pd.DataFrame([
        {
            "id": 1, 
            "prompt": "write the number 42 as Roman numeral", 
            "answer": "XLII"
        },
        {
            "id": 2, 
            "prompt": "Compute 15 XOR 9 where AND, OR, XOR are bitwise operations. Give the decimal result.", 
            "answer": "6"
        },
        {
            "id": 3, 
            "prompt": "Solve for x: 2x - 6 = 10", 
            "answer": "8"
        },
        {
            "id": 4, 
            "prompt": "Compute 5^3 mod 7.", 
            "answer": "6"
        },
        {
            "id": 5, 
            "prompt": "Given the sequence: 3, 7, 11, 15, ...\nWhat is the 5th term (1-indexed)?", 
            "answer": "19"
        }
    ])

def pd_dummy_test():
    import pandas as pd
    return pd.DataFrame([
        {
            "id": 101, 
            "prompt": "write the number 125 as Roman numeral"
        },
        {
            "id": 102, 
            "prompt": "Compute 12 AND 5 bitwise. Give decimal result."
        }
    ])


if __name__ == "__main__":
    main()
