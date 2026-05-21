import os
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─── PATHS CONFIGURATION ────────────────────────────────────────────────
# Resolve standard directory structures relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_PROCESSED_DIR  # Phase 1 SFT files output destination

# Create folders pro-actively
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Fine-tuning artifacts paths
ADAPTER_DIR = PROJECT_ROOT / "models" / "lora_adapter"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ─── HARDWARE PROFILE ──────────────────────────────────────────────────
# Hardware switches configured for high-performance Local Workstation (Xeon + RTX 4000)
# vs Cloud Kaggle GPU environment (Dual T4)
if HAS_TORCH:
    PROFILE = "workstation" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 18e9 else "kaggle"
else:
    PROFILE = "kaggle"

if PROFILE == "workstation":
    # High-performance local workstation
    GPU_MEM_HEADROOM_GB = 18.0  # RTX 4000 has 20GB VRAM
    device_map = "auto"
    attn_implementation = "sdpa"  # Scaled Dot Product Attention (fast and native on PyTorch 2.x)
    SFT_BATCH_SIZE = 2
    SFT_GRAD_ACCUM = 4
    optim_algo = "paged_adamw_8bit"
else:
    # Tight Kaggle dual T4 constraints
    GPU_MEM_HEADROOM_GB = 13.0  # Dual T4s have 15GB each
    device_map = "auto"
    attn_implementation = "eager"  # Safer fallback to prevent OOM
    SFT_BATCH_SIZE = 1
    SFT_GRAD_ACCUM = 8
    optim_algo = "paged_adamw_8bit"


# ─── MODEL & LORA SPECIFICATION ───────────────────────────────────────
MODEL_HF_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# Local path fallback if model has been cached or downloaded to Kaggle dataset
LOCAL_MODEL_PATH = "/kaggle/input/nvidia-nemotron-nano-30b/"
MODEL_ID = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else MODEL_HF_ID

# LoRA Constraints (rank must be <= 32 per competition rules)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]


# ─── PHASE 1: DATA PREPARATION & SYNTHESIS ───────────────────────────
COT_CACHE_FILE = DATA_PROCESSED_DIR / "cot_cache.jsonl"
TRAIN_SFT_OUTPUT = DATA_PROCESSED_DIR / "train_sft.jsonl"
SYNTHETIC_PER_TYPE = 200  # Number of synthetic puzzles per solver category


# ─── PHASE 2: SFT + GRPO TRAINING ────────────────────────────────────
SFT_MAX_SEQ_LENGTH = 2048
SFT_EPOCHS = 2
SFT_LR = 2e-4
SFT_WARMUP_RATIO = 0.05

RUN_GRPO = True
GRPO_STEPS = 100
GRPO_LR = 5e-5
GRPO_BATCH_SIZE = 1
GRPO_GRAD_ACCUM = 4
GRPO_N_ROLLOUTS = 4  # Number of generations per prompt for policy optimization


# ─── PHASE 3: ADAPTIVE COMPUTE & INFERENCE ───────────────────────────
N_VOTES_DEFAULT = 5
MAX_NEW_TOKENS = 1500
INFERENCE_TEMP = 0.7

CATEGORY_VOTES = {
    "bit_manipulation": 3,
    "algebraic_equation": 5,
    "modular_arithmetic": 5,
    "sequence_pattern": 3,
    "combinatorics": 7,  # Hard category gets more compute budget
    "number_theory": 5,
    "other": 3,
}

RETRY_LOW_CONF = True
LOW_CONF_THRESHOLD = 0.5
RETRY_N_VOTES = 7
RETRY_TEMP = 0.9
