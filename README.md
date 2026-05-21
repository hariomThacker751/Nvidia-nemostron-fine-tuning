# 🏆 NVIDIA Nemotron Reasoning Pipeline
> A production-grade, modular, and containerized machine learning pipeline designed to fine-tune and evaluate the **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** hybrid model. Pre-configured for high-end local workstations (such as multi-Xeon CPUs with 224GB RAM and RTX 4000 20GB VRAM) and cloud Kaggle dual-T4 execution contexts.

---

## 🔬 Pipeline Architecture & Features

This pipeline restructures logical reasoning tasks into a reproducible, modular deep learning repository.

```mermaid
graph TD
    A[data/raw/train.csv] --> B[Phase 1: EDA & Prep]
    B --> C[Tag Puzzle Category]
    B --> D[ deterministic Programmatic Solvers]
    B --> E[Synthetic Data Generation]
    C & D & E --> F[data/processed/train_sft.jsonl]
    
    F --> G[Phase 2: Fine-Tuning Stage]
    G --> H[Stage A: SFT QLoRA 4-bit]
    H --> I[Stage B: GRPO RL Alignment]
    I --> J[models/lora_adapter]
    
    J --> K[Phase 3: Adaptive Consistency Inference]
    K --> L[Category Router & Specialized System Cards]
    K --> M[SC@N Majority Voting with Adaptive Compute]
    K --> N[Retry Routines for Uncertainty]
    L & M & N --> O[submission.csv]
```

### Key Technical Innovations
*   **Tiered Deterministic Solvers:** Includes programmatic solvers for 6 puzzle styles (Roman, Gravity, Bitwise logic, Substitution cryptarithms, Units, Ciphers).
*   **Synthetic Augmentation:** Automatically generates up to 5x data expansions dynamically.
*   **Stage-B GRPO Training:** Aligns models using outcome-based reinforcement learning, rewarding logical consistency, `<think>` boundaries, and exact mathematical matches.
*   **Adaptive Self-Consistency:** Distributes VRAM memory maps dynamically and directs high voting budgets (e.g. N=7) strictly towards complex categories like combinatorics.

---

## 📂 Repository Directory Map

```text
nemotron-reasoning-pipeline/
├── .github/workflows/
│   └── python-app.yml       # Standard CI: auto-linting and pytest checks upon push
├── data/
│   ├── raw/                     # Raw train.csv and test.csv competition inputs
│   └── processed/               # Tagged records, cache maps, and train_sft.jsonl outputs
├── src/
│   ├── config.py                # Hyperparameters, directory path states, hardware switches
│   ├── solvers.py               # Algorithmic deterministic solvers (bitwise, roman, gravity, symbolic)
│   ├── utils.py                 # Mathematical canonical formatting, match validation
│   ├── data_prep.py             # Phase 1 logic runner: cache tracking, synthetic builder
│   ├── train.py                 # Phase 2 logic runner: QLoRA setups, SFT and GRPO train engines
│   └── inference.py             # Phase 3 logic runner: adaptive votings, retry routines
├── notebooks/
│   ├── 01_data_prep.ipynb       # Wrapping Phase 1 data pipelines
│   ├── 02_training.ipynb        # Wrapping Phase 2 deep learning training
│   └── 03_inference.ipynb       # Wrapping Phase 3 consistency evaluations
├── scripts/
│   ├── run_pipeline.py          # Unified CLI entrypoint to drive pipeline steps
│   ├── setup_env.bat            # One-click Windows virtualenv builder
│   └── setup_env.sh             # Linux bash environment automated script
├── tests/
│   ├── test_solvers.py          # Unit tests verifying mathematical solvers
│   └── test_utils.py            # Unit tests verifying latex normalizations & matching
├── Dockerfile                   # Multi-stage, reproducible CUDA-ready runner image
├── README.md                    # Core project documentation
└── requirements.txt             # Dependency constraints specifying PEFT, TRL, etc.
```

---

## 🛠️ Workstation Setup & Setup Guides

### 1. Local Workstation Setup (Windows / CMD)
Double-click `scripts/setup_env.bat` or run it directly in your command shell to automatically establish a virtual environment and configure dependencies:
```cmd
scripts\setup_env.bat
```

To activate the virtual environment manually later:
```cmd
:: Command Prompt
.venv\Scripts\activate.bat

:: PowerShell
.venv\Scripts\Activate.ps1
```

### 2. UNIX / Linux Setup
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

---

## 🚀 Unified Pipeline CLI Commands

Drive execution dynamically using the Python orchestrator script `scripts/run_pipeline.py`.

### 🔬 Run Phase 1: Data Preparation
Prepares datasets, runs programmatic solvers to build offline CoT caches, triggers synthetic puzzle generation, and token limits:
```bash
python scripts/run_pipeline.py --phase prep
```

### 🚀 Run Phase 2: QLoRA SFT & GRPO Training
Performs SFT training followed by outcome-based policy reinforcement learning:
```bash
python scripts/run_pipeline.py --phase train
```

### 🏆 Run Phase 3: Adaptive Consistency Inference
Generates submissions (`submission.csv`) using majority voting, customized prompt cards, and retry passes for uncertain outputs:
```bash
python scripts/run_pipeline.py --phase infer
```

### 🔁 Run Full Sequential Pipeline End-to-End
```bash
python scripts/run_pipeline.py --phase all
```

### 🪲 Debug Mode (Dry Run)
You can append the `--limit` flag to any command to run a fast dry run on a micro-set of data in seconds without expensive compute resources or API calls:
```bash
python scripts/run_pipeline.py --phase all --limit 5
```

---

## 🧪 Unit Testing

We employ `pytest` to verify mathematical solvers and LaTeX normalizers. Run unit tests locally before pushing to git:
```bash
pytest tests/
```

Unit tests will also execute automatically on every push or pull request to the `master`/`main` branches via GitHub Actions.

---

## 🐳 Container Run (Docker)

To run the pipeline in a fully isolated container environment:
```bash
# Build the runner image
docker build -t nemotron-runner .

# Execute unit test suites in container
docker run --gpus all -it nemotron-runner pytest

# Run pipeline preparation inside container
docker run --gpus all -v $(pwd)/data:/app/data -it nemotron-runner python scripts/run_pipeline.py --phase prep
```
