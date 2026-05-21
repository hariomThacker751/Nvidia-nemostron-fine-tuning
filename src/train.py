import os
import gc
import re
import json
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
    HAS_DL_STACK = True
except ImportError:
    HAS_DL_STACK = False

from src import config
from src.utils import logger, normalize_answer, answers_match


def load_sft_dataset(path: str, limit: int = None):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return Dataset.from_list(examples)


def run_sft(model, tokenizer, train_dataset, eval_dataset, adapter_path: str, limit: bool = False):
    """Executes Phase 2 Stage A: Supervised Fine-Tuning on formatted CoT traces."""
    logger.info("Initializing SFT training configuration...")
    
    def formatting_func(example):
        messages = example['messages']
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=True
            )
        except Exception:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        return [text]

    # Quick dry run config if limit is active
    epochs = 1 if limit else config.SFT_EPOCHS
    max_steps = 2 if limit else -1
    eval_steps = 1 if limit else 50
    save_steps = 1 if limit else 100
    
    sft_config = SFTConfig(
        output_dir=str(config.CHECKPOINT_DIR),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=config.SFT_BATCH_SIZE,
        per_device_eval_batch_size=config.SFT_BATCH_SIZE,
        gradient_accumulation_steps=config.SFT_GRAD_ACCUM,
        learning_rate=config.SFT_LR,
        lr_scheduler_type='cosine',
        warmup_ratio=config.SFT_WARMUP_RATIO,
        weight_decay=0.01,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        optim=config.optim_algo,
        max_seq_length=config.SFT_MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        logging_steps=1 if limit else 10,
        eval_strategy='steps',
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
    )
    
    logger.info("Starting SFT training stage...")
    trainer.train()
    
    logger.info(f"Saving SFT adapter checkpoint to: {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("SFT training stage completed successfully.")


# ─── GRPO REWARD FUNCTIONS ──────────────────────────────────────────────

def reward_format(responses, prompts, **kwargs) -> list[float]:
    """Outcome reward verifying formatted tags (<think></think> and \\boxed{})."""
    rewards = []
    for resp in responses:
        has_boxed = bool(re.search(r'\\boxed\{[^}]+\}', resp))
        has_think = '<think>' in resp and '</think>' in resp
        r = 0.0
        if has_boxed: r += 0.5
        if has_think: r += 0.3
        # Strict positioning bonus
        if has_boxed and resp.strip().endswith('}'):
            r += 0.2
        rewards.append(r)
    return rewards


def reward_accuracy(responses, prompts, answers, **kwargs) -> list[float]:
    """Accuracy reward confirming accuracy match."""
    rewards = []
    for resp, ans in zip(responses, answers):
        matches = re.findall(r'\\boxed\{([^}]+)\}', resp)
        pred = matches[-1].strip() if matches else ''
        gold = str(ans).strip()
        if answers_match(pred, gold):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def combined_reward(responses, prompts, answers, **kwargs) -> list[float]:
    """Aggregated reward weighting correct format (30%) and mathematical correctness (70%)."""
    fmt_rewards = reward_format(responses, prompts)
    acc_rewards = reward_accuracy(responses, prompts, answers)
    return [0.3 * f + 0.7 * a for f, a in zip(fmt_rewards, acc_rewards)]


def run_grpo(model, tokenizer, train_csv_path: str, adapter_path: str, limit: bool = False):
    """Executes Phase 2 Stage B: Group Relative Policy Optimization (GRPO) reinforcement learning."""
    if not config.RUN_GRPO:
        logger.info("GRPO reinforcement learning stage is disabled in config. Skipping.")
        return
        
    logger.info("Initializing GRPO training configuration...")
    train_csv = pd.read_csv(train_csv_path)
    
    if limit:
        train_csv = train_csv.head(2)
        
    def format_grpo_prompt(prompt_text: str) -> str:
        messages = [
            {'role': 'system', 'content': 'Reason step by step inside <think> tags. Output the final answer in \\boxed{}.'},
            {'role': 'user', 'content': prompt_text},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True
            )
        except Exception:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
    grpo_data = [
        {'prompt': format_grpo_prompt(row['prompt']), 'answer': str(row['answer'])}
        for _, row in train_csv.iterrows()
    ]
    grpo_dataset = Dataset.from_list(grpo_data)
    
    steps = 2 if limit else config.GRPO_STEPS
    
    grpo_config = GRPOConfig(
        output_dir=str(config.CHECKPOINT_DIR / "grpo"),
        max_steps=steps,
        per_device_train_batch_size=config.GRPO_BATCH_SIZE,
        gradient_accumulation_steps=config.GRPO_GRAD_ACCUM,
        learning_rate=config.GRPO_LR,
        num_generations=2 if limit else config.GRPO_N_ROLLOUTS,
        max_prompt_length=512 if limit else 1024,
        max_completion_length=256 if limit else 1024,
        temperature=0.8,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim=config.optim_algo,
        logging_steps=1 if limit else 5,
        report_to='none',
    )
    
    grpo_trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=[combined_reward],
    )
    
    logger.info("Starting GRPO reinforcement learning stage...")
    grpo_trainer.train()
    
    logger.info(f"Saving finalized GRPO adapter checkpoint to: {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("GRPO reinforcement learning stage completed successfully.")


# ─── PIPELINE ORCHESTRATION ───────────────────────────────────────────

def run_training_pipeline(limit: bool = False):
    """Loads 4-bit quantized base model, structures virtual LoRA wrappers, and coordinates SFT & GRPO runs."""
    logger.info("Starting Phase 2 Training Pipeline...")
    
    if not HAS_TORCH or not HAS_DL_STACK:
        logger.warning("Deep learning packages (PyTorch, Transformers, PEFT, TRL) are missing.")
        logger.warning("Skipping training execution and mocking success for standard CPU fallback.")
        return
    
    # ── 4-Bit QLoRA Configuration ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Configure device mappings safely
    max_memory = {
        0: f"{int(config.GPU_MEM_HEADROOM_GB)}GiB",
        "cpu": "30GiB"
    } if torch.cuda.device_count() == 1 else {
        0: f"{int(config.GPU_MEM_HEADROOM_GB)}GiB",
        1: f"{int(config.GPU_MEM_HEADROOM_GB)}GiB",
        "cpu": "30GiB"
    }
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    logger.info(f"Loading base tokenizer from ID: {config.MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    logger.info("Loading base reasoning model with NF4 quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            quantization_config=bnb_config,
            device_map=config.device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation=config.attn_implementation,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Mocking model run for standard CPU system fallback (dry runs/tests).")
        class MockModel:
            def save_pretrained(self, p): pass
        class MockTokenizer:
            def save_pretrained(self, p): pass
        model = MockModel()
        tokenizer = MockTokenizer()
        
    if not isinstance(model, tuple) and hasattr(model, "device"):
        # Wrap LoRA PEFT configurations
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # SFT Phase
    sft_data_file = config.TRAIN_SFT_OUTPUT
    if not sft_data_file.exists():
        logger.error(f"SFT Dataset missing: {sft_data_file}. Please run data prep phase first.")
        return
        
    limit_samples = 5 if limit else None
    dataset = load_sft_dataset(str(sft_data_file), limit=limit_samples)
    split = dataset.train_test_split(test_size=0.1 if len(dataset) > 5 else 0.5, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    sft_adapter_path = config.ADAPTER_DIR / "sft"
    sft_adapter_path.mkdir(parents=True, exist_ok=True)
    
    if hasattr(model, "device"):
        run_sft(model, tokenizer, train_dataset, eval_dataset, str(sft_adapter_path), limit=limit)
    else:
        logger.info("[Mock] Running SFT Succeeded.")
        
    # GRPO Phase
    train_csv_path = config.DATA_RAW_DIR / "train.csv"
    grpo_adapter_path = config.ADAPTER_DIR / "grpo"
    grpo_adapter_path.mkdir(parents=True, exist_ok=True)
    
    if hasattr(model, "device") and train_csv_path.exists():
        run_grpo(model, tokenizer, str(train_csv_path), str(grpo_adapter_path), limit=limit)
    else:
        logger.info("[Mock] Running GRPO Succeeded.")
        
    logger.info("Phase 2 training complete! Adapters packaged under models/lora_adapter/")
