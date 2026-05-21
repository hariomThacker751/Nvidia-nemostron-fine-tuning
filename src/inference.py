import re
import gc
import json
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers import LogitsProcessor, LogitsProcessorList
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    class LogitsProcessor: pass
    class LogitsProcessorList:
        def __init__(self, *args, **kwargs): pass

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from src import config
from src.utils import logger, normalize_answer, extract_boxed_answer
from src.solvers import deterministic_answer

# ─── INFERENCE CATEGORY ROUTING ────────────────────────────────────────

CATEGORY_PROMPTS = {
    'bit_manipulation': (
        'You specialize in bitwise operations. '
        'Always convert to binary, show each bit column, then convert back. '
        'Answer in \\boxed{}.'
    ),
    'algebraic_equation': (
        'You specialize in algebra. '
        'Show each algebraic step. Verify by substituting back. '
        'Answer in \\boxed{}.'
    ),
    'modular_arithmetic': (
        'You specialize in number theory. '
        'Use Fermat\'s Little Theorem or direct computation. '
        'Show intermediate modular reductions. Answer in \\boxed{}.'
    ),
    'sequence_pattern': (
        'You specialize in sequences. '
        'State the rule explicitly. Show the formula. '
        'Verify with known terms. Answer in \\boxed{}.'
    ),
    'combinatorics': (
        'You specialize in combinatorics. '
        'Identify whether order matters. '
        'Show the formula and computation. Answer in \\boxed{}.'
    ),
    'number_theory': (
        'You specialize in number theory. '
        'Show factorization or GCD/LCM computation step by step. '
        'Answer in \\boxed{}.'
    ),
}

DEFAULT_SYSTEM = (
    'Reason step by step inside <think> tags. '
    'Show all work. Answer in \\boxed{}.'
)


def tag_puzzle_category(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ['bit', 'xor', 'and ', 'or ', 'shift', 'binary', 'bitwise']):
        return 'bit_manipulation'
    if any(w in p for w in ['equation', 'solve', 'algebra']):
        return 'algebraic_equation'
    if any(w in p for w in ['sequence', 'next', 'pattern', 'series']):
        return 'sequence_pattern'
    if any(w in p for w in ['mod ', 'modulo', 'remainder']):
        return 'modular_arithmetic'
    if any(w in p for w in ['count', 'how many', 'combinations']):
        return 'combinatorics'
    if any(w in p for w in ['prime', 'factor', 'gcd', 'lcm']):
        return 'number_theory'
    return 'other'


# ─── VOTING & SELF-CONSISTENCY ALGORITHMS ─────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract boxed math answer, fallback to last numeric entity."""
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    # Numeric regex fallback
    lines = str(text).strip().split('\n')
    for line in reversed(lines):
        nums = re.findall(r'-?\d+\.?\d*', line)
        if nums:
            return nums[-1]
    return None


def compute_answer_confidence(candidates: list[str]) -> tuple[str, float]:
    """Perform majority voting with math normalization checks."""
    if not candidates:
        return 'unknown', 0.0
        
    normalized = [normalize_answer(c) for c in candidates if c]
    if not normalized:
        return candidates[0] if candidates else 'unknown', 0.0
        
    counts = Counter(normalized)
    best_norm, best_count = counts.most_common(1)[0]
    confidence = best_count / len(candidates)
    
    # Map normalized token back to original generated text
    for orig, norm in zip(candidates, normalized):
        if norm == best_norm:
            return orig, confidence
    return best_norm, confidence


class BoxedBitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, is_bit_problem):
        self.tokenizer = tokenizer
        self.is_bit_problem = is_bit_problem
        try:
            self.allowed_ids = list({tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["0", "1", "}"]})
        except Exception:
            self.allowed_ids = []

    def __call__(self, input_ids, scores):
        if not HAS_TORCH or not self.is_bit_problem or not self.allowed_ids:
            return scores
        try:
            context = self.tokenizer.decode(input_ids[0][-20:])
            if "\\boxed{" in context and "}" not in context.split("\\boxed{")[-1]:
                mask = torch.ones_like(scores, dtype=torch.bool)
                mask[:, self.allowed_ids] = False
                scores = scores.masked_fill(mask, -float("inf"))
        except Exception:
            pass
        return scores


def inference_self_consistency(
    prompt_text: str,
    tokenizer,
    model,
    n_votes: int = 3,
    max_new_tokens: int = 1000,
    temperature: float = 0.7,
    system_msg: str = None
) -> dict:
    """Run multiple randomized rollout loops and vote on standard solutions."""
    # Check if model has standard PyTorch weights or is mock
    if not HAS_TORCH or tokenizer is None or not hasattr(model, "device"):
        # Mock mode
        return {
            'answer': 'mock_answer',
            'confidence': 1.0,
            'all_answers': ['mock_answer'],
            'n_agree': n_votes,
        }

    if system_msg is None:
        system_msg = DEFAULT_SYSTEM
        
    messages = [
        {'role': 'system', 'content': system_msg},
        {'role': 'user', 'content': prompt_text},
    ]
    
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
    except Exception:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    is_bit = "8-bit binary" in prompt_text.lower() or "bit manipulation" in prompt_text.lower()
    processors = LogitsProcessorList([BoxedBitLogitsProcessor(tokenizer, is_bit)])
    
    all_answers = []
    with torch.no_grad():
        for _ in range(n_votes):
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(n_votes > 1),
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=processors,
            )
            gen_tokens = output[0][inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            all_answers.append(extract_answer(text))
            
    best_answer, confidence = compute_answer_confidence(all_answers)
    return {
        'answer': best_answer,
        'confidence': confidence,
        'all_answers': all_answers,
        'n_agree': int(confidence * n_votes),
    }


def adaptive_inference(prompt_text: str, tokenizer, model) -> dict:
    """Route puzzles to custom system prompt templates with adjusted vote counts."""
    category = tag_puzzle_category(prompt_text)
    system_msg = CATEGORY_PROMPTS.get(category, DEFAULT_SYSTEM)
    n_votes = config.CATEGORY_VOTES.get(category, 3)
    
    result = inference_self_consistency(
        prompt_text=prompt_text,
        tokenizer=tokenizer,
        model=model,
        n_votes=n_votes,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.INFERENCE_TEMP,
        system_msg=system_msg
    )
    result['category'] = category
    result['n_votes'] = n_votes
    return result


# ─── INFERENCE SYSTEM RUNNER ──────────────────────────────────────────

def run_inference_pipeline(limit: int = None):
    """Loads packaged adapter from Phase 2 training, routes inputs, retries low confidence, and exports output."""
    logger.info("Initializing Phase 3 Inference Pipeline...")
    
    # Resolve adapter paths
    grpo_adapter = config.ADAPTER_DIR / "grpo"
    sft_adapter = config.ADAPTER_DIR / "sft"
    adapter_dataset = grpo_adapter if grpo_adapter.exists() else sft_adapter
    
    logger.info(f"Resolved model checkpoint adapter to load: {adapter_dataset}")
    
    # Quantization Setup
    if HAS_TORCH and HAS_TRANSFORMERS:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        max_memory = {
            0: f"{int(config.GPU_MEM_HEADROOM_GB)}GiB",
            "cpu": "30GiB"
        } if torch.cuda.device_count() == 1 else {
            0: f"{int(config.GPU_MEM_HEADROOM_GB)}GiB",
            1: f"{int(config.GPU_MEM_HEADROOM_GB)}GiB",
            "cpu": "30GiB"
        }
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        bnb_config = None
        max_memory = None
    
    # Fallback tokenizer load
    tokenizer_path = adapter_dataset if adapter_dataset.exists() else config.MODEL_ID
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    if HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            tokenizer = None
    else:
        tokenizer = None
        
    logger.info(f"Loading quantized base model: {config.MODEL_ID}")
    try:
        if not HAS_TORCH or not HAS_TRANSFORMERS or not HAS_PEFT:
            raise ImportError("Deep learning packages are missing.")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            quantization_config=bnb_config,
            device_map=config.device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation=config.attn_implementation,
        )
        
        if adapter_dataset.exists():
            logger.info(f"Layering LoRA adapter weights onto model...")
            model = PeftModel.from_pretrained(base_model, str(adapter_dataset))
        else:
            logger.warning("No adapter weights found under models/lora_adapter. Running raw model baseline.")
            model = base_model
            
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model layers: {e}")
        logger.warning("Mocking model run for standard CPU fallback (dry runs/tests).")
        model = object()  # mock placeholder
        
    # Read test file
    test_csv_path = config.DATA_RAW_DIR / "test.csv"
    if not test_csv_path.exists():
        logger.error(f"Test csv dataset missing: {test_csv_path}. Creating a dummy fallback test.csv.")
        dummy_df = pd.DataFrame([
            {"id": 1, "prompt": "Compute 12 AND 5 bitwise. Dec value?"},
            {"id": 2, "prompt": "Solve 3x + 4 = 10"}
        ])
        dummy_df.to_csv(test_csv_path, index=False)
        
    test_df = pd.read_csv(test_csv_path)
    if limit:
        test_df = test_df.head(limit)
        
    logger.info(f"Running inference on {len(test_df)} records...")
    
    answers = {}
    unresolved = []
    
    for _, row in test_df.iterrows():
        ans, conf, reason = deterministic_answer(row['prompt'])
        if ans:
            logger.info(f"Solved puzzle {row['id']} deterministically: {ans} ({reason})")
            answers[row['id']] = {
                'answer': ans,
                'confidence': 1.0,
                'category': tag_puzzle_category(row['prompt']),
                'n_votes': 1,
                'all_answers': [ans]
            }
        else:
            unresolved.append(row)
            
    logger.info(f"Deterministic solvers answered {len(answers)} / {len(test_df)} records.")
    
    results = []
    # Add deterministic results
    for qid, res in answers.items():
        results.append({
            'id': qid,
            'answer': res['answer'],
            'confidence': res['confidence'],
            'category': res['category'],
            'n_votes': res['n_votes'],
            'all_answers': str(res['all_answers'])
        })
        
    if unresolved:
        logger.info(f"Running model inference on remaining {len(unresolved)} unresolved records...")
        
        model_results = []
        for _, row in tqdm(pd.DataFrame(unresolved).iterrows(), total=len(unresolved), desc='Inference'):
            res = adaptive_inference(row['prompt'], tokenizer, model)
            model_results.append({
                'id': row['id'],
                'answer': res['answer'],
                'confidence': res['confidence'],
                'category': res['category'],
                'n_votes': res['n_votes'],
                'all_answers': str(res['all_answers'])
            })
            
        model_results_df = pd.DataFrame(model_results)
        
        # Retry logic for low confidence model responses
        if config.RETRY_LOW_CONF and not model_results_df.empty and hasattr(model, "device"):
            low_conf = model_results_df[model_results_df['confidence'] < config.LOW_CONF_THRESHOLD]
            if not low_conf.empty:
                logger.info(f"Re-running inference for {len(low_conf)} low-confidence predictions...")
                id_to_prompt = {row['id']: row['prompt'] for row in unresolved}
                for row_id in tqdm(low_conf['id'].tolist(), desc='Retrying Low Conf'):
                    prompt = id_to_prompt[row_id]
                    res = inference_self_consistency(
                        prompt_text=prompt,
                        tokenizer=tokenizer,
                        model=model,
                        n_votes=config.RETRY_N_VOTES,
                        max_new_tokens=config.MAX_NEW_TOKENS,
                        temperature=config.RETRY_TEMP
                    )
                    mask = model_results_df['id'] == row_id
                    model_results_df.loc[mask, 'answer'] = res['answer']
                    model_results_df.loc[mask, 'confidence'] = res['confidence']
                    
        # Append model results to final results list
        for _, row in model_results_df.iterrows():
            results.append(row.to_dict())
            
    results_df = pd.DataFrame(results)
    
    # Normalize answers
    if not results_df.empty:
        results_df['answer'] = results_df['answer'].apply(lambda x: normalize_answer(x) if x else 'unknown')
        
        # Sort results to match original test_df ID sequence
        id_to_pos = {row_id: i for i, row_id in enumerate(test_df['id'])}
        results_df['sort_key'] = results_df['id'].map(id_to_pos)
        results_df = results_df.sort_values('sort_key').drop(columns=['sort_key'])
    else:
        results_df = pd.DataFrame(columns=['id', 'answer'])
    
    # Export submissions.csv
    submission_path = config.PROJECT_ROOT / "submission.csv"
    results_df[['id', 'answer']].to_csv(submission_path, index=False)
    
    logger.info(f"Phase 3 complete! Packaged submission.csv export to: {submission_path}")
    logger.info(f"Submission shape: {results_df[['id', 'answer']].shape}")
