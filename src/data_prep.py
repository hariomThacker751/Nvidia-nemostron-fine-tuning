import re
import json
import random
from pathlib import Path
import pandas as pd
import sympy as sp
from tqdm.auto import tqdm

from src import config
from src.utils import logger, extract_boxed_answer
import src.solvers as solvers

# ─── EDA & CATEGORIZATION ──────────────────────────────────────────────

def categorize_puzzle(prompt: str) -> str:
    """Heuristics to classify puzzles for downstream token routing & prompts."""
    p = prompt.lower()
    if any(w in p for w in ['bit', 'xor', 'and ', 'or ', 'shift', 'binary', 'bitwise']):
        return 'bit_manipulation'
    if any(w in p for w in ['equation', 'solve', 'algebra', '= ?', 'find x', 'find the value']):
        return 'algebraic_equation'
    if any(w in p for w in ['sequence', 'next', 'pattern', 'series']):
        return 'sequence_pattern'
    if any(w in p for w in ['mod ', 'modulo', 'remainder', 'divisible']):
        return 'modular_arithmetic'
    if any(w in p for w in ['count', 'how many', 'combinations', 'permutations']):
        return 'combinatorics'
    if any(w in p for w in ['logic', 'true', 'false', 'boolean', 'if and only', 'implies']):
        return 'propositional_logic'
    if any(w in p for w in ['matrix', 'determinant', 'eigenvalue', 'vector']):
        return 'linear_algebra'
    if any(w in p for w in ['prime', 'factor', 'gcd', 'lcm', 'divisor']):
        return 'number_theory'
    if any(w in p for w in ['probability', 'chance', 'expected', 'random']):
        return 'probability'
    if any(w in p for w in ['graph', 'node', 'edge', 'path', 'cycle']):
        return 'graph_theory'
    return 'other'


# ─── CO T GENERATION ROUTINES ──────────────────────────────────────────

TEMPLATES = {
    'bit_manipulation': """<think>
Let me work through this bit manipulation problem step by step.
Step 1: Identify the operation and operands.
Step 2: Convert numbers to binary representation.
Step 3: Apply the bitwise operation column by column.
Step 4: Convert the result back to the required format.
Step 5: Verify: re-apply the operation to confirm.
</think>
Working through the bit manipulation:
{reasoning}
The answer is \\boxed{{{answer}}}""",
    'algebraic_equation': """<think>
Let me solve this algebraic equation systematically.
Step 1: Identify what we're solving for.
Step 2: Isolate the variable using algebraic operations.
Step 3: Simplify each side.
Step 4: Check the solution satisfies the original equation.
</think>
Solving the equation:
{reasoning}
Therefore, \\boxed{{{answer}}}""",
    'sequence_pattern': """<think>
Let me identify the pattern in this sequence.
Step 1: List out the given terms.
Step 2: Compute differences or ratios between consecutive terms.
Step 3: State the rule explicitly.
Step 4: Apply the rule to find the target term.
Step 5: Verify by applying the rule to known terms.
</think>
Analyzing the sequence:
{reasoning}
The answer is \\boxed{{{answer}}}""",
    'modular_arithmetic': """<think>
Working with modular arithmetic.
Step 1: Identify the modulus and the expression.
Step 2: Simplify using modular arithmetic properties.
Step 3: Apply Fermat's Little Theorem or direct computation as needed.
Step 4: Reduce to the canonical range [0, modulus-1].
</think>
Solving modular arithmetic:
{reasoning}
Therefore, \\boxed{{{answer}}}""",
}


def build_template_cot(row) -> str:
    """Fallback generator yielding high-quality templated reasonings."""
    category = row.get('category', 'other')
    template = TEMPLATES.get(category, """<think>
Let me analyze this problem carefully.
Step 1: Understand what is being asked.
Step 2: Identify the key operations.
Step 3: Compute step by step.
Step 4: Verify the answer.
</think>
Working through the problem:
{reasoning}
Therefore, \\boxed{{{answer}}}""")
    return template.format(
        reasoning=f"Given: {row['prompt']}\nComputed answer: {row['answer']}",
        answer=row['answer']
    )


def generate_cot_offline(prompt: str) -> str | None:
    """Use the programmatic solver logic to generate a rich, accurate CoT trace."""
    ans, conf, reason = solvers.deterministic_answer(prompt)
    if ans is None:
        return None
    cot = (
        f"<think>\n"
        f"Applying programmatic reasoning solver:\n"
        f"Strategy: {reason}\n"
        f"Confidence: {conf}\n"
        f"Derived answer exactly:\n"
        f"{ans}\n"
        f"</think>\n"
        f"\nTherefore, \\boxed{{{ans}}}"
    )
    return cot


# ─── SYNTHETIC PUZZLE GENERATORS ───────────────────────────────────────

def gen_bit_manipulation_example(counter_state):
    ops = ['AND', 'OR', 'XOR']
    op = random.choice(ops)
    a, b = random.randint(0, 255), random.randint(0, 255)
    ans = a & b if op == 'AND' else (a | b if op == 'OR' else a ^ b)
    
    prompt = f'Compute {a} {op} {b} where AND, OR, XOR are bitwise operations. Give the decimal result.'
    reasoning = (
        f'<think>\n'
        f'a = {a} = {bin(a)}\n'
        f'b = {b} = {bin(b)}\n'
        f'a {op} b = {bin(a)} {op} {bin(b)} = {bin(ans)}\n'
        f'{bin(ans)} in decimal = {ans}\n'
        f'</think>\n'
        f'Computing bitwise {op}:\n  {a} = {bin(a)}\n  {b} = {bin(b)}\n  {a} {op} {b} = {ans}\n\\boxed{{{ans}}}'
    )
    counter_state[0] += 1
    return {'id': counter_state[0], 'prompt': prompt, 'answer': str(ans), 'cot': reasoning, 'category': 'bit_manipulation', 'synthetic': True}


def gen_linear_equation(counter_state):
    a = random.randint(1, 20)
    b = random.randint(-50, 50)
    c = random.randint(-100, 100)
    x_val = sp.Rational(c - b, a)
    prompt = f'Solve for x: {a}x + {b} = {c}'
    reasoning = (
        f'<think>\nGiven: {a}x + {b} = {c}\n'
        f'Subtract {b} from both sides: {a}x = {c} - ({b}) = {c-b}\n'
        f'Divide both sides by {a}: x = {c-b}/{a} = {x_val}\n'
        f'Verify: {a}*({x_val}) + {b} = {a*x_val} + {b} = {a*x_val + b} ✓\n</think>\n'
        f'Solving {a}x + {b} = {c}:\n{a}x = {c - b}\nx = {x_val}\n\\boxed{{{x_val}}}'
    )
    counter_state[0] += 1
    return {'id': counter_state[0], 'prompt': prompt, 'answer': str(x_val), 'cot': reasoning, 'category': 'algebraic_equation', 'synthetic': True}


def gen_modular_example(counter_state):
    base = random.randint(2, 100)
    exp  = random.randint(2, 20)
    mod  = random.choice([7, 11, 13, 17, 19, 23, 31, 97, 101])
    ans  = pow(base, exp, mod)
    prompt = f'Compute {base}^{exp} mod {mod}.'
    reasoning = (
        f'<think>\nCompute {base}^{exp} mod {mod} using fast exponentiation.\n'
        f'{base}^{exp} = {base**exp}\n{base**exp} mod {mod} = {ans}\n</think>\n'
        f'Using modular exponentiation:\n{base}^{exp} mod {mod} = {ans}\n\\boxed{{{ans}}}'
    )
    counter_state[0] += 1
    return {'id': counter_state[0], 'prompt': prompt, 'answer': str(ans), 'cot': reasoning, 'category': 'modular_arithmetic', 'synthetic': True}


def gen_arithmetic_sequence(counter_state):
    a0 = random.randint(1, 50)
    d  = random.randint(1, 20)
    n  = random.randint(5, 15)
    target_idx = random.randint(n+1, n+10)
    seq = [a0 + d*i for i in range(n)]
    answer = a0 + d * target_idx
    prompt = f'Given the sequence: {", ".join(map(str, seq))}, ...\nWhat is the {target_idx+1}th term (1-indexed)?'
    reasoning = (
        f'<think>\nTerms: {", ".join(map(str, seq))}\n'
        f'Differences: {", ".join(str(seq[i+1]-seq[i]) for i in range(len(seq)-1))}\n'
        f'Common difference d = {d}\nFirst term a0 = {a0}\n'
        f'Formula: a(n) = a0 + (n-1)*d = {a0} + (n-1)*{d}\n'
        f'a({target_idx+1}) = {a0} + {target_idx}*{d} = {answer}\n</think>\n'
        f'Arithmetic sequence with first term {a0} and common difference {d}.\n'
        f'a({target_idx+1}) = {a0} + {target_idx}×{d} = {answer}\n\\boxed{{{answer}}}'
    )
    counter_state[0] += 1
    return {'id': counter_state[0], 'prompt': prompt, 'answer': str(answer), 'cot': reasoning, 'category': 'sequence_pattern', 'synthetic': True}


# ─── DATA EXPORT & PACKAGING ──────────────────────────────────────────

def format_sft_example(prompt: str, cot_response: str) -> dict:
    nemotron_system = (
        'A conversation between a user and an AI assistant. '
        'The AI assistant reasons step-by-step inside <think> tags '
        'and outputs the final answer in \\boxed{} format.'
    )
    return {
        'messages': [
            {'role': 'system',    'content': nemotron_system},
            {'role': 'user',      'content': prompt},
            {'role': 'assistant', 'content': cot_response},
        ]
    }


def validate_example(example: dict) -> bool:
    """Verify boxed answer exists."""
    assistant_text = example['messages'][2]['content']
    return bool(extract_boxed_answer(assistant_text))


def run_data_prep(train_path: Path, test_path: Path, output_dir: Path, limit: int = None):
    """Executes EDA, runs programmatic CoT generation, appends synthetic records, and saves the final JSONL."""
    logger.info("Initializing Phase 1: Data Preparation Pipeline...")
    
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training csv dataset: {train_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    
    if limit:
        logger.info(f"Limiting inputs to top {limit} records for fast debugging.")
        train_df = train_df.head(limit)
        if not test_df.empty:
            test_df = test_df.head(limit)
            
    # Tag Categories
    logger.info("Categorizing prompts...")
    train_df['category'] = train_df['prompt'].apply(categorize_puzzle)
    train_df.to_csv(output_dir / "train_categorized.csv", index=False)
    
    # Load / Initialize CoT caches
    cot_cache = {}
    cot_cache_path = config.COT_CACHE_FILE
    if cot_cache_path.exists():
        logger.info(f"Loading cached traces from: {cot_cache_path}")
        with open(cot_cache_path) as f:
            for line in f:
                item = json.loads(line)
                cot_cache[str(item['id'])] = item['cot']
                
    # Run deterministic CoT solver
    logger.info("Generating offline deterministic CoT reasoning chains...")
    offline_count = 0
    with open(cot_cache_path, 'a' if cot_cache_path.exists() else 'w') as cache_f:
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc='Solving Puzzles'):
            row_id = str(row['id'])
            if row_id in cot_cache:
                continue
                
            cot_text = generate_cot_offline(row['prompt'])
            if cot_text is not None and '\\boxed{' in cot_text:
                offline_count += 1
            else:
                cot_text = build_template_cot(row)
                
            cot_cache[row_id] = cot_text
            cache_f.write(json.dumps({'id': row_id, 'cot': cot_text}) + '\n')
            
    logger.info(f"CoT cache synced. Total: {len(cot_cache)} (Offline deterministic solver found: {offline_count})")
    
    # Generate Synthetic Puzzles
    synthetic_examples = []
    syn_id_counter = [100000]
    
    # Scale synthetic count based on limit switch
    run_synthetic_count = config.SYNTHETIC_PER_TYPE if not limit else min(5, limit)
    logger.info(f"Generating synthetic augmentations ({run_synthetic_count} per category)...")
    
    generators = [
        gen_bit_manipulation_example,
        gen_linear_equation,
        gen_modular_example,
        gen_arithmetic_sequence
    ]
    
    for gen_fn in generators:
        for _ in range(run_synthetic_count):
            try:
                synthetic_examples.append(gen_fn(syn_id_counter))
            except Exception as e:
                logger.debug(f"Generator failure: {e}")
                
    syn_df = pd.DataFrame(synthetic_examples)
    logger.info(f"Generated {len(syn_df)} synthetic records.")
    
    # SFT formatting and validation
    logger.info("Validating and formatting outputs into standard Nemotron JSONL...")
    sft_examples = []
    skipped = 0
    
    # Process original dataset
    for _, row in train_df.iterrows():
        row_id = str(row['id'])
        cot = cot_cache.get(row_id, build_template_cot(row))
        if '\\boxed{' not in cot:
            cot = cot + f"\n\nTherefore, \\boxed{{{row['answer']}}}"
        ex = format_sft_example(row['prompt'], cot)
        if validate_example(ex):
            sft_examples.append(ex)
        else:
            skipped += 1
            
    # Process synthetic dataset
    for _, row in syn_df.iterrows():
        ex = format_sft_example(row['prompt'], row['cot'])
        if validate_example(ex):
            sft_examples.append(ex)
        else:
            skipped += 1
            
    # Shuffle dataset
    random.seed(42)
    random.shuffle(sft_examples)
    
    # Save SFT File
    sft_path = config.TRAIN_SFT_OUTPUT
    with open(sft_path, 'w') as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + '\n')
            
    logger.info(f"SFT Dataset complete! Saved {len(sft_examples)} samples to: {sft_path} (skipped: {skipped})")
    
    # Token metrics overview
    char_lens = [len(' '.join(m['content'] for m in ex['messages'])) for ex in sft_examples]
    approx_tokens = [cl // 4 for cl in char_lens]
    if approx_tokens:
        p95 = int(pd.Series(approx_tokens).quantile(0.95))
        logger.info(f"Approximate Token length metric: Mean={int(sum(approx_tokens)/len(approx_tokens))}, P95={p95}")
        logger.info(f"Recommended MAX_SEQ_LENGTH configuration: {int(p95 * 1.2)}")
    
    logger.info("Phase 1 complete! Saved outputs successfully.")
