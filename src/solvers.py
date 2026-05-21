# Programmatic deterministic solvers and verification helpers for different reasoning puzzles.
import re
import os
import json
import time
import itertools
from decimal import Decimal
from src.utils import clean_answer, normalize_answer, logger


def classify_prompt(prompt) -> str:
    """Classify the puzzle category based on prompt patterns."""
    lower = str(prompt).lower()
    if "8-bit binary" in lower or "bit manipulation" in lower:
        return "bitwise"
    if "secret encryption" in lower or "decrypt the following text" in lower:
        return "cipher"
    if "gravitational constant" in lower:
        return "gravity"
    if "different numeral system" in lower:
        return "roman"
    if "secret unit conversion" in lower:
        return "unit_conversion"
    if "transformation rules is applied to equations" in lower:
        return "symbolic"
    return "unknown"


def roman_numeral(n: int) -> str:
    """Convert integer to Roman numeral."""
    table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out = []
    for value, token in table:
        count, n = divmod(n, value)
        out.append(token * count)
    return "".join(out)


def infer_rounded_factor(observations) -> Decimal:
    """Calculate the exact mathematical interval for the hidden factor based on 0.005 rounding bounds."""
    low_bound = Decimal("-1e20")
    high_bound = Decimal("1e20")
    
    for x_raw, y_raw in observations:
        x = Decimal(str(x_raw))
        y = Decimal(str(y_raw))
        # Since y was rounded to 2 decimal places, true y was in [y - 0.005, y + 0.005)
        eps = Decimal("0.005")
        obs_low = (y - eps) / x
        obs_high = (y + eps) / x
        
        low_bound = max(low_bound, obs_low)
        high_bound = min(high_bound, obs_high)
        
    if low_bound <= high_bound:
        return (low_bound + high_bound) / Decimal(2)
        
    # Fallback to median
    ratios = [Decimal(str(y)) / Decimal(str(x)) for x, y in observations]
    return sum(ratios) / Decimal(len(ratios))


def solve_roman(prompt):
    match = re.search(r"write the number\s+(\d+)", str(prompt), re.I)
    if not match:
        return None, 0.0, "missing target number"
    n = int(match.group(1))
    return roman_numeral(n), 1.0, f"standard roman conversion for {n}"


def q2(value):
    val = round(float(value), 2)
    s = f"{val:.2f}".rstrip("0").rstrip(".")
    return s


def solve_gravity(prompt):
    prompt = str(prompt)
    obs = []
    for t, d in re.findall(r"t\s*=\s*([0-9.]+)s,\s*distance\s*=\s*([0-9.]+)\s*m", prompt):
        t_dec = Decimal(t)
        obs.append((Decimal("0.5") * t_dec * t_dec, Decimal(d)))
    target = re.search(r"falling distance for t\s*=\s*([0-9.]+)s", prompt, re.I)
    if not obs or not target:
        return None, 0.0, "missing gravity observations or target"
    g = infer_rounded_factor(obs)
    t = Decimal(target.group(1))
    raw = Decimal("0.5") * g * t * t
    return q2(raw), 1.0, "exact interval arithmetic gravity fit"


def solve_unit_conversion(prompt):
    prompt = str(prompt)
    obs = [(Decimal(a), Decimal(b)) for a, b in re.findall(r"([0-9.]+)\s*m\s+becomes\s+([0-9.]+)", prompt)]
    target = re.search(r"convert the following measurement:\s*([0-9.]+)\s*m", prompt, re.I)
    if not obs or not target:
        return None, 0.0, "missing unit observations or target"
    factor = infer_rounded_factor(obs)
    raw = Decimal(target.group(1)) * factor
    return q2(raw), 1.0, "exact interval arithmetic unit conversion"


def parse_cipher_pairs(prompt):
    before = str(prompt).split("Now, decrypt the following text:")[0]
    pairs = []
    for line in before.splitlines():
        if "->" in line:
            enc, dec = line.split("->", 1)
            pairs.append((enc.strip(), dec.strip()))
    return pairs


def _try_shift_cipher(pairs, target_text):
    shifts = []
    for enc, dec in pairs:
        if len(enc) != len(dec):
            continue
        for ec, dc in zip(enc, dec):
            if ec == " " and dc == " ":
                continue
            if ec.isalpha() and dc.isalpha():
                shift = (ord(dc.lower()) - ord(ec.lower())) % 26
                shifts.append(shift)
    if not shifts:
        return None
    if len(set(shifts)) != 1:
        return None
    shift = shifts[0]
    result = []
    for ch in target_text:
        if ch.isalpha():
            base = ord('a') if ch.islower() else ord('A')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)


def _solve_cryptogram(pairs, target_words, vocab):
    char_map = {}
    for enc, dec in pairs:
        if len(enc) != len(dec): return None, 0.0
        for ec, dc in zip(enc, dec):
            if ec in char_map and char_map[ec] != dc:
                return None, 0.0
            char_map[ec] = dc

    def word_matches(enc_word, cand_word, current_map):
        if len(enc_word) != len(cand_word): return False
        temp_map = {}
        for ec, cc in zip(enc_word, cand_word):
            if ec in current_map:
                if current_map[ec] != cc: return False
            elif ec in temp_map:
                if temp_map[ec] != cc: return False
            else:
                temp_map[ec] = cc
        return True

    start_time = time.time()
    def backtrack(word_index, current_map):
        if time.time() - start_time > 1.0: return None
        if word_index == len(target_words):
            return current_map
        enc_word = target_words[word_index]
        if all(c in current_map for c in enc_word):
            if "".join(current_map[c] for c in enc_word) in vocab:
                return backtrack(word_index + 1, current_map)
            return None
        for cand in vocab:
            if word_matches(enc_word, cand, current_map):
                new_map = current_map.copy()
                for ec, cc in zip(enc_word, cand):
                    new_map[ec] = cc
                res = backtrack(word_index + 1, new_map)
                if res: return res
        return None

    final_map = backtrack(0, char_map)
    if final_map:
        return " ".join("".join(final_map[c] for c in w) for w in target_words), 1.0
    return None, 0.0


def solve_cipher(prompt):
    prompt = str(prompt)
    target = re.search(r"decrypt the following text:\s*(.+)", prompt, re.I | re.S)
    if not target:
        return None, 0.0, "missing cipher target"
    target_text = target.group(1).strip()
    pairs = parse_cipher_pairs(prompt)

    word_pairs = []
    for enc, dec in pairs:
        for ew, dw in zip(enc.split(), dec.split()):
            word_pairs.append((ew, dw))

    vocab = set()
    # Check parent project root or current working dir for cipher_vocab.json
    vocab_paths = ["cipher_vocab.json", "../cipher_vocab.json"]
    for vp in vocab_paths:
        if os.path.exists(vp):
            try:
                with open(vp) as f:
                    vocab = set(json.load(f))
                break
            except Exception:
                pass
    for _, dw in word_pairs:
        vocab.add(dw)

    res, conf = _solve_cryptogram(word_pairs, target_text.split(), vocab)
    if res:
        return res, conf, "exact dictionary cryptogram match"

    shift_result = _try_shift_cipher(pairs, target_text)
    if shift_result:
        return shift_result, 0.70, "Caesar shift cipher"

    return None, 0.0, "unseen cipher symbol — no strategy resolved"


def _extract_symbolic_target(prompt):
    delimiters = [
        "Now, determine the result for:",
        "Now, determine the result for the following equation:",
        "What is the result for:",
        "Determine the result for:",
        "Apply the rule to:",
        "Now apply it to:",
        "Result for:",
    ]
    for delim in delimiters:
        if delim.lower() in prompt.lower():
            idx = prompt.lower().index(delim.lower())
            before = prompt[:idx]
            after = prompt[idx + len(delim):].strip()
            target = after.splitlines()[0].strip() if after else ""
            if target:
                return before, target
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    if len(lines) >= 2:
        for line in reversed(lines):
            if " = " not in line and len(line) <= 20:
                return "\n".join(lines[:-1]), line
    return None, None


def solve_symbolic_substitution(prompt):
    prompt = str(prompt)
    before, target_str = _extract_symbolic_target(prompt)
    if not target_str:
        return None, 0.0, "missing symbolic target"
        
    equations = []
    for line in before.splitlines():
        if " = " in line:
            left, right = line.split(" = ", 1)
            left = left.strip()
            right = right.strip()
            if not left or not right: continue
            
            mid = len(left) // 2
            op1 = left[:mid]
            op = left[mid:mid+1]
            op2 = left[mid+1:]
            equations.append((op1, op, op2, right))
            
    if len(target_str) < 3:
        return None, 0.0, "invalid target length"
        
    mid = len(target_str) // 2
    target_lhs = (target_str[:mid], target_str[mid:mid+1], target_str[mid+1:])
    
    if not equations:
        return None, 0.0, "no equations found"

    op_chars = set()
    for op1, op, op2, res in equations:
        op_chars.add(op)
    op_chars.add(target_lhs[1])
    
    ordered_digits = []
    for op1, op, op2, res in equations:
        for c in op1 + op2 + res:
            if c not in ordered_digits and c not in op_chars:
                ordered_digits.append(c)
    for c in target_lhs[0] + target_lhs[2]:
        if c not in ordered_digits and c not in op_chars:
            ordered_digits.append(c)
            
    if len(ordered_digits) > 10:
        return None, 0.0, "too many unique characters"
        
    ops_choices = ['+', '-', '*', '//']
    
    for ops_p in itertools.product(ops_choices, repeat=len(op_chars)):
        o_map = dict(zip(op_chars, ops_p))
        
        start_time = time.time()
        def backtrack(char_idx, d_map, used):
            if time.time() - start_time > 1.0: return False
            for op1, op, op2, res in equations:
                if all(c in d_map for c in op1+op2+res):
                    if len(op1) > 1 and d_map[op1[0]] == '0': return False
                    if len(op2) > 1 and d_map[op2[0]] == '0': return False
                    if len(res) > 1 and d_map[res[0]] == '0': return False
                    
                    v1 = int("".join(d_map[c] for c in op1))
                    v2 = int("".join(d_map[c] for c in op2))
                    vr = int("".join(d_map[c] for c in res))
                    
                    o = o_map[op]
                    if o == '+':
                        if v1 + v2 != vr: return False
                    elif o == '-':
                        if v1 - v2 != vr: return False
                    elif o == '*':
                        if v1 * v2 != vr: return False
                    elif o == '//':
                        if v2 == 0 or v1 % v2 != 0 or v1 // v2 != vr: return False

            if char_idx == len(ordered_digits):
                return d_map
                
            c = ordered_digits[char_idx]
            for d in "0123456789":
                if d not in used:
                    d_map[c] = d
                    used.add(d)
                    res_map = backtrack(char_idx + 1, d_map, used)
                    if res_map: return res_map
                    used.remove(d)
                    del d_map[c]
            return False
            
        final_d_map = backtrack(0, {}, set())
        if final_d_map:
            v1 = int("".join(final_d_map[c] for c in target_lhs[0]))
            v2 = int("".join(final_d_map[c] for c in target_lhs[2]))
            o = o_map[target_lhs[1]]
            
            target_val = None
            if o == '+': target_val = v1 + v2
            elif o == '-': target_val = v1 - v2
            elif o == '*': target_val = v1 * v2
            elif o == '//': 
                if v2 != 0 and v1 % v2 == 0:
                    target_val = v1 // v2
            
            if target_val is None:
                continue
                
            inv_d_map = {v: k for k, v in final_d_map.items()}
            target_res_str = str(target_val)
            
            res_enc = []
            valid_enc = True
            for d in target_res_str:
                if d == '-':
                    res_enc.append('-')
                elif d in inv_d_map:
                    res_enc.append(inv_d_map[d])
                else:
                    valid_enc = False
                    break
            
            if valid_enc:
                return "".join(res_enc), 1.0, "exact cryptarithm cipher match"
                
    return None, 0.0, "no valid cryptarithm mapping found"


def bit_feature_bank():
    tiers = []
    
    t0 = [("0", lambda b: 0), ("1", lambda b: 1)]
    for i in range(8):
        t0.append((f"x{i}", lambda b, i=i: int(b[i])))
        t0.append((f"not x{i}", lambda b, i=i: 1 - int(b[i])))
    tiers.append(t0)
    
    t1 = []
    for i in range(8):
        for j in range(i+1, 8):
            t1.append((f"x{i}&x{j}", lambda b, i=i, j=j: int(b[i]) & int(b[j])))
            t1.append((f"x{i}|x{j}", lambda b, i=i, j=j: int(b[i]) | int(b[j])))
            t1.append((f"x{i}^x{j}", lambda b, i=i, j=j: int(b[i]) ^ int(b[j])))
            t1.append((f"~(x{i}^x{j})", lambda b, i=i, j=j: 1 - (int(b[i]) ^ int(b[j]))))
            t1.append((f"~(x{i}&x{j})", lambda b, i=i, j=j: 1 - (int(b[i]) & int(b[j]))))
            t1.append((f"~(x{i}|x{j})", lambda b, i=i, j=j: 1 - (int(b[i]) | int(b[j]))))
    tiers.append(t1)
    
    t2 = []
    for i in range(8):
        for j in range(i+1, 8):
            for k in range(j+1, 8):
                t2.append((f"maj{i}{j}{k}", lambda b, i=i, j=j, k=k: int(int(b[i]) + int(b[j]) + int(b[k]) >= 2)))
                t2.append((f"xor3{i}{j}{k}", lambda b, i=i, j=j, k=k: int(b[i]) ^ int(b[j]) ^ int(b[k])))
    tiers.append(t2)
    
    return tiers


BIT_TIERS = bit_feature_bank()


def solve_bitwise(prompt):
    prompt = str(prompt)
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    target = re.search(r"output for:\s*([01]{8})", prompt, re.I)
    if not pairs or not target:
        return None, 0.0, "missing bit examples or target"

    target_bits = target.group(1)
    int_pairs = [(int(x, 2), int(y, 2)) for x, y in pairs]
    target_int = int(target_bits, 2)

    # 1. Structural / Arithmetic Search
    for K in range(256):
        if all((x + K) % 256 == y for x, y in int_pairs):
            return format((target_int + K) % 256, '08b'), 1.0, f"arithmetic: (x + {K}) % 256"
        if all((x - K) % 256 == y for x, y in int_pairs):
            return format((target_int - K) % 256, '08b'), 1.0, f"arithmetic: (x - {K}) % 256"
        if all((K - x) % 256 == y for x, y in int_pairs):
            return format((K - target_int) % 256, '08b'), 1.0, f"arithmetic: ({K} - x) % 256"
        if all((x ^ K) == y for x, y in int_pairs):
            return format(target_int ^ K, '08b'), 1.0, f"arithmetic: x ^ {K}"
        if all(((x ^ K) + K) % 256 == y for x, y in int_pairs):
            return format(((target_int ^ K) + K) % 256, '08b'), 1.0, f"arithmetic: (x ^ {K}) + {K}"

    for shift in range(1, 8):
        if all(((x << shift) & 255) | (x >> (8 - shift)) == y for x, y in int_pairs):
            res = ((target_int << shift) & 255) | (target_int >> (8 - shift))
            return format(res, '08b'), 1.0, f"rotate left {shift}"
        if all(((x >> shift) | ((x << (8 - shift)) & 255)) == y for x, y in int_pairs):
            res = (target_int >> shift) | ((target_int << (8 - shift)) & 255)
            return format(res, '08b'), 1.0, f"rotate right {shift}"

    if all(int(format(x, '08b')[::-1], 2) == y for x, y in int_pairs):
        return format(target_int, '08b')[::-1], 1.0, "bit reversal"

    # 2. Tiered Logic Gate Search
    out = []
    for pos in range(8):
        expected = tuple(int(y[pos]) for _, y in pairs)
        found_fn = None
        for tier in BIT_TIERS:
            for fn_name, fn in tier:
                if tuple(fn(x) for x, _ in pairs) == expected:
                    found_fn = fn
                    break
            if found_fn:
                break
                
        if not found_fn:
            return None, 0.0, f"no matching rule for bit position {pos}"
            
        out.append(str(found_fn(target_bits)))

    return "".join(out), 0.95, "exact tiered boolean logic"


SOLVERS = {
    "bitwise": solve_bitwise,
    "cipher": solve_cipher,
    "gravity": solve_gravity,
    "roman": solve_roman,
    "unit_conversion": solve_unit_conversion,
    "symbolic": solve_symbolic_substitution,
}


def deterministic_answer(prompt):
    """Main solver entrypoint classifying and solving the prompt."""
    task_type = classify_prompt(prompt)
    solver = SOLVERS.get(task_type)
    if solver is None:
        return None, 0.0, "unknown task"
    return solver(prompt)


def verify_deterministic(prompt, answer) -> tuple:
    """Verifies a prediction against the programmatic solver, returning match flag, confidence, and reason."""
    predicted, confidence, reason = deterministic_answer(prompt)
    if predicted is None:
        return None, confidence, reason
    return normalize_answer(predicted) == normalize_answer(answer), confidence, reason
