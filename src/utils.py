import re
import logging
from decimal import Decimal
import sympy as sp
from fractions import Fraction

# ─── LOGGING CONFIGURATION ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("NemotronPipeline")


# ─── CORE CLEANERS & VERIFIERS ─────────────────────────────────────────

def clean_answer(answer) -> str:
    """Strip whitespace, normalize spacing, and remove training dots."""
    text = str(answer).strip()
    text = re.sub(r"\s+", " ", text)
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def is_numeric(s: str) -> bool:
    """Safely check if a string can be converted to float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def normalize_answer(ans: str) -> str:
    """
    Standardize mathematically equivalent answers into a canonical form.
    Handles LaTeX formatting (e.g. \\frac, \\sqrt, \\times), SymPy evaluation,
    and trailing decimal zeroes to match gold standards cleanly.
    """
    if ans is None:
        return ""
    
    # Pre-clean string spacing
    ans = clean_answer(ans)
    
    # Remove LaTeX-specific symbols and format commands
    ans = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', ans)
    ans = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', ans)
    ans = ans.replace('\\times', '*').replace('\\cdot', '*')
    ans = re.sub(r'\\text\{[^}]+\}', '', ans)
    
    # Try direct SymPy simplification and float evaluation
    try:
        val = float(sp.sympify(ans))
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}".rstrip("0").rstrip(".")
    except Exception:
        pass
    
    # Decimal fallback (e.g. 2.00 -> 2)
    try:
        dec = Decimal(ans)
        if "." in ans:
            return format(dec.normalize(), "f").rstrip("0").rstrip(".")
    except Exception:
        pass
        
    return ans.lower().strip()


def answers_match(pred: str, gold: str, tol=1e-2) -> bool:
    """
    Check if a predicted answer matches a gold standard.
    Includes exact string comparison, normalized comparison, and numeric tolerance.
    """
    if pred is None or gold is None:
        return False
        
    pred_str = str(pred).strip()
    gold_str = str(gold).strip()
    
    # Exact matching
    if pred_str.lower() == gold_str.lower():
        return True
        
    # Standard normalization matching
    norm_pred = normalize_answer(pred_str)
    norm_gold = normalize_answer(gold_str)
    if norm_pred == norm_gold:
        return True
        
    # Decimal/Floating point tolerance check
    try:
        if is_numeric(norm_pred) and is_numeric(norm_gold):
            return abs(float(norm_pred) - float(norm_gold)) <= tol
    except Exception:
        pass
        
    return False


def extract_boxed_answer(text: str) -> str | None:
    """
    Extract the contents of the final \\boxed{...} tag from a string.
    Returns None if no boxed expression is resolved.
    """
    if not text:
        return None
    matches = re.findall(r'\\boxed\{([^}]+)\}', str(text))
    if matches:
        return clean_answer(matches[-1])
    return None
