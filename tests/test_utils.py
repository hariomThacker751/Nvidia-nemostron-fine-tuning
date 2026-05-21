import pytest
from src.utils import clean_answer, is_numeric, normalize_answer, answers_match, extract_boxed_answer

def test_clean_answer():
    assert clean_answer("  hello   world.  ") == "hello world"
    assert clean_answer("42.0\n") == "42.0"
    assert clean_answer("ans.") == "ans"


def test_is_numeric():
    assert is_numeric("123") is True
    assert is_numeric("12.34") is True
    assert is_numeric("-5e-3") is True
    assert is_numeric("abc") is False
    assert is_numeric(None) is False


def test_normalize_answer():
    # LaTeX normalizations
    assert normalize_answer("\\frac{3}{4}") == "0.75"
    assert normalize_answer("2.000") == "2"
    assert normalize_answer("1/2") == "0.5"
    
    # Text tag stripping
    assert normalize_answer("42\\text{ meters}") == "42"
    
    # SymPy evaluations
    assert normalize_answer("sqrt(16)") == "4"
    assert normalize_answer("3 * 5") == "15"


def test_answers_match():
    # Exact Match
    assert answers_match("abc", "abc") is True
    
    # Normalized Match
    assert answers_match("\\frac{1}{2}", "0.5") is True
    assert answers_match("2.00", "2") is True
    
    # Tolerance Match
    assert answers_match("10.005", "10.01", tol=1e-2) is True
    assert answers_match("10.05", "10.01", tol=1e-2) is False


def test_extract_boxed_answer():
    text = "The solution leads to \\boxed{42}."
    assert extract_boxed_answer(text) == "42"
    
    text_multi = "First we find \\boxed{10}, then finally \\boxed{0.75}."
    assert extract_boxed_answer(text_multi) == "0.75"
    
    assert extract_boxed_answer("No boxed content.") is None
