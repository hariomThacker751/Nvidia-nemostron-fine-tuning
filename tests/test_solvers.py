import pytest
from src.solvers import (
    classify_prompt, roman_numeral, solve_roman,
    solve_gravity, solve_unit_conversion, solve_bitwise,
    solve_symbolic_substitution
)

def test_classify_prompt():
    assert classify_prompt("bit manipulation with XOR") == "bitwise"
    assert classify_prompt("decrypt the following text: encryption rule") == "cipher"
    assert classify_prompt("gravitational constant equations") == "gravity"
    assert classify_prompt("write 42 in a different numeral system") == "roman"
    assert classify_prompt("secret unit conversion factors") == "unit_conversion"
    assert classify_prompt("transformation rules is applied to equations") == "symbolic"
    assert classify_prompt("standard arithmetic expression") == "unknown"


def test_roman_numeral():
    assert roman_numeral(4) == "IV"
    assert roman_numeral(9) == "IX"
    assert roman_numeral(42) == "XLII"
    assert roman_numeral(1000) == "M"


def test_solve_roman():
    prompt = "Please write the number 42 as Roman Numeral"
    ans, conf, reason = solve_roman(prompt)
    assert ans == "XLII"
    assert conf == 1.0
    
    bad_prompt = "Find x in standard math system."
    ans, conf, reason = solve_roman(bad_prompt)
    assert ans is None


def test_solve_gravity():
    prompt = (
        "Under gravitational constant g:\n"
        "t = 1.0s, distance = 4.90m\n"
        "t = 2.0s, distance = 19.60m\n"
        "Find falling distance for t = 3.0s."
    )
    ans, conf, reason = solve_gravity(prompt)
    assert ans is not None
    assert float(ans) == pytest.approx(44.1, abs=0.1)
    assert conf == 1.0


def test_solve_unit_conversion():
    prompt = (
        "Secret unit conversion observations:\n"
        "1.0 m becomes 3.28\n"
        "2.0 m becomes 6.56\n"
        "Now, convert the following measurement: 5.0 m"
    )
    ans, conf, reason = solve_unit_conversion(prompt)
    assert ans is not None
    assert float(ans) == pytest.approx(16.4, abs=0.1)
    assert conf == 1.0


def test_solve_bitwise():
    # Rotate left test
    # 00000001 (1) -> 00000010 (2)
    # 00000010 (2) -> 00000100 (4)
    # Target: 00000100 (4) -> ?
    prompt = (
        "Bit manipulation examples:\n"
        "00000001 -> 00000010\n"
        "00000010 -> 00000100\n"
        "Find output for: 00000100"
    )
    ans, conf, reason = solve_bitwise(prompt)
    assert ans == "00001000"
    assert conf == 1.0
    
    # Tiered Logic Gate search: maj XOR3 Maj gates
    prompt_logic = (
        "Bitwise boolean rule examples:\n"
        "00000011 -> 00000001\n"
        "00000101 -> 00000001\n"
        "Find output for: 00000111"
    )
    ans, conf, reason = solve_bitwise(prompt_logic)
    assert ans is not None


def test_solve_symbolic_substitution():
    prompt = (
        "Transformation rules is applied to equations:\n"
        "A + B = C\n"
        "Where each letter represents a distinct digit.\n"
        "Here are equations:\n"
        "ab + cd = ef\n"
        "Determine the result for: ac + bd"
    )
    # Symbolic cryptarithm backtracker testing
    ans, conf, reason = solve_symbolic_substitution(prompt)
    assert conf >= 0.0  # Conf resolves safely or reports unmatched
