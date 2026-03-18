"""Tests for answer extraction and math normalization."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infrastructure.generation import (
    extract_math_answer,
    extract_mcq_answer,
    normalize_math_answer,
    check_answer_correct,
    find_horl,
)


class TestExtractMathAnswer:
    def test_boxed_simple(self):
        assert extract_math_answer(r'The answer is \boxed{42}') == '42'

    def test_boxed_fraction(self):
        assert extract_math_answer(r'\boxed{\frac{3}{4}}') == r'\frac{3}{4}'

    def test_boxed_negative(self):
        assert extract_math_answer(r'\boxed{-7}') == '-7'

    def test_multiple_boxed_takes_last(self):
        text = r'First attempt \boxed{10}, wait \boxed{42}'
        assert extract_math_answer(text) == '42'

    def test_answer_is_pattern(self):
        assert extract_math_answer('Therefore, the answer is 42') == '42'

    def test_no_answer(self):
        assert extract_math_answer('I am not sure about this problem') is None

    def test_boxed_with_nested_braces(self):
        assert extract_math_answer(r'\boxed{2^{10}}') == '2^{10}'


class TestExtractMCQ:
    def test_answer_is_A(self):
        assert extract_mcq_answer('The answer is (A)') == 'A'

    def test_answer_is_B_no_parens(self):
        assert extract_mcq_answer('The answer is B') == 'B'

    def test_therefore_pattern(self):
        assert extract_mcq_answer('Therefore, the answer is (C)') == 'C'

    def test_last_letter_fallback(self):
        text = 'After analysis... the correct choice is D.'
        result = extract_mcq_answer(text)
        assert result == 'D'


class TestNormalizeMath:
    def test_remove_dollar(self):
        assert normalize_math_answer('$42$') == '42'

    def test_remove_commas(self):
        assert normalize_math_answer('1,000') == '1000'

    def test_fraction(self):
        assert normalize_math_answer(r'\frac{1}{2}') == '0.5'

    def test_trailing_zeros(self):
        assert normalize_math_answer('42.0') == '42'

    def test_percentage(self):
        assert normalize_math_answer('50%') == '50'


class TestCheckCorrect:
    def test_math_exact(self):
        assert check_answer_correct('42', '42', 'math') is True

    def test_math_normalized(self):
        assert check_answer_correct('$42$', '42', 'math') is True

    def test_math_wrong(self):
        assert check_answer_correct('41', '42', 'math') is False

    def test_mcq_correct(self):
        assert check_answer_correct('B', 'B', 'gpqa') is True

    def test_mcq_case_insensitive(self):
        assert check_answer_correct('b', 'B', 'gpqa') is True

    def test_none_extracted(self):
        assert check_answer_correct(None, '42', 'math') is False


class TestFindHORL:
    def test_finds_boxed_answer(self):
        text = r'Let me think... first \boxed{10} but actually \boxed{42}'
        pos = find_horl(text, '42', 'math')
        assert pos is not None
        assert pos > 0

    def test_no_correct_answer(self):
        text = r'Let me think... \boxed{10}'
        pos = find_horl(text, '42', 'math')
        assert pos is None

    def test_mcq_horl(self):
        text = 'Thinking... the answer is B'
        pos = find_horl(text, 'B', 'gpqa')
        assert pos is not None


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
