from __future__ import annotations

from qiffusion.qwen_eval import extract_code


def test_extract_code_repairs_single_trailing_json_brace() -> None:
    parsed, code = extract_code('{"code": "def add(a, b):\\n    return a + b}"}')

    assert parsed is True
    assert code == "def add(a, b):\n    return a + b"


def test_extract_code_keeps_valid_python_dict_brace() -> None:
    parsed, code = extract_code('{"code": "def build():\\n    return {\\u0027ok\\u0027: True}"}')

    assert parsed is True
    assert code == "def build():\n    return {'ok': True}"
