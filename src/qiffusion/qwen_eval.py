from __future__ import annotations

import ast
import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import FunctionType
from typing import Final, NotRequired, TypedDict

from qiffusion.qwen_bridge import (
    DEFAULT_OLLAMA_MODEL,
    PREFERRED_MODEL_ID,
    QwenBridgeReport,
    ollama_executable,
    ollama_has_qwen,
)

CODING_FIXTURE_PROMPT: Final = (
    "Return JSON only with one key named code. The code value must be Python source. "
    "Define exactly these Python functions and no imports: "
    "add(a, b) returns a + b; "
    "count_even(values) returns the number of even integers in values; "
    "reverse_words(text) returns the whitespace-separated words in reverse order joined by single spaces. "
    "Examples: count_even([1, 2, 4, 5]) == 2; reverse_words('one two three') == 'three two one'. "
    "Do not print, import, use markdown, JavaScript, explanations, or tests."
)


class FixtureResult(TypedDict):
    name: str
    status: str
    error: NotRequired[str]


@dataclass(frozen=True, slots=True)
class FixtureCase:
    name: str
    args: tuple[object, ...]
    expected: object


FIXTURE_CASES: Final = (
    FixtureCase("add", (2, 3), 5),
    FixtureCase("add", (-1, 4), 3),
    FixtureCase("count_even", ([1, 2, 4, 5],), 2),
    FixtureCase("count_even", ([],), 0),
    FixtureCase("reverse_words", ("one two three",), "three two one"),
    FixtureCase("reverse_words", ("  solo ",), "solo"),
)
REQUIRED_FUNCTIONS: Final = tuple(dict.fromkeys(case.name for case in FIXTURE_CASES))


def run_ollama_fixture(model: str) -> tuple[bool, str]:
    http_ok, http_response = run_ollama_http_fixture(model)
    if http_ok:
        return (True, http_response)
    executable = ollama_executable()
    if executable is None:
        return (False, http_response)
    command = [
        executable,
        "run",
        model,
        "--format",
        "json",
        "--think",
        "false",
        CODING_FIXTURE_PROMPT,
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=120.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (False, "ollama run timed out")
    except OSError as exc:
        return (False, f"ollama run failed: {exc}")
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"ollama returned {result.returncode}"
        return (False, message)
    return (True, result.stdout)


def run_ollama_http_fixture(model: str) -> tuple[bool, str]:
    if os.environ.get("QIFFUSION_DISABLE_OLLAMA_HTTP") == "1":
        return (False, "ollama HTTP disabled")
    host = os.environ.get("QIFFUSION_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    payload = json.dumps(
        {
            "model": model,
            "prompt": CODING_FIXTURE_PROMPT,
            "stream": False,
            "format": "json",
            "think": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, json.JSONDecodeError, urllib.error.URLError) as exc:
        return (False, f"ollama HTTP failed: {exc}")
    generated = body.get("response") if isinstance(body, dict) else None
    if not isinstance(generated, str) or generated == "":
        return (False, "ollama HTTP response lacked generated text")
    return (True, generated)


def extract_code(response: str) -> tuple[bool, str]:
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end <= start:
        return (False, "missing JSON object")
    try:
        payload = json.loads(response[start : end + 1])
    except json.JSONDecodeError as exc:
        return (False, f"invalid JSON: {exc}")
    if not isinstance(payload, dict):
        return (False, "JSON payload is not an object")
    code = payload.get("code")
    if not isinstance(code, str) or code.strip() == "":
        return (False, "JSON payload lacks non-empty code string")
    return (True, code)


def run_code_smoke(code: str) -> tuple[bool, str, list[FixtureResult]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return (False, f"generated code is not Python: {exc.msg}", [])
    blocked = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
    if any(isinstance(node, blocked) for node in ast.walk(tree)):
        return (False, "generated code uses blocked syntax for the smoke fixture", [])
    namespace: dict[str, object] = {
        "__builtins__": {
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "range": range,
            "reversed": reversed,
            "str": str,
            "sum": sum,
        }
    }
    try:
        exec(compile(tree, "<qwen-fixture>", "exec"), namespace)
    except Exception as exc:
        return (False, f"generated code raised during load: {exc}", [])
    missing = [name for name in REQUIRED_FUNCTIONS if not isinstance(namespace.get(name), FunctionType)]
    if missing:
        return (False, f"generated code is missing functions: {', '.join(missing)}", [])
    results: list[FixtureResult] = []
    for case in FIXTURE_CASES:
        candidate = namespace[case.name]
        if not isinstance(candidate, FunctionType):
            return (False, f"{case.name} is not a function", results)
        try:
            observed = candidate(*case.args)
        except Exception as exc:
            results.append({"name": case.name, "status": "fail", "error": str(exc)})
            return (False, f"{case.name} raised during smoke check: {exc}", results)
        if observed != case.expected:
            error = f"expected {case.expected!r}, got {observed!r}"
            results.append({"name": case.name, "status": "fail", "error": error})
            return (False, f"{case.name} returned an incorrect result: {error}", results)
        results.append({"name": case.name, "status": "pass"})
    return (True, "code smoke passed", results)


def qwen_eval(model: str = DEFAULT_OLLAMA_MODEL) -> QwenBridgeReport:
    if not ollama_has_qwen(model):
        return {
            "backend": "qwen_bridge",
            "model_id": PREFERRED_MODEL_ID,
            "status": "prerequisite_missing",
            "engine": "ollama",
            "notes": [f"local Ollama model not found: {model}"],
            "fixtures_status": "not_run",
            "code_smoke_status": "not_run",
            "candidate_source": "none",
            "coding_capability_claim": False,
        }
    ok, response = run_ollama_fixture(model)
    if not ok:
        return eval_failure(model, "fail", "not_run", response)
    parsed, code = extract_code(response)
    if not parsed:
        report = eval_failure(model, "fail", "not_run", code)
        report["raw_response"] = response
        return report
    smoke_ok, smoke_message, fixture_results = run_code_smoke(code)
    if not smoke_ok:
        report = eval_failure(model, "pass", "fail", smoke_message)
        report["generated_code"] = code
        report["fixture_results"] = fixture_results
        return report
    return {
        "backend": "qwen_bridge",
        "model_id": PREFERRED_MODEL_ID,
        "status": "available",
        "engine": "ollama",
        "notes": ["local Ollama Qwen coding fixture passed"],
        "fixtures_status": "pass",
        "code_smoke_status": "pass",
        "candidate_source": f"ollama:{model}",
        "coding_capability_claim": True,
        "fixture_results": fixture_results,
        "generated_code": code,
    }


def eval_failure(model: str, fixtures_status: str, code_smoke_status: str, message: str) -> QwenBridgeReport:
    return {
        "backend": "qwen_bridge",
        "model_id": PREFERRED_MODEL_ID,
        "status": "available",
        "engine": "ollama",
        "notes": ["local Ollama Qwen model found, but coding fixture did not pass"],
        "fixtures_status": fixtures_status,
        "code_smoke_status": code_smoke_status,
        "candidate_source": f"ollama:{model}",
        "coding_capability_claim": False,
        "smoke_error": message,
    }
