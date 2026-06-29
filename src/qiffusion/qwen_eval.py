from __future__ import annotations

import ast
import json
import subprocess
from types import FunctionType
from typing import Final

from qiffusion.qwen_bridge import (
    DEFAULT_OLLAMA_MODEL,
    PREFERRED_MODEL_ID,
    QwenBridgeReport,
    ollama_executable,
    ollama_has_qwen,
)

CODING_FIXTURE_PROMPT: Final = (
    "Return JSON only with one key named code. The code value must be Python source. "
    "It must define exactly this function: def add(a, b): return a + b. "
    "Do not use markdown, JavaScript, explanations, or tests."
)


def run_ollama_fixture(model: str) -> tuple[bool, str]:
    executable = ollama_executable()
    if executable is None:
        return (False, "ollama executable not found")
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


def run_code_smoke(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return (False, f"generated code is not Python: {exc.msg}")
    allowed = (
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.BinOp,
        ast.Add,
        ast.Name,
        ast.Load,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return (False, "generated code uses unsupported syntax for the smoke fixture")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "add":
        return (False, "generated code must define exactly one add function")
    namespace: dict[str, object] = {"__builtins__": {}}
    try:
        exec(compile(tree, "<qwen-fixture>", "exec"), namespace)
    except Exception as exc:
        return (False, f"generated code raised during load: {exc}")
    candidate = namespace.get("add")
    if not isinstance(candidate, FunctionType):
        return (False, "add is not a function")
    try:
        checks = (candidate(2, 3) == 5, candidate(-1, 4) == 3)
    except Exception as exc:
        return (False, f"add raised during smoke check: {exc}")
    if not all(checks):
        return (False, "add returned an incorrect result")
    return (True, "code smoke passed")


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
    smoke_ok, smoke_message = run_code_smoke(code)
    if not smoke_ok:
        report = eval_failure(model, "pass", "fail", smoke_message)
        report["generated_code"] = code
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
