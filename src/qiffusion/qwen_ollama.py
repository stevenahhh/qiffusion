from __future__ import annotations

import ast
import json
import os
import subprocess
import urllib.error
import urllib.request

from qiffusion.qwen_bridge import ollama_executable


def run_ollama_fixture(model: str, prompt: str) -> tuple[bool, str]:
    http_ok, http_response = run_ollama_http_fixture(model, prompt)
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
        prompt,
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


def run_ollama_http_fixture(model: str, prompt: str) -> tuple[bool, str]:
    if os.environ.get("QIFFUSION_DISABLE_OLLAMA_HTTP") == "1":
        return (False, "ollama HTTP disabled")
    host = os.environ.get("QIFFUSION_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0},
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
    return (True, clean_code_value(code))


def clean_code_value(code: str) -> str:
    candidate = code.strip()
    if not candidate.endswith("}"):
        return candidate
    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        if exc.msg != "unmatched '}'":
            return candidate
    else:
        return candidate
    trimmed = candidate[:-1].rstrip()
    try:
        ast.parse(trimmed)
    except SyntaxError:
        return candidate
    return trimmed
