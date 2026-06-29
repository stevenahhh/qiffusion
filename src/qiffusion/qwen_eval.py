from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass

from qiffusion.qwen_bridge import (
    DEFAULT_OLLAMA_MODEL,
    PREFERRED_MODEL_ID,
    QwenBridgeReport,
    TaskResult,
    ollama_executable,
    ollama_has_qwen,
)
from qiffusion.qwen_tasks import (
    CODING_TASKS,
    CodingTask,
    run_task_smoke,
    task_prompt,
)


@dataclass(frozen=True, slots=True)
class EvalProgress:
    model: str
    task_results: tuple[TaskResult, ...]
    generated: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskFailure:
    task: CodingTask
    run: int
    statuses: tuple[str, str]
    message: str


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
    return (True, code)


def qwen_eval(model: str = DEFAULT_OLLAMA_MODEL, runs: int = 1) -> QwenBridgeReport:
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
            "runs": runs,
        }
    task_results: list[TaskResult] = []
    generated: list[str] = []
    fixture_results = []
    for run_number in range(1, runs + 1):
        for task in CODING_TASKS:
            ok, response = run_ollama_fixture(model, task_prompt(task))
            if not ok:
                failure = TaskFailure(task, run_number, ("fail", "not_run"), response)
                return task_failure(progress(model, task_results, generated), failure)
            parsed, code = extract_code(response)
            if not parsed:
                failure = TaskFailure(task, run_number, ("fail", "not_run"), code)
                report = task_failure(progress(model, task_results, generated), failure)
                report["raw_response"] = response
                return report
            generated.append(f"# run {run_number} {task.name}\n{code}")
            smoke_ok, smoke_message, task_fixtures = run_task_smoke(code, task)
            fixture_results.extend(task_fixtures)
            if not smoke_ok:
                failure = TaskFailure(task, run_number, ("pass", "fail"), smoke_message)
                return task_failure(progress(model, task_results, generated), failure)
            task_results.append({"name": task.name, "run": run_number, "status": "pass", "generated_code": code})
    return {
        "backend": "qwen_bridge",
        "model_id": PREFERRED_MODEL_ID,
        "status": "available",
        "engine": "ollama",
        "notes": ["local Ollama Qwen independent coding tasks passed"],
        "fixtures_status": "pass",
        "code_smoke_status": "pass",
        "candidate_source": f"ollama:{model}",
        "coding_capability_claim": True,
        "fixture_results": fixture_results,
        "generated_code": "\n\n".join(generated),
        "runs": runs,
        "task_results": task_results,
    }


def progress(model: str, task_results: list[TaskResult], generated: list[str]) -> EvalProgress:
    return EvalProgress(model, tuple(task_results), tuple(generated))


def task_failure(progress: EvalProgress, failure: TaskFailure) -> QwenBridgeReport:
    fixtures_status, code_smoke_status = failure.statuses
    task_results = [
        *progress.task_results,
        {"name": failure.task.name, "run": failure.run, "status": "fail", "error": failure.message},
    ]
    report = eval_failure(progress.model, fixtures_status, code_smoke_status, failure.message)
    report["generated_code"] = "\n\n".join(progress.generated)
    report["task_results"] = task_results
    return report


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
