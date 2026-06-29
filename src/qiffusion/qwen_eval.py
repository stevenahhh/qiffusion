from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from qiffusion.qwen_bridge import (
    DEFAULT_OLLAMA_MODEL,
    FixtureResult,
    PREFERRED_MODEL_ID,
    QwenBridgeReport,
    TaskResult,
    ollama_has_qwen,
)
from qiffusion.qwen_file_tasks import (
    FILE_EDIT_TASKS,
    file_edit_prompt,
    run_file_edit_smoke,
)
from qiffusion.qwen_ollama import extract_code, run_ollama_fixture
from qiffusion.qwen_repair_tasks import REPAIR_TASKS, repair_prompt, run_repair_smoke
from qiffusion.qwen_tasks import (
    CODING_TASKS,
    run_task_smoke,
    task_prompt,
)

SmokeRunner: TypeAlias = Callable[[str], tuple[bool, str, list[FixtureResult]]]


@dataclass(frozen=True, slots=True)
class EvalProgress:
    model: str
    task_results: tuple[TaskResult, ...]
    generated: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskFailure:
    task_name: str
    run: int
    statuses: tuple[str, str]
    message: str


@dataclass(frozen=True, slots=True)
class PromptEval:
    name: str
    label: str
    prompt: str
    smoke: SmokeRunner


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
    fixture_results: list[FixtureResult] = []
    for run_number in range(1, runs + 1):
        for prompt_eval in prompt_evals():
            report = run_prompt_eval(model, run_number, prompt_eval, task_results, generated, fixture_results)
            if report is not None:
                return report
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


def prompt_evals() -> list[PromptEval]:
    evals: list[PromptEval] = []
    for task in CODING_TASKS:
        evals.append(PromptEval(task.name, task.name, task_prompt(task), lambda code, task=task: run_task_smoke(code, task)))
    for task in FILE_EDIT_TASKS:
        evals.append(
            PromptEval(
                task.name,
                f"file-edit {task.name}",
                file_edit_prompt(task),
                lambda code, task=task: run_file_edit_smoke(code, task),
            )
        )
    for task in REPAIR_TASKS:
        evals.append(
            PromptEval(
                task.name,
                f"repair {task.name}",
                repair_prompt(task),
                lambda code, task=task: run_repair_smoke(code, task),
            )
        )
    return evals


def run_prompt_eval(
    model: str,
    run_number: int,
    prompt_eval: PromptEval,
    task_results: list[TaskResult],
    generated: list[str],
    fixture_results: list[FixtureResult],
) -> QwenBridgeReport | None:
    ok, response = run_ollama_fixture(model, prompt_eval.prompt)
    if not ok:
        failure = TaskFailure(prompt_eval.name, run_number, ("fail", "not_run"), response)
        return task_failure(progress(model, task_results, generated), failure)
    parsed, code = extract_code(response)
    if not parsed:
        failure = TaskFailure(prompt_eval.name, run_number, ("fail", "not_run"), code)
        report = task_failure(progress(model, task_results, generated), failure)
        report["raw_response"] = response
        return report
    generated.append(f"# run {run_number} {prompt_eval.label}\n{code}")
    smoke_ok, smoke_message, task_fixtures = prompt_eval.smoke(code)
    fixture_results.extend(task_fixtures)
    if not smoke_ok:
        failure = TaskFailure(prompt_eval.name, run_number, ("pass", "fail"), smoke_message)
        return task_failure(progress(model, task_results, generated), failure)
    task_results.append({"name": prompt_eval.name, "run": run_number, "status": "pass", "generated_code": code})
    return None


def progress(model: str, task_results: list[TaskResult], generated: list[str]) -> EvalProgress:
    return EvalProgress(model, tuple(task_results), tuple(generated))


def task_failure(progress: EvalProgress, failure: TaskFailure) -> QwenBridgeReport:
    fixtures_status, code_smoke_status = failure.statuses
    task_results = [
        *progress.task_results,
        {"name": failure.task_name, "run": failure.run, "status": "fail", "error": failure.message},
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
