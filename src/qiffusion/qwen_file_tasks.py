from __future__ import annotations

import ast
from dataclasses import dataclass
from types import FunctionType
from typing import Final, TypeAlias

from qiffusion.qwen_bridge import FixtureResult
from qiffusion.qwen_tasks import JsonValue, make_namespace

SourceText: TypeAlias = str


@dataclass(frozen=True, slots=True)
class FileEditCase:
    args: tuple[JsonValue, ...]
    expected: JsonValue


@dataclass(frozen=True, slots=True)
class FileEditTask:
    name: str
    source: SourceText
    request: str
    cases: tuple[FileEditCase, ...]


FILE_EDIT_TASKS: Final = (
    FileEditTask(
        name="classify_temperature",
        source=(
            "def classify_temperature(celsius):\n"
            "    if celsius > 30:\n"
            "        return 'hot'\n"
            "    return 'cold'\n"
        ),
        request=(
            "Edit classify_temperature with these exact ordered branches: "
            "if celsius <= 0 return 'freezing'; else if celsius <= 20 return 'cold'; "
            "else if celsius <= 30 return 'warm'; otherwise return 'hot'."
        ),
        cases=(
            FileEditCase((-5,), "freezing"),
            FileEditCase((10,), "cold"),
            FileEditCase((25,), "warm"),
            FileEditCase((40,), "hot"),
        ),
    ),
)


def file_edit_prompt(task: FileEditTask) -> str:
    checks = "; ".join(
        f"{task.name}({', '.join(repr(arg) for arg in case.args)}) == {case.expected!r}"
        for case in task.cases
    )
    return (
        "Return JSON only with one key named code. The code value must be the full edited Python file. "
        "Do not return a diff. Do not use markdown or explanations. "
        "Keep the existing public function name and do not add imports. "
        f"Current file:\n{task.source}\n"
        f"Requested edit: {task.request} "
        f"The edited file must satisfy these checks: {checks}."
    )


def run_file_edit_smoke(code: str, task: FileEditTask) -> tuple[bool, str, list[FixtureResult]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return (False, f"edited file is not Python: {exc.msg}", [])
    blocked = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
    if any(isinstance(node, blocked) for node in ast.walk(tree)):
        return (False, "edited file uses blocked syntax for the file-edit fixture", [])
    namespace = make_namespace()
    try:
        exec(compile(tree, "<qwen-file-edit-fixture>", "exec"), namespace)
    except Exception as exc:  # noqa: BROAD_EXCEPT_OK
        return (False, f"edited file raised during load: {exc}", [])
    candidate = namespace.get(task.name)
    if not isinstance(candidate, FunctionType):
        return (False, f"edited file is missing function: {task.name}", [])
    results: list[FixtureResult] = []
    for case in task.cases:
        try:
            observed = candidate(*case.args)
        except Exception as exc:  # noqa: BROAD_EXCEPT_OK
            results.append({"name": task.name, "status": "fail", "error": str(exc)})
            return (False, f"{task.name} raised during file-edit smoke check: {exc}", results)
        if observed != case.expected:
            error = f"expected {case.expected!r}, got {observed!r}"
            results.append({"name": task.name, "status": "fail", "error": error})
            return (False, f"{task.name} returned an incorrect file-edit result: {error}", results)
        results.append({"name": task.name, "status": "pass"})
    return (True, "file-edit smoke passed", results)
