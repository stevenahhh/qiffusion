from __future__ import annotations

import ast
from dataclasses import dataclass
from types import FunctionType
from typing import Final

from qiffusion.qwen_bridge import FixtureResult
from qiffusion.qwen_tasks import FixtureCase, JsonValue, make_namespace


@dataclass(frozen=True, slots=True)
class RepairTask:
    name: str
    source: str
    failure: str
    cases: tuple[FixtureCase, ...]


REPAIR_TASKS: Final = (
    RepairTask(
        name="slugify_title",
        source=(
            "def slugify_title(title):\n"
            "    \"\"\"Return a URL slug for a title.\"\"\"\n"
            "    return title.lower().replace(' ', '-')\n"
        ),
        failure=(
            "The function leaves leading/trailing hyphens and repeated hyphens when a title has "
            "outer, repeated, tab, or newline whitespace. Fix it by splitting on arbitrary "
            "whitespace, lowercasing words, and joining them with one hyphen."
        ),
        cases=(
            FixtureCase(("Hello World",), "hello-world"),
            FixtureCase(("  Multiple   Spaces ",), "multiple-spaces"),
            FixtureCase(("Tabs\tWork",), "tabs-work"),
            FixtureCase(("Line\nBreak",), "line-break"),
        ),
    ),
)


def repair_prompt(task: RepairTask) -> str:
    checks = "; ".join(
        f"{task.name}({', '.join(repr(arg) for arg in case.args)}) == {case.expected!r}"
        for case in task.cases
    )
    return (
        "Return JSON only with one key named code. The code value must be the full fixed Python file. "
        "Do not return a diff. Do not use markdown or explanations. "
        "Keep the existing public function name and do not add imports. "
        f"Broken file:\n{task.source}\n"
        f"Bug report: {task.failure} "
        f"The fixed file must satisfy these tests: {checks}."
    )


def run_repair_smoke(code: str, task: RepairTask) -> tuple[bool, str, list[FixtureResult]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return (False, f"fixed file is not Python: {exc.msg}", [])
    blocked = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
    if any(isinstance(node, blocked) for node in ast.walk(tree)):
        return (False, "fixed file uses blocked syntax for the repair fixture", [])
    namespace = make_namespace()
    try:
        exec(compile(tree, "<qwen-repair-fixture>", "exec"), namespace)
    except Exception as exc:  # noqa: BROAD_EXCEPT_OK
        return (False, f"fixed file raised during load: {exc}", [])
    candidate = namespace.get(task.name)
    if not isinstance(candidate, FunctionType):
        return (False, f"fixed file is missing function: {task.name}", [])
    return run_repair_cases(candidate, task)


def run_repair_cases(candidate: FunctionType, task: RepairTask) -> tuple[bool, str, list[FixtureResult]]:
    results: list[FixtureResult] = []
    for case in task.cases:
        try:
            observed = candidate(*case.args)
        except Exception as exc:  # noqa: BROAD_EXCEPT_OK
            results.append({"name": task.name, "status": "fail", "error": str(exc)})
            return (False, f"{task.name} raised during repair smoke check: {exc}", results)
        if observed != case.expected:
            error = f"expected {case.expected!r}, got {observed!r}"
            results.append({"name": task.name, "status": "fail", "error": error})
            return (False, f"{task.name} returned an incorrect repair result: {error}", results)
        results.append({"name": task.name, "status": "pass"})
    return (True, "repair smoke passed", results)
