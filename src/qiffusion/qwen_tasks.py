from __future__ import annotations

import ast
from dataclasses import dataclass
from types import FunctionType
from typing import Final, TypeAlias

from qiffusion.qwen_bridge import FixtureResult

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"]


@dataclass(frozen=True, slots=True)
class FixtureCase:
    args: tuple[JsonValue, ...]
    expected: JsonValue


@dataclass(frozen=True, slots=True)
class CodingTask:
    name: str
    signature: str
    behavior: str
    examples: tuple[str, ...]
    cases: tuple[FixtureCase, ...]


CODING_TASKS: Final = (
    CodingTask(
        name="add",
        signature="def add(a, b):",
        behavior="return a + b",
        examples=("add(2, 3) == 5", "add(-1, 4) == 3"),
        cases=(FixtureCase((2, 3), 5), FixtureCase((-1, 4), 3)),
    ),
    CodingTask(
        name="count_even",
        signature="def count_even(values):",
        behavior="return the number of even integers in values",
        examples=("count_even([1, 2, 4, 5]) == 2", "count_even([]) == 0"),
        cases=(FixtureCase(([1, 2, 4, 5],), 2), FixtureCase(([],), 0)),
    ),
    CodingTask(
        name="reverse_words",
        signature="def reverse_words(text):",
        behavior=(
            "return whitespace-separated words in reverse order joined by single spaces; "
            "use text.split() with no separator so repeated or outer whitespace is ignored; "
            "preserving the original word order is incorrect"
        ),
        examples=(
            "reverse_words('one two three') == 'three two one'",
            "reverse_words('alpha beta') == 'beta alpha'",
            "reverse_words('  solo ') == 'solo'",
        ),
        cases=(FixtureCase(("one two three",), "three two one"), FixtureCase(("  solo ",), "solo")),
    ),
)


def task_prompt(task: CodingTask) -> str:
    return (
        "Return JSON only with one key named code. The code value must be Python source. "
        f"Define exactly one Python function with this signature: {task.signature} "
        f"It must {task.behavior}. "
        f"Examples: {'; '.join(task.examples)}. "
        "Do not print, import, define extra functions, use markdown, JavaScript, explanations, or tests."
    )


def run_task_smoke(code: str, task: CodingTask) -> tuple[bool, str, list[FixtureResult]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return (False, f"generated code is not Python: {exc.msg}", [])
    blocked = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
    if any(isinstance(node, blocked) for node in ast.walk(tree)):
        return (False, "generated code uses blocked syntax for the smoke fixture", [])
    namespace = make_namespace()
    try:
        exec(compile(tree, "<qwen-fixture>", "exec"), namespace)
    except Exception as exc:  # noqa: BROAD_EXCEPT_OK
        return (False, f"generated code raised during load: {exc}", [])
    candidate = namespace.get(task.name)
    if not isinstance(candidate, FunctionType):
        return (False, f"generated code is missing function: {task.name}", [])
    results: list[FixtureResult] = []
    for case in task.cases:
        try:
            observed = candidate(*case.args)
        except Exception as exc:  # noqa: BROAD_EXCEPT_OK
            results.append({"name": task.name, "status": "fail", "error": str(exc)})
            return (False, f"{task.name} raised during smoke check: {exc}", results)
        if observed != case.expected:
            error = f"expected {case.expected!r}, got {observed!r}"
            results.append({"name": task.name, "status": "fail", "error": error})
            return (False, f"{task.name} returned an incorrect result: {error}", results)
        results.append({"name": task.name, "status": "pass"})
    return (True, "code smoke passed", results)


def make_namespace() -> dict[str, JsonValue | dict[str, type[list] | type[int] | type[str] | FunctionType]]:
    return {
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
