from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path
from typing import Final, Literal, TypedDict

PREFERRED_MODEL_ID: Final = "Qwen/Qwen3.5-4B"
Status = Literal["available", "prerequisite_missing"]


class QwenBridgeReport(TypedDict):
    backend: str
    model_id: str
    status: Status
    engine: str | None
    notes: list[str]
    fixtures_status: str
    code_smoke_status: str
    candidate_source: str
    coding_capability_claim: bool


@dataclass(frozen=True, slots=True)
class Readiness:
    available: bool
    notes: tuple[str, ...]


def hf_cache_roots() -> tuple[Path, ...]:
    configured = os.environ.get("QIFFUSION_HF_HOME")
    roots: list[Path] = []
    if configured is not None and configured != "":
        roots.append(Path(configured))
    for name in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(name)
        if value is not None and value != "":
            roots.append(Path(value))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return tuple(dict.fromkeys(roots))


def discover_hf_snapshot(model_id: str) -> Readiness:
    if os.environ.get("QIFFUSION_DISABLE_HF") == "1":
        return Readiness(False, ())
    cache_name = "models--" + model_id.replace("/", "--")
    notes: list[str] = []
    for root in hf_cache_roots():
        snapshots = root / cache_name / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in snapshots.iterdir():
            if snapshot_is_complete(snapshot):
                return Readiness(True, ("local Hugging Face snapshot found",))
            if snapshot.is_dir():
                notes.append(f"local Hugging Face snapshot incomplete: {snapshot.name}")
    return Readiness(False, tuple(notes))


def snapshot_is_complete(snapshot: Path) -> bool:
    required = ("config.json", "tokenizer.json")
    if not snapshot.is_dir():
        return False
    if not all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0 for name in required):
        return False
    weights = tuple(snapshot.glob("*.safetensors")) + tuple(snapshot.glob("*.bin"))
    return any(path.is_file() and path.stat().st_size > 0 for path in weights)


def has_hf_snapshot(model_id: str) -> bool:
    return discover_hf_snapshot(model_id).available


def runtime_is_ready() -> Readiness:
    if util.find_spec("transformers") is None:
        return Readiness(False, ("transformers is not installed",))
    if util.find_spec("torch") is None:
        return Readiness(False, ("torch is not installed",))
    torch_version = metadata.version("torch")
    if version_key(torch_version) < (2, 4):
        return Readiness(False, (f"torch {torch_version} is below required 2.4",))
    return Readiness(True, ("transformers runtime is ready",))


def version_key(value: str) -> tuple[int, int]:
    parts = value.split("+", maxsplit=1)[0].split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    return (major, minor)


def ollama_has_qwen() -> bool:
    if os.environ.get("QIFFUSION_DISABLE_OLLAMA") == "1":
        return False
    executable = shutil.which("ollama")
    if executable is None:
        return False
    try:
        result = subprocess.run([executable, "list"], text=True, capture_output=True, timeout=10.0, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    names = [line.split()[0].lower() for line in result.stdout.splitlines()[1:] if line.split()]
    return any("qwen" in name and "4b" in name for name in names)


def gguf_roots() -> tuple[Path, ...]:
    if os.environ.get("QIFFUSION_DISABLE_GGUF") == "1":
        return ()
    configured = os.environ.get("QIFFUSION_GGUF_ROOTS")
    if configured is None or configured == "":
        return (Path.cwd(),)
    return tuple(Path(root) for root in configured.split(os.pathsep) if root != "")


def has_qwen_gguf() -> bool:
    scanned = 0
    for root in gguf_roots():
        if not root.exists():
            continue
        for path in root.rglob("*.gguf"):
            scanned += 1
            lowered = path.name.lower()
            if "qwen" in lowered and "4b" in lowered:
                return True
            if scanned >= 5000:
                return False
    return False


def qwen_status() -> QwenBridgeReport:
    hf = discover_hf_snapshot(PREFERRED_MODEL_ID)
    runtime = runtime_is_ready()
    if hf.available and runtime.available:
        return {
            "backend": "qwen_bridge",
            "model_id": PREFERRED_MODEL_ID,
            "status": "available",
            "engine": "transformers",
            "notes": [*hf.notes, *runtime.notes],
            "fixtures_status": "not_run",
            "code_smoke_status": "not_run",
            "candidate_source": "none",
            "coding_capability_claim": False,
        }
    if ollama_has_qwen():
        return {
            "backend": "qwen_bridge",
            "model_id": PREFERRED_MODEL_ID,
            "status": "available",
            "engine": "ollama",
            "notes": ["local Ollama Qwen 4B model found"],
            "fixtures_status": "not_run",
            "code_smoke_status": "not_run",
            "candidate_source": "none",
            "coding_capability_claim": False,
        }
    if has_qwen_gguf():
        return {
            "backend": "qwen_bridge",
            "model_id": PREFERRED_MODEL_ID,
            "status": "available",
            "engine": "llama.cpp",
            "notes": ["local Qwen 4B GGUF found"],
            "fixtures_status": "not_run",
            "code_smoke_status": "not_run",
            "candidate_source": "none",
            "coding_capability_claim": False,
        }
    return {
        "backend": "qwen_bridge",
        "model_id": PREFERRED_MODEL_ID,
        "status": "prerequisite_missing",
        "engine": None,
        "notes": [*hf.notes, *runtime.notes, "no runnable local Qwen 4B engine found; no download attempted"],
        "fixtures_status": "not_run",
        "code_smoke_status": "not_run",
        "candidate_source": "none",
        "coding_capability_claim": False,
    }
