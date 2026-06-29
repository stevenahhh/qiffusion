from __future__ import annotations

import os
import shutil
import subprocess
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


def has_hf_snapshot(model_id: str) -> bool:
    if os.environ.get("QIFFUSION_DISABLE_HF") == "1":
        return False
    cache_name = "models--" + model_id.replace("/", "--")
    for root in hf_cache_roots():
        snapshots = root / cache_name / "snapshots"
        if snapshots.is_dir() and any((path / "config.json").is_file() for path in snapshots.iterdir()):
            return True
    return False


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
    home = Path.home()
    return (
        Path.cwd(),
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "lm-studio" / "models",
        home / ".lmstudio" / "models",
    )


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
    if has_hf_snapshot(PREFERRED_MODEL_ID):
        return {
            "backend": "qwen_bridge",
            "model_id": PREFERRED_MODEL_ID,
            "status": "available",
            "engine": "transformers",
            "notes": ["local Hugging Face snapshot found"],
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
        "notes": ["no local Qwen 4B engine found; no download attempted"],
        "fixtures_status": "not_run",
        "code_smoke_status": "not_run",
        "candidate_source": "none",
        "coding_capability_claim": False,
    }
