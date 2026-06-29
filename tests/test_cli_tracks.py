from __future__ import annotations

import json
import sys
from pathlib import Path

from qiffusion.cli import main


def write_fake_ollama(tmp_path: Path, *, model_list: str, code: str) -> None:
    script = tmp_path / "fake_ollama.py"
    script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "import sys",
                "model_list = " + repr(model_list),
                "code = " + repr(code),
                "command = sys.argv[1] if len(sys.argv) > 1 else ''",
                "if command == 'list':",
                "    print('NAME ID SIZE MODIFIED')",
                "    if model_list:",
                "        print(model_list)",
                "elif command == 'run':",
                "    print(json.dumps({'code': code}))",
                "else:",
                "    raise SystemExit(2)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    launcher = tmp_path / "ollama.cmd"
    launcher.write_text(f'@echo off\n"{sys.executable}" "{script}" %*\n', encoding="utf-8")


def test_qwen_status_writes_prerequisite_report_when_no_engine(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("QIFFUSION_HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("QIFFUSION_DISABLE_HF", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_OLLAMA", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_GGUF", "1")
    output = tmp_path / "qwen.json"

    exit_code = main(["qwen-status", "--out", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert report == stdout
    assert report["backend"] == "qwen_bridge"
    assert report["model_id"] == "Qwen/Qwen3.5-4B"
    assert report["status"] == "prerequisite_missing"
    assert report["coding_capability_claim"] is False


def test_qwen_status_rejects_incomplete_hf_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "hf" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "bad"
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).write_text("", encoding="utf-8")
    monkeypatch.setenv("QIFFUSION_HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("QIFFUSION_DISABLE_OLLAMA", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_GGUF", "1")
    output = tmp_path / "qwen.json"

    exit_code = main(["qwen-status", "--out", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "prerequisite_missing"
    assert "incomplete" in " ".join(report["notes"])
    assert report["coding_capability_claim"] is False


def test_qwen_eval_promotes_after_fake_ollama_code_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_fake_ollama(
        tmp_path,
        model_list="qwen3.5:4b abc 3.4 GB now",
        code=(
            "def add(a, b):\n"
            "    return a + b\n"
            "def count_even(values):\n"
            "    return sum(1 for value in values if value % 2 == 0)\n"
            "def reverse_words(text):\n"
            "    return ' '.join(reversed(text.split()))\n"
        ),
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("QIFFUSION_DISABLE_HF", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_GGUF", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_OLLAMA_HTTP", "1")
    output = tmp_path / "qwen-eval.json"

    exit_code = main(["qwen-eval", "--out", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "available"
    assert report["engine"] == "ollama"
    assert report["fixtures_status"] == "pass"
    assert report["code_smoke_status"] == "pass"
    assert report["candidate_source"] == "ollama:qwen3.5:4b"
    assert report["coding_capability_claim"] is True
    assert {item["name"] for item in report["fixture_results"]} == {
        "add",
        "count_even",
        "reverse_words",
    }


def test_qwen_eval_rejects_incomplete_generated_suite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_fake_ollama(
        tmp_path,
        model_list="qwen3.5:4b abc 3.4 GB now",
        code="def add(a, b):\n    return a + b\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("QIFFUSION_DISABLE_HF", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_GGUF", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_OLLAMA_HTTP", "1")
    output = tmp_path / "qwen-eval.json"

    exit_code = main(["qwen-eval", "--out", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["fixtures_status"] == "pass"
    assert report["code_smoke_status"] == "fail"
    assert report["coding_capability_claim"] is False
    assert "count_even" in report["smoke_error"]


def test_qwen_eval_missing_ollama_model_does_not_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_fake_ollama(tmp_path, model_list="", code="def add(a, b):\n    return a + b\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("QIFFUSION_DISABLE_HF", "1")
    monkeypatch.setenv("QIFFUSION_DISABLE_GGUF", "1")
    output = tmp_path / "qwen-eval.json"

    exit_code = main(["qwen-eval", "--model", "missing-qwen", "--out", str(output)])

    assert exit_code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "prerequisite_missing"
    assert report["fixtures_status"] == "not_run"
    assert report["code_smoke_status"] == "not_run"
    assert report["coding_capability_claim"] is False


def test_backend_status_writes_diffusion_scaffold_report(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "diffusion.json"

    exit_code = main(["backend-status", "--backend", "diffusion", "--out", str(output)])

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert report == stdout
    assert report["backend"] == "diffusion"
    assert report["status"] == "scaffold_ready"
    assert report["coding_capability_claim"] is False
    assert report["candidate_source"] == "none"
