# Diffusion Training

This milestone proves the local train -> sample -> eval loop for a tiny diffusion language model. It is not a useful model yet.

## Exact Commands

```powershell
python -m qiffusion.cli diffusion-train --steps 20 --seed 11 --max-examples 24 --out evidence/final-tiny.pt --report-out evidence/final-train.json
python -m qiffusion.cli diffusion-sample --checkpoint evidence/final-tiny.pt --prompt "def" --steps 8 --seed 11 --out evidence/final-sample.json
python -m qiffusion.cli diffusion-eval --checkpoint evidence/final-tiny.pt --runs 1 --out evidence/final-eval.json
python -m qiffusion.cli status --report evidence/final-eval.json
```

The eval report is intentionally shared-gate compatible. The tiny model should remain `coding_capability_claim=false` unless generated Python passes executable code smoke.

## Artifact Policy

Generated checkpoints are runtime artifacts and are not committed. Keep `.pt`, `.safetensors`, cache directories, and generated model files under ignored paths such as `evidence/`, `checkpoints/`, `models/`, or `.omo/ulw-loop/.../evidence/`.

JSON and text evidence may be committed only when it is small, stable, and useful for review. Large training outputs stay ignored.

## Not 4B Yet

This path is not 4B yet. It uses deterministic byte tokenization, a tiny GRU denoiser, CPU training, and a conservative gate. The output is expected to fail practical coding smoke until training data, objective quality, model size, and sampling improve.

## Scaling Path

The next scale steps are:

1. Replace the byte tokenizer with a stronger code-aware tokenizer.
2. Expand the corpus with verified local code, exported teacher snippets, and benchmark-style repair tasks.
3. Move from tiny GRU denoising to LLaDA, Dream, DiffuCoder, or DiffusionGemma-style block/canvas denoising.
4. Add AR initialization or distillation only as explicit training data or initialization, never as a hidden sampler fallback.
5. Promote only through the shared gate after generated code passes executable coding benchmarks.
