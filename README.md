# qiffusion

## 최종 예정 형태

`qiffusion`의 최종 목표는 **실제 코드 작성/디버깅/도구 호출까지 포함한 실사용 가능한 대화형 생성 능력**을 갖춘 Qwen 기반 디퓨전 LLM 레포로 수렴하는 것입니다.

이 레포는 다음의 형태를 지향합니다.

- `qiffusion` 코어는 디퓨전형 denoiser를 중심으로 학습/샘플링/평가가 독립적으로 동작
- Qwen(현재는 Qwen/Qwen3.5-4B 계열) 경량 브릿지를 `teacher`로 사용하여 고품질 합성데이터를 생성
- 멀티턴 대화, 코딩 과제, 에이전트형 동작용 데이터는 전부 하나의 공통 게이트(`qiffusion` 공유 게이트)로 묶어 검증
- 성능 공개 전에는 캡처된 증거(실행 가능한 코드 smoke, 추론 정확도 지표, 데이터 라인리지)로만 주장

## 현재 상태(요약)

- 레포의 공개 엔트리포인트는 `python -m qiffusion.cli`입니다.
- 현재는 디퓨전 골격 + Qwen 브릿지 + 증거 파일 생성 명령이 준비된 실험 단계입니다.
- 코드 캡슐러빌리티 주장보다 **재현 가능한 증거 기반 개선 루프**를 우선하는 방향입니다.

## 핵심 루프(최종 운영 목표)

1. 데이터 축적
   - 로컬 코드/문서/대화 데이터 수집
   - Qwen 브릿지 기반 teacher trace 생성
   - 데이터 메타(원본, 버전, 라이선스, 품질 스코어) 보존
2. 디퓨전 학습
   - Qwen tokenizer 호환 파이프라인에 맞춘 학습 스텝
   - 스몰 배치 스모크부터 시작해 점진적 스케일 업
3. 검증
   - 코드 smoke, 회귀 테스트, 멀티턴 품질 점검
   - 실패 시 데이터/학습 설정에 대한 반복 개선
4. 게이트 통과 시 공개
   - 증거 패키지 정리 후 다음 단계 계획으로 이관

## 레포 구성 (예정)

- `src/qiffusion/`  
  - 핵심 엔진, trainer, sampler, evaluator
- `docs/`  
  - 실험 규약, 데이터 정책, 확장 계획
- `artifacts/`  
  - 증거 보고서(JSON), 로그, 체크포인트 메타데이터
- `.omo/`  
  - 작업 계획, 진행 상태, 규칙 기반 워크플로우
- `.github/`  
  - CI/CD 및 자동 검증 규칙(예정)

## 사용 방법

### 최소 실행

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m qiffusion.cli plan
```

### Qwen 브릿지 상태 확인

```powershell
.\.venv\Scripts\python.exe -m qiffusion.cli qwen-status --out evidence/qwen-status.json
.\.venv\Scripts\python.exe -m qiffusion.cli qwen-eval --out evidence/qwen-eval.json
```

### 디퓨전 스모크 루프

```powershell
.\.venv\Scripts\python.exe -m qiffusion.cli backend-status --backend diffusion --out evidence/diffusion-status.json
.\.venv\Scripts\python.exe -m qiffusion.cli diffusion-train --steps 20 --seed 11 --max-examples 24 --out evidence/final-tiny.pt --report-out evidence/final-train.json
.\.venv\Scripts\python.exe -m qiffusion.cli diffusion-sample --checkpoint evidence/final-tiny.pt --prompt "def" --steps 8 --seed 11 --out evidence/final-sample.json
.\.venv\Scripts\python.exe -m qiffusion.cli diffusion-eval --checkpoint evidence/final-tiny.pt --runs 1 --out evidence/final-eval.json
.\.venv\Scripts\python.exe -m qiffusion.cli status --report evidence/final-eval.json
```

## 공개 및 배포

이 레포는 공개 저장소입니다.  
최종 산출물은 원격 저장소([https://github.com/stevenahhh/qiffusion.git](https://github.com/stevenahhh/qiffusion.git))에 커밋·푸시됩니다.

## 설계 원칙

- 코드나 모델 성능 주장은 근거 기반 명확한 기준 통과 후에만 제시
- 실험 실패 데이터는 즉시 폐기하지 않고, 개선 포인트로 재활용
- 향후 브릿지/디퓨전 병렬 개선 대신, 공유 게이트 기반 순환 개선을 우선

## 참고 문서

- `docs/diffusion-training.md` : 실험/평가/확장 경로의 상세 규칙
- `.omo/plans/` : 단계별 실행 계획
