# Migration Context (읽고 시작할 것)

이 문서는 이전 세션(Claude)에서 진행한 작업을 새 컨텍스트에서 이어받기 위한 요약이다. 프로젝트 자체의 설계는 여기 없다 — 아래 "다음에 읽을 문서" 순서대로 읽을 것.

## 지금 당장 할 일

**`TODO.md`를 먼저 읽어라.** NURBS patch가 실제 형태(aspect ratio)와 다르게 fitting되는 버그의 원인 분석과 수정 방향이 재현 스크립트와 함께 전부 정리되어 있다. 이번 세션은 원인 분석까지만 했고 **수정은 아직 안 했다.**

## 다음에 읽을 문서 (이 순서대로)

1. `docs/architecture.md` — OSN-GS가 뭘 하려는 프로젝트인지 (surface-centric 3DGS, NURBS 중간 표현)
2. `docs/nurbs_construction.md` — Gaussian → NURBS 생성 전체 파이프라인, 수식/코드 참조 지도 포함. 매우 상세함.
3. `docs/voxel_role.md` — voxel bootstrap이 그 파이프라인에서 정확히 어떤 역할을 하는지
4. `TODO.md` — 지금 풀어야 할 구체적 버그
5. `nurbs_constructor_benchmark/README.md` — 실제 production 경로(`TorchOSNGSPipeline.initialize()`)를 synthetic scene으로 검증하는 도구. `TODO.md`의 재현/검증에 이걸 쓴다.

## 이번 세션에서 한 일 (시간순 요약)

1. 프로젝트 전체 프레임워크 구조 평가 (완성도, 미구현 부분 파악).
2. `--low_vram` 버그 수정, NURBS를 진짜 rational B-spline evaluator로 구현(Cox-de Boor), voxel region 정규화/경계탐지 GPU 벡터화.
3. **레거시 numpy 프로토타입 프레임워크 전체 삭제** (~25개 파일, ~940줄) — `osn_gs.core.framework` 계열, `osn_gs.gaussian.certain_gaussians` 등. 실제 torch 학습 경로(`train.py`, `scripts/train_osn_gs_torch.py`, 모든 `torch_*.py`)는 이걸 전혀 참조하지 않았음을 grep으로 확인 후 진행. 모든 `osn_gs/**/__init__.py`를 실제 `torch_*` 심볼만 export하도록 정리. 죽은 테스트를 실제 torch 경로를 검증하는 `tests/test_torch_pipeline_smoke.py`로 교체.
4. 학습 스트리밍 성능 문제 진단: fallback rasterizer 여부 확인법, bulk streaming이 고정 50ms 딜레이로 50~70MB 메시지를 연달아 쏴서 relay/브라우저가 못 따라가는 문제 → 페이로드 크기 비례 딜레이로 수정.
5. `nurbs_constructor_benchmark`(다른 에이전트가 만들었으나 본인 환경에 torch 없어 한 번도 실행 못 해본 상태였음) 구조 평가, 실제로 처음 실행/검증. `density_gradient` scene 추가(밀도 불균일 데이터로 adaptive voxel density를 실제로 스트레스 테스트하는 유일한 scene). `NURBS_output/<scene>/{point_cloud.ply,nurbs_surface.json}` export 기능 추가 — 이 과정에서 트레이너의 실제 파일 저장 코드가 multi-patch 중 patch 0만 저장하던 기존 버그를 발견해 `nurbs_intermediate_payload()` 공용 헬퍼로 통합 수정(트레이너와 벤치마크가 같은 함수 사용, 전체 `patches[]` 포함하도록 개선).
6. `docs/nurbs_construction.md` 작성 (Gaussian→NURBS 전체 파이프라인 기술 문서).
7. 사용자가 렌더러 스크린샷을 공유 → NURBS_output 결과를 렌더러(`WebRenderer`)에 띄운 것에 대한 3가지 질문에 답하는 과정에서, 실제 `WebRenderer/util/NurbsGeometry.js`(사용자가 최신 파일을 업로드해줌)를 읽고 노란 곡선=`base_curves` Bézier, cyan 격자=진짜 iso-parametric wireframe(control polygon 아님)임을 코드 근거로 확인.
8. "fitting issue가 맞다"는 (GPT의) 판단을 검증하는 과정에서 실제로 `crease` scene의 patch 0/1/3이 진짜 형태 대비(aspect ratio 5.3:1) 정사각형에 가까운 grid(2.33:1)로 fitting되는 진짜 버그를 수치로 확인 → `docs/voxel_role.md`, `TODO.md` 작성.
9. `voxel-surface-regions` 브랜치를 `main`에 병합 (충돌 3개 해결, 병합 근거 확인 후 진행). `TODO.md`를 `voxel-surface-regions`에 커밋.

## 현재 git 상태

```text
main                   19150b9  (origin/main보다 11 커밋 앞섬, push 안 함)
voxel-surface-regions  e9ae98a  (origin/voxel-surface-regions보다 2 커밋 앞섬, push 안 함)  <- 현재 브랜치
```

**원격(origin)에는 아직 아무것도 push하지 않았다.** 로컬 `main`은 `voxel-surface-regions`의 모든 내용을 병합해서 최신 상태다. 두 브랜치 모두 working tree clean.

## 알아둘 것

- **동시 편집 환경**: 이 리포는 Claude(나)와 다른 에이전트(Codex로 추정, `docs/worklogs/`에 작업 기록 남김)가 같은 파일을 동시에 편집하고 있었다. 파일을 읽을 때 이전 세션 기억보다 **실제 파일 내용을 항상 우선**할 것.
- **개발 환경**: Windows, `.venv/Scripts/python.exe`에 torch 설치됨(CUDA 사용 가능한 머신도 있었음: RTX 5080). 테스트는 `PYTHONPATH="c:/Projects/OSN-GS" .venv/Scripts/python.exe -B -m unittest discover -s tests` (현재 26개 통과). 벤치마크는 `python -m nurbs_constructor_benchmark`.
- **렌더러**: `WebRenderer/`(로컬 vendored 복사본)와 `../3DGS_Renderer/`(형제 디렉터리, 별도 git repo) 두 곳에 비슷한 렌더러 코드가 있다. `RENDERER_INPUT_FORMAT.md`가 입력 포맷을 문서화한다. NURBS 렌더링 코드는 `WebRenderer/util/NurbsGeometry.js` + `Shaders/nurbs_geometry.wgsl` (사용자가 최근에 업로드해준 최신 버전).
- **Korean/English 혼용**: 코드 주석과 문서는 한글/영어가 섞여 있다. Korean 인코딩 사고가 과거에 한 번 있었음(`Agent.md`의 "Korean Markdown Encoding Rules" 참고) — `.md`/한글 주석 수정 시 UTF-8 유지에 주의.
- **사용자 작업 스타일**: 큰 작업 전엔 계획/근거를 먼저 설명하고 확인받는 걸 선호함(예: "작업을 바로 하진 말고" 지시). 파괴적 작업(대량 삭제 등)은 사전 확인 필요.
