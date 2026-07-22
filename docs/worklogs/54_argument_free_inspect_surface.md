# 인자 없는 `inspect-surface` 실행

## 수행 작업

- `osn-gs inspect-surface`의 `--source_path` 기본값을 노트북 로컬 Dataset 셀과 같은 `DATASET`으로 설정했다.
- 기본 출력 경로를 노트북 OSN-GS `MODEL_ROOT` 아래의 `output/osn_gs_scene/inspect-surface`로 변경했다.
- 학습 stream cache와 분리하기 위해 스냅샷을 전용 폴더의 `renderer_snapshot.json`으로 직접 저장한다.
- 노트북 Train 셀의 Stage 1 constructor 기본값도 검사 CLI에 추가해 초기화 레시피가 일치하도록 했다.

## 결과

- 필수 인자가 없어져 `.venv\Scripts\osn-gs.exe inspect-surface`만으로 실행할 수 있다.
- `surface_quality.json`과 `surface_quality.txt`는 같은 전용 폴더에 기록되며, renderer용 단일 스냅샷도 함께 보존된다.

## 평가

- `python -B -m py_compile scripts/devtools/inspect_visible_surface.py`와 `inspect_visible_surface.py --help`를 통과했다.
- Stage 1/NURBS/voxel/covariance의 기본값은 현재 노트북 Train 셀 값과 일치한다.

## 남은 위험

- 노트북에서 사용자가 `DATA_ROOT` 또는 `MODEL_ROOT`를 수동으로 다른 위치로 바꾸면 PowerShell CLI가 실행 중인 노트북 커널의 값을 자동으로 읽지는 않는다. 이 경우 `-s` 또는 `--output`으로 명시적으로 재정의해야 한다.