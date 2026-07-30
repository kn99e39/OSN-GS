# Worklog 127 — 노트북 canonical ADC-NURBS 실행 설정

## 작업

- `colab_train_3dgs.ipynb`의 OSN-GS Train 셀에서 삭제된 legacy/voxel/Stage-1 CLI 옵션 전달을 제거했다.
- 현재 `train.py`가 지원하는 canonical covariance/reliability visible-NURBS 옵션만 전달하도록 정리했다.
- 기본값을 `OSN_VISIBLE_NURBS_UPDATE_SCHEDULE = 'adc_post_commit'`으로 두고 `--visible_nurbs_update_schedule adc_post_commit`를 명시했다.

## 결과

- 노트북 Train 셀은 구조적 ADC의 clone/split/prune가 실제 commit된 뒤 NURBS 재구성을 시도한다. no-op ADC와 opacity reset만으로는 중복 재구성하지 않는다.
- JSON 파싱과 현재 Train 셀의 옵션 whitelist 검증을 통과했다.

## 주의

- 이는 재구성 **시도**를 ADC마다 연결하는 설정이다. 현재 실제 `DATASET`은 canonical reliable region을 만들지 못해 각 시도가 `no_admissible_region` fail-closed empty payload가 될 수 있다. 상세 근거는 Worklog 126을 따른다.
