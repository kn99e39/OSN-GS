# 44. 노트북 저장 출력 단계 오류 수정

## 작업 내용

5000 iteration 이후 노트북 학습이 중단된 실행 결과를 확인했다. output/osn_gs_scene/5000/에 point_cloud.ply, render.ppm, metrics.txt만 존재하고 nurbs_surface.json과 checkpoint.pt가 없었으므로 학습 루프가 아니라 저장 출력 단계의 오류로 범위를 좁혔다.

TorchOSNGSTrainer.save_outputs()의 실행 순서를 추적한 결과, _save_nurbs_intermediate()에서 voxel_regions payload 내부의 CPU Torch Tensor가 json.dumps()에 직접 전달되는 문제가 원인이었다. NURBS JSON 저장이 실패해 checkpoint 단계까지 도달하지 못했다.

## 수정 결과

- osn_gs/core/torch_trainer.py
  - voxel region payload의 Tensor 필드를 모두 Python list로 변환
  - nurbs_surface.json 파일 직렬화가 가능하도록 수정
- colab_train_3dgs.ipynb
  - 학습 subprocess가 비정상 종료되면 마지막 출력 tail 40줄을 captured train.py output 블록으로 표시
  - stderr는 기존처럼 stdout과 병합하여 수집

## 검증

- 노트북 JSON 파싱 성공
- torch_trainer.py Python 구문 검사 성공
- pytest는 현재 환경에 설치되어 있지 않아 실행하지 못했다.

## 남은 위험

실제 5000 iteration CUDA 학습을 재실행해 nurbs_surface.json과 checkpoint.pt까지 생성되는지 확인해야 한다. 대형 Gaussian 상태의 JSON/체크포인트 파일 크기와 저장 시간도 별도로 관찰할 필요가 있다.