# Worklog 27 — Mode-Aware Representative Selection Phase 2 Exact 최적화

## 목표

Worklog 26 이후 남은 mode-aware representative selection 병목을 줄인다. cell mode splitting, stable-ID 순서, FPS 정책은 고정하며, v3 replay artifact의 candidate 및 FPS trace와 완전히 일치하는 구현만 채택한다.

## 고정 기준

- Replay artifact: `C:\tmp\osn_gs_mode_aware_selection_replay_v3.pt`
- SHA-256: `c4e1bcacb73c95516afbd1f43783f65c0d996cb7c0b7623f772a6b98d736786c`
- Input: 138,766 Gaussian / 895 cells / 2,898 candidate modes / 2,048 final representatives
- Production hot path는 artifact loading·detailed trace 수집을 하지 않는다.

## Baseline profile

v3 replay warm 5회, CUDA synchronize 기준:

- Representative selection median: 4.278s, P90: 4.304s
- Current NumPy/Python exact cell splitter median: 1.273s (29.8%)

따라서 native splitter는 아직 착수 조건을 만족하지 않는다. 나머지 약 70%를 차지하는 aggregate·medoid·candidate payload가 우선 대상이다.

## 진행 기록

- Tensor segmented reduction의 첫 시도는 candidate representative exact equality를 깨뜨려 폐기했다. scatter reduction의 누적 순서가 기존 per-mode 연산 순서와 달라 medoid가 바뀔 수 있음을 확인했다.
- 이미 CPU에 존재하는 source arrays를 재사용한 NumPy aggregate/medoid도 v3에서 medoid 8개가 달라져 폐기했다. GPU aggregate와 CPU aggregate의 수치 순서는 interchange할 수 없다.
- 채택: mode별 opacity sum·centroid·거리 텐서 계산은 기존 Torch 경로를 그대로 유지하고, 각 member 거리의 `float(CUDA tensor)` 138,766회를 `torch.cat(...).cpu().tolist()` 1회로 바꿨다. 이후 nearest-distance/stable-ID Python tie-break도 기존과 동일한 순서로 수행한다.
- Exact gate: v3 public representative 2,048개, compact candidate의 cell/mode/source-count/opacity mass, 그리고 2,048-step FPS `selected_candidate_indices`가 모두 정확히 일치했다. mode splitting 코드는 변경하지 않았으므로 detailed per-member trace의 정책도 그대로 보존된다.
- v3 replay warm 5회/CUDA synchronize 재측정: selection median 2.324s, P90 2.382s. 기존 4.278s 대비 median 45.7% 단축(약 1.84x)이다. exact splitter median은 1.239s로 이제 selection 시간의 53.3%다.
- 다음 판단: native exact splitter는 이제 profiler상 가장 큰 병목이지만, 별도 native/CUDA 실험은 replay artifact의 per-step mode trace를 직접 비교하는 전용 gate를 먼저 갖춘 뒤에만 검토한다. 이번 단계에서는 착수하지 않는다.