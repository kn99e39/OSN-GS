# Worklog 28 — Native Exact Cell Splitter Gate

## 결론

C++ CPU native splitter prototype은 실제 v3 replay에서 exact assignment를 재현하지 못했다. 따라서 production backend로 연결하지 않았고, Worklog 27의 Python/NumPy reference splitter와 bulk medoid-distance transfer 경로를 그대로 유지한다.

## 고정 oracle

- Artifact: `C:\tmp\osn_gs_mode_aware_selection_replay_v3.pt`
- SHA-256: `c4e1bcacb73c95516afbd1f43783f65c0d996cb7c0b7623f772a6b98d736786c`
- 입력: 138,766 Gaussian, 895 cells, 2,898 candidate modes, 2,048 final representatives

## Prototype 결과

- 선택: CPU C++ extension. cell 간 독립성은 native loop로 처리하고, cell 내부 member stream·mode iteration·strict highest-alignment tie 처리를 순차로 보존하도록 구현했다.
- 빌드: Windows MSVC와 PyTorch C++ extension으로 성공했다. `ninja`는 환경에 이미 설치되어 있었고 Python Scripts PATH 및 compiler output decode 설정만 필요했다.
- 성능: native splitter warm median `2.533ms`.
- Exact gate: mode count는 895개 cell에서 일치했지만, Gaussian-to-mode assignment는 불일치했다. 첫 불일치는 stable-ID ordered stream offset 16,826 / source index 100,929에서 reference mode 0, native mode 1이었다.

## 판정

`np.dot`의 실제 floating-point 동작을 C++의 3항 scalar dot product로 대체한 것만으로도 discrete mode assignment가 달라졌다. mode count가 같거나 속도가 빠르다는 사실은 승인 근거가 될 수 없다.

CUDA는 더 다른 reduction/FMA/rounding 경로를 사용하므로, 이 CPU prototype이 exact gate를 통과하지 못한 상황에서 추가 구현하지 않았다. artifact의 per-step trace를 정확히 재현할 수 있는 native numeric backend가 별도로 입증되기 전까지 native splitter는 production 후보가 아니다.

## 유지되는 production 상태

- Worklog 27의 `torch.cat(...).cpu().tolist()` 기반 bulk medoid-distance transfer를 유지한다.
- Representative selection 기준값은 median `2.324s`, P90 `2.382s`이며, Python exact splitter median은 `1.239s`이다.
- production hot path에는 artifact loading, detailed trace collection, native JIT build를 추가하지 않았다.

## 남은 위험 및 다음 조건

현재 native 접근의 이론적 성능은 충분하지만, authoritative `np.dot` semantics가 discrete selection을 바꾼다는 것이 확인됐다. 재시도하려면 단순 C++/CUDA reimplementation이 아니라 reference dot backend까지 포함한 per-step exact gate와 cross-platform build/maintenance 근거가 먼저 필요하다.