# 워크로그 식별자 규칙

파일명 앞의 번호는 작업 기록의 안정 식별자다. 대부분은 단일 정수 `NN`을 쓴다. 같은 시점의 독립 작업이 병렬로 기록되어 기존 번호가 충돌한 경우에는 과거 링크를 깨지 않도록 `NNa`, `NNb` suffix를 사용한다.

| 식별자 | 기록 |
|---|---|
| 60-A | Held-out eval 파이프라인 연결과 smoke test |
| 60-B | Proxy-Based Surface Decomposition Stage 0 기준선 |
| 73-A | anisotropy 격차 1차 조사 |
| 73-B | Urgent_Work 계획 폐기와 현재 게이트 정리 |
| 77-A | Boundary-Conditioned Phase C observation evidence |
| 77-B | anisotropy parity ablation 결과 |

새 worklog는 현재 가장 큰 정수 식별자 다음 번호를 사용한다. 병렬 작업을 같은 번호로 추가해야 하면 suffix를 부여하고, 이 표·파일 제목·모든 Markdown 링크를 같은 변경에서 함께 갱신한다.
