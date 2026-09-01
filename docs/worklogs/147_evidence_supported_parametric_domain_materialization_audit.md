# Worklog 147 — Evidence-Supported Parametric Domain Materialization Audit

## 상태: 중단 — WL145 exact baseline mismatch

WL147은 WL139 representative를 변경하지 않고, WL145의 frozen support-domain
annotation만으로 full-domain materialization과 support-constrained
materialization을 비교하도록 지시했다. Phase 1의 exact replay를 먼저
수행했으나, 현재 보존된 WL145 artifact와 Worklog 147 계약의 기준값이
일치하지 않아 지시문에 따라 이 batch를 중단했다.

## 확인한 immutable 입력

- WL145 case: `tabletop_broad_planar_clean`
- per-view event union: `1586 x 3`
- WL145 fit-input/event-union hash: `79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78`
- frozen representative: `3840 x 3` sampled points와 `3840 x 3` normals가 들어 있는
  `output/145_genuine_physical_sheet_oracle_clean_support_representative_audit/tabletop_broad_planar_clean/clean_support_representative/wl139_frozen_representative.npz`
- support annotation grid: `96 x 40`
- support mask hash replay: `23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9`

## 실제 구현의 replay 결과

커밋된 WL145 구현(`6f7482e`)의 `_domain_accounting`을 변경 없이 적용했다.
좌표는 WL145 PCA chart의 physical `u/v` 좌표이며, 각 raw oracle point를
`floor(normalized_uv * [96, 40])`로 binning하고 clip한다. 하나라도 점유된
grid bin은 supported vertex이고, 나머지는 unsupported vertex다. 면적 계산에서
cell은 네 꼭짓점이 모두 supported인 경우에만 supported로 집계되지만, WL145
annotation 자체는 vertex 기반이다.

| 항목 | WL147 계약 | 현재 artifact/source exact replay |
|---|---:|---:|
| supported vertices | `248 / 3840` | `314 / 3840` |
| unsupported vertices | `3592 / 3840` | `3526 / 3840` |
| support mask hash | 기준값 미제공 | `23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9` |

현재 WL145 JSON report도 `314/3526`과 동일한 mask hash를 기록한다. 반면
`docs/README.md` 및 WL145 prose는 `248/3592`를 기록한다. 따라서 현재
보존된 report/source는 서로 exact replay되지만, WL147이 요구한 historical
248 baseline을 재현하지 못한다. raw→representative와 representative→raw
거리도 report에는 각각 `1.3276h / 2.1729h` 및 `32.4048h / 77.9380h`로
기록되어 있고, WL145 prose의 fixed-residual/area 수치와도 일부 차이가 있다.

## 판정과 보류한 작업

- historical support annotation을 재현하지 못했으므로 `ARM A/B`를 실행하지
  않았다.
- replacement support mask, new threshold, dilation/erosion, refit, trimming,
  topology repair, continuation은 만들지 않았다.
- WL139 representative, WL145 oracle/provenance, renderer, checkpoint,
  Candidate B 및 canonical production code는 변경하지 않았다.
- 따라서 WL147의 architecture attribution이나 quantitative A/B verdict는
  아직 제시할 수 없다.

다음 작업은 248 기준을 실제로 생성한 WL145 report/mask provenance를 복구한
뒤, 그것을 immutable input으로 고정하고 WL147을 새로 시작하는 것이다. 현재
artifact만으로 248을 맞추기 위해 규칙을 바꾸는 것은 WL147 계약상 허용되지
않는다.
