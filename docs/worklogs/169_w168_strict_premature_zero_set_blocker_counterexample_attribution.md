# Worklog 169 — W168 Strict Premature Zero-Set Blocker Counterexample 원인 Attribution

## 1. Intent alignment

W168의 strict premature relation `z_s < z*`가 historical zero-set의 실제 geometric front-bias인지, float64 ray/intersection·analytic-depth arithmetic의 finite-precision ordering인지 진단했다. W168 fixture와 artifact, historical `SparseProjectiveTSDF` construction/extraction, raw zero-set geometry, W167 first-hit contract, strict semantics 및 모든 component를 변경하지 않았다. epsilon, tolerance, offset, filtering 또는 correction은 추가하지 않았다.

## 2. W168 reproduction

W168의 네 `fixture_audit.npz`를 읽기 전용 기준으로 사용하고 동일 fixture를 다시 구성했다. analytic depth/hit, zero-set first-hit depth/status, triangle ID, component ID, premature mask가 fixture별로 모두 exact 또는 bitwise equal이었다. W168 root report와 네 NPZ의 SHA-256를 W169 실행 전후 비교했으며 모두 동일했다.

W168 premature count도 fronto-parallel `84`, oblique `1,306`, sphere `0`, layered `1,038`, 합계 `2,428`개로 exact 재현됐다.

## 3. Premature magnitude accounting

| fixture | count | world min / median / mean / p95 / max | normalized `/h` min / median / mean / p95 / max |
|---|---:|---|---|
| fronto-parallel plane | 84 | 모두 `4.4408921e-16` | 모두 `8.8817842e-15` |
| oblique plane | 1,306 | `1.5355e-10 / 4.7457e-9 / 5.8224e-9 / 1.3258e-8 / 1.5084e-8` | `3.0711e-9 / 9.4914e-8 / 1.1645e-7 / 2.6517e-7 / 3.0167e-7` |
| curved sphere | 0 | 해당 없음 | 해당 없음 |
| layered two-sheet | 1,038 | `4.4409e-16 / 4.4409e-16 / 4.6163e-16 / 4.4409e-16 / 8.8818e-16` | `8.8818e-15 / 8.8818e-15 / 9.2326e-15 / 8.8818e-15 / 1.7764e-14` |

큰 premature-ray count를 큰 geometry displacement로 해석하지 않았다. prevalence는 `2,428`개지만 stable geometric displacement의 최댓값도 oblique fixture에서 `1.5084e-8 world`, 즉 `3.0167e-7 h`다.

## 4. Hit-point analytic residuals

각 premature ray의 frozen first-hit triangle ID를 유지하고 hit XYZ에서 analytic residual을 기록했다. plane/layered fixture는 exact plane equation `(p-c)·n`, sphere는 radial residual `||p-c||-r`을 사용했다.

- fronto-parallel: high-precision hit residual은 전부 `0`이었다.
- layered: high-precision depth는 analytic front plane과 exact equal이었다. hit residual의 약 `1.25e-28 world`는 Decimal operation rounding scale이며 ordering delta는 exact `0`이다.
- oblique: high-precision hit residual은 전부 음수이며 `/h` min/median/mean/p95/max가 `-2.8494e-7 / -8.9840e-8 / -1.0618e-7 / -5.7270e-9 / -2.7856e-9`다.
- sphere: W168 premature ray가 없어 attribution population이 없다.

## 5. Hit-triangle vertex residuals

모든 premature ray에 대해 선택된 frozen first-hit triangle의 세 vertex를 별도로 평가했다.

- fronto-parallel: `252/252`개 vertex가 exact surface였다.
- layered: analytic front surface를 기준으로 `3,114/3,114`개 vertex가 exact surface였다.
- oblique: 총 `3,918`개 중 analytic plane 앞 `2,955`, exact surface `0`, 뒤 `963`이었다. triangle 자체가 plane을 횡단하지만 실제 first-hit 위치는 앞에 놓였다.
- sphere: premature ray가 없어 평가 대상 vertex가 없다.

## 6. High-precision ordering check

Windows의 `np.longdouble`은 float64보다 높은 mantissa precision을 제공하지 않으므로, frozen float input을 `Decimal.from_float`로 exact 변환한 뒤 80-digit Möller–Trumbore와 analytic plane/quadratic intersection을 premature selected triangle에만 적용했다.

| fixture | high-precision strict premature 유지 | exact equal로 제거 | attribution |
|---|---:|---:|---|
| fronto-parallel plane | 0 | 84 | `NUMERICAL_ORDERING_ONLY` |
| oblique plane | 1,306 | 0 | `GEOMETRIC_FRONT_BIAS` |
| curved sphere | 0 | 0 | `NO_PREMATURE_RAYS` |
| layered two-sheet | 0 | 1,038 | `NUMERICAL_ORDERING_ONLY` |

invalid/unresolved high-precision selected-triangle case와 ordering reversal은 모두 `0`이다. oblique의 1,306개는 high precision에서도 sign과 magnitude가 안정적으로 유지됐다.

## 7. Geometry vs numerical attribution

질문에 대한 답은 **`MIXED`**다. W168 strict premature counterexample에는 두 원인이 모두 존재한다.

- fronto-parallel 및 layered population `1,122`개는 finite-precision comparison artifact다.
- oblique population `1,306`개는 frozen zero-set triangle intersection이 analytic plane보다 실제로 앞에 있는 intrinsic geometric front-bias다.
- high-precision에서 strictly premature로 남은 ray는 `1,306`, exact equal로 바뀐 ray는 `1,122`, reversal과 unresolved는 각각 `0`이다.

## 8. Architecture interpretation

W168 counterexample 전체를 “수치 오차뿐”이라고 해석할 수 없고, 반대로 count `2,428` 전체를 material한 geometry displacement라고 해석해서도 안 된다. stable geometric front-bias는 실제이지만 최대 크기는 `3.0167e-7 h`다. W168의 strict no-epsilon failure는 그대로 보존하며, W169에서 correction이나 architecture 변경은 승인하지 않는다.

## 9. Retained / rejected / open

- 결과: `output/169_w168_strict_premature_zero_set_blocker_counterexample_attribution/`
- fixture별 `premature_attribution.npz`에는 ray/triangle/component ID, float64 depth, hit/vertex residual, Decimal depth/residual 문자열과 attribution을 보존했다.
- 새 visualization은 만들지 않았다. numeric report가 exact diagnostic이며 PNG/PPM, W153 replay-cache temp mirror도 생성하지 않았다.
- W169 focused tests: `4 passed`; analytic residual, exact/front-shift high-precision reference, no-epsilon attribution, layered W168 reproduction/magnitude accounting을 확인했다. W168/W169 combined tests도 `9 passed`했다.
- retained: W168 result, strict semantics, historical geometry/extraction, W167 first-hit, 모든 component.
- rejected: epsilon/tolerance, geometry offset/repair, component filtering/trusted subset, blocker semantics 수정, NURBS/global aggregation.
- open: 이 attribution에 대한 향후 architecture response는 별도 승인된 batch가 필요하다.
