# 28. UV 트림 마스크 수명주기

날짜: 2026-07-15

## 작업

trimming이 활성인 경우 maintenance가 Gaussian UV binding을 갱신한 뒤 모든 patch의 UV support mask를 다시 구축하도록 변경했다. maintenance report는 support_masks_refreshed를 노출한다.

## 검증

NURBS surface와 training-regression test module을 실행했다.

결과: 24개 test 통과.

## 결과

초기화 시점의 trim mask가 maintenance UV projection 이후 stale 상태로 남지 않는다. TODO lifecycle 항목을 제거했다. mask와 UV version metadata는 이후 diagnostic 과제로 남아 있다.
