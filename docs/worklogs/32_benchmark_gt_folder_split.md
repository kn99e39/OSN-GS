# 32. Benchmark GT Renderer Folder 분리

날짜: 2026-07-16

## 작업

synthetic constructor renderer export를 generated nurbs_surface.json과 GT nurbs_surface_gt.json이 한 folder에 함께 있던 방식에서, renderer가 직접 불러올 수 있는 sibling folder 구조로 변경했다. generated output은 NURBS_output/scene 아래의 point_cloud.ply와 nurbs_surface.json이며, GT는 NURBS_output/scene_gt 아래의 nurbs_surface.json이다.

## 검증

sine CPU benchmark를 실행했다. export는 NURBS_output/sine에 point_cloud.ply와 nurbs_surface.json을 생성했고, NURBS_output/sine_gt에는 nurbs_surface.json 하나를 생성했다.
