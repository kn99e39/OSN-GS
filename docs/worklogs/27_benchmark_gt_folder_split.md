# 27. Benchmark GT Renderer Folder Split

Date: 2026-07-16

## Work

Changed synthetic constructor renderer export from one folder containing generated nurbs_surface.json plus GT nurbs_surface_gt.json to separate renderer-loadable sibling folders. Generated output is NURBS_output/scene with point_cloud.ply and nurbs_surface.json. GT is NURBS_output/scene_gt with nurbs_surface.json.

## Verification

Ran the sine CPU benchmark. The export produced NURBS_output/sine containing point_cloud.ply and nurbs_surface.json, and NURBS_output/sine_gt containing one nurbs_surface.json.
