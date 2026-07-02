# OSN-GS Architecture

OSN-GS??愿痢〓맂 ?쒕㈃ ?꾩뿉 ?ъ궗??Gaussian?ㅼ쓣 ?댁슜??NURBS 湲곕컲??parametric surface瑜?援ъ꽦?섍퀬, 洹?援ъ“???뱀꽦??諛뷀깢?쇰줈 媛?ㅼ쭊 ?쒕㈃??Gaussian??諛곗튂?섎뒗 3D Gaussian Splatting ?꾨젅?꾩썙?ъ씠??

?듭떖 紐⑺몴??湲곗〈 3DGS媛 ?쏀븳 鍮꾧?痢??곸뿭???쒕㈃ 援ъ“瑜??⑥닚 visibility, MVS, pseudo-view 蹂닿컯???꾨땲??愿痢??쒕㈃?먯꽌 異붿텧??援ъ“??prior濡??덉륫?섎뒗 寃껋씠??

## Core Idea

1. 湲곗〈 3DGS? 媛숈씠 珥덇린 Gaussian???앹꽦?섍퀬 ?숈뒿???쒖옉?쒕떎.
2. 珥덇린 Gaussian?ㅼ쓣 point cloud濡?媛꾩＜??愿痢〓맂 ?쒕㈃ ?꾩쓽 base curve瑜?異붿젙?쒕떎.
3. base curve?ㅼ쓽 援ъ“???곗냽?? 怨〓쪧, 諛⑺뼢?? 諛섎났 ?⑦꽩???댁슜??occluded space????묐릺??curve瑜??앹꽦?쒕떎.
4. 愿痢?curve? 異붿젙 curve瑜??④퍡 ?ъ슜??NURBS surface瑜?援ъ꽦?쒕떎.
5. NURBS surface ?꾩쓽 鍮꾧?痢??곸뿭??uncertain Gaussian???섑뵆留곹븳??
6. ?대?吏 湲곕컲 ?숈뒿??諛섎났?섎㈃??certain Gaussian? ?쇰컲 3DGS 諛⑹떇?쇰줈 理쒖쟻?뷀븯怨? uncertain Gaussian? ?뚮뜑留?loss? surface consistency瑜??댁슜???꾩튂? surface basis瑜?媛깆떊?쒕떎.

## Motivation

### Why 3DGS

鍮꾧?痢??쒕㈃???덉륫?섎젮硫?愿痢??쒕㈃?쇰줈遺??紐낆떆?곸씤 援ъ“ representation???살쓣 ???덉뼱???쒕떎. NeRF??NeuS 怨꾩뿴? scene??implicit field濡??쒗쁽?섎?濡? 愿痢〓맂 ?쒕㈃ ?꾩쓽 援ъ“???⑦꽩??吏곸젒 異붿텧???덈줈??Gaussian 諛곗튂濡??곌껐?섍린 ?대졄??

3DGS??Gaussian???꾩튂, covariance, opacity, color媛 紐낆떆?곸쑝濡?議댁옱?섎?濡??ㅼ쓬 ?묒뾽???곹빀?섎떎.

- 愿痢??쒕㈃ point cloud 異붿텧
- surface curve fitting
- 鍮꾧?痢??곸뿭 ?꾨낫 ?꾩튂 ?섑뵆留?
- certain/uncertain Gaussian 遺꾨━ ?숈뒿
- density control ?⑦꽩 遺꾩꽍

### Why NURBS

鍮꾧?痢??쒕㈃??Gaussian??諛곗튂?섎젮硫??꾩껜 ?쒕㈃ 援ъ“瑜??쒗쁽?섎뒗 以묎컙 representation???꾩슂?섎떎.

Mesh??vertex 諛곗튂?????蹂꾨룄 prior媛 ?꾩슂?섍퀬, 愿痢〓맂 vertex 媛꾩쓽 援ъ“??愿怨꾨? ?덉젙?곸쑝濡?異붿텧?섍린 ?대졄?? SDF??愿痢??대?吏?먯꽌 ?쒕㈃??李얜뒗 ??媛뺤젏???덉쑝?? 愿痢〓릺吏 ?딆? ?쒕㈃??援ъ“?곸쑝濡??몄궫?섎뒗 ?곗뿉??吏곸젒?곸씤 ?쒖빟??遺議깊븯??

NURBS??control point, knot, degree, weight瑜??듯빐 ?곗냽?곸씤 parametric surface瑜??쒗쁽?????덉쑝誘濡??ㅼ쓬 ?μ젏???덈떎.

- 愿痢?curve?먯꽌 鍮꾧?痢?curve濡?援ъ“瑜??뺤옣?섍린 ?쎈떎.
- smoothness, curvature, continuity瑜?紐낆떆?곸쑝濡??쒖뼱?????덈떎.
- surface parameter domain ?꾩뿉??Gaussian???덉젙?곸쑝濡??섑뵆留곹븷 ???덈떎.
- base curve ?ш퀎?곗쓣 ?듯빐 uncertain Gaussian???꾩튂 蹂댁젙怨??곌껐?섍린 ?쎈떎.

## Current Stage Boundary

현재 구현의 우선순위는 전체 3DGS 확장 루프가 아니라 **Stage 1: Visible NURBS Reconstruction**이다.

Stage 1 입력은 COLMAP/초기 Gaussian에서 얻은 관측 Gaussian center와 color이며, 출력은 visible surface만 설명하는 NURBS-like parametric representation이다.

Stage 1에서 수행하는 작업:

- 관측 Gaussian center를 visible surface point cloud로 사용한다.
- visible point cloud에서 base curve set을 추정한다.
- PCA 기반 2D parameter domain 위에 visible surface control grid를 fitting한다.
- `(u, v) -> xyz` surface evaluator와 smoothness regularization을 제공한다.

Stage 1에서 의도적으로 제외하는 작업:

- occlusion curve prediction
- occluded surface extrapolation
- uncertain Gaussian sampling
- uncertain/certain joint optimization
- uncertain confidence 기반 pruning/promotion

이 제외 항목들은 Stage 2 이후에 별도 파이프라인으로 다시 연결한다.
## High-Level Pipeline

```text
Scene Loader
    -> Initial 3DGS Gaussians
    -> Observed Surface Point Cloud
    -> Base Curve Fitting
    -> Occlusion Curve Prediction
    -> NURBS Surface Construction
    -> Uncertain Gaussian Sampling
    -> Joint Optimization
    -> Curve and Surface Update
```

## Gaussian Types

### Certain Gaussian

Certain Gaussian? 湲곗〈 3DGS ?숈뒿 怨쇱젙?먯꽌 ?앹꽦?섍굅?? 異⑸텇??愿痢?洹쇨굅媛 ?덈뒗 Gaussian?대떎.

二쇱슂 ?숈뒿 ?좏샇:

- image similarity loss
- opacity and scale regularization
- standard 3DGS adaptive density control
- color and spherical harmonics optimization

?대떦 紐⑤뱢:

- `osn_gs/gaussian/certain_gaussians.py`
- `osn_gs/gaussian/projection.py`
- `osn_gs/losses/image_similarity.py`

### Uncertain Gaussian

Uncertain Gaussian? NURBS surface??鍮꾧?痢??곸뿭 ?꾩뿉 諛곗튂?섎뒗 Gaussian?대떎. ??Gaussian? 吏곸젒 愿痢〓맂 ?쒕㈃?먯꽌 ??寃껋씠 ?꾨땲誘濡? 珥덇린?먮뒗 援ъ“??異붿젙??湲곕컲???꾨낫濡?痍④툒?쒕떎.

以묒슂???댁꽍:

Uncertain Gaussian??image similarity loss瑜??ш쾶 諛쒖깮?쒗궓?ㅻ㈃, ?대뒗 ?대떦 Gaussian???섎せ???꾩튂 ?먮뒗 ?섎せ??surface ?꾩뿉 諛곗튂?섏뿀??媛?μ꽦???섎??쒕떎. ?곕씪???⑥닚??Gaussian parameter留?理쒖쟻?뷀븯??寃껋씠 ?꾨땲?? NURBS base curve? surface ?먯껜瑜??ш퀎?고븯???좏샇濡??ъ슜?쒕떎.

二쇱슂 ?숈뒿 ?좏샇:

- image similarity residual
- NURBS surface consistency
- neighboring certain Gaussian cluster prior
- curve smoothness and continuity prior

?대떦 紐⑤뱢:

- `osn_gs/gaussian/uncertain_gaussians.py`
- `osn_gs/surface/sampling.py`
- `osn_gs/optim/curve_update.py`
- `osn_gs/losses/uncertainty.py`

## Surface Reconstruction

### Observed Point Cloud

珥덇린 Gaussian??center瑜?point cloud濡??ъ슜?쒕떎. ?꾩슂?섎떎硫?opacity, scale, normal confidence, visibility score瑜?湲곗??쇰줈 愿痢??쒕㈃???대떦?섎뒗 Gaussian留??꾪꽣留곹븳??

?대떦 紐⑤뱢:

- `osn_gs/surface/point_cloud.py`

### Base Curve Fitting

愿痢??쒕㈃ point cloud?먯꽌 base curve瑜??앹꽦?쒕떎. 珥덇린 援ы쁽?먯꽌???ㅼ쓬 湲곗???怨좊젮?쒕떎.

- local geometry grouping
- normal consistency
- principal direction estimation
- color cluster consistency
- camera visibility confidence

?대떦 紐⑤뱢:

- `osn_gs/surface/base_curves.py`
- `osn_gs/surface/structural_prior.py`

### Occlusion Curve Prediction

base curve??諛⑺뼢?? 怨〓쪧, 媛꾧꺽, 諛섎났?깆쓣 ?댁슜??occluded space ?덉쓽 curve瑜?異붿젙?쒕떎. ???④퀎??OSN-GS???듭떖 李⑤퀎?먯씠??

異붿젙??curve??吏곸젒 ?뺣떟???꾨땲??NURBS surface瑜??앹꽦?섍린 ?꾪븳 structural hypothesis濡?痍④툒?쒕떎.

?대떦 紐⑤뱢:

- `osn_gs/surface/occlusion_curves.py`
- `osn_gs/surface/structural_prior.py`

### NURBS Surface Construction

愿痢?base curve? 異붿젙 occlusion curve瑜??④퍡 ?ъ슜??NURBS surface瑜?留뚮뱺??

NURBS surface???ㅼ쓬 ?뺣낫瑜?媛吏꾨떎.

- control points
- degree
- knot vectors
- weights
- parameter domain
- observed/occluded region mask

?대떦 紐⑤뱢:

- `osn_gs/surface/nurbs_surface.py`

## Color Assignment

Uncertain Gaussian???됱긽? 吏곸젒 愿痢〓맂 ?됱긽???놁쑝誘濡?certain Gaussian???됱긽 遺꾪룷瑜?湲곕컲?쇰줈 珥덇린?뷀븳??

珥덇린 ?꾨왂:

1. Certain Gaussian???됱긽 ?먮뒗 spherical harmonics coefficient 湲곗??쇰줈 clustering?쒕떎.
2. 媛?uncertain Gaussian??媛??媛源뚯슫 surface region ?먮뒗 curve neighborhood??cluster???좊떦?쒕떎.
3. ?좊떦??cluster??color prior瑜?uncertain Gaussian??怨듭쑀?쒕떎.
4. ?숈뒿 以?image residual????븘吏??諛⑺뼢?쇰줈 color parameter瑜??쒗븳?곸쑝濡??낅뜲?댄듃?쒕떎.

?대떦 紐⑤뱢:

- `osn_gs/gaussian/color_clusters.py`

## Adaptive Density Control

Certain Gaussian? 湲곗〈 3DGS??adaptive density control???곕Ⅸ??

Uncertain Gaussian? ?낅┰?곸쑝濡?densify/prune?섍린蹂대떎, 媛숈? color/geometry cluster???랁븳 certain Gaussian??ADC ?⑦꽩??紐⑤갑?쒕떎.

珥덇린 ?꾨왂:

- 媛숈? cluster??certain Gaussian split/clone/prune ?듦퀎瑜?湲곕줉?쒕떎.
- surface parameter domain ?꾩뿉??uncertain Gaussian??density瑜?蹂댁젙?쒕떎.
- image residual??吏?띿쟻?쇰줈 ??uncertain Gaussian? ?꾩튂 ?대룞 ?먮뒗 curve update ?꾨낫濡??섍릿??
- confidence媛 異⑸텇???믪븘吏?uncertain Gaussian? certain Gaussian?쇰줈 ?밴꺽?????덈떎.

?대떦 紐⑤뱢:

- `osn_gs/gaussian/density_control.py`

## Training Loop

```text
for iteration in training_iterations:
    batch = scene_loader.sample_views()

    rendered = rasterizer.render(
        certain_gaussians,
        uncertain_gaussians,
        cameras=batch.cameras,
    )

    certain_loss = image_similarity(rendered, batch.images)
    uncertain_loss = uncertainty_loss(rendered, batch.images, nurbs_surface)
    surface_loss = nurbs_regularization(nurbs_surface)

    total_loss = certain_loss + uncertain_loss + surface_loss
    total_loss.backward()

    update_certain_gaussians()
    update_uncertain_gaussians()

    if should_update_curves(iteration):
        update_base_curves()
        update_occlusion_curves()
        rebuild_nurbs_surface()
        resample_uncertain_gaussians()

    if should_run_density_control(iteration):
        run_certain_adc()
        run_uncertain_adc_from_cluster_patterns()
```

?대떦 紐⑤뱢:

- `osn_gs/core/trainer.py`
- `osn_gs/core/pipeline.py`
- `osn_gs/core/state.py`
- `osn_gs/render/rasterizer_adapter.py`

## Module Responsibilities

### `osn_gs/core`

?꾩껜 ?꾨젅?꾩썙?ъ쓽 ?ㅽ뻾 ?먮쫫??愿由ы븳??

- `framework.py`: OSN-GS ?곸쐞 API
- `pipeline.py`: surface construction怨?Gaussian update ?④퀎 ?곌껐
- `trainer.py`: ?숈뒿 猷⑦봽
- `state.py`: Gaussian, NURBS, optimizer, iteration state 蹂닿?

### `osn_gs/gaussian`

Certain/uncertain Gaussian???앹꽦, ?낅뜲?댄듃, ?됱긽, density control???대떦?쒕떎.

- `certain_gaussians.py`: 愿痢?湲곕컲 Gaussian container? update
- `uncertain_gaussians.py`: NURBS 湲곕컲 Gaussian container? confidence 愿由?
- `projection.py`: Gaussian projection 諛?observed surface point 異붿텧
- `color_clusters.py`: ?됱긽 湲곕컲 cluster prior
- `density_control.py`: certain ADC? uncertain ADC 紐⑤갑 ?뺤콉

### `osn_gs/surface`

Point cloud?먯꽌 curve, NURBS surface, Gaussian sampling ?꾩튂瑜??앹꽦?쒕떎.

- `point_cloud.py`: Gaussian center 湲곕컲 point cloud 蹂?섍낵 ?꾪꽣留?
- `base_curves.py`: 愿痢??쒕㈃ base curve fitting
- `occlusion_curves.py`: 鍮꾧?痢??곸뿭 curve prediction
- `nurbs_surface.py`: NURBS surface representation
- `sampling.py`: NURBS surface ??Gaussian sampling
- `structural_prior.py`: curve continuity, curvature, repetition prior

### `osn_gs/losses`

?대?吏 湲곕컲 loss? surface 愿??regularization???뺤쓽?쒕떎.

- `image_similarity.py`: L1, SSIM, perceptual ?뺥깭??image loss
- `nurbs_regularization.py`: smoothness, curvature, continuity regularization
- `uncertainty.py`: uncertain Gaussian confidence? position correction loss

### `osn_gs/render`

湲곗〈 3DGS rasterizer瑜?OSN-GS ?숈뒿 猷⑦봽??留욊쾶 媛먯떬??

- `rasterizer_adapter.py`: certain/uncertain Gaussian???④퍡 ?뚮뜑留곹븯??adapter

### `osn_gs/optim`

Curve? surface 媛깆떊 ?뺤콉, scheduler瑜??대떦?쒕떎.

- `curve_update.py`: uncertain residual??湲곕컲?쇰줈 base curve? occlusion curve ?ш퀎??
- `schedulers.py`: curve update, NURBS rebuild, ADC ?ㅽ뻾 二쇨린 愿由?

### `osn_gs/data`

Scene, camera, image batch瑜?濡쒕뱶?쒕떎.

- `scene_loader.py`: dataset entry point
- `cameras.py`: camera parameter wrapper

## Key Research Questions

1. 愿痢?Gaussian?먯꽌 ?덉젙?곸씤 base curve瑜??대뼸寃?異붿텧??寃껋씤媛?
2. Occluded curve prediction?먯꽌 ?대뼡 structural prior媛 媛???④낵?곸씤媛?
3. Uncertain Gaussian??image loss瑜??꾩튂 ?ㅻ쪟, ?됱긽 ?ㅻ쪟, ?쒕㈃ ?ㅻ쪟 以?臾댁뾿?쇰줈 ?댁꽍??寃껋씤媛?
4. ?몄젣 uncertain Gaussian??certain Gaussian?쇰줈 ?밴꺽??寃껋씤媛?
5. Certain Gaussian??ADC ?⑦꽩??uncertain Gaussian???대뒓 ?뺣룄源뚯? 紐⑤갑?쒗궗 寃껋씤媛?
6. NURBS surface update媛 ?덈Т ??쓣 ???숈뒿 ?덉젙?깆씠 源⑥?吏 ?딅룄濡??대뼡 scheduler瑜???寃껋씤媛?

## Implementation Roadmap

### Phase 1: Skeleton and Baseline Bridge

- 湲곗〈 3DGS ?숈뒿 肄붾뱶? ?곌껐??rasterizer adapter ?묒꽦
- Gaussian container ?명꽣?섏씠???뺤쓽
- Scene loader? config 援ъ“ ?뺤쓽
- 湲곕낯 train script ?묒꽦

?꾩옱 援ы쁽 ?곹깭:

- `TorchGaussianModel`? 3DGS??`GaussianModel` ?띿꽦 怨꾩빟怨??좎궗?섍쾶 `get_xyz`, `get_features`, `get_opacity`, `get_scaling`, `get_rotation`???쒓났?쒕떎.
- `TorchRasterizerAdapter`??`diff_gaussian_rasterization`???ㅼ튂?섏뼱 ?덉쑝硫?CUDA rasterizer瑜??ъ슜?섍퀬, ?놁쑝硫?Torch fallback renderer瑜??ъ슜?쒕떎.
- `TorchOSNGSTrainer`???숈뒿 寃곌낵濡?PLY, PPM preview, Torch checkpoint, metrics file????ν븳??

### Phase 2: Observed Surface Curves

- Gaussian center瑜?point cloud濡?蹂??
- 愿痢?confidence 湲곕컲 filtering
- base curve fitting prototype ?묒꽦
- curve visualization ?꾧뎄 ?묒꽦

### Phase 3: NURBS Surface and Uncertain Gaussian

- NURBS surface representation 援ы쁽
- surface parameter domain sampling 援ы쁽
- uncertain Gaussian container 援ы쁽
- color cluster 湲곕컲 珥덇린 ?됱긽 ?좊떦

### Phase 4: Joint Optimization

- certain/uncertain Gaussian joint rendering
- uncertainty loss 異붽?
- curve update scheduler 異붽?
- NURBS rebuild? uncertain resampling ?곌껐

### Phase 5: ADC and Promotion Policy

- certain cluster蹂?ADC ?⑦꽩 湲곕줉
- uncertain Gaussian density control 援ы쁽
- uncertain to certain promotion rule 援ы쁽
- ablation ?ㅽ뿕 援ъ꽦

## Expected Contributions

- 愿痢??쒕㈃??援ъ“???뱀꽦???댁슜??鍮꾧?痢??쒕㈃??異붿젙?섎뒗 3DGS ?뺤옣 ?꾨젅?꾩썙??
- NURBS 湲곕컲 parametric surface瑜?3DGS ?숈뒿 猷⑦봽??寃고빀?섎뒗 諛⑸쾿
- Certain/uncertain Gaussian 遺꾨━? confidence 湲곕컲 媛깆떊 ?꾨왂
- Color cluster? ADC pattern transfer瑜??댁슜??鍮꾧?痢?Gaussian 珥덇린??諛?density control


## Ongoing Context Log

- 2026-07-01: User requested that whenever the environment, project situation, or task direction changes, the relevant `.md` files should be updated with that context instead of relying only on chat history.
- 2026-07-01: NURBS is an intermediate representation, not a replacement final output. Training should keep Gaussian primitives as the main output while preserving visible NURBS reconstruction data for later visualization tools.
- 2026-07-01: The Colab training notebook should pass NURBS-related configuration alongside OSN-GS training/Gaussian primitive output handling so downstream visualization can consume both Gaussian and NURBS artifacts.

- 2026-07-01: WebRenderer PLY compatibility request. Renderer requires Graphdeco-style Gaussian fields `x`, `y`, `z`, `f_dc_0..2`, raw `opacity`, optional raw log `scale_0..2`, and `rot_0..3`. OSN-GS has corresponding primitives in `TorchGaussianModel`, so `save_ply` should emit those names instead of debug-only RGB/`scale_x` fields.
- 2026-07-01: Notebook output packaging now includes NURBS visualization data. OSN-GS output inspection creates `visualization_manifest.json` under `MODEL_ROOT`, pairing each `point_cloud.ply` with its sibling `nurbs_surface.json` so external tools can load Gaussian primitives and the visible NURBS intermediate together.
- 2026-07-02: Added `visible_surface_resolution_scale` so Stage 1 visible NURBS control-grid density can be increased from the notebook Train cell without changing the base U/V parameters. Final resolution is computed from `visible_surface_resolution_u/v * scale`.
