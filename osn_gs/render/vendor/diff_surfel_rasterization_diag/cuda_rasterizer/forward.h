/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		// float* isovals,
		// float3* normals,
		float* transMats,
		float* colors,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float2* points_xy_image,
		const float* features,
		const float* transMats,
		const float* depths,
		const float4* normal_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_others,
		// OSN-GS DIAGNOSTIC ADDITION (not in the canonical vendored copy):
		// global surfel index of the median-transmittance-crossing (T>0.5)
		// contributor at each pixel, -1 where none crossed. Read-only,
		// diagnostic-only -- see osn_gs/render/torch_surfel_contribution_diagnostics.py.
		int* out_representative_id,
		// OSN-GS DIAGNOSTIC ADDITION (worklog 108): per-PRIMITIVE (size P,
		// not H*W) 0/1 flag -- set to 1 iff this surfel passed every forward
		// acceptance check (depth/power/alpha/test_T) for >=1 pixel in this
		// view, i.e. the SAME "accepted contributor" semantics as the
		// canonical forward kernel's own compositing loop, captured in the
		// same execution as out_representative_id above.
		int* out_forward_accepted,
		// OSN-GS DIAGNOSTIC ADDITION (worklog 110): bounded per-pixel
		// accepted-contributor provenance -- (H, W, OSN_GS_MAX_CONTRIB_SLOTS)
		// global surfel ids (-1 = unused slot) and matching pre/post-median
		// flags (see config.h and forward.cu for the exact traversal-order
		// semantics), plus the true uncapped per-pixel accepted count.
		int* out_contrib_ids,
		int* out_contrib_post_median,
		int* out_contrib_count,
		// OSN-GS DIAGNOSTIC ADDITION (worklog 118): the exact low-pass-filter
		// provenance of the median-transmittance-crossing (T>0.5) event at
		// each pixel -- rho3d (true ray-plane intersection distance in the
		// surfel's own local uv system) and rho2d (screen-space low-pass
		// floor), plus the surfel-local intersection coordinates `s` that
		// `depth` is itself derived from at that same event. -1 where no
		// contributor crossed T=0.5. Never used to change acceptance/depth;
		// read-only, diagnostic-only.
		float* out_median_rho3d,
		float* out_median_rho2d,
		float* out_median_s_u,
		float* out_median_s_v);
}


#endif
