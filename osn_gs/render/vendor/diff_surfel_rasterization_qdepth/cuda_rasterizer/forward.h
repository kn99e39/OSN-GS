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
		float* out_median_s_v,
		// OSN-GS DIAGNOSTIC ADDITION (worklog 120, Candidate D): arbitrary-
		// query-depth probe of the canonical traversal. `query_depths` is
		// (H, W, OSN_GS_MAX_QUERY_SLOTS) camera-space z; a slot value <= 0 is
		// an unused slot. For every used slot the kernel reports, WITHOUT
		// altering any canonical computation:
		//   out_query_T            -- running transmittance T at the moment
		//                             canonical traversal first reaches a
		//                             contributor whose own `depth` >= this
		//                             slot's query depth (i.e. T accumulated
		//                             from every accepted contributor strictly
		//                             before it), or T at canonical
		//                             termination / list exhaustion if the
		//                             query depth is never reached.
		//   out_query_terminated   -- 1 iff the canonical termination
		//                             condition (`T * (1 - alpha) < 0.0001`,
		//                             the kernel's own, unmodified) fired at a
		//                             contributor strictly before the query
		//                             depth was reached.
		//   out_query_reached      -- 1 iff traversal actually reached a
		//                             contributor at or beyond the query depth.
		//   out_query_prefix_count -- number of ACCEPTED contributors composited
		//                             before resolution (provenance only).
		const float* query_depths,
		float* out_query_T,
		int* out_query_terminated,
		int* out_query_reached,
		int* out_query_prefix_count,
		// OSN-GS DIAGNOSTIC ADDITION (worklog 121, D value provenance).
		// Purely additive: none of the outputs above is read, written, or
		// ordered differently because of these.
		//   out_query_resolution_depth  -- per-pixel `depth` of the accepted
		//                                  (or termination) event that resolved
		//                                  the slot; -1 if never resolved.
		//   out_query_termination_alpha -- the canonical `alpha` at the
		//                                  termination event, written ONLY for
		//                                  slots whose verdict is terminated=1,
		//                                  so `test_T = T_pre * (1 - alpha)` can
		//                                  be reconstructed host-side; -1 else.
		//   out_query_late_front_count  -- accepted events processed AFTER the
		//                                  slot resolved whose own per-pixel
		//                                  depth is still < the query depth
		//                                  (traversal-order vs physical-depth
		//                                  gap); 0 when resolved with none, -1
		//                                  when never resolved.
		//   out_pixel_inversion_count   -- (H, W) accepted events whose depth is
		//                                  below the running max accepted depth.
		//   out_pixel_max_backward_jump -- (H, W) largest such backward step.
		float* out_query_resolution_depth,
		float* out_query_termination_alpha,
		int* out_query_late_front_count,
		int* out_pixel_inversion_count,
		float* out_pixel_max_backward_jump,
		// OSN-GS DIAGNOSTIC ADDITION (worklog 122, candidate B frontier
		// validation). Exhaustive accounting of accepted contributors that
		// occur AFTER the canonical median-surface event, using the SAME
		// post-median test worklog 110 already uses (`T <= 0.5` at
		// acceptance, T pre-update). Purely additive.
		//   primitive_component            (P,) frozen visible component id
		//                                  per surfel, -1 unresolved; empty
		//                                  disables categories 1-3.
		//   primitive_representative_class (P,) 2 = median representative in
		//                                  THIS view, 1 = in another view
		//                                  only, 0 = never; empty disables 4-6.
		//   out_post_median_counts         (H, W, 8) per-category counts
		//   out_post_median_weights        (H, W, 8) per-category sum of the
		//                                  canonical compositing weight w=alpha*T
		//   out_total_accepted_weight      (H, W) sum of w over ALL accepted
		//                                  contributors, so a post-median
		//                                  contribution FRACTION is computable
		//   out_post_median_depth_stats    (H, W, 3) sum / min / max of
		//                                  (contributor depth - median depth)
		const int* primitive_component,
		const int* primitive_representative_class,
		int* out_post_median_counts,
		float* out_post_median_weights,
		float* out_total_accepted_weight,
		float* out_post_median_depth_stats);
}


#endif
