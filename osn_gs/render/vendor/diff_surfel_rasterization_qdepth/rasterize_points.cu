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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	// OSN-GS DIAGNOSTIC ADDITION (worklog 120, Candidate D): (H, W,
	// OSN_GS_MAX_QUERY_SLOTS) float32 camera-space-z probe depths; a slot
	// <= 0 is unused. Pass an EMPTY tensor to disable the probe entirely,
	// in which case the four query outputs come back at their fill values
	// and the kernel takes exactly the canonical path.
	const torch::Tensor& query_depths,
	// OSN-GS DIAGNOSTIC ADDITION (worklog 122): per-primitive provenance for
	// post-median accounting. Either may be EMPTY to disable its categories.
	const torch::Tensor& primitive_component,
	const torch::Tensor& primitive_representative_class)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  // OSN-GS DIAGNOSTIC ADDITION: per-pixel global surfel index of the T>0.5
  // (median-depth) contributor, -1 where none crossed.
  torch::Tensor out_representative_id = torch::full({H, W}, -1, int_opts);
  // OSN-GS DIAGNOSTIC ADDITION (worklog 108): per-PRIMITIVE (size P, not
  // H*W) 0/1 flag, set iff this surfel passed every forward acceptance
  // check for >=1 pixel in this view (the same forward execution as
  // out_representative_id above).
  torch::Tensor out_forward_accepted = torch::full({P}, 0, int_opts);
  // OSN-GS DIAGNOSTIC ADDITION (worklog 110): bounded per-pixel accepted-
  // contributor provenance -- see cuda_rasterizer/config.h for
  // OSN_GS_MAX_CONTRIB_SLOTS and forward.cu for the exact write semantics.
  // Sparse/streamed by construction (H*W*K, K small), never a full H*W*P
  // matrix. -1 fill for unused ids; out_contrib_count is the true, uncapped
  // per-pixel accepted-contributor count (detects slot-array truncation).
  torch::Tensor out_contrib_ids = torch::full({H, W, OSN_GS_MAX_CONTRIB_SLOTS}, -1, int_opts);
  torch::Tensor out_contrib_post_median = torch::full({H, W, OSN_GS_MAX_CONTRIB_SLOTS}, 0, int_opts);
  torch::Tensor out_contrib_count = torch::full({H, W}, 0, int_opts);
  // OSN-GS DIAGNOSTIC ADDITION (worklog 118): low-pass provenance at the
  // exact same T>0.5 median event -- see forward.cu/forward.h. -1 fill
  // (rho3d/rho2d) matches "no median event at this pixel".
  torch::Tensor out_median_rho3d = torch::full({H, W}, -1.0, float_opts);
  torch::Tensor out_median_rho2d = torch::full({H, W}, -1.0, float_opts);
  torch::Tensor out_median_s_u = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor out_median_s_v = torch::full({H, W}, 0.0, float_opts);
  // OSN-GS DIAGNOSTIC ADDITION (worklog 120, Candidate D): canonical
  // traversal state probed at arbitrary query depths. Fill values mark
  // "never written" (an unused slot, or the probe disabled): T = -1,
  // terminated/reached = -1, prefix_count = -1. The kernel only ever
  // writes slots whose input depth was > 0, so a -1 that survives is
  // always an unused slot and never a silently-defaulted decision.
  const bool query_enabled = query_depths.numel() > 0;
  torch::Tensor out_query_T = torch::full({H, W, OSN_GS_MAX_QUERY_SLOTS}, -1.0, float_opts);
  torch::Tensor out_query_terminated = torch::full({H, W, OSN_GS_MAX_QUERY_SLOTS}, -1, int_opts);
  torch::Tensor out_query_reached = torch::full({H, W, OSN_GS_MAX_QUERY_SLOTS}, -1, int_opts);
  torch::Tensor out_query_prefix_count = torch::full({H, W, OSN_GS_MAX_QUERY_SLOTS}, -1, int_opts);
  // OSN-GS DIAGNOSTIC ADDITION (worklog 121, D value provenance). Same -1
  // "never written" convention as the four outputs above; the two per-pixel
  // depth-order counters are always written, so they start at 0.
  torch::Tensor out_query_resolution_depth = torch::full({H, W, OSN_GS_MAX_QUERY_SLOTS}, -1.0, float_opts);
  torch::Tensor out_query_termination_alpha = torch::full({H, W, OSN_GS_MAX_QUERY_SLOTS}, -1.0, float_opts);
  torch::Tensor out_query_late_front_count = torch::full({H, W, OSN_GS_MAX_QUERY_SLOTS}, -1, int_opts);
  torch::Tensor out_pixel_inversion_count = torch::full({H, W}, 0, int_opts);
  torch::Tensor out_pixel_max_backward_jump = torch::full({H, W}, 0.0, float_opts);
  // OSN-GS DIAGNOSTIC ADDITION (worklog 122): exhaustive post-median
  // accounting. Always written, so these start at 0 rather than -1.
  torch::Tensor out_post_median_counts = torch::full({H, W, OSN_GS_POST_MEDIAN_CATEGORIES}, 0, int_opts);
  torch::Tensor out_post_median_weights = torch::full({H, W, OSN_GS_POST_MEDIAN_CATEGORIES}, 0.0, float_opts);
  torch::Tensor out_total_accepted_weight = torch::full({H, W}, 0.0, float_opts);
  torch::Tensor out_post_median_depth_stats = torch::full({H, W, 3}, 0.0, float_opts);
  const bool component_enabled = primitive_component.numel() > 0;
  const bool representative_enabled = primitive_representative_class.numel() > 0;
  if (component_enabled) {
    CHECK_INPUT(primitive_component);
    if (primitive_component.numel() != P) { AT_ERROR("primitive_component must be empty or have P elements"); }
  }
  if (representative_enabled) {
    CHECK_INPUT(primitive_representative_class);
    if (primitive_representative_class.numel() != P) { AT_ERROR("primitive_representative_class must be empty or have P elements"); }
  }
  if (query_enabled)
  {
    CHECK_INPUT(query_depths);
    if (query_depths.numel() != (long long)H * (long long)W * (long long)OSN_GS_MAX_QUERY_SLOTS) {
      AT_ERROR("query_depths must be empty or have exactly H*W*OSN_GS_MAX_QUERY_SLOTS elements");
    }
  }

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_others.contiguous().data<float>(),
		out_representative_id.contiguous().data<int>(),
		out_forward_accepted.contiguous().data<int>(),
		out_contrib_ids.contiguous().data<int>(),
		out_contrib_post_median.contiguous().data<int>(),
		out_contrib_count.contiguous().data<int>(),
		out_median_rho3d.contiguous().data<float>(),
		out_median_rho2d.contiguous().data<float>(),
		out_median_s_u.contiguous().data<float>(),
		out_median_s_v.contiguous().data<float>(),
		query_enabled ? query_depths.contiguous().data<float>() : nullptr,
		out_query_T.contiguous().data<float>(),
		out_query_terminated.contiguous().data<int>(),
		out_query_reached.contiguous().data<int>(),
		out_query_prefix_count.contiguous().data<int>(),
		out_query_resolution_depth.contiguous().data<float>(),
		out_query_termination_alpha.contiguous().data<float>(),
		out_query_late_front_count.contiguous().data<int>(),
		out_pixel_inversion_count.contiguous().data<int>(),
		out_pixel_max_backward_jump.contiguous().data<float>(),
		component_enabled ? primitive_component.contiguous().data<int>() : nullptr,
		representative_enabled ? primitive_representative_class.contiguous().data<int>() : nullptr,
		out_post_median_counts.contiguous().data<int>(),
		out_post_median_weights.contiguous().data<float>(),
		out_total_accepted_weight.contiguous().data<float>(),
		out_post_median_depth_stats.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_others, radii, geomBuffer, binningBuffer, imgBuffer, out_representative_id, out_forward_accepted, out_contrib_ids, out_contrib_post_median, out_contrib_count, out_median_rho3d, out_median_rho2d, out_median_s_u, out_median_s_v, out_query_T, out_query_terminated, out_query_reached, out_query_prefix_count, out_query_resolution_depth, out_query_termination_alpha, out_query_late_front_count, out_pixel_inversion_count, out_pixel_max_backward_jump, out_post_median_counts, out_post_median_weights, out_total_accepted_weight, out_post_median_depth_stats);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  transMat_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_others.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dtransMat.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
