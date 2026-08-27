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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

// OSN-GS DIAGNOSTIC ADDITION (worklog 110): fixed per-pixel capacity for the
// accepted-contributor provenance slots (`out_contrib_ids`/`out_contrib_
// post_median` in forward.cu/rasterize_points.cu) -- a bounded, sparse/
// streamed representation, NOT a full (H*W*P) matrix. Pixels with more than
// this many accepted contributors have their overflow silently truncated in
// the slot arrays but the true count is still recorded exactly in
// `out_contrib_count`, so truncation is always detectable, never hidden.
#define OSN_GS_MAX_CONTRIB_SLOTS 16

// OSN-GS DIAGNOSTIC ADDITION (worklog 120, Candidate D): fixed per-pixel
// capacity for arbitrary-depth QUERY PROBES. Each slot carries one query
// depth (camera-space z, the same quantity the canonical loop's `depth`
// variable holds) and receives back the canonical traversal state at that
// depth. Purely additive: no canonical value is read differently, written
// differently, or gated on these slots. Pixels needing more than this many
// simultaneous probes are handled by the Python driver issuing additional
// render passes -- never by dropping queries.
#define OSN_GS_MAX_QUERY_SLOTS 8

// OSN-GS DIAGNOSTIC ADDITION (worklog 122): number of post-median
// accounting categories. EXHAUSTIVE per-pixel aggregates -- deliberately
// not a bounded contributor list, because worklog 110 measured 97.4% of
// its 16-slot arrays as truncated and that truncation would bias this
// measurement. Categories:
//   0 all post-median accepted contributors
//   1 same frozen visible component as the median representative
//   2 different visible component
//   3 unresolved component provenance
//   4 contributor is itself a median representative in THIS view
//   5 ... in another training view only
//   6 ... never a median representative anywhere
//   7 accepted through the rho2d screen-space low-pass branch
//   8 per-pixel depth is IN FRONT of the median depth despite being
//     post-median in canonical traversal order (the centre-depth sort is
//     not monotone in per-pixel intersection depth -- worklog 121)
//   9 per-pixel depth is at or behind the median depth
#define OSN_GS_POST_MEDIAN_CATEGORIES 10

#endif