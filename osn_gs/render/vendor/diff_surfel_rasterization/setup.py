#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_surfel_rasterization",
    packages=['diff_surfel_rasterization'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            # OSN-GS PACKAGING-ONLY DEVIATION: glm headers are shared with the
            # vendored 3DGS rasterizer instead of being duplicated here. The
            # header trees are byte-identical -- see
            # third_party/GLM_SHARED_WITH_3DGS.md. No compiled source differs.
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diff_gaussian_rasterization", "third_party", "glm")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
