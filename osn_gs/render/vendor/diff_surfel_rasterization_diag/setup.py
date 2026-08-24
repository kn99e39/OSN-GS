from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

# OSN-GS Worklog 107 diagnostic-only build. A sibling of
# ../diff_surfel_rasterization/setup.py with exactly one difference: the
# extension/package name, so this never collides with (or is mistaken for)
# the canonical vendored package. Source diffs are documented and marked
# "OSN-GS DIAGNOSTIC ADDITION" inline in this directory's own .cu/.h files.

setup(
    name="diff_surfel_rasterization_diag",
    packages=['diff_surfel_rasterization_diag'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_rasterization_diag._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diff_gaussian_rasterization", "third_party", "glm")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
