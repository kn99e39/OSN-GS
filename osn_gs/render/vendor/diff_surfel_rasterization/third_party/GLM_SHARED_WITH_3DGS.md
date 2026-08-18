# glm is shared with the vendored 3DGS rasterizer

The official `diff-surfel-rasterization` repository carries glm as a git
submodule at `third_party/glm` (pinned to
`5c46b9c07008ae65cb81ab79cd677ecc1934b903`, glm 0.9.9.9).

OSN-GS already vendors that exact header tree for the 3DGS rasterizer at
`osn_gs/render/vendor/diff_gaussian_rasterization/third_party/glm`. The two
`glm/` header trees are byte-identical (verified with `diff -r`; only glm's own
docs/tests/CI files, none of which are compiled, are absent from the OSN-GS
copy), so this vendored tree reuses the existing copy instead of duplicating
3 MB of identical headers.

This is a PACKAGING-ONLY deviation from the official repository layout. No
compiled source differs. `osn_gs/render/diff_surfel_loader.py` and
`setup.py` point `-I` at the shared path.
