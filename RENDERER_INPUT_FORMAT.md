# 3DGS Renderer Input Format

This document describes the file and WebSocket formats accepted by
`3DGS_Renderer`.

For an SSH-forwarded trainer server, enter the local tunnel endpoint in the
renderer, for example `ws://localhost:8080`. The transport address changes,
but this WebSocket payload format does not.

## 1. Local File Input

The file picker accepts multiple files and directory selection:

```html
<input type="file" accept=".ply,.json" multiple webkitdirectory>
```

Supported files:

- Gaussian `.ply`
- NURBS `nurbs_surface.json`

Files with unsupported extensions are ignored.

## 2. Iteration Naming

When files are loaded together, the renderer groups them by iteration.

Recommended directory layout:

```text
scene_output/
  iteration_001000/
    point_cloud.ply
    nurbs_surface.json
  iteration_003000/
    point_cloud.ply
    nurbs_surface.json
  iteration_010000/
    point_cloud.ply
    nurbs_surface.json
```

The directory name must match:

```text
iteration_<integer>
```

For example, `iteration_010000` becomes iteration `10000`.

If no `iteration_<integer>` directory is found, the renderer falls back to
the file order and uses the zero-based file index as the iteration value.

Files with the same inferred iteration are combined into one snapshot.

## 3. Gaussian PLY Format

The PLY parser accepts these PLY encodings:

```text
format ascii 1.0
format binary_little_endian 1.0
```

`binary_big_endian` is not supported.

The PLY must contain a `vertex` element and these required properties:

```text
x
y
z
f_dc_0
f_dc_1
f_dc_2
opacity
```

For a complete Graphdeco 3DGS PLY, also include:

```text
scale_0
scale_1
scale_2
rot_0
rot_1
rot_2
rot_3
```

Typical header:

```text
ply
format binary_little_endian 1.0
element vertex <COUNT>
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
```

The parser interprets Graphdeco parameters as follows:

```text
position = [x, y, z]
scale    = exp([scale_0, scale_1, scale_2])
rotation = [rot_0, rot_1, rot_2, rot_3]
opacity  = sigmoid(opacity)
color    = clamp01(0.5 + 0.28209479177387814 * [f_dc_0, f_dc_1, f_dc_2])
```

Additional PLY properties are preserved in the raw Gaussian data where
possible, but the renderer currently uses the DC color and does not evaluate
view-dependent spherical-harmonic colors.

The PLY parser does not support list properties on Gaussian vertices.

## 4. NURBS Surface JSON

The filename must be exactly:

```text
nurbs_surface.json
```

For a single-patch payload, the required field is:

```json
{
  "control_grid": [
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
  ]
}
```

The renderer treats the first array axis as **U** and the second as **V**:

```text
control_grid[u][v] = [x, y, z]
control_grid_shape = [U, V, 3]
```

`u` and `v` are both evaluated over the normalized domain `[0, 1]`. The
renderer generates clamped-uniform knot vectors from each patch's control
count and degree. Explicit `knots_u` / `knots_v` fields are not read yet, so a
producer must not rely on custom knot vectors being preserved by this renderer.

Supported optional top-level fields:

```json
{
  "type": "nurbs_surface",
  "iteration": 10000,
  "degree_u": 1,
  "degree_v": 1,
  "weights": [[1.0, 1.0], [1.0, 1.0]],
  "control_grid_shape": [2, 2, 3],
  "observed_v_max": 0.5,
  "base_curves": [[[0, 0, 0], [1, 0, 0]]],
  "occlusion_curves": [[[0, 1, 0], [1, 1, 0]]]
}
```

For a multi-patch surface, `patches[]` is the authoritative renderable surface
set and is required instead of the single top-level `control_grid`. Top-level
`control_grid`, `weights`, and degrees remain a backward-compatible primary
patch representation, but are **not** rendered again when `patches[]` is
non-empty.

Every patch is independently normalized and evaluated. A patch has its own
`control_grid`, `weights`, `degree_u`, `degree_v`, and optional
`observed_v_max`. When a patch omits `observed_v_max`, the top-level value is
used; when both omit it, the default is `0.5`.

`patch_id` is optional syntactically, but multi-patch producers should provide
a unique numeric ID for every patch. The renderer uses it for status metadata,
mouse selection, and highlight lookup. If omitted, the patch's array index is
used. Duplicate IDs make selection ambiguous and must be avoided.

```json
{
  "type": "nurbs_surface",
  "iteration": 10000,
  "observed_v_max": 0.5,
  "base_curves": [[[0, 0, 0], [0.5, 0.3, 0], [1, 0, 0]]],
  "occlusion_curves": [],
  "patches": [
    {
      "patch_id": 0,
      "control_grid_shape": [3, 3, 3],
      "control_grid": [
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
        [[1, 0, 0], [1, 1, 0], [1, 2, 0]],
        [[2, 0, 0], [2, 1, 0], [2, 2, 0]]
      ],
      "weights": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
      "degree_u": 2,
      "degree_v": 2,
      "observed_v_max": 0.6
    },
    {
      "patch_id": 1,
      "control_grid_shape": [3, 2, 3],
      "control_grid": [
        [[3, 0, 0], [3, 1, 0]],
        [[4, 0, 0], [4, 1, 0]],
        [[5, 0, 0], [5, 1, 0]]
      ],
      "weights": [[1, 1], [1, 1], [1, 1]],
      "degree_u": 2,
      "degree_v": 1
    }
  ]
}
```

Each patch is tessellated independently. Missing weights are treated as one,
and flattened arrays are reshaped with `control_grid_shape`. Degrees are
clamped to the corresponding control count minus one. An invalid patch is
skipped without discarding valid patches; its ID is reported in renderer
metadata.

### U/V Curves, Boundaries, and Patch Selection

The cyan wireframe is made from evaluated NURBS iso-parametric curves, not from
the control polygon:

- **U iso-curves**: `v` is fixed and `u` varies from `0` to `1`.
- **V iso-curves**: `u` is fixed and `v` varies from `0` to `1`.
- **Patch boundary**: the four evaluated edges `v=0`, `u=1`, `v=1`, and `u=0`.

`NURBS Surface` displays the filled surface and each patch boundary. Clicking a
surface patch highlights that patch and its boundary, and reports its
`patch_id` in the Statistics panel. `NURBS Curves` and `Point Cloud + NURBS
Curves` display evaluated U/V curves and boundaries for every patch.

`base_curves` and `occlusion_curves` are top-level diagnostic polylines. They
are rendered once for the payload and are not duplicated for every patch.

The renderer generates surface triangles, boundaries, and visualization curves
from this JSON. The available NURBS-related modes are:

- `NURBS Surface`
- `NURBS Curves`
- `Point Cloud + NURBS Curves`

In the combined mode, the NURBS curve pass is rendered after the point-cloud
pass and does not use a depth buffer, so curves are visually placed in front
of the point cloud. This combined mode uses a sharper point-cloud falloff than
the standalone `Point Cloud` mode; the standalone mode remains unchanged.

## 5. WebSocket Snapshot Format

Every snapshot must contain a numeric `iteration` and either `gaussians[]` or
packed Gaussian arrays.

### Object-based snapshot

```json
{
  "iteration": 10000,
  "parameterSpace": "graphdeco",
  "gaussians": [
    {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "f_dc_0": 0.0,
      "f_dc_1": 0.0,
      "f_dc_2": 0.0,
      "opacity": 0.0,
      "scale_0": -2.0,
      "scale_1": -2.0,
      "scale_2": -2.0,
      "rot_0": 1.0,
      "rot_1": 0.0,
      "rot_2": 0.0,
      "rot_3": 0.0
    }
  ],
  "nurbs_surface": null,
  "metadata": {}
}
```

`parameterSpace` values:

- `graphdeco`: scales are log-scales and opacity is a logit; the renderer
  applies `exp()` and `sigmoid()`.
- `render` or omitted: scales and opacity are already in render space.

The normalized render-space object form is:

```json
{
  "iteration": 10000,
  "parameterSpace": "render",
  "gaussians": [
    {
      "position": [0.0, 0.0, 0.0],
      "scale": [0.1, 0.1, 0.1],
      "rotation": [1.0, 0.0, 0.0, 0.0],
      "opacity": 0.8,
      "color": [0.7, 0.5, 0.3]
    }
  ]
}
```

### Packed snapshot

```json
{
  "iteration": 10000,
  "parameterSpace": "graphdeco",
  "count": 2,
  "positions": [x0, y0, z0, x1, y1, z1],
  "scales": [sx0, sy0, sz0, sx1, sy1, sz1],
  "rotations": [r0, r1, r2, r3, r0, r1, r2, r3],
  "colors": [r0, g0, b0, r1, g1, b1],
  "opacities": [a0, a1],
  "ids": [0, 1],
  "nurbsSurface": null,
  "metadata": {}
}
```

`positions`, `scales`, and `colors` are flattened arrays with three values per
Gaussian. `rotations` has four values per Gaussian. `count` is optional when
it can be inferred from `positions.length / 3`.

## 6. NURBS Over WebSocket

The WebSocket message may include the same NURBS JSON object under either key:

```json
{
  "iteration": 10000,
  "gaussians": [],
  "nurbs_surface": {
    "control_grid": [[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]],
    "degree_u": 1,
    "degree_v": 1,
    "weights": [[1, 1], [1, 1]],
    "base_curves": [],
    "occlusion_curves": []
  }
}
```

`nurbsSurface` is also accepted as an alias for `nurbs_surface`.

The renderer converts the received NURBS surface into GPU-ready geometry on
the client. The sender does not need to send pre-tessellated vertices.

## 7. Minimal Handoff Checklist

For a new training framework, provide:

1. A Graphdeco-compatible `point_cloud.ply`, or a WebSocket snapshot using the
   exact fields above.
2. A numeric iteration number, preferably represented by a directory named
   `iteration_<integer>` for local files.
3. If NURBS visualization is needed, a `nurbs_surface.json` file or a
   `nurbs_surface` WebSocket field containing `control_grid`.
4. The correct `parameterSpace` value so scales and opacity are not decoded
   twice or left undecoded.
