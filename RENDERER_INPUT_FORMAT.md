# 3DGS Renderer Input Format

This document describes the file and WebSocket formats accepted by
`3DGS_Renderer`.

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

Required field:

```json
{
  "control_grid": [
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
  ]
}
```

Supported optional fields:

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

The renderer generates surface triangles and visualization curves from this
JSON. The available NURBS-related modes are:

- `NURBS Surface`
- `NURBS Curves`
- `Point Cloud + NURBS Curves`

In the combined mode, the NURBS curve pass is rendered after the point-cloud
pass and does not use a depth buffer, so curves are visually placed in front
of the point cloud.

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

