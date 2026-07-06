from __future__ import annotations

"""COLMAP and Graphdeco-style dataset loader for OSN-GS.

This module lets OSN-GS consume a standard 3DGS dataset layout:

scene_root/
  images/
  sparse/0/cameras.bin or cameras.txt
  sparse/0/images.bin or images.txt
  sparse/0/points3D.bin or points3D.txt
"""

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from osn_gs.data.torch_scene import TorchScene
from osn_gs.render.torch_fallback import TorchCamera
from osn_gs.utils.torch_ops import require_torch


CAMERA_MODEL_PARAM_COUNTS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_NAME_TO_ID = {name: model_id for model_id, (name, _) in CAMERA_MODEL_PARAM_COUNTS.items()}


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass
class ColmapPointCloud:
    xyz: np.ndarray
    rgb: np.ndarray


def load_colmap_scene(
    scene_root: str | Path,
    device: str = "cuda",
    image_device: str | None = None,
    image_dir_name: str = "images",
    sparse_dir_name: str = "sparse/0",
    image_downscale: int = 1,
    max_images: int = 0,
) -> TorchScene:
    """Convert a COLMAP scene into the TorchScene format used by OSN-GS."""

    torch = require_torch()
    scene_root = Path(scene_root)
    if image_device is None:
        image_device = "auto"
    image_device = str(image_device).lower()
    image_root = scene_root / image_dir_name
    sparse_root = scene_root / sparse_dir_name
    if not image_root.exists():
        raise FileNotFoundError(f"Missing COLMAP image directory: {image_root}")
    if not sparse_root.exists():
        raise FileNotFoundError(f"Missing COLMAP sparse directory: {sparse_root}")

    cameras = read_colmap_cameras(sparse_root)
    images = read_colmap_images(sparse_root)
    point_cloud = read_colmap_points3d(sparse_root)

    if point_cloud.xyz.shape[0] == 0:
        raise ValueError(f"No sparse points found in {sparse_root}")

    ordered_images = sorted(images.values(), key=lambda image: image.name)
    if max_images > 0:
        ordered_images = ordered_images[:max_images]

    torch_cameras: list[TorchCamera] = []
    image_tensors = []
    for image in ordered_images:
        colmap_camera = cameras[image.camera_id]
        image_path = resolve_image_path(image_root, image.name)
        image_tensor, height, width = load_image_tensor(image_path, device="cpu", downscale=image_downscale)
        fovx, fovy = camera_fovs(colmap_camera, width=width, height=height, downscale=image_downscale)
        world_view, full_proj, center = camera_matrices(image.qvec, image.tvec, fovx, fovy, device=device)
        torch_cameras.append(
            TorchCamera(
                image_height=height,
                image_width=width,
                world_view_transform=world_view,
                full_proj_transform=full_proj,
                camera_center=center,
                FoVx=fovx,
                FoVy=fovy,
                image_name=image.name,
            )
        )
        image_tensors.append(image_tensor)

    if not image_tensors:
        raise ValueError(f"No training images found under {image_root}")

    points = torch.as_tensor(point_cloud.xyz, dtype=torch.float32, device=device)
    colors = torch.as_tensor(point_cloud.rgb, dtype=torch.float32, device=device)
    image_bytes = sum(int(image.numel() * image.element_size()) for image in image_tensors)
    print(
        "OSN-GS image storage: cpu-staged "
        f"images={image_bytes / (1024 ** 3):.2f}GB "
        "transfer=per-view",
        flush=True,
    )
    extent = estimate_scene_extent(point_cloud.xyz)
    return TorchScene(
        initial_points=points,
        initial_colors=colors,
        cameras=torch_cameras,
        images=image_tensors,
        device=device,
        extent=extent,
    )


def read_colmap_cameras(sparse_root: Path) -> dict[int, ColmapCamera]:
    binary_path = sparse_root / "cameras.bin"
    text_path = sparse_root / "cameras.txt"
    if binary_path.exists():
        return read_cameras_binary(binary_path)
    if text_path.exists():
        return read_cameras_text(text_path)
    raise FileNotFoundError(f"Missing cameras.bin/cameras.txt in {sparse_root}")


def read_colmap_images(sparse_root: Path) -> dict[int, ColmapImage]:
    binary_path = sparse_root / "images.bin"
    text_path = sparse_root / "images.txt"
    if binary_path.exists():
        return read_images_binary(binary_path)
    if text_path.exists():
        return read_images_text(text_path)
    raise FileNotFoundError(f"Missing images.bin/images.txt in {sparse_root}")


def read_colmap_points3d(sparse_root: Path) -> ColmapPointCloud:
    binary_path = sparse_root / "points3D.bin"
    text_path = sparse_root / "points3D.txt"
    if binary_path.exists():
        return read_points3d_binary(binary_path)
    if text_path.exists():
        return read_points3d_text(text_path)
    raise FileNotFoundError(f"Missing points3D.bin/points3D.txt in {sparse_root}")


def read_next_bytes(handle: BinaryIO, num_bytes: int, fmt: str) -> tuple[Any, ...]:
    data = handle.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file.")
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as handle:
        (num_cameras,) = read_next_bytes(handle, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(handle, 24, "iiQQ")
            model_name, num_params = CAMERA_MODEL_PARAM_COUNTS[model_id]
            params = np.array(read_next_bytes(handle, 8 * num_params, "d" * num_params), dtype=np.float64)
            cameras[int(camera_id)] = ColmapCamera(int(camera_id), model_name, int(width), int(height), params)
    return cameras


def read_cameras_text(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = np.array([float(value) for value in parts[4:]], dtype=np.float64)
        cameras[camera_id] = ColmapCamera(camera_id, model, width, height, params)
    return cameras


def read_images_binary(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("rb") as handle:
        (num_images,) = read_next_bytes(handle, 8, "Q")
        for _ in range(num_images):
            image_id = read_next_bytes(handle, 4, "i")[0]
            qvec = np.array(read_next_bytes(handle, 32, "dddd"), dtype=np.float64)
            tvec = np.array(read_next_bytes(handle, 24, "ddd"), dtype=np.float64)
            camera_id = read_next_bytes(handle, 4, "i")[0]
            name = read_null_terminated_string(handle)
            (num_points2d,) = read_next_bytes(handle, 8, "Q")
            handle.seek(int(num_points2d) * 24, 1)
            images[int(image_id)] = ColmapImage(int(image_id), qvec, tvec, int(camera_id), name)
    return images


def read_images_text(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array([float(value) for value in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(value) for value in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name)
        idx += 1
    return images


def read_points3d_binary(path: Path) -> ColmapPointCloud:
    xyz_values = []
    rgb_values = []
    with path.open("rb") as handle:
        (num_points,) = read_next_bytes(handle, 8, "Q")
        for _ in range(num_points):
            read_next_bytes(handle, 8, "Q")
            xyz = read_next_bytes(handle, 24, "ddd")
            rgb = read_next_bytes(handle, 3, "BBB")
            read_next_bytes(handle, 8, "d")
            (track_length,) = read_next_bytes(handle, 8, "Q")
            handle.seek(int(track_length) * 8, 1)
            xyz_values.append(xyz)
            rgb_values.append(rgb)
    return ColmapPointCloud(
        xyz=np.asarray(xyz_values, dtype=np.float32),
        rgb=np.asarray(rgb_values, dtype=np.float32) / 255.0,
    )


def read_points3d_text(path: Path) -> ColmapPointCloud:
    xyz_values = []
    rgb_values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        xyz_values.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb_values.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return ColmapPointCloud(
        xyz=np.asarray(xyz_values, dtype=np.float32),
        rgb=np.asarray(rgb_values, dtype=np.float32) / 255.0,
    )


def read_null_terminated_string(handle: BinaryIO) -> str:
    chars = bytearray()
    while True:
        char = handle.read(1)
        if char == b"":
            raise EOFError("Unexpected end of COLMAP binary string.")
        if char == b"\x00":
            return chars.decode("utf-8")
        chars.extend(char)


def resolve_image_path(image_root: Path, image_name: str) -> Path:
    path = image_root / image_name
    if path.exists():
        return path
    candidates = list(image_root.rglob(Path(image_name).name))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing COLMAP image file: {path}")


def load_image_tensor(path: Path, device: str, downscale: int = 1) -> tuple[Any, int, int]:
    torch = require_torch()
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if downscale > 1:
            width = max(1, image.width // downscale)
            height = max(1, image.height // downscale)
            image = image.resize((width, height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device).permute(2, 0, 1).contiguous()
    return tensor, int(tensor.shape[1]), int(tensor.shape[2])


def camera_fovs(camera: ColmapCamera, width: int, height: int, downscale: int = 1) -> tuple[float, float]:
    fx, fy = camera_focals(camera)
    fx = fx / max(downscale, 1)
    fy = fy / max(downscale, 1)
    return focal_to_fov(fx, width), focal_to_fov(fy, height)


def camera_focals(camera: ColmapCamera) -> tuple[float, float]:
    model = camera.model
    params = camera.params
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        return float(params[0]), float(params[0])
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        return float(params[0]), float(params[1])
    raise ValueError(f"Unsupported COLMAP camera model: {model}")


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = qvec
    return np.array(
        [
            [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q3 * q1 + 2 * q0 * q2],
            [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
            [2 * q3 * q1 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2],
        ],
        dtype=np.float32,
    )


def camera_matrices(qvec: np.ndarray, tvec: np.ndarray, fovx: float, fovy: float, device: str) -> tuple[Any, Any, Any]:
    torch = require_torch()
    rotation = qvec_to_rotmat(qvec)
    world_view_np = np.eye(4, dtype=np.float32)
    world_view_np[:3, :3] = rotation
    world_view_np[:3, 3] = tvec.astype(np.float32)
    world_view = torch.as_tensor(world_view_np, dtype=torch.float32, device=device).transpose(0, 1).contiguous()
    projection = projection_matrix(0.01, 100.0, fovx, fovy, device=device).transpose(0, 1).contiguous()
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    center_np = -rotation.T @ tvec.astype(np.float32)
    center = torch.as_tensor(center_np, dtype=torch.float32, device=device)
    return world_view, full_projection, center


def projection_matrix(znear: float, zfar: float, fovx: float, fovy: float, device: str) -> Any:
    torch = require_torch()
    tan_half_y = math.tan(fovy * 0.5)
    tan_half_x = math.tan(fovx * 0.5)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right
    matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
    matrix[0, 0] = 2.0 * znear / (right - left)
    matrix[1, 1] = 2.0 * znear / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


def estimate_scene_extent(points: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 1.0
    center = np.mean(points, axis=0, keepdims=True)
    distances = np.linalg.norm(points - center, axis=1)
    return float(np.percentile(distances, 90) * 1.1)
