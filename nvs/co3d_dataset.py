"""CO3D dataset loaders with an Objaverse-like interface.

The classes in this module mirror the structure of ``objaverse_dataset.py`` so
that downstream code can swap datasets without large refactors. Images are
cropped to square patches, composited on a white background, and cameras are
normalised so the centroid is at the origin with maximum radius 0.5.
"""

from __future__ import annotations

import gzip
import json
import os.path as osp
import random
import socket
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from scipy import ndimage as nd
from torch.utils.data import Dataset
import cv2

# ---------------------------------------------------------------------------
# Environment-specific defaults
# ---------------------------------------------------------------------------
HOSTNAME = socket.gethostname()

CO3D_DIR: Optional[str] = None
CO3D_ANNOTATION_DIR: Optional[str] = None
CO3D_DEPTH_DIR: Optional[str] = None

if "grogu" in HOSTNAME:
    CO3D_DIR = "/grogu/datasets/co3d"
    CO3D_ANNOTATION_DIR = "/grogu/user/amylin2/co3d_v2_annotations"
    CO3D_DEPTH_DIR = "/grogu/datasets/co3d"
elif "bridges" in HOSTNAME:
    CO3D_DIR = "/ocean/projects/cis240058p/jzhang24/co3d_v2_cropped"
    CO3D_ANNOTATION_DIR = (
        "/ocean/projects/cis240058p/jzhang24/co3d_v2_annotations_cropped"
    )
    CO3D_DEPTH_DIR = "/ocean/projects/cis240058p/jzhang24/co3d_depths"
else:
    warnings.warn(f"No default CO3D paths configured for host {HOSTNAME}.")


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]

assert len(TRAINING_CATEGORIES) + len(TEST_CATEGORIES) == 51


BBOX_SCALE = 2.0


@dataclass
class FrameData:
    filepath: str
    bbox: Sequence[float]
    rotation: np.ndarray
    translation: np.ndarray
    principal_point: Sequence[float]
    focal_length: Sequence[float]


def square_bbox(bbox: Sequence[float], padding: float = 0.0, tight: bool = False) -> np.ndarray:
    bbox_arr = np.array(bbox, dtype=float)
    center = (bbox_arr[:2] + bbox_arr[2:]) / 2.0
    extents = (bbox_arr[2:] - bbox_arr[:2]) / 2.0
    scale = (max(extents) if tight else min(extents)) * (1.0 + padding)
    square = np.array(
        [center[0] - scale, center[1] - scale, center[0] + scale, center[1] + scale],
        dtype=float,
    )
    return square


def clamp_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return np.array([x0, y0, x1, y1], dtype=int)


def jitter_bbox(bbox: np.ndarray, scale: float) -> np.ndarray:
    center = (bbox[:2] + bbox[2:]) / 2.0
    half_size = (bbox[2:] - bbox[:2]) / 2.0
    jitter_half = half_size * scale
    jittered = np.array(
        [center[0] - jitter_half[0], center[1] - jitter_half[1], center[0] + jitter_half[0], center[1] + jitter_half[1]],
        dtype=float,
    )
    return jittered


def compute_centered_square_bbox(
    image_width: int,
    image_height: int,
    object_bbox: Sequence[float],
    scale: float,
) -> np.ndarray:
    bbox_arr = np.array(object_bbox, dtype=float)
    width = max(1.0, bbox_arr[2] - bbox_arr[0])
    height = max(1.0, bbox_arr[3] - bbox_arr[1])
    center = (bbox_arr[:2] + bbox_arr[2:]) / 2.0

    base_half = max(width, height) * 0.5
    desired_half = base_half * max(scale, 1.0)
    max_half = min(image_width, image_height) * 0.5
    half = min(desired_half, max_half)
    if half < 0.5:
        half = max_half if max_half > 0.0 else 0.5

    side = int(max(1, min(image_width, image_height, np.floor(2.0 * half))))
    half = side / 2.0

    min_cx = half
    max_cx = image_width - half
    min_cy = half
    max_cy = image_height - half
    if min_cx > max_cx:
        min_cx = max_cx = image_width / 2.0
    if min_cy > max_cy:
        min_cy = max_cy = image_height / 2.0

    cx = float(np.clip(center[0], min_cx, max_cx))
    cy = float(np.clip(center[1], min_cy, max_cy))

    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + side
    y1 = y0 + side

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > image_width:
        shift = x1 - image_width
        x0 -= shift
        x1 = image_width
    if y1 > image_height:
        shift = y1 - image_height
        y0 -= shift
        y1 = image_height

    x0 = max(0, min(x0, image_width - 1))
    y0 = max(0, min(y0, image_height - 1))
    x1 = max(x0 + 1, min(x1, image_width))
    y1 = max(y0 + 1, min(y1, image_height))

    side = min(x1 - x0, y1 - y0)
    x1 = x0 + side
    y1 = y0 + side
    if x1 > image_width:
        shift = x1 - image_width
        x0 -= shift
        x1 = image_width
    if y1 > image_height:
        shift = y1 - image_height
        y0 -= shift
        y1 = image_height

    return np.array([x0, y0, x1, y1], dtype=int)


def fill_depth_nearest(depth: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError("Expected 2D depth map for filling.")

    invalid_mask = invalid_mask.astype(bool)
    if not invalid_mask.any():
        return depth
    if (~invalid_mask).sum() == 0:
        return depth

    indices = nd.distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
    filled = depth.copy()
    nearest = depth[tuple(indices)]
    filled[invalid_mask] = nearest[invalid_mask]
    return filled

def transform_intrinsics(
    image: Image.Image,
    bbox: Sequence[float],
    principal_point_ndc: Sequence[float],
    focal_length_ndc: Sequence[float],
    resize_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Convert normalized intrinsics to pixel-space intrinsics after applying a crop and resize.
    """
    # --- Original image size ---
    W_orig, H_orig = image.width, image.height
    half_box = np.array([W_orig / 2.0, H_orig / 2.0], dtype=np.float32)
    org_scale = float(min(half_box))

    # --- Convert normalized intrinsics to pixel space (before crop) ---
    focal_px = np.array(focal_length_ndc, dtype=np.float32) * org_scale
    principal_px = half_box - np.array(principal_point_ndc, dtype=np.float32) * org_scale

    # --- Crop adjustment ---
    bbox = np.array(bbox, dtype=np.float32)
    x0, y0, x1, y1 = bbox
    crop_w, crop_h = x1 - x0, y1 - y0

    # Shift principal point so (0,0) is now top-left of crop
    principal_px -= np.array([x0, y0], dtype=np.float32)

    # --- Resize adjustment ---
    new_w, new_h = resize_shape
    scale_x = new_w / crop_w
    scale_y = new_h / crop_h

    focal_px[0] *= scale_x
    focal_px[1] *= scale_y
    principal_px[0] *= scale_x
    principal_px[1] *= scale_y

    # --- Construct new K matrix (pixel-space intrinsics) ---
    K = np.array([
        [focal_px[0], 0.0, principal_px[0]],
        [0.0, focal_px[1], principal_px[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K

# def transform_intrinsics(
#     image: Image.Image,
#     bbox: Sequence[int],
#     principal_point: Sequence[float],
#     focal_length: Sequence[float],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     half_box = np.array([image.width, image.height], dtype=np.float32) / 2.0
#     org_scale = float(min(half_box))

#     principal_point_px = half_box - (np.array(principal_point) * org_scale)
#     focal_length_px = np.array(focal_length) * org_scale
#     principal_point_px -= np.array(bbox[:2])

#     new_bbox = (np.array(bbox[2:]) - np.array(bbox[:2])) / 2.0
#     new_scale = float(min(new_bbox))

#     new_principal_ndc = (new_bbox - principal_point_px) / new_scale
#     new_focal_ndc = focal_length_px / new_scale
#     return new_principal_ndc.astype(np.float32), new_focal_ndc.astype(np.float32)


# def ndc_to_pixel_intrinsics(principal_ndc: np.ndarray, focal_ndc: np.ndarray, image_size: int) -> np.ndarray:
#     half_size = image_size / 2.0
#     fx = focal_ndc[0] * half_size
#     fy = focal_ndc[1] * half_size
#     cx = (principal_ndc[0] + 1.0) * half_size
#     cy = (principal_ndc[1] + 1.0) * half_size
#     return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


AXIS_P3D_TO_OPENCV = np.array(
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)


def pytorch3d_extrinsics_to_opencv_c2w(
    R: np.ndarray, T: np.ndarray
) -> np.ndarray:
    """Convert PyTorch3D (R, T) camera extrinsics to an OpenCV-style C2W matrix."""

    w2c_pt3d = np.eye(4, dtype=np.float32)
    w2c_pt3d[:3, :3] = R.astype(np.float32).T
    w2c_pt3d[:3, 3] = T.astype(np.float32)

    c2w_pt3d = np.linalg.inv(w2c_pt3d)

    R_c2w_pt3d = c2w_pt3d[:3, :3]
    T_c2w_pt3d = c2w_pt3d[:3, 3]

    R_c2w_opencv = R_c2w_pt3d @ AXIS_P3D_TO_OPENCV
    T_c2w_opencv = T_c2w_pt3d

    c2w_opencv = np.eye(4, dtype=np.float32)
    c2w_opencv[:3, :3] = R_c2w_opencv
    c2w_opencv[:3, 3] = T_c2w_opencv
    return c2w_opencv


class _Co3dSequenceStore:
    def __init__(
        self,
        categories: Sequence[str],
        split: str,
        annotation_dir: str,
        min_frames: int,
    ) -> None:
        start_time = time.time()
        self.rotations: Dict[str, List[FrameData]] = {}
        self.category_map: Dict[str, str] = {}
        self.low_quality_sequences: List[str] = []

        for category in categories:
            annotation_path = osp.join(annotation_dir, f"{category}_{split}.jgz")
            with gzip.open(annotation_path, "r") as handle:
                annotation = json.loads(handle.read())

            kept = 0
            for sequence_name, frames in annotation.items():
                if len(frames) < min_frames:
                    continue

                filtered: List[FrameData] = []
                bad_sequence = False
                for frame in frames:
                    det = np.linalg.det(frame["R"])
                    if (np.abs(frame["T"]) > 1e5).any() or det < 0.99 or det > 1.01:
                        bad_sequence = True
                        self.low_quality_sequences.append(sequence_name)
                        break
                    filtered.append(
                        FrameData(
                            filepath=frame["filepath"],
                            bbox=frame["bbox"],
                            rotation=np.array(frame["R"], dtype=np.float32),
                            translation=np.array(frame["T"], dtype=np.float32),
                            principal_point=frame["principal_point"],
                            focal_length=frame["focal_length"],
                        )
                    )

                if not bad_sequence:
                    self.rotations[sequence_name] = filtered
                    self.category_map[sequence_name] = category
                    kept += 1

            # print(f"Loaded {kept} sequences from category '{category}'.")

        self.sequence_list = sorted(self.rotations.keys())
        elapsed = time.time() - start_time
        print(f"CO3D metadata cache ready ({len(self.sequence_list)} sequences, {elapsed:.2f}s).")


class _Co3dBaseDataset(Dataset):
    def __init__(
        self,
        categories: Sequence[str],
        split: str,
        patch_size: int,
        depth_size: Optional[int],
        co3d_dir: Optional[str],
        co3d_annotation_dir: Optional[str],
        co3d_depth_dir: Optional[str],
        load_depth: bool,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.depth_size = depth_size or patch_size
        self.load_depth = load_depth

        self.co3d_dir = co3d_dir or CO3D_DIR
        self.co3d_annotation_dir = co3d_annotation_dir or CO3D_ANNOTATION_DIR
        self.co3d_depth_dir = co3d_depth_dir or CO3D_DEPTH_DIR

        if self.co3d_dir is None or self.co3d_annotation_dir is None:
            raise RuntimeError("CO3D dataset paths are not configured.")
        if self.load_depth and self.co3d_depth_dir is None:
            raise RuntimeError("Depth loading requested but depth directory is unset.")

        self.store = _Co3dSequenceStore(categories, split, self.co3d_annotation_dir, min_frames=2)

    def __len__(self) -> int:
        return len(self.store.sequence_list)

    def _get_sequence_metadata(self, sequence_name: str) -> List[FrameData]:
        return self.store.rotations[sequence_name]

    def _load_mask(self, category: str, sequence_name: str, filepath: str, target_size: Tuple[int, int]) -> Image.Image:
        mask_name = osp.basename(filepath).replace(".jpg", ".png")
        mask_path = osp.join(self.co3d_dir, category, sequence_name, "masks", mask_name)
        mask_image = Image.open(mask_path).convert("L")
        if mask_image.size != target_size:
            mask_image = mask_image.resize(target_size, resample=Image.NEAREST)
        return mask_image

    def _load_depth(self, category: str, sequence_name: str, filepath: str) -> np.ndarray:
        depth_file = osp.basename(filepath).replace(".jpg", ".jpg.geometric.png")
        depth_path = osp.join(self.co3d_depth_dir, category, sequence_name, "depths", depth_file)
        depth_image = Image.open(depth_path)
        raw = np.frombuffer(np.array(depth_image, dtype=np.uint16), dtype=np.float16).astype(np.float32)
        depth = raw.reshape((depth_image.size[1], depth_image.size[0]))
        return depth

    def _prepare_frame(
        self,
        frame: FrameData,
        category: str,
        sequence_name: str,
    ) -> Dict[str, Any]:
        image_path = osp.join(self.co3d_dir, frame.filepath)
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        # mask_image = self._load_mask(category, sequence_name, frame.filepath, original_size)
        # mask_array = (np.array(mask_image) > 125).astype(np.uint8) * 255
        # mask_binary_image = Image.fromarray(mask_array)

        object_bbox = np.array(frame.bbox, dtype=float)
        good_bbox = ((object_bbox[2:] - object_bbox[:2]) > 30).all()
        if not good_bbox:
            object_bbox = np.array([0.0, 0.0, float(image.width), float(image.height)], dtype=float)

        bbox_int = compute_centered_square_bbox(image.width, image.height, object_bbox, BBOX_SCALE)

        K = transform_intrinsics(
            image,
            bbox_int,
            frame.principal_point,
            frame.focal_length,
            (self.patch_size, self.patch_size),
        )

        cropped_image = image.crop(tuple(bbox_int))
        # cropped_mask = mask_binary_image.crop(tuple(bbox_int))

        resized_image = cropped_image.resize((self.patch_size, self.patch_size), resample=Image.BILINEAR)
        # resized_mask = cropped_mask.resize((self.patch_size, self.patch_size), resample=Image.NEAREST)

        image_np = np.asarray(resized_image).astype(np.float32)
        # mask_np = (np.asarray(resized_mask) > 125).astype(np.float32)

        depth_np: Optional[np.ndarray] = None
        depth_mask_np: Optional[np.ndarray] = None
        if self.load_depth:
            depth_full = self._load_depth(category, sequence_name, frame.filepath)

            if depth_full.shape[::-1] != original_size:
                scale_x = depth_full.shape[1] / original_size[0]
                scale_y = depth_full.shape[0] / original_size[1]
                bbox_depth = np.array(
                    [
                        int(bbox_int[0] * scale_x),
                        int(bbox_int[1] * scale_y),
                        int(bbox_int[2] * scale_x),
                        int(bbox_int[3] * scale_y),
                    ]
                )
            else:
                bbox_depth = bbox_int

            depth_crop = depth_full[bbox_depth[1] : bbox_depth[3], bbox_depth[0] : bbox_depth[2]]
            depth_image = Image.fromarray(depth_crop.astype(np.float32))
            depth_resized = depth_image.resize((self.depth_size, self.depth_size), resample=Image.NEAREST)
            depth_np = np.asarray(depth_resized, dtype=np.float32)
            invalid_mask = np.logical_or(depth_np < 1e-3, np.isnan(depth_np))
            valid_mask = ~invalid_mask
            if valid_mask.any() and not valid_mask.all():
                depth_np = fill_depth_nearest(depth_np, invalid_mask)
            elif not valid_mask.any():
                depth_np = np.full_like(depth_np, np.inf, dtype=np.float32)
            # depth_mask_np = valid_mask.astype(np.uint8)

        c2w = pytorch3d_extrinsics_to_opencv_c2w(frame.rotation, frame.translation)

        return {
            "image": image_np,
            # "mask": mask_np,
            "K": K,
            "c2w": c2w,
            "depth": depth_np,
            "depth_mask": depth_mask_np,
            "path": image_path,
        }

    @staticmethod
    def _normalise_scene(c2ws: np.ndarray, depths: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        
        # first make the first camera identity
        ref0_c2w = c2ws[0]
        c2ws = np.einsum("ij,vjk->vik", np.linalg.inv(ref0_c2w), c2ws)
        
        centers = c2ws[:, :3, 3]
        centroid = centers.mean(axis=0)
        centers -= centroid
        max_radius = np.linalg.norm(centers, axis=1).max()
        scale = 1.0
        if max_radius > 0:
            scale = 0.5 / max_radius
            centers *= scale
        c2ws[:, :3, 3] = centers

        scaled_depths = None
        if depths is not None:
            scaled_depths = depths.copy()
            valid = np.logical_and(scaled_depths >= 0.0, ~np.isnan(scaled_depths))
            scaled_depths[valid] *= scale
            scaled_depths[~valid] = np.inf
        return c2ws, scaled_depths, scale


class Co3dTrainDataset(_Co3dBaseDataset):
    """Training split of CO3D with RealEstate10k-style view sampling."""

    def __init__(
        self,
        categories: Sequence[str] | str = ("seen",),
        patch_size: int = 224,
        depth_size: Optional[int] = None,
        input_views: int = 2,
        supervise_views: int = 1,
        min_frame_dist: int = 25,
        max_frame_dist: int = 100,
        load_depth: bool = False,
        co3d_dir: Optional[str] = None,
        co3d_annotation_dir: Optional[str] = None,
        co3d_depth_dir: Optional[str] = None,
    ) -> None:
        if isinstance(categories, str):
            categories = [categories]
        if "seen" in categories:
            categories = TRAINING_CATEGORIES
        if "full" in categories:
            categories = TRAINING_CATEGORIES + TEST_CATEGORIES

        self.input_views = input_views
        self.supervise_views = supervise_views
        self.min_frame_dist = min_frame_dist
        self.max_frame_dist = max_frame_dist

        super().__init__(
            categories=categories,
            split="train",
            patch_size=patch_size,
            depth_size=depth_size,
            co3d_dir=co3d_dir,
            co3d_annotation_dir=co3d_annotation_dir,
            co3d_depth_dir=co3d_depth_dir,
            load_depth=load_depth,
        )

    def _select_views_from_range(self, n_frames: int) -> Optional[List[int]]:
        total_needed = self.input_views + self.supervise_views
        if n_frames < total_needed:
            return None

        max_dist = min(n_frames - 1, self.max_frame_dist)
        min_dist = min(max(1, self.min_frame_dist), max_dist)
        if max_dist <= 1:
            indices = random.sample(range(n_frames), total_needed)
            return indices

        # Select a temporal range
        frame_dist = random.randint(min_dist, max_dist)
        if n_frames <= frame_dist:
            return None
        start_index = random.randint(0, n_frames - frame_dist - 1)
        end_index = start_index + frame_dist

        # Sample input_views frames within the temporal range
        temporal_range = list(range(start_index, end_index + 1))
        if self.input_views == 1:
            # Single view: pick randomly from the range
            context_indices = [random.choice(temporal_range)]
        elif self.input_views == 2:
            # Two views: use start and end
            context_indices = [start_index, end_index]
        else:
            # Multiple views: ensure start and end are included, then sample intermediate frames
            if len(temporal_range) >= self.input_views:
                # Include start and end, sample the rest from intermediate frames
                intermediate = temporal_range[1:-1]
                if len(intermediate) >= self.input_views - 2:
                    sampled_intermediate = random.sample(intermediate, self.input_views - 2)
                    context_indices = [start_index, end_index] + sampled_intermediate
                else:
                    raise RuntimeError("Frame range should be larger than number of input views.")
            else:
                raise RuntimeError("Frame range should be larger than number of input views.")

        # Sample supervise views from frames not in context
        supervise_pool = [idx for idx in range(n_frames) if idx not in context_indices]
        if len(supervise_pool) < self.supervise_views:
            return None
        
        supervise_indices = random.sample(supervise_pool, self.supervise_views)
        indices = context_indices + supervise_indices
        return indices
    
    def _select_views_random(self, n_frames: int) -> Optional[List[int]]:
        """Randomly select input views and target views without temporal constraints."""
        total_needed = self.input_views + self.supervise_views
        if n_frames < total_needed:
            return None
        
        # Randomly sample all needed frames
        indices = random.sample(range(n_frames), total_needed)
        return indices

    def __getitem__(self, _: int) -> Dict[str, Any]:
        attempt = 0
        while True:
            sequence_name = random.choice(self.store.sequence_list)
            # sequence_name = self.store.sequence_list[0]
            frames = self._get_sequence_metadata(sequence_name)

            # New fix: if we train with more views, use random sampling
            if self.input_views >= 4:
                indices = self._select_views_random(len(frames))
            else:
                indices = self._select_views_from_range(len(frames))

            attempt += 1
            if indices is None:
                if attempt > 20:
                    raise RuntimeError("Unable to sample views with requested spacing.")
                continue
            break

        category = self.store.category_map[sequence_name]
        
        frame_outputs = [
            self._prepare_frame(
                frames[idx],
                category,
                sequence_name,
            )
            for idx in indices
        ]

        images = np.stack([f["image"] for f in frame_outputs], axis=0)
        # masks = np.stack([f["mask"] for f in frame_outputs], axis=0)
        Ks = np.stack([f["K"] for f in frame_outputs], axis=0)
        c2ws = np.stack([f["c2w"] for f in frame_outputs], axis=0)
        paths = [f["path"] for f in frame_outputs]

        depths = None
        # depth_masks = None
        if self.load_depth:
            depth_list: List[np.ndarray] = []
            # depth_mask_list: List[np.ndarray] = []
            for frame in frame_outputs:
                depth_arr = frame["depth"]
                # mask_arr = frame.get("depth_mask")
                if depth_arr is None:
                    depth_arr = np.full((self.depth_size, self.depth_size), np.inf, dtype=np.float32)
                    # mask_arr = np.zeros((self.depth_size, self.depth_size), dtype=np.uint8)
                # else:
                #     if mask_arr is None:
                #         mask_arr = np.logical_and(depth_arr >= 0.0, ~np.isnan(depth_arr)).astype(np.uint8)
                #     else:
                #         mask_arr = mask_arr.astype(np.uint8, copy=False)
                depth_list.append(depth_arr)
                # depth_mask_list.append(mask_arr)

            depths = np.stack(depth_list, axis=0)
            # depth_masks = np.stack(depth_mask_list, axis=0)

        c2ws_norm, depths_norm, scale = self._normalise_scene(c2ws, depths)

        result: Dict[str, Any] = {
            "sequence": sequence_name,
            "category": category,
            "image": torch.from_numpy(images).float(),
            # "mask": torch.from_numpy(masks).float(),
            "K": torch.from_numpy(Ks).float(),
            "camtoworld": torch.from_numpy(c2ws_norm).float(),
            "image_path": paths,
            "context_indices": list(range(self.input_views)),
            "target_indices": list(range(self.input_views, self.input_views + self.supervise_views)),
            "scene_scale": scale,
            "frame_indices": indices,
        }


        if depths_norm is not None:
            depths_tensor = torch.from_numpy(depths_norm).float()
            result["depths"] = depths_tensor
            result["context_depths"] = depths_tensor[:self.input_views].unsqueeze(-1)
            # if depth_masks is not None:
            #     result["depth_masks"] = torch.from_numpy(depth_masks).long()[:self.input_views].unsqueeze(-1)

        return result


class Co3dEvalDataset(_Co3dBaseDataset):
    """Evaluation split of CO3D with configurable category selection."""

    def __init__(
        self,
        index_file: str,
        categories: Sequence[str] | str = ("seen",),
        patch_size: int = 224,
        depth_size: Optional[int] = None,
        input_views: int = 2,
        load_depth: bool = False,
        co3d_dir: Optional[str] = None,
        co3d_annotation_dir: Optional[str] = None,
        co3d_depth_dir: Optional[str] = None,
        first_n: Optional[int] = None,
        render_video: bool = False,
    ) -> None:
        if isinstance(categories, str):
            categories = [categories]
        # if isinstance(categories, str):
        #     categories = [categories]
        if "unseen" in categories:
            categories = TEST_CATEGORIES
        elif "seen" in categories:
            categories = TRAINING_CATEGORIES
        elif "full" in categories:
            categories = TRAINING_CATEGORIES + TEST_CATEGORIES

        self.input_views = input_views
        with open(index_file, "r") as handle:
            self.index_data = json.load(handle)

        super().__init__(
            categories=categories,
            split="test",
            patch_size=patch_size,
            depth_size=depth_size,
            co3d_dir=co3d_dir,
            co3d_annotation_dir=co3d_annotation_dir,
            co3d_depth_dir=co3d_depth_dir,
            load_depth=load_depth,
        )
        self.render_video = render_video
        self.sequence_order = [
            seq for seq in self.store.sequence_list if seq in self.index_data
        ]
        if first_n is not None:
            self.sequence_order = self.sequence_order[:first_n]
        if not self.sequence_order:
            raise RuntimeError("No sequences from index file match the loaded metadata.")

    def __len__(self) -> int:
        return len(self.sequence_order)

    def _sort_video_frames(self, indices: List[int], frames: List[FrameData]) -> List[int]:
        """Sort frame indices for video rendering using bidirectional greedy extension with direction consistency.
        
        Args:
            indices: List of frame indices to sort
            frames: List of FrameData for the sequence
            
        Returns:
            Sorted list of frame indices for smooth camera trajectory
        """
        if len(indices) <= 2:
            return indices
        
        # Get c2w matrices for all frames in indices
        c2ws = []
        for idx in indices:
            frame = frames[idx]
            c2w = pytorch3d_extrinsics_to_opencv_c2w(frame.rotation, frame.translation)
            c2ws.append(c2w)
        c2ws = np.stack(c2ws, axis=0)  # (N, 4, 4)
        
        # Extract camera centers and rotations
        centers = c2ws[:, :3, 3]  # (N, 3)
        rotations = c2ws[:, :3, :3]  # (N, 3, 3)
        
        def rotation_distance(R1: np.ndarray, R2: np.ndarray) -> float:
            """Geodesic distance on SO(3)."""
            R_diff = R1.T @ R2
            trace = np.trace(R_diff)
            cos_angle = np.clip((trace - 1) / 2, -1, 1)
            return np.arccos(cos_angle)
        
        def pose_distance(i: int, j: int) -> float:
            """Combined distance metric for camera poses."""
            trans_dist = np.linalg.norm(centers[i] - centers[j])
            rot_dist = rotation_distance(rotations[i], rotations[j])
            return trans_dist + 1.0 * rot_dist
        
        def direction_consistency(prev_idx: int, curr_idx: int, next_idx: int) -> float:
            """Measure how consistent the movement direction is (0 = same direction, 1 = opposite).
            
            Considers both translation velocity and rotation velocity consistency.
            """
            # Translation direction consistency
            v1 = centers[curr_idx] - centers[prev_idx]
            v2 = centers[next_idx] - centers[curr_idx]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            
            if norm1 < 1e-8 or norm2 < 1e-8:
                trans_consistency = 0.0  # No penalty if barely moving
            else:
                cos_sim = np.dot(v1, v2) / (norm1 * norm2)
                trans_consistency = (1 - cos_sim) / 2  # 0 if same direction, 1 if opposite
            
            # Rotation direction consistency (compare angular velocities)
            R_diff1 = rotations[prev_idx].T @ rotations[curr_idx]
            R_diff2 = rotations[curr_idx].T @ rotations[next_idx]
            # Use Frobenius norm of difference as a simple consistency measure
            rot_consistency = np.linalg.norm(R_diff1 - R_diff2) / 4.0  # Normalize roughly to [0, 1]
            
            return trans_consistency + 0.5 * rot_consistency
        
        def score_extension(path: List[int], candidate: int, direction: str, visited: List[bool]) -> float:
            """Score a candidate extension (lower is better)."""
            direction_weight = 1.0
            
            if direction == 'forward':
                curr_idx = path[-1]
                dist = pose_distance(curr_idx, candidate)
                if len(path) >= 2:
                    prev_idx = path[-2]
                    dir_penalty = direction_consistency(prev_idx, curr_idx, candidate)
                    return dist + direction_weight * dir_penalty
                return dist
            else:  # backward
                curr_idx = path[0]
                dist = pose_distance(curr_idx, candidate)
                if len(path) >= 2:
                    next_idx = path[1]
                    # For backward: candidate -> path[0] -> path[1]
                    dir_penalty = direction_consistency(candidate, curr_idx, next_idx)
                    return dist + direction_weight * dir_penalty
                return dist
        
        # Start with first frame (local index 0)
        n = len(indices)
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        # Bidirectional greedy extension
        for _ in range(n - 1):
            best_score = float('inf')
            best_idx = -1
            best_direction = 'forward'
            
            # Try extending forward
            for j in range(n):
                if not visited[j]:
                    score = score_extension(path, j, 'forward', visited)
                    if score < best_score:
                        best_score = score
                        best_idx = j
                        best_direction = 'forward'
            
            # Try extending backward
            for j in range(n):
                if not visited[j]:
                    score = score_extension(path, j, 'backward', visited)
                    if score < best_score:
                        best_score = score
                        best_idx = j
                        best_direction = 'backward'
            
            # Extend in the best direction
            if best_direction == 'forward':
                path.append(best_idx)
            else:
                path.insert(0, best_idx)
            visited[best_idx] = True
        
        # --- Refinement phase: iteratively relocate worst frames ---
        def compute_frame_penalty(path: List[int], pos: int) -> float:
            """Compute the score penalty for a frame at position pos in the path."""
            penalty = 0.0
            idx = path[pos]
            
            # Distance to neighbors
            if pos > 0:
                penalty += pose_distance(path[pos - 1], idx)
            if pos < len(path) - 1:
                penalty += pose_distance(idx, path[pos + 1])
            
            # Direction consistency penalty
            if pos > 0 and pos < len(path) - 1:
                penalty += direction_consistency(path[pos - 1], idx, path[pos + 1])
            
            return penalty
        
        def compute_insertion_cost(path: List[int], idx: int, insert_pos: int) -> float:
            """Compute the cost of inserting idx at insert_pos in path."""
            cost = 0.0
            
            # Cost of connecting to neighbors at new position
            if insert_pos > 0:
                cost += pose_distance(path[insert_pos - 1], idx)
            if insert_pos < len(path):
                cost += pose_distance(idx, path[insert_pos])
            
            # Direction consistency at new position
            if insert_pos > 0 and insert_pos < len(path):
                cost += direction_consistency(path[insert_pos - 1], idx, path[insert_pos])
            
            # Bonus: subtract the cost of the edge we're breaking
            if insert_pos > 0 and insert_pos < len(path):
                cost -= pose_distance(path[insert_pos - 1], path[insert_pos])
            
            return cost
        
        def compute_removal_benefit(path: List[int], pos: int) -> float:
            """Compute the benefit of removing frame at pos (negative = good to remove)."""
            benefit = compute_frame_penalty(path, pos)
            
            # If removing creates a new edge, subtract its cost
            if pos > 0 and pos < len(path) - 1:
                new_edge_cost = pose_distance(path[pos - 1], path[pos + 1])
                benefit -= new_edge_cost
            
            return benefit
        
        max_iterations = 2 * n  # Limit iterations
        for _ in range(max_iterations):
            # Find the frame with highest removal benefit (worst fitting frame)
            worst_pos = -1
            worst_benefit = -float('inf')
            
            for pos in range(len(path)):
                benefit = compute_removal_benefit(path, pos)
                if benefit > worst_benefit:
                    worst_benefit = benefit
                    worst_pos = pos
            
            if worst_benefit <= 0:
                # No frame benefits from relocation
                break
            
            # Remove the worst frame
            worst_idx = path[worst_pos]
            path_without = path[:worst_pos] + path[worst_pos + 1:]
            
            # Find the best position to reinsert it
            best_insert_pos = worst_pos
            best_insert_cost = float('inf')
            
            for insert_pos in range(len(path_without) + 1):
                cost = compute_insertion_cost(path_without, worst_idx, insert_pos)
                if cost < best_insert_cost:
                    best_insert_cost = cost
                    best_insert_pos = insert_pos
            
            # Only relocate if it improves the path
            if best_insert_pos != worst_pos:
                path = path_without[:best_insert_pos] + [worst_idx] + path_without[best_insert_pos:]
            else:
                # No improvement possible for this frame, try next worst
                break
        
        # Map back to original frame indices
        return [indices[i] for i in path]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sequence_name = self.sequence_order[index]
        frames = self._get_sequence_metadata(sequence_name)
        category = self.store.category_map[sequence_name]

        if sequence_name not in self.index_data:
            raise KeyError(f"Sequence {sequence_name} missing from index file.")

        entry = self.index_data[sequence_name]
        context_indices = entry.get("context_view_indices")
        if context_indices is None:
            raise KeyError(
                "Index file entries must contain 'context_view_indices'."
            )
        context_indices = list(map(int, context_indices))

        if len(context_indices) != self.input_views:
            raise ValueError(
                f"Sequence {sequence_name} expected {self.input_views} context views, got {len(context_indices)}."
            )

        if self.render_video:
            # Use all frames in the sequence as targets (excluding context views)
            max_video_length = 120
            all_frame_indices = list(range(len(frames)))
            target_indices = [idx for idx in all_frame_indices if idx not in context_indices]
            target_indices = self._sort_video_frames(target_indices, frames)
            if len(target_indices) > max_video_length:
                target_indices = target_indices[:max_video_length]
        else:
            target_indices = entry.get("target_view_indices")
            if target_indices is None:
                raise KeyError(
                    "Index file entries must contain 'target_view_indices'."
                )
            target_indices = list(map(int, target_indices))

        all_indices = context_indices + target_indices

        frame_outputs = [
            self._prepare_frame(
                frames[idx],
                category,
                sequence_name,
            )
            for idx in all_indices
        ]

        images = np.stack([f["image"] for f in frame_outputs], axis=0)
        # masks = np.stack([f["mask"] for f in frame_outputs], axis=0)
        Ks = np.stack([f["K"] for f in frame_outputs], axis=0)
        c2ws = np.stack([f["c2w"] for f in frame_outputs], axis=0)
        paths = [f["path"] for f in frame_outputs]

        depths = None
        # depth_masks = None
        if self.load_depth:
            depth_list = []
            depth_mask_list = []
            for frame in frame_outputs:
                depth_arr = frame["depth"]
                # mask_arr = frame.get("depth_mask")
                if depth_arr is None:
                    depth_arr = np.full((self.depth_size, self.depth_size), np.inf, dtype=np.float32)
                    # mask_arr = np.zeros((self.depth_size, self.depth_size), dtype=np.uint8)
                # else:
                #     if mask_arr is None:
                #         mask_arr = np.logical_and(depth_arr >= 0.0, ~np.isnan(depth_arr)).astype(np.uint8)
                #     else:
                #         mask_arr = mask_arr.astype(np.uint8, copy=False)
                depth_list.append(depth_arr)
                # depth_mask_list.append(mask_arr)

            depths = np.stack(depth_list, axis=0)
            # depth_masks = np.stack(depth_mask_list, axis=0)

        c2ws_norm, depths_norm, scale = self._normalise_scene(c2ws, depths)
        camtoworld = torch.from_numpy(c2ws_norm).float()
        K = torch.from_numpy(Ks).float()

        result: Dict[str, Any] = {
            "sequence": sequence_name,
            "category": category,
            "image": torch.from_numpy(images).float(),
            # "mask": torch.from_numpy(masks).float(),
            "K": K,
            "camtoworld": camtoworld,
            "image_path": paths,
            "context_indices": context_indices,
            "target_indices": target_indices,
            "scene_scale": scale,
            "frame_indices": all_indices,
        }

        if depths_norm is not None:
            depths_tensor = torch.from_numpy(depths_norm).float()
            result["depths"] = depths_tensor
            result["context_depths"] = depths_tensor[:self.input_views].unsqueeze(-1)
            # if depth_masks is not None:
            #     result["depth_masks"] = torch.from_numpy(depth_masks).long()[:self.input_views].unsqueeze(-1)

        return result

def interpolate_camera_traj(
    c2w_ref: torch.Tensor, 
    K_ref: torch.Tensor, 
    num_frames: int = 24
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interpolate camera trajectory from reference views using SLERP for rotation
    and linear interpolation for translation.
    
    Args:
        c2w_ref: (2, 4, 4) reference camera-to-world matrices
        K_ref: (2, 3, 3) reference intrinsic matrices
        num_frames: Number of interpolated frames to generate
        
    Returns:
        c2w_traj: (num_frames, 4, 4) interpolated camera-to-world matrices
        K_traj: (num_frames, 3, 3) interpolated intrinsic matrices
    """
    from scipy.spatial.transform import Rotation, Slerp
    
    # Extract rotation matrices and translations
    R0 = c2w_ref[0, :3, :3].numpy()
    R1 = c2w_ref[1, :3, :3].numpy()
    t0 = c2w_ref[0, :3, 3].numpy()
    t1 = c2w_ref[1, :3, 3].numpy()
    
    # Convert rotation matrices to Rotation objects for SLERP
    rotations = Rotation.from_matrix(np.stack([R0, R1]))
    slerp = Slerp([0, 1], rotations)
    
    # Generate interpolation weights
    t_values = np.linspace(0, 1, num_frames)
    
    # Interpolate rotations using SLERP
    interp_rotations = slerp(t_values)
    interp_R = interp_rotations.as_matrix()  # (num_frames, 3, 3)
    
    # Linearly interpolate translations
    interp_t = (1 - t_values[:, None]) * t0 + t_values[:, None] * t1  # (num_frames, 3)
    
    # Construct interpolated c2w matrices
    c2w_traj = np.zeros((num_frames, 4, 4), dtype=np.float32)
    c2w_traj[:, :3, :3] = interp_R
    c2w_traj[:, :3, 3] = interp_t
    c2w_traj[:, 3, 3] = 1.0
    
    # Linearly interpolate intrinsics
    K0 = K_ref[0].numpy()
    K1 = K_ref[1].numpy()
    K_traj = np.zeros((num_frames, 3, 3), dtype=np.float32)
    for i, t in enumerate(t_values):
        K_traj[i] = (1 - t) * K0 + t * K1
    
    return torch.from_numpy(c2w_traj).float(), torch.from_numpy(K_traj).float()
