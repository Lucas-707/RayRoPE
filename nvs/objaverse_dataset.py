import json
import os, random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import imageio.v2 as imageio
import OpenEXR, Imath

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .re10k_dataset import resize_crop_with_subpixel_accuracy

def scale_down_extrinsics(c2w: np.ndarray, scale: float = 0.02) -> np.ndarray:
    # Scale down the camera with respect the world origin
    c2w_scaled = c2w.copy()
    c2w_scaled[:3, 3] *= scale
    return c2w_scaled

def normalize_view1_to_identity(c2ws: np.ndarray) -> np.ndarray:
    ref0_c2w = c2ws[0]
    c2ws_out = np.einsum("ij, vjk -> vik", np.linalg.inv(ref0_c2w), c2ws)
    return c2ws_out


def get_normalize_scale(c2w: np.ndarray, max_radius: float) -> np.ndarray:
    """Normalize camera-to-world matrices by scaling so max camera center distance equals max_radius.
    """
    # For c2w, camera center in world coords is just the translation: c2w[:, :3, 3]
    camera_centers = c2w[:, :3, 3]  # Shape: (B, 3)
    # Compute distance of each camera center to origin
    distances = np.linalg.norm(camera_centers, axis=1)  # Shape: (B,)
    max_distance = np.max(distances)
    # Compute scaling factor
    if max_distance > 0:
        scale = max_radius / max_distance
    else:
        scale = 1.0
    return scale

def load_objaverse_cameras(cameras_json_path: str) -> Dict[str, Any]:
    """Load camera information from Objaverse cameras.json file."""
    with open(cameras_json_path, "r") as f:
        cameras_data = json.load(f)
    return cameras_data


def parse_objaverse_camera(camera_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Parse individual camera info to get intrinsic and extrinsic matrices.
    
    Args:
        camera_info: Camera dictionary from cameras.json
        
    Returns:
        K: 3x3 intrinsic matrix
        c2w: 4x4 camera-to-world matrix
    """
    # Extract intrinsics
    intrinsics = camera_info["intrinsics"]
    fx, fy = intrinsics["focal_length"]
    cx, cy = intrinsics["principal_point"]
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Extract extrinsics
    extrinsics = camera_info["extrinsics"]

    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

    # c2w_blender = np.array(extrinsics["camera_to_world"], dtype=np.float32)
    # R_blender = c2w_blender[:3, :3]
    # t_blender = c2w_blender[:3, 3:4]
    # R_opencv = R_blender @ blender2opencv[:3, :3]
    # c2w_3by4 = np.concatenate([R_opencv, t_blender], axis=1)

    c2w_3by4 = np.array(extrinsics["camera_to_world"], dtype=np.float32) @ blender2opencv
    
    # Construct camera-to-world matrix
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :4] = np.array(c2w_3by4, dtype=np.float32)

    return K, c2w


def load_objaverse_frames(
    views_dir: str,
    cameras_data: Dict[str, Any],
    frame_ids: List[int],
    depth_ids: Optional[List[int]] = None,
    patch_size: int = 256,
    camera_pose_only: bool = False,
    normalize_ids: Optional[List[int]] = None,
    normalize_scale: Optional[float] = None,
) -> Union[Dict[str, Any], np.ndarray]:
    """Load frames from Objaverse dataset.
    
    Args:
        data_dir: Path to the object directory
        cameras_data: Loaded cameras data from cameras.json
        frame_ids: List of frame indices to load
        patch_size: Target image size (images will be resized to patch_size x patch_size)
        camera_pose_only: If True, only return camera poses
        normalize_ids: View indices used to estimate the global normalization scale.
            When None, falls back to the provided frame_ids.
        
    Returns:
        Dictionary with images, K matrices, camera poses, and image paths
        or just camera poses if camera_pose_only=True
    """
    cameras = cameras_data["cameras"]

    if normalize_scale is None:
        if normalize_ids is None:
            normalize_ids = frame_ids
        if len(normalize_ids) > 1:
            normalize_ids = list(dict.fromkeys(normalize_ids))

        normalize_c2ws = np.stack([
            parse_objaverse_camera(cameras[norm_id])[1]
            for norm_id in normalize_ids
        ])
        normalize_scale = get_normalize_scale(normalize_c2ws, max_radius=0.5)
    
    # Shortcut for loading only camera poses
    if camera_pose_only:
        c2ws = []
        for frame_id in frame_ids:
            camera_info = cameras[frame_id]
            _, c2w = parse_objaverse_camera(camera_info)
            c2w_scaled = c2w.copy()
            c2w_scaled[:3, 3] *= normalize_scale
            c2ws.append(c2w_scaled)
        return np.stack(c2ws)
    
    # Load images and camera parameters
    images, Ks, c2ws, abs_image_paths = [], [], [], []
    # views_dir = os.path.join(data_dir, "views")
    
    for frame_id in frame_ids:
        camera_info = cameras[frame_id]
        image_name = camera_info["image_name"]
        
        # Load image
        abs_image_path = os.path.join(views_dir, image_name)
        image = imageio.imread(abs_image_path)[..., :3]
        
        # Parse camera parameters
        K, c2w = parse_objaverse_camera(camera_info)
        
        # Resize and crop image using the same method as RealEstate10k
        image, K = resize_crop_with_subpixel_accuracy(image, K, patch_size)
        
        images.append(image)
        Ks.append(K)
        c2ws.append(c2w)
        abs_image_paths.append(abs_image_path)

    # normalize the extrinsics using the precomputed scale
    c2ws = np.stack(c2ws)
    c2ws[:, :3, 3] *= normalize_scale
    # c2ws = normalize_view1_to_identity(c2ws)

    # Compute foreground mask: 1 for foreground, 0 for background (pure white)
    stacked_images = np.stack(images)
    white_threshold = 250
    is_white = np.all(stacked_images >= white_threshold, axis=-1)  # (N, H, W)
    masks = (~is_white).astype(np.float32)[..., None]  # (N, H, W, 1), 1 for foreground

    if depth_ids:
        depths = []
        for depth_id in depth_ids:
            camera_info = cameras[depth_id]
            image_name = camera_info["image_name"]
            depth_name = image_name.replace(".jpg", "_depth.exr")
            abs_depth_path = os.path.join(views_dir, depth_name)
            f = OpenEXR.InputFile(abs_depth_path) 
            dw = f.header()['dataWindow']
            sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
            depth = np.frombuffer(f.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(sz)
            depth = depth[..., None]  # (H, W, 1)
            depth = depth * normalize_scale
            depths.append(depth)
    
    return {
        "image": stacked_images,
        "mask": masks,
        "depth" : np.stack(depths) if depth_ids else None,
        "K": np.stack(Ks),
        "camtoworld": np.stack(c2ws),
        "image_path": abs_image_paths,
        "normalize_scale": normalize_scale,
    }


class ObjaverseTrainDataset(Dataset):
    """Objaverse training dataset."""
    
    def __init__(
        self,
        scenes: List[str],
        index_file: str,
        patch_size: int = 256,
        input_views: int = 2,
        supervise_views: int = 6,
        get_depth: bool = False,
        get_mask: bool = False,
    ):
        """
        Args:
            scenes: List of scene directories (object directories)
            index_file: Path to the index JSON file (e.g., objaverse_index_train_context2.json)
            patch_size: Target image patch size
            supervise_views: Number of target views to supervise during training
        """
        self.scenes = scenes
        self.patch_size = patch_size
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.get_depth = get_depth
        self.get_mask = get_mask

        # Load index file
        with open(index_file, "r") as f:
            self.index_data = json.load(f)
        
        # Filter scenes to only include those in the index
        self.valid_scenes = []
        for scene_path in scenes:
            object_uid = os.path.basename(scene_path)
            if object_uid in self.index_data:
                self.valid_scenes.append(scene_path)
        
        # print(f"ObjaverseTrainDataset: {len(self.valid_scenes)} valid scenes out of {len(scenes)} total scenes")
    
    def __len__(self) -> int:
        return len(self.valid_scenes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:        
        scene_path = self.valid_scenes[idx]
        object_uid = os.path.basename(scene_path)
        
        # Get context and target views from index
        scene_info = self.index_data[object_uid]
        context_files = scene_info["context_view_files"]
        target_files = scene_info["target_view_files"]
        
        # Load cameras data
        cameras_json_path = os.path.join(scene_path, "cameras.json")
        cameras_data = load_objaverse_cameras(cameras_json_path)
        
        # Map image files to view IDs
        cameras = cameras_data["cameras"]
        file_to_view_id = {}
        for camera_info in cameras:
            view_id = camera_info["view_id"]
            image_name = camera_info["image_name"]
            file_to_view_id[image_name] = view_id
        
        # Get context view IDs
        context_view_ids = [file_to_view_id[fname] for fname in context_files]
        assert len(context_view_ids) == self.input_views
        
        # Get target view IDs (only use first supervise_views for training)
        all_target_view_ids = [file_to_view_id[fname] for fname in target_files]
        assert len(all_target_view_ids) >= self.supervise_views
        target_view_ids = random.sample(all_target_view_ids, self.supervise_views)
        # Combine context and target views
        all_view_ids = context_view_ids + target_view_ids
        normalize_ids = context_view_ids + all_target_view_ids
        
        # Load frames
        data = load_objaverse_frames(
            f"{scene_path}/views", 
            cameras_data, 
            all_view_ids,
            depth_ids=context_view_ids if self.get_depth else None,
            patch_size=self.patch_size,
            normalize_ids=normalize_ids,
        )
        
        # Convert to torch tensors and normalize poses (same as RE10K dataset)
        camtoworld = torch.from_numpy(data["camtoworld"]).float()
        K = torch.from_numpy(data["K"]).float()
        image = torch.from_numpy(data["image"]).float()
        image_path = data["image_path"]
        
        results = {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "image_path": image_path,
        }
    
        if self.get_depth:
            results["context_depths"] = torch.from_numpy(data["depth"]).float()
        if self.get_mask:
            results["mask"] = torch.from_numpy(data["mask"]).float()

        return results

class ObjaverseEvalDataset(Dataset):
    """Objaverse evaluation dataset."""
    
    def __init__(
        self,
        scenes: List[str],
        index_file: str,
        patch_size: int = 256,
        input_views: int = 2,
        supervise_views: int = 3,
        first_n: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        get_depth: bool = False,
        render_video: bool = False,
    ):
        """
        Args:
            scenes: List of scene directories (object directories)
            index_file: Path to the index JSON file (e.g., objaverse_index_test_context2.json)
            patch_size: Target image patch size
            input_views: Number of input/context views
            supervise_views: Number of target views for evaluation
            first_n: If provided, only use the first n scenes
            rank: Process rank for distributed evaluation
            world_size: Total number of processes for distributed evaluation
        """
        self.scenes = scenes
        self.patch_size = patch_size
        self.input_views = input_views
        # self.supervise_views = supervise_views
        self.get_depth = get_depth
        self.render_video = render_video
        if self.render_video:
            assert self.input_views >= 2, "Need at least 2 input views to render video."
        
        # Load index file
        with open(index_file, "r") as f:
            self.index_data = json.load(f)
        
        # Filter scenes to only include those in the index
        self.valid_scenes = []
        for scene_path in scenes:
            object_uid = os.path.basename(scene_path)
            if object_uid in self.index_data:
                self.valid_scenes.append(scene_path)
        
        # Sort scenes for consistent ordering
        self.valid_scenes = sorted(self.valid_scenes)
        
        # print(f"ObjaverseEvalDataset: {len(self.valid_scenes)} valid scenes out of {len(scenes)} total scenes")
        
        # Apply first_n limit if specified
        if first_n is not None:
            self.valid_scenes = self.valid_scenes[:first_n]
            # print(f"ObjaverseEvalDataset: Limited to first {len(self.valid_scenes)} scenes")
        
        # Apply distributed evaluation if rank and world_size are specified
        if rank is not None and world_size is not None:
            self.valid_scenes = self.valid_scenes[rank::world_size]
            # print(f"ObjaverseEvalDataset: Rank {rank}/{world_size} using {len(self.valid_scenes)} scenes")
    
    def __len__(self) -> int:
        return len(self.valid_scenes)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_path = self.valid_scenes[idx]
        object_uid = os.path.basename(scene_path)
        
        # Get context and target views from index
        scene_info = self.index_data[object_uid]
        context_files = scene_info["context_view_files"]
        target_files = scene_info["target_view_files"]
        
        # Load cameras data
        cameras_json_path = os.path.join(scene_path, "cameras.json")
        cameras_data = load_objaverse_cameras(cameras_json_path)
        
        # Map image files to view IDs
        cameras = cameras_data["cameras"]
        file_to_view_id = {}
        for camera_info in cameras:
            view_id = camera_info["view_id"]
            image_name = camera_info["image_name"]
            file_to_view_id[image_name] = view_id
        
        # Get context view IDs
        context_view_ids = [file_to_view_id[fname] for fname in context_files]
        assert len(context_view_ids) == self.input_views
        
        # Get ALL target view IDs for evaluation
        target_view_ids = [file_to_view_id[fname] for fname in target_files]
        # assert len(target_view_ids) >= self.supervise_views

        # Combine context and target views
        all_view_ids = context_view_ids + target_view_ids
        
        normalize_ids = all_view_ids + [0,3,6,9,12,15,18,21]
        # Load frames
        data = load_objaverse_frames(
            f"{scene_path}/views", 
            cameras_data, 
            all_view_ids, 
            depth_ids=context_view_ids if self.get_depth else None,
            patch_size=self.patch_size,
            normalize_ids=normalize_ids,
        )
        
        # Convert to torch tensors and normalize poses (same as RE10K dataset)
        camtoworld = torch.from_numpy(data["camtoworld"]).float()
        K = torch.from_numpy(data["K"]).float()
        image = torch.from_numpy(data["image"]).float()
        image_path = data["image_path"]

        if self.render_video:
            c2w_ref = camtoworld[:self.input_views]
            K_ref = K[:self.input_views]
            image_ref = image[:self.input_views]
            image_path_ref = image_path[:self.input_views]
            
            cameras_json_path_circ = os.path.join(scene_path, "cameras_circ_traj.json")
            cameras_data_circ = load_objaverse_cameras(cameras_json_path_circ)
            normalize_scale = data["normalize_scale"]
            num_frames = 60 # hard coded

            data_circ = load_objaverse_frames(
                f"{scene_path}/circ_traj", 
                cameras_data_circ, 
                list(range(num_frames)), 
                patch_size=self.patch_size,
                normalize_scale=normalize_scale,
            )

            c2w_circ = torch.from_numpy(data_circ["camtoworld"]).float()
            K_circ = torch.from_numpy(data_circ["K"]).float()
            image_circ = torch.from_numpy(data_circ["image"]).float()
            image_path_circ = data_circ["image_path"]

            camtoworld = torch.cat([c2w_ref, c2w_circ], dim=0)
            K = torch.cat([K_ref, K_circ], dim=0)
            image = torch.cat([image_ref, image_circ], dim=0)
            image_path = image_path_ref + image_path_circ
        
        results =  {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "image_path": image_path,
            "scene": idx,
        }

        if self.get_depth:
            results["context_depths"] = torch.from_numpy(data["depth"]).float()
        return results

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
