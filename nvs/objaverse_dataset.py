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

from .dataset import resize_crop_with_subpixel_accuracy

def scale_down_extrinsics(c2w: np.ndarray, scale: float = 0.02) -> np.ndarray:
    # Scale down the camera with respect the world origin
    c2w_scaled = c2w.copy()
    c2w_scaled[:3, 3] *= scale
    return c2w_scaled

def normalize_poses(c2w: np.ndarray, max_radius: float) -> np.ndarray:
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
    # Scale the c2w matrices (only the translation part)
    c2w_normalized = c2w.copy()
    c2w_normalized[:, :3, 3] *= scale
    
    return c2w_normalized, scale

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
    c2w_3by4 = np.array(extrinsics["camera_to_world"], dtype=np.float32) @ blender2opencv
    
    # Construct camera-to-world matrix
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :4] = np.array(c2w_3by4, dtype=np.float32)

    return K, c2w


def load_objaverse_frames(
    data_dir: str,
    cameras_data: Dict[str, Any],
    frame_ids: List[int],
    depth_ids: Optional[List[int]] = None,
    patch_size: int = 256,
    camera_pose_only: bool = False,
) -> Union[Dict[str, Any], np.ndarray]:
    """Load frames from Objaverse dataset.
    
    Args:
        data_dir: Path to the object directory
        cameras_data: Loaded cameras data from cameras.json
        frame_ids: List of frame indices to load
        patch_size: Target image size (images will be resized to patch_size x patch_size)
        camera_pose_only: If True, only return camera poses
        
    Returns:
        Dictionary with images, K matrices, camera poses, and image paths
        or just camera poses if camera_pose_only=True
    """
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    cameras = cameras_data["cameras"]
    
    # Shortcut for loading only camera poses
    if camera_pose_only:
        c2ws = []
        for frame_id in frame_ids:
            camera_info = cameras[frame_id]
            _, c2w = parse_objaverse_camera(camera_info)
            c2ws.append(c2w)
        return np.stack(c2ws)
    
    # Load images and camera parameters
    images, Ks, c2ws, abs_image_paths = [], [], [], []
    views_dir = os.path.join(data_dir, "views")
    
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

    # normalize the extrinsics
    c2ws = np.stack(c2ws)
    c2ws, normalize_scale = normalize_poses(c2ws, max_radius=0.5)

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
        "image": np.stack(images),
        "depth" : np.stack(depths) if depth_ids else None,
        "K": np.stack(Ks),
        "camtoworld": np.stack(c2ws),
        "image_path": abs_image_paths,
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
        target_view_ids = [file_to_view_id[fname] for fname in target_files]
        assert len(target_view_ids) >= self.supervise_views
        target_view_ids = random.sample(target_view_ids, self.supervise_views)
        # Combine context and target views
        all_view_ids = context_view_ids + target_view_ids
        
        # Load frames
        data = load_objaverse_frames(
            scene_path, 
            cameras_data, 
            all_view_ids,
            depth_ids=context_view_ids if self.get_depth else None,
            patch_size=self.patch_size
        )
        
        # Convert to torch tensors and normalize poses (same as RE10K dataset)
        camtoworld = torch.from_numpy(data["camtoworld"]).float()
        K = torch.from_numpy(data["K"]).float()
        image = torch.from_numpy(data["image"]).float()
        image_path = data["image_path"]
        context_depths = torch.from_numpy(data["depth"]).float() if self.get_depth else []
        # context_depths[context_depths > 1000.0] = torch.inf
        
        return {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "context_depths": context_depths,
            "image_path": image_path,
        }


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
        
        # Load frames
        data = load_objaverse_frames(
            scene_path, 
            cameras_data, 
            all_view_ids, 
            depth_ids=context_view_ids if self.get_depth else None,
            patch_size=self.patch_size
        )
        
        # Convert to torch tensors and normalize poses (same as RE10K dataset)
        camtoworld = torch.from_numpy(data["camtoworld"]).float()
        K = torch.from_numpy(data["K"]).float()
        image = torch.from_numpy(data["image"]).float()
        image_path = data["image_path"]

        context_depths = torch.from_numpy(data["depth"]).float() if self.get_depth else []
        
        return {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "context_depths": context_depths,
            "image_path": image_path,
            "scene": idx,
        }

