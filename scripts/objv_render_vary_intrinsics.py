"""
Exaample usage:

python scripts/objv_render_vary_intrinsics.py -- \
    --obj_path /grogu/datasets/objaverse/glbs/000-097/24a550a5d6e54402809a75eb8654230e.glb \
    --output_dir ./test_blender \

python scripts/objv_render_vary_intrinsics.py -- \
    --obj_path /grogu/datasets/objaverse/glbs/000-145/98bc91e1119d40b3b703de8ca975266c.glb \
    --output_dir ./test_blender \
    --render_depth

"""
import sys
import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import glob
from pathlib import Path
import bpy
from mathutils import Vector
import numpy as np
import imageio.v2 as imageio
import cv2
import json
# from contextlib import redirect_stdout




def setup_blender_scene():
    """Setup Blender scene with render settings based on command line arguments."""
    context = bpy.context
    scene = context.scene
    render = scene.render

    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = args.resolution_x
    render.resolution_y = args.resolution_y
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True


def get_object_dimensions():
    """Get the dimensions of the loaded 3D object after normalization."""
    bbox_min, bbox_max = scene_bbox()
    dimensions = bbox_max - bbox_min
    height = dimensions.z  # Z is typically the height
    width = max(dimensions.x, dimensions.y)  # Use the larger of X or Y as width
    depth = min(dimensions.x, dimensions.y)  # Use the smaller as depth
    return height, width, depth, dimensions


def sample_camera_intrinsics_and_poses(
        num_views: int, 
        min_fov: float,
        max_fov: float, 
        object_height: float, 
        target_coverage: float = 0.6, 
        seed: int = None,
        fix_radial: bool = False):
    """
    Sample camera intrinsics (FOV) and corresponding poses.
    
    If fix_radial=False (default):
        For each spherical angle, generates 3 views:
        - Original: sampled FOV and distance
        - Variant 1: same angles and distance, different FOV
        - Variant 2: same angles and FOV, different distance
    
    If fix_radial=True:
        For each spherical angle, generates 1 view with fixed FOV=40 degrees
    
    Args:
        num_views: Number of unique spherical angles 
        min_fov: Minimum vertical field of view in degrees (ignored if fix_radial=True)
        max_fov: Maximum vertical field of view in degrees (ignored if fix_radial=True)
        object_height: Height of the 3D object
        target_coverage: Target coverage of object height in image (0.8 = 80% of image height)
        seed: Random seed for reproducibility (optional)
        fix_radial: If True, use fixed FOV=40 degrees and single view per angle
        
    Returns:
        fovs: Array of vertical field of view values in degrees
        camera_distances: Array of camera distances
        positions: Array of camera positions (3, total_views)
        elevations: Array of elevation angles in degrees
        azimuths: Array of azimuth angles in degrees
        view_indices: Array of view indices
        radial_indices: Array of radial indices (always 0 if fix_radial=True)
    """
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Handle fixed radial case
    if fix_radial:
        # Use fixed FOV of 40 degrees
        fixed_fov = 40.0
        fovs_orig = np.full(num_views, fixed_fov)
        
        # Sample elevations uniformly from -10 to 30 degrees
        elevations_orig = np.random.uniform(-10, 30, num_views)
        
        # Sample azimuths deterministically from 0 to 360 degrees
        azimuths_orig = np.linspace(0, 360, num_views, endpoint=False)
        rand_offset_lim = 360 / num_views / 2
        azimuths_orig += np.random.uniform(-rand_offset_lim, rand_offset_lim, num_views)
        
        # Compute fixed camera distances
        fov_rad = np.radians(fixed_fov / 2)
        camera_distances_orig = (object_height / target_coverage) / (2 * np.tan(fov_rad))
        camera_distances_orig = np.full(num_views, camera_distances_orig)
        
        # Only one view per angle
        total_views = num_views
        fovs = fovs_orig.copy()
        camera_distances = camera_distances_orig.copy()
        elevations = elevations_orig.copy()
        azimuths = azimuths_orig.copy()
        view_indices = np.arange(num_views, dtype=int)
        radial_indices = np.zeros(num_views, dtype=int)
        
        print(f"Generated {total_views} views with fixed FOV={fixed_fov}° and fixed distance={camera_distances_orig[0]:.3f}")
        
    else:
        # Original radial sampling behavior
        # Convert FOV to focal length for sampling in log space
        # For vertical FOV: focal_length = image_height / (2 * tan(fov/2))
        # We'll use a reference image height of 100 pixels for focal length calculation
        reference_height = 100.0
        
        # Convert min/max FOV to focal lengths
        min_fov_rad = np.radians(min_fov)
        max_fov_rad = np.radians(max_fov)
        max_focal_length = reference_height / (2 * np.tan(min_fov_rad / 2))  # min FOV -> max focal length
        min_focal_length = reference_height / (2 * np.tan(max_fov_rad / 2))  # max FOV -> min focal length
        
        # Sample linearly in log space of focal length for original views
        log_min_focal = np.log(min_focal_length)
        log_max_focal = np.log(max_focal_length)
        log_focal_lengths_orig = np.random.uniform(log_min_focal, log_max_focal, num_views)
        focal_lengths_orig = np.exp(log_focal_lengths_orig)
        
        # Convert focal lengths back to FOVs for original views
        fovs_orig = 2 * np.arctan(reference_height / (2 * focal_lengths_orig))
        fovs_orig = np.degrees(fovs_orig)  # Convert to degrees
        
        # Sample elevations uniformly from -10 to 30 degrees
        elevations_orig = np.random.uniform(-10, 30, num_views)
        
        # Sample azimuths deterministically from 0 to 360 degrees
        azimuths_orig = np.linspace(0, 360, num_views, endpoint=False)
        rand_offset_lim = 360 / num_views / 2
        azimuths_orig += np.random.uniform(-rand_offset_lim, rand_offset_lim, num_views)
        
        # Compute camera distances for original views
        fov_rads_orig = np.radians(fovs_orig / 2)
        camera_distances_orig = (object_height / target_coverage) / (2 * np.tan(fov_rads_orig))
        camera_distances_orig = camera_distances_orig * np.random.uniform(0.9, 1.1, num_views)
        
        # Now generate 3 views per spherical angle
        total_views = 3 * num_views
        fovs = np.zeros(total_views)
        camera_distances = np.zeros(total_views)
        elevations = np.zeros(total_views)
        azimuths = np.zeros(total_views)
        view_indices = np.zeros(total_views, dtype=int)
        radial_indices = np.zeros(total_views, dtype=int)
        
        for i in range(num_views):
            base_idx = i * 3
            
            # View 0: Original view
            fovs[base_idx] = fovs_orig[i]
            camera_distances[base_idx] = camera_distances_orig[i]
            elevations[base_idx] = elevations_orig[i]
            azimuths[base_idx] = azimuths_orig[i]
            view_indices[base_idx] = i
            radial_indices[base_idx] = 0
            
            # View 1: Same angles and distance, different FOV
            # Sample new FOV in log focal length space
            log_focal_new = log_focal_lengths_orig[i] * np.random.uniform(0.8, 1.2)
            focal_new = np.exp(log_focal_new)
            fov_new = 2 * np.arctan(reference_height / (2 * focal_new))
            fov_new = np.degrees(fov_new)
            
            fovs[base_idx + 1] = fov_new
            camera_distances[base_idx + 1] = camera_distances_orig[i]
            elevations[base_idx + 1] = elevations_orig[i]
            azimuths[base_idx + 1] = azimuths_orig[i]
            view_indices[base_idx + 1] = i
            radial_indices[base_idx + 1] = 1
            
            # View 2: Same angles and FOV, different distance
            # Compute new distance based on original FOV but with random variation
            fov_rad = np.radians(fovs_orig[i] / 2)
            base_distance = (object_height / target_coverage) / (2 * np.tan(fov_rad))
            distance_new = base_distance * np.random.uniform(0.8, 1.2)  # Wider range for distance variation
            
            fovs[base_idx + 2] = fovs_orig[i]
            camera_distances[base_idx + 2] = distance_new
            elevations[base_idx + 2] = elevations_orig[i]
            azimuths[base_idx + 2] = azimuths_orig[i]
            view_indices[base_idx + 2] = i
            radial_indices[base_idx + 2] = 2
        
        print(f"Generated {total_views} total views ({num_views} angles × 3 views each)")
        print("Original FOVs (degrees): ", fovs_orig)
        print("Original camera distances: ", camera_distances_orig)
    
    # Convert to spherical coordinates and then to Cartesian (vectorized)
    elev_rads = np.radians(elevations)
    azim_rads = np.radians(azimuths)
    
    # Convert elevation to polar angle (90° - elevation)
    polar_angles = np.radians(90 - elevations)
    
    # Vectorized computation of positions
    positions = np.zeros((3, total_views))
    positions[0, :] = camera_distances * np.sin(polar_angles) * np.cos(azim_rads)  # x
    positions[1, :] = camera_distances * np.sin(polar_angles) * np.sin(azim_rads)  # y
    positions[2, :] = camera_distances * np.cos(polar_angles)  # z
    
    return fovs, camera_distances, positions, elevations, azimuths, view_indices, radial_indices


def add_lighting() -> None:
    # delete the default light
    try:
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()
    except:
        pass
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")
    return


def collect_material_states():
    """Capture original material state so we can toggle transparency settings."""
    materials = {}
    for obj in scene_meshes():
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            key = id(mat)
            if key in materials:
                continue
            materials[key] = {
                "material": mat,
                "blend_method": getattr(mat, "blend_method", None),
                "shadow_method": getattr(mat, "shadow_method", None),
            }
    return list(materials.values())


def apply_depth_material_overrides(material_states):
    """Force alpha-blended Eevee materials to write to depth when needed."""
    if args.engine != "BLENDER_EEVEE":
        return

    transparent_modes = {"BLEND", "ADD", "MULTIPLY"}
    for state in material_states:
        mat = state["material"]
        if hasattr(mat, "blend_method"):
            blend_method = getattr(mat, "blend_method", None)
            if blend_method in transparent_modes:
                mat.blend_method = "HASHED"

        if hasattr(mat, "shadow_method"):
            shadow_method = getattr(mat, "shadow_method", None)
            if shadow_method in {None, "NONE"}:
                mat.shadow_method = "HASHED"


def restore_material_states(material_states):
    """Restore materials to their original transparency configuration."""
    for state in material_states:
        mat = state["material"]
        original_blend = state.get("blend_method")
        if hasattr(mat, "blend_method") and original_blend is not None:
            mat.blend_method = original_blend

        original_shadow = state.get("shadow_method")
        if hasattr(mat, "shadow_method"):
            if original_shadow is not None:
                mat.shadow_method = original_shadow
            else:
                # Reset to Eevee default when the original value was not set explicitly.
                mat.shadow_method = 'OPAQUE'


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    # print("bbox max, min: ", bbox_max, bbox_min)
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    scene = bpy.context.scene
    cam = scene.objects["Camera"]
    # Initial position will be set per view
    cam.location = (0, 0, 3.0)  # Default initial position
    # Configure camera for vertical FOV
    cam.data.sensor_fit = 'VERTICAL'  # Ensure angle corresponds to image height
    # Initial FOV will be set per view
    cam.data.angle = np.radians((args.min_fov + args.max_fov) / 2)
    print(f"Camera setup - Vertical FOV range: {args.min_fov}° to {args.max_fov}°")
    cam.data.sensor_width = 32
    cam.data.sensor_height = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def setup_depth_rendering():
    """Enable depth pass for rendering metric depth."""
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_z = True


def render_and_save_depth(scene, depth_path):
    """Render and save metric depth map in OpenEXR format."""
    # Set up compositor nodes to output depth
    scene.use_nodes = True
    tree = scene.node_tree
    
    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # Create nodes
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    output_file = tree.nodes.new('CompositorNodeOutputFile')
    
    # Configure output
    output_dir = os.path.dirname(depth_path)
    base_name = os.path.basename(depth_path).replace('.exr', '')
    
    output_file.base_path = output_dir
    output_file.file_slots[0].path = base_name + '_'  # Blender will append frame number
    output_file.format.file_format = 'OPEN_EXR'
    output_file.format.color_depth = '32'
    output_file.format.color_mode = 'RGB'  # OpenEXR doesn't support 'BW', use 'RGB' for single channel
    
    # Connect depth output
    tree.links.new(render_layers.outputs['Depth'], output_file.inputs[0])
    
    # Render
    bpy.ops.render.render(write_still=False)
    
    # Blender adds frame number suffix, rename to desired name
    generated_path = os.path.join(output_dir, base_name + '_0001.exr')
    if os.path.exists(generated_path):
        os.rename(generated_path, depth_path)
    
    # Clean up compositor
    tree.nodes.clear()
    scene.use_nodes = False



def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    scene = bpy.context.scene
    
    #os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    # load the object
    print(f"object file: ", object_file)
    load_object(object_file)
    material_states = collect_material_states()

    # print the object info
    # loaded_obj = bpy.data.objects[3]
    # print(loaded_obj.name, loaded_obj.type, loaded_obj.location, [v[:] for v in loaded_obj.bound_box], loaded_obj.scale)

    object_uid = os.path.basename(object_file).split(".")[0]
    object_folder = object_file.split("/")[-2]
    normalize_scene()
    loaded_obj = bpy.data.objects[3]
    
    # Get object dimensions after normalization
    object_height, object_width, object_depth, object_dimensions = get_object_dimensions()
    print(f"Object dimensions - Height: {object_height:.3f}, Width: {object_width:.3f}, Depth: {object_depth:.3f}")
    object_dimension = max(object_dimensions)
    
    add_lighting()
    cam, cam_constraint = setup_camera()
    
    # Enable depth rendering if requested
    if args.render_depth or args.render_depth_only:
        setup_depth_rendering()
    
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    
    # Sample camera intrinsics and poses using object dimensions
    fovs, camera_distances, positions, elevations, azimuths, view_indices, radial_indices = sample_camera_intrinsics_and_poses(
        args.num_views, args.min_fov, args.max_fov, object_dimension, target_coverage=args.target_coverage, seed=args.seed, fix_radial=args.fix_radial
    )
    
    total_views = len(fovs)
    if args.fix_radial:
        print(f"Sampled {total_views} views with fixed FOV=40° and fixed distance")
    else:
        print(f"Sampled {total_views} total views ({args.num_views} angles × 3 views each) with FOVs from {args.min_fov}° to {args.max_fov}°")
    print(f"Camera distances range: {camera_distances.min():.2f} to {camera_distances.max():.2f}")

    # Initialize list to store all camera parameters
    all_camera_params = []

    for i in range(total_views):
        # Set camera position and FOV
        point = positions[:, i].tolist()
        fov = fovs[i]
        cam_dist = camera_distances[i]
        elevation = elevations[i]
        azimuth = azimuths[i]
        view_idx = view_indices[i]
        radial_idx = radial_indices[i]
        
        cam.location = point
        # Set camera vertical FOV (for image height)
        # In Blender, we need to ensure the angle corresponds to vertical FOV
        # by setting the sensor fit mode appropriately
        cam.data.sensor_fit = 'VERTICAL'  # Ensure FOV corresponds to image height
        cam.data.angle = np.radians(fov)
        
        # Choose file naming based on fix_radial option
        if args.fix_radial:
            img_name = f"{view_idx:03d}.png"
        else:
            img_name = f"{view_idx:03d}_{radial_idx}.png"
        
        # render the image (skip if render_depth_only is True)
        if not args.render_depth_only:
            render_path = os.path.join(args.output_dir, object_uid, "views", img_name)
            scene.render.filepath = render_path
            
            bpy.ops.render.render(write_still=True)
        
        # Render depth if enabled
        if args.render_depth or args.render_depth_only:
            depth_name = img_name.replace('.png', '_depth.exr')
            depth_path = os.path.join(args.output_dir, object_uid, "views", depth_name)
            apply_depth_material_overrides(material_states)
            render_and_save_depth(scene, depth_path)
            restore_material_states(material_states)
        
        # Calculate camera parameters for saving
        focal_length = cam.data.lens
        sensor_width = cam.data.sensor_width
        sensor_height = cam.data.sensor_height
        image_width = bpy.context.scene.render.resolution_x
        image_height = bpy.context.scene.render.resolution_y
        
        # Calculate focal length in pixels
        focal_x = (focal_length / sensor_width) * image_width
        focal_y = (focal_length / sensor_height) * image_height
        
        # Principal point (assuming center of image)
        principal_x = image_width / 2.0
        principal_y = image_height / 2.0
        
        # Create camera rotation matrix (look-at transformation)
        # Camera looks towards origin (0,0,0) from position 'point'
        forward = -np.array(point) / np.linalg.norm(point)  # Camera looks towards origin
        up = np.array([0, 0, 1])  # World up vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Rotation matrix (world to camera)
        R = np.array([right, up, -forward])
        
        # Translation vector (world to camera)
        T = -R @ np.array(point)
        
        # Create 3x4 extrinsics matrices
        # World to camera transform: [R | T]
        world_to_camera = np.hstack([R, T.reshape(-1, 1)])  # 3x4 matrix
        
        # Camera to world transform: [R^T | -R^T * T]
        R_inv = R.T
        T_inv = -R_inv @ T
        camera_to_world = np.hstack([R_inv, T_inv.reshape(-1, 1)])  # 3x4 matrix
        
        # Camera parameters for this view
        view_camera_params = {
            "view_id": i,
            "view_idx": int(view_idx),
            "radial_idx": int(radial_idx),
            "image_name": img_name.replace(".png", ".jpg"),  # Since we convert to jpg later
            "intrinsics": {
                "focal_length": [focal_x, focal_y],
                "principal_point": [principal_x, principal_y],
                "image_size": [image_width, image_height],
                "fov_degrees": float(fov)
            },
            "extrinsics": {
                "world_to_camera": world_to_camera.tolist(),  # 3x4 matrix
                "camera_to_world": camera_to_world.tolist(),  # 3x4 matrix
                "camera_position": point,
                "elevation_degrees": float(elevation),
                "azimuth_degrees": float(azimuth),
                "camera_distance": float(cam_dist)
            }
        }
        
        all_camera_params.append(view_camera_params)

    # Save all camera parameters in a single JSON file
    if not os.path.exists(os.path.join(args.output_dir, object_uid)):
        os.makedirs(os.path.join(args.output_dir, object_uid))
    
    camera_data = {
        "object_uid": object_uid,
        "num_angles": args.num_views,
        "total_views": total_views,
        "views_per_angle": 1 if args.fix_radial else 3,
        "fix_radial": args.fix_radial,
        "object_info": {
            "height": float(object_height),
            "width": float(object_width),
            "depth": float(object_depth),
            "dimensions": [float(d) for d in object_dimensions]
        },
        "sampling_params": {
            "min_fov_degrees": args.min_fov if not args.fix_radial else 40.0,
            "max_fov_degrees": args.max_fov if not args.fix_radial else 40.0,
            "target_coverage": args.target_coverage,
            "random_seed": args.seed,
            "elevation_range": [-10, 30],
            "azimuth_range": [0, 360]
        },
        "cameras": all_camera_params
    }
    
    # Add radial index description only if not using fix_radial
    if not args.fix_radial:
        camera_data["radial_idx_description"] = {
            "0": "Original sampled view",
            "1": "Same angles and distance, different FOV", 
            "2": "Same angles and FOV, different distance"
        }
    
    with open(os.path.join(args.output_dir, object_uid, "cameras.json"), 'w') as json_file:
        json.dump(camera_data, json_file, indent=2)

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_path",
        dest="obj_paths",
        action="append",
        required=True,
        help="Path to a .glb or .fbx file; repeat flag to process multiple objects.",
    )
    parser.add_argument("--background", type=str, default="white")
    parser.add_argument("--output_dir", type=str, default="/grogu/user/yuwu3/objaverse80k_sp/data")
    parser.add_argument("--dump_log", type=str, default="/home/yuwu3/VideoData/objaverse/dump.out")
    parser.add_argument(
        "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
    )
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--min_fov", type=float, default=20.0, help="Minimum vertical field of view in degrees (for image height)")
    parser.add_argument("--max_fov", type=float, default=80.0, help="Maximum vertical field of view in degrees (for image height)")
    parser.add_argument("--target_coverage", type=float, default=0.6, help="Target object coverage in image (0.8 = 80% of image height)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--fix_radial", action="store_true", default=False, help="Use fixed FOV of 40 degrees and single view per angle")
    parser.add_argument("--save_mask", action="store_true", default=False, help="Save mask images")
    parser.add_argument("--render_depth", action="store_true", default=False, help="Render and save metric depth maps")
    parser.add_argument("--render_depth_only", action="store_true", default=False, help="Only render depth maps, skip RGB rendering")
    parser.add_argument("--resolution_x", type=int, default=256, help="Image width resolution")
    parser.add_argument("--resolution_y", type=int, default=256, help="Image height resolution")


    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    if not args.obj_paths:
        raise ValueError("At least one --obj_path must be provided.")

    total_start = time.time()

    # Seed random number generators for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Setup Blender scene with command line arguments
    setup_blender_scene()

    open(args.dump_log, "w").close()
    old_stdout = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.close(sys.stdout.fileno())
    fd = os.open(args.dump_log, os.O_WRONLY)

    try:
        for local_path in args.obj_paths:
            object_start = time.time()
            print(f"=== Rendering {local_path} ===", flush=True)
            save_images(local_path)

            if not args.render_depth_only:
                object_uid = Path(local_path).stem
                view_glob = os.path.join(args.output_dir, object_uid, "views", "*.png")
                for img in glob.glob(view_glob):
                    img_origin = imageio.imread(img)
                    img_mask = img_origin[:, :, 3]
                    img_mask = ((img_mask > 0) * 255).astype(np.uint8)
                    img_jpg = img_origin[:, :, :3]
                    if args.background == "white":
                        mask3 = np.repeat(img_mask[:, :, np.newaxis], 3).reshape(
                            args.resolution_y, args.resolution_x, 3
                        )
                        img_jpg[mask3 == 0] = 255

                    img_jpg = cv2.cvtColor(img_jpg, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(img.replace(".png", ".jpg"), img_jpg)

                    # Only save mask images if save_mask is True
                    if args.save_mask:
                        imageio.imwrite(img.replace(".png", "_mask.jpg"), img_mask)

                    os.remove(img)

            object_elapsed = time.time() - object_start
            print(
                f"Finished {local_path} in {object_elapsed:.2f} seconds",
                flush=True,
            )
    finally:
        os.close(fd)
        os.dup(old_stdout)
        os.close(old_stdout)

    total_elapsed = time.time() - total_start
    print(
        f"Completed {len(args.obj_paths)} objects in {total_elapsed:.2f} seconds",
        flush=True,
    )
        