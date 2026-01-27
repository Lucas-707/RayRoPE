import submitit
import json
import pandas as pd
import os, sys, time
from pathlib import Path
import glob
import subprocess
from tqdm import tqdm
import pickle as pkl
import torch

'''
nohup python submitit_batch_render.py > submitit_batch_render.log 2>&1 &

'''

OBJV_DIR = "" # Todo: Set the path where the renderings and annotations will be saved
OBJV_GLB_ROOT = "" # Todo: Set the path to the downloaded Objaverse assets

# Configuration dictionary - Edit these parameters as needed
CONFIG = {
    # Data paths
    "glb_root": OBJV_GLB_ROOT,
    "output_dir": OBJV_DIR,
    "csv_path": "../assets/kiuisobj_v1_merged_80K.csv",
    "blender_script_path": "objv_render_vary_intrinsics.py",
    
    # Dataset selection
    "start_index": 0,
    "num_obj": 80000,
    
    # Blender rendering arguments
    "blender_args": {
        "num_views": "8",
        "min_fov": "20.0",
        "max_fov": "80.0",
        "target_coverage": "0.6",
        "seed": "1",
        "resolution_x": "256",
        "resolution_y": "256",
        "fix_radial": False,  # If True, use fixed radial distances
        "render_depth_only": False,  # If True, render only depth maps
        "render_depth": True,  # If True, render depth maps
        "save_mask": False,  # Set to True to include --save_mask flag
        "engine": "BLENDER_EEVEE",  # or "CYCLES"
    },
    
    # SLURM job configuration
    "slurm_config": {
        "log_folder": f"{OBJV_DIR}/logs",
        "slurm_array_parallelism": 100,
        "slurm_partition": "all",
        "cpus_per_task": 4,
        "gpus_per_node": 1,
        "timeout_min": 60,
        "slurm_exclude": "",
        "slurm_max_num_timeout": 3,
    },
    
    # Job management
    "max_job_per_split": 1000,
    "num_obj_per_job": 10,
    "sleep_time": 10,
    
    # Processing mode: "get_obj_paths", "get_unfinished_obj", or "get_singular_obj"
    "processing_mode": "get_unfinished_obj",
}


def _chunk_list(seq, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for idx in range(0, len(seq), chunk_size):
        yield seq[idx: idx + chunk_size]


def _render_task(obj_paths):
    if not obj_paths:
        print("No object paths provided to render task", flush=True)
        return

    # @ VARY INTRINSICS RENDER
    command = [
        "python",
        CONFIG["blender_script_path"],
        "--",
        "--output_dir",
        CONFIG["output_dir"],
        "--num_views",
        CONFIG["blender_args"]["num_views"],
        "--min_fov",
        CONFIG["blender_args"]["min_fov"],
        "--max_fov",
        CONFIG["blender_args"]["max_fov"],
        "--target_coverage",
        CONFIG["blender_args"]["target_coverage"],
        "--seed",
        CONFIG["blender_args"]["seed"],
        "--resolution_x",
        CONFIG["blender_args"]["resolution_x"],
        "--resolution_y",
        CONFIG["blender_args"]["resolution_y"],
        "--engine",
        CONFIG["blender_args"]["engine"],
    ]

    for obj_path in obj_paths:
        command.extend(["--obj_path", obj_path])

    # Add optional save_mask flag if enabled
    if CONFIG["blender_args"]["save_mask"]:
        command.append("--save_mask")
    if CONFIG["blender_args"]["fix_radial"]:
        command.append("--fix_radial")
    if CONFIG["blender_args"]["render_depth_only"]:
        command.append("--render_depth_only")
    if CONFIG["blender_args"]["render_depth"]:
        command.append("--render_depth")

    print(" ".join(command))
    process = subprocess.Popen(command, shell=False, stdout=subprocess.DEVNULL)
    process.wait()

def get_obj_paths(glb_root, start_index, num_obj):
    df = pd.read_csv(CONFIG["csv_path"])
    if start_index + num_obj + 1 > len(df):
        df = df[start_index:]
    else:
        df = df[start_index: start_index + num_obj + 1]
    obj_paths = []
    for i, row in df.iterrows():
        partition = row['partition']
        uid = row['uid']
        obj_path = f"{glb_root}/{partition}/{uid}.glb"
        obj_paths.append(obj_path)

    return obj_paths

def get_unfinished_obj(
        glb_root, 
        start_index=None, 
        num_obj=None, 
        output_root=None,
    ):
    if output_root is None:
        output_root = CONFIG["output_dir"]
    if start_index is None:
        start_index = CONFIG["start_index"]
    if num_obj is None:
        num_obj = CONFIG["num_obj"]
        
    df = pd.read_csv(CONFIG["csv_path"])
    
    # Apply start_index and num_obj filtering
    if start_index + num_obj + 1 > len(df):
        df = df[start_index:]
    else:
        df = df[start_index: start_index + num_obj + 1]
    
    def add_unfinished(path, reason=None):
        if path not in unfinished_seen:
            unfinished_seen.add(path)
            if reason:
                print(reason, flush=True)
            unfinished_paths.append(path)

    unfinished_paths = []
    unfinished_seen = set()
    for i, row in df.iterrows():
        partition = row['partition']
        uid = row['uid']
        output_path = f"{output_root}/{uid}"
        obj_path = f"{glb_root}/{partition}/{uid}.glb"
        if not os.path.exists(output_path):
            add_unfinished(obj_path)
        else:
            # Check for vary_intrinsics output format
            view_dir = Path(f"{output_path}/views")
            cameras_json = Path(f"{output_path}/cameras.json")
            
            # Remove any leftover PNG files
            for png_file in view_dir.glob('*.png'):
                png_file.unlink()
            
            # Check if cameras.json exists and has expected number of views
            if not cameras_json.exists():
                add_unfinished(obj_path)
                continue
            try:
                with cameras_json.open('r') as json_file:
                    json.load(json_file)
            except (json.JSONDecodeError, OSError) as exc:
                add_unfinished(
                    obj_path,
                    f"corrupted cameras.json for object {uid}: {exc}",
                )
                continue

                
            # Count JPG files in views directory
            num_views = len(list(view_dir.glob('*.jpg'))) if view_dir.exists() else 0
            num_depths = len(list(view_dir.glob('*_depth.exr'))) if view_dir.exists() else 0
            
            # Expected number of views from config
            expected_views = int(CONFIG["blender_args"]["num_views"])
            if not CONFIG["blender_args"]["fix_radial"]:
                expected_views = expected_views * 3
            
            if num_views < expected_views:
                add_unfinished(obj_path)
            if CONFIG["blender_args"]["render_depth_only"]:
                if num_depths < expected_views:
                    add_unfinished(obj_path)
        
        if i % 1000 == 0:
            print(
                f"Checked {i} objects; unfinished: {len(unfinished_paths)}",
                flush=True,
            )
    print(
        f"Checked {len(df)} objects; unfinished: {len(unfinished_paths)}",
        flush=True,
    )
    if len(unfinished_paths) < 100:
        print("Unfinished object paths:", flush=True)
        for path in unfinished_paths:
            print(path, flush=True)
    return unfinished_paths
        
def get_singular_obj(glb_root, output_root=None):
    if output_root is None:
        output_root = CONFIG["output_dir"]
    df = pd.read_csv(CONFIG["csv_path"])
    obj_paths = []
    for i, row in df.iterrows():
        partition = row['partition']
        uid = row['uid']
        obj_path = f"{glb_root}/{partition}/{uid}.glb"
        output_cam_path = f"{output_root}/{uid}/camera"

        cam0 = pkl.load(open(f"{output_cam_path}/000.pkl", 'rb'))
        cam99 = pkl.load(open(f"{output_cam_path}/099.pkl", 'rb'))
        try:
            torch.linalg.inv(cam0.R)
            torch.linalg.inv(cam99.R)
        except Exception:
            print(f"singular view object {uid}", flush=True)
            obj_paths.append(obj_path)
            continue
        
        if (cam0.get_camera_center() - torch.tensor([0,0,4])).norm() < 1e-3 or (cam99.get_camera_center() - torch.tensor([0,0,-4])).norm() < 1e-3:
            print(f"singular view object {uid}", flush=True)
            obj_paths.append(obj_path)

    with open("./singular_view_obj_path.pt", 'wb') as pickle_file:
        pkl.dump(obj_paths, pickle_file)
    return obj_paths
            

        
def launch_render_jobs():
    # Get object paths based on processing mode
    if CONFIG["processing_mode"] == "get_obj_paths":
        all_obj_path = get_obj_paths(CONFIG["glb_root"], CONFIG["start_index"], CONFIG["num_obj"])
    elif CONFIG["processing_mode"] == "get_unfinished_obj":
        all_obj_path = get_unfinished_obj(CONFIG["glb_root"], CONFIG["start_index"], CONFIG["num_obj"])
    elif CONFIG["processing_mode"] == "get_singular_obj":
        all_obj_path = get_singular_obj(CONFIG["glb_root"])
    else:
        raise ValueError(f"Invalid processing_mode: {CONFIG['processing_mode']}")

    for file_path in all_obj_path:
        if not os.path.isfile(file_path):
            print(f"{file_path} does not exist", flush=True)

    total_objects = len(all_obj_path)
    if total_objects == 0:
        print("No jobs to launch", flush=True)
        return

    num_per_job = int(CONFIG.get("num_obj_per_job", 1))
    if num_per_job <= 0:
        raise ValueError("CONFIG['num_obj_per_job'] must be a positive integer")

    obj_batches = list(_chunk_list(all_obj_path, num_per_job))
    total_jobs = len(obj_batches)
    print(
        f"number of total jobs = {total_jobs} (covering {total_objects} objects)",
        flush=True,
    )

    # Setup SLURM executor with config parameters
    executor = submitit.AutoExecutor(
        folder=CONFIG["slurm_config"]["log_folder"], 
        slurm_max_num_timeout=CONFIG["slurm_config"]["slurm_max_num_timeout"]
    )
    executor.update_parameters(
        slurm_array_parallelism=CONFIG["slurm_config"]["slurm_array_parallelism"],
        slurm_partition=CONFIG["slurm_config"]["slurm_partition"],
        cpus_per_task=CONFIG["slurm_config"]["cpus_per_task"],
        gpus_per_node=CONFIG["slurm_config"]["gpus_per_node"],
        timeout_min=CONFIG["slurm_config"]["timeout_min"],
        slurm_exclude=CONFIG["slurm_config"]["slurm_exclude"]
    )

    max_concurrent = min(CONFIG["max_job_per_split"], total_jobs)
    sleep_time = CONFIG["sleep_time"]
    start_time = time.time()

    job_iter = iter(obj_batches)
    running_jobs = []

    # Prime the queue up to the concurrency limit.
    for _ in range(max_concurrent):
        obj_batch = next(job_iter, None)
        if obj_batch is None:
            break
        job = executor.submit(_render_task, obj_batch)
        running_jobs.append((job, obj_batch))

    submitted = len(running_jobs)
    submitted_objects = sum(len(batch) for _, batch in running_jobs)
    finished = 0
    finished_objects = 0
    print(
        f"Submitted {submitted} initial jobs (limit {max_concurrent}) covering {submitted_objects} objects",
        flush=True,
    )

    while running_jobs:
        time.sleep(sleep_time)
        elapsed_minutes = (time.time() - start_time) / 60

        next_batch = []
        for job, obj_batch in running_jobs:
            if job.done():
                try:
                    job.result()
                except Exception as exc:
                    print(f"Job failed for batch {obj_batch}: {exc}", flush=True)
                finished += 1
                finished_objects += len(obj_batch)
                next_batch_paths = next(job_iter, None)
                if next_batch_paths is not None:
                    new_job = executor.submit(_render_task, next_batch_paths)
                    next_batch.append((new_job, next_batch_paths))
                    submitted += 1
                    submitted_objects += len(next_batch_paths)
            else:
                next_batch.append((job, obj_batch))

        running_jobs = next_batch
        print(
            f"{elapsed_minutes:.1f} min elapsed | finished {finished}/{total_jobs} jobs ({finished_objects}/{total_objects} objects) "
            f"| in-flight {len(running_jobs)} | submitted {submitted} jobs ({submitted_objects}/{total_objects} objects)",
            flush=True,
        )

    total_minutes = (time.time() - start_time) / 60
    print(
        f"All jobs completed in {total_minutes:.1f} minutes for {total_objects} objects",
        flush=True,
    )

if __name__ == '__main__':
    launch_render_jobs()