#!/usr/bin/env python3
# python ./scripts/re10k_lvsm2prope_data.py
# python ./scripts/re10k_lvsm2prope_data.py > re10k_lvsm2prope_transform.log 2>&1
"""
Script to transform RE10K dataset structure from LVSM format to PROPE format.

Original structure:
./re10k
└── test
    ├── full_list.txt
    ├── images/
    │   ├── seq1/
    │   │   ├── 00000.png
    │   │   ├── 00001.png
    │   │   └── ...
    │   └── seq2/
    │       └── ...
    └── metadata/
        ├── seq1.json
        ├── seq2.json
        └── ...

Desired structure:
./re10k_processed
└── test/
    ├── seq1/
    │   ├── images/
    │   │   ├── 00000.png (symlink)
    │   │   ├── 00001.png (symlink)
    │   │   └── ...
    │   └── transforms.json
    └── seq2/
        ├── images/
        └── transforms.json
"""

import os
import json
import glob
from pathlib import Path
# from tqdm import tqdm
import numpy as np
import sys
import argparse


def load_metadata(metadata_path: str) -> dict:
    """Load metadata JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def convert_metadata_to_transforms(metadata: dict) -> dict:
    """Convert LVSM metadata format to PROPE transforms format."""
    frames = metadata["frames"]
    if not frames:
        return None
        
    # Get image dimensions from the first frame
    first_frame = frames[0]
    fxfycxcy = first_frame["fxfycxcy"]
    fx, fy, cx, cy = fxfycxcy
    
    # Assume standard image size for RE10K (640x360)
    w, h = 640, 360
    
    transforms = {
        "w": w,
        "h": h,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "frames": []
    }
    
    for frame in frames:
        # Extract world-to-camera matrix
        w2c = np.array(frame["w2c"], dtype=np.float32)
        
        # Convert to 4x4 matrix if it's 3x4
        if w2c.shape == (3, 4):
            w2c_4x4 = np.eye(4, dtype=np.float32)
            w2c_4x4[:3, :4] = w2c
            w2c = w2c_4x4
        
        # Convert world-to-camera to camera-to-world
        c2w = np.linalg.inv(w2c)
        
        # Extract image filename from path
        image_path = frame["image_path"]
        image_filename = os.path.basename(image_path)
        
        frame_data = {
            "file_path": f"images/{image_filename}",
            "transform_matrix": c2w.tolist()
        }
        transforms["frames"].append(frame_data)
    
    return transforms


def is_sequence_processed(seq_name: str, output_dir: str, split: str = "test") -> bool:
    """Check if a sequence has already been processed."""
    output_seq_dir = os.path.join(output_dir, split, seq_name)
    output_images_dir = os.path.join(output_seq_dir, "images")
    output_transforms_path = os.path.join(output_seq_dir, "transforms.json")
    
    # Check if both the images directory and transforms.json exist
    if os.path.exists(output_images_dir) and os.path.exists(output_transforms_path):
        # Additional check: make sure images directory is not empty
        image_files = [f for f in os.listdir(output_images_dir) if f.endswith('.png')]
        if not image_files:
            return False
        
        # Check if number of images matches transforms.json entries
        try:
            with open(output_transforms_path, 'r') as f:
                transforms = json.load(f)
            
            num_transforms = len(transforms.get('frames', []))
            num_images = len(image_files)
            
            if num_images != num_transforms:
                return False
                
            return True
        except (json.JSONDecodeError, KeyError):
            return False
    
    return False


def process_sequence(seq_name: str, source_dir: str, output_dir: str, split: str = "test"):
    """Process a single sequence from LVSM format to PROPE format."""
    # Check if sequence is already processed
    if is_sequence_processed(seq_name, output_dir, split):
        print(f"Sequence {seq_name} already processed, skipping...", flush=True)
        return True
    
    # Paths
    source_images_dir = os.path.join(source_dir, split, "images", seq_name)
    source_metadata_path = os.path.join(source_dir, split, "metadata", f"{seq_name}.json")
    
    output_seq_dir = os.path.join(output_dir, split, seq_name)
    output_images_dir = os.path.join(output_seq_dir, "images")
    output_transforms_path = os.path.join(output_seq_dir, "transforms.json")
    
    # Check if source files exist
    if not os.path.exists(source_images_dir):
        print(f"Warning: Images directory not found for sequence {seq_name}", flush=True)
        return False
        
    if not os.path.exists(source_metadata_path):
        print(f"Warning: Metadata file not found for sequence {seq_name}", flush=True)
        return False
    
    # Create output directories
    os.makedirs(output_seq_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Load and convert metadata
    try:
        metadata = load_metadata(source_metadata_path)
        transforms = convert_metadata_to_transforms(metadata)
        
        if transforms is None:
            print(f"Warning: No frames found in metadata for sequence {seq_name}", flush=True)
            return False
            
    except Exception as e:
        print(f"Error processing metadata for sequence {seq_name}: {e}", flush=True)
        return False
    
    # Create symbolic links for images
    try:
        image_files = sorted(glob.glob(os.path.join(source_images_dir, "*.png")))
        
        for image_file in image_files:
            image_filename = os.path.basename(image_file)
            target_path = os.path.join(output_images_dir, image_filename)
            
            # Remove existing symlink if it exists
            if os.path.islink(target_path):
                os.unlink(target_path)
            elif os.path.exists(target_path):
                os.remove(target_path)
            
            # Create symbolic link
            os.symlink(os.path.abspath(image_file), target_path)
            
    except Exception as e:
        print(f"Error creating symlinks for sequence {seq_name}: {e}", flush=True)
        return False
    
    # Save transforms.json
    try:
        with open(output_transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)
    except Exception as e:
        print(f"Error saving transforms for sequence {seq_name}: {e}", flush=True)
        return False
    
    return True


def transform_dataset(source_dir: str, output_dir: str, split: str = "test"):
    """Transform the entire dataset from LVSM format to PROPE format."""
    
    print(f"Transforming dataset from {source_dir} to {output_dir}", flush=True)
    print(f"Processing split: {split}", flush=True)
    
    # Get list of sequences from the images directory
    images_dir = os.path.join(source_dir, split, "images")
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    sequences = [d for d in os.listdir(images_dir) 
                if os.path.isdir(os.path.join(images_dir, d))]
    sequences.sort()
    
    print(f"Found {len(sequences)} sequences to process", flush=True)
    
    # Process each sequence
    successful = 0
    failed = 0
    skipped = 0
    failed_sequences = []
    
    for idx, seq_name in enumerate(sequences):
        print(f"Processing sequence {idx+1}/{len(sequences)}: {seq_name}", flush=True)
        sys.stdout.flush()
        
        if is_sequence_processed(seq_name, output_dir, split):
            print(f"  -> Skipped (already processed)", flush=True)
            skipped += 1
        elif process_sequence(seq_name, source_dir, output_dir, split):
            print(f"  -> Successfully processed", flush=True)
            successful += 1
        else:
            print(f"  -> Failed to process", flush=True)
            failed += 1
            failed_sequences.append(seq_name)
    
    print(f"\nProcessing complete:", flush=True)
    print(f"  Successful: {successful}", flush=True)
    print(f"  Skipped: {skipped}", flush=True)
    print(f"  Failed: {failed}", flush=True)
    print(f"  Total: {len(sequences)}", flush=True)
    
    # Print failed sequences if any
    if failed_sequences:
        print(f"\nFailed sequences ({len(failed_sequences)}):", flush=True)
        for seq_name in failed_sequences:
            print(f"  - {seq_name}", flush=True)
    else:
        print("\nAll sequences processed successfully!", flush=True)
    
    # Copy full_list.txt if it exists
    source_full_list = os.path.join(source_dir, split, "full_list.txt")
    if os.path.exists(source_full_list):
        output_full_list = os.path.join(output_dir, split, "full_list.txt")
        os.makedirs(os.path.dirname(output_full_list), exist_ok=True)
        
        # Create symbolic link for full_list.txt
        if os.path.islink(output_full_list):
            os.unlink(output_full_list)
        elif os.path.exists(output_full_list):
            os.remove(output_full_list)
        os.symlink(os.path.abspath(source_full_list), output_full_list)
        print(f"Linked full_list.txt", flush=True)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Transform RE10K dataset from LVSM format to PROPE format.')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to the source RE10K dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory for processed dataset')
    
    args = parser.parse_args()
    
    # Transform the specified split
    transform_dataset(args.source_dir, args.output_dir, split='test')
    transform_dataset(args.source_dir, args.output_dir, split='train')

    # seq_name = "33913957b62dabc4"
    # seq_name = "1ec011c8e0b341e5"
    # process_sequence(seq_name, args.source_dir, args.output_dir, split=args.split)


if __name__ == "__main__":
    main()