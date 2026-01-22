"""
Script to check for missing frames in the RealEstate10K dataset.

This script compares the expected number of frames (from metafiles) with the actual 
number of image files in the processed dataset directories to identify sequences 
with missing frames. It distinguishes between completely missing sequences and 
incomplete sequences (exist but missing some frames).

Usage:
    To run the script and save all output to a log file:
    
    python scripts/check_re10k_data.py > missing_frames_check_lvsmdata.log 2>&1
    
    Or to see output in terminal AND save to log file:
    
    python scripts/check_re10k_data.py | tee missing_frames_check.log

    The script prints a comprehensive final report showing:
    - Number of completely missing sequences
    - Number of incomplete sequences (missing some frames)
    - Detailed breakdown of each problematic sequence
"""

import json
import numpy as np
import os, sys


def check_re10k_frames():
    """
    Loop over the JSON file to check for missing frames in RE10K dataset.
    For each entry seq_name, read the corresponding metafiles at /grogu/user/yuwu3/RealEstate10K/metafiles/test,
    then find how many frames there should be. Given a dataset path, count how many images exist in the 
    corresponding sequence. Log the sequence if there are fewer frames in data than in metafile.
    """
    
    # Paths
    json_file = "/home/yuwu3/prope/assets/evaluation_index_re10k_video.json"
    metafiles_dir = "/grogu/user/yuwu3/RealEstate10K/metafiles/test"
    dataset_path = "/grogu/user/yuwu3/re10k_processed/test"
    
    # Load the JSON file
    print("Loading evaluation index JSON file...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} sequences in JSON file")
    
    missing_sequences = []  # Completely missing (no images directory)
    incomplete_sequences = []  # Exists but missing some frames
    total_checked = 0
    null_entries = 0
    errors = []
    
    for seq_name, seq_data in data.items():
        # Skip null entries
        if seq_data is None:
            null_entries += 1
            continue
            
        total_checked += 1
        
        # Paths for this sequence
        metafile_path = os.path.join(metafiles_dir, f"{seq_name}.txt")
        sequence_dir = os.path.join(dataset_path, seq_name)
        images_dir = os.path.join(sequence_dir, "images")
        
        # Check if metafile exists
        if not os.path.exists(metafile_path):
            errors.append(f"Metafile not found for {seq_name}")
            continue
            
        # Count lines in metafile (each line = 1 frame)
        try:
            with open(metafile_path, 'r') as f:
                expected_frames = len(f.readlines()) - 1 # the first line is seq name
        except Exception as e:
            errors.append(f"Could not read metafile for {seq_name}: {e}")
            continue
            
        # Check if sequence directory exists
        if not os.path.exists(images_dir):
            missing_sequences.append({
                'seq_name': seq_name,
                'expected_frames': expected_frames
            })
            continue
            
        # Count actual images
        try:
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            actual_frames = len(image_files)
        except Exception as e:
            errors.append(f"Could not list images for {seq_name}: {e}")
            continue
        
        # Check if there are missing frames
        if actual_frames < expected_frames:
            missing_frames = expected_frames - actual_frames
            incomplete_sequences.append({
                'seq_name': seq_name,
                'expected_frames': expected_frames,
                'actual_frames': actual_frames,
                'missing_frames': missing_frames
            })
        
        # Progress indicator
        if total_checked % 1000 == 0:
            print(f"Checked {total_checked} sequences...")
    
    # Final Report
    print(f"\n" + "=" * 70)
    print("FINAL REPORT: RE10K Dataset Missing Frames Analysis")
    print("=" * 70)
    
    print(f"Total sequences in JSON: {len(data)}")
    print(f"Null entries skipped: {null_entries}")
    print(f"Sequences checked: {total_checked}")
    print(f"Sequences completely missing: {len(missing_sequences)}")
    print(f"Sequences incomplete (missing some frames): {len(incomplete_sequences)}")
    print(f"Sequences complete: {total_checked - len(missing_sequences) - len(incomplete_sequences)}")
    
    if errors:
        print(f"Errors encountered: {len(errors)}")
    
    # Missing sequences (completely absent)
    if missing_sequences:
        print(f"\n--- COMPLETELY MISSING SEQUENCES ({len(missing_sequences)}) ---")
        total_missing_frames = 0
        for seq_info in missing_sequences:
            print(f"{seq_info['seq_name']}: Expected {seq_info['expected_frames']} frames")
            total_missing_frames += seq_info['expected_frames']
        print(f"Total frames missing from absent sequences: {total_missing_frames}")
    
    # Incomplete sequences (exist but missing frames)
    if incomplete_sequences:
        print(f"\n--- INCOMPLETE SEQUENCES ({len(incomplete_sequences)}) ---")
        total_incomplete_frames = 0
        for seq_info in incomplete_sequences:
            print(f"{seq_info['seq_name']}: Expected {seq_info['expected_frames']}, Got {seq_info['actual_frames']}, Missing {seq_info['missing_frames']}")
            total_incomplete_frames += seq_info['missing_frames']
        print(f"Total frames missing from incomplete sequences: {total_incomplete_frames}")
    
    # Summary
    total_missing = (sum(s['expected_frames'] for s in missing_sequences) + 
                    sum(s['missing_frames'] for s in incomplete_sequences))
    print(f"\n--- SUMMARY ---")
    print(f"Total missing frames across all sequences: {total_missing}")
    print(f"Sequences with issues: {len(missing_sequences) + len(incomplete_sequences)} out of {total_checked}")
    
    if errors:
        print(f"\n--- ERRORS ---")
        for error in errors[:10]:  # Show first 10 errors
            print(f"ERROR: {error}")
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors")
    
    print("=" * 70)


if __name__ == "__main__":
    check_re10k_frames()

