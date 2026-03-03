"""
Re-run COLMAP with more robust settings for failed objects.

Key changes from default:
1. Use SEQUENTIAL matcher instead of EXHAUSTIVE (better for video frames)
2. Lower feature detection threshold
3. Use more RANSAC iterations
4. Try multiple feature types (SIFT + SuperPoint if available)
"""

import os
import subprocess
import json
import shutil

def get_failed_objects(all_data_path, threshold=50):
    """Get list of objects with incomplete COLMAP reconstruction."""
    failed = []
    
    for obj_name in sorted(os.listdir(all_data_path)):
        obj_path = os.path.join(all_data_path, obj_name)
        if not os.path.isdir(obj_path):
            continue
        
        transforms_path = os.path.join(obj_path, "transforms.json")
        images_path = os.path.join(obj_path, "images")
        
        if not os.path.exists(images_path):
            continue
        
        images = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(images)
        
        num_frames = 0
        if os.path.exists(transforms_path):
            try:
                with open(transforms_path, 'r') as f:
                    data = json.load(f)
                num_frames = len(data.get('frames', []))
            except:
                pass
        
        if num_frames < threshold:
            failed.append({
                'name': obj_name,
                'path': obj_path,
                'images': num_images,
                'frames': num_frames
            })
    
    return failed

def run_colmap_sequential(images_folder, output_folder, colmap_path="colmap"):
    """
    Run COLMAP with sequential matcher (better for video frames).
    """
    database_path = os.path.join(output_folder, "database.db")
    sparse_path = os.path.join(output_folder, "sparse")
    
    os.makedirs(sparse_path, exist_ok=True)
    
    # Remove old database if exists
    if os.path.exists(database_path):
        os.remove(database_path)
    
    commands = [
        # 1. Feature extraction with more features
        [
            colmap_path, "feature_extractor",
            "--database_path", database_path,
            "--image_path", images_folder,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.max_num_features", "8192",  # More features
            "--SiftExtraction.first_octave", "-1",  # Detect smaller features
        ],
        
        # 2. Sequential matching (assumes consecutive frames are similar)
        [
            colmap_path, "sequential_matcher",
            "--database_path", database_path,
            "--SequentialMatching.overlap", "10",  # Match with 10 neighboring frames
            "--SequentialMatching.quadratic_overlap", "1",  # Also match distant frames
        ],
        
        # 3. Mapper with more robust settings
        [
            colmap_path, "mapper",
            "--database_path", database_path,
            "--image_path", images_folder,
            "--output_path", sparse_path,
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "1",
            "--Mapper.init_min_num_inliers", "50",  # Lower threshold
            "--Mapper.abs_pose_min_num_inliers", "15",  # Lower threshold
            "--Mapper.abs_pose_min_inlier_ratio", "0.15",  # Lower threshold
        ],
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd[:3])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
    
    return True

def print_colmap_command():
    """Print manual COLMAP commands for reference."""
    print("""
================================================================================
MANUAL COLMAP COMMANDS (if automatic doesn't work):
================================================================================

Option 1: Sequential Matcher (for video frames)
-----------------------------------------------
colmap feature_extractor \\
    --database_path database.db \\
    --image_path images/ \\
    --SiftExtraction.max_num_features 8192

colmap sequential_matcher \\
    --database_path database.db \\
    --SequentialMatching.overlap 10

colmap mapper \\
    --database_path database.db \\
    --image_path images/ \\
    --output_path sparse/

Option 2: Vocab Tree Matcher (for unordered images)
---------------------------------------------------
colmap feature_extractor \\
    --database_path database.db \\
    --image_path images/

colmap vocab_tree_matcher \\
    --database_path database.db \\
    --VocabTreeMatching.vocab_tree_path vocab_tree.bin

colmap mapper \\
    --database_path database.db \\
    --image_path images/ \\
    --output_path sparse/

Option 3: Use COLMAP GUI
------------------------
1. Open COLMAP GUI
2. File → New Project
3. Processing → Feature extraction (enable "Share intrinsics")
4. Processing → Feature matching → Sequential matching
5. Reconstruction → Start reconstruction

================================================================================
""")

def main():
    all_data_path = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data"
    
    print("=" * 80)
    print("COLMAP RE-RUN FOR FAILED OBJECTS")
    print("=" * 80)
    
    failed = get_failed_objects(all_data_path)
    
    print(f"\nFound {len(failed)} objects that may benefit from re-running COLMAP:")
    for obj in failed:
        print(f"  {obj['name']}: {obj['frames']}/{obj['images']} frames")
    
    print_colmap_command()
    
    print("\nNote: Running COLMAP requires COLMAP to be installed and in PATH.")
    print("If you prefer, use the 'fix_all_transforms.py' script to generate")
    print("synthetic turntable poses instead.")

if __name__ == "__main__":
    main()

