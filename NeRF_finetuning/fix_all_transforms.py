"""
Automatically fix all transforms.json files that have incomplete COLMAP reconstruction.
Uses turntable pose generation to create poses for ALL images.

This script will:
1. Find all objects with fewer registered frames than total images
2. Generate synthetic turntable poses for all images
3. Create new transforms.json files (with backup of originals)
"""

import os
import json
import math
import numpy as np
from pathlib import Path
import shutil

def look_at(camera_pos, target, up=np.array([0, 1, 0])):
    """Create a look-at transformation matrix."""
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)
    
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = new_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = camera_pos
    
    return c2w

def generate_turntable_poses(num_frames, radius=4.0, elevation_deg=30.0):
    """Generate camera poses for turntable capture."""
    poses = []
    elevation_rad = math.radians(elevation_deg)
    angle_step = 2 * math.pi / num_frames
    target = np.array([0, 0, 0])
    
    for i in range(num_frames):
        angle = i * angle_step
        x = radius * math.cos(angle) * math.cos(elevation_rad)
        z = radius * math.sin(angle) * math.cos(elevation_rad)
        y = radius * math.sin(elevation_rad)
        
        camera_pos = np.array([x, y, z])
        c2w = look_at(camera_pos, target)
        poses.append(c2w)
    
    return poses

def fix_transforms_json(obj_path, camera_angle_x=0.6911, image_size=(512, 512)):
    """Fix transforms.json for an object using turntable poses."""
    images_path = os.path.join(obj_path, "images")
    transforms_path = os.path.join(obj_path, "transforms.json")
    
    if not os.path.exists(images_path):
        return None, "No images folder"
    
    # Get images
    images = sorted([f for f in os.listdir(images_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not images:
        return None, "No images found"
    
    num_frames = len(images)
    
    # Backup original
    if os.path.exists(transforms_path):
        backup_path = os.path.join(obj_path, "transforms_colmap_backup.json")
        if not os.path.exists(backup_path):
            shutil.copy(transforms_path, backup_path)
    
    # Generate poses
    poses = generate_turntable_poses(num_frames)
    
    # Calculate focal length
    w, h = image_size
    focal_length = w / (2 * math.tan(camera_angle_x / 2))
    
    # Build transforms
    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": []
    }
    
    for img_name, pose in zip(images, poses):
        frame = {
            "file_path": f"./images/{img_name}",
            "transform_matrix": pose.tolist(),
            "w": w,
            "h": h,
            "fl_x": focal_length,
            "fl_y": focal_length,
            "cx": w / 2,
            "cy": h / 2,
            "camera_model": "PINHOLE"
        }
        transforms["frames"].append(frame)
    
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    return num_frames, "OK"

def main():
    all_data_path = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data"
    
    print("=" * 80)
    print("FIXING ALL TRANSFORMS.JSON FILES")
    print("=" * 80)
    
    results = {"fixed": [], "skipped": [], "errors": []}
    
    for obj_name in sorted(os.listdir(all_data_path)):
        obj_path = os.path.join(all_data_path, obj_name)
        if not os.path.isdir(obj_path):
            continue
        
        transforms_path = os.path.join(obj_path, "transforms.json")
        images_path = os.path.join(obj_path, "images")
        
        if not os.path.exists(images_path):
            results["errors"].append((obj_name, "No images folder"))
            continue
        
        # Count current frames
        images = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(images)
        
        current_frames = 0
        if os.path.exists(transforms_path):
            try:
                with open(transforms_path, 'r') as f:
                    data = json.load(f)
                current_frames = len(data.get('frames', []))
            except:
                pass
        
        # Fix if incomplete (less than 90% of images registered)
        if current_frames < num_images * 0.9:
            new_frames, status = fix_transforms_json(obj_path)
            if new_frames:
                results["fixed"].append((obj_name, current_frames, new_frames))
                print(f"✅ Fixed {obj_name}: {current_frames} → {new_frames} frames")
            else:
                results["errors"].append((obj_name, status))
                print(f"❌ Error {obj_name}: {status}")
        else:
            results["skipped"].append((obj_name, current_frames))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Fixed: {len(results['fixed'])} objects")
    print(f"Skipped (already complete): {len(results['skipped'])} objects")
    print(f"Errors: {len(results['errors'])} objects")
    
    if results["fixed"]:
        print("\n📝 Fixed objects:")
        for name, old, new in results["fixed"]:
            print(f"   {name}: {old} → {new} frames")

if __name__ == "__main__":
    main()

