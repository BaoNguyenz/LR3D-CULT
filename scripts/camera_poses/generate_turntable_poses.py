"""
Generate camera poses for turntable captures.
Assumes the object is at the center and camera rotates around it.

This bypasses COLMAP entirely and generates synthetic camera poses
based on the assumption of a turntable/rotating capture setup.
"""

import os
import json
import math
import numpy as np
from pathlib import Path
import shutil

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix for rotation around axis by theta radians.
    axis: 'x', 'y', or 'z'
    """
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def look_at(camera_pos, target, up=np.array([0, 1, 0])):
    """
    Create a look-at transformation matrix.
    """
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)
    
    # Build camera-to-world matrix
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = new_up
    c2w[:3, 2] = -forward  # Camera looks along -Z
    c2w[:3, 3] = camera_pos
    
    return c2w

def generate_turntable_poses(
    num_frames,
    radius=4.0,           # Distance from camera to object center
    height=0.0,           # Camera height relative to object center
    elevation_deg=30.0,   # Camera elevation angle in degrees
    start_angle_deg=0.0,  # Starting rotation angle
    full_rotation=True,   # If True, rotate full 360 degrees
    rotation_axis='y'     # Axis of rotation (usually 'y' for turntable)
):
    """
    Generate camera poses for a turntable capture.
    
    Args:
        num_frames: Number of frames/images
        radius: Distance from camera to object
        height: Vertical offset of camera
        elevation_deg: Elevation angle (0 = horizontal, 90 = top-down)
        start_angle_deg: Starting angle of rotation
        full_rotation: Whether to do full 360° or partial
        rotation_axis: Axis around which camera rotates
    
    Returns:
        List of 4x4 transformation matrices (camera-to-world)
    """
    poses = []
    
    elevation_rad = math.radians(elevation_deg)
    start_angle_rad = math.radians(start_angle_deg)
    
    # Calculate angle step
    if full_rotation:
        total_angle = 2 * math.pi
    else:
        total_angle = 2 * math.pi * (num_frames - 1) / num_frames
    
    angle_step = total_angle / num_frames
    
    target = np.array([0, 0, 0])  # Object at origin
    
    for i in range(num_frames):
        angle = start_angle_rad + i * angle_step
        
        # Camera position on a circle
        x = radius * math.cos(angle) * math.cos(elevation_rad)
        z = radius * math.sin(angle) * math.cos(elevation_rad)
        y = radius * math.sin(elevation_rad) + height
        
        camera_pos = np.array([x, y, z])
        
        # Create look-at matrix
        c2w = look_at(camera_pos, target)
        poses.append(c2w)
    
    return poses

def create_transforms_json(
    images_folder,
    output_path,
    camera_angle_x=0.6911,  # ~39.6 degrees, typical phone camera FOV
    radius=4.0,
    elevation_deg=30.0,
    image_size=(512, 512)
):
    """
    Create transforms.json for PixelNeRF/NeRF training.
    """
    # Get list of images
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    images = sorted([f for f in os.listdir(images_folder) if f.endswith(image_extensions)])
    
    if not images:
        print(f"No images found in {images_folder}")
        return None
    
    num_frames = len(images)
    print(f"Found {num_frames} images")
    
    # Generate poses
    poses = generate_turntable_poses(
        num_frames=num_frames,
        radius=radius,
        elevation_deg=elevation_deg
    )
    
    # Calculate focal length from FOV
    w, h = image_size
    focal_length = w / (2 * math.tan(camera_angle_x / 2))
    
    # Build transforms.json
    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": []
    }
    
    for i, (img_name, pose) in enumerate(zip(images, poses)):
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
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"Saved transforms.json with {num_frames} frames to {output_path}")
    return transforms

def process_failed_objects(all_data_path, min_frames_threshold=50):
    """
    Process objects that have fewer frames than expected in their transforms.json
    """
    failed_objects = []
    
    for obj_name in sorted(os.listdir(all_data_path)):
        obj_path = os.path.join(all_data_path, obj_name)
        if not os.path.isdir(obj_path):
            continue
        
        transforms_path = os.path.join(obj_path, "transforms.json")
        images_path = os.path.join(obj_path, "images")
        
        if not os.path.exists(transforms_path) or not os.path.exists(images_path):
            continue
        
        # Count images
        images = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(images)
        
        # Count frames in transforms.json
        with open(transforms_path, 'r') as f:
            data = json.load(f)
        num_frames = len(data.get('frames', []))
        
        # Check if COLMAP failed to register enough frames
        if num_frames < min_frames_threshold and num_images > num_frames:
            failed_objects.append({
                'name': obj_name,
                'path': obj_path,
                'images': num_images,
                'registered_frames': num_frames
            })
    
    return failed_objects

def regenerate_transforms_for_object(obj_path, backup=True):
    """
    Regenerate transforms.json for a single object using turntable poses.
    """
    images_path = os.path.join(obj_path, "images")
    transforms_path = os.path.join(obj_path, "transforms.json")
    
    if not os.path.exists(images_path):
        print(f"No images folder found in {obj_path}")
        return False
    
    # Backup original transforms.json
    if backup and os.path.exists(transforms_path):
        backup_path = os.path.join(obj_path, "transforms_colmap_backup.json")
        if not os.path.exists(backup_path):
            shutil.copy(transforms_path, backup_path)
            print(f"Backed up original transforms.json to {backup_path}")
    
    # Create new transforms.json
    create_transforms_json(
        images_folder=images_path,
        output_path=transforms_path,
        camera_angle_x=0.6911,  # Standard FOV
        radius=4.0,
        elevation_deg=30.0,
        image_size=(512, 512)
    )
    
    return True

def main():
    all_data_path = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data"
    
    print("=" * 80)
    print("TURNTABLE POSE GENERATOR")
    print("=" * 80)
    
    # Find failed objects
    print("\nFinding objects with incomplete COLMAP reconstruction...")
    failed = process_failed_objects(all_data_path, min_frames_threshold=50)
    
    print(f"\nFound {len(failed)} objects that need regeneration:")
    for obj in failed:
        print(f"  {obj['name']}: {obj['registered_frames']}/{obj['images']} frames registered")
    
    # Ask for confirmation
    print("\n" + "=" * 80)
    print("OPTIONS:")
    print("1. Regenerate transforms.json for ALL failed objects")
    print("2. Regenerate for a specific object")
    print("3. Just list failed objects (no changes)")
    print("=" * 80)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\nRegenerating transforms.json for all failed objects...")
        for obj in failed:
            print(f"\nProcessing {obj['name']}...")
            regenerate_transforms_for_object(obj['path'])
        print("\nDone!")
        
    elif choice == '2':
        obj_name = input("Enter object name: ").strip()
        obj_path = os.path.join(all_data_path, obj_name)
        if os.path.exists(obj_path):
            regenerate_transforms_for_object(obj_path)
        else:
            print(f"Object {obj_name} not found!")
            
    else:
        print("\nNo changes made.")

if __name__ == "__main__":
    main()

