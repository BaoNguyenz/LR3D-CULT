"""
Generate transforms.json from metadata.json for all objects.
This bypasses COLMAP and uses the exact render parameters.

The metadata.json contains spherical camera coordinates:
- azimuth_deg: horizontal rotation angle (0-360)
- elevation_deg: vertical angle from horizontal plane
- radius: distance from object center
- position_world: camera position in world coordinates
"""

import os
import json
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

def spherical_to_cartesian(azimuth_deg, elevation_deg, radius):
    """Convert spherical coordinates to cartesian (x, y, z)"""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    
    x = radius * math.cos(elevation) * math.sin(azimuth)
    y = radius * math.sin(elevation)
    z = radius * math.cos(elevation) * math.cos(azimuth)
    
    return np.array([x, y, z])

def look_at_matrix(camera_pos, target=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
    """
    Create a look-at camera-to-world transformation matrix.
    Camera looks from camera_pos towards target.
    """
    # Forward vector (camera looks along negative z in camera space)
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # Right vector
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        # Handle case where forward is parallel to up
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recalculate up to ensure orthogonality
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Build rotation matrix (camera-to-world)
    # NeRF convention: camera looks along -Z, Y is up, X is right
    rotation = np.eye(4)
    rotation[:3, 0] = right
    rotation[:3, 1] = up
    rotation[:3, 2] = -forward  # Camera looks along -Z
    rotation[:3, 3] = camera_pos
    
    return rotation

def create_transform_matrix_from_metadata(frame_data):
    """
    Create 4x4 transform matrix from metadata frame data.
    Uses the position_world directly from the metadata.
    """
    pos = frame_data['position_world']
    spherical = frame_data['spherical']
    
    # Camera position
    camera_pos = np.array([pos['x'], pos['y'], pos['z']])
    
    # Target is the object center (origin)
    target = np.array([0, 0, 0])
    
    # Y-up convention (as specified in metadata)
    up = np.array([0, 1, 0])
    
    # Create look-at matrix
    transform = look_at_matrix(camera_pos, target, up)
    
    return transform.tolist()

def process_metadata(metadata_path, images_path, obj_name, output_path, image_size=512, focal_length=None):
    """
    Process a single metadata.json file and create transforms.json
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Calculate camera_angle_x from focal length
    # FOV = 2 * atan(sensor_width / (2 * focal_length))
    # For a square image, sensor_width = image_size
    if focal_length is None:
        # Default FOV ~39.6 degrees (0.6911 radians) - typical for rendering
        camera_angle_x = 0.6911
    else:
        camera_angle_x = 2 * math.atan(image_size / (2 * focal_length))
    
    frames = []
    frame_list = metadata.get('frames', [])
    
    # Group frames by camera and frame number
    cam1_frames = []
    cam2_frames = []
    
    for frame_data in frame_list:
        camera_name = frame_data.get('camera', '')
        if 'CAM1' in camera_name:
            cam1_frames.append(frame_data)
        elif 'CAM2' in camera_name:
            cam2_frames.append(frame_data)
    
    # Sort by frame number
    cam1_frames.sort(key=lambda x: x['frame'])
    cam2_frames.sort(key=lambda x: x['frame'])
    
    # Process CAM1 frames (0001-0090)
    for i, frame_data in enumerate(cam1_frames):
        frame_idx = i + 1  # 1-indexed
        image_filename = f"{obj_name}_{frame_idx:04d}.png"
        image_path = os.path.join(images_path, image_filename)
        
        if os.path.exists(image_path):
            transform_matrix = create_transform_matrix_from_metadata(frame_data)
            frames.append({
                "file_path": f"./images/{image_filename}",
                "transform_matrix": transform_matrix,
                "w": image_size,
                "h": image_size,
                "fl_x": image_size / (2 * math.tan(camera_angle_x / 2)),
                "fl_y": image_size / (2 * math.tan(camera_angle_x / 2)),
                "cx": image_size / 2.0,
                "cy": image_size / 2.0,
                "camera_model": "PINHOLE"
            })
    
    # Process CAM2 frames (0091-0180)
    for i, frame_data in enumerate(cam2_frames):
        frame_idx = 90 + i + 1  # 91-indexed for CAM2
        image_filename = f"{obj_name}_{frame_idx:04d}.png"
        image_path = os.path.join(images_path, image_filename)
        
        if os.path.exists(image_path):
            transform_matrix = create_transform_matrix_from_metadata(frame_data)
            frames.append({
                "file_path": f"./images/{image_filename}",
                "transform_matrix": transform_matrix,
                "w": image_size,
                "h": image_size,
                "fl_x": image_size / (2 * math.tan(camera_angle_x / 2)),
                "fl_y": image_size / (2 * math.tan(camera_angle_x / 2)),
                "cx": image_size / 2.0,
                "cy": image_size / 2.0,
                "camera_model": "PINHOLE"
            })
    
    # Create transforms.json
    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }
    
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    return len(frames)

def process_all_objects(all_data_path, backup_old=True):
    """
    Process all objects in All_data folder
    """
    object_folders = sorted([f for f in os.listdir(all_data_path) 
                            if os.path.isdir(os.path.join(all_data_path, f))])
    
    results = {
        'success': [],
        'no_metadata': [],
        'error': []
    }
    
    print(f"Found {len(object_folders)} object folders")
    print("=" * 80)
    
    for obj_name in tqdm(object_folders, desc="Processing objects"):
        obj_path = os.path.join(all_data_path, obj_name)
        metadata_path = os.path.join(obj_path, "metadata.json")
        images_path = os.path.join(obj_path, "images")
        transforms_path = os.path.join(obj_path, "transforms.json")
        
        # Check if metadata.json exists
        if not os.path.exists(metadata_path):
            results['no_metadata'].append(obj_name)
            continue
        
        # Check if images folder exists
        if not os.path.exists(images_path):
            results['error'].append((obj_name, "No images folder"))
            continue
        
        try:
            # Backup old transforms.json
            if backup_old and os.path.exists(transforms_path):
                backup_path = os.path.join(obj_path, "transforms_colmap_backup.json")
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy(transforms_path, backup_path)
            
            # Generate new transforms.json
            frame_count = process_metadata(
                metadata_path, 
                images_path, 
                obj_name,
                transforms_path
            )
            results['success'].append((obj_name, frame_count))
            
        except Exception as e:
            results['error'].append((obj_name, str(e)))
    
    return results

def print_results(results):
    """Print processing results"""
    print("\n" + "=" * 80)
    print("PROCESSING RESULTS")
    print("=" * 80)
    
    print(f"\n✅ Successfully processed: {len(results['success'])} objects")
    print(f"⚠️  No metadata.json: {len(results['no_metadata'])} objects")
    print(f"❌ Errors: {len(results['error'])} objects")
    
    if results['success']:
        print("\n📊 Frame counts for successful objects:")
        frame_counts = [fc for _, fc in results['success']]
        print(f"   Min: {min(frame_counts)}, Max: {max(frame_counts)}, Avg: {sum(frame_counts)/len(frame_counts):.1f}")
    
    if results['no_metadata']:
        print("\n⚠️  Objects without metadata.json:")
        for obj in results['no_metadata'][:10]:
            print(f"   - {obj}")
        if len(results['no_metadata']) > 10:
            print(f"   ... and {len(results['no_metadata']) - 10} more")
    
    if results['error']:
        print("\n❌ Objects with errors:")
        for obj, err in results['error'][:10]:
            print(f"   - {obj}: {err}")
        if len(results['error']) > 10:
            print(f"   ... and {len(results['error']) - 10} more")

def main():
    all_data_path = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data"
    
    print("=" * 80)
    print("GENERATING TRANSFORMS.JSON FROM METADATA")
    print("(Bypassing COLMAP - using exact render parameters)")
    print("=" * 80)
    print(f"\nPath: {all_data_path}")
    
    results = process_all_objects(all_data_path, backup_old=True)
    print_results(results)

if __name__ == "__main__":
    main()

