"""
Analyze the relationship between camera_angle_x and frame count
"""

import os
import json
import math
from pathlib import Path

def analyze_camera_angles(all_data_path):
    results = []
    
    for obj_name in sorted(os.listdir(all_data_path)):
        obj_path = os.path.join(all_data_path, obj_name)
        if not os.path.isdir(obj_path):
            continue
            
        transforms_path = os.path.join(obj_path, "transforms.json")
        if not os.path.exists(transforms_path):
            continue
            
        try:
            with open(transforms_path, 'r') as f:
                data = json.load(f)
                
            camera_angle_x = data.get('camera_angle_x', 0)
            frames_count = len(data.get('frames', []))
            
            # Convert to degrees for easier understanding
            fov_degrees = math.degrees(camera_angle_x)
            
            results.append({
                'name': obj_name,
                'camera_angle_x': camera_angle_x,
                'fov_degrees': fov_degrees,
                'frames': frames_count
            })
        except Exception as e:
            print(f"Error reading {obj_name}: {e}")
    
    return results

def print_analysis(results):
    # Standard FOV range (35-45 degrees is typical for most cameras)
    NORMAL_FOV_MIN = 30  # degrees
    NORMAL_FOV_MAX = 50  # degrees
    
    abnormal_fov = []
    low_frames = []
    
    print("=" * 100)
    print("CAMERA ANGLE ANALYSIS")
    print("=" * 100)
    
    # Find objects with abnormal FOV
    for r in results:
        if r['fov_degrees'] < NORMAL_FOV_MIN or r['fov_degrees'] > NORMAL_FOV_MAX:
            abnormal_fov.append(r)
        if r['frames'] <= 10:
            low_frames.append(r)
    
    # Check correlation
    print("\n📊 SUMMARY:")
    print(f"Total objects: {len(results)}")
    print(f"Objects with abnormal FOV (<{NORMAL_FOV_MIN}° or >{NORMAL_FOV_MAX}°): {len(abnormal_fov)}")
    print(f"Objects with low frames (<=10): {len(low_frames)}")
    
    # Print abnormal FOV objects
    print("\n" + "=" * 100)
    print("⚠️  ABNORMAL FOV (outside 30-50 degrees range):")
    print("=" * 100)
    print(f"{'Object':<25} {'camera_angle_x':<18} {'FOV (degrees)':<15} {'Frames':<10} {'Status'}")
    print("-" * 100)
    
    for r in sorted(abnormal_fov, key=lambda x: x['fov_degrees']):
        status = "🔴 TOO LOW" if r['fov_degrees'] < NORMAL_FOV_MIN else "🟠 TOO HIGH"
        frame_status = " ⚠️ LOW FRAMES" if r['frames'] <= 10 else ""
        print(f"{r['name']:<25} {r['camera_angle_x']:<18.6f} {r['fov_degrees']:<15.2f} {r['frames']:<10} {status}{frame_status}")
    
    # Print correlation analysis
    print("\n" + "=" * 100)
    print("🔗 CORRELATION: Objects with LOW FRAMES vs ABNORMAL FOV")
    print("=" * 100)
    
    low_frames_abnormal = [r for r in low_frames if r in abnormal_fov]
    low_frames_normal = [r for r in low_frames if r not in abnormal_fov]
    
    print(f"\nObjects with LOW frames AND abnormal FOV: {len(low_frames_abnormal)}")
    print(f"Objects with LOW frames but normal FOV: {len(low_frames_normal)}")
    
    if low_frames_abnormal:
        print("\n📍 Low frames + Abnormal FOV:")
        for r in low_frames_abnormal:
            print(f"   {r['name']}: {r['frames']} frames, FOV={r['fov_degrees']:.2f}°")
    
    if low_frames_normal:
        print("\n📍 Low frames + Normal FOV:")
        for r in low_frames_normal:
            print(f"   {r['name']}: {r['frames']} frames, FOV={r['fov_degrees']:.2f}°")
    
    # Statistics by frame count
    print("\n" + "=" * 100)
    print("📈 FOV STATISTICS BY FRAME COUNT:")
    print("=" * 100)
    
    frame_ranges = [
        (1, 2, "1-2 frames"),
        (3, 10, "3-10 frames"),
        (11, 50, "11-50 frames"),
        (51, 90, "51-90 frames"),
        (91, 180, "91-180 frames"),
    ]
    
    for min_f, max_f, label in frame_ranges:
        subset = [r for r in results if min_f <= r['frames'] <= max_f]
        if subset:
            fovs = [r['fov_degrees'] for r in subset]
            avg_fov = sum(fovs) / len(fovs)
            min_fov = min(fovs)
            max_fov = max(fovs)
            print(f"\n{label} ({len(subset)} objects):")
            print(f"   FOV range: {min_fov:.2f}° - {max_fov:.2f}°")
            print(f"   Average FOV: {avg_fov:.2f}°")
            
            # Count abnormal in this range
            abnormal_in_range = sum(1 for f in fovs if f < NORMAL_FOV_MIN or f > NORMAL_FOV_MAX)
            print(f"   Abnormal FOV count: {abnormal_in_range}/{len(subset)} ({100*abnormal_in_range/len(subset):.1f}%)")

def main():
    all_data_path = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data"
    
    print("Analyzing camera angles in transforms.json files...")
    results = analyze_camera_angles(all_data_path)
    print_analysis(results)

if __name__ == "__main__":
    main()

