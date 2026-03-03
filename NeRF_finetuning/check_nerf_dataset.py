"""
Script to check NeRF dataset quality after COLMAP processing.
Checks each object folder for images and transforms.json
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def check_dataset(all_data_path):
    """Check all object folders in All_data directory"""
    
    results = []
    
    # Get all object folders
    object_folders = sorted([f for f in os.listdir(all_data_path) 
                            if os.path.isdir(os.path.join(all_data_path, f))])
    
    print(f"Found {len(object_folders)} object folders\n")
    print("=" * 80)
    
    for obj_name in object_folders:
        obj_path = os.path.join(all_data_path, obj_name)
        images_path = os.path.join(obj_path, "images")
        transforms_path = os.path.join(obj_path, "transforms.json")
        
        # Check images folder
        image_count = 0
        has_images_folder = os.path.exists(images_path)
        if has_images_folder:
            images = [f for f in os.listdir(images_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_count = len(images)
        
        # Check transforms.json
        has_transforms = os.path.exists(transforms_path)
        transforms_frame_count = 0
        if has_transforms:
            try:
                with open(transforms_path, 'r') as f:
                    data = json.load(f)
                    if 'frames' in data:
                        transforms_frame_count = len(data['frames'])
            except Exception as e:
                transforms_frame_count = -1  # Error reading
        
        results.append({
            'name': obj_name,
            'has_images_folder': has_images_folder,
            'image_count': image_count,
            'has_transforms': has_transforms,
            'transforms_frames': transforms_frame_count
        })
    
    return results

def print_results(results):
    """Print results in a formatted way"""
    
    # Group by frame count ranges
    low_frames = []      # <= 10 frames
    medium_frames = []   # 11-50 frames
    good_frames = []     # 51-80 frames
    excellent_frames = [] # > 80 frames
    issues = []          # Missing files or mismatch
    
    for r in results:
        if not r['has_images_folder'] or not r['has_transforms']:
            issues.append(r)
        elif r['transforms_frames'] <= 10:
            low_frames.append(r)
        elif r['transforms_frames'] <= 50:
            medium_frames.append(r)
        elif r['transforms_frames'] <= 80:
            good_frames.append(r)
        else:
            excellent_frames.append(r)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total objects: {len(results)}")
    print(f"  - Excellent (>80 frames): {len(excellent_frames)}")
    print(f"  - Good (51-80 frames): {len(good_frames)}")
    print(f"  - Medium (11-50 frames): {len(medium_frames)}")
    print(f"  - Low (<=10 frames): {len(low_frames)}")
    print(f"  - Issues (missing files): {len(issues)}")
    
    # Print problematic ones (low frames)
    if low_frames:
        print("\n" + "=" * 80)
        print("⚠️  LOW FRAME COUNT (<=10 frames) - May have COLMAP issues:")
        print("=" * 80)
        for r in sorted(low_frames, key=lambda x: x['transforms_frames']):
            print(f"  {r['name']}: {r['transforms_frames']} frames in transforms.json, {r['image_count']} images")
    
    # Print issues
    if issues:
        print("\n" + "=" * 80)
        print("❌ ISSUES (missing images folder or transforms.json):")
        print("=" * 80)
        for r in issues:
            status = []
            if not r['has_images_folder']:
                status.append("NO images folder")
            if not r['has_transforms']:
                status.append("NO transforms.json")
            print(f"  {r['name']}: {', '.join(status)}")
    
    # Print medium frames
    if medium_frames:
        print("\n" + "=" * 80)
        print("⚡ MEDIUM FRAME COUNT (11-50 frames):")
        print("=" * 80)
        for r in sorted(medium_frames, key=lambda x: x['transforms_frames']):
            print(f"  {r['name']}: {r['transforms_frames']} frames")
    
    # Print detailed list
    print("\n" + "=" * 80)
    print("DETAILED LIST (all objects):")
    print("=" * 80)
    print(f"{'Object Name':<25} {'Images':<10} {'Transforms':<12} {'Status'}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['transforms_frames']):
        status = "✅"
        if not r['has_images_folder'] or not r['has_transforms']:
            status = "❌ MISSING"
        elif r['transforms_frames'] <= 2:
            status = "🔴 VERY LOW"
        elif r['transforms_frames'] <= 10:
            status = "🟡 LOW"
        elif r['transforms_frames'] <= 50:
            status = "🟠 MEDIUM"
        elif r['transforms_frames'] <= 80:
            status = "🟢 GOOD"
        else:
            status = "✅ EXCELLENT"
            
        print(f"{r['name']:<25} {r['image_count']:<10} {r['transforms_frames']:<12} {status}")

def main():
    all_data_path = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data"
    
    print("Checking NeRF dataset quality...")
    print(f"Path: {all_data_path}\n")
    
    results = check_dataset(all_data_path)
    print_results(results)
    
    # Save results to file
    output_file = os.path.join(os.path.dirname(all_data_path), "dataset_check_report.txt")
    
    # Redirect print to file as well
    import sys
    from io import StringIO
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    print_results(results)
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"NeRF Dataset Check Report\n")
        f.write(f"Path: {all_data_path}\n")
        f.write(output)
    
    print(f"\n📄 Report saved to: {output_file}")

if __name__ == "__main__":
    main()
