"""
Chuẩn hóa dataset gốm cho PixelNeRF training.

Workflow:
1. Quét tất cả object từ SRC_ROOT (không cần train/val/test sẵn)
2. Lọc objects có >= N_FRAMES frame hợp lệ
3. Shuffle và chia theo tỉ lệ 70:15:15 (train:val:test)
4. Chọn N_FRAMES frame chia đều quanh 360°
5. Resize ảnh về RESIZE_TO
6. Tạo transforms.json với format chuẩn PixelNeRF

Output format:
{
  "camera_angle_x": 0.6699...,
  "frames": [
    {
      "file_path": "./0001.png",
      "w": 128,
      "h": 128,
      "transform_matrix": [[...], [...], [...], [...]]
    },
    ...
  ]
}
"""

import os
import json
import shutil
import random
from pathlib import Path
import math

from PIL import Image
import numpy as np

# ================== CONFIG ==================
# Source: folder chứa các object (mỗi object có 180 ảnh + transforms.json)
SRC_ROOT = Path(r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\DATA_tam_nay_thi_het_cuu")

# Destination: folder output với cấu trúc train/val/test
DST_ROOT = Path(r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\Dataset_test")

# Số frame mỗi object (chọn đều từ 180 frames)
N_FRAMES = 90

# Kích thước ảnh output
RESIZE_TO = (128, 128)

# Các định dạng ảnh hợp lệ
IMG_EXTS = {".png", ".jpg", ".jpeg"}

# Random seed để reproducible
RANDOM_SEED = 42

# Tỉ lệ split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.2
TEST_RATIO = 0.1
# ============================================


def normalize_file_path(rel: str) -> str:
    """Chuẩn hóa file_path trong transforms.json -> chỉ còn tên file."""
    rel = rel.replace("\\", "/")
    rel = rel.lstrip("./")
    if rel.startswith("images/"):
        rel = rel[len("images/"):]
    return os.path.basename(rel)


def list_valid_frames(obj_dir: Path, tf_data: dict):
    """
    Trả về danh sách TẤT CẢ frame hợp lệ:
      - Ảnh tồn tại
      - Có đuôi hợp lệ
    Format: [(frame_dict, src_img_path), ...]
    """
    images_dir = obj_dir / "images"
    search_dir = images_dir if images_dir.exists() and images_dir.is_dir() else obj_dir

    # Map tên file -> path
    img_map = {}
    for p in search_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            img_map[p.name] = p

    frames = tf_data.get("frames", [])
    all_valid = []

    for idx, fr in enumerate(frames):
        fname = normalize_file_path(fr.get("file_path", ""))
        src_img = img_map.get(fname)
        if src_img is None:
            continue
        if src_img.suffix.lower() not in IMG_EXTS:
            continue
        all_valid.append((fr, src_img))

    return all_valid


def select_evenly_spaced_frames(all_valid, n_frames: int):
    """
    Chọn n_frames frame chia đều quanh 0–360 độ (dựa trên thứ tự trong all_valid).
    all_valid: list[(frame_dict, img_path)].
    """
    M = len(all_valid)
    if M == 0:
        return []

    if M <= n_frames:
        return all_valid

    # Chọn index chia đều bằng linspace
    indices = np.linspace(0, M - 1, n_frames, dtype=int)
    indices = sorted(set(indices.tolist()))

    # Nếu vì set() mà bị thiếu thì bù thêm
    while len(indices) < n_frames:
        for extra in range(M):
            if extra not in indices:
                indices.append(extra)
                if len(indices) >= n_frames:
                    break
    indices = sorted(indices)

    chosen = [all_valid[i] for i in indices]
    return chosen


def process_object(obj_dir: Path, dst_obj_dir: Path) -> bool:
    """
    Xử lý một object:
    - Đọc transforms.json
    - Lọc valid frames
    - Chọn N_FRAMES frame chia đều
    - Resize ảnh + rename 0001.png, 0002.png, ...
    - Ghi transforms.json mới với format chuẩn PixelNeRF
    """
    tf_path = obj_dir / "transforms.json"
    if not tf_path.exists():
        print(f"  ! Bỏ qua {obj_dir.name}: không có transforms.json")
        return False

    with tf_path.open("r", encoding="utf-8") as f:
        tf_data = json.load(f)

    all_valid = list_valid_frames(obj_dir, tf_data)
    M = len(all_valid)
    if M < N_FRAMES:
        print(f"  ! Bỏ qua {obj_dir.name}: chỉ có {M} frame hợp lệ (< {N_FRAMES})")
        return False

    chosen = select_evenly_spaced_frames(all_valid, N_FRAMES)
    print(f"  ✓ {obj_dir.name}: {M} frames → chọn {len(chosen)} frames")

    dst_obj_dir.mkdir(parents=True, exist_ok=True)

    new_frames = []

    # Copy + Resize ảnh
    for i, (fr, src_img) in enumerate(chosen, start=1):
        new_name = f"{i:04d}.png"
        dst_img_path = dst_obj_dir / new_name

        im = Image.open(src_img).convert("RGB")
        im = im.resize(RESIZE_TO, Image.BILINEAR)
        im.save(dst_img_path)

        # Chỉ giữ các field cần thiết cho PixelNeRF
        new_frame = {
            "file_path": f"./{new_name}",
            "w": RESIZE_TO[0],
            "h": RESIZE_TO[1],
            "transform_matrix": fr.get("transform_matrix", [])
        }
        new_frames.append(new_frame)

    # Tạo transforms.json mới với format đơn giản
    tf_new = {
        "camera_angle_x": tf_data.get("camera_angle_x", 0.6911),
        "frames": new_frames
    }

    with (dst_obj_dir / "transforms.json").open("w", encoding="utf-8") as f:
        json.dump(tf_new, f, ensure_ascii=False, indent=2)

    return True


def gather_all_objects():
    """
    Gom tất cả object từ SRC_ROOT.
    Hỗ trợ cả 2 cấu trúc:
    - SRC_ROOT/object_name/...
    - SRC_ROOT/train|val|test/object_name/...
    """
    all_obj_dirs = []

    # Kiểm tra xem có cấu trúc train/val/test không
    has_splits = any((SRC_ROOT / split).exists() for split in ["train", "val", "test"])

    if has_splits:
        # Quét từ train/val/test
        for split in ["train", "val", "test"]:
            split_dir = SRC_ROOT / split
            if not split_dir.exists():
                continue
            for obj in sorted(split_dir.iterdir()):
                if obj.is_dir():
                    all_obj_dirs.append(obj)
    else:
        # Quét trực tiếp từ SRC_ROOT
        for obj in sorted(SRC_ROOT.iterdir()):
            if obj.is_dir():
                # Bỏ qua các folder không phải object (ví dụ: scripts, configs)
                tf_path = obj / "transforms.json"
                if tf_path.exists():
                    all_obj_dirs.append(obj)

    print(f"Tổng số object tìm được: {len(all_obj_dirs)}")

    # Lọc objects có đủ frames
    eligible = []
    for obj_dir in all_obj_dirs:
        tf_path = obj_dir / "transforms.json"
        if not tf_path.exists():
            print(f"  ! {obj_dir.name}: không có transforms.json, bỏ.")
            continue
        try:
            tf_data = json.load(open(tf_path, "r", encoding="utf-8"))
        except Exception as e:
            print(f"  ! {obj_dir.name}: lỗi đọc transforms.json: {e}, bỏ.")
            continue

        all_valid = list_valid_frames(obj_dir, tf_data)
        if len(all_valid) < N_FRAMES:
            print(f"  ! {obj_dir.name}: chỉ có {len(all_valid)} frame hợp lệ (< {N_FRAMES}), bỏ.")
            continue

        eligible.append(obj_dir)

    print(f"Object đủ điều kiện (>= {N_FRAMES} frames): {len(eligible)}")
    return eligible


def split_objects(eligible_obj_dirs):
    """
    Chia danh sách object thành train/val/test theo tỉ lệ 70:15:15.
    """
    random.seed(RANDOM_SEED)
    obj_dirs = eligible_obj_dirs[:]
    random.shuffle(obj_dirs)

    n_total = len(obj_dirs)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val

    train_objs = obj_dirs[:n_train]
    val_objs = obj_dirs[n_train:n_train + n_val]
    test_objs = obj_dirs[n_train + n_val:]

    print(f"\n{'='*50}")
    print(f"SPLIT {int(TRAIN_RATIO*100)}:{int(VAL_RATIO*100)}:{int(TEST_RATIO*100)}")
    print(f"{'='*50}")
    print(f"Total : {n_total}")
    print(f"Train : {len(train_objs)} ({len(train_objs)/n_total*100:.1f}%)")
    print(f"Val   : {len(val_objs)} ({len(val_objs)/n_total*100:.1f}%)")
    print(f"Test  : {len(test_objs)} ({len(test_objs)/n_total*100:.1f}%)")

    return {
        "train": train_objs,
        "val": val_objs,
        "test": test_objs,
    }


def build_new_dataset(splits_map):
    """
    Tạo cấu trúc:
      DST_ROOT/train/obj_name/{0001.png, 0002.png, ..., transforms.json}
      DST_ROOT/val/...
      DST_ROOT/test/...
    """
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_name, obj_dirs in splits_map.items():
        dst_split_dir = DST_ROOT / split_name
        dst_split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*50}")
        print(f"Processing: {split_name.upper()}")
        print(f"{'='*50}")
        
        success_count = 0
        for obj_dir in obj_dirs:
            dst_obj_dir = dst_split_dir / obj_dir.name
            ok = process_object(obj_dir, dst_obj_dir)
            if ok:
                success_count += 1
        
        summary[split_name] = success_count
        print(f"\n📊 {split_name}: {success_count}/{len(obj_dirs)} objects")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    total = 0
    for k, v in summary.items():
        print(f"{k:10s}: {v} objects")
        total += v
    print(f"{'Total':10s}: {total} objects")

    return summary


def main():
    print("="*60)
    print("CHUẨN HÓA DATASET GỐM CHO PIXELNERF")
    print("="*60)
    print(f"\nSRC_ROOT  : {SRC_ROOT}")
    print(f"DST_ROOT  : {DST_ROOT}")
    print(f"N_FRAMES  : {N_FRAMES}")
    print(f"RESIZE_TO : {RESIZE_TO}")
    print(f"SPLIT     : {int(TRAIN_RATIO*100)}:{int(VAL_RATIO*100)}:{int(TEST_RATIO*100)}")
    print()

    if not SRC_ROOT.exists():
        raise SystemExit(f"❌ SRC_ROOT không tồn tại: {SRC_ROOT}")

    # Bước 1: Thu thập và lọc objects
    print("\n[1/3] Thu thập objects...")
    eligible = gather_all_objects()
    if len(eligible) == 0:
        raise SystemExit("❌ Không có object nào đủ số frame!")

    # Bước 2: Chia train/val/test
    print("\n[2/3] Chia train/val/test...")
    splits_map = split_objects(eligible)

    # Bước 3: Xử lý và tạo dataset mới
    print("\n[3/3] Tạo dataset mới...")
    build_new_dataset(splits_map)

    print(f"\n🎉 Hoàn tất! Dataset mới tại: {DST_ROOT}")
    print(f"\nOutput format:")
    print("""
{
  "camera_angle_x": 0.6699...,
  "frames": [
    {
      "file_path": "./0001.png",
      "w": 128,
      "h": 128,
      "transform_matrix": [[...], [...], [...], [...]]
    },
    ...
  ]
}
""")


if __name__ == "__main__":
    main()
