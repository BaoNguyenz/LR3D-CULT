import os
import json
import shutil
import random
from pathlib import Path
import math

from PIL import Image
import numpy as np

# ================== CONFIG ==================
SRC_ROOT = Path(r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\dataset_pottery_ver3")

DST_ROOT = Path(r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\dataset_pottery_ver3_90nv")

N_FRAMES = 90

RESIZE_TO = (128, 128)

IMG_EXTS = {".png", ".jpg", ".jpeg"}

RANDOM_SEED = 42
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
            # print(f"    ! Không tìm thấy ảnh {fname}, bỏ qua frame {idx}.")
            continue
        if src_img.suffix.lower() not in IMG_EXTS:
            # print(f"    ! {fname} không phải ảnh hợp lệ, bỏ qua frame {idx}.")
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
        # Nếu ít hoặc bằng N_FRAMES thì dùng hết
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
    - Đọc transforms.json
    - Lọc tất cả valid frames
    - Nếu < N_FRAMES -> bỏ
    - Nếu >= N_FRAMES -> chọn N_FRAMES frame chia đều
    - Resize ảnh + rename 0001.png, 0002.png, ...
    - Ghi transforms.json mới (file_path = ./0001.png ...)
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
    print(f"  - {obj_dir.name}: tổng {M} frame hợp lệ, chọn {len(chosen)} frame để export")

    dst_obj_dir.mkdir(parents=True, exist_ok=True)

    new_frames = []

    # Dùng frame đầu tiên gốc để lấy w,h cũ (nếu có) cho scale_ratio
    orig_w = tf_data.get("w", None)
    if orig_w is None and len(tf_data.get("frames", [])) > 0:
        orig_w = tf_data["frames"][0].get("w", None)

    # COPY + RESIZE ảnh
    for i, (fr, src_img) in enumerate(chosen, start=1):
        new_name = f"{i:04d}{src_img.suffix.lower()}"
        dst_img_path = dst_obj_dir / new_name

        im = Image.open(src_img).convert("RGB")
        im = im.resize(RESIZE_TO, Image.BILINEAR)
        im.save(dst_img_path)

        fr = dict(fr)  # clone
        fr["file_path"] = f"./{new_name}"
        fr["w"] = RESIZE_TO[0]
        fr["h"] = RESIZE_TO[1]
        new_frames.append(fr)

    # Ghi transforms.json mới
    tf_new = dict(tf_data)
    tf_new["frames"] = new_frames
    tf_new["w"] = RESIZE_TO[0]
    tf_new["h"] = RESIZE_TO[1]

    # Nếu có intrinsics global (fl_x, cx, ...) thì scale theo width
    if orig_w is not None and orig_w > 0:
        scale_ratio = RESIZE_TO[0] / float(orig_w)
        for key in ("fl_x", "fl_y", "cx", "cy"):
            if key in tf_new:
                tf_new[key] = float(tf_new[key]) * scale_ratio

    with (dst_obj_dir / "transforms.json").open("w", encoding="utf-8") as f:
        json.dump(tf_new, f, ensure_ascii=False, indent=2)

    return True


def gather_all_objects():
    """
    Gom tất cả object từ SRC_ROOT/train, val, test.
    Chỉ giữ những object có >= N_FRAMES frame hợp lệ.
    """
    all_obj_dirs = []

    for split in ["train", "val", "test"]:
        split_dir = SRC_ROOT / split
        if not split_dir.exists():
            continue
        for obj in sorted(split_dir.iterdir()):
            if obj.is_dir():
                all_obj_dirs.append(obj)

    print(f"Tổng số object tìm được (chưa lọc frame): {len(all_obj_dirs)}")

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
            print(f"  ! {obj_dir.name}: chỉ có {len(all_valid)} frame hợp lệ, bỏ.")
            continue

        eligible.append(obj_dir)

    print(f"Object đủ điều kiện (>= {N_FRAMES} frame): {len(eligible)}")
    return eligible


def split_objects_8_1_1(eligible_obj_dirs):
    """
    Chia danh sách object thành train/val/test theo tỉ lệ 8:1:1.
    """
    random.seed(RANDOM_SEED)
    obj_dirs = eligible_obj_dirs[:]
    random.shuffle(obj_dirs)

    n_total = len(obj_dirs)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_objs = obj_dirs[:n_train]
    val_objs = obj_dirs[n_train:n_train + n_val]
    test_objs = obj_dirs[n_train + n_val:]

    print("\n===== SPLIT 8:1:1 =====")
    print(f"Total: {n_total}")
    print(f"train: {len(train_objs)}")
    print(f"val  : {len(val_objs)}")
    print(f"test : {len(test_objs)}")

    return {
        "train": train_objs,
        "val": val_objs,
        "test": test_objs,
    }


def build_new_dataset(splits_map):
    """
    Tạo cấu trúc:
      DST_ROOT/train/obj_name/{0001.png,0002.png,transforms.json}
      DST_ROOT/val/...
      DST_ROOT/test/...
    """
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_name, obj_dirs in splits_map.items():
        dst_split_dir = DST_ROOT / split_name
        dst_split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n======= Xử lý split mới: {split_name} =======")
        used = 0
        for obj_dir in obj_dirs:
            print(f"\n🧱 Object: {obj_dir.name}")
            dst_obj_dir = dst_split_dir / obj_dir.name
            ok = process_object(obj_dir, dst_obj_dir)
            if ok:
                used += 1
        summary[split_name] = used
        print(f"\n📊 Split {split_name}: dùng được {used}/{len(obj_dirs)} object")

    print("\n===== TỔNG KẾT CUỐI =====")
    for k, v in summary.items():
        print(f"{k}: {v} object")


def main():
    print("=== Chuẩn hóa dataset gốm cho PixelNeRF ===")
    print(f"SRC_ROOT: {SRC_ROOT}")
    print(f"DST_ROOT: {DST_ROOT}")
    print(f"N_FRAMES mỗi object: {N_FRAMES}")
    print(f"Resize: {RESIZE_TO}")
    print("Split tỉ lệ: train:val:test = 8:1:1")
    print()

    if not SRC_ROOT.exists():
        raise SystemExit(f"SRC_ROOT không tồn tại: {SRC_ROOT}")

    eligible = gather_all_objects()
    if len(eligible) == 0:
        raise SystemExit("Không có object nào đủ số frame, dừng.")

    splits_map = split_objects_8_1_1(eligible)
    build_new_dataset(splits_map)

    print("\n🎉 Hoàn tất tạo dataset mới:", DST_ROOT)


if __name__ == "__main__":
    main()
