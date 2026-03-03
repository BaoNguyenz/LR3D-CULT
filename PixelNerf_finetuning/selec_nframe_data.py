import json
import shutil
import random
from pathlib import Path
import numpy as np

# ========== CẤU HÌNH ==========
SRC_ROOT = Path(r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\dataset_pottery_trans_128")
DST_ROOT = Path(r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\dataset_pottery_trans_128_50nv")
N_FRAMES = 50
IMG_EXTS = {".png", ".jpg", ".jpeg"}
RANDOM_SEED = 42
# ==============================

def normalize_file_path(rel: str) -> str:
    if rel is None:
        return ""
    rel = rel.replace("\\", "/").lstrip("./")
    if rel.startswith("images/"):
        rel = rel[len("images/"):]
    return rel.split("/")[-1]

def list_valid_frames(obj_dir: Path, tf_data: dict):
    images_dir = obj_dir / "images"
    search_dir = images_dir if images_dir.exists() and images_dir.is_dir() else obj_dir

    img_map = {}
    for p in search_dir.iterdir() if search_dir.exists() else []:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            img_map[p.name] = p

    frames = tf_data.get("frames", [])
    all_valid = []
    for fr in frames:
        fname = normalize_file_path(fr.get("file_path", ""))
        src_img = img_map.get(fname)
        if src_img is None:
            continue
        if src_img.suffix.lower() not in IMG_EXTS:
            continue
        all_valid.append((fr, src_img))
    return all_valid

def select_evenly_spaced_frames(all_valid, n_frames: int):
    M = len(all_valid)
    if M == 0:
        return []
    if M <= n_frames:
        return all_valid
    # numpy linspace to choose indices uniformly
    indices = np.linspace(0, M - 1, n_frames, dtype=int)
    indices = sorted(set(indices.tolist()))
    # fill if set removed duplicates
    i = 0
    while len(indices) < n_frames:
        if i not in indices:
            indices.append(i)
        i += 1
        if i >= M:
            i = 0
    indices = sorted(indices)[:n_frames]
    chosen = [all_valid[idx] for idx in indices]
    return chosen

def process_object(obj_dir: Path, dst_obj_dir: Path) -> bool:
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
    print(f"  - {obj_dir.name}: tổng {M} frame hợp lệ, chọn {len(chosen)} frame để copy")

    dst_obj_dir.mkdir(parents=True, exist_ok=True)

    new_frames = []
    for i, (fr, src_img) in enumerate(chosen, start=1):
        new_name = f"{i:04d}{src_img.suffix.lower()}"
        dst_img_path = dst_obj_dir / new_name
        try:
            shutil.copy2(src_img, dst_img_path)  # copy nguyên file, giữ alpha
        except Exception as e:
            print(f"    ! Lỗi copy {src_img} -> {dst_img_path}: {e}. Bỏ frame.")
            continue

        fr_new = dict(fr)
        fr_new["file_path"] = f"./{new_name}"
        # giữ nguyên fr_new["w"], fr_new["h"], transform_matrix, intrinsics...
        new_frames.append(fr_new)

    if not new_frames:
        print(f"  ! {obj_dir.name} không còn frame hợp lệ sau copy, bỏ object.")
        return False

    tf_new = dict(tf_data)
    tf_new["frames"] = new_frames
    # giữ nguyên tf_new["w"], tf_new["h"], intrinsics, v.v.

    dst_tf_path = dst_obj_dir / "transforms.json"
    with dst_tf_path.open("w", encoding="utf-8") as f:
        json.dump(tf_new, f, ensure_ascii=False, indent=2)

    print(f"  + [{obj_dir.name}] Hoàn tất: {len(new_frames)} frames written to {dst_obj_dir}")
    return True

def find_object_dirs(src_root: Path):
    """Tìm các thư mục object trực tiếp dưới SRC_ROOT hoặc trong train/val/test."""
    obj_dirs = []
    # Check top-level folders
    for p in sorted(src_root.iterdir()):
        if p.is_dir() and (p / "transforms.json").exists():
            obj_dirs.append(p)
    # Also check train/val/test subfolders (if dataset uses them)
    for split in ["train", "val", "test"]:
        sp = src_root / split
        if sp.exists() and sp.is_dir():
            for p in sorted(sp.iterdir()):
                if p.is_dir() and (p / "transforms.json").exists():
                    obj_dirs.append(p)
    # unique
    obj_dirs = sorted(set(obj_dirs))
    return obj_dirs

def main():
    random.seed(RANDOM_SEED)
    print("Start uniform N_FRAMES copy")
    print("SRC_ROOT:", SRC_ROOT)
    print("DST_ROOT:", DST_ROOT)
    print("N_FRAMES:", N_FRAMES)
    print()

    if not SRC_ROOT.exists():
        raise SystemExit(f"SRC_ROOT không tồn tại: {SRC_ROOT}")
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    obj_dirs = find_object_dirs(SRC_ROOT)
    print(f"Found {len(obj_dirs)} objects with transforms.json")

    n_ok = 0
    n_fail = 0
    for od in obj_dirs:
        print(f"\nProcessing object: {od.name}")
        dst_obj_dir = DST_ROOT / od.name
        ok = process_object(od, dst_obj_dir)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print("\n=== DONE ===")
    print(f"Success: {n_ok}, Failed: {n_fail}")

if __name__ == "__main__":
    main()
