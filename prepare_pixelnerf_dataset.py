import os
import shutil
import json
from pathlib import Path
from PIL import Image

# === 1. Cấu hình đường dẫn gốc ===
ROOT = r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\All_data"

# Các thư mục sẽ xóa
DELETE_DIRS = ["colmap", "fewshot"]

# Các phần mở rộng file sẽ xóa
DELETE_EXTS = {".mtl", ".obj", ".bak"}

# Các file tên đặc biệt sẽ xóa
SPECIAL_DELETE_NAMES = {"metadata", "transforms.backup", "transforms.old"}

# Định dạng ảnh hợp lệ
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def clean_object_dir(obj_dir: Path):
    print(f"\n=== Xử lý object: {obj_dir.name} ===")

    # 1. Xóa thư mục rác
    for d in DELETE_DIRS:
        p = obj_dir / d
        if p.exists():
            shutil.rmtree(p)
            print(f"  - Xóa folder: {p}")

    # 2. Xóa file rác
    for f in list(obj_dir.iterdir()):
        if f.is_dir():
            continue
        
        name = f.stem
        ext = f.suffix.lower()

        if name in SPECIAL_DELETE_NAMES:
            f.unlink()
            print(f"  - Xóa file special: {f.name}")
            continue
        
        if ext in DELETE_EXTS:
            f.unlink()
            print(f"  - Xóa file: {f.name}")
            continue
        
        # Xóa preview PNG trùng tên object
        if ext in IMG_EXTS and name == obj_dir.name:
            f.unlink()
            print(f"  - Xóa preview: {f.name}")
            continue

    # 3. Kiểm tra images/ và transforms.json
    images_dir = obj_dir / "images"
    tf_path = obj_dir / "transforms.json"

    if not images_dir.exists():
        print("  ! Bỏ qua: không có images/")
        return

    if not tf_path.exists():
        print("  ! Bỏ qua: không có transforms.json")
        return

    # 4. Đọc transforms.json
    with tf_path.open("r", encoding="utf-8") as f:
        tf_data = json.load(f)

    frames = tf_data.get("frames", [])
    print(f"  - Frames trong transforms.json: {len(frames)}")

    # 5. Tạo folder tạm cho ảnh sạch
    tmp_dir = obj_dir / "images_clean"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    # 6. Lọc frames hợp lệ
    new_frames = []
    counter = 0
    dropped = 0

    for idx, fr in enumerate(frames):
        rel = fr.get("file_path", "").lstrip("./").replace("\\", "/")
        
        if rel.startswith("images/"):
            img_name = rel.split("/", 1)[1]
        else:
            img_name = rel

        img_path = images_dir / img_name

        if not img_path.exists():
            print(f"    ! Frame {idx}: MISSING ảnh -> DROP")
            dropped += 1
            continue
        
        if img_path.suffix.lower() not in IMG_EXTS:
            print(f"    ! Frame {idx}: Không phải ảnh -> DROP")
            dropped += 1
            continue

        counter += 1
        new_name = f"{counter:04d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, tmp_dir / new_name)

        fr["file_path"] = f"./images/{new_name}"
        new_frames.append(fr)

    print(f"  - Frame hợp lệ: {len(new_frames)} (drop {dropped})")

    if len(new_frames) == 0:
        print("  ! Không có frame hợp lệ -> SKIP object.")
        shutil.rmtree(tmp_dir)
        return

    # 7. Xóa images/ cũ và thay bằng images_clean/
    shutil.rmtree(images_dir)
    tmp_dir.rename(images_dir)
    print("  ✅ Đã tạo images/ sạch.")

    # 8. Ghi transforms.json mới (không backup)
    tf_data["frames"] = new_frames
    with tf_path.open("w", encoding="utf-8") as f:
        json.dump(tf_data, f, ensure_ascii=False, indent=2)

    print("  ✅ Đã ghi transforms.json mới.")


def main():
    root = Path(ROOT)
    assert root.exists(), f"ROOT không tồn tại: {ROOT}"

    print(f"Root: {root}")
    for obj in sorted(root.iterdir()):
        if obj.is_dir():
            clean_object_dir(obj)

    print("\n🎉 Hoàn tất CLEAN dataset cho PixelNeRF (Không backup).")


if __name__ == "__main__":
    main()
