import os
import json
from pathlib import Path
from shutil import move

ROOT = r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\dataset_pottery"
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def fix_one_obj(obj_dir: Path):
    print(f"\n=== Xử lý: {obj_dir} ===")

    images_dir = obj_dir / "images"
    tf_path = obj_dir / "transforms.json"

    if not images_dir.exists():
        print("  - Không có folder images/, bỏ qua.")
        return

    # 1. Move ảnh từ images/ ra ngoài
    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if not imgs:
        print("  - images/ không có ảnh, bỏ qua.")
        return

    for img in imgs:
        dst = obj_dir / img.name
        if dst.exists():
            print(f"    ! Bỏ qua {img.name} vì {dst} đã tồn tại.")
            continue
        print(f"  - Move {img.name} -> {dst.name}")
        move(str(img), str(dst))

    # Xoá folder images nếu rỗng
    try:
        images_dir.rmdir()
        print("  - Đã xoá folder images/ (rỗng).")
    except OSError:
        print("  - Folder images/ chưa rỗng, không xoá được (kiểm tra lại).")

    # 2. Sửa transforms.json nếu có
    if tf_path.exists():
        with tf_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        fixed = 0
        for fr in frames:
            fp = fr.get("file_path", "")
            if "images/" in fp:
                new_fp = fp.replace("images/", "")
                fr["file_path"] = new_fp
                fixed += 1

        if fixed > 0:
            with tf_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  - Đã sửa {fixed} file_path trong transforms.json (bỏ prefix images/).")
        else:
            print("  - transforms.json không chứa 'images/', không cần sửa.")
    else:
        print("  - Không có transforms.json, bỏ qua.")


def main():
    root = Path(ROOT)
    assert root.exists(), f"ROOT không tồn tại: {ROOT}"

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            print(f"\n!!! Split {split} không tồn tại, bỏ qua.")
            continue

        print(f"\n===== Split: {split} =====")
        for obj_dir in sorted(split_dir.iterdir()):
            if obj_dir.is_dir():
                fix_one_obj(obj_dir)

    print("\n✅ Hoàn tất chuẩn hóa dataset cho PixelNeRF (bỏ folder images/).")


if __name__ == "__main__":
    main()
