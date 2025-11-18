import os
from pathlib import Path

# Thư mục chứa train/val/test mà lúc nãy mình move
DATASET_ROOT = r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\dataset_pottery"

# Thư mục lưu các file txt mô tả split
SPLIT_ROOT = os.path.join(DATASET_ROOT, "splits")


def find_scene_path(obj_dir: Path, split_name: str) -> str:
    """
    Trả về đường dẫn tương đối cho 1 object dùng trong .txt
    Ưu tiên dùng folder pixelnerf/, fallback dùng root object.
    """
    # Ưu tiên kiểu: <split>/<obj_name>/pixelnerf nếu tồn tại transforms.json trong đó
    px_dir = obj_dir / "pixelnerf"
    if (px_dir / "transforms.json").exists():
        rel = Path(split_name) / obj_dir.name / "pixelnerf"
        return str(rel).replace("\\", "/")

    # Nếu không có pixelnerf/ nhưng có transforms.json ngay dưới object
    if (obj_dir / "transforms.json").exists():
        rel = Path(split_name) / obj_dir.name
        return str(rel).replace("\\", "/")

    # Không có gì dùng được -> bỏ
    return None


def main():
    root = Path(DATASET_ROOT)
    assert root.exists(), f"DATASET_ROOT không tồn tại: {DATASET_ROOT}"

    os.makedirs(SPLIT_ROOT, exist_ok=True)

    split_files = {
        "train": open(os.path.join(SPLIT_ROOT, "train.txt"), "w", encoding="utf-8"),
        "val": open(os.path.join(SPLIT_ROOT, "val.txt"), "w", encoding="utf-8"),
        "test": open(os.path.join(SPLIT_ROOT, "test.txt"), "w", encoding="utf-8"),
    }

    counts = {k: 0 for k in split_files.keys()}

    try:
        for split_name in ["train", "val", "test"]:
            split_dir = root / split_name
            if not split_dir.exists():
                print(f"[WARN] Không tìm thấy folder split: {split_dir}")
                continue

            for obj_name in sorted(os.listdir(split_dir)):
                obj_dir = split_dir / obj_name
                if not obj_dir.is_dir():
                    continue

                scene_rel = find_scene_path(obj_dir, split_name)
                if scene_rel is None:
                    print(f"  ! Bỏ qua {obj_dir} (không tìm thấy transforms.json hợp lệ)")
                    continue

                split_files[split_name].write(scene_rel + "\n")
                counts[split_name] += 1

        print("\n=== TÓM TẮT SỐ LƯỢNG SCENE GHI VÀO TXT ===")
        for k, v in counts.items():
            print(f"  {k:5s}: {v} object")

        print("\nCác file split đã tạo:")
        print(f"  - {os.path.join(SPLIT_ROOT, 'train.txt')}")
        print(f"  - {os.path.join(SPLIT_ROOT, 'val.txt')}")
        print(f"  - {os.path.join(SPLIT_ROOT, 'test.txt')}")

    finally:
        for f in split_files.values():
            f.close()


if __name__ == "__main__":
    main()
