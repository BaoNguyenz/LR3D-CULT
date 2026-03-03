import os
import shutil
import random
from collections import defaultdict

import matplotlib.pyplot as plt

# ================== CẤU HÌNH ==================

# Thư mục gốc hiện tại chứa tất cả object đã chuẩn hóa cho PixelNeRF
SOURCE_ROOT = r"Final_data_trans"

# Thư mục mới sẽ chứa train/val/test
DEST_ROOT = r"Final_data_trans"

# Chia theo tỷ lệ 70 / 15 / 15 cho từng loại
TRAIN_RATIO = 0.70
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Nếu muốn thử trước không move thư mục, để True
DRY_RUN = False

# Seed để lần sau chạy lại vẫn ra split giống nhau
RANDOM_SEED = 42

# Định nghĩa các loại theo prefix
TYPE_PREFIXES = {
    "bat_gom": ["bat_gom_"],
    "binh_gom": ["binh_gom_"],
    "binh_gom_bt": ["binh_gom_bt_"],
    "chen_gom": ["chen_gom_"],
    "dia_gom": ["dia_gom_"],
    "ly_dong": ["ly_dong_"],
}

# Nếu folder không khớp prefix nào ở trên thì xếp vào "other"
OTHER_TYPE_NAME = "other"

# =============================================


def detect_type(folder_name: str) -> str:
    """Xác định type dựa trên prefix của tên folder."""
    for tname, prefixes in TYPE_PREFIXES.items():
        for p in prefixes:
            if folder_name.startswith(p):
                return tname
    return OTHER_TYPE_NAME


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    random.seed(RANDOM_SEED)

    # 1. Liệt kê tất cả object trong SOURCE_ROOT
    all_objs = [
        d for d in os.listdir(SOURCE_ROOT)
        if os.path.isdir(os.path.join(SOURCE_ROOT, d))
    ]
    print(f"Tổng số object tìm thấy trong SOURCE_ROOT: {len(all_objs)}\n")

    # 2. Gom theo type
    type_to_objs = defaultdict(list)
    for obj_name in all_objs:
        tname = detect_type(obj_name)
        type_to_objs[tname].append(obj_name)

    # In sơ bộ số lượng mỗi loại
    print("Số object theo từng loại (before split):")
    for tname, objs in sorted(type_to_objs.items()):
        print(f"  - {tname}: {len(objs)} object")
    print()

    # 3. Chuẩn bị thư mục đích
    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(DEST_ROOT, split))

    # 4. Chia train/val/test từng loại
    split_counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    for tname, objs in sorted(type_to_objs.items()):
        objs_sorted = sorted(objs)
        random.shuffle(objs_sorted)

        n = len(objs_sorted)
        if n == 0:
            continue

        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        # Đảm bảo không bị lệch tổng
        n_test = n - n_train - n_val

        train_objs = objs_sorted[:n_train]
        val_objs = objs_sorted[n_train:n_train + n_val]
        test_objs = objs_sorted[n_train + n_val:]

        # Ghi log
        print(f"Loại: {tname}")
        print(f"  Tổng: {n}")
        print(f"  -> Train: {len(train_objs)} | Val: {len(val_objs)} | Test: {len(test_objs)}")

        # Cập nhật thống kê
        split_counts["train"][tname] += len(train_objs)
        split_counts["val"][tname] += len(val_objs)
        split_counts["test"][tname] += len(test_objs)

        # 5. Move (hoặc dry-run)
        for split_name, obj_list in [("train", train_objs),
                                     ("val", val_objs),
                                     ("test", test_objs)]:
            for obj_name in obj_list:
                src = os.path.join(SOURCE_ROOT, obj_name)
                dst = os.path.join(DEST_ROOT, split_name, obj_name)
                if DRY_RUN:
                    print(f"  [DRY_RUN] {src}  ->  {dst}")
                else:
                    shutil.move(src, dst)

        print()

    # 6. In tổng kết sau khi split
    print("\n================= TỔNG KẾT SPLIT =================")
    all_types = sorted(type_to_objs.keys())

    for tname in all_types:
        tr = split_counts["train"][tname]
        va = split_counts["val"][tname]
        te = split_counts["test"][tname]
        print(f"{tname:10s} | train: {tr:3d} | val: {va:3d} | test: {te:3d} | total: {tr + va + te:3d}")

    # 7. Plot kết quả
    print("\nVẽ biểu đồ split theo loại...")

    types_for_plot = [t for t in all_types if (split_counts["train"][t] +
                                              split_counts["val"][t] +
                                              split_counts["test"][t]) > 0]

    train_vals = [split_counts["train"][t] for t in types_for_plot]
    val_vals = [split_counts["val"][t] for t in types_for_plot]
    test_vals = [split_counts["test"][t] for t in types_for_plot]

    x = range(len(types_for_plot))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar([i - width for i in x], train_vals, width=width, label="Train")
    plt.bar(x, val_vals, width=width, label="Val")
    plt.bar([i + width for i in x], test_vals, width=width, label="Test")

    plt.xticks(x, types_for_plot, rotation=30)
    plt.ylabel("Số object")
    plt.title("Phân chia train / val / test theo từng loại object gốm")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n🎉 Hoàn tất chia train/val/test. Kiểm tra lại thư mục:")
    print(f"  - {os.path.join(DEST_ROOT, 'train')}")
    print(f"  - {os.path.join(DEST_ROOT, 'val')}")
    print(f"  - {os.path.join(DEST_ROOT, 'test')}")


if __name__ == "__main__":
    main()
