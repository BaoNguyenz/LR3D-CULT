import os
import subprocess
import shutil

# === CHỈNH ĐƯỜNG DẪN GỐC CỦA BẠN TẠI ĐÂY ===
ROOT = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data_few_shot"
# ===========================================

# Tìm colmap trong PATH; nếu không thấy thì dùng đường dẫn .BAT
COLMAP = shutil.which("colmap")
if COLMAP is None:
    COLMAP = r"C:\Program Files\colmap-x64-windows-cuda\COLMAP.bat"  # <-- đổi nếu khác

def run(args):
    """
    args: list[str] chưa quote.
    Dùng list2cmdline để quote đúng chuẩn Windows rồi chạy shell=True.
    """
    cmdline = subprocess.list2cmdline(args)
    print(">>", cmdline)
    subprocess.check_call(cmdline, shell=True)

def run_colmap(obj_dir):
    images_dir = os.path.join(obj_dir, "images")
    if not os.path.isdir(images_dir):
        print(f"[BỎ QUA] {obj_dir} - không có thư mục images/")
        return

    colmap_dir = os.path.join(obj_dir, "colmap")
    sparse_dir = os.path.join(colmap_dir, "sparse")
    dense_dir = os.path.join(colmap_dir, "dense")
    db_path = os.path.join(colmap_dir, "database.db")

    # Tạo đủ thư mục cha (để COLMAP không kêu ExistsDir)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)

    print(f"\n=== ĐANG XỬ LÝ: {obj_dir} ===")

    # 1) Feature extraction
    run([
        COLMAP, "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1"
    ])

    # 2) Exhaustive matcher
    run([
        COLMAP, "exhaustive_matcher",
        "--database_path", db_path
    ])

    # 3) Mapper
    run([
        COLMAP, "mapper",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir
    ])

    print(f"[OK] Đã hoàn tất COLMAP cho {obj_dir}")

def main():
    # Kiểm tra colmap .bat tồn tại
    if not os.path.exists(COLMAP):
        raise SystemExit(f"Không tìm thấy COLMAP tại: {COLMAP}")
    # Duyệt lần lượt tất cả object
    for name in sorted(os.listdir(ROOT)):
        obj_path = os.path.join(ROOT, name)
        if os.path.isdir(obj_path):
            run_colmap(obj_path)

if __name__ == "__main__":
    main()
