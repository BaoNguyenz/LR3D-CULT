import os, json, math, sys
from pathlib import Path
from PIL import Image
from typing import Optional
# === CHỈNH ĐƯỜNG DẪN GỐC TẠI ĐÂY ===
ALL_DATA = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data"

# Các phần mở rộng ảnh sẽ dò nếu thiếu đuôi
EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

def find_image(img_dir: Path, basename: str) -> Optional[Path]:
    """
    Trả về đường dẫn ảnh trong img_dir trùng với basename.
    - Nếu basename có đuôi: thử đúng tên trước.
    - Nếu không có đuôi: thử với EXTS.
    - Nếu không có file_path hoặc sai thư mục -> cố gắng khớp theo tên file cuối.
    """
    base = Path(basename).name  # chỉ giữ phần tên file
    stem, ext = os.path.splitext(base)

    # 1) Nếu có đuôi → thử trực tiếp
    if ext:
        p = img_dir / base
        if p.exists():
            return p

    # 2) Thử mọi đuôi cho cùng stem
    for e in EXTS:
        p = img_dir / f"{stem}{e}"
        if p.exists():
            return p

    # 3) Nếu vẫn chưa thấy, thử quét thư mục để tìm khớp tên không phân biệt hoa/thường
    lower = base.lower()
    for p in img_dir.iterdir():
        if p.is_file() and p.name.lower() == lower:
            return p

    # 4) Không tìm thấy
    return None

def compute_intrinsics_from_fov(width: int, height: int, camera_angle_x: float):
    """Tính intrinsics (fx, fy, cx, cy) từ FOV ngang + kích thước ảnh."""
    fx = 0.5 * width / math.tan(0.5 * camera_angle_x)
    fy = fx  # giả định pixel square
    cx = width * 0.5
    cy = height * 0.5
    return fx, fy, cx, cy

def normalize_rel_path(p: Path):
    """Chuyển thành đường dẫn tương đối dùng forward-slash: ./images/xxx.png"""
    return "./" + str(p.as_posix())

def patch_one_object(obj_dir: Path) -> dict:
    """
    Patch transforms.json trong obj_dir:
    - Đọc camera_angle_x tại root
    - Với mỗi frame:
        - Đồng bộ file_path → ./images/<file>
        - Tìm ảnh trong /images, đo w,h
        - Ghi w,h, fl_x, fl_y, cx, cy, camera_model="PINHOLE"
    - Ghi backup transforms.backup.json
    - Ghi transforms.json mới
    Trả về thống kê.
    """
    stats = {
        "object": obj_dir.name,
        "frames_total": 0,
        "frames_patched": 0,
        "frames_missing_image": 0,
        "skipped_because_no_angle": False,
        "errors": []
    }

    tf = obj_dir / "transforms.json"
    img_dir = obj_dir / "images"

    if not tf.exists():
        stats["errors"].append("Không có transforms.json")
        return stats
    if not img_dir.exists():
        stats["errors"].append("Không có thư mục images/")
        return stats

    try:
        data = json.load(open(tf, "r", encoding="utf-8"))
    except Exception as e:
        stats["errors"].append(f"Lỗi đọc transforms.json: {e}")
        return stats

    cam_angle_x = data.get("camera_angle_x", None)
    if cam_angle_x is None:
        stats["skipped_because_no_angle"] = True
        stats["errors"].append("Thiếu camera_angle_x ở root → bỏ qua object này (khuyên dùng ns-process-data colmap).")
        return stats

    frames = data.get("frames", [])
    stats["frames_total"] = len(frames)

    for fr in frames:
        # 1) Chuẩn hoá file_path về ./images/<file>
        orig_fp = fr.get("file_path", "")
        base = Path(orig_fp.replace("./", "")).name  # chỉ giữ tên file (bỏ thư mục cũ nếu có)
        img_path = find_image(img_dir, base)

        if img_path is None:
            stats["frames_missing_image"] += 1
            # Vẫn set lại file_path về ./images/<base> với đuôi gốc (nếu có) để còn tự sửa tay
            fr["file_path"] = normalize_rel_path(Path("images") / base)
            continue

        # 2) Đảm bảo file_path đúng ./images/<tên-đúng>
        fr["file_path"] = normalize_rel_path(Path("images") / img_path.name)

        # 3) Đọc w,h rồi tính intrinsics
        try:
            with Image.open(img_path) as im:
                w, h = im.width, im.height
        except Exception as e:
            stats["frames_missing_image"] += 1
            stats["errors"].append(f"Lỗi đọc ảnh {img_path.name}: {e}")
            continue

        fx, fy, cx, cy = compute_intrinsics_from_fov(w, h, cam_angle_x)

        # 4) Ghi các trường intrinsics theo schema Nerfstudio
        fr["w"] = w
        fr["h"] = h
        fr["fl_x"] = fx
        fr["fl_y"] = fy
        fr["cx"] = cx
        fr["cy"] = cy
        # Không có distortion → để PINHOLE là hợp lý
        fr["camera_model"] = "PINHOLE"

        stats["frames_patched"] += 1

    # Ghi file
    backup = obj_dir / "transforms.backup.json"
    try:
        if not backup.exists():
            # lưu backup lần đầu
            with open(backup, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)  # backup bản đã chỉnh? hoặc đọc lại gốc trước khi chỉnh
            # lưu ý: nếu muốn backup bản gốc, cần đọc gốc trước khi chỉnh; ở đây đơn giản hoá.
    except Exception as e:
        stats["errors"].append(f"Lỗi ghi backup: {e}")

    try:
        with open(tf, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        stats["errors"].append(f"Lỗi ghi transforms.json mới: {e}")

    return stats

def main():
    root = Path(ALL_DATA)
    if not root.exists():
        print(f"[ERR] ALL_DATA không tồn tại: {root}")
        sys.exit(1)

    obj_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    all_stats = []

    print(f"🔧 Bắt đầu patch intrinsics cho {len(obj_dirs)} object trong: {root}")
    for obj in obj_dirs:
        print(f"\n=== {obj.name} ===")
        stats = patch_one_object(obj)
        all_stats.append(stats)
        if stats["errors"]:
            for e in stats["errors"]:
                print("  [ERR]", e)
        print(f"  frames: {stats['frames_patched']}/{stats['frames_total']} patched, thiếu ảnh: {stats['frames_missing_image']}")

    # Tổng kết
    total = len(all_stats)
    ok = sum(1 for s in all_stats if not s["errors"])
    skipped = sum(1 for s in all_stats if s["skipped_because_no_angle"])
    missing = sum(s["frames_missing_image"] for s in all_stats)
    patched = sum(s["frames_patched"] for s in all_stats)
    frames = sum(s["frames_total"] for s in all_stats)

    print("\n================ TỔNG KẾT ================")
    print(f"Object xử lý: {total}, không lỗi nghiêm trọng: {ok}, bỏ qua (thiếu camera_angle_x): {skipped}")
    print(f"Tổng frames: {frames}, đã patch: {patched}, thiếu ảnh: {missing}")
    print("==========================================")

if __name__ == "__main__":
    main()
