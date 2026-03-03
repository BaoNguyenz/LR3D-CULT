# -*- coding: utf-8 -*-
"""
Auto-pipeline cho toàn bộ dataset:
- Chọn sparse model tốt nhất (points3D.bin lớn nhất)
- colmap model_converter -> TXT
- cameras.txt + images.txt -> transforms.json (chuẩn NeRF paper)
"""

import os, re, json, math, shutil, subprocess, sys
from pathlib import Path
from typing import Optional 

# === CHỈNH ĐƯỜNG DẪN GỐC Ở ĐÂY ===
ROOT = r"E:\LET_ME_COOK\Captone\NeRF_finetuning\All_data_few_shot"
# ================================

# Tìm colmap
COLMAP = shutil.which("colmap")
if COLMAP is None:
    COLMAP = r"C:\Program Files\colmap-x64-windows-cuda\COLMAP.bat"

def run_cmd(args):
    """Chạy lệnh Windows an toàn (đường dẫn có dấu cách)."""
    cmdline = subprocess.list2cmdline(args)
    print(">>", cmdline)
    subprocess.check_call(cmdline, shell=True)

# ---------- ĐỌC CAMERAS.TXT ----------
def parse_cameras_txt(fp):
    cams = {}
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            toks = line.split()
            cam_id = int(toks[0])
            model = toks[1]
            width = float(toks[2]); height = float(toks[3])
            params = list(map(float, toks[4:]))

            # Lấy fx, fy, cx, cy theo model
            # Tham khảo colmap models:
            # SIMPLE_PINHOLE: f, cx, cy
            # PINHOLE: fx, fy, cx, cy
            # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            # RADIAL: f, cx, cy, k1, k2
            # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
            if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"]:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                # Mặc định an toàn
                fx = fy = params[0]
                cx, cy = params[1], params[2]

            cams[cam_id] = dict(model=model, width=width, height=height,
                                fx=fx, fy=fy, cx=cx, cy=cy, raw_params=params)
    return cams

# ---------- QUAT → ROT ----------
def qvec2rotmat(q):
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q
    # chuẩn hóa đề phòng
    s = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/s, qx/s, qy/s, qz/s
    R = [
        [1 - 2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1 - 2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1 - 2*(qx*qx+qy*qy)],
    ]
    return R

def matmul3(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def matvec3(A, v):
    return [sum(A[i][k]*v[k] for k in range(3)) for i in range(3)]

def invert_pose(R, t):
    # camera-to-world = [R|t]^{-1} = [R^T | -R^T t]
    Rt = [[R[0][0], R[1][0], R[2][0]],
          [R[0][1], R[1][1], R[2][1]],
          [R[0][2], R[1][2], R[2][2]]]
    tt = matvec3(Rt, [-t[0], -t[1], -t[2]])
    return Rt, tt

def compose_4x4(R, t):
    return [
        [R[0][0], R[0][1], R[0][2], t[0]],
        [R[1][0], R[1][1], R[1][2], t[1]],
        [R[2][0], R[2][1], R[2][2], t[2]],
        [0,0,0,1]
    ]

# ---------- ĐỌC IMAGES.TXT ----------
def parse_images_txt(fp):
    """Trả về list dict: {name, qvec, tvec, cam_id}"""
    images = []
    with open(fp, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = " ".join(parts[9:])
        # dòng kế tiếp là POINTS2D -> bỏ qua
        i += 2 if i+1 < len(lines) else 1
        images.append(dict(id=image_id, qvec=[qw,qx,qy,qz], tvec=[tx,ty,tz],
                           cam_id=cam_id, name=name))
    # Giữ nguyên thứ tự theo id tăng
    images.sort(key=lambda x: x["id"])
    return images

# ---------- CHỌN SPARSE TỐT NHẤT ----------
def choose_best_sparse(sparse_dir: Path) -> Optional[Path]:
    candidates = []
    for sub in sorted(sparse_dir.iterdir()):
        if not sub.is_dir(): 
            continue
        p3d = sub / "points3D.bin"
        img = sub / "images.bin"
        if p3d.exists() or img.exists():
            size_p3d = p3d.stat().st_size if p3d.exists() else 0
            size_img = img.stat().st_size if img.exists() else 0
            score = (size_p3d, size_img)   # ưu tiên points3D lớn
            candidates.append((score, sub))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]

# ---------- BUILD TRANSFORMS.JSON ----------
def build_transforms(obj_dir: Path, model_txt_dir: Path, cams, imgs):
    # Detect ảnh ext
    img_dir = obj_dir / "images"
    exts = ["png","jpg","jpeg","PNG","JPG","JPEG"]
    available = {p.name for ext in exts for p in img_dir.glob(f"*.{ext}")}
    if not available:
        print(f"[WARN] Không tìm thấy ảnh trong {img_dir}")
    # camera_angle_x từ fx & width (NeRF paper)
    # chọn camera đầu tiên
    any_cam = cams[imgs[0]["cam_id"]]
    fx = any_cam["fx"]; width = any_cam["width"]
    camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))

    frames = []
    for im in imgs:
        R = qvec2rotmat(im["qvec"])
        t = im["tvec"]
        # COLMAP cho world-to-cam: [R|t], ta cần cam-to-world
        Rc2w, tc2w = invert_pose(R, t)

        # Convert sang OpenGL/NeRF (flip trục y & z)
        # (giống các colmap2nerf phổ biến)
        F = [[1,0,0],[0,-1,0],[0,0,-1]]
        Rc2w = matmul3(Rc2w, F)

        # 4x4
        T = compose_4x4(Rc2w, tc2w)

        fname = Path(im["name"]).name
        if fname not in available:
            # nếu ảnh đã rename 0001.png… thì map theo basename
            pass
        frames.append({
            "file_path": f"./images/{fname}",
            "transform_matrix": T,
        })

    out = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }
    return out

# ---------- MAIN ----------
def main():
    if not os.path.exists(COLMAP):
        print(f"[ERR] Không tìm thấy COLMAP tại: {COLMAP}")
        sys.exit(1)

    root = Path(ROOT)
    for obj in sorted(root.iterdir()):
        if not obj.is_dir(): 
            continue
        img_dir = obj / "images"
        if not img_dir.exists():
            continue

        colmap_dir = obj / "colmap"
        sparse_dir = colmap_dir / "sparse"
        if not sparse_dir.exists():
            print(f"[SKIP] {obj.name}: chưa có colmap/sparse/")
            continue

        best = choose_best_sparse(sparse_dir)
        if best is None:
            print(f"[SKIP] {obj.name}: không tìm thấy model hợp lệ.")
            continue

        print(f"\n=== {obj.name}: dùng model {best.name} ===")

        # 1) Convert BIN -> TXT (an toàn chạy lại)
        run_cmd([COLMAP, "model_converter",
                 "--input_path", str(best),
                 "--output_path", str(best),
                 "--output_type", "TXT"])

        cam_txt = best/"cameras.txt"
        img_txt = best/"images.txt"
        if not cam_txt.exists() or not img_txt.exists():
            print(f"[ERR] Thiếu TXT ở {best}")
            continue

        cams = parse_cameras_txt(cam_txt)
        imgs = parse_images_txt(img_txt)
        if not imgs:
            print(f"[ERR] Không đọc được images.txt ở {best}")
            continue

        transforms = build_transforms(obj, best, cams, imgs)
        out_path = obj/"transforms.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transforms, f, indent=2)
        print(f"[OK] Saved {out_path}")

if __name__ == "__main__":
    main()
