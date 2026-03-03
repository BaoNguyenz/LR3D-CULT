import os, json, numpy as np

# ======= THAY ĐƯỜNG DẪN NÀY THEO MÁY BẠN =======
ROOT = r"E:\LET ME COOK\Captone\NeRF_finetuning\All_data"
# =================================================

def read_cameras_txt(path):
    """Đọc fx, fy, cx, cy từ cameras.txt"""
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if not l.startswith("#")]
    if not lines:
        raise ValueError("Không tìm thấy cameras.txt hoặc file rỗng")
    parts = lines[0].split()
    cam_model = parts[1]
    params = list(map(float, parts[4:]))
    if "SIMPLE_PINHOLE" in cam_model:
        fx = fy = params[0]; cx, cy = params[1:3]
    elif "PINHOLE" in cam_model:
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Không hỗ trợ camera model {cam_model}")
    return fx, fy, cx, cy

def qvec2rotmat(qvec):
    """Chuyển quaternion thành ma trận quay 3x3"""
    q = np.array(qvec, dtype=np.float64)
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

def read_images_txt(path):
    """Đọc pose từng ảnh (world-to-camera)"""
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        if not lines[i]: continue
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = list(map(float, parts[1:5]))
        tvec = list(map(float, parts[5:8]))
        file_name = parts[9]
        images[image_id] = {"qvec": qvec, "tvec": tvec, "file_name": file_name}
    return images

def save_transforms_json(obj_dir):
    sparse_dir = os.path.join(obj_dir, "colmap", "sparse", "0")
    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    images_path = os.path.join(sparse_dir, "images.txt")
    if not (os.path.exists(cameras_path) and os.path.exists(images_path)):
        print(f"[BỎ QUA] {obj_dir} - chưa có dữ liệu COLMAP")
        return

    fx, fy, cx, cy = read_cameras_txt(cameras_path)
    images = read_images_txt(images_path)

    frames = []
    for img in images.values():
        R = qvec2rotmat(img["qvec"])
        t = np.array(img["tvec"]).reshape(3, 1)
        Rt = np.concatenate([R, t], axis=1)
        Rt = np.vstack([Rt, np.array([0,0,0,1])])
        c2w = np.linalg.inv(Rt)
        c2w[:,1:3] *= -1  # chuyển hệ trục từ COLMAP (OpenCV) sang OpenGL/NGP

        frame = {
            "file_path": f"images/{img['file_name']}",
            "transform_matrix": c2w.tolist(),
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy
        }
        frames.append(frame)

    out = {"camera_model": "PINHOLE", "frames": frames}
    out_path = os.path.join(obj_dir, "transforms.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] {obj_dir} → transforms.json ({len(frames)} ảnh)")

def main():
    for name in os.listdir(ROOT):
        obj_path = os.path.join(ROOT, name)
        if os.path.isdir(obj_path):
            save_transforms_json(obj_path)

if __name__ == "__main__":
    main()
