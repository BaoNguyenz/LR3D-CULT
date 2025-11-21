#!/usr/bin/env python3
"""
Prepare dataset for PixelNeRF.

- Pick N_FRAMES evenly from source images (spread across 0..360).
- Resize images (optional).
- Copy images directly into each object folder (NOT images/ subfolder).
- Create transforms.json containing 'camera_angle_x' and frames list, where each frame:
    { "file_path": "./0001.png", "w": W, "h": H, "transform_matrix": [[...],...] }
- Split objects into train/val/test by ratio (default 0.8/0.1/0.1).
- Skip objects with fewer than N_FRAMES images.

Usage example:
python prepare_pixelnerf_dataset.py \
  --src "E:/LET_ME_COOK/Captone/PixelNerf_finetuning/all_objects" \
  --dst "E:/LET_ME_COOK/Captone/PixelNerf_finetuning/dataset_pixelnerf_nv90" \
  --n 90 --resize 128 128 --fov 50 --dist 1.5 --elev 0 --seed 42
"""
import os, sys, json, math, random, argparse
from pathlib import Path
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg"}

def list_images(d: Path):
    if not d.exists(): return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def pick_even_indices(L, n):
    if n == 1: return [0]
    return [int(round(i*(L-1)/(n-1))) for i in range(n)]

def make_cam2world_y(theta_rad, dist=1.5, elev_rad=0.0):
    # camera pos in spherical coords
    x = dist * math.cos(elev_rad) * math.sin(theta_rad)
    y = dist * math.sin(elev_rad)
    z = dist * math.cos(elev_rad) * math.cos(theta_rad)
    pos = [x,y,z]
    fx,fy,fz = -x,-y,-z
    fn = math.sqrt(fx*fx+fy*fy+fz*fz)+1e-9
    forward = [fx/fn, fy/fn, fz/fn]
    up_world = [0.0,1.0,0.0]
    # right = up x forward
    rx = up_world[1]*forward[2] - up_world[2]*forward[1]
    ry = up_world[2]*forward[0] - up_world[0]*forward[2]
    rz = up_world[0]*forward[1] - up_world[1]*forward[0]
    rn = math.sqrt(rx*rx+ry*ry+rz*rz)+1e-9
    right = [rx/rn, ry/rn, rz/rn]
    # up = forward x right
    ux = forward[1]*right[2] - forward[2]*right[1]
    uy = forward[2]*right[0] - forward[0]*right[2]
    uz = forward[0]*right[1] - forward[1]*right[0]
    mat = [
        [right[0], ux, forward[0], pos[0]],
        [right[1], uy, forward[1], pos[1]],
        [right[2], uz, forward[2], pos[2]],
        [0.0,      0.0, 0.0,       1.0]
    ]
    return mat

def prepare_obj(src_obj, dst_obj, n_frames, resize_wh, fov_deg, dist, elev_deg):
    # support images either in src_obj/images/ or directly in src_obj/
    images_dir = src_obj / "images"
    imgs = list_images(images_dir) if images_dir.exists() else list_images(src_obj)
    if len(imgs) < n_frames:
        return False, f"not enough images ({len(imgs)})"
    idxs = pick_even_indices(len(imgs), n_frames)
    dst_obj.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, idx in enumerate(idxs, start=1):
        src_img = imgs[idx]
        ext = src_img.suffix.lower()
        newname = f"{i:04d}{ext}"
        dst_path = dst_obj / newname
        im = Image.open(src_img).convert("RGB")
        if resize_wh is not None:
            im = im.resize(resize_wh, Image.LANCZOS)
        im.save(dst_path)
        frames.append({"file_path": f"./{newname}", "w": im.width, "h": im.height})
    # build transforms.json
    cam_angle_x = math.radians(fov_deg)
    out = {"camera_angle_x": cam_angle_x, "frames": []}
    elev_rad = math.radians(elev_deg)
    for i in range(n_frames):
        theta = 2.0*math.pi*(i) / n_frames
        mat = make_cam2world_y(theta, dist=dist, elev_rad=elev_rad)
        fr = dict(frames[i])
        fr["transform_matrix"] = mat
        out["frames"].append(fr)
    with open(dst_obj / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return True, f"ok, copied {len(frames)} images"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--n", type=int, default=90, help="NV per object")
    p.add_argument("--resize", nargs=2, type=int, default=[128,128], help="W H (0 0 to keep)")
    p.add_argument("--fov", type=float, default=50.0)
    p.add_argument("--dist", type=float, default=1.5)
    p.add_argument("--elev", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", nargs=3, type=float, default=[0.8,0.1,0.1])
    args = p.parse_args()

    SRC = Path(args.src)
    DST = Path(args.dst)
    N = args.n
    resize_wh = (args.resize[0], args.resize[1]) if not (args.resize[0]==0 and args.resize[1]==0) else None
    random.seed(args.seed)
    objs = sorted([d for d in SRC.iterdir() if d.is_dir()])
    good = []
    bad = []
    for d in objs:
        cnt = len(list_images(d / "images")) if (d / "images").exists() else len(list_images(d))
        if cnt >= N:
            good.append(d)
        else:
            bad.append((d, cnt))
    print(f"Found {len(objs)} objects; will use {len(good)} (skip {len(bad)})")
    # split
    random.shuffle(good)
    n = len(good)
    ntrain = int(round(n * args.split[0]))
    nval = int(round(n * args.split[1]))
    if ntrain + nval > n:
        nval = max(0, n - ntrain)
    ntest = n - ntrain - nval
    train = good[:ntrain]; val = good[ntrain:ntrain+nval]; test = good[ntrain+nval:]
    print("Split: train", len(train), "val", len(val), "test", len(test))

    for splitname, listobjs in [("train",train),("val",val),("test",test)]:
        for d in listobjs:
            ok, msg = prepare_obj(d, DST / splitname / d.name, N, resize_wh, args.fov, args.dist, args.elev)
            if not ok:
                print("  SKIP", d.name, msg)
            else:
                print("  DONE", d.name, msg)
    print("Done. Output at:", DST)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Prepare dataset for PixelNeRF.

- Pick N_FRAMES evenly from source images (spread across 0..360).
- Resize images (optional).
- Copy images directly into each object folder (NOT images/ subfolder).
- Create transforms.json containing 'camera_angle_x' and frames list, where each frame:
    { "file_path": "./0001.png", "w": W, "h": H, "transform_matrix": [[...],...] }
- Split objects into train/val/test by ratio (default 0.8/0.1/0.1).
- Skip objects with fewer than N_FRAMES images.

Usage example:
python prepare_pixelnerf_dataset.py \
  --src "E:/LET_ME_COOK/Captone/PixelNerf_finetuning/all_objects" \
  --dst "E:/LET_ME_COOK/Captone/PixelNerf_finetuning/dataset_pixelnerf_nv90" \
  --n 90 --resize 128 128 --fov 50 --dist 1.5 --elev 0 --seed 42
"""
import os, sys, json, math, random, argparse
from pathlib import Path
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg"}

def list_images(d: Path):
    if not d.exists(): return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def pick_even_indices(L, n):
    if n == 1: return [0]
    return [int(round(i*(L-1)/(n-1))) for i in range(n)]

def make_cam2world_y(theta_rad, dist=1.5, elev_rad=0.0):
    # camera pos in spherical coords
    x = dist * math.cos(elev_rad) * math.sin(theta_rad)
    y = dist * math.sin(elev_rad)
    z = dist * math.cos(elev_rad) * math.cos(theta_rad)
    pos = [x,y,z]
    fx,fy,fz = -x,-y,-z
    fn = math.sqrt(fx*fx+fy*fy+fz*fz)+1e-9
    forward = [fx/fn, fy/fn, fz/fn]
    up_world = [0.0,1.0,0.0]
    # right = up x forward
    rx = up_world[1]*forward[2] - up_world[2]*forward[1]
    ry = up_world[2]*forward[0] - up_world[0]*forward[2]
    rz = up_world[0]*forward[1] - up_world[1]*forward[0]
    rn = math.sqrt(rx*rx+ry*ry+rz*rz)+1e-9
    right = [rx/rn, ry/rn, rz/rn]
    # up = forward x right
    ux = forward[1]*right[2] - forward[2]*right[1]
    uy = forward[2]*right[0] - forward[0]*right[2]
    uz = forward[0]*right[1] - forward[1]*right[0]
    mat = [
        [right[0], ux, forward[0], pos[0]],
        [right[1], uy, forward[1], pos[1]],
        [right[2], uz, forward[2], pos[2]],
        [0.0,      0.0, 0.0,       1.0]
    ]
    return mat

def prepare_obj(src_obj, dst_obj, n_frames, resize_wh, fov_deg, dist, elev_deg):
    # support images either in src_obj/images/ or directly in src_obj/
    images_dir = src_obj / "images"
    imgs = list_images(images_dir) if images_dir.exists() else list_images(src_obj)
    if len(imgs) < n_frames:
        return False, f"not enough images ({len(imgs)})"
    idxs = pick_even_indices(len(imgs), n_frames)
    dst_obj.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, idx in enumerate(idxs, start=1):
        src_img = imgs[idx]
        ext = src_img.suffix.lower()
        newname = f"{i:04d}{ext}"
        dst_path = dst_obj / newname
        im = Image.open(src_img).convert("RGB")
        if resize_wh is not None:
            im = im.resize(resize_wh, Image.LANCZOS)
        im.save(dst_path)
        frames.append({"file_path": f"./{newname}", "w": im.width, "h": im.height})
    # build transforms.json
    cam_angle_x = math.radians(fov_deg)
    out = {"camera_angle_x": cam_angle_x, "frames": []}
    elev_rad = math.radians(elev_deg)
    for i in range(n_frames):
        theta = 2.0*math.pi*(i) / n_frames
        mat = make_cam2world_y(theta, dist=dist, elev_rad=elev_rad)
        fr = dict(frames[i])
        fr["transform_matrix"] = mat
        out["frames"].append(fr)
    with open(dst_obj / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return True, f"ok, copied {len(frames)} images"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--n", type=int, default=90, help="NV per object")
    p.add_argument("--resize", nargs=2, type=int, default=[128,128], help="W H (0 0 to keep)")
    p.add_argument("--fov", type=float, default=50.0)
    p.add_argument("--dist", type=float, default=1.5)
    p.add_argument("--elev", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", nargs=3, type=float, default=[0.8,0.1,0.1])
    args = p.parse_args()

    SRC = Path(args.src)
    DST = Path(args.dst)
    N = args.n
    resize_wh = (args.resize[0], args.resize[1]) if not (args.resize[0]==0 and args.resize[1]==0) else None
    random.seed(args.seed)
    objs = sorted([d for d in SRC.iterdir() if d.is_dir()])
    good = []
    bad = []
    for d in objs:
        cnt = len(list_images(d / "images")) if (d / "images").exists() else len(list_images(d))
        if cnt >= N:
            good.append(d)
        else:
            bad.append((d, cnt))
    print(f"Found {len(objs)} objects; will use {len(good)} (skip {len(bad)})")
    # split
    random.shuffle(good)
    n = len(good)
    ntrain = int(round(n * args.split[0]))
    nval = int(round(n * args.split[1]))
    if ntrain + nval > n:
        nval = max(0, n - ntrain)
    ntest = n - ntrain - nval
    train = good[:ntrain]; val = good[ntrain:ntrain+nval]; test = good[ntrain+nval:]
    print("Split: train", len(train), "val", len(val), "test", len(test))

    for splitname, listobjs in [("train",train),("val",val),("test",test)]:
        for d in listobjs:
            ok, msg = prepare_obj(d, DST / splitname / d.name, N, resize_wh, args.fov, args.dist, args.elev)
            if not ok:
                print("  SKIP", d.name, msg)
            else:
                print("  DONE", d.name, msg)
    print("Done. Output at:", DST)

if __name__ == "__main__":
    main()
