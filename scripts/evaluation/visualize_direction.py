# visualize_transforms_fixed.py
# - Compute object center from cam1 (frames 1..90)
# - Plot camera positions, object center, and translucent view-direction lines
# - Save result to /mnt/data/transforms_view_dirs_fixed.png
#
# Requires: matplotlib, numpy

import json, math
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

TRANSFORM_PATH = Path(r"E:\LET_ME_COOK\Captone\PixelNerf_finetuning\dataset_pottery_ver3\bat_gom_1\transforms.json")   # <-- your file
OUT_PNG = TRANSFORM_PATH.with_name("transforms_view_dirs_fixed.png")

def extract_position(mat):
    # mat: 4x4 list-of-lists
    # Prefer translation in last column if last row == [0,0,0,1]
    if all(abs(mat[3][i] - v) < 1e-6 for i, v in enumerate([0,0,0,1])):
        return (float(mat[0][3]), float(mat[1][3]), float(mat[2][3]))
    # else try translation in last row
    if abs(mat[3][3] - 1.0) < 1e-6:
        return (float(mat[3][0]), float(mat[3][1]), float(mat[3][2]))
    # fallback: column
    return (float(mat[0][3]), float(mat[1][3]), float(mat[2][3]))

# Load transforms
data = json.load(open(TRANSFORM_PATH, "r", encoding="utf-8"))
frames = data.get("frames", [])
if len(frames) == 0:
    raise SystemExit("No frames in transforms.json")

# Extract positions
positions = [extract_position(fr["transform_matrix"]) for fr in frames]
pos_arr = np.array(positions)

# Define cam1 indices (frames 1..90 -> indices 0..89)
n = len(frames)
cam1_cnt = min(90, n)
cam1_idx = list(range(0, cam1_cnt))
cam2_idx = list(range(cam1_cnt, min(2*cam1_cnt, n)))

# Compute object center from cam1 positions (mean)
cam1_positions = pos_arr[cam1_idx]
center = cam1_positions.mean(axis=0)
center_arr = center

# Prepare 3D plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Plot cam1 positions
if len(cam1_idx) > 0:
    p1 = pos_arr[cam1_idx]
    ax.scatter(p1[:,0], p1[:,2], p1[:,1], marker='o', s=40, label='cam1 positions')

# Plot cam2 positions
if len(cam2_idx) > 0:
    p2 = pos_arr[cam2_idx]
    ax.scatter(p2[:,0], p2[:,2], p2[:,1], marker='^', s=40, label='cam2 positions')

# Plot object center (from cam1)
ax.scatter([center_arr[0]], [center_arr[2]], [center_arr[1]],
           marker='*', s=220, color='red', label='object center (cam1 mean)')

# Draw translucent view-direction lines from each camera toward center
for i, p in enumerate(pos_arr):
    px, py, pz = p[0], p[1], p[2]
    vx, vy, vz = center_arr[0]-px, center_arr[1]-py, center_arr[2]-pz
    # line coordinates (swap for plotting: X, Z, Y)
    line_x = [px, center_arr[0]]
    line_y = [pz, center_arr[2]]
    line_z = [py, center_arr[1]]
    ax.plot(line_x, line_y, line_z, linewidth=0.9, alpha=0.18, color='gray')

    # small arrow to indicate direction (quiver)
    vec = np.array([vx, vy, vz], dtype=float)
    dist = np.linalg.norm(vec)
    if dist > 1e-6:
        # arrow components in plotting coordinates: dx = x->center, dz = z->center, dy = y->center
        # normalized small arrow
        head_len = min(dist * 0.15, max(0.3, dist*0.05))
        dirn = vec / dist * head_len
        # quiver params: (x,y,z, u,v,w) but our plotting axes mapping uses (x,z,y) ordering
        ax.quiver(px, pz, py, dirn[0], dirn[2], dirn[1], length=1.0, normalize=False, linewidth=0.8, alpha=0.65)

# Annotate a few sample frames to help debugging (first, mid, last)
sample_idxs = [0, max(0, cam1_cnt//2), cam1_cnt-1, cam1_cnt, cam1_cnt + cam1_cnt//2, min(n-1, 2*cam1_cnt-1)]
for idx in sample_idxs:
    if idx < n:
        p = pos_arr[idx]
        ax.text(p[0], p[2], p[1], f'{idx+1}', size=8, zorder=10)

# Axis labels and aspect
ax.set_xlabel('X (world)')
ax.set_ylabel('Z (world)')
ax.set_zlabel('Y (world)')
ax.set_title('Camera positions, object center (cam1 mean), and view directions')

# Equal-ish aspect ratio
max_range = max(pos_arr[:,0].max()-pos_arr[:,0].min(),
                pos_arr[:,2].max()-pos_arr[:,2].min(),
                pos_arr[:,1].max()-pos_arr[:,1].min())
mid_x = 0.5*(pos_arr[:,0].max()+pos_arr[:,0].min())
mid_y = 0.5*(pos_arr[:,2].max()+pos_arr[:,2].min())
mid_z = 0.5*(pos_arr[:,1].max()+pos_arr[:,1].min())
ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

ax.legend()
plt.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print("Saved fixed visualization to:", OUT_PNG)
plt.show()
