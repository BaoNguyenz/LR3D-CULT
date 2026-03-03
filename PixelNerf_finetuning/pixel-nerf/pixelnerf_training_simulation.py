"""
============================================================================
MÔ PHỎNG QUÁ TRÌNH TRAINING PIXELNERF
============================================================================

Config:
- Input views (NV): 3
- Batch size (SB): 4
- Image size: 128x128
- Ray batch size: 128 (số rays sample mỗi iteration)

Giải thích chi tiết cách PixelNeRF tính ray và training.
"""

import numpy as np
import math

# ============================================================================
# STEP 1: KHỞI TẠO CÁC THAM SỐ
# ============================================================================

print("=" * 80)
print("STEP 1: KHỞI TẠO THAM SỐ")
print("=" * 80)

# Config
BATCH_SIZE = 4          # SB: số objects trong 1 batch
N_VIEWS = 3             # NV: số input views
IMAGE_SIZE = 128        # H = W = 128
RAY_BATCH_SIZE = 128    # Số rays sample mỗi iteration
FOCAL_LENGTH = 177.78   # fl = 711.12 / 4 (scaled từ 512 -> 128)
Z_NEAR = 0.5
Z_FAR = 6.0
N_COARSE = 64           # Số sample points trên mỗi ray (coarse)
N_FINE = 32             # Số sample points (fine)

print(f"""
Batch size (SB)     : {BATCH_SIZE} objects
Input views (NV)    : {N_VIEWS} views/object
Image size          : {IMAGE_SIZE}x{IMAGE_SIZE}
Ray batch size      : {RAY_BATCH_SIZE}
Focal length        : {FOCAL_LENGTH:.2f}
z_near, z_far       : {Z_NEAR}, {Z_FAR}
Coarse samples      : {N_COARSE}
Fine samples        : {N_FINE}
""")

# ============================================================================
# STEP 2: TẠO UNPROJECTION MAP
# ============================================================================

print("=" * 80)
print("STEP 2: TẠO UNPROJECTION MAP")
print("=" * 80)

def unproj_map(width, height, focal, c=None):
    """
    Tạo unprojection map: mỗi pixel (x, y) -> unit vector trong camera space.
    
    Công thức:
        X = (x - cx) / fx
        Y = -(y - cy) / fy  (âm vì trục Y ngược)
        Z = -1              (camera nhìn theo -Z)
        
    Sau đó normalize thành unit vector.
    """
    if c is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = c
    
    fx = fy = focal
    
    # Tạo grid tọa độ pixel
    Y, X = np.meshgrid(
        np.arange(height, dtype=np.float32) - cy,
        np.arange(width, dtype=np.float32) - cx,
        indexing='ij'
    )
    
    # Chuyển sang camera space
    X = X / fx
    Y = -Y / fy  # Âm vì trục Y ngược
    Z = -np.ones_like(X)  # Camera nhìn theo -Z
    
    # Stack và normalize
    unproj = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    unproj = unproj / np.linalg.norm(unproj, axis=-1, keepdims=True)
    
    return unproj

# Tạo unprojection map
cam_unproj_map = unproj_map(IMAGE_SIZE, IMAGE_SIZE, FOCAL_LENGTH)

print(f"""
Unprojection map shape: {cam_unproj_map.shape}
Giải thích: Mỗi pixel (i, j) ánh xạ thành 1 unit vector hướng ray trong camera space.

Ví dụ một số pixels:
  - Center pixel ({IMAGE_SIZE//2}, {IMAGE_SIZE//2}): {cam_unproj_map[IMAGE_SIZE//2, IMAGE_SIZE//2]}
    → Hướng thẳng (0, 0, -1) vì ở tâm ảnh
    
  - Top-left (0, 0): {cam_unproj_map[0, 0]}
    → Hướng lệch sang trái-trên
    
  - Bottom-right ({IMAGE_SIZE-1}, {IMAGE_SIZE-1}): {cam_unproj_map[-1, -1]}
    → Hướng lệch sang phải-dưới
""")

# ============================================================================
# STEP 3: TẠO RAYS TỪ CAMERA POSE
# ============================================================================

print("=" * 80)
print("STEP 3: TẠO RAYS TỪ CAMERA POSE")
print("=" * 80)

def gen_rays(pose, width, height, focal, z_near, z_far, c=None):
    """
    Tạo rays từ camera pose.
    
    Input:
        - pose: (4, 4) camera-to-world transformation matrix
        
    Output:
        - rays: (H, W, 8) = [origin_x, origin_y, origin_z, 
                            dir_x, dir_y, dir_z,
                            near, far]
    
    Công thức:
        1. Camera origin = pose[:3, 3] (translation phần)
        2. Ray direction = pose[:3, :3] @ unproj_map (rotation phần)
    """
    unproj = unproj_map(width, height, focal, c)  # (H, W, 3)
    
    # Camera origin (vị trí camera trong world space)
    cam_origin = pose[:3, 3]  # (3,)
    
    # Camera rotation matrix (camera-to-world)
    cam_rot = pose[:3, :3]  # (3, 3)
    
    # Transform ray directions từ camera space sang world space
    # ray_dir[i,j] = R @ unproj[i,j]
    ray_dirs = np.einsum('ij,hwj->hwi', cam_rot, unproj)  # (H, W, 3)
    
    # Camera origins cho tất cả pixels (giống nhau)
    cam_origins = np.broadcast_to(cam_origin, (height, width, 3))
    
    # Near và far bounds
    nears = np.full((height, width, 1), z_near)
    fars = np.full((height, width, 1), z_far)
    
    # Concat thành rays: [origin(3), direction(3), near(1), far(1)]
    rays = np.concatenate([cam_origins, ray_dirs, nears, fars], axis=-1)
    
    return rays  # (H, W, 8)

# Ví dụ với 1 camera pose (identity + translation)
example_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 4],  # Camera ở vị trí (0, 0, 4)
    [0, 0, 0, 1]
], dtype=np.float32)

rays_example = gen_rays(example_pose, IMAGE_SIZE, IMAGE_SIZE, FOCAL_LENGTH, Z_NEAR, Z_FAR)

print(f"""
Camera pose:
{example_pose}

Rays shape: {rays_example.shape} = (H, W, 8)

Ray format: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, near, far]

Ví dụ rays:
  - Center ray: {rays_example[IMAGE_SIZE//2, IMAGE_SIZE//2]}
    → Origin: (0, 0, 4), Direction: (0, 0, -1) - hướng thẳng vào object
    
  - Corner ray: {rays_example[0, 0]}
    → Origin: (0, 0, 4), Direction: hướng lệch về góc

Tổng số rays cho 1 ảnh: {IMAGE_SIZE * IMAGE_SIZE} = {IMAGE_SIZE}×{IMAGE_SIZE}
""")

# ============================================================================
# STEP 4: SAMPLE RAYS CHO TRAINING
# ============================================================================

print("=" * 80)
print("STEP 4: SAMPLE RAYS CHO TRAINING")
print("=" * 80)

def sample_rays_for_training(all_images, all_poses, all_rays, n_views, ray_batch_size):
    """
    Sample random rays từ tất cả views để training.
    
    Trong training, ta KHÔNG render toàn bộ ảnh (quá tốn memory),
    mà chỉ sample một số random rays.
    """
    SB, NV, C, H, W = all_images.shape
    
    # Với mỗi object, chọn ngẫu nhiên n_views làm input
    src_view_indices = []
    for obj_idx in range(SB):
        indices = np.random.choice(NV, n_views, replace=False)
        src_view_indices.append(indices)
    
    # Sample random pixel indices
    total_pixels = NV * H * W
    sampled_indices = np.random.randint(0, total_pixels, ray_batch_size)
    
    # Convert flat index -> (view_idx, row, col)
    view_idx = sampled_indices // (H * W)
    pixel_idx = sampled_indices % (H * W)
    row_idx = pixel_idx // W
    col_idx = pixel_idx % W
    
    return {
        'src_view_indices': np.array(src_view_indices),
        'sampled_rays': {
            'view_idx': view_idx,
            'row_idx': row_idx,
            'col_idx': col_idx
        }
    }

# Simulate
print(f"""
Training sample strategy:

1. Cho mỗi object trong batch (SB={BATCH_SIZE}):
   - Chọn ngẫu nhiên {N_VIEWS} views làm INPUT (để encode features)
   - Chọn ngẫu nhiên {RAY_BATCH_SIZE} rays từ TẤT CẢ views làm TARGET

2. Input views: Dùng để extract image features qua CNN encoder
3. Target rays: Dùng để tính loss giữa rendered color và ground truth

Memory efficient:
   - Không render toàn bộ {IMAGE_SIZE}×{IMAGE_SIZE} = {IMAGE_SIZE*IMAGE_SIZE} pixels
   - Chỉ render {RAY_BATCH_SIZE} rays mỗi iteration
""")

# ============================================================================
# STEP 5: ENCODING INPUT VIEWS
# ============================================================================

print("=" * 80)
print("STEP 5: ENCODING INPUT VIEWS")
print("=" * 80)

print(f"""
CNN Encoder (ResNet-34 backbone):

Input:  src_images shape: (SB, NV, 3, H, W) = ({BATCH_SIZE}, {N_VIEWS}, 3, {IMAGE_SIZE}, {IMAGE_SIZE})

Process:
  1. Reshape: (SB×NV, 3, H, W) = ({BATCH_SIZE * N_VIEWS}, 3, {IMAGE_SIZE}, {IMAGE_SIZE})
  2. Qua ResNet-34 backbone
  3. Output feature maps tại nhiều scales:
     - Layer 1: ({BATCH_SIZE * N_VIEWS}, 64, {IMAGE_SIZE//2}, {IMAGE_SIZE//2})
     - Layer 2: ({BATCH_SIZE * N_VIEWS}, 128, {IMAGE_SIZE//4}, {IMAGE_SIZE//4})
     - Layer 3: ({BATCH_SIZE * N_VIEWS}, 256, {IMAGE_SIZE//8}, {IMAGE_SIZE//8})
     - Layer 4: ({BATCH_SIZE * N_VIEWS}, 512, {IMAGE_SIZE//16}, {IMAGE_SIZE//16})

Latent size: 512 (sau khi combine các scales)

Encoder KHÔNG dùng để generate rays, mà dùng để:
  - Extract visual features từ input images
  - Các features này sẽ được QUERY tại vị trí 3D points khi rendering
""")

# ============================================================================
# STEP 6: RENDERING - SAMPLE POINTS ALONG RAYS
# ============================================================================

print("=" * 80)
print("STEP 6: RENDERING - SAMPLE POINTS ALONG RAY")
print("=" * 80)

def sample_points_along_ray(ray, n_samples):
    """
    Sample điểm dọc theo ray sử dụng stratified sampling.
    
    ray: [origin(3), direction(3), near(1), far(1)]
    
    Công thức:
        point = origin + t * direction
        với t ∈ [near, far]
    """
    origin = ray[:3]
    direction = ray[3:6]
    near = ray[6]
    far = ray[7]
    
    # Stratified sampling: chia [near, far] thành n_samples bins
    # Sample 1 điểm random trong mỗi bin
    t_vals = np.linspace(near, far, n_samples + 1)
    t_samples = []
    
    for i in range(n_samples):
        t_lo = t_vals[i]
        t_hi = t_vals[i + 1]
        t = np.random.uniform(t_lo, t_hi)
        t_samples.append(t)
    
    t_samples = np.array(t_samples)
    
    # Tính 3D points
    points = origin[None, :] + t_samples[:, None] * direction[None, :]
    
    return t_samples, points

# Ví dụ
example_ray = rays_example[IMAGE_SIZE//2, IMAGE_SIZE//2]
t_samples, sampled_points = sample_points_along_ray(example_ray, N_COARSE)

print(f"""
Stratified Sampling trên 1 ray:

Ray: origin={example_ray[:3]}, direction={example_ray[3:6]}
Near={example_ray[6]}, Far={example_ray[7]}

Chia đoạn [near, far] = [{Z_NEAR}, {Z_FAR}] thành {N_COARSE} bins
Sample 1 điểm random trong mỗi bin → {N_COARSE} points

t values (distance từ camera):
  First 5: {t_samples[:5]}
  Last 5:  {t_samples[-5:]}

3D points (x, y, z):
  First 3: 
{sampled_points[:3]}
  Last 3:
{sampled_points[-3:]}

Tổng số 3D queries cho 1 batch:
  = SB × ray_batch_size × n_coarse
  = {BATCH_SIZE} × {RAY_BATCH_SIZE} × {N_COARSE}
  = {BATCH_SIZE * RAY_BATCH_SIZE * N_COARSE} points
""")

# ============================================================================
# STEP 7: QUERY FEATURES VÀ MLP PREDICTION
# ============================================================================

print("=" * 80)
print("STEP 7: QUERY FEATURES VÀ MLP PREDICTION")
print("=" * 80)

print(f"""
Với mỗi 3D point, PixelNeRF cần:

1. PROJECT point về input images:
   - Transform point từ world space → camera space của mỗi input view
   - Project: uv = -xy/z * focal + center
   
2. SAMPLE feature từ feature maps:
   - Bilinear interpolation tại vị trí (u, v)
   - Kết quả: latent vector (512,) cho mỗi input view
   
3. AGGREGATE features từ nhiều views:
   - Average pooling across {N_VIEWS} views
   - Output: (512,)

4. MLP prediction:
   Input:  latent (512) + positional_encoding(xyz) (39)
   Hidden: 4 layers, 512 units
   Output: (r, g, b, σ) = (4,)

Shape flow cho 1 iteration:
  - Input 3D points: ({BATCH_SIZE * RAY_BATCH_SIZE * N_COARSE}, 3)
  - After projection: ({BATCH_SIZE * RAY_BATCH_SIZE * N_COARSE * N_VIEWS}, 2)
  - Sampled features: ({BATCH_SIZE * RAY_BATCH_SIZE * N_COARSE * N_VIEWS}, 512)
  - After aggregation: ({BATCH_SIZE * RAY_BATCH_SIZE * N_COARSE}, 512)
  - MLP output: ({BATCH_SIZE * RAY_BATCH_SIZE * N_COARSE}, 4)
""")

# ============================================================================
# STEP 8: VOLUME RENDERING
# ============================================================================

print("=" * 80)
print("STEP 8: VOLUME RENDERING")
print("=" * 80)

def volume_rendering(rgbs, sigmas, t_vals):
    """
    NeRF volume rendering equation.
    
    Công thức:
        C = Σ T_i × α_i × c_i
        
    Với:
        - δ_i = t_{i+1} - t_i (khoảng cách giữa 2 samples)
        - α_i = 1 - exp(-σ_i × δ_i)  (opacity)
        - T_i = Π_{j<i} (1 - α_j)    (transmittance)
    """
    n_samples = len(t_vals)
    
    # Tính deltas (khoảng cách giữa các samples)
    deltas = np.concatenate([
        t_vals[1:] - t_vals[:-1],
        [1e10]  # Infinity cho sample cuối
    ])
    
    # Tính alphas (opacity)
    alphas = 1 - np.exp(-sigmas * deltas)
    
    # Tính transmittance
    # T_i = (1-α_0) × (1-α_1) × ... × (1-α_{i-1})
    T = np.cumprod(np.concatenate([[1.0], 1 - alphas[:-1] + 1e-10]))
    
    # Weights = T × α
    weights = T * alphas
    
    # Final color = weighted sum of colors
    rgb_final = np.sum(weights[:, None] * rgbs, axis=0)
    
    # Depth = weighted sum of t values
    depth_final = np.sum(weights * t_vals)
    
    return rgb_final, depth_final, weights

# Simulate với random values
np.random.seed(42)
fake_rgbs = np.random.rand(N_COARSE, 3)  # Random colors
fake_sigmas = np.random.rand(N_COARSE) * 10  # Random densities

rgb_rendered, depth_rendered, weights = volume_rendering(fake_rgbs, fake_sigmas, t_samples)

print(f"""
Volume Rendering (Alpha Compositing):

Với mỗi ray, ta có {N_COARSE} samples, mỗi sample có (r,g,b,σ)

Bước 1: Tính δ_i = t_{{i+1}} - t_i
Bước 2: Tính α_i = 1 - exp(-σ_i × δ_i)   [opacity của sample i]
Bước 3: Tính T_i = Π(1 - α_j) for j < i  [xác suất ray chưa bị chặn]
Bước 4: Tính weights w_i = T_i × α_i
Bước 5: Final color C = Σ w_i × c_i
Bước 6: Final depth D = Σ w_i × t_i

Ví dụ kết quả:
  - Rendered RGB: {rgb_rendered}
  - Rendered Depth: {depth_rendered:.4f}
  - Sum of weights: {weights.sum():.4f} (should be ~1.0 for opaque)
""")

# ============================================================================
# STEP 9: LOSS VÀ BACKPROP
# ============================================================================

print("=" * 80)
print("STEP 9: LOSS VÀ BACKPROP")
print("=" * 80)

print(f"""
Loss function: L2 Loss (MSE)

Loss = ||RGB_predicted - RGB_groundtruth||²

Với coarse + fine sampling:
  Total Loss = λ_coarse × L_coarse + λ_fine × L_fine
  
Thường: λ_coarse = 1.0, λ_fine = 1.0

Backprop flow:
  Loss → MLP weights
       → CNN encoder weights (nếu không freeze)
       
Optimizer: Adam với lr = 1e-4

Mỗi iteration:
  - Forward: {BATCH_SIZE} objects × {RAY_BATCH_SIZE} rays = {BATCH_SIZE * RAY_BATCH_SIZE} rays
  - Backward: Gradient qua MLP + Encoder
""")

# ============================================================================
# STEP 10: TỔNG KẾT PIPELINE
# ============================================================================

print("=" * 80)
print("STEP 10: TỔNG KẾT TRAINING PIPELINE")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PIXELNERF TRAINING PIPELINE SUMMARY                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INPUT:                                                                      ║
║    • Batch: {BATCH_SIZE} objects                                                        ║
║    • Mỗi object: {N_VIEWS} input views (128×128×3)                                   ║
║    • Camera poses + focal lengths                                            ║
║                                                                              ║
║  STEP 1: ENCODE INPUT IMAGES                                                 ║
║    • ResNet-34 encoder                                                       ║
║    • Input: ({BATCH_SIZE}×{N_VIEWS}, 3, 128, 128)                                            ║
║    • Output: Feature maps tại nhiều scales                                   ║
║                                                                              ║
║  STEP 2: GENERATE RAYS                                                       ║
║    • Tạo unprojection map từ focal length                                    ║
║    • Transform với camera pose → world space rays                            ║
║    • Sample {RAY_BATCH_SIZE} random rays/object                                       ║
║                                                                              ║
║  STEP 3: SAMPLE POINTS ALONG RAYS                                            ║
║    • Stratified sampling: {N_COARSE} coarse + {N_FINE} fine points/ray                 ║
║    • Total: {BATCH_SIZE}×{RAY_BATCH_SIZE}×{N_COARSE} = {BATCH_SIZE * RAY_BATCH_SIZE * N_COARSE:,} 3D points                          ║
║                                                                              ║
║  STEP 4: QUERY FEATURES                                                      ║
║    • Project 3D points → 2D locations trên input images                      ║
║    • Bilinear sample features từ feature maps                                ║
║    • Aggregate features across {N_VIEWS} views                                      ║
║                                                                              ║
║  STEP 5: MLP PREDICTION                                                      ║
║    • Input: features + positional encoding                                   ║
║    • Output: (r, g, b, σ) cho mỗi 3D point                                   ║
║                                                                              ║
║  STEP 6: VOLUME RENDERING                                                    ║
║    • Alpha compositing: C = Σ T_i × α_i × c_i                                ║
║    • Output: RGB color + depth cho mỗi ray                                   ║
║                                                                              ║
║  STEP 7: COMPUTE LOSS & BACKPROP                                             ║
║    • L2 loss với ground truth pixel colors                                   ║
║    • Backprop → Update MLP + Encoder weights                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY INSIGHT - KHÁC BIỆT VỚI VANILLA NERF:

1. VANILLA NERF:
   - Train 1 MLP riêng cho MỖI scene
   - Không có encoder
   - Input: chỉ có (x, y, z, viewdir)
   
2. PIXELNERF:
   - 1 model cho TẤT CẢ scenes
   - CNN encoder extract features từ input views
   - Input: (x, y, z, viewdir) + IMAGE FEATURES
   - → Có thể generalize sang unseen objects!
""")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)

