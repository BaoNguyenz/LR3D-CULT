import os
import sys
import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import warnings

# Fix OpenMP conflict on Windows (set before importing torch)
if sys.platform == 'win32':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

if sys.platform == "win32":
    # Allow OpenMP duplicates (common on Windows with torch + other libs)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    from pyhocon import ConfigFactory
except ImportError:
    ConfigFactory = None
    warnings.warn("pyhocon not installed. Config loading may fail.")

# Optional libraries for PLY export
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False

try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    PlyData = None
    PlyElement = None
    HAS_PLYFILE = False

# Import PixelNeRF modules
try:
    from model import make_model
    from render import NeRFRenderer
    import util
except ImportError as e:
    print(f"Error importing PixelNeRF modules: {e}")
    print("Make sure you're running from the repo root and src/ is in path")
    sys.exit(1)


# ============================================================================
# Utility functions
# ============================================================================

def load_json(path: Path) -> dict:
    """Load JSON file"""
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def camera_intrinsics_from_json(js: dict, frame: dict) -> Tuple[float, float, float, float, int, int]:
    """
    Extract camera intrinsics from transforms.json
    Returns: fx, fy, cx, cy, W, H
    """
    W = frame.get('w', js.get('w', None))
    H = frame.get('h', js.get('h', None))
    if W is None or H is None:
        raise ValueError("Image width/height not found in transforms.json")
    
    # Try explicit intrinsics first
    fx = frame.get('fl_x', frame.get('fx', None)) or js.get('fl_x', js.get('fx', None))
    fy = frame.get('fl_y', frame.get('fy', None)) or js.get('fl_y', js.get('fy', None))
    cx = frame.get('cx', None) or js.get('cx', None)
    cy = frame.get('cy', None) or js.get('cy', None)
    
    if fx and fy and cx and cy:
        return float(fx), float(fy), float(cx), float(cy), int(W), int(H)
    
    # Fallback to camera_angle_x
    cam_angle_x = js.get('camera_angle_x', None)
    if cam_angle_x is not None:
        fx = 0.5 * float(W) / math.tan(float(cam_angle_x) / 2.0)
        fy = fx
        cx = float(W) / 2.0
        cy = float(H) / 2.0
        return fx, fy, cx, cy, int(W), int(H)
    
    # Last resort: assume reasonable defaults
    fx = 0.5 * float(W)
    fy = 0.5 * float(W)
    cx = float(W) / 2.0
    cy = float(H) / 2.0
    return fx, fy, cx, cy, int(W), int(H)


def pose_to_tensor(frame: dict, device: torch.device) -> torch.Tensor:
    """Convert transform_matrix from JSON to torch tensor"""
    tm = frame.get('transform_matrix', None) or frame.get('transform', None)
    if tm is None:
        raise ValueError("frame missing 'transform_matrix'")
    arr = np.array(tm, dtype=np.float32)
    if arr.shape != (4, 4):
        arr = arr.reshape(4, 4)
    return torch.from_numpy(arr).to(device)


def load_image(image_path: Path, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Load image and convert to tensor format expected by PixelNeRF
    Returns: (3, H, W) tensor in range [-1, 1]
    """
    pil_img = Image.open(str(image_path)).convert('RGB')
    if size is not None:
        pil_img = pil_img.resize(size, Image.LANCZOS)
    img_array = np.array(pil_img).astype(np.float32) / 255.0
    # Convert to [-1, 1] range (PixelNeRF format)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
    img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor


def depth_to_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
    depth_mask: Optional[np.ndarray] = None,
    z_near: float = 0.1,
    z_far: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth map to point cloud
    Args:
        depth: (H, W) depth map
        rgb: (H, W, 3) RGB image
        fx, fy, cx, cy: camera intrinsics
        c2w: (4, 4) camera-to-world transformation matrix
        depth_mask: optional mask for valid depth pixels
    Returns:
        points: (N, 3) point cloud in world coordinates
        colors: (N, 3) RGB colors
    """
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    # Filter valid depth
    if depth_mask is None:
        depth_mask = np.isfinite(depth) & (depth > z_near) & (depth < z_far)
    
    # Convert to camera coordinates
    z = depth[depth_mask]
    x = (i[depth_mask] - cx) * z / fx
    y = -(j[depth_mask] - cy) * z / fy  # Negative y due to image coordinates
    
    pts_cam = np.stack([x, y, -z], axis=-1)  # (N, 3) in camera space
    
    # Transform to world coordinates
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = (pts_cam @ R.T) + t.reshape(1, 3)
    
    # Extract colors
    colors = rgb[depth_mask]  # (N, 3)
    
    return pts_world, colors


def save_ply_open3d(points: np.ndarray, colors: Optional[np.ndarray], out_path: Path):
    """Save PLY using open3d"""
    if not HAS_OPEN3D:
        raise RuntimeError("open3d not available")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1.01:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved PLY to {out_path} ({len(points)} points)")


def save_ply_plyfile(points: np.ndarray, colors: Optional[np.ndarray], out_path: Path):
    """Save PLY using plyfile library"""
    if not HAS_PLYFILE:
        raise RuntimeError("plyfile not available")
    
    points = points.reshape(-1, 3)
    if colors is not None:
        colors = colors.reshape(-1, 3)
        if colors.max() <= 1.01:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        vertex = np.array(
            [(float(x), float(y), float(z), int(r), int(g), int(b))
             for (x, y, z), (r, g, b) in zip(points, colors)],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
    else:
        vertex = np.array(
            [(float(x), float(y), float(z)) for x, y, z in points],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(str(out_path))
    print(f"Saved PLY to {out_path} ({len(points)} points)")


# ============================================================================
# Model loading
# ============================================================================

def load_model(
    weights_path: Path,
    config_path: Optional[Path],
    device: torch.device
) -> torch.nn.Module:
    """
    Load PixelNeRF model from checkpoint
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    # Load config
    if config_path is None or not config_path.exists():
        # Try default config
        default_configs = [
            Path("conf/default_mv.conf"),
            Path("conf/default.conf"),
        ]
        config_path = None
        for cfg in default_configs:
            if cfg.exists():
                config_path = cfg
                break
        
        if config_path is None:
            warnings.warn("No config file found. Model may not initialize correctly.")
            # Create minimal config
            if ConfigFactory is None:
                raise RuntimeError("pyhocon not installed and no config file found")
            conf = ConfigFactory.parse_string("""
                model {
                    encoder {
                        type = resnet
                        d_out = 256
                    }
                    use_encoder = true
                    use_xyz = false
                    normalize_z = true
                    use_code = true
                    use_code_viewdirs = true
                    use_viewdirs = false
                    use_global_encoder = false
                    mlp_coarse {
                        type = resnetfc
                        n_blocks = 5
                        d_hidden = 256
                        combine_layer = 3
                        combine_type = average
                    }
                    mlp_fine {
                        type = resnetfc
                        n_blocks = 5
                        d_hidden = 256
                        combine_layer = 3
                        combine_type = average
                    }
                }
                renderer {
                    n_coarse = 64
                    n_fine = 128
                    using_fine = true
                    white_bkgd = false
                }
            """)
        else:
            conf = ConfigFactory.parse_file(str(config_path))
    else:
        conf = ConfigFactory.parse_file(str(config_path))
    
    print(f"Loading model from {weights_path}")
    print(f"Using config: {config_path if config_path else 'default/minimal'}")
    
    # Create model
    model = make_model(conf["model"]).to(device)
    
    # Load weights
    checkpoint = torch.load(str(weights_path), map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, torch.nn.Module):
        model.load_state_dict(checkpoint.state_dict())
    elif isinstance(checkpoint, dict):
        # Try common keys
        state_dict = (
            checkpoint.get('state_dict') or
            checkpoint.get('model_state_dict') or
            checkpoint.get('net_state_dict') or
            checkpoint
        )
        # Remove 'module.' prefix if present (from DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:] if k.startswith('module.') else k: v
                         for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print("Model loaded successfully")
    return model, conf


# ============================================================================
# Main inference function
# ============================================================================

def render_depth_rgb(
    render_fn,
    pose: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    H: int,
    W: int,
    z_near: float = 0.1,
    z_far: float = 10.0,
    ray_batch_size: int = 50000,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render depth/RGB for a single target pose (assumes model.encode already run).
    """
    if device is None:
        raise ValueError("device must be provided for rendering")

    pose = pose.unsqueeze(0) if pose.dim() == 2 else pose
    focal = torch.tensor([[fx, fy]], device=device, dtype=torch.float32)
    c = torch.tensor([[cx, cy]], device=device, dtype=torch.float32)

    rays = util.gen_rays(
        pose,
        W,
        H,
        focal,
        z_near,
        z_far,
        c=c,
    ).reshape(1, H * W, 8).to(device)

    all_rgb = []
    all_depth = []

    with torch.no_grad():
        rays_flat = rays.reshape(-1, 8)
        for start in range(0, rays_flat.shape[0], ray_batch_size):
            ray_batch = rays_flat[start : start + ray_batch_size]
            rgb, depth = render_fn(ray_batch[None])
            all_rgb.append(rgb[0].cpu())
            all_depth.append(depth[0].cpu())

    rgb = torch.cat(all_rgb, dim=0).reshape(H, W, 3).numpy()
    depth = torch.cat(all_depth, dim=0).reshape(H, W).numpy()
    depth = depth * (z_far - z_near) + z_near

    return depth, rgb


# ============================================================================
# Main script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export PixelNeRF model to PLY point cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using images directory
  python export_ply.py --weights checkpoint.pth \\
      --images_dir dataset/test/object --transforms dataset/test/object/transforms.json \\
      --out result.ply --device cuda:0

  # Using specific images
  python export_ply.py --weights checkpoint.pth \\
      --images 0001.png,0012.png,0034.png --transforms transforms.json \\
      --out result.ply --device 0

  # With custom config
  python export_ply.py --weights checkpoint.pth \\
      --images_dir dataset/test/object --transforms transforms.json \\
      --out result.ply --config conf/default_mv.conf --device cuda:0
        """
    )
    
    parser.add_argument(
        '--weights', '-w',
        type=str,
        required=True,
        help='Path to model checkpoint/weights file (.pth)'
    )
    
    parser.add_argument(
        '--images', '-i',
        type=str,
        default=None,
        help='Comma-separated list of image files (e.g., "0001.png,0012.png,0034.png")'
    )
    
    parser.add_argument(
        '--images_dir', '-d',
        type=str,
        default=None,
        help='Directory containing images (will use frames from transforms.json)'
    )
    
    parser.add_argument(
        '--transforms', '-t',
        type=str,
        required=True,
        help='Path to transforms.json file'
    )
    
    parser.add_argument(
        '--out', '-o',
        type=str,
        required=True,
        help='Output PLY file path'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device: cpu, cuda, cuda:0, cuda:1, etc. (default: cpu)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to model config file (.conf). If not provided, uses default.'
    )
    
    parser.add_argument(
        '--frame_indices',
        type=str,
        default=None,
        help='Comma-separated frame indices to use (e.g., "0,12,34"). Default: first 3 frames'
    )
    
    parser.add_argument(
        '--z_near',
        type=float,
        default=0.1,
        help='Near plane depth (default: 0.1)'
    )
    
    parser.add_argument(
        '--z_far',
        type=float,
        default=10.0,
        help='Far plane depth (default: 10.0)'
    )
    
    parser.add_argument(
        '--ray_batch_size',
        type=int,
        default=50000,
        help='Ray batch size for rendering (default: 50000)'
    )
    
    parser.add_argument(
        '--max_points',
        type=int,
        default=1000000,
        help='Maximum points in output PLY (downsample if exceeded, default: 1000000)'
    )
    
    parser.add_argument(
        '--prefer_plyfile',
        action='store_true',
        help='Prefer plyfile over open3d for PLY export'
    )
    
    args = parser.parse_args()
    
    # Parse device
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)
    else:
        # Try to parse as integer GPU ID
        try:
            gpu_id = int(args.device)
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{gpu_id}')
            else:
                print("CUDA not available, using CPU")
                device = torch.device('cpu')
        except ValueError:
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load transforms.json
    transforms_path = Path(args.transforms)
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")
    
    transforms_data = load_json(transforms_path)
    frames = transforms_data.get('frames', [])
    if not frames:
        raise ValueError("No frames found in transforms.json")
    
    print(f"Found {len(frames)} frames in transforms.json")
    
    # Determine which frames to use
    if args.frame_indices:
        frame_indices = [int(x.strip()) for x in args.frame_indices.split(',')]
    elif args.images:
        # If specific images provided, find their indices
        image_names = [Path(x.strip()).name for x in args.images.split(',')]
        frame_indices = []
        for img_name in image_names:
            for idx, frame in enumerate(frames):
                frame_path = Path(frame.get('file_path', '')).name
                if frame_path == img_name or frame_path.endswith(img_name):
                    frame_indices.append(idx)
                    break
        if len(frame_indices) == 0:
            raise ValueError(f"Could not find images {image_names} in transforms.json")
    else:
        # Default: use first 3 frames
        frame_indices = list(range(min(3, len(frames))))
    
    print(f"Using frame indices: {frame_indices}")
    
    # Determine base directory for images
    if args.images_dir:
        base_dir = Path(args.images_dir)
    else:
        base_dir = transforms_path.parent
    
    # Load images and poses
    images_list = []
    poses_list = []
    focal_pairs = []
    c_list = []
    frame_params: List[Dict[str, float]] = []
    H, W = None, None
    
    for idx in frame_indices:
        if idx >= len(frames):
            raise IndexError(f"Frame index {idx} out of range (0-{len(frames)-1})")
        
        frame = frames[idx]
        file_path = frame.get('file_path', '')
        if not file_path:
            raise ValueError(f"Frame {idx} missing 'file_path'")
        
        # Resolve image path
        image_path = base_dir / file_path
        if not image_path.exists():
            # Try just the filename
            image_path = base_dir / Path(file_path).name
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {base_dir / file_path}")
        
        # Get intrinsics
        fx, fy, cx, cy, frame_W, frame_H = camera_intrinsics_from_json(transforms_data, frame)
        if H is None:
            H, W = frame_H, frame_W
        elif H != frame_H or W != frame_W:
            print(f"Warning: Frame {idx} has different size ({frame_H}x{frame_W}) than first frame ({H}x{W})")
        
        # Load image
        img_tensor = load_image(image_path, size=(W, H))
        images_list.append(img_tensor)
        
        # Get pose
        pose = pose_to_tensor(frame, device)
        poses_list.append(pose)
        
        # Store intrinsics
        focal_pairs.append(torch.tensor([fx, fy], dtype=torch.float32))
        c_list.append(torch.tensor([cx, cy], dtype=torch.float32))
        frame_params.append({"fx": fx, "fy": fy, "cx": cx, "cy": cy})
        
        print(f"Loaded frame {idx}: {image_path.name} ({W}x{H})")
    
    # Stack images and poses
    images = torch.stack(images_list).to(device)  # (NS, 3, H, W)
    poses = torch.stack(poses_list).to(device)  # (NS, 4, 4)
    if focal_pairs:
        focal_value = torch.stack(focal_pairs).mean().item()
    else:
        focal_value = 1.0
    focal_encode = torch.tensor([focal_value], device=device)
    if c_list:
        c_stack = torch.stack(c_list).to(device)
        c_encode = c_stack.mean(dim=0, keepdim=True)
    else:
        c_encode = None
    
    print(f"Loaded {len(images_list)} images, shape: {images.shape}")
    
    # Load model
    weights_path = Path(args.weights)
    config_path = Path(args.config) if args.config else None
    
    model, conf = load_model(weights_path, config_path, device)
    
    # Create renderer
    renderer = NeRFRenderer.from_conf(
        conf["renderer"],
        lindisp=False,
        eval_batch_size=args.ray_batch_size
    ).to(device)
    render_par = renderer.bind_parallel(model, [device], simple_output=True).eval()

    # Encode all selected source views simultaneously (few-shot setting)
    model.encode(
        images.unsqueeze(0),
        poses.unsqueeze(0),
        focal_encode,
        c=c_encode,
    )
    # Render depth and RGB for each target pose and combine point clouds
    all_points = []
    all_colors = []
    
    print("Rendering depth maps...")
    with torch.no_grad():
        for i, idx in enumerate(frame_indices):
            print(f"Rendering frame {idx} ({i+1}/{len(frame_indices)})...")

            params = frame_params[i]
            depth, rgb = render_depth_rgb(
                render_par,
                poses[i],
                params["fx"],
                params["fy"],
                params["cx"],
                params["cy"],
                H,
                W,
                args.z_near,
                args.z_far,
                args.ray_batch_size,
                device,
            )

            frame = frames[idx]
            fx, fy, cx, cy, _, _ = camera_intrinsics_from_json(transforms_data, frame)
            c2w = pose_to_tensor(frame, torch.device('cpu')).numpy()
            points, colors = depth_to_point_cloud(
                depth, rgb,
                fx, fy, cx, cy,
                c2w,
                z_near=args.z_near,
                z_far=args.z_far
            )
            
            all_points.append(points)
            all_colors.append(colors)
            print(f"  Extracted {len(points)} points from frame {idx}")
    
    # Combine all point clouds
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    
    print(f"Total points: {len(all_points)}")
    
    # Downsample if needed
    if len(all_points) > args.max_points:
        print(f"Downsampling from {len(all_points)} to {args.max_points} points...")
        indices = np.random.choice(len(all_points), size=args.max_points, replace=False)
        all_points = all_points[indices]
        all_colors = all_colors[indices]
    
    # Save PLY
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving PLY to {out_path}...")
    
    if args.prefer_plyfile or not HAS_OPEN3D:
        if not HAS_PLYFILE:
            raise RuntimeError("Neither open3d nor plyfile available. Install one: pip install open3d or pip install plyfile")
        save_ply_plyfile(all_points, all_colors, out_path)
    else:
        save_ply_open3d(all_points, all_colors, out_path)
    
    print("Done!")


if __name__ == '__main__':
    main()
