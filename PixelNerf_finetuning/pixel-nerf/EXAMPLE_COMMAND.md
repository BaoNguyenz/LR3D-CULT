# Ví dụ lệnh chạy với checkpoint pn_90nv_v6_b2_200e

## Checkpoint path
```
checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest
```

## Lệnh mẫu

### Ví dụ 1: Với dataset_pottery/test/bat_gom_6
```bash
python export_ply.py \
    --weights checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest \
    --images_dir dataset_pottery/test/bat_gom_6 \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result_bat_gom_6.ply \
    --device cuda:0 \
    --frame_indices 0,12,34
```

### Ví dụ 2: Với config tùy chỉnh (nếu có)
```bash
python export_ply.py \
    --weights checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest \
    --images_dir dataset_pottery/test/bat_gom_6 \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result.ply \
    --device cuda:0 \
    --config conf/default_mv.conf \
    --frame_indices 0,12,34 \
    --z_near 0.1 \
    --z_far 10.0
```

### Ví dụ 3: Chỉ định ảnh cụ thể
```bash
python export_ply.py \
    --weights checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest \
    --images 0001.png,0012.png,0034.png \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result.ply \
    --device 0
```

### Ví dụ 4: Chạy trên CPU (nếu không có GPU)
```bash
python export_ply.py \
    --weights checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest \
    --images_dir dataset_pottery/test/bat_gom_6 \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result.ply \
    --device cpu \
    --ray_batch_size 10000
```

## Lưu ý

1. **Checkpoint file**: File `pixel_nerf_latest` không có extension `.pth`, script sẽ tự động xử lý.

2. **Config file**: Nếu experiment `pn_90nv_v6_b2_200e` có config riêng trong `expconf.conf`, script có thể tự động tìm. Nếu không, sẽ dùng default config.

3. **Frame indices**: Mặc định dùng 3 frames đầu tiên (0, 1, 2). Có thể chỉ định khác bằng `--frame_indices`.

4. **Device**: 
   - `cuda:0` hoặc `0` cho GPU đầu tiên
   - `cpu` cho CPU
   - Script tự động fallback về CPU nếu CUDA không available

## Kiểm tra checkpoint

Để kiểm tra checkpoint có load được không:
```python
import torch
ckpt = torch.load('checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest', map_location='cpu')
print(type(ckpt))
if isinstance(ckpt, dict):
    print("Keys:", list(ckpt.keys())[:10])
```

