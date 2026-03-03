# Lệnh chạy export_ply.py

## Lệnh đầy đủ (Windows PowerShell)

```powershell
# Bước 1: Kích hoạt conda environment (nếu dùng conda)
conda activate pixelnerf

# Bước 2: Chạy script
python export_ply.py --weights checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest --images_dir dataset_pottery/test/bat_gom_6 --transforms dataset_pottery/test/bat_gom_6/transforms.json --out result_bat_gom_6.ply --device cuda:0 --frame_indices 0,12,34
```

## Lệnh đầy đủ (Linux/Mac/Colab)

```bash
python export_ply.py \
    --weights checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest \
    --images_dir dataset_pottery/test/bat_gom_6 \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result_bat_gom_6.ply \
    --device cuda:0 \
    --frame_indices 0,12,34
```

## Lệnh một dòng (Windows CMD)

```cmd
python export_ply.py --weights checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest --images_dir dataset_pottery/test/bat_gom_6 --transforms dataset_pottery/test/bat_gom_6/transforms.json --out result_bat_gom_6.ply --device cuda:0 --frame_indices 0,12,34
```

## Kiểm tra trước khi chạy

1. **Kích hoạt environment:**
   ```bash
   conda activate pixelnerf
   # hoặc
   source activate pixelnerf
   ```

2. **Kiểm tra dependencies:**
   ```python
   python -c "import torch; import numpy; from PIL import Image; print('OK')"
   ```

3. **Kiểm tra checkpoint tồn tại:**
   ```python
   import os
   print(os.path.exists('checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest'))
   ```

4. **Kiểm tra dataset:**
   ```python
   import os
   print(os.path.exists('dataset_pottery/test/bat_gom_6/transforms.json'))
   ```

## Troubleshooting

### Lỗi: ModuleNotFoundError: No module named 'torch'
**Giải pháp:** Kích hoạt conda environment:
```bash
conda activate pixelnerf
```

### Lỗi: CUDA not available
**Giải pháp:** Dùng CPU hoặc kiểm tra CUDA:
```bash
# Dùng CPU
--device cpu

# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Lỗi: Weights not found
**Giải pháp:** Kiểm tra đường dẫn checkpoint:
```python
import os
print(os.path.abspath('checkpoints/pn_90nv_v6_b2_200e/pixel_nerf_latest'))
```

### Lỗi: Image not found
**Giải pháp:** Kiểm tra file_path trong transforms.json có đúng relative path không.

## Output mong đợi

Script sẽ:
1. Load model từ checkpoint
2. Load 3 images (frames 0, 12, 34)
3. Render depth maps
4. Tạo point cloud
5. Lưu file `result_bat_gom_6.ply`

Thời gian chạy: ~1-5 phút tùy GPU và số lượng points.

