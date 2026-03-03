# Export PLY từ PixelNeRF Model

Script `export_ply.py` để export point cloud từ model PixelNeRF đã train.

## Tóm tắt

Script này thực hiện:
1. ✅ Load model PixelNeRF từ checkpoint weight file
2. ✅ Đọc 2-3 ảnh input và file transforms.json (NeRF format)
3. ✅ Chạy inference để render depth/RGB
4. ✅ Tạo point cloud và xuất ra file PLY duy nhất
5. ✅ Hỗ trợ CPU và GPU
6. ✅ Có kiểm tra lỗi và validation

## Files đã tạo

- **`export_ply.py`**: Script chính (hoàn chỉnh, có thể chạy trên Colab)
- **`COLAB_SETUP.md`**: Hướng dẫn chi tiết setup và sử dụng trên Colab
- **`colab_example.py`**: Các cell mẫu để copy-paste vào Colab notebook

## Quick Start

### Trên Colab

1. Clone repo và cài dependencies:
```python
!git clone https://github.com/sxyu/pixel-nerf.git
%cd pixel-nerf
!pip install pyhocon opencv-python dotmap imageio imageio-ffmpeg open3d
```

2. Chạy script:
```bash
python export_ply.py \
    --weights checkpoint.pth \
    --images_dir dataset_pottery/test/bat_gom_6 \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result.ply \
    --device cuda:0
```

### Trên máy local

```bash
# Cài dependencies từ environment.yml
conda env create -f environment.yml
conda activate pixelnerf

# Hoặc cài thủ công
pip install pyhocon opencv-python dotmap imageio open3d torch torchvision

# Chạy script
python export_ply.py \
    --weights checkpoint.pth \
    --images_dir dataset_pottery/test/bat_gom_6 \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result.ply \
    --device cuda:0
```

## Ví dụ lệnh cụ thể

### Ví dụ 1: Với dataset_pottery/test/bat_gom_6
```bash
python export_ply.py \
    --weights /path/to/checkpoint.pth \
    --images_dir dataset_pottery/test/bat_gom_6 \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result_bat_gom_6.ply \
    --device cuda:0 \
    --frame_indices 0,12,34 \
    --z_near 0.1 \
    --z_far 10.0
```

### Ví dụ 2: Chỉ định ảnh cụ thể
```bash
python export_ply.py \
    --weights checkpoint.pth \
    --images 0001.png,0012.png,0034.png \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result.ply \
    --device 0
```

### Ví dụ 3: Với config tùy chỉnh
```bash
python export_ply.py \
    --weights checkpoint.pth \
    --images_dir dataset_pottery/test/bat_gom_6 \
    --transforms dataset_pottery/test/bat_gom_6/transforms.json \
    --out result.ply \
    --device cuda:0 \
    --config conf/default_mv.conf \
    --max_points 500000
```

## Tham số CLI

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--weights`, `-w` | Đường dẫn đến checkpoint file | **Bắt buộc** |
| `--images`, `-i` | Danh sách ảnh (comma-separated) | None |
| `--images_dir`, `-d` | Thư mục chứa ảnh | None |
| `--transforms`, `-t` | Đường dẫn đến transforms.json | **Bắt buộc** |
| `--out`, `-o` | Đường dẫn file PLY output | **Bắt buộc** |
| `--device` | Device: cpu, cuda, cuda:0, etc. | cpu |
| `--config`, `-c` | Đường dẫn config file | Tự động tìm |
| `--frame_indices` | Chỉ số frames (comma-separated) | 0,1,2 |
| `--z_near` | Near plane depth | 0.1 |
| `--z_far` | Far plane depth | 10.0 |
| `--ray_batch_size` | Batch size cho ray rendering | 50000 |
| `--max_points` | Số điểm tối đa trong PLY | 1000000 |
| `--prefer_plyfile` | Ưu tiên plyfile thay vì open3d | False |

## Định dạng transforms.json

File JSON với cấu trúc NeRF standard:

```json
{
  "camera_angle_x": 0.6646597230498723,
  "w": 128,
  "h": 128,
  "frames": [
    {
      "file_path": "./0001.png",
      "w": 128,
      "h": 128,
      "transform_matrix": [
        [r11, r12, r13, tx],
        [r21, r22, r23, ty],
        [r31, r32, r33, tz],
        [0,   0,   0,   1 ]
      ]
    },
    ...
  ]
}
```

Script tự động tính camera intrinsics từ `camera_angle_x` hoặc sử dụng `fl_x`, `fl_y`, `cx`, `cy` nếu có.

## Dependencies

### Bắt buộc
- `torch`, `torchvision`
- `numpy`
- `Pillow`
- `pyhocon` (để đọc config)

### Tùy chọn (cần ít nhất 1)
- `open3d` (khuyến nghị) - `pip install open3d`
- `plyfile` - `pip install plyfile`

### Từ repo
- Các module trong `src/` (model, render, util, data)

## Giả định và lưu ý

1. **Checkpoint format**: Script hỗ trợ nhiều format:
   - State dict trực tiếp
   - Dict với keys: 'state_dict', 'model_state_dict', 'net_state_dict'
   - Model object đã được save

2. **Config file**: Nếu không chỉ định, script sẽ:
   - Tìm `conf/default_mv.conf`
   - Tìm `conf/default.conf`
   - Tạo config tối thiểu nếu không tìm thấy

3. **Frames**: Mặc định sử dụng 3 frames đầu tiên. Có thể chỉ định bằng `--frame_indices`.

4. **Point cloud**: Kết quả là point cloud kết hợp từ tất cả các frames được chọn.

5. **Rendering**: Script render từ mỗi source view và kết hợp point clouds lại.

## Troubleshooting

### Lỗi import modules
```
Error: No module named 'model'
```
**Giải pháp**: Đảm bảo đang chạy từ thư mục gốc của repo và `src/` có trong Python path.

### CUDA out of memory
**Giải pháp**: Giảm `--ray_batch_size` (ví dụ: `--ray_batch_size 10000`)

### Config không tìm thấy
**Giải pháp**: Chỉ định `--config` với đường dẫn đầy đủ hoặc đảm bảo file `conf/default_mv.conf` tồn tại.

### Ảnh không tìm thấy
**Giải pháp**: Kiểm tra `file_path` trong transforms.json có đúng relative path từ `--images_dir` không.

### Depth map không hợp lệ
**Giải pháp**: Điều chỉnh `--z_near` và `--z_far` phù hợp với scene của bạn.

## Kết quả

Script tạo ra file PLY chứa:
- Point cloud trong world coordinates
- RGB colors cho mỗi point
- Có thể mở bằng MeshLab, CloudCompare, hoặc open3d

## Liên hệ

Nếu gặp vấn đề, kiểm tra:
1. Log output của script
2. Format của transforms.json
3. Checkpoint có đúng format không
4. Config file có đúng không

