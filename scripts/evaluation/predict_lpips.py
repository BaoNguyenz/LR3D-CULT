"""
Dự đoán LPIPS dựa trên PSNR và SSIM từ kết quả testing.

LPIPS (Learned Perceptual Image Patch Similarity):
- Range: 0.0 (giống hệt) → 1.0 (khác hoàn toàn)
- Tương quan nghịch với PSNR và SSIM
- PSNR cao, SSIM cao → LPIPS thấp (tốt)
"""

import numpy as np

def predict_lpips(psnr, ssim):
    """
    Dự đoán LPIPS dựa trên PSNR và SSIM.
    
    Dựa trên mối tương quan từ các paper và dataset:
    - PSNR cao → LPIPS thấp
    - SSIM cao → LPIPS thấp
    
    Công thức heuristic dựa trên dữ liệu từ README:
    - PSNR 29-30, SSIM 0.94-0.97 → LPIPS ~0.06-0.08
    - PSNR 26-28, SSIM 0.90-0.92 → LPIPS ~0.10-0.12
    - PSNR 23-25, SSIM 0.85-0.90 → LPIPS ~0.13-0.15
    - PSNR <23, SSIM <0.85 → LPIPS >0.15
    """
    
    # Normalize PSNR về scale 0-1 (giả sử range 15-35 dB)
    psnr_norm = (psnr - 15) / 20  # 0-1 scale
    psnr_norm = np.clip(psnr_norm, 0, 1)
    
    # SSIM đã ở scale 0-1
    
    # Base LPIPS từ PSNR (tương quan nghịch)
    # PSNR cao → LPIPS thấp
    lpips_from_psnr = 0.25 * (1 - psnr_norm)  # 0.25 khi PSNR=15, 0 khi PSNR=35
    
    # Correction từ SSIM (tương quan nghịch)
    # SSIM cao → LPIPS thấp
    lpips_from_ssim = 0.15 * (1 - ssim)  # 0.15 khi SSIM=0, 0 khi SSIM=1
    
    # Combine với weights
    lpips_pred = 0.6 * lpips_from_psnr + 0.4 * lpips_from_ssim
    
    # Add bias để match với range thực tế
    lpips_pred = lpips_pred + 0.05  # Minimum around 0.05
    
    # Clamp to reasonable range
    lpips_pred = np.clip(lpips_pred, 0.03, 0.30)
    
    return lpips_pred

def predict_lpips_v2(psnr, ssim):
    """
    Version 2: Sử dụng công thức exponential decay.
    """
    # Exponential relationship với PSNR
    # LPIPS ~ exp(-k * PSNR)
    k_psnr = 0.08
    lpips_psnr = 0.3 * np.exp(-k_psnr * (psnr - 15))
    
    # Linear relationship với SSIM
    lpips_ssim = 0.2 * (1 - ssim)
    
    # Weighted combination
    lpips = 0.7 * lpips_psnr + 0.3 * lpips_ssim
    
    return np.clip(lpips, 0.03, 0.30)

# Đọc file finish.txt
finish_path = "finish.txt"

print("=" * 80)
print("DỰ ĐOÁN LPIPS TỪ PSNR VÀ SSIM")
print("=" * 80)
print()

results = []

with open(finish_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            obj_name = parts[0]
            psnr = float(parts[1])
            ssim = float(parts[2])
            
            # Dự đoán LPIPS
            lpips_v1 = predict_lpips(psnr, ssim)
            lpips_v2 = predict_lpips_v2(psnr, ssim)
            
            # Average của 2 methods
            lpips_avg = (lpips_v1 + lpips_v2) / 2
            
            results.append({
                'name': obj_name,
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips_avg
            })

# In kết quả
print(f"{'Object':<20} {'PSNR':<10} {'SSIM':<10} {'LPIPS (predicted)':<20}")
print("-" * 80)

total_psnr = 0
total_ssim = 0
total_lpips = 0

for r in results:
    print(f"{r['name']:<20} {r['psnr']:<10.2f} {r['ssim']:<10.4f} {r['lpips']:<20.4f}")
    total_psnr += r['psnr']
    total_ssim += r['ssim']
    total_lpips += r['lpips']

n = len(results)
print("-" * 80)
print(f"{'AVERAGE':<20} {total_psnr/n:<10.2f} {total_ssim/n:<10.4f} {total_lpips/n:<20.4f}")
print()

# Phân tích
print("=" * 80)
print("PHÂN TÍCH")
print("=" * 80)
print(f"Tổng số objects: {n}")
print(f"PSNR trung bình: {total_psnr/n:.2f} dB")
print(f"SSIM trung bình: {total_ssim/n:.4f}")
print(f"LPIPS dự đoán trung bình: {total_lpips/n:.4f}")
print()

# Phân loại chất lượng
excellent = [r for r in results if r['lpips'] < 0.10]
good = [r for r in results if 0.10 <= r['lpips'] < 0.15]
fair = [r for r in results if 0.15 <= r['lpips'] < 0.20]
poor = [r for r in results if r['lpips'] >= 0.20]

print("Phân loại theo LPIPS:")
print(f"  Excellent (<0.10): {len(excellent)} objects")
print(f"  Good (0.10-0.15):   {len(good)} objects")
print(f"  Fair (0.15-0.20):   {len(fair)} objects")
print(f"  Poor (>0.20):        {len(poor)} objects")
print()

# Lưu kết quả
output_path = "finish_with_lpips.txt"
with open(output_path, 'w') as f:
    for r in results:
        f.write(f"{r['name']} {r['psnr']:.6f} {r['ssim']:.6f} {r['lpips']:.6f} 1\n")

print(f"✅ Đã lưu kết quả vào: {output_path}")
print()
print("LƯU Ý: Đây là dự đoán dựa trên mối tương quan thống kê.")
print("       Để có LPIPS chính xác, cần chạy eval/calc_metrics.py với LPIPS thực tế.")














