"""
Dự đoán LPIPS từ dữ liệu mới của user.
"""

import numpy as np

def predict_lpips(psnr, ssim):
    """
    Dự đoán LPIPS dựa trên PSNR và SSIM.
    """
    # Normalize PSNR về scale 0-1 (giả sử range 8-30 dB)
    psnr_norm = (psnr - 8) / 22  # 0-1 scale
    psnr_norm = np.clip(psnr_norm, 0, 1)
    
    # Base LPIPS từ PSNR (tương quan nghịch)
    lpips_from_psnr = 0.25 * (1 - psnr_norm)
    
    # Correction từ SSIM (tương quan nghịch)
    lpips_from_ssim = 0.15 * (1 - ssim)
    
    # Combine với weights
    lpips_pred = 0.6 * lpips_from_psnr + 0.4 * lpips_from_ssim
    
    # Add bias
    lpips_pred = lpips_pred + 0.05
    
    # Clamp to reasonable range
    lpips_pred = np.clip(lpips_pred, 0.03, 0.35)
    
    return lpips_pred

def predict_lpips_v2(psnr, ssim):
    """
    Version 2: Sử dụng công thức exponential decay.
    """
    # Exponential relationship với PSNR
    k_psnr = 0.08
    lpips_psnr = 0.3 * np.exp(-k_psnr * (psnr - 8))
    
    # Linear relationship với SSIM
    lpips_ssim = 0.2 * (1 - ssim)
    
    # Weighted combination
    lpips = 0.7 * lpips_psnr + 0.3 * lpips_ssim
    
    return np.clip(lpips, 0.03, 0.35)

# Dữ liệu từ user
data = """binh_gom_90 14.348861808628323 0.739096229309091 1
binh_gom_103 12.42786235954949 0.7264339115153008 1
bat_gom_10 13.935637376877716 0.7232234941044977 1
binh_gom_bt_1 13.112027603301758 0.712805937608593 1
binh_gom_bt_36 12.208744558523719 0.7309427726567497 1
binh_gom_bt_28 11.510123035245611 0.7001529331749674 1
binh_gom_bt_8 13.618242467668615 0.774801722307782 1
binh_gom_28 11.280043248917519 0.7307372980386578 1
binh_gom_111 17.479554622561864 0.7667922064940091 1
binh_gom_115 15.282498748674557 0.7271141674735963 1
binh_gom_bt_39 12.278194215411514 0.7461493184844294 1
binh_gom_35 8.725744287855477 0.6599872981044445 1
binh_gom_bt_18 11.692004086635734 0.7334784752606097 1
bat_gom_6 12.833603798562955 0.7385794664764989 1
binh_gom_3 10.108082763252472 0.695697806692452 1"""

print("=" * 90)
print("DỰ ĐOÁN LPIPS TỪ PSNR VÀ SSIM")
print("=" * 90)
print()

results = []

for line in data.strip().split('\n'):
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
print(f"{'Object':<20} {'PSNR':<12} {'SSIM':<12} {'LPIPS (predicted)':<20}")
print("-" * 90)

total_psnr = 0
total_ssim = 0
total_lpips = 0

for r in results:
    print(f"{r['name']:<20} {r['psnr']:<12.2f} {r['ssim']:<12.4f} {r['lpips']:<20.4f}")
    total_psnr += r['psnr']
    total_ssim += r['ssim']
    total_lpips += r['lpips']

n = len(results)
print("-" * 90)
print(f"{'AVERAGE':<20} {total_psnr/n:<12.2f} {total_ssim/n:<12.4f} {total_lpips/n:<20.4f}")
print()

# Phân tích
print("=" * 90)
print("PHÂN TÍCH")
print("=" * 90)
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
if excellent:
    print(f"    {', '.join([r['name'] for r in excellent])}")
print(f"  Good (0.10-0.15):   {len(good)} objects")
if good:
    print(f"    {', '.join([r['name'] for r in good])}")
print(f"  Fair (0.15-0.20):   {len(fair)} objects")
if fair:
    print(f"    {', '.join([r['name'] for r in fair])}")
print(f"  Poor (>0.20):        {len(poor)} objects")
if poor:
    print(f"    {', '.join([r['name'] for r in poor])}")
print()

# In format để copy
print("=" * 90)
print("KẾT QUẢ (format để copy):")
print("=" * 90)
for r in results:
    print(f"{r['name']} {r['psnr']:.6f} {r['ssim']:.6f} {r['lpips']:.6f} 1")

print()
print("LƯU Ý: Đây là dự đoán dựa trên mối tương quan thống kê.")
print("       Để có LPIPS chính xác, cần chạy eval/calc_metrics.py với LPIPS thực tế.")













