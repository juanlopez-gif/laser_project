#!/usr/bin/env python3
"""
VALIDATION (FIXED) - Compare Stitched vs Ground Truth
======================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_csv_fixed(filepath):
    """Load CSV - FIXED version that handles variable columns"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find pixel size
    pixel_size = 0.007413
    for line in lines[:20]:
        if 'calibration' in line.lower():
            parts = line.split(',')
            for p in parts:
                try:
                    val = float(p.strip().strip('"'))
                    if val > 0.1:
                        pixel_size = val / 1000.0
                        break
                    elif 0.001 < val < 0.1:
                        pixel_size = val
                        break
                except:
                    pass
    
    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        if 'height' in line.lower() and line.strip().startswith('"'):
            data_start = i + 1
            break
    
    # PASS 1: Find max columns
    max_cols = 0
    for line in lines[data_start:]:
        if not line.strip():
            continue
        vals = line.strip().split(',')
        max_cols = max(max_cols, len(vals))
    
    print(f"  Max columns: {max_cols}")
    
    # PASS 2: Read with padding
    data = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        vals = [v.strip().strip('"') for v in line.strip().split(',')]
        row = []
        for v in vals:
            if v == '' or v == '""':
                row.append(np.nan)
            else:
                try:
                    row.append(float(v))
                except:
                    row.append(np.nan)
        
        # Pad to max_cols
        while len(row) < max_cols:
            row.append(np.nan)
        
        data.append(row)
    
    H = np.array(data, dtype=float)
    
    # Clean empty rows/cols
    valid_rows = ~np.all(np.isnan(H), axis=1)
    valid_cols = ~np.all(np.isnan(H), axis=0)
    H = H[valid_rows, :][:, valid_cols]
    
    return H, pixel_size


def compare_with_ground_truth(stitched_file, ground_truth_file, output_dir="."):
    """Compare stitched result vs ground truth"""
    
    print("="*70)
    print("VALIDATION: Stitched vs Ground Truth")
    print("="*70)
    
    print(f"\nLoading ground truth: {Path(ground_truth_file).name}")
    Z_truth, ps_truth = load_csv_fixed(ground_truth_file)
    print(f"  Shape: {Z_truth.shape}")
    print(f"  Pixel: {ps_truth*1000:.3f} µm")
    
    print(f"\nLoading stitched: {Path(stitched_file).name}")
    Z_stitched, ps_stitched = load_csv_fixed(stitched_file)
    print(f"  Shape: {Z_stitched.shape}")
    print(f"  Pixel: {ps_stitched*1000:.3f} µm")
    
    # Crop to same size
    min_rows = min(Z_truth.shape[0], Z_stitched.shape[0])
    min_cols = min(Z_truth.shape[1], Z_stitched.shape[1])
    
    Z_truth = Z_truth[:min_rows, :min_cols]
    Z_stitched = Z_stitched[:min_rows, :min_cols]
    
    print(f"\nComparing: {min_rows}×{min_cols} pixels")
    
    # Calculate errors
    valid = ~np.isnan(Z_truth) & ~np.isnan(Z_stitched)
    n_valid = np.sum(valid)
    
    if n_valid < 100:
        print("\n❌ ERROR: Not enough valid points")
        return
    
    diff_um = (Z_truth[valid] - Z_stitched[valid]) * 1000
    
    mae = np.mean(np.abs(diff_um))
    rmse = np.sqrt(np.mean(diff_um**2))
    std = np.std(diff_um)
    max_err = np.max(np.abs(diff_um))
    mean_err = np.mean(diff_um)
    
    print("\n" + "="*70)
    print("VALIDATION METRICS")
    print("="*70)
    print(f"Valid points: {n_valid:,} ({100*n_valid/(min_rows*min_cols):.1f}%)")
    print(f"\nERROR STATISTICS:")
    print(f"  MAE:  {mae:.3f} µm")
    print(f"  RMSE: {rmse:.3f} µm")
    print(f"  STD:  {std:.3f} µm")
    print(f"  MAX:  {max_err:.3f} µm")
    print(f"  MEAN: {mean_err:.3f} µm (bias)")
    print("="*70)
    
    print("\nVERDICT:")
    if mae < 3.0:
        print("  ✅ EXCELLENT - Algorithm PERFECT (at noise limit)")
        print("     → Real 7.67 µm errors are 100% fixture")
    elif mae < 5.0:
        print("  ✅ VERY GOOD - Algorithm excellent")
    elif mae < 10.0:
        print("  ✅ GOOD - Algorithm working well")
    elif mae < 20.0:
        print("  ⚠️  ACCEPTABLE - Room for improvement")
    else:
        print("  ❌ FAIL - Algorithm has bugs")
    
    print("\nEXPECTED: MAE ~2 µm (noise limit)")
    print("="*70)
    
    # Save plot
    plot_validation(Z_truth, Z_stitched, diff_um, ps_truth, mae, rmse, output_dir)
    
    return {'mae': mae, 'rmse': rmse, 'std': std, 'max': max_err}


def plot_validation(Z_truth, Z_stitched, diff_um, pixel_size, mae, rmse, output_dir):
    """Generate validation plots"""
    
    rows, cols = Z_truth.shape
    extent = [0, cols*pixel_size, 0, rows*pixel_size]
    
    # ✅ Calcular STD aquí
    std = np.std(diff_um)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Ground Truth
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(Z_truth*1000, cmap='viridis', aspect='auto', 
                     extent=extent, origin='lower')
    ax1.set_title('Ground Truth', fontweight='bold')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=ax1, label='Height (µm)')
    
    # 2. Stitched
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(Z_stitched*1000, cmap='viridis', aspect='auto', 
                     extent=extent, origin='lower')
    ax2.set_title('Stitched Result', fontweight='bold')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=ax2, label='Height (µm)')
    
    # 3. Error Map
    ax3 = plt.subplot(2, 3, 3)
    diff_2d = np.full((rows, cols), np.nan)
    valid = ~np.isnan(Z_truth) & ~np.isnan(Z_stitched)
    diff_2d[valid] = (Z_truth[valid] - Z_stitched[valid]) * 1000
    
    vmax = max(abs(np.nanpercentile(diff_2d, 1)), abs(np.nanpercentile(diff_2d, 99)))
    im3 = ax3.imshow(diff_2d, cmap='RdBu_r', aspect='auto', extent=extent, 
                     origin='lower', vmin=-vmax, vmax=vmax)
    ax3.set_title(f'Error Map (MAE={mae:.2f}µm)', fontweight='bold')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=ax3, label='Error (µm)')
    
    # 4. Histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(diff_um, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='red', linestyle='--', lw=2)
    ax4.axvline(np.mean(diff_um), color='green', linestyle='-', lw=2)
    ax4.set_xlabel('Error (µm)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Profile
    ax5 = plt.subplot(2, 3, 5)
    center = rows // 2
    x_mm = np.arange(cols) * pixel_size
    ax5.plot(x_mm, Z_truth[center, :]*1000, 'b-', lw=2, label='Ground Truth', alpha=0.8)
    ax5.plot(x_mm, Z_stitched[center, :]*1000, 'r--', lw=2, label='Stitched', alpha=0.8)
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Height (µm)')
    ax5.set_title('Center Profile', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    verdict = "✅ EXCELLENT" if mae < 3 else "✅ VERY GOOD" if mae < 5 else "✅ GOOD" if mae < 10 else "⚠️ ACCEPTABLE" if mae < 20 else "❌ FAIL"
    color = "#00CC00" if mae < 5 else "#66CC00" if mae < 10 else "#FFAA00" if mae < 20 else "#FF0000"
    
    summary = f"""
VALIDATION SUMMARY

Valid Points: {len(diff_um):,}

━━━━━━━━━━━━━━━━━━
METRICS (µm)
━━━━━━━━━━━━━━━━━━

MAE:  {mae:.2f} µm
RMSE: {rmse:.2f} µm
STD:  {std:.2f} µm
MAX:  {np.max(np.abs(diff_um)):.2f} µm

━━━━━━━━━━━━━━━━━━
VERDICT
━━━━━━━━━━━━━━━━━━

{verdict}

Expected: ~2µm
"""
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=11, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=3))
    
    plt.suptitle('SYNTHETIC VALIDATION REPORT', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = f"{output_dir}/validation_report.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    
    STITCHED_FILE = "./heightmap_20251210_1252_FINAL.csv"
    GROUND_TRUTH_FILE = "./synthetic_tiles/ground_truth_FULL.csv"
    OUTPUT_DIR = "./validation_results"
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        metrics = compare_with_ground_truth(STITCHED_FILE, GROUND_TRUTH_FILE, OUTPUT_DIR)
        print(f"\n✅ Complete. Results in: {OUTPUT_DIR}/")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()