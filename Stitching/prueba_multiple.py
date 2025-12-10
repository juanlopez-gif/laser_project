"""
VR-6000 Height Map Stitching - VERSIÓN DEFINITIVA
==================================================
"""

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import re


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StitchConfig:
    pixel_size_mm: float = 0.007413
    expected_overlap_mm: float = 3.0
    max_delta_z_mm: float = 0.15
    min_snr: float = 3.0
    max_row_shift_px: int = 15
    smoothing_sigma: float = 1.0
    use_fallback: bool = True


@dataclass
class StitchResult:
    heightmap: np.ndarray
    overlap_px: int
    overlap_mm: float
    row_shift: int
    delta_z_mm: float
    delta_z_std_mm: float
    correlation_snr: float
    valid_overlap_points: int
    used_fallback: bool = False
    original_file_width_px: int = 0


def validate_heightmap(H: np.ndarray, name: str) -> None:
    if H is None or H.size == 0:
        raise ValueError(f"{name}: Empty heightmap")
    if H.ndim != 2:
        raise ValueError(f"{name}: Must be 2D array")
    valid_fraction = np.sum(~np.isnan(H)) / H.size
    if valid_fraction < 0.1:
        raise ValueError(f"{name}: Too many NaN values")


def find_valid_range(H: np.ndarray) -> Tuple[int, int]:
    valid_mask = ~np.isnan(H)
    cols_with_data = np.sum(valid_mask, axis=0)
    valid_cols = np.where(cols_with_data > H.shape[0] * 0.1)[0]
    if len(valid_cols) == 0:
        return 0, H.shape[1]
    return int(valid_cols[0]), int(valid_cols[-1] + 1)


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) < 10:
        return np.zeros_like(data), valid_mask
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)
    normalized = data.copy()
    if std_val > 1e-10:
        normalized = (data - mean_val) / std_val
    else:
        normalized = data - mean_val
    normalized[~valid_mask] = 0
    return normalized, valid_mask


def smooth_heightmap(H: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(H, sigma=sigma, mode='constant', cval=np.nan)


def load_vr6000_csv(filepath: str) -> Tuple[np.ndarray, float]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        pixel_size = None
        for i, line in enumerate(lines[:50]):
            if 'xy' in line.lower() and 'calibration' in line.lower():
                numbers = re.findall(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)', line)
                if numbers:
                    value = float(numbers[0])
                    if value > 0.1:
                        pixel_size = value / 1000.0
                        print(f"✓ XY Calibration: {value} um = {pixel_size} mm")
                    else:
                        pixel_size = value
                        print(f"✓ XY Calibration: {value} mm")
                    break
        
        if pixel_size is None:
            print(f"\n⚠️  No XY Calibration en {Path(filepath).name}")
            value_input = input(f"XY Calibration: ").strip()
            value = float(re.findall(r'(\d+\.?\d*)', value_input)[0])
            if value > 0.1:
                pixel_size = value / 1000.0
            else:
                pixel_size = value
        
        data_start = 0
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('Height,'):
                parts = line_stripped.split(',')
                if all(p == '' for p in parts[1:]):
                    data_start = i + 1
                    break
        if data_start == 0:
            data_start = 22
        
        data = []
        for line in lines[data_start:]:
            if not line.strip():
                continue
            vals = [x.strip().strip('"') for x in line.strip().split(',')]
            row = []
            for v in vals:
                if v == '' or v == '""':
                    row.append(np.nan)
                else:
                    try:
                        row.append(float(v))
                    except:
                        row.append(np.nan)
            data.append(row)
        
        H = np.array(data, dtype=float)
        H[H < -10] = np.nan
        H[H > 10] = np.nan
        
        logger.info(f"Loaded {Path(filepath).name}: {H.shape}")
        return H, pixel_size
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


def extract_search_bands(H_left, H_right, expected_overlap_px, config):
    left_start, left_end = find_valid_range(H_left)
    right_start, right_end = find_valid_range(H_right)
    search_margin = int(expected_overlap_px * 0.5)
    band_width = expected_overlap_px + search_margin
    band_left_start = max(left_start, left_end - band_width)
    band_left = H_left[:, band_left_start:left_end].copy()
    band_right_end = min(right_end, right_start + band_width)
    band_right = H_right[:, right_start:band_right_end].copy()
    band_left = smooth_heightmap(band_left, config.smoothing_sigma)
    band_right = smooth_heightmap(band_right, config.smoothing_sigma)
    band_left_norm, _ = normalize_data(band_left)
    band_right_norm, _ = normalize_data(band_right)
    return band_left_norm, band_right_norm


def find_overlap_via_correlation(band_left, band_right, H_left, config):
    try:
        correlation = fftconvolve(band_left, band_right[::-1, ::-1], mode='full')
        max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        max_corr_value = correlation[max_idx]
        corr_row, corr_col = max_idx
        corr_std = np.nanstd(correlation)
        snr = max_corr_value / corr_std if corr_std > 0 else 0
        overlap_cols = corr_col - (band_right.shape[1] - 1)
        left_start, left_end = find_valid_range(H_left)
        band_width = band_left.shape[1]
        band_left_start = max(left_start, left_end - band_width)
        overlap_absolute = left_end - (band_left_start + overlap_cols)
        
        if overlap_absolute <= 0 or overlap_absolute > min(H_left.shape[1], band_right.shape[1]):
            if config.use_fallback:
                return None, 0, snr
            raise ValueError(f"Invalid overlap: {overlap_absolute}")
        
        if snr < config.min_snr:
            if config.use_fallback:
                return None, 0, snr
            raise ValueError(f"Low SNR: {snr:.2f}")
        
        row_shift = corr_row - (band_left.shape[0] - 1)
        if abs(row_shift) > config.max_row_shift_px:
            row_shift = np.clip(row_shift, -config.max_row_shift_px, config.max_row_shift_px)
        
        return overlap_absolute, row_shift, snr
    except Exception as e:
        if config.use_fallback:
            return None, 0, 0.0
        raise


def calculate_height_offset(H_left, H_right, overlap_px, row_shift, config):
    min_rows = min(H_left.shape[0], H_right.shape[0])
    overlap_start_left = H_left.shape[1] - overlap_px
    overlap_end_left = H_left.shape[1]
    overlap_start_right = 0
    overlap_end_right = overlap_px
    
    if row_shift > 0:
        left_crop = H_left[row_shift:row_shift+min_rows, overlap_start_left:overlap_end_left]
        right_crop = H_right[0:min_rows, overlap_start_right:overlap_end_right]
    else:
        left_crop = H_left[0:min_rows, overlap_start_left:overlap_end_left]
        right_crop = H_right[-row_shift:-row_shift+min_rows, overlap_start_right:overlap_end_right]
    
    min_overlap_rows = min(left_crop.shape[0], right_crop.shape[0])
    min_overlap_cols = min(left_crop.shape[1], right_crop.shape[1])
    region_left = left_crop[:min_overlap_rows, :min_overlap_cols]
    region_right = right_crop[:min_overlap_rows, :min_overlap_cols]
    valid_overlap = ~np.isnan(region_left) & ~np.isnan(region_right)
    
    if np.sum(valid_overlap) < 10:
        return 0.0, 0.0
    
    diff = region_right[valid_overlap] - region_left[valid_overlap]
    delta_z = np.nanmedian(diff)
    delta_z_std = np.nanstd(diff)
    return delta_z, delta_z_std


def assemble_stitched_heightmap(H_left, H_right, overlap_px, row_shift, delta_z):
    H_right_corrected = H_right - delta_z
    
    if row_shift > 0:
        usable_rows_left = H_left.shape[0] - row_shift
        usable_rows_right = H_right.shape[0]
    else:
        usable_rows_left = H_left.shape[0]
        usable_rows_right = H_right.shape[0] + row_shift
    
    final_rows = min(usable_rows_left, usable_rows_right)
    final_cols = H_left.shape[1] + H_right.shape[1] - overlap_px
    H_stitched = np.full((final_rows, final_cols), np.nan)
    
    if row_shift > 0:
        H_stitched[:, :H_left.shape[1]] = H_left[row_shift:row_shift+final_rows, :]
    else:
        H_stitched[:, :H_left.shape[1]] = H_left[0:final_rows, :]
    
    start_col = H_left.shape[1] - overlap_px
    
    if row_shift > 0:
        new_data = H_right_corrected[0:final_rows, :]
    else:
        offset = -row_shift
        new_data = H_right_corrected[offset:offset+final_rows, :]
    
    old_data = H_stitched[:, start_col:start_col+H_right.shape[1]]
    H_stitched[:, start_col:start_col+H_right.shape[1]] = np.where(
        ~np.isnan(new_data),
        new_data,
        old_data
    )
    
    overlap_region = slice(start_col, H_left.shape[1])
    
    if row_shift > 0:
        left_overlap = H_left[row_shift:row_shift+final_rows, -overlap_px:]
        right_overlap = H_right_corrected[0:final_rows, :overlap_px]
    else:
        left_overlap = H_left[0:final_rows, -overlap_px:]
        offset = -row_shift
        right_overlap = H_right_corrected[offset:offset+final_rows, :overlap_px]
    
    if left_overlap.shape == right_overlap.shape:
        stacked = np.dstack((left_overlap, right_overlap))
        with np.errstate(invalid='ignore'):
            overlap_avg = np.nanmean(stacked, axis=2)
            old_overlap = H_stitched[:, overlap_region]
            H_stitched[:, overlap_region] = np.where(
                ~np.isnan(overlap_avg),
                overlap_avg,
                old_overlap
            )
    
    return H_stitched


def stitch_two_heightmaps(H_left, H_right, config, override_overlap_mm=None, original_right_width=None):
    validate_heightmap(H_left, "H_left")
    validate_heightmap(H_right, "H_right")
    
    overlap_mm = override_overlap_mm if override_overlap_mm is not None else config.expected_overlap_mm
    expected_overlap_px = int(overlap_mm / config.pixel_size_mm)
    
    band_left, band_right = extract_search_bands(H_left, H_right, expected_overlap_px, config)
    overlap_px, row_shift, snr = find_overlap_via_correlation(band_left, band_right, H_left, config)
    
    used_fallback = False
    if overlap_px is None:
        overlap_px = expected_overlap_px
        row_shift = 0
        snr = 0.0
        used_fallback = True
    
    delta_z, delta_z_std = calculate_height_offset(H_left, H_right, overlap_px, row_shift, config)
    H_stitched = assemble_stitched_heightmap(H_left, H_right, overlap_px, row_shift, delta_z)
    valid_overlap_points = int(np.sum(~np.isnan(H_stitched[:, H_left.shape[1]-overlap_px:H_left.shape[1]])))
    
    logger.info(f"Complete: {H_stitched.shape}, overlap: {overlap_px*config.pixel_size_mm:.2f}mm")
    
    return StitchResult(
        heightmap=H_stitched, overlap_px=overlap_px, overlap_mm=overlap_px*config.pixel_size_mm,
        row_shift=row_shift, delta_z_mm=delta_z, delta_z_std_mm=delta_z_std,
        correlation_snr=snr, valid_overlap_points=valid_overlap_points,
        used_fallback=used_fallback,
        original_file_width_px=original_right_width if original_right_width else H_right.shape[1]
    )


def configure_overlaps(num_files, default_overlap):
    print("\n" + "="*70)
    print("OVERLAP CONFIGURATION")
    print("="*70)
    print(f"Files: {num_files} ({num_files-1} operations)")
    print(f"Default: {default_overlap:.2f}mm")
    print("\n1. Same for all\n2. Different for each\n3. Auto (use default)")
    
    choice = input("\nChoice [1/2/3]: ").strip()
    
    if choice == "1":
        overlap = input(f"Overlap (mm) [{default_overlap:.2f}]: ").strip()
        overlap = float(overlap) if overlap else default_overlap
        return {i: overlap for i in range(num_files - 1)}
    elif choice == "2":
        overlaps = {}
        for i in range(num_files - 1):
            overlap = input(f"File {i+1}->{i+2} (mm) [{default_overlap:.2f}]: ").strip()
            overlaps[i] = float(overlap) if overlap else default_overlap
        return overlaps
    else:
        return {i: default_overlap for i in range(num_files - 1)}


def plot_heightmap_with_overlaps(H_final, all_results, pixel_size_mm, first_file_width_px, output_prefix):
    """✅ HEIGHTMAP CON OVERLAPS MARCADOS"""
    from matplotlib.colors import LinearSegmentedColormap
    
    vmin, vmax = -0.063, 0.065
    colors = ['#00008B', '#0000FF', '#00FFFF', '#FFFF00', '#FFA500', '#FF0000']
    positions = [0.0, 0.20, 0.38, 0.56, 0.73, 1.0]
    cmap = LinearSegmentedColormap.from_list('vr6200', list(zip(positions, colors)), N=256)
    
    fig, ax = plt.subplots(figsize=(20, 6))
    width = H_final.shape[1] * pixel_size_mm
    height = H_final.shape[0] * pixel_size_mm
    extent = [0, width, 0, height]
    
    im = ax.imshow(H_final, cmap=cmap, aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
    
    # ✅ MARCAR OVERLAPS
    current_pos = first_file_width_px
    
    for i, result in enumerate(all_results):
        overlap_start_px = current_pos - result.overlap_px
        overlap_end_px = current_pos
        
        overlap_start_mm = overlap_start_px * pixel_size_mm
        overlap_end_mm = overlap_end_px * pixel_size_mm
        
        ax.axvline(overlap_start_mm, color='white', linewidth=2, linestyle='--', alpha=0.8)
        ax.axvline(overlap_end_mm, color='black', linewidth=2, linestyle='--', alpha=0.8)
        ax.axvspan(overlap_start_mm, overlap_end_mm, alpha=0.15, color='yellow')
        
        mid_overlap = (overlap_start_mm + overlap_end_mm) / 2
        ax.text(mid_overlap, height * 0.95, f'Overlap {i+1}\n{result.overlap_mm:.1f}mm', 
                ha='center', va='top', fontsize=9,
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        current_pos += (result.original_file_width_px - result.overlap_px)
    
    ax.set_xlabel('Width (mm)', fontsize=12)
    ax.set_ylabel('Height (mm)', fontsize=12)
    ax.set_title(f'FINAL STITCHED HEIGHTMAP (WITH OVERLAPS)\n{width:.1f}x{height:.1f}mm | {len(all_results)} overlaps',
                fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Height (mm)')
    plt.tight_layout()
    
    fname = f"{output_prefix}_WITH_overlaps.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {fname}")


def plot_heightmap_clean(H_final, pixel_size_mm, output_prefix):
    """✅ HEIGHTMAP LIMPIO"""
    from matplotlib.colors import LinearSegmentedColormap
    
    vmin, vmax = -0.063, 0.065
    colors = ['#00008B', '#0000FF', '#00FFFF', '#FFFF00', '#FFA500', '#FF0000']
    positions = [0.0, 0.20, 0.38, 0.56, 0.73, 1.0]
    cmap = LinearSegmentedColormap.from_list('vr6200', list(zip(positions, colors)), N=256)
    
    fig, ax = plt.subplots(figsize=(20, 6))
    width = H_final.shape[1] * pixel_size_mm
    height = H_final.shape[0] * pixel_size_mm
    extent = [0, width, 0, height]
    
    im = ax.imshow(H_final, cmap=cmap, aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
    
    ax.set_xlabel('Width (mm)', fontsize=12)
    ax.set_ylabel('Height (mm)', fontsize=12)
    ax.set_title(f'FINAL STITCHED HEIGHTMAP (CLEAN)\n{width:.1f}x{height:.1f}mm',
                fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Height (mm)')
    plt.tight_layout()
    
    fname = f"{output_prefix}_CLEAN.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {fname}")


def plot_profiles_with_overlaps(H_final, all_results, pixel_size_mm, first_file_width_px, output_prefix):
    """✅ 5 PERFILES CON OVERLAPS MARCADOS (EN MM)"""
    logger.info("Generating 5 continuity profiles...")
    
    fig, axes = plt.subplots(5, 1, figsize=(18, 14))
    rows_to_check = [H_final.shape[0]//5, 2*H_final.shape[0]//5, H_final.shape[0]//2, 
                     3*H_final.shape[0]//5, 4*H_final.shape[0]//5]
    
    for i, row in enumerate(rows_to_check):
        ax = axes[i]
        profile = H_final[row, :]
        valid = ~np.isnan(profile)
        
        # ✅ CONVERTIR A MM
        cols_mm = np.arange(len(profile)) * pixel_size_mm
        
        ax.plot(cols_mm[valid], profile[valid]*1000, 'b-', linewidth=1.5, alpha=0.8)
        
        # ✅ MARCAR OVERLAPS
        current_pos = first_file_width_px
        
        for j, result in enumerate(all_results):
            overlap_start_px = current_pos - result.overlap_px
            overlap_end_px = current_pos
            
            overlap_start_mm = overlap_start_px * pixel_size_mm
            overlap_end_mm = overlap_end_px * pixel_size_mm
            
            ax.axvline(overlap_start_mm, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(overlap_end_mm, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvspan(overlap_start_mm, overlap_end_mm, alpha=0.15, color='yellow')
            
            current_pos += (result.original_file_width_px - result.overlap_px)
        
        ax.set_ylabel('Height (μm)', fontsize=10)
        ax.set_title(f'Profile at row {row} ({100*row/H_final.shape[0]:.0f}%)', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Position (mm)', fontsize=11)
    plt.suptitle('PROFILE CONTINUITY - All Overlaps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fname = f"{output_prefix}_profiles.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {fname}")


def save_csv_with_header(filepath, heightmap, pixel_size_mm):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('"Stitched Heightmap - Shipyard 4.0 MVP2"\n')
        f.write(f'"Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"\n')
        f.write(f'"XY Calibration: {pixel_size_mm*1000:.3f} um"\n')
        f.write(f'"Dimensions: {heightmap.shape[1]} x {heightmap.shape[0]} pixels"\n')
        f.write(f'"Physical Size: {heightmap.shape[1]*pixel_size_mm:.2f} x {heightmap.shape[0]*pixel_size_mm:.2f} mm"\n')
        f.write(f'"Height Range: [{np.nanmin(heightmap):.6f}, {np.nanmax(heightmap):.6f}] mm"\n')
        f.write('""\n"Height"\n')
        np.savetxt(f, heightmap, delimiter=',', fmt='%.6f')
    logger.info(f"CSV saved: {Path(filepath).name}")


def stitch_multiple_files(csv_files, output_dir=".", output_prefix="stitched", config=None, interactive=True):
    if config is None:
        config = StitchConfig()
    if len(csv_files) < 2:
        raise ValueError("Need at least 2 files")
    
    if interactive:
        overlaps = configure_overlaps(len(csv_files), config.expected_overlap_mm)
    else:
        overlaps = {i: config.expected_overlap_mm for i in range(len(csv_files) - 1)}
    
    logger.info("="*70)
    logger.info(f"STITCHING {len(csv_files)} FILES")
    logger.info("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    H_result, pixel_size_mm = load_vr6000_csv(csv_files[0])
    config.pixel_size_mm = pixel_size_mm
    first_file_width_px = H_result.shape[1]
    
    all_results = []
    
    for i, csv_file in enumerate(csv_files[1:]):
        logger.info(f"\n[{i+2}/{len(csv_files)}] Stitching {Path(csv_file).name}...")
        H_next, _ = load_vr6000_csv(csv_file)
        original_next_width = H_next.shape[1]
        
        result = stitch_two_heightmaps(H_result, H_next, config, 
                                       override_overlap_mm=overlaps[i],
                                       original_right_width=original_next_width)
        H_result = result.heightmap
        all_results.append(result)
    
    output_csv = output_path / f"{output_prefix}_FINAL.csv"
    save_csv_with_header(str(output_csv), H_result, pixel_size_mm)
    
    logger.info("\n" + "="*70)
    logger.info("GENERATING PLOTS")
    logger.info("="*70)
    
    plot_heightmap_with_overlaps(H_result, all_results, pixel_size_mm, first_file_width_px, str(output_path / output_prefix))
    plot_heightmap_clean(H_result, pixel_size_mm, str(output_path / output_prefix))
    plot_profiles_with_overlaps(H_result, all_results, pixel_size_mm, first_file_width_px, str(output_path / output_prefix))
    
    # ✅ CALCULAR LONGITUD PIEZA
    valid_cols = []
    for col in range(H_result.shape[1]):
        if np.sum(~np.isnan(H_result[:, col])) > H_result.shape[0] * 0.1:
            valid_cols.append(col)
    
    if valid_cols:
        piece_length_px = valid_cols[-1] - valid_cols[0] + 1
        piece_length_mm = piece_length_px * pixel_size_mm
    else:
        piece_length_mm = H_result.shape[1] * pixel_size_mm
    
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Files stitched: {len(csv_files)}")
    logger.info(f"XY Calibration: {pixel_size_mm*1000:.3f} um")
    logger.info(f"Final shape: {H_result.shape} pixels")
    logger.info(f"Piece length: {piece_length_mm:.2f} mm")
    logger.info(f"Piece height: {H_result.shape[0]*pixel_size_mm:.2f} mm")
    
    for i, res in enumerate(all_results):
        logger.info(f"  Overlap {i+1}: {res.overlap_mm:.2f} mm")
    
    return H_result, all_results, pixel_size_mm


if __name__ == "__main__":
    config = StitchConfig(
        pixel_size_mm=0.007413, expected_overlap_mm=3.0, max_delta_z_mm=0.15,
        min_snr=3.0, max_row_shift_px=15, smoothing_sigma=1.0, use_fallback=True
    )
    
    csv_files = [
        r"C:\Users\javie\Desktop\Universidad\Cuarto-Univesrity of Rhode Island\Primer Cuatrimestre\ISE 401 TFG\ProyectoLaser\Stitching\1.1Height.csv",
        r"C:\Users\javie\Desktop\Universidad\Cuarto-Univesrity of Rhode Island\Primer Cuatrimestre\ISE 401 TFG\ProyectoLaser\Stitching\1.2Height.csv",
        r"C:\Users\javie\Desktop\Universidad\Cuarto-Univesrity of Rhode Island\Primer Cuatrimestre\ISE 401 TFG\ProyectoLaser\Stitching\1.3Height.csv",
        r"C:\Users\javie\Desktop\Universidad\Cuarto-Univesrity of Rhode Island\Primer Cuatrimestre\ISE 401 TFG\ProyectoLaser\Stitching\1.4Height.csv"
    ]
    
    H_final, results, pixel_size = stitch_multiple_files(
        csv_files, output_dir=".", output_prefix=f"heightmap_{datetime.now().strftime('%Y%m%d_%H%M')}",
        config=config, interactive=True
    )
    
    logger.info("✅ COMPLETED")