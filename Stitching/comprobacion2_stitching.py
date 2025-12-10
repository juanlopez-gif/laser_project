"""
VR-6000 Height Map Stitching - COMPLETAMENTE CORREGIDO
=======================================================
Versión con assemble_stitched_heightmap() ARREGLADO
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
class OverlapMetrics:
    """Métricas de calidad del overlap"""
    mae_um: float
    rmse_um: float
    std_um: float
    max_um: float
    n_points: int
    verdict: str  # "EXCELLENT", "GOOD", "ACCEPTABLE", "FAIL"
    
    
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
    overlap_metrics: Optional[OverlapMetrics] = None


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
            if line_stripped.startswith('Height,') or line_stripped.startswith('"Height"'):
                parts = line_stripped.split(',')
                if all(p == '' or p == '""' for p in parts[1:]):
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
            if row:
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
    """
    ✅ VERSIÓN COMPLETAMENTE CORREGIDA
    Ensambla dos heightmaps con overlap correctamente
    """
    H_right_corrected = H_right - delta_z
    
    # Calcular dimensiones finales y offsets
    if row_shift > 0:
        # Right está más arriba que left
        usable_rows = min(H_left.shape[0] - row_shift, H_right.shape[0])
        left_start_row = row_shift
        right_start_row = 0
    else:
        # Left está más arriba que right (o iguales si row_shift=0)
        usable_rows = min(H_left.shape[0], H_right.shape[0] + row_shift)
        left_start_row = 0
        right_start_row = -row_shift
    
    final_rows = usable_rows
    final_cols = H_left.shape[1] + H_right.shape[1] - overlap_px
    
    # Crear canvas vacío
    H_stitched = np.full((final_rows, final_cols), np.nan)
    
    # ✅ PASO 1: Copiar toda la parte de H_left
    H_stitched[:, :H_left.shape[1]] = H_left[left_start_row:left_start_row+final_rows, :]
    
    # ✅ PASO 2: Preparar H_right alineado
    start_col = H_left.shape[1] - overlap_px
    right_data = H_right_corrected[right_start_row:right_start_row+final_rows, :]
    
    # ✅ PASO 3: Copiar la parte NO-overlap de H_right (después del overlap)
    # Solo la parte que NO se solapa
    H_stitched[:, H_left.shape[1]:] = right_data[:, overlap_px:]
    
    # ✅ PASO 4: Promediar en la región de overlap
    left_overlap = H_left[left_start_row:left_start_row+final_rows, -overlap_px:]
    right_overlap = H_right_corrected[right_start_row:right_start_row+final_rows, :overlap_px]
    
    if left_overlap.shape == right_overlap.shape:
        # Promediar donde ambos tienen datos válidos
        valid_left = ~np.isnan(left_overlap)
        valid_right = ~np.isnan(right_overlap)
        both_valid = valid_left & valid_right
        
        overlap_avg = np.full_like(left_overlap, np.nan)
        overlap_avg[both_valid] = (left_overlap[both_valid] + right_overlap[both_valid]) / 2.0
        overlap_avg[valid_left & ~valid_right] = left_overlap[valid_left & ~valid_right]
        overlap_avg[~valid_left & valid_right] = right_overlap[~valid_left & valid_right]
        
        # Escribir el promedio en la región de overlap
        H_stitched[:, start_col:H_left.shape[1]] = overlap_avg
    
    return H_stitched


def verify_overlap_quality(H_left, H_right, overlap_px, row_shift, delta_z, 
                          pixel_size_mm, pair_name="", output_dir=None) -> OverlapMetrics:
    """
    ✅ VERSIÓN CORREGIDA con protecciones contra arrays vacíos
    Calcula métricas de calidad del overlap entre dos heightmaps
    """
    
    # Extraer regiones de overlap alineadas
    min_rows = min(H_left.shape[0], H_right.shape[0])
    overlap_start_left = H_left.shape[1] - overlap_px
    overlap_end_left = H_left.shape[1]
    overlap_start_right = 0
    overlap_end_right = overlap_px
    
    # Aplicar row_shift
    if row_shift > 0:
        left_overlap = H_left[row_shift:row_shift+min_rows, overlap_start_left:overlap_end_left]
        right_overlap = H_right[0:min_rows, overlap_start_right:overlap_end_right]
    else:
        left_overlap = H_left[0:min_rows, overlap_start_left:overlap_end_left]
        right_overlap = H_right[-row_shift:-row_shift+min_rows, overlap_start_right:overlap_end_right]
    
    # Recortar a mismas dimensiones
    min_overlap_rows = min(left_overlap.shape[0], right_overlap.shape[0])
    min_overlap_cols = min(left_overlap.shape[1], right_overlap.shape[1])
    left_overlap = left_overlap[:min_overlap_rows, :min_overlap_cols]
    right_overlap = right_overlap[:min_overlap_rows, :min_overlap_cols]
    
    # Aplicar corrección de delta_z
    right_overlap_corrected = right_overlap - delta_z
    
    # Calcular diferencias
    valid_mask = ~np.isnan(left_overlap) & ~np.isnan(right_overlap_corrected)
    
    if np.sum(valid_mask) < 100:
        logger.warning(f"{pair_name}: Insufficient valid points for verification")
        return OverlapMetrics(
            mae_um=np.nan, rmse_um=np.nan, std_um=np.nan, 
            max_um=np.nan, n_points=0, verdict="INSUFFICIENT_DATA"
        )
    
    diff_mm = left_overlap[valid_mask] - right_overlap_corrected[valid_mask]
    diff_um = diff_mm * 1000  # Convertir a micrones
    
    # ✅ PROTECCIÓN 1: Check mínimo de puntos
    if len(diff_um) < 10:
        logger.warning(f"{pair_name}: Too few points ({len(diff_um)})")
        return OverlapMetrics(
            mae_um=np.nan, rmse_um=np.nan, std_um=np.nan, 
            max_um=np.nan, n_points=0, verdict="INSUFFICIENT_DATA"
        )
    
    # Filtrar outliers extremos (> 3σ) SOLO si hay variabilidad
    mean_diff = np.mean(diff_um)
    std_diff = np.std(diff_um)
    
    # ✅ PROTECCIÓN 2: Solo filtrar si STD > umbral mínimo
    if std_diff > 0.1:  # Solo filtrar si hay variabilidad real
        outlier_mask = np.abs(diff_um - mean_diff) < 3 * std_diff
        diff_um_filtered = diff_um[outlier_mask]
        n_outliers = len(diff_um) - len(diff_um_filtered)
    else:
        # Datos muy uniformes (sintéticos perfectos), no filtrar
        diff_um_filtered = diff_um
        n_outliers = 0
    
    # ✅ PROTECCIÓN 3: Verificar que quedaron puntos después de filtrar
    if len(diff_um_filtered) < 10:
        logger.warning(f"{pair_name}: All points filtered as outliers")
        # Usar datos sin filtrar
        diff_um_filtered = diff_um
        n_outliers = 0
    
    # Calcular métricas
    mae = np.mean(np.abs(diff_um_filtered))
    rmse = np.sqrt(np.mean(diff_um_filtered**2))
    std = np.std(diff_um_filtered)
    max_err = np.max(np.abs(diff_um_filtered))
    n_points = len(diff_um_filtered)
    
    # Determinar veredicto
    if mae < 5:
        verdict = "EXCELLENT"
    elif mae < 10:
        verdict = "GOOD"
    elif mae < 15:
        verdict = "ACCEPTABLE"
    else:
        verdict = "FAIL"
    
    metrics = OverlapMetrics(
        mae_um=mae, rmse_um=rmse, std_um=std, 
        max_um=max_err, n_points=n_points, verdict=verdict
    )
    
    # Log resultados
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERLAP VERIFICATION: {pair_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Overlap size: {overlap_px} px ({overlap_px*pixel_size_mm:.2f} mm)")
    logger.info(f"Valid points: {n_points:,}")
    if n_outliers > 0:
        logger.info(f"Outliers removed: {n_outliers} ({100*n_outliers/len(diff_um):.1f}%)")
    logger.info(f"\nERROR METRICS:")
    logger.info(f"  MAE:  {mae:.2f} µm")
    logger.info(f"  RMSE: {rmse:.2f} µm")
    logger.info(f"  STD:  {std:.2f} µm")
    logger.info(f"  MAX:  {max_err:.2f} µm")
    logger.info(f"\nVERDICT: {verdict}")
    logger.info(f"{'='*60}\n")
    
    return metrics


def stitch_two_heightmaps(H_left, H_right, config, override_overlap_mm=None, 
                         original_right_width=None, pair_name="", verify_quality=True,
                         verification_output_dir=None):
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

    print(f"\n🔍 DEBUG Pair {pair_name}:")
    print(f"  Overlap detected: {overlap_px} px ({overlap_px*config.pixel_size_mm:.2f} mm)")
    print(f"  Row shift: {row_shift} px")
    print(f"  Delta Z: {delta_z*1000:.2f} µm")
    print(f"  SNR: {snr:.2f}")
    print(f"  Used fallback: {used_fallback}")
    
    # Verificación de calidad (antes de ensamblar)
    overlap_metrics = None
    if verify_quality and not used_fallback:
        try:
            overlap_metrics = verify_overlap_quality(
                H_left, H_right, overlap_px, row_shift, delta_z,
                config.pixel_size_mm, pair_name, verification_output_dir
            )
        except Exception as e:
            logger.warning(f"Could not verify overlap quality: {e}")
    
    # ✅ Ensamblar heightmap con función CORREGIDA
    H_stitched = assemble_stitched_heightmap(H_left, H_right, overlap_px, row_shift, delta_z)
    valid_overlap_points = int(np.sum(~np.isnan(H_stitched[:, H_left.shape[1]-overlap_px:H_left.shape[1]])))
    
    logger.info(f"Complete: {H_stitched.shape}, overlap: {overlap_px*config.pixel_size_mm:.2f}mm")
    
    return StitchResult(
        heightmap=H_stitched, overlap_px=overlap_px, overlap_mm=overlap_px*config.pixel_size_mm,
        row_shift=row_shift, delta_z_mm=delta_z, delta_z_std_mm=delta_z_std,
        correlation_snr=snr, valid_overlap_points=valid_overlap_points,
        used_fallback=used_fallback,
        original_file_width_px=original_right_width if original_right_width else H_right.shape[1],
        overlap_metrics=overlap_metrics
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
    """HEIGHTMAP CON OVERLAPS MARCADOS"""
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
    
    # Marcar overlaps
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
        
        quality_label = ""
        if result.overlap_metrics:
            quality_label = f"\n{result.overlap_metrics.verdict}"
        
        ax.text(mid_overlap, height * 0.95, f'Overlap {i+1}\n{result.overlap_mm:.1f}mm{quality_label}', 
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
    """HEIGHTMAP LIMPIO"""
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
    """5 PERFILES CON OVERLAPS MARCADOS"""
    logger.info("Generating 5 continuity profiles...")
    
    fig, axes = plt.subplots(5, 1, figsize=(18, 14))
    rows_to_check = [H_final.shape[0]//5, 2*H_final.shape[0]//5, H_final.shape[0]//2, 
                     3*H_final.shape[0]//5, 4*H_final.shape[0]//5]
    
    for i, row in enumerate(rows_to_check):
        ax = axes[i]
        profile = H_final[row, :]
        valid = ~np.isnan(profile)
        
        cols_mm = np.arange(len(profile)) * pixel_size_mm
        
        ax.plot(cols_mm[valid], profile[valid]*1000, 'b-', linewidth=1.5, alpha=0.8)
        
        # Marcar overlaps
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


def generate_quality_report(all_results, output_file):
    """Genera reporte de texto con resumen de calidad"""
    
    n_total = len(all_results)
    n_excellent = sum(1 for r in all_results if r.overlap_metrics and r.overlap_metrics.verdict == "EXCELLENT")
    n_good = sum(1 for r in all_results if r.overlap_metrics and r.overlap_metrics.verdict == "GOOD")
    n_acceptable = sum(1 for r in all_results if r.overlap_metrics and r.overlap_metrics.verdict == "ACCEPTABLE")
    n_fail = sum(1 for r in all_results if r.overlap_metrics and r.overlap_metrics.verdict == "FAIL")
    n_pass = n_excellent + n_good + n_acceptable
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STITCHING QUALITY VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total pairs analyzed: {n_total}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"PASS:  {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)\n")
        f.write(f"FAIL:  {n_fail}/{n_total} ({100*n_fail/n_total:.1f}%)\n\n")
        
        f.write(f"  EXCELLENT:  {n_excellent}\n")
        f.write(f"  GOOD:       {n_good}\n")
        f.write(f"  ACCEPTABLE: {n_acceptable}\n")
        f.write(f"  FAIL:       {n_fail}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        for i, result in enumerate(all_results):
            pair_name = f"{i+1}->{i+2}"
            f.write(f"Pair {pair_name}:\n")
            f.write(f"  Overlap: {result.overlap_mm:.2f} mm ({result.overlap_px} px)\n")
            f.write(f"  Delta Z: {result.delta_z_mm*1000:.2f} µm\n")
            f.write(f"  SNR:     {result.correlation_snr:.2f}\n")
            
            if result.overlap_metrics:
                m = result.overlap_metrics
                f.write(f"  MAE:     {m.mae_um:.2f} µm\n")
                f.write(f"  RMSE:    {m.rmse_um:.2f} µm\n")
                f.write(f"  STD:     {m.std_um:.2f} µm\n")
                f.write(f"  MAX:     {m.max_um:.2f} µm\n")
                f.write(f"  Points:  {m.n_points:,}\n")
                f.write(f"  VERDICT: {m.verdict}\n")
            else:
                f.write(f"  VERDICT: NO_VERIFICATION\n")
            
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("-" * 80 + "\n")
        
        if n_fail == 0:
            if n_excellent + n_good >= n_total * 0.8:
                recommendation = "EXCELLENT quality."
            else:
                recommendation = "ACCEPTABLE quality."
        else:
            recommendation = f"WARNING: {n_fail} pair(s) FAILED."
        
        f.write(f"{recommendation}\n\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Quality report saved: {Path(output_file).name}")


def stitch_multiple_files(csv_files, output_dir=".", output_prefix="stitched", config=None, 
                         interactive=True, verify_quality=True):
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
    
    verification_dir = output_path / "verification" if verify_quality else None
    if verification_dir:
        verification_dir.mkdir(parents=True, exist_ok=True)
    
    H_result, pixel_size_mm = load_vr6000_csv(csv_files[0])
    config.pixel_size_mm = pixel_size_mm
    first_file_width_px = H_result.shape[1]
    
    all_results = []
    
    for i, csv_file in enumerate(csv_files[1:]):
        logger.info(f"\n[{i+2}/{len(csv_files)}] Stitching {Path(csv_file).name}...")
        H_next, _ = load_vr6000_csv(csv_file)
        original_next_width = H_next.shape[1]
        
        result = stitch_two_heightmaps(
            H_result, H_next, config, 
            override_overlap_mm=overlaps[i],
            original_right_width=original_next_width,
            pair_name=f"{i+1}->{i+2}",
            verify_quality=verify_quality,
            verification_output_dir=str(verification_dir) if verification_dir else None
        )
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
    
    if verify_quality:
        generate_quality_report(all_results, output_path / f"{output_prefix}_quality_report.txt")
    
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
        quality_str = ""
        if res.overlap_metrics:
            quality_str = f" | Quality: {res.overlap_metrics.verdict} (MAE={res.overlap_metrics.mae_um:.2f}µm)"
        logger.info(f"  Overlap {i+1}: {res.overlap_mm:.2f} mm{quality_str}")
    
    return H_result, all_results, pixel_size_mm


if __name__ == "__main__":
    config = StitchConfig(
        pixel_size_mm=0.007413, expected_overlap_mm=3.0, max_delta_z_mm=0.15,
        min_snr=3.0, max_row_shift_px=15, smoothing_sigma=1.0, use_fallback=True
    )
    
    csv_files = [
        r"C:\Users\javie\Desktop\Universidad\Cuarto-Univesrity of Rhode Island\Primer Cuatrimestre\ISE 401 TFG\ProyectoLaser\synthetic_tiles\synthetic_tile_1.csv",
        r"C:\Users\javie\Desktop\Universidad\Cuarto-Univesrity of Rhode Island\Primer Cuatrimestre\ISE 401 TFG\ProyectoLaser\synthetic_tiles\synthetic_tile_2.csv",
        r"C:\Users\javie\Desktop\Universidad\Cuarto-Univesrity of Rhode Island\Primer Cuatrimestre\ISE 401 TFG\ProyectoLaser\synthetic_tiles\synthetic_tile_3.csv",
        r"C:\Users\javie\Desktop\Universidad\Cuarto-Univesrity of Rhode Island\Primer Cuatrimestre\ISE 401 TFG\ProyectoLaser\synthetic_tiles\synthetic_tile_4.csv"
    ]
    
    H_final, results, pixel_size = stitch_multiple_files(
        csv_files, output_dir=".", output_prefix=f"heightmap_{datetime.now().strftime('%Y%m%d_%H%M')}",
        config=config, interactive=True, verify_quality=True
    )
    
    logger.info("✅ COMPLETED")