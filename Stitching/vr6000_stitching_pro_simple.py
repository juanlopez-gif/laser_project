"""
VR-6000 Height Map Stitching - Professional Version
====================================================
Refactored version with:
- Proper logging instead of prints
- Modular code (no duplication)
- Input validation and sanity checks
- Error handling
- Clean separation of concerns

Author: Juan López - Shipyard 4.0 MVP2
Date: 2025-11-21
"""

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StitchConfig:
    """Configuration parameters for stitching"""
    pixel_size_mm: float = 0.007413  # VR-6000 default
    expected_overlap_mm: float = 5.0
    max_delta_z_mm: float = 0.1  # Maximum allowed height difference
    min_snr: float = 5.0  # Minimum correlation SNR
    max_row_shift_px: int = 10  # Maximum vertical misalignment
    smoothing_sigma: float = 1.0


@dataclass
class StitchResult:
    """Results from stitching operation"""
    heightmap: np.ndarray
    overlap_px: int
    overlap_mm: float
    row_shift: int
    delta_z_mm: float
    delta_z_std_mm: float
    correlation_snr: float
    valid_overlap_points: int


# ============================================================================
# CORE UTILITIES
# ============================================================================

def validate_heightmap(H: np.ndarray, name: str) -> None:
    """Validate heightmap data"""
    if H is None or H.size == 0:
        raise ValueError(f"{name}: Empty heightmap")
    
    if H.ndim != 2:
        raise ValueError(f"{name}: Must be 2D array, got {H.ndim}D")
    
    valid_fraction = np.sum(~np.isnan(H)) / H.size
    if valid_fraction < 0.1:
        raise ValueError(f"{name}: Too many NaN values ({valid_fraction:.1%} valid)")
    
    logger.debug(f"{name} validated: {H.shape}, {valid_fraction:.1%} valid data")


def find_valid_range(H: np.ndarray) -> Tuple[int, int]:
    """Find column range with valid data"""
    valid_mask = ~np.isnan(H)
    cols_with_data = np.sum(valid_mask, axis=0)
    valid_cols = np.where(cols_with_data > H.shape[0] * 0.1)[0]
    
    if len(valid_cols) == 0:
        return 0, H.shape[1]
    
    return valid_cols[0], valid_cols[-1] + 1


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize data for correlation, handling NaN values"""
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
    """Apply Gaussian smoothing to heightmap"""
    return ndimage.gaussian_filter(H, sigma=sigma, mode='constant', cval=np.nan)


# ============================================================================
# CSV LOADING
# ============================================================================

def load_vr6000_csv(filepath: str) -> np.ndarray:
    """Load height map from VR-6000 CSV file"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find data start
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '"Height"':
                data_start = i + 1
                break
        
        if data_start == 0:
            data_start = 22
        
        # Parse data
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
        
        # Filter outliers
        H[H < -10] = np.nan
        H[H > 10] = np.nan
        
        logger.info(f"Loaded {Path(filepath).name}: {H.shape}, range [{np.nanmin(H):.4f}, {np.nanmax(H):.4f}] mm")
        
        return H
        
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        raise


# ============================================================================
# CORRELATION & ALIGNMENT
# ============================================================================

def extract_search_bands(H_left: np.ndarray, H_right: np.ndarray, 
                        expected_overlap_px: int, config: StitchConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Extract bands for correlation search"""
    left_start, left_end = find_valid_range(H_left)
    right_start, right_end = find_valid_range(H_right)
    
    logger.debug(f"Valid ranges - LEFT: [{left_start}:{left_end}], RIGHT: [{right_start}:{right_end}]")
    
    # Define search bands
    search_margin = int(expected_overlap_px * 0.5)
    band_width = expected_overlap_px + search_margin
    
    band_left_start = max(left_start, left_end - band_width)
    band_left = H_left[:, band_left_start:left_end].copy()
    
    band_right_end = min(right_end, right_start + band_width)
    band_right = H_right[:, right_start:band_right_end].copy()
    
    # Smooth and normalize
    band_left = smooth_heightmap(band_left, config.smoothing_sigma)
    band_right = smooth_heightmap(band_right, config.smoothing_sigma)
    
    band_left_norm, _ = normalize_data(band_left)
    band_right_norm, _ = normalize_data(band_right)
    
    return band_left_norm, band_right_norm


def find_overlap_via_correlation(band_left: np.ndarray, band_right: np.ndarray,
                                 H_left: np.ndarray, config: StitchConfig) -> Tuple[int, int, float]:
    """Find overlap using FFT correlation"""
    correlation = fftconvolve(band_left, band_right[::-1, ::-1], mode='full')
    
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    max_corr_value = correlation[max_idx]
    corr_row, corr_col = max_idx
    
    corr_std = np.nanstd(correlation)
    snr = max_corr_value / corr_std if corr_std > 0 else 0
    
    logger.debug(f"Correlation - max: {max_corr_value:.1f}, SNR: {snr:.2f}")
    
    # Validate SNR
    if snr < config.min_snr:
        raise ValueError(f"Low correlation SNR: {snr:.2f} < {config.min_snr}")
    
    # Calculate overlap
    overlap_cols = corr_col - (band_right.shape[1] - 1)
    
    # Calculate from original image bounds
    left_start, left_end = find_valid_range(H_left)
    band_width = band_left.shape[1]
    band_left_start = max(left_start, left_end - band_width)
    overlap_absolute = left_end - (band_left_start + overlap_cols)
    
    # Validate overlap
    if overlap_absolute <= 0 or overlap_absolute > min(H_left.shape[1], band_right.shape[1]):
        raise ValueError(f"Invalid overlap: {overlap_absolute} pixels")
    
    # Calculate row shift
    row_shift = corr_row - (band_left.shape[0] - 1)
    
    if abs(row_shift) > config.max_row_shift_px:
        raise ValueError(f"Excessive vertical shift: {row_shift} px > {config.max_row_shift_px} px")
    
    return overlap_absolute, row_shift, snr


# ============================================================================
# HEIGHT ALIGNMENT
# ============================================================================

def calculate_height_offset(H_left: np.ndarray, H_right: np.ndarray,
                           overlap_px: int, row_shift: int, 
                           config: StitchConfig) -> Tuple[float, float]:
    """Calculate median height offset between overlapping regions"""
    min_rows = min(H_left.shape[0], H_right.shape[0])
    
    overlap_start_left = H_left.shape[1] - overlap_px
    overlap_end_left = H_left.shape[1]
    overlap_start_right = 0
    overlap_end_right = overlap_px
    
    # Extract regions accounting for row shift
    if row_shift > 0:
        left_crop = H_left[row_shift:row_shift+min_rows, overlap_start_left:overlap_end_left]
        right_crop = H_right[0:min_rows, overlap_start_right:overlap_end_right]
    else:
        left_crop = H_left[0:min_rows, overlap_start_left:overlap_end_left]
        right_crop = H_right[-row_shift:-row_shift+min_rows, overlap_start_right:overlap_end_right]
    
    # Align shapes
    min_overlap_rows = min(left_crop.shape[0], right_crop.shape[0])
    min_overlap_cols = min(left_crop.shape[1], right_crop.shape[1])
    region_left = left_crop[:min_overlap_rows, :min_overlap_cols]
    region_right = right_crop[:min_overlap_rows, :min_overlap_cols]
    
    valid_overlap = ~np.isnan(region_left) & ~np.isnan(region_right)
    
    if np.sum(valid_overlap) < 10:
        logger.warning("Very few valid overlap points")
        return 0.0, 0.0
    
    diff = region_right[valid_overlap] - region_left[valid_overlap]
    delta_z = np.nanmedian(diff)
    delta_z_std = np.nanstd(diff)
    
    logger.debug(f"Height offset: {delta_z*1000:.2f} ± {delta_z_std*1000:.2f} μm")
    
    # Validate delta Z
    if abs(delta_z) > config.max_delta_z_mm:
        raise ValueError(f"Excessive height difference: {delta_z*1000:.1f} μm > {config.max_delta_z_mm*1000:.1f} μm")
    
    return delta_z, delta_z_std


# ============================================================================
# STITCHING ASSEMBLY
# ============================================================================

def assemble_stitched_heightmap(H_left: np.ndarray, H_right: np.ndarray,
                                overlap_px: int, row_shift: int,
                                delta_z: float) -> np.ndarray:
    """Assemble final stitched heightmap"""
    # Correct height offset
    H_right_corrected = H_right - delta_z
    
    # Calculate final dimensions
    if row_shift > 0:
        usable_rows_left = H_left.shape[0] - row_shift
        usable_rows_right = H_right.shape[0]
    else:
        usable_rows_left = H_left.shape[0]
        usable_rows_right = H_right.shape[0] + row_shift
    
    final_rows = min(usable_rows_left, usable_rows_right)
    final_cols = H_left.shape[1] + H_right.shape[1] - overlap_px
    
    H_stitched = np.full((final_rows, final_cols), np.nan)
    
    # Place LEFT
    if row_shift > 0:
        H_stitched[:, :H_left.shape[1]] = H_left[row_shift:row_shift+final_rows, :]
    else:
        H_stitched[:, :H_left.shape[1]] = H_left[0:final_rows, :]
    
    # Place RIGHT
    start_col = H_left.shape[1] - overlap_px
    if row_shift > 0:
        H_stitched[:, start_col:start_col+H_right.shape[1]] = H_right_corrected[0:final_rows, :]
    else:
        offset = -row_shift
        H_stitched[:, start_col:start_col+H_right.shape[1]] = H_right_corrected[offset:offset+final_rows, :]
    
    # Average overlap region
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
        H_stitched[:, overlap_region] = np.nanmean(stacked, axis=2)
    
    return H_stitched


# ============================================================================
# MAIN STITCHING FUNCTION
# ============================================================================

def stitch_two_heightmaps(H_left: np.ndarray, H_right: np.ndarray,
                         config: Optional[StitchConfig] = None) -> StitchResult:
    """
    Stitch two height maps together
    
    Parameters:
    -----------
    H_left, H_right : np.ndarray
        Height maps to stitch
    config : StitchConfig
        Configuration parameters
        
    Returns:
    --------
    StitchResult with heightmap and metadata
    """
    if config is None:
        config = StitchConfig()
    
    # Validate inputs
    validate_heightmap(H_left, "H_left")
    validate_heightmap(H_right, "H_right")
    
    logger.info(f"Stitching: {H_left.shape} + {H_right.shape}")
    
    # Calculate expected overlap in pixels
    expected_overlap_px = int(config.expected_overlap_mm / config.pixel_size_mm)
    logger.debug(f"Expected overlap: {config.expected_overlap_mm}mm = {expected_overlap_px}px")
    
    # Extract and correlate search bands
    band_left, band_right = extract_search_bands(H_left, H_right, expected_overlap_px, config)
    overlap_px, row_shift, snr = find_overlap_via_correlation(band_left, band_right, H_left, config)
    
    logger.info(f"Found overlap: {overlap_px}px = {overlap_px*config.pixel_size_mm:.2f}mm, row_shift: {row_shift}px")
    
    # Calculate height offset
    delta_z, delta_z_std = calculate_height_offset(H_left, H_right, overlap_px, row_shift, config)
    
    # Assemble final heightmap
    H_stitched = assemble_stitched_heightmap(H_left, H_right, overlap_px, row_shift, delta_z)
    
    # Count valid overlap points for quality metric
    valid_overlap_points = int(np.sum(~np.isnan(H_stitched[:, H_left.shape[1]-overlap_px:H_left.shape[1]])))
    
    logger.info(f"Stitching complete: {H_stitched.shape}, SNR: {snr:.1f}")
    
    return StitchResult(
        heightmap=H_stitched,
        overlap_px=overlap_px,
        overlap_mm=overlap_px * config.pixel_size_mm,
        row_shift=row_shift,
        delta_z_mm=delta_z,
        delta_z_std_mm=delta_z_std,
        correlation_snr=snr,
        valid_overlap_points=valid_overlap_points
    )


# ============================================================================
# VISUALIZATION - MODULAR PLOTTING
# ============================================================================

def plot_single_heightmap(ax, H, extent, title, overlap_mm=None, vmin=None, vmax=None):
    """Plot a single heightmap with optional overlap line"""
    im = ax.imshow(H, cmap='terrain', aspect='auto', extent=extent,
                   vmin=vmin, vmax=vmax, origin='lower')
    
    if overlap_mm is not None:
        ax.axvline(overlap_mm, color='red', linewidth=3, linestyle='--',
                  label=f'Overlap: {overlap_mm:.2f}mm')
        ax.legend(fontsize=10)
    
    ax.set_xlabel('Width (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white')
    plt.colorbar(im, ax=ax, label='Height (mm)')
    
    return im


def plot_left_right(H_left, H_right, pixel_size_mm, vmin, vmax):
    """Plot original left and right heightmaps"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # LEFT
    left_width = H_left.shape[1] * pixel_size_mm
    left_height = H_left.shape[0] * pixel_size_mm
    extent_left = [0, left_width, 0, left_height]
    plot_single_heightmap(ax1, H_left, extent_left, 
                         f'LEFT Original\n{left_width:.1f} × {left_height:.1f} mm',
                         vmin=vmin, vmax=vmax)
    
    # RIGHT
    right_width = H_right.shape[1] * pixel_size_mm
    right_height = H_right.shape[0] * pixel_size_mm
    extent_right = [0, right_width, 0, right_height]
    plot_single_heightmap(ax2, H_right, extent_right,
                         f'RIGHT Original\n{right_width:.1f} × {right_height:.1f} mm',
                         vmin=vmin, vmax=vmax)
    
    plt.tight_layout()
    return fig


def plot_stitched_full(H_stitched, H_left, result, pixel_size_mm, vmin, vmax):
    """Plot full stitched result with overlap markers"""
    fig, ax = plt.subplots(figsize=(18, 6))
    
    width = H_stitched.shape[1] * pixel_size_mm
    height = H_stitched.shape[0] * pixel_size_mm
    extent = [0, width, 0, height]
    
    im = ax.imshow(H_stitched, cmap='terrain', aspect='auto', extent=extent,
                   vmin=vmin, vmax=vmax, origin='lower')
    
    # Mark overlap region
    left_end_mm = H_left.shape[1] * pixel_size_mm - result.overlap_mm
    left_boundary_mm = H_left.shape[1] * pixel_size_mm
    
    ax.axvline(left_end_mm, color='red', linewidth=3, linestyle='--',
              label='Overlap start', alpha=0.8)
    ax.axvline(left_boundary_mm, color='cyan', linewidth=3, linestyle='--',
              label='LEFT end', alpha=0.8)
    ax.axvspan(left_end_mm, left_boundary_mm, alpha=0.2, color='yellow')
    
    ax.set_xlabel('Total width (mm)', fontsize=11)
    ax.set_ylabel('Height (mm)', fontsize=11)
    ax.set_title(f'STITCHED RESULT\n{width:.1f} × {height:.1f} mm | Overlap: {result.overlap_mm:.2f}mm | SNR: {result.correlation_snr:.1f}',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, color='white')
    plt.colorbar(im, ax=ax, label='Height (mm)')
    
    plt.tight_layout()
    return fig


def plot_zoom_region(H_stitched, H_left, result, pixel_size_mm, vmin, vmax):
    """Plot zoomed view of stitching region"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    overlap_px = result.overlap_px
    zoom_start_px = max(0, H_left.shape[1] - 2*overlap_px)
    zoom_end_px = min(H_stitched.shape[1], H_left.shape[1] + 2*overlap_px)
    zoom_data = H_stitched[:, zoom_start_px:zoom_end_px]
    
    zoom_start_mm = zoom_start_px * pixel_size_mm
    zoom_end_mm = zoom_end_px * pixel_size_mm
    height_mm = H_stitched.shape[0] * pixel_size_mm
    extent_zoom = [zoom_start_mm, zoom_end_mm, 0, height_mm]
    
    im = ax.imshow(zoom_data, cmap='terrain', aspect='auto', extent=extent_zoom,
                   vmin=vmin, vmax=vmax, origin='lower')
    
    left_end_mm = H_left.shape[1] * pixel_size_mm - result.overlap_mm
    left_boundary_mm = H_left.shape[1] * pixel_size_mm
    
    ax.axvline(left_end_mm, color='red', linewidth=3, linestyle='--', alpha=0.9)
    ax.axvline(left_boundary_mm, color='cyan', linewidth=3, linestyle='--', alpha=0.9)
    ax.axvspan(left_end_mm, left_boundary_mm, alpha=0.25, color='yellow')
    
    ax.set_xlabel('Position (mm)', fontsize=11)
    ax.set_ylabel('Height (mm)', fontsize=11)
    ax.set_title('ZOOM - Stitching Region', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white')
    plt.colorbar(im, ax=ax, label='Height (mm)')
    
    plt.tight_layout()
    return fig


def plot_profiles(H_stitched, H_left, result):
    """Plot horizontal profiles through stitching region"""
    fig, axes = plt.subplots(5, 1, figsize=(16, 12))
    
    overlap_start = H_left.shape[1] - result.overlap_px
    overlap_end = H_left.shape[1]
    
    rows_to_check = [
        H_stitched.shape[0] // 4,
        H_stitched.shape[0] // 3,
        H_stitched.shape[0] // 2,
        2 * H_stitched.shape[0] // 3,
        3 * H_stitched.shape[0] // 4
    ]
    
    for i, row in enumerate(rows_to_check):
        ax = axes[i]
        profile = H_stitched[row, :]
        valid = ~np.isnan(profile)
        cols = np.arange(len(profile))
        
        ax.plot(cols[valid], profile[valid], 'b-', linewidth=1, alpha=0.8)
        ax.axvline(overlap_start, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(overlap_end, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvspan(overlap_start, overlap_end, alpha=0.15, color='yellow')
        
        ax.set_ylabel('Height (mm)', fontsize=10)
        ax.set_title(f'Profile at row {row} ({100*row/H_stitched.shape[0]:.0f}%)', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Column (pixels)', fontsize=11)
    plt.suptitle('Profile Continuity Verification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def generate_verification_plots(H_left: np.ndarray, H_right: np.ndarray,
                               result: StitchResult, pixel_size_mm: float,
                               output_prefix: str) -> List[str]:
    """Generate all verification plots"""
    logger.info("Generating verification plots...")
    
    H_stitched = result.heightmap
    vmin = np.nanmin(H_stitched)
    vmax = np.nanmax(H_stitched)
    
    output_files = []
    
    # 1. Left & Right originals
    fig = plot_left_right(H_left, H_right, pixel_size_mm, vmin, vmax)
    fname = f"{output_prefix}_originals.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    output_files.append(fname)
    logger.debug(f"Saved {fname}")
    
    # 2. Full stitched result
    fig = plot_stitched_full(H_stitched, H_left, result, pixel_size_mm, vmin, vmax)
    fname = f"{output_prefix}_stitched.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    output_files.append(fname)
    logger.debug(f"Saved {fname}")
    
    # 3. Zoom region
    fig = plot_zoom_region(H_stitched, H_left, result, pixel_size_mm, vmin, vmax)
    fname = f"{output_prefix}_zoom.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    output_files.append(fname)
    logger.debug(f"Saved {fname}")
    
    # 4. Profiles
    fig = plot_profiles(H_stitched, H_left, result)
    fname = f"{output_prefix}_profiles.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    output_files.append(fname)
    logger.debug(f"Saved {fname}")
    
    logger.info(f"Generated {len(output_files)} verification plots")
    return output_files


# ============================================================================
# MULTI-FILE STITCHING
# ============================================================================

def stitch_multiple_files(csv_files: List[str], 
                         output_dir: str = ".",
                         output_prefix: str = "stitched",
                         config: Optional[StitchConfig] = None) -> Tuple[np.ndarray, List[StitchResult]]:
    """
    Stitch multiple CSV files in sequence
    
    Parameters:
    -----------
    csv_files : list of str
        CSV file paths in order (left to right)
    output_dir : str
        Output directory
    output_prefix : str
        Prefix for output files
    config : StitchConfig
        Configuration parameters
        
    Returns:
    --------
    H_final : np.ndarray
        Final stitched heightmap
    results : list of StitchResult
        Results from each stitching operation
    """
    if config is None:
        config = StitchConfig()
    
    if len(csv_files) < 2:
        raise ValueError("Need at least 2 CSV files")
    
    logger.info(f"="*70)
    logger.info(f"STITCHING {len(csv_files)} FILES")
    logger.info(f"="*70)
    
    # Load first file
    logger.info(f"[1/{len(csv_files)}] Loading base...")
    H_result = load_vr6000_csv(csv_files[0])
    
    all_results = []
    
    # Stitch sequentially
    for i, csv_file in enumerate(csv_files[1:], start=2):
        logger.info(f"[{i}/{len(csv_files)}] Stitching {Path(csv_file).name}...")
        
        H_next = load_vr6000_csv(csv_file)
        result = stitch_two_heightmaps(H_result, H_next, config)
        
        H_result = result.heightmap
        all_results.append(result)
        
        logger.info(f"Result: {H_result.shape}, overlap: {result.overlap_mm:.2f}mm, ΔZ: {result.delta_z_mm*1000:.2f}μm")
    
    # Save result
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_path / f"{output_prefix}.csv"
    np.savetxt(output_csv, H_result, delimiter=',', fmt='%.6f')
    logger.info(f"Saved result: {output_csv}")
    
    # Generate plots (for 2-file case)
    if len(csv_files) == 2:
        H_first = load_vr6000_csv(csv_files[0])
        H_last = load_vr6000_csv(csv_files[1])
        generate_verification_plots(
            H_first, H_last, all_results[0],
            config.pixel_size_mm,
            str(output_path / output_prefix)
        )
    
    # Summary
    logger.info(f"="*70)
    logger.info("SUMMARY")
    logger.info(f"="*70)
    logger.info(f"Files processed: {len(csv_files)}")
    logger.info(f"Final shape: {H_result.shape}")
    logger.info(f"Dimensions: {H_result.shape[1]*config.pixel_size_mm:.2f} × {H_result.shape[0]*config.pixel_size_mm:.2f} mm")
    logger.info(f"Height range: [{np.nanmin(H_result):.4f}, {np.nanmax(H_result):.4f}] mm")
    
    if all_results:
        avg_overlap = np.mean([r.overlap_mm for r in all_results])
        avg_snr = np.mean([r.correlation_snr for r in all_results])
        logger.info(f"Average overlap: {avg_overlap:.2f}mm")
        logger.info(f"Average SNR: {avg_snr:.1f}")
    
    return H_result, all_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # Example configuration
    config = StitchConfig(
        pixel_size_mm=0.007413,
        expected_overlap_mm=4.5,
        max_delta_z_mm=0.1,
        min_snr=5.0,
        max_row_shift_px=10,
        smoothing_sigma=1.0
    )

    # List your CSV files
    csv_files = [
        "/home/isecapstone/project_laser/files/1.3Height.csv",
        "/home/isecapstone/project_laser/files/1.4Height.csv"
    ]

    # ✅ Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ✅ Run stitching con parámetros correctos
    H_final, results = stitch_multiple_files(
        csv_files,
        output_dir=".",  # Guardar en directorio actual
        output_prefix=f"heightmap_FINAL_PRO_{timestamp}",
        config=config
    )

    logger.info("✅ COMPLETED")