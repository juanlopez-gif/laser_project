#!/usr/bin/env python3
"""
REALISTIC Synthetic Data Generator
===================================
Simulates real VR-6000 scanning with:
- Per-tile Z offset (calibration drift)
- Per-tile noise (sensor noise)
- Small tilt variation
"""
import numpy as np
from pathlib import Path

# CONFIG
PIXEL_SIZE = 0.007413  # mm
#PART_LENGTH = 20.0  # mm
PART_WIDTH = 6.0    # mm
TILE_WIDTH = 7.6    # mm
OVERLAP = 3.0       # mm
OUTPUT_DIR = "synthetic_tiles"

# REALISM PARAMETERS
Z_OFFSET_RANGE = 0.010  # ±10 µm offset per tile (calibration drift)
NOISE_STD = 0.015       # 15 µm noise per tile (sensor noise)
TILT_RANGE = 0.0001     # ±0.1 µm/mm tilt variation

# Calculate pixels
nx = int(PART_LENGTH / PIXEL_SIZE)
ny = int(PART_WIDTH / PIXEL_SIZE)

print(f"Generating REALISTIC {PART_LENGTH}×{PART_WIDTH} mm part")
print(f"Resolution: {nx}×{ny} pixels")
print(f"Realism: Z offset ±{Z_OFFSET_RANGE*1000:.1f}µm, noise {NOISE_STD*1000:.1f}µm STD")

# BASE GEOMETRY: Almost flat with small peaks
Z_base = np.zeros((ny, nx))

# Add small "picos" (peaks) - random locations
n_peaks = 8
for _ in range(n_peaks):
    px = np.random.randint(0, nx)
    py = np.random.randint(0, ny)
    for i in range(ny):
        for j in range(nx):
            dist = np.sqrt((i-py)**2 + (j-px)**2)
            if dist < 50:
                Z_base[i, j] += 0.025 * np.exp(-(dist**2)/500)  # 25 µm peak

print(f"Base geometry range: [{Z_base.min()*1000:.1f}, {Z_base.max()*1000:.1f}] µm")

# SAVE GROUND TRUTH (without noise/offset)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

with open(f'{OUTPUT_DIR}/ground_truth_IDEAL.csv', 'w') as f:
    f.write('"XY Calibration","7.413","um"\n')
    f.write(f'"Horizontal","{nx}"\n')
    f.write(f'"Vertical","{ny}"\n')
    f.write('"Height"\n')
    for row in Z_base:
        f.write(','.join([f'"{v:.6f}"' for v in row]) + '\n')

print(f"✓ Saved: ground_truth_IDEAL.csv (no noise/offset)")

# SPLIT INTO TILES WITH REALISTIC VARIATIONS
tile_width_px = int(TILE_WIDTH / PIXEL_SIZE)
overlap_px = int(OVERLAP / PIXEL_SIZE)
step_px = tile_width_px - overlap_px

print(f"\nSplitting into tiles with realistic variations:")
print(f"  Tile width: {TILE_WIDTH}mm ({tile_width_px}px)")
print(f"  Overlap: {OVERLAP}mm ({overlap_px}px)")
print(f"  Step: {TILE_WIDTH - OVERLAP}mm ({step_px}px)")

tile_num = 1
start = 0
tile_offsets = []

while start < nx:
    end = min(start + tile_width_px, nx)
    
    # Extract base geometry
    tile_base = Z_base[:, start:end].copy()
    
    # ✅ ADD REALISTIC VARIATIONS PER TILE:
    
    # 1. Random Z offset (simulates calibration drift between scans)
    z_offset = np.random.uniform(-Z_OFFSET_RANGE, Z_OFFSET_RANGE)
    
    # 2. Random noise (simulates sensor noise)
    noise = np.random.normal(0, NOISE_STD, tile_base.shape)
    
    # 3. Small tilt (simulates slight angle variation)
    tilt_x = np.random.uniform(-TILT_RANGE, TILT_RANGE)
    tilt_y = np.random.uniform(-TILT_RANGE, TILT_RANGE)
    
    x_coords = np.arange(tile_base.shape[1]) * PIXEL_SIZE
    y_coords = np.arange(tile_base.shape[0]) * PIXEL_SIZE
    X, Y = np.meshgrid(x_coords, y_coords)
    tilt = tilt_x * X + tilt_y * Y
    
    # Combine all effects
    tile_realistic = tile_base + z_offset + noise + tilt
    
    # Save tile
    with open(f'{OUTPUT_DIR}/synthetic_tile_{tile_num}.csv', 'w') as f:
        f.write('"XY Calibration","7.413","um"\n')
        f.write(f'"Horizontal","{tile_realistic.shape[1]}"\n')
        f.write(f'"Vertical","{tile_realistic.shape[0]}"\n')
        f.write('"Height"\n')
        for row in tile_realistic:
            f.write(','.join([f'"{v:.6f}"' for v in row]) + '\n')
    
    tile_offsets.append(z_offset)
    
    print(f"  Tile {tile_num}: cols [{start}:{end}] → {tile_realistic.shape}")
    print(f"    Z offset: {z_offset*1000:+.2f} µm | Tilt: ({tilt_x*1000:.3f}, {tilt_y*1000:.3f}) µm/mm")
    
    tile_num += 1
    start += step_px
    
    if start + tile_width_px//2 >= nx:
        break

# CALCULATE EXPECTED OVERLAP ERRORS
print(f"\n{'='*70}")
print("EXPECTED OVERLAP ERRORS (before stitching correction):")
print(f"{'='*70}")

for i in range(len(tile_offsets)-1):
    delta_z = tile_offsets[i+1] - tile_offsets[i]
    print(f"  Pair {i+1}→{i+2}: ΔZ = {delta_z*1000:+.2f} µm")
    print(f"    Expected MAE: ~{NOISE_STD*1000:.1f} µm (from sensor noise)")

print(f"\n✅ DONE! Generated {tile_num-1} REALISTIC tiles in {OUTPUT_DIR}/")
print(f"\nREALISM SUMMARY:")
print(f"  - Each tile has random Z offset: ±{Z_OFFSET_RANGE*1000:.1f} µm")
print(f"  - Each tile has independent noise: {NOISE_STD*1000:.1f} µm STD")
print(f"  - Each tile has small tilt: ±{TILT_RANGE*1000:.3f} µm/mm")
print(f"  - Overlap regions are NOT identical (simulates real scans)")
print("\nNEXT:")
print("  1. python comprobacion2_stitching_FIXED.py")
print("  2. python validate_FIXED.py")
print("\nEXPECTED RESULTS:")
print(f"  - MAE should be ~{NOISE_STD*1000:.1f} µm (not 0!)")
print(f"  - Delta Z correction should find the Z offsets")
print(f"  - Stitched result should be close to ground_truth_IDEAL.csv")