#!/usr/bin/env python3
"""
TEST GENERATOR - Add deliberate offsets to test stitching correction
======================================================================
"""
import numpy as np
from pathlib import Path

# CONFIG
PIXEL_SIZE = 0.007413  # mm
PART_LENGTH = 20.0  # mm
PART_WIDTH = 6.0    # mm
TILE_WIDTH = 7.6    # mm
OVERLAP = 3.0       # mm
OUTPUT_DIR = "synthetic_tiles"

nx = int(PART_LENGTH / PIXEL_SIZE)
ny = int(PART_WIDTH / PIXEL_SIZE)

print(f"Generating {PART_LENGTH}×{PART_WIDTH} mm part")
print(f"Resolution: {nx}×{ny} pixels")

# BASE: Flat with variation
Z = np.random.normal(0, 0.015, (ny, nx))

# Add peaks
n_peaks = 8
for _ in range(n_peaks):
    px = np.random.randint(0, nx)
    py = np.random.randint(0, ny)
    for i in range(ny):
        for j in range(nx):
            dist = np.sqrt((i-py)**2 + (j-px)**2)
            if dist < 50:
                Z[i, j] += 0.025 * np.exp(-(dist**2)/500)

print(f"Height range: [{Z.min()*1000:.1f}, {Z.max()*1000:.1f}] µm")

# SAVE GROUND TRUTH
Path(OUTPUT_DIR).mkdir(exist_ok=True)

with open(f'{OUTPUT_DIR}/ground_truth_FULL.csv', 'w') as f:
    f.write('"XY Calibration","7.413","um"\n')
    f.write(f'"Horizontal","{nx}"\n')
    f.write(f'"Vertical","{ny}"\n')
    f.write('"Height"\n')
    for row in Z:
        f.write(','.join([f'"{v:.6f}"' for v in row]) + '\n')

print(f"✓ Saved: ground_truth_FULL.csv")

# SPLIT INTO TILES + ADD DELIBERATE OFFSETS
tile_width_px = int(TILE_WIDTH / PIXEL_SIZE)
overlap_px = int(OVERLAP / PIXEL_SIZE)
step_px = tile_width_px - overlap_px

print(f"\nSplitting with DELIBERATE OFFSETS:")
print(f"  Tile width: {TILE_WIDTH}mm ({tile_width_px}px)")
print(f"  Overlap: {OVERLAP}mm ({overlap_px}px)")

# ✅ Offsets deliberados en altura (para que el stitching los corrija)
offsets_um = [0, 20, -15, 10]  # µm

tile_num = 1
start = 0

while start < nx:
    end = min(start + tile_width_px, nx)
    tile = Z[:, start:end].copy()
    
    # ✅ AÑADIR OFFSET DELIBERADO
    offset_mm = offsets_um[tile_num-1] / 1000.0
    tile = tile + offset_mm
    
    print(f"  Tile {tile_num}: cols [{start}:{end}] → offset: {offsets_um[tile_num-1]:+.0f} µm")
    
    with open(f'{OUTPUT_DIR}/synthetic_tile_{tile_num}.csv', 'w') as f:
        f.write('"XY Calibration","7.413","um"\n')
        f.write(f'"Horizontal","{tile.shape[1]}"\n')
        f.write(f'"Vertical","{tile.shape[0]}"\n')
        f.write('"Height"\n')
        for row in tile:
            f.write(','.join([f'"{v:.6f}"' for v in row]) + '\n')
    
    tile_num += 1
    start += step_px
    
    if start + tile_width_px//2 >= nx:
        break

print(f"\n✅ DONE! Generated {tile_num-1} tiles with deliberate offsets")
print("\n⚠️  EXPECTED BEHAVIOR:")
print("   1. Stitching should detect delta_z = offset values")
print("   2. Stitching should CORRECT these offsets")
print("   3. Validation should show MAE ~2-5 µm (residual from correction)")
print("\nNEXT:")
print("  1. python comprobacion2_stitching_FIXED.py")
print("  2. python validate_FIXED.py")