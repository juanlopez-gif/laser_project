#!/usr/bin/env python3
"""
CRITICAL TEST - Verify overlap regions in stitched result
==========================================================
This checks if the stitching actually changed values in overlap regions
"""
import numpy as np

def load_csv(filepath):
    """Load CSV and return numpy array"""
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    data_start = 0
    for i, line in enumerate(lines):
        if 'Height' in line and line.strip().startswith('"Height"'):
            data_start = i + 1
            break
    
    data = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        try:
            row = [float(v.strip('"')) for v in line.strip().split(',') if v.strip() and v.strip() != '']
            if row:
                data.append(row)
        except ValueError:
            # Skip lines that can't be converted to float (headers)
            continue
    
    return np.array(data)

print("="*70)
print("CRITICAL OVERLAP TEST")
print("="*70)

# Load files
print("\nLoading files...")
truth = load_csv('synthetic_tiles/ground_truth_FULL.csv')
stitched = load_csv('heightmap_20251210_1152_FINAL.csv')

print(f"Ground truth shape: {truth.shape}")
print(f"Stitched shape: {stitched.shape}")

# Test overlaps
# Tile boundaries based on 1025px width, 404px overlap
# Tile 1: cols 0-1025
# Tile 2: cols 621-1646 (overlap: 621-1025)
# Tile 3: cols 1242-2267 (overlap: 1242-1646)
# Tile 4: cols 1863-2697 (overlap: 1863-2267)

overlaps = [
    ("Overlap 1 (Tile 1-2)", 621, 1025),
    ("Overlap 2 (Tile 2-3)", 1242, 1646),
    ("Overlap 3 (Tile 3-4)", 1863, 2267)
]

print("\n" + "="*70)
print("OVERLAP REGION ANALYSIS")
print("="*70)

for name, start, end in overlaps:
    print(f"\n{name}: cols [{start}:{end}]")
    
    overlap_truth = truth[:, start:end]
    overlap_stitched = stitched[:, start:end]
    
    diff = (overlap_truth - overlap_stitched) * 1000  # µm
    
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_err = np.max(np.abs(diff))
    
    print(f"  MAE:  {mae:.3f} µm")
    print(f"  RMSE: {rmse:.3f} µm")
    print(f"  MAX:  {max_err:.3f} µm")
    
    if mae < 0.001:
        print(f"  ⚠️  WARNING: MAE = 0! This means no averaging happened!")
    elif mae < 1.0:
        print(f"  ✅ GOOD: Small error from averaging")
    else:
        print(f"  ❌ FAIL: Large error detected")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print("""
If ALL overlaps show MAE = 0.000 µm:
  → The stitching is NOT actually averaging overlap regions
  → There might be a bug in assemble_stitched_heightmap()
  → DO NOT send email yet

If overlaps show MAE > 0 µm:
  → The stitching IS working correctly
  → Small MAE means algorithm is good
  → OK to send email
""")

print("="*70)