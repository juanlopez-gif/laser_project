"""
SYNTHETIC VALIDATION - VR-6000 Stitching Algorithm
====================================================
Test básico: superficie con ondulación suave + ruido realista
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ============================================================================
# PASO 1: GENERAR PIEZA SINTÉTICA
# ============================================================================

def generate_synthetic_part(length_mm=20.0, width_mm=6.0, pixel_size_mm=0.007413):
    """
    Genera heightmap sintético SIMPLE pero realista
    
    Características:
    - Ondulación suave (simula deformación mecánica leve)
    - Ruido gaussiano (simula ruido del sensor VR-6000)
    - Base plana con leve inclinación
    """
    
    # Calcular dimensiones en pixeles
    nx = int(length_mm / pixel_size_mm)
    ny = int(width_mm / pixel_size_mm)
    
    print(f"Generating synthetic part: {length_mm}×{width_mm} mm")
    print(f"Resolution: {nx}×{ny} pixels ({pixel_size_mm*1000:.3f} µm/pixel)")
    
    # Crear grids de coordenadas
    x = np.linspace(0, length_mm, nx)
    y = np.linspace(0, width_mm, ny)
    X, Y = np.meshgrid(x, y)
    
    # Componente 1: Base plana con leve inclinación (10 µm/mm en X)
    Z_base = 0.01 * X  # 10 µm cada mm en dirección X
    
    # Componente 2: Ondulación suave (simula deformación)
    # Amplitud de 30 µm, periodo de 8 mm
    Z_wave = 0.030 * np.sin(2 * np.pi * X / 8.0)
    
    # Componente 3: Ruido del sensor (±2 µm STD, típico del VR-6000)
    np.random.seed(42)  # Reproducible
    Z_noise = np.random.normal(0, 0.002, (ny, nx))  # 2 µm standard deviation
    
    # Combinar todas las componentes
    Z_total = Z_base + Z_wave + Z_noise
    
    print(f"Height range: [{Z_total.min()*1000:.1f}, {Z_total.max()*1000:.1f}] µm")
    
    return X, Y, Z_total


# ============================================================================
# PASO 2: GUARDAR COMO CSV EN FORMATO VR-6000
# ============================================================================

def save_as_vr6000_csv(Z, pixel_size_mm, filepath, tile_number=1):
    """
    Guarda heightmap en formato idéntico al VR-6000
    Compatible con tu función load_vr6000_csv()
    """
    
    rows, cols = Z.shape
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Header (formato exacto del VR-6000)
        f.write('"Measured date","2025-12-10 12:00:00"\n')
        f.write('"Model","VR-6000"\n')
        f.write('"Measurement unit model","VR-6100"\n')
        f.write('"Data type","ImageDataCsv"\n')
        f.write('"File version","1000"\n')
        f.write(f'"Measurement data name","SYNTHETIC_TILE_{tile_number}"\n')
        f.write('"Viewer capture method","Manual"\n')
        f.write('"Camera magnification","High magnification"\n')
        f.write('"Magnification","40"\n')
        f.write('"Capture image type","Normal"\n')
        f.write('"Height measurement mode","Fine"\n')
        f.write('"Measurement direction","Both sides"\n')
        f.write(f'"XY Calibration","{pixel_size_mm*1000:.3f}","um"\n')  # ✅ CRÍTICO
        f.write('"Output image data","Height"\n')
        f.write(f'"Horizontal","{cols}"\n')
        f.write(f'"Vertical","{rows}"\n')
        f.write(f'"Minimum value","{Z.min():.6f}"\n')
        f.write(f'"Maximum value","{Z.max():.6f}"\n')
        f.write('"Unit","mm"\n')
        f.write('"Reference data name",""\n')
        f.write('\n')
        f.write('"Height"\n')
        
        
        # Data: cada fila como valores separados por comas
        for row in Z:
            # Formato: valores entre comillas, coma-separados
            line = ','.join([f'"{v:.6f}"' for v in row])
            f.write(line + '\n')
    
    print(f"✓ Saved: {Path(filepath).name} ({rows}×{cols})")


# ============================================================================
# PASO 3: DIVIDIR EN TILES CON OVERLAP
# ============================================================================

def split_into_tiles(Z, pixel_size_mm, tile_width_mm=7.6, overlap_mm=3.0, output_dir="."):
    """
    Divide heightmap en tiles con overlap (simula escaneo real)
    
    Args:
        Z: heightmap completo
        tile_width_mm: ancho de cada tile (típicamente ~7.6mm para VR-6000)
        overlap_mm: overlap entre tiles consecutivos
        output_dir: directorio donde guardar tiles
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calcular step en mm y pixeles
    step_mm = tile_width_mm - overlap_mm
    step_px = int(step_mm / pixel_size_mm)
    tile_width_px = int(tile_width_mm / pixel_size_mm)
    
    total_width_px = Z.shape[1]
    
    print(f"\n{'='*60}")
    print(f"SPLITTING INTO TILES")
    print(f"{'='*60}")
    print(f"Tile width: {tile_width_mm} mm ({tile_width_px} px)")
    print(f"Overlap: {overlap_mm} mm ({tile_width_px - step_px} px)")
    print(f"Step: {step_mm} mm ({step_px} px)")
    
    # Generar tiles
    tiles = []
    tile_files = []
    tile_num = 1
    
    start_px = 0
    while start_px < total_width_px:
        end_px = min(start_px + tile_width_px, total_width_px)
        
        # Extraer tile
        tile = Z[:, start_px:end_px]
        
        # Guardar como CSV
        filename = output_path / f"synthetic_tile_{tile_num}.csv"
        save_as_vr6000_csv(tile, pixel_size_mm, filename, tile_num)
        
        tiles.append(tile)
        tile_files.append(str(filename))
        
        print(f"  Tile {tile_num}: columns [{start_px}:{end_px}] → {tile.shape}")
        
        tile_num += 1
        start_px += step_px
        
        # Si el siguiente tile sería muy pequeño, terminar
        if start_px + tile_width_px // 2 >= total_width_px:
            break
    
    print(f"{'='*60}")
    print(f"Generated {len(tiles)} tiles with {overlap_mm}mm overlap")
    print(f"{'='*60}\n")
    
    return tiles, tile_files


# ============================================================================
# PASO 4: VISUALIZACIÓN
# ============================================================================

def plot_synthetic_part(X, Y, Z, tiles, output_file="synthetic_part.png"):
    """Plot del heightmap original y división en tiles"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Heightmap completo
    im = axes[0].imshow(Z * 1000, cmap='viridis', aspect='auto', 
                       extent=[0, X.max(), 0, Y.max()], origin='lower')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    axes[0].set_title('Synthetic Part - Ground Truth', fontweight='bold', fontsize=14)
    plt.colorbar(im, ax=axes[0], label='Height (µm)')
    
    # Marcar divisiones de tiles
    pixel_size = X[0, 1] - X[0, 0]
    cumulative_width = 0
    for i, tile in enumerate(tiles):
        tile_width_mm = tile.shape[1] * pixel_size
        if i > 0:  # No marcar el inicio del primer tile
            axes[0].axvline(cumulative_width, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7)
        cumulative_width += tile_width_mm
        if i < len(tiles) - 1:  # No marcar el final del último tile
            axes[0].axvline(cumulative_width, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7)
    
    # Plot 2: Perfil central
    center_row = Z.shape[0] // 2
    profile = Z[center_row, :] * 1000
    x_profile = np.linspace(0, X.max(), len(profile))
    
    axes[1].plot(x_profile, profile, 'b-', linewidth=2, label='Height profile')
    axes[1].set_xlabel('X (mm)', fontsize=12)
    axes[1].set_ylabel('Height (µm)', fontsize=12)
    axes[1].set_title('Center Profile (Y = 3mm)', fontweight='bold', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Marcar tiles en el perfil
    cumulative_width = 0
    for i, tile in enumerate(tiles):
        tile_width_mm = tile.shape[1] * pixel_size
        if i > 0:
            axes[1].axvline(cumulative_width, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
        cumulative_width += tile_width_mm
        if i < len(tiles) - 1:
            axes[1].axvline(cumulative_width, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_file}")
    plt.close()


# ============================================================================
# MAIN: GENERAR TODO
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("SYNTHETIC VR-6000 DATA GENERATOR")
    print("="*70)
    print()
    
    # Parámetros
    PIXEL_SIZE_MM = 0.007413  # VR-6000 típico
    PART_LENGTH_MM = 20.0     # Largo de la pieza
    PART_WIDTH_MM = 6.0       # Ancho de la pieza
    TILE_WIDTH_MM = 7.6       # Ancho de cada tile
    OVERLAP_MM = 3.0          # Overlap entre tiles
    OUTPUT_DIR = "./synthetic_tiles"
    
    # Generar pieza sintética
    X, Y, Z_ground_truth = generate_synthetic_part(
        length_mm=PART_LENGTH_MM,
        width_mm=PART_WIDTH_MM,
        pixel_size_mm=PIXEL_SIZE_MM
    )
    
    # Dividir en tiles
    tiles, tile_files = split_into_tiles(
        Z_ground_truth,
        pixel_size_mm=PIXEL_SIZE_MM,
        tile_width_mm=TILE_WIDTH_MM,
        overlap_mm=OVERLAP_MM,
        output_dir=OUTPUT_DIR
    )
    
    # Visualizar
    plot_synthetic_part(X, Y, Z_ground_truth, tiles, 
                       output_file=f"{OUTPUT_DIR}/synthetic_part_visualization.png")
    
    # Guardar ground truth completo para comparación posterior
    save_as_vr6000_csv(Z_ground_truth, PIXEL_SIZE_MM, 
                      f"{OUTPUT_DIR}/ground_truth_FULL.csv", tile_number=0)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files in: {OUTPUT_DIR}/")
    print(f"  - {len(tile_files)} tile CSV files")
    print(f"  - 1 ground truth CSV file")
    print(f"  - 1 visualization PNG\n")
    
    print("NEXT STEPS:")
    print("-" * 70)
    print("1. Run your stitching algorithm on the tile files:")
    print(f"   python tu_script_stitching.py")
    print()
    print("2. Compare stitched result with ground_truth_FULL.csv")
    print()
    print("3. Expected result if algorithm is perfect:")
    print("   MAE < 3 µm  (limited by 2µm noise we added)")
    print("="*70)
    
    # Print lista de archivos generados
    print("\nGenerated tile files:")
    for i, f in enumerate(tile_files, 1):
        print(f"  {i}. {Path(f).name}")