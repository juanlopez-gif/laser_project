import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import ezdxf
import cv2
from datetime import datetime
from scipy.spatial.distance import cdist

# --- CONFIGURACIÓN ---
file = "VR-20251110_173541_Height.csv"
skip_header = 22
pixel_size = 1.853e-3

# --- LECTURA ---
print(f"Leyendo archivo: {file}")
z = pd.read_csv(file, skiprows=skip_header, header=None).to_numpy()
ny, nx = z.shape

print(f"\n--- INFORMACIÓN DEL SCAN ---")
print(f"Dimensiones: {nx} x {ny} píxeles")
print(f"Área física: {nx*pixel_size:.2f} x {ny*pixel_size:.2f} mm")
print(f"Altura mínima: {np.nanmin(z):.3f} mm")
print(f"Altura máxima: {np.nanmax(z):.3f} mm")
print(f"Rango total: {np.nanmax(z) - np.nanmin(z):.3f} mm")
print(f"Valores NaN: {np.isnan(z).sum()} ({100*np.isnan(z).sum()/z.size:.1f}%)")

# --- COORDENADAS ---
x = np.arange(nx) * pixel_size
y = np.arange(ny) * pixel_size
X, Y = np.meshgrid(x, y)

# *** COLORMAP CORRECTO DEL VR-6200 ***
vmin = -0.063
vmax = 0.05

colors = ['#00008B', '#0000FF', '#00FFFF', '#FFFF00', '#FFA500', '#FF0000']
positions = [0.0, 0.20, 0.38, 0.56, 0.73, 1.0]

cmap_custom = LinearSegmentedColormap.from_list('vr6200', 
                                                 list(zip(positions, colors)),
                                                 N=256)

# --- VISUALIZACIÓN 2D (4 gráficas) ---
fig1, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1) Mapa de alturas
im1 = axes[0, 0].imshow(z, extent=[x.min(), x.max(), y.max(), y.min()],
                         cmap=cmap_custom, aspect='auto',
                         vmin=vmin, vmax=vmax)
axes[0, 0].set_xlabel('X [mm]')
axes[0, 0].set_ylabel('Y [mm]')
axes[0, 0].set_title('Mapa de Alturas')
axes[0, 0].set_xlim(0, x.max())
axes[0, 0].set_ylim(y.max(), 0)
cbar1 = plt.colorbar(im1, ax=axes[0, 0], label='Altura (mm)')

# 2) Histograma
axes[0, 1].hist(z[~np.isnan(z)].flatten(), bins=100, color='steelblue', edgecolor='black')
axes[0, 1].axvline(x=0.05, color='red', linewidth=2, linestyle='--', label='Max (0.05)')
axes[0, 1].axvline(x=-0.063, color='darkblue', linewidth=2, linestyle='--', label='Min (-0.063)')
axes[0, 1].axvline(x=0, color='#FFFF00', linewidth=1.5, linestyle='-', label='Ref (0)')
axes[0, 1].set_xlabel('Altura [mm]')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].set_title('Distribución de Alturas')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# 3) Perfil X
mid_y = ny // 2
axes[1, 0].plot(x, z[mid_y, :], 'b-', linewidth=1.5)
axes[1, 0].axhline(y=0.05, color='red', linewidth=1, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-0.063, color='darkblue', linewidth=1, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=0, color='#FFFF00', linewidth=1, linestyle='-', alpha=0.7)
axes[1, 0].set_xlabel('X [mm]')
axes[1, 0].set_ylabel('Altura [mm]')
axes[1, 0].set_title(f'Perfil en Y = {y[mid_y]:.2f} mm')
axes[1, 0].set_xlim(0, x.max())
axes[1, 0].grid(True, alpha=0.3)

# 4) Perfil Y
mid_x = nx // 2
axes[1, 1].plot(y, z[:, mid_x], 'r-', linewidth=1.5)
axes[1, 1].axhline(y=0.05, color='red', linewidth=1, linestyle='--', alpha=0.7)
axes[1, 1].axhline(y=-0.063, color='darkblue', linewidth=1, linestyle='--', alpha=0.7)
axes[1, 1].axhline(y=0, color='#FFFF00', linewidth=1, linestyle='-', alpha=0.7)
axes[1, 1].set_xlabel('Y [mm]')
axes[1, 1].set_ylabel('Altura [mm]')
axes[1, 1].set_title(f'Perfil en X = {x[mid_x]:.2f} mm')
axes[1, 1].set_xlim(0, y.max())
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# --- 3D ---
fig2 = plt.figure(figsize=(14, 10))
ax = fig2.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, z, cmap=cmap_custom, 
                       linewidth=0, antialiased=True,
                       alpha=0.9,
                       vmin=vmin, vmax=vmax)

ax.invert_yaxis()

ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Altura [mm]')
ax.set_title('Vista 3D del Scan VR-6200')
ax.set_xlim(0, x.max())
ax.set_ylim(y.max(), 0)

cbar2 = fig2.colorbar(surf, ax=ax, shrink=0.5, label='Altura (mm)')

ax.set_box_aspect([nx*pixel_size, ny*pixel_size, (vmax-vmin)*5])
ax.view_init(elev=30, azim=45)

# --- MAPA DE DEFECTOS >0.07 ---
threshold = 0.07
defectos = z > threshold

fig3, ax3 = plt.subplots(figsize=(10, 8))

# Crear máscara binaria
mapa_defectos = np.zeros_like(z)
mapa_defectos[defectos] = 1

im3 = ax3.imshow(mapa_defectos, extent=[x.min(), x.max(), y.max(), y.min()],
                 cmap='Reds', aspect='auto', vmin=0, vmax=1, alpha=0.5)
ax3.set_xlabel('X [mm]')
ax3.set_ylabel('Y [mm]')
ax3.set_title(f'Defectos > {threshold} mm con Trayectoria Optimizada')
ax3.set_xlim(0, x.max())
ax3.set_ylim(y.max(), 0)
ax3.grid(True, alpha=0.3, color='black')

num_defectos = np.sum(defectos)
area_defectos = num_defectos * (pixel_size ** 2)

plt.tight_layout()

# *** ANÁLISIS Y GENERACIÓN DE ARCHIVO DXF ***
print(f"\n--- ANÁLISIS DE DEFECTOS (>{threshold} mm) ---")
print(f"Píxeles defectuosos: {num_defectos} ({100*num_defectos/z.size:.2f}%)")
print(f"Área total defectos: {area_defectos:.4f} mm²")

# --- GENERAR DXF CON CONTORNOS Y TRAYECTORIA OPTIMIZADA ---
# Extraer nombre base del CSV y timestamp
csv_basename = file.replace('.csv', '').replace('_Height', '')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dxf = f"{csv_basename}_LaserPath_{timestamp}.dxf"

if num_defectos > 0:
    # Convertir máscara a uint8 para OpenCV
    defectos_uint8 = (defectos * 255).astype(np.uint8)
    
    # Encontrar contornos con OpenCV
    contours, hierarchy = cv2.findContours(defectos_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n✓ Encontrados {len(contours)} contornos de defectos")
    
    # --- CALCULAR CENTROIDES DE CADA CONTORNO ---
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            centroids.append([cx, cy])
        else:
            # Si no hay momento, usar el primer punto
            centroids.append([contour[0][0][0], contour[0][0][1]])
    
    centroids = np.array(centroids)
    
    # --- OPTIMIZAR TRAYECTORIA (NEAREST NEIGHBOR) ---
    if len(centroids) > 1:
        # Empezar desde el centroide más cercano a (0,0)
        distances_from_origin = np.linalg.norm(centroids, axis=1)
        start_idx = np.argmin(distances_from_origin)
        
        trayectoria_indices = [start_idx]
        remaining = list(range(len(centroids)))
        remaining.remove(start_idx)
        
        while remaining:
            last_idx = trayectoria_indices[-1]
            last_centroid = centroids[last_idx]
            
            # Encontrar el más cercano
            distances = [np.linalg.norm(centroids[i] - last_centroid) for i in remaining]
            nearest_idx = remaining[np.argmin(distances)]
            
            trayectoria_indices.append(nearest_idx)
            remaining.remove(nearest_idx)
    else:
        trayectoria_indices = list(range(len(centroids)))
    
    # Reordenar contornos según trayectoria optimizada
    contours_ordered = [contours[i] for i in trayectoria_indices]
    
    # --- CALCULAR LONGITUD TOTAL DE TRAYECTORIA ---
    total_travel_distance = 0
    for i in range(len(centroids) - 1):
        idx_current = trayectoria_indices[i]
        idx_next = trayectoria_indices[i + 1]
        dist_px = np.linalg.norm(centroids[idx_next] - centroids[idx_current])
        dist_mm = dist_px * pixel_size
        total_travel_distance += dist_mm
    
    print(f"✓ Trayectoria optimizada - Distancia total: {total_travel_distance:.2f} mm")
    
    # --- VISUALIZAR TRAYECTORIA EN EL MAPA ---
    for i in range(len(trayectoria_indices)):
        idx = trayectoria_indices[i]
        cx_mm = centroids[idx][0] * pixel_size
        cy_mm = centroids[idx][1] * pixel_size
        
        # Dibujar centroide
        ax3.plot(cx_mm, cy_mm, 'bo', markersize=8, markeredgecolor='black', markeredgewidth=1)
        ax3.text(cx_mm + 0.02, cy_mm + 0.02, str(i+1), fontsize=10, color='blue', fontweight='bold')
        
        # Dibujar línea a siguiente centroide
        if i < len(trayectoria_indices) - 1:
            idx_next = trayectoria_indices[i + 1]
            cx_next_mm = centroids[idx_next][0] * pixel_size
            cy_next_mm = centroids[idx_next][1] * pixel_size
            ax3.plot([cx_mm, cx_next_mm], [cy_mm, cy_next_mm], 'g-', linewidth=2, alpha=0.7)
    
    ax3.text(0.02, 0.98, 
             f'Defectos: {num_defectos} px\nÁrea: {area_defectos:.4f} mm²\nZonas: {len(contours)}\nTrayectoria: {total_travel_distance:.2f} mm',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- GENERAR ARCHIVO DXF ---
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Añadir información como texto en el DXF
    msp.add_text(
        f"Source: {file}",
        dxfattribs={'layer': 'INFO', 'height': 0.1}
    ).set_placement((0, -0.3))
    
    msp.add_text(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        dxfattribs={'layer': 'INFO', 'height': 0.1}
    ).set_placement((0, -0.5))
    
    msp.add_text(
        f"Threshold: >{threshold}mm | Defects: {len(contours)} | Travel: {total_travel_distance:.2f}mm",
        dxfattribs={'layer': 'INFO', 'height': 0.1}
    ).set_placement((0, -0.7))
    
    # Dibujar contornos en orden optimizado
    for i, contour_idx in enumerate(trayectoria_indices):
        contour = contours[contour_idx]
        
        # Convertir de índices de píxel a coordenadas físicas
        points_physical = []
        for point in contour:
            col, row = point[0]
            x_pos = col * pixel_size
            y_pos = row * pixel_size
            points_physical.append((x_pos, y_pos))
        
        # Añadir polyline cerrada
        if len(points_physical) >= 3:
            msp.add_lwpolyline(
                points_physical,
                dxfattribs={'layer': f'DEFECT_{i+1:03d}', 'color': 1, 'closed': True}
            )
            
            # Añadir número de orden en el centroide
            cx_mm = centroids[contour_idx][0] * pixel_size
            cy_mm = centroids[contour_idx][1] * pixel_size
            msp.add_text(
                str(i + 1),
                dxfattribs={'layer': 'ORDER', 'height': 0.05, 'color': 3}
            ).set_placement((cx_mm, cy_mm))
    
    # Añadir líneas de trayectoria entre centroides
    for i in range(len(trayectoria_indices) - 1):
        idx_current = trayectoria_indices[i]
        idx_next = trayectoria_indices[i + 1]
        
        cx1 = centroids[idx_current][0] * pixel_size
        cy1 = centroids[idx_current][1] * pixel_size
        cx2 = centroids[idx_next][0] * pixel_size
        cy2 = centroids[idx_next][1] * pixel_size
        
        msp.add_line((cx1, cy1), (cx2, cy2), dxfattribs={'layer': 'TRAJECTORY', 'color': 2})
    
    doc.saveas(output_dxf)
    print(f"✓ Archivo DXF generado: {output_dxf}")
    print(f"  Contornos exportados: {len(contours)} (en orden optimizado)")
    
    # Calcular área total
    total_area = 0
    for contour in contours:
        area_px = cv2.contourArea(contour)
        area_mm2 = area_px * (pixel_size ** 2)
        total_area += area_mm2
    
    print(f"  Área total contornos: {total_area:.4f} mm²")
else:
    print("\n⚠ No hay defectos >0.07mm para generar DXF")

# *** MOSTRAR GRÁFICOS ***
print("\n✓ Visualización completada")
print(f"\nDefectos ROJOS (>0.05): {np.sum(z > 0.05)} px ({100*np.sum(z > 0.05)/z.size:.2f}%)")
print(f"Defectos AZULES (<-0.063): {np.sum(z < -0.063)} px ({100*np.sum(z < -0.063)/z.size:.2f}%)")

plt.show()