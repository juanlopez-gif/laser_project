import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

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

# --- INTERPOLACIÓN PARA 3D SUAVIZADO ---
print("\n--- GENERANDO SUPERFICIE 3D SUAVIZADA ---")
factor = 3  # Factor de interpolación (3x más denso)

# Crear grid interpolado
x_interp = np.linspace(x.min(), x.max(), nx * factor)
y_interp = np.linspace(y.min(), y.max(), ny * factor)
X_interp, Y_interp = np.meshgrid(x_interp, y_interp)

# Preparar datos para interpolación (eliminar NaN)
points = np.column_stack([X.ravel(), Y.ravel()])
values = z.ravel()
mask = ~np.isnan(values)

print(f"Interpolando de {nx}x{ny} a {nx*factor}x{ny*factor} puntos...")

# Interpolar con método cúbico para mayor suavidad
z_interp = griddata(points[mask], values[mask], (X_interp, Y_interp), method='cubic')

print("✓ Interpolación completada")

# --- 3D SUAVIZADO ---
fig2 = plt.figure(figsize=(14, 10))
ax = fig2.add_subplot(111, projection='3d')

surf = ax.plot_surface(X_interp, Y_interp, z_interp, 
                       cmap=cmap_custom, 
                       linewidth=0, 
                       antialiased=True,
                       alpha=0.9,
                       vmin=vmin, 
                       vmax=vmax)

ax.invert_yaxis()

ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Altura [mm]')
ax.set_title('Vista 3D Suavizada del Scan VR-6200')
ax.set_xlim(0, x.max())
ax.set_ylim(y.max(), 0)

cbar2 = fig2.colorbar(surf, ax=ax, shrink=0.5, label='Altura (mm)')

ax.set_box_aspect([nx*pixel_size, ny*pixel_size, (vmax-vmin)*5])
ax.view_init(elev=30, azim=45)

# *** ESTADÍSTICAS FINALES ***
print("\n✓ Visualización completada")
print(f"\nDefectos ROJOS (>0.05): {np.sum(z > 0.05)} px ({100*np.sum(z > 0.05)/z.size:.2f}%)")
print(f"Defectos AZULES (<-0.063): {np.sum(z < -0.063)} px ({100*np.sum(z < -0.063)/z.size:.2f}%)")

plt.show()