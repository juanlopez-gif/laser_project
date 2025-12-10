import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, LightSource
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

# --- PREPARAR DATOS PARA INTERPOLACIÓN ---
points = np.column_stack([X.ravel(), Y.ravel()])
values = z.ravel()
mask = ~np.isnan(values)

# =============================================================================
# MÉTODO 1: SUBMUESTREO (Más rápido, menos "Minecraft")
# =============================================================================
print("\n--- MÉTODO 1: SUBMUESTREO ---")
step = 4  # Mostrar 1 de cada 4 puntos
X_sub = X[::step, ::step]
Y_sub = Y[::step, ::step]
z_sub = z[::step, ::step]

fig1 = plt.figure(figsize=(14, 10))
ax1 = fig1.add_subplot(111, projection='3d')

surf1 = ax1.plot_surface(X_sub, Y_sub, z_sub, 
                         cmap=cmap_custom, 
                         linewidth=0, 
                         antialiased=True,
                         alpha=0.9,
                         vmin=vmin, 
                         vmax=vmax)

ax1.invert_yaxis()
ax1.set_xlabel('X [mm]')
ax1.set_ylabel('Y [mm]')
ax1.set_zlabel('Altura [mm]')
ax1.set_title('MÉTODO 1: Submuestreo (1 de cada 4 puntos)\nMás rápido, menos detalle', 
              fontsize=12, fontweight='bold')
ax1.set_xlim(0, x.max())
ax1.set_ylim(y.max(), 0)
fig1.colorbar(surf1, ax=ax1, shrink=0.5, label='Altura (mm)')
ax1.set_box_aspect([nx*pixel_size, ny*pixel_size, (vmax-vmin)*5])
ax1.view_init(elev=30, azim=45)

print("✓ Método 1 completado")

# =============================================================================
# MÉTODO 2: INTERPOLACIÓN + GOURAUD SHADING (Más suave)
# =============================================================================
print("\n--- MÉTODO 2: INTERPOLACIÓN + GOURAUD SHADING ---")
factor = 3

x_interp = np.linspace(x.min(), x.max(), nx * factor)
y_interp = np.linspace(y.min(), y.max(), ny * factor)
X_interp, Y_interp = np.meshgrid(x_interp, y_interp)

print(f"Interpolando de {nx}x{ny} a {nx*factor}x{ny*factor} puntos...")
z_interp = griddata(points[mask], values[mask], (X_interp, Y_interp), method='cubic')

fig2 = plt.figure(figsize=(14, 10))
ax2 = fig2.add_subplot(111, projection='3d')

# CLAVE: shade=True para Gouraud shading
surf2 = ax2.plot_surface(X_interp, Y_interp, z_interp, 
                         cmap=cmap_custom, 
                         linewidth=0, 
                         antialiased=True,
                         shade=True,  # ← ACTIVAR SHADING
                         alpha=0.9,
                         vmin=vmin, 
                         vmax=vmax)

ax2.invert_yaxis()
ax2.set_xlabel('X [mm]')
ax2.set_ylabel('Y [mm]')
ax2.set_zlabel('Altura [mm]')
ax2.set_title('MÉTODO 2: Interpolación + Gouraud Shading\nSuave y detallado', 
              fontsize=12, fontweight='bold')
ax2.set_xlim(0, x.max())
ax2.set_ylim(y.max(), 0)
fig2.colorbar(surf2, ax=ax2, shrink=0.5, label='Altura (mm)')
ax2.set_box_aspect([nx*pixel_size, ny*pixel_size, (vmax-vmin)*5])
ax2.view_init(elev=30, azim=45)

print("✓ Método 2 completado")

# =============================================================================
# MÉTODO 3: INTERPOLACIÓN + LIGHTSOURCE (Iluminación realista)
# =============================================================================
print("\n--- MÉTODO 3: INTERPOLACIÓN + LIGHTSOURCE ---")

fig3 = plt.figure(figsize=(14, 10))
ax3 = fig3.add_subplot(111, projection='3d')

# Crear fuente de luz personalizada
ls = LightSource(azdeg=315, altdeg=45)

# Calcular colores con iluminación
# Normalizar z para el colormap
z_norm = (z_interp - vmin) / (vmax - vmin)
z_norm = np.clip(z_norm, 0, 1)

# Aplicar colormap
rgb = cmap_custom(z_norm)

# Aplicar iluminación (shade_rgb requiere que rgb sea (n, m, 3))
illuminated = ls.shade_rgb(rgb[:, :, :3], z_interp)

surf3 = ax3.plot_surface(X_interp, Y_interp, z_interp,
                         facecolors=illuminated,
                         linewidth=0,
                         antialiased=True,
                         shade=False,  # Ya aplicamos shade manualmente
                         alpha=0.9)

ax3.invert_yaxis()
ax3.set_xlabel('X [mm]')
ax3.set_ylabel('Y [mm]')
ax3.set_zlabel('Altura [mm]')
ax3.set_title('MÉTODO 3: Interpolación + LightSource\nIluminación realista con sombras', 
              fontsize=12, fontweight='bold')
ax3.set_xlim(0, x.max())
ax3.set_ylim(y.max(), 0)

# Colorbar manual (porque usamos facecolors)
from matplotlib import cm
from matplotlib.colors import Normalize
norm = Normalize(vmin=vmin, vmax=vmax)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap_custom)
fig3.colorbar(mappable, ax=ax3, shrink=0.5, label='Altura (mm)')

ax3.set_box_aspect([nx*pixel_size, ny*pixel_size, (vmax-vmin)*5])
ax3.view_init(elev=30, azim=45)

print("✓ Método 3 completado")

# =============================================================================
# MOSTRAR COMPARACIÓN
# =============================================================================
print("\n" + "="*60)
print("COMPARACIÓN DE MÉTODOS:")
print("="*60)
print("MÉTODO 1 (Submuestreo):")
print("  ✓ Muy rápido de renderizar")
print("  ✓ Menos efecto 'Minecraft'")
print("  ✗ Pierde detalles finos")
print()
print("MÉTODO 2 (Interpolación + Gouraud):")
print("  ✓ Superficie muy suave")
print("  ✓ Mantiene todos los detalles")
print("  ✓ Shading automático")
print("  ✗ Más lento (interpolación)")
print()
print("MÉTODO 3 (Interpolación + LightSource):")
print("  ✓ Iluminación más realista")
print("  ✓ Muestra relieve con sombras")
print("  ✓ Aspecto más profesional")
print("  ✗ Configuración más compleja")
print("="*60)
print("\n✓ Las 3 gráficas están listas. Compara visualmente.")
print("  Usa el ratón para rotar y hacer zoom en cada una.")

plt.show()