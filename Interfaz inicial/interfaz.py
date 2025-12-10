import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider, TextBox
from scipy.interpolate import griddata
import cv2

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
factor = 3

x_interp = np.linspace(x.min(), x.max(), nx * factor)
y_interp = np.linspace(y.min(), y.max(), ny * factor)
X_interp, Y_interp = np.meshgrid(x_interp, y_interp)

points = np.column_stack([X.ravel(), Y.ravel()])
values = z.ravel()
mask = ~np.isnan(values)

print(f"Interpolando de {nx}x{ny} a {nx*factor}x{ny*factor} puntos...")
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

# --- INTERFAZ INTERACTIVA DE FILTROS ---
fig3 = plt.figure(figsize=(14, 10))

# Espacio para gráfica principal y leyenda externa
ax_main = plt.subplot2grid((6, 3), (0, 0), rowspan=5, colspan=2)
ax_legend = plt.subplot2grid((6, 3), (0, 2), rowspan=5)
ax_legend.axis('off')

# Valores iniciales
nivel1_init = 0.070
nivel2_init = 0.065
nivel3_init = 0.060

# Colores
color1 = '#000000'  # Negro
color2 = '#FF1493'  # Rosa fuerte
color3 = '#00FF00'  # Verde fosforito

# Función para contar nodos (contornos) usando OpenCV
def contar_nodos(mascara):
    if np.sum(mascara) == 0:
        return 0
    mascara_uint8 = (mascara * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mascara_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

# Crear mapa inicial SIN INTERPOLACIÓN
def crear_mapa_niveles(n1, n2, n3):
    mapa = np.ones((*z.shape, 3))  # RGB
    
    # Nivel 3 (más bajo) - Verde fosforito
    mask3 = (z >= n3) & (z < n2)
    mapa[mask3] = [0, 1, 0]  # Verde
    
    # Nivel 2 - Rosa
    mask2 = (z >= n2) & (z < n1)
    mapa[mask2] = [1, 0.078, 0.576]  # Rosa
    
    # Nivel 1 (más alto) - Negro
    mask1 = z >= n1
    mapa[mask1] = [0, 0, 0]  # Negro
    
    return mapa, mask1, mask2, mask3

mapa_inicial, mask1, mask2, mask3 = crear_mapa_niveles(nivel1_init, nivel2_init, nivel3_init)
im_filtro = ax_main.imshow(mapa_inicial, extent=[x.min(), x.max(), y.max(), y.min()],
                           aspect='auto', origin='upper')

ax_main.set_xlabel('X [mm]')
ax_main.set_ylabel('Y [mm]')
ax_main.set_xlim(0, x.max())
ax_main.set_ylim(y.max(), 0)
ax_main.grid(True, alpha=0.3)

# Sliders para los 3 niveles
ax_s1 = plt.axes([0.08, 0.12, 0.55, 0.02])
ax_s2 = plt.axes([0.08, 0.08, 0.55, 0.02])
ax_s3 = plt.axes([0.08, 0.04, 0.55, 0.02])

slider1 = Slider(ax_s1, 'Nivel 1 (Negro)', 0.0, 0.15, valinit=nivel1_init, 
                 valstep=0.001, color='black')
slider2 = Slider(ax_s2, 'Nivel 2 (Rosa)', 0.0, 0.15, valinit=nivel2_init, 
                 valstep=0.001, color='deeppink')
slider3 = Slider(ax_s3, 'Nivel 3 (Verde)', 0.0, 0.15, valinit=nivel3_init, 
                 valstep=0.001, color='lime')

# Variable para almacenar el texto de la leyenda
legend_text = None

# Función de actualización
def update(val):
    global legend_text
    
    n1 = slider1.val
    n2 = slider2.val
    n3 = slider3.val
    
    # Asegurar orden correcto
    if n2 >= n1:
        n2 = n1 - 0.001
        slider2.set_val(n2)
    if n3 >= n2:
        n3 = n2 - 0.001
        slider3.set_val(n3)
    
    mapa_actualizado, mask1, mask2, mask3 = crear_mapa_niveles(n1, n2, n3)
    im_filtro.set_data(mapa_actualizado)
    
    # Calcular estadísticas
    pixels1 = np.sum(mask1)
    pixels2 = np.sum(mask2)
    pixels3 = np.sum(mask3)
    
    area1 = pixels1 * (pixel_size ** 2)
    area2 = pixels2 * (pixel_size ** 2)
    area3 = pixels3 * (pixel_size ** 2)
    
    # Contar nodos
    nodos1 = contar_nodos(mask1)
    nodos2 = contar_nodos(mask2)
    nodos3 = contar_nodos(mask3)
    
    # Actualizar título con valores actuales
    ax_main.set_title(
        f'Filtrado por Niveles | N1≥{n1:.3f}mm | N2:[{n2:.3f}-{n1:.3f})mm | N3:[{n3:.3f}-{n2:.3f})mm',
        fontsize=11, fontweight='bold'
    )
    
    # Limpiar leyenda anterior
    ax_legend.clear()
    ax_legend.axis('off')
    
    # Crear nueva leyenda con estadísticas
    legend_info = (
        f"NIVEL 1 (Negro)\n"
        f"Umbral: ≥ {n1:.3f} mm\n"
        f"Píxeles: {pixels1}\n"
        f"Área: {area1:.4f} mm²\n"
        f"Nodos: {nodos1}\n"
        f"\n"
        f"NIVEL 2 (Rosa)\n"
        f"Umbral: [{n2:.3f}, {n1:.3f}) mm\n"
        f"Píxeles: {pixels2}\n"
        f"Área: {area2:.4f} mm²\n"
        f"Nodos: {nodos2}\n"
        f"\n"
        f"NIVEL 3 (Verde)\n"
        f"Umbral: [{n3:.3f}, {n2:.3f}) mm\n"
        f"Píxeles: {pixels3}\n"
        f"Área: {area3:.4f} mm²\n"
        f"Nodos: {nodos3}\n"
        f"\n"
        f"RESTO (Blanco)\n"
        f"Umbral: < {n3:.3f} mm"
    )
    
    ax_legend.text(0.05, 0.95, legend_info, 
                   transform=ax_legend.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig3.canvas.draw_idle()

slider1.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)

# Llamar update inicial para mostrar estadísticas
update(None)

print("\n✓ Visualización completada")
print(f"\nDefectos ROJOS (>0.05): {np.sum(z > 0.05)} px ({100*np.sum(z > 0.05)/z.size:.2f}%)")
print(f"Defectos AZULES (<-0.063): {np.sum(z < -0.063)} px ({100*np.sum(z < -0.063)/z.size:.2f}%)")

plt.show()