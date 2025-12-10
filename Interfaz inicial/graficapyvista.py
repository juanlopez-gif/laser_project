import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

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

colors_hex = ['#00008B', '#0000FF', '#00FFFF', '#FFFF00', '#FFA500', '#FF0000']
positions = [0.0, 0.20, 0.38, 0.56, 0.73, 1.0]

# Crear colormap de matplotlib
cmap_mpl = LinearSegmentedColormap.from_list('vr6200', 
                                              list(zip(positions, colors_hex)),
                                              N=256)

# --- PREPARAR MALLA ESTRUCTURADA PARA PYVISTA ---
print("\n--- CREANDO MALLA 3D ---")

# INVERTIR EL EJE Y para que (0,0) esté arriba a la izquierda
Y_invertido = y.max() - Y

# PyVista necesita puntos en formato (n_points, 3)
points = np.c_[X.ravel(), Y_invertido.ravel(), z.ravel()]

# Crear StructuredGrid
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [nx, ny, 1]

# Añadir valores de altura como scalar
grid.point_data['Altura'] = z.ravel()

print(f"✓ Malla creada: {grid.n_points} puntos, {grid.n_cells} celdas")

# --- CREAR VISUALIZACIÓN CON PYVISTA ---
print("\n--- GENERANDO VISUALIZACIÓN 3D ---")

# Crear plotter
plotter = pv.Plotter(window_size=[1400, 1000])

# Añadir malla con colormap
plotter.add_mesh(
    grid,
    scalars='Altura',
    cmap=cmap_mpl,
    clim=[vmin, vmax],
    smooth_shading=True,      # ← CLAVE: Shading suave
    show_edges=False,
    lighting=True,            # ← Iluminación realista
    specular=0.5,             # Reflejo especular
    specular_power=15,
    ambient=0.3,
    diffuse=0.7,
    interpolate_before_map=True  # Interpolación suave
)

# Añadir barra de color
plotter.add_scalar_bar(
    title='Altura (mm)',
    n_labels=6,
    italic=False,
    bold=True,
    title_font_size=14,
    label_font_size=12,
    position_x=0.85,
    position_y=0.1
)

# Configurar cámara
plotter.camera_position = 'iso'
plotter.camera.azimuth = 45
plotter.camera.elevation = 30
plotter.camera.zoom(1.2)

# Añadir ejes
plotter.add_axes(
    xlabel='X [mm]',
    ylabel='Y [mm]',
    zlabel='Z [mm]',
    line_width=3,
    labels_off=False
)

# Título
plotter.add_text(
    'Vista 3D Profesional - VR-6200 Scan\n(PyVista OpenGL Rendering)',
    position='upper_edge',
    font_size=14,
    color='black',
    font='arial'
)

# Fondo blanco
plotter.background_color = 'white'

# Información en la esquina
info_text = (
    f"Dimensiones: {nx}×{ny} píxeles\n"
    f"Área: {nx*pixel_size:.2f}×{ny*pixel_size:.2f} mm\n"
    f"Rango Z: [{np.nanmin(z):.3f}, {np.nanmax(z):.3f}] mm"
)
plotter.add_text(
    info_text,
    position='lower_left',
    font_size=10,
    color='black'
)

# Instrucciones de uso
instrucciones = (
    "Controles:\n"
    "- Click izquierdo: Rotar\n"
    "- Click derecho: Pan\n"
    "- Scroll: Zoom\n"
    "- 'r': Reset cámara\n"
    "- 's': Screenshot\n"
    "- 'q': Salir"
)
plotter.add_text(
    instrucciones,
    position='upper_right',
    font_size=9,
    color='gray'
)

print("\n" + "="*60)
print("CONTROLES INTERACTIVOS:")
print("="*60)
print("  Click izquierdo + arrastrar → Rotar vista")
print("  Click derecho + arrastrar  → Mover (pan)")
print("  Rueda del ratón           → Zoom in/out")
print("  Tecla 'r'                 → Reset cámara")
print("  Tecla 's'                 → Guardar captura (screenshot.png)")
print("  Tecla 'q' o cerrar ventana → Salir")
print("="*60)

# Mostrar
print("\n✓ Abriendo ventana de visualización...")
print("  (La ventana puede tardar unos segundos en aparecer)")

plotter.show()

print("\n✓ Visualización cerrada")

# --- OPCIONAL: EXPORTAR MALLA ---
print("\n¿Deseas exportar la malla 3D?")
print("  Formatos disponibles: VTK, STL, OBJ, PLY")
print("  (Puedes descomentar las siguientes líneas)")

# Descomentar para exportar:
# grid.save('superficie_vr6200.vtk')
# print("✓ Malla exportada a: superficie_vr6200.vtk")

# Para exportar a STL (útil para CAD):
# surface = grid.extract_surface()
# surface.save('superficie_vr6200.stl')
# print("✓ Superficie exportada a: superficie_vr6200.stl")