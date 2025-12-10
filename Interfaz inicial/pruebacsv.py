import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN ---
csv_file = "Niveles_Nodos_20251113_144858.csv"  # Cambia por tu archivo CSV

print(f"Leyendo archivo CSV: {csv_file}")

# Leer CSV
df = pd.read_csv(csv_file)

print(f"Total de filas leídas: {len(df)}")
print(f"\nPrimeras filas del CSV:")
print(df.head(10))

# Filtrar solo las filas con coordenadas (las que tienen X, Y, Z)
df_puntos = df[df['X_mm'].notna()].copy()

print(f"\nTotal de puntos con coordenadas: {len(df_puntos)}")

# Verificar niveles presentes
niveles_presentes = df_puntos['Nivel'].unique()
print(f"Niveles encontrados: {niveles_presentes}")

# --- COLORES POR NIVEL ---
colores_nivel = {
    4: '#000000',  # Negro
    3: '#FF1493',  # Rosa
    2: '#00FF00',  # Verde fosforito
}

# --- CREAR VISUALIZACIÓN CON LEYENDA EXTERNA ---
fig = plt.figure(figsize=(14, 10))

# Crear dos subplots: gráfica principal y panel de leyenda
ax_main = plt.subplot2grid((1, 3), (0, 0), colspan=2)
ax_legend = plt.subplot2grid((1, 3), (0, 2))
ax_legend.axis('off')

# Plotear cada nivel con su color
for nivel in sorted(niveles_presentes, reverse=True):  # De mayor a menor
    df_nivel = df_puntos[df_puntos['Nivel'] == nivel]
    
    color = colores_nivel.get(nivel, 'gray')
    nombre_nivel = {4: 'Negro', 3: 'Rosa', 2: 'Verde'}.get(nivel, f'Nivel {nivel}')
    
    # Contar nodos únicos en este nivel
    nodos = df_nivel['Nodo'].unique()
    num_nodos = len(nodos)
    
    ax_main.scatter(df_nivel['X_mm'], df_nivel['Y_mm'], 
                    c=color, s=3, alpha=0.9)
    
    print(f"  Nivel {nivel} ({nombre_nivel}): {len(df_nivel)} puntos en {num_nodos} nodos")

# Invertir eje Y para coincidir con la visualización original
ax_main.invert_yaxis()

ax_main.set_xlabel('X [mm]', fontsize=12)
ax_main.set_ylabel('Y [mm]', fontsize=12)
ax_main.set_title(f'Verificación Visual del CSV: {csv_file}', fontsize=14, fontweight='bold')
ax_main.grid(True, alpha=0.3)

# Ajustar aspecto para que no salga cuadrado forzosamente
# ax_main.set_aspect('equal', adjustable='box')  # REMOVIDO para evitar que sea cuadrado

# --- CREAR LEYENDA EXTERNA CON ESTADÍSTICAS ---
# Leer información de resumen por nivel
df_resumen = df[df['Altura_media_mm'].notna()].copy()

# Construir texto de leyenda
legend_text = "INFORMACIÓN POR NIVEL\n" + "="*30 + "\n\n"

for nivel in sorted(niveles_presentes, reverse=True):
    nombre_nivel = {4: 'Negro', 3: 'Rosa', 2: 'Verde'}.get(nivel, f'Nivel {nivel}')
    df_nivel = df_puntos[df_puntos['Nivel'] == nivel]
    nodos = df_nivel['Nodo'].unique()
    num_nodos = len(nodos)
    num_puntos = len(df_nivel)
    
    # Obtener info de resumen de este nivel
    df_nivel_resumen = df_resumen[df_resumen['Nivel'] == nivel]
    area_total = df_nivel_resumen['Area_mm2'].sum()
    
    legend_text += f"NIVEL {nivel} ({nombre_nivel})\n"
    legend_text += f"Nodos: {num_nodos}\n"
    legend_text += f"Puntos: {num_puntos}\n"
    legend_text += f"Área total: {area_total:.6f} mm²\n"
    legend_text += "\n"

# Añadir estadísticas generales
legend_text += "="*30 + "\n"
legend_text += "RANGOS GENERALES\n"
legend_text += f"X: [{df_puntos['X_mm'].min():.3f}, {df_puntos['X_mm'].max():.3f}] mm\n"
legend_text += f"Y: [{df_puntos['Y_mm'].min():.3f}, {df_puntos['Y_mm'].max():.3f}] mm\n"
legend_text += f"Z: [{df_puntos['Z_mm'].min():.3f}, {df_puntos['Z_mm'].max():.3f}] mm\n"

ax_legend.text(0.05, 0.95, legend_text,
               transform=ax_legend.transAxes,
               fontsize=10,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# --- ESTADÍSTICAS EN CONSOLA ---
print(f"\n--- ESTADÍSTICAS GENERALES ---")
print(f"Rango X: {df_puntos['X_mm'].min():.6f} - {df_puntos['X_mm'].max():.6f} mm")
print(f"Rango Y: {df_puntos['Y_mm'].min():.6f} - {df_puntos['Y_mm'].max():.6f} mm")
print(f"Rango Z: {df_puntos['Z_mm'].min():.6f} - {df_puntos['Z_mm'].max():.6f} mm")

print(f"\n--- RESUMEN POR NIVEL Y NODO ---")
for _, row in df_resumen.iterrows():
    print(f"Nivel {int(row['Nivel'])} - Nodo {int(row['Nodo'])}: "
          f"Altura media = {row['Altura_media_mm']:.6f} mm, "
          f"Área = {row['Area_mm2']:.6f} mm²")

plt.tight_layout()
plt.show()

print("\n✓ Visualización completada")