# Proyecto de Análisis de Superficies VR-6200

Este proyecto proporciona herramientas para analizar y procesar datos de altura obtenidos del escáner láser VR-6200, permitiendo la detección de defectos, visualización 2D/3D y exportación de datos para corrección láser.

---

## 📁 Estructura de Archivos

### 🔧 Scripts Principales

#### `interfazmejorada.py` ⭐ (PRINCIPAL)
**Script interactivo con interfaz gráfica para análisis de niveles de defectos.**

**Funcionalidades:**
- Visualización 2D completa (mapa de alturas, histograma, perfiles X/Y)
- Vista 3D suavizada mediante interpolación cúbica
- **Interfaz interactiva** con sliders para definir 3 niveles de umbral personalizados
- Clasificación por colores:
  - **Nivel 4 (Negro)**: Defectos críticos (≥ umbral alto)
  - **Nivel 3 (Rosa)**: Defectos moderados (umbral medio)
  - **Nivel 2 (Verde)**: Defectos leves (umbral bajo)
- Contador de nodos (zonas contiguas del mismo nivel)
- **Botón de exportación CSV** con todos los puntos por nivel y nodo
- Panel lateral con estadísticas en tiempo real

**Uso:**
```bash
python interfazmejorada.py
```

---

#### `pruebacsv.py`
**Visualizador de archivos CSV exportados para verificación.**

**Funcionalidades:**
- Lee archivos CSV generados por `interfazmejorada.py`
- Genera visualización 2D con código de colores por nivel
- Leyenda externa con estadísticas completas
- Verificación visual de la correcta exportación de datos

**Uso:**
```bash
python pruebacsv.py
```
> Edita la variable `csv_file` con el nombre de tu archivo CSV

---

#### `pruebamapytrayectoria.py`
**Análisis completo con generación de trayectorias optimizadas.**

**Funcionalidades:**
- Visualización 2D (4 gráficas: mapa, histograma, perfiles)
- Vista 3D de la superficie
- Detección automática de defectos (umbral fijo >0.07 mm)
- **Generación de archivo DXF** con:
  - Contornos de zonas defectuosas
  - Trayectoria optimizada (algoritmo nearest neighbor)
  - Cálculo de distancia total de recorrido
- Identificación de nodos mediante OpenCV
- Mapa con trayectoria numerada

**Uso:**
```bash
python pruebamapytrayectoria.py
```
# Visualización 3D Profesional con PyVista

## Propósito

Este script proporciona una **visualización 3D de alta calidad** de los datos de superficie obtenidos del escáner láser VR-6200, utilizando PyVista con renderizado OpenGL acelerado por hardware. A diferencia de las visualizaciones tradicionales con Matplotlib que pueden presentar un aspecto pixelado o "blocoso", este script genera superficies completamente suavizadas con iluminación realista, ideal para presentaciones profesionales, análisis detallado y documentación técnica.

## Funcionalidades Principales

### 🎨 Renderizado de Alta Calidad
- **Smooth shading**: Elimina completamente el efecto "Minecraft" mediante interpolación suave entre puntos
- **Iluminación realista**: Aplica modelos de iluminación especular y difusa para resaltar el relieve de la superficie
- **Colormap VR-6200**: Utiliza la escala de colores calibrada del escáner (azul oscuro a rojo) con rangos de -0.063 mm a 0.05 mm
- **Renderizado OpenGL**: Aprovecha aceleración por hardware para visualización fluida y de alta resolución

### 🖱️ Interactividad Total
- **Rotación libre**: Click izquierdo + arrastrar para explorar la superficie desde cualquier ángulo
- **Desplazamiento (Pan)**: Click derecho + arrastrar para mover la vista sin cambiar la perspectiva
- **Zoom dinámico**: Rueda del ratón para acercamiento/alejamiento suave
- **Reset de cámara**: Tecla 'r' para volver a la vista por defecto (isométrica a 45°)
- **Captura de pantalla**: Tecla 's' para exportar imágenes en alta resolución directamente desde la visualización

### 📊 Información Contextual
- **Barra de escala lateral**: Muestra la correspondencia color-altura en milímetros
- **Ejes tridimensionales**: Sistema de coordenadas XYZ con etiquetas en mm
- **Estadísticas del scan**: Dimensiones en píxeles, área física y rango de alturas
- **Orientación correcta**: Sistema de coordenadas con origen (0,0) en esquina superior izquierda, coincidiendo con la convención de los mapas 2D

### 💾 Capacidades de Exportación (Opcional)
El script permite exportar la malla 3D en múltiples formatos profesionales:
- **VTK**: Formato estándar para visualización científica (compatible con ParaView)
- **STL**: Para software CAD e impresión 3D
- **OBJ**: Compatible con software de modelado 3D (Blender, Maya, 3ds Max)
- **PLY**: Para procesamiento avanzado de nubes de puntos

## Ventajas sobre Visualización Tradicional

Este enfoque con PyVista supera las limitaciones de Matplotlib 3D al proporcionar:
- **Calidad visual profesional**: Sin artefactos de pixelado, superficies completamente lisas
- **Rendimiento superior**: Renderizado acelerado por GPU, sin ralentizaciones con datos densos
- **Interactividad fluida**: Rotación y zoom en tiempo real sin lag
- **Exportación flexible**: Capacidad de generar imágenes de publicación y modelos 3D reutilizables

Este script es ideal para análisis detallado de defectos superficiales, presentaciones técnicas, documentación de calidad y cualquier aplicación que requiera visualización 3D de precisión de datos topográficos obtenidos mediante perfilometría láser.
---

#### `interopolacion.py`
**Script de interpolación de datos para suavizado de superficies.**

**Funcionalidades:**
- Interpolación cúbica de datos de altura
- Aumento de resolución mediante factor configurable
- Generación de superficie suavizada para visualización 3D
- Reducción de aspecto "pixelado" en representaciones 3D

**Uso:**
```bash
python interopolacion.py
```

---

### 📊 Archivos de Datos

#### `VR-20251110_173541_Height.csv`
Archivo de datos crudos exportado del escáner VR-6200.

**Formato:**
- 22 líneas de encabezado (metadatos del scan)
- Matriz de valores de altura en mm
- Dimensiones típicas: 1024 × 768 píxeles
- Resolución: 1.853 µm/píxel

---

#### `Niveles_Nodos_YYYYMMDD_HHMMSS.csv`
Archivos CSV generados por `interfazmejorada.py`.

**Estructura:**
```csv
Nivel,Nodo,Altura_media_mm,Area_mm2,X_mm,Y_mm,Z_mm
4,1,0.072345,0.001523,,,
4,1,,,,0.123456,0.456789,0.072
4,1,,,,0.124567,0.457890,0.073
...
```

**Contenido:**
- Primera fila por nodo: resumen (altura media, área)
- Filas siguientes: coordenadas (X, Y, Z) de todos los puntos del nodo
- Agrupado por Nivel y Nodo

---

## 🚀 Flujo de Trabajo Recomendado

1. **Análisis inicial**: Ejecuta `pruebamapytrayectoria.py` para visión general y DXF
2. **Análisis detallado**: Usa `interfazmejorada.py` para ajustar umbrales personalizados
3. **Exportación**: Genera CSV con umbrales óptimos usando el botón "Exportar CSV"
4. **Verificación**: Ejecuta `pruebacsv.py` para comprobar visualmente los datos exportados

---

## 📦 Dependencias

```bash
pip install numpy pandas matplotlib scipy opencv-python ezdxf
```

### Librerías utilizadas:
- `numpy`: Procesamiento numérico
- `pandas`: Manejo de datos CSV
- `matplotlib`: Visualización 2D/3D
- `scipy`: Interpolación y procesamiento de señales
- `opencv-python` (cv2): Detección de contornos y nodos
- `ezdxf`: Generación de archivos DXF para CAD

---

## 🎯 Características Principales

### Detección de Defectos
- Clasificación multinivel configurable
- Detección de zonas contiguas (nodos)
- Cálculo de áreas y alturas medias

### Visualización
- Mapas de calor con colormap VR-6200
- Superficies 3D suavizadas
- Perfiles de línea X/Y
- Histogramas de distribución

### Exportación
- Archivos DXF con trayectorias optimizadas
- CSV detallado por nivel y nodo
- Metadatos incluidos (timestamp, umbrales, estadísticas)

---

## 📝 Notas

- Los archivos CSV generados incluyen timestamp para evitar sobrescrituras
- La interpolación 3D no afecta los datos exportados (se exportan datos originales)
- El algoritmo de trayectoria usa "nearest neighbor" para minimizar distancia de recorrido
- Los colores están calibrados según estándar VR-6200

---

## 👨‍💻 Autor

Proyecto de análisis láser - TFG ISE 401  
University of Rhode Island - Cuarto Año

---

## 📄 Licencia

Proyecto académico - Uso educativo