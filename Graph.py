import vedo
vedo.settings.default_backend = 'vtk' #mejor compatibilidad con jupyter
from vedo import show 
from vedo import Plotter

import brainrender
from brainrender import Animation
from brainrender import Scene, settings
from brainrender.actors import Point
from brainrender.video import VideoMaker
from brainglobe_atlasapi import BrainGlobeAtlas

import numpy as np
import os
from pathlib import Path


# Configuración de BrainRender y Vedo
brainrender.settings.BACKGROUND_COLOR = "white"
brainrender.settings.ROOT_ALPHA = 0.1  # Transparencia del cerebro
brainrender.settings.SHOW_AXES = False
brainrender.settings.SHADER_STYLE = "plastic"  # Opciones: metallic, plastic, shiny, glossy, cartoon


def create_data(points_loc, colors, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    atlas_res = 39
    AP = 371 * atlas_res
    DV = 72 * atlas_res
    ML = 266 * atlas_res
    bregma = (AP, DV, ML)

    points = {name: compute_point(bregma, offset) for name, offset in points_loc.items()}

    coords_array = np.array(list(points.values()))
    np.save(os.path.join(output_dir, "points_coords.npy"), coords_array)

    names_array = np.array(list(points.keys()))
    np.save(os.path.join(output_dir, "points_names.npy"), names_array)

    colors_array = np.array(colors)
    np.save(os.path.join(output_dir, "points_colors.npy"), colors_array)

def compute_point(base, offset):
    return tuple(b + o for b, o in zip(base, offset))



def get_camera_settings(scene, save_path="camera_data.npy", print_camera=True):
    """
    Obtiene y guarda la configuración de la cámara después del renderizado.

    :param scene: Objeto Scene renderizado.
    :param save_path: Ruta donde se guardará la configuración de la cámara.
    :param print_camera: Si es True, imprime la configuración de la cámara.
    :return: Diccionario con los parámetros de la cámara.
    """
    cam_position = scene.plotter.camera.GetPosition()
    cam_focal = scene.plotter.camera.GetFocalPoint()
    cam_viewup = scene.plotter.camera.GetViewUp()
    cam_clip = scene.plotter.camera.GetClippingRange()

    camera_data = {
        "pos": cam_position,
        "focal": cam_focal,
        "viewup": cam_viewup,
        "clipping_range": cam_clip,
    }

    # Guardar la configuración de la cámara para usos futuros
    np.save(save_path, camera_data)

    if print_camera:
        print("\n📸 Características de la cámara después del renderizado:")
        print(f"🔹 Posición: {cam_position}")
        print(f"🔹 Punto focal: {cam_focal}")
        print(f"🔹 Vector de vista arriba: {cam_viewup}")
        print(f"🔹 Rango de recorte: {cam_clip}")
        print(f"💾 Configuración de cámara guardada en {save_path}")

    return camera_data

def load_previous_camera(save_path="camera_data.npy"):
    """
    Carga la configuración de la cámara desde un archivo si existe.

    :param save_path: Ruta del archivo donde está guardada la configuración.
    :return: Diccionario con los parámetros de la cámara si existe, de lo contrario None.
    """
    if os.path.exists(save_path):
        print(f"🔄 Cargando configuración de cámara previa desde {save_path}")
        return np.load(save_path, allow_pickle=True).item()  # Cargar como diccionario
    else:
        print("⚠️ No se encontraron datos previos de cámara, usando valores por defecto.")
        return None


def generate_video(scene, save_folder, video_name="test_video", duration=15, fps=15, azimuth=1, elevation=0, roll=0):
    """
    Genera un video de una escena de BrainRender con rotación de la cámara.

    :param scene: (Scene) Escena de BrainRender a renderizar.
    :param save_folder: (str) Directorio donde se guardará el video.
    :param video_name: (str) Nombre del video (sin extensión). Default: "test_video".
    :param duration: (int) Duración del video en segundos. Default: 15.
    :param fps: (int) Fotogramas por segundo. Default: 15.
    :param azimuth: (int) Rotación lateral en cada frame. Default: 1.
    :param elevation: (int) Rotación vertical en cada frame. Default: 0.
    :param roll: (int) Giro de la cámara en cada frame. Default: 0.
    """

    # Crear objeto de video
    vm = VideoMaker(scene, name=video_name, save_fld=save_folder)

    # Generar video con los parámetros dados
    vm.make_video(
        duration=duration, 
        fps=fps,       
        azimuth=azimuth,    
        elevation=elevation,  
        roll=roll        
    )

    # Cerrar la escena después de renderizar
    scene.close()

def graph(points, camera_data=None, interactive=True, labels=False, mode="CA1d", print_camera=True):
    """
    Genera una escena en BrainRender con regiones cerebrales y puntos.

    :param points: Lista de objetos Point.
    :param camera_data: Configuración de la cámara (si está vacía, carga valores predeterminados).
    :param interactive: Si es True, permite interacción con la escena.
    :param labels: Si es True, muestra etiquetas sobre los puntos.
    :param mode: Modo de visualización, inclye regiones para visualizacion, puede ser "CA1d", "CA1v", "RSC", o "PFC".
    :return: Objeto Scene renderizado.
    """
    # Validar el modo ingresado
    valid_modes = {"CA1d", "CA1v", "RSC", "PFC"}
    if mode not in valid_modes:
        raise ValueError(f"Modo inválido. Debe ser uno de: {valid_modes}")
        
    # Si no se proporciona camera_data, intenta cargar datos previos
    if not camera_data:
        camera_data = load_previous_camera()

    # Si aún no hay datos previos, usar valores predeterminados
    if not camera_data:
        camera_data = {"pos": (10000, 10000, 10000), "viewup": (0, -1, 0), "clipping_range": (5000, 20000)}

    # Crear la escena
    scene = Scene(atlas_name="whs_sd_rat_39um", screenshots_folder='screenshots')
    scene.root.alpha(0)

    # Agregar regiones cerebrales con diferentes estilos según el modo
    if mode == "CA1d":
        CA= scene.add_brain_region(
            "CA",  alpha=0.1, silhouette=False, hemisphere="left"
        )
    elif mode == "CA1v":
        CA, DG = scene.add_brain_region(
            "CA", "DG", alpha=0.1, silhouette=False, hemisphere="left"
        )
    elif mode == "RSC":
        scene.add_brain_region(
            "RSD", "RSG", alpha=0.1, silhouette=False, hemisphere="left"
        )
    elif mode == "PFC":
        scene.add_brain_region(
            "IL", "PrL", "Cg1", "Cg2", alpha=0.1, silhouette=False, hemisphere="left"
        )

    Brain = scene.add_brain_region("Brain", alpha=0.05, silhouette=False, hemisphere="left", color="white")

    # Agregar puntos a la escena
    for point in points:
        scene.add(point, size=100)
        if labels:
            scene.add_label(point, point.name, size=200, radius=None)

    # Renderizar la escena
    scene.render(camera=camera_data, interactive=interactive)

    # Obtener la configuración de la cámara después del renderizado y guardarla
    get_camera_settings(scene, save_path="camera_data.npy", print_camera=print_camera)

    return scene


def data_loader(region_name, base_dir="npy_points"):
    """
    Carga las coordenadas, nombres y colores de una región específica desde archivos .npy 
    y crea objetos Point.

    :param region_name: Nombre de la región (por ejemplo, "CA1v").
    :param base_dir: Directorio base donde se encuentran los archivos (por defecto, "npy_points").
    :return: Lista de objetos Point.
    """
    # Construir las rutas de los archivos
    coords_file = os.path.join(base_dir, f"{region_name}_offsets_coords.npy")
    names_file = os.path.join(base_dir, f"{region_name}_offsets_ids.npy")
    colors_file = os.path.join(base_dir, f"{region_name}_offsets_colors.npy")

    # Verificar la existencia de los archivos
    if not os.path.exists(coords_file):
        raise FileNotFoundError(f"No se encontró el archivo de coordenadas: {coords_file}")
    if not os.path.exists(names_file):
        raise FileNotFoundError(f"No se encontró el archivo de nombres: {names_file}")

    # Cargar coordenadas y nombres
    loaded_coords = np.load(coords_file)
    loaded_names = np.load(names_file, allow_pickle=True)

    
    # Cargar colores o generar aleatorios si el archivo no existe
    if os.path.exists(colors_file):
        loaded_colors = np.load(colors_file)
    else:
        loaded_colors = np.random.rand(len(loaded_names), 3)

    # Crear objetos Point
    
    points = []
    for coords, name, color in zip(loaded_coords, loaded_names, loaded_colors):
        #print(coords, name, color)  # Depuración
        points.append(Point(pos=coords, name=name, color=color, radius=100))
    
    return points

def capturas(points, multi, region):
    mode=region

    sagital = {
        "pos": (-1046.9756515348963, 1212.954955375364, -75871.55746288373),
        "viewup": (0, -1, 0),
        "clipping_range": (50000, 20000),
    }

    frontal = {
        "pos": (-26645, 8244, -9576),
        "viewup": (0, -1, 0),
        "clipping_range": (19531, 40903),
    }

    top = {
        "pos": (1073, -66580, -12129),
        "viewup": (-1, 0, 0),
        "clipping_range": (27262, 45988),
    }

    three_quarters_camera = {
        "pos": (-22310.0, -3802.0, 13811.0),
        "viewup": (0, -1, 0),
        "clipping_range": (13625, 119925),
    }

    capturas_config = [
    (sagital, "_sagital"),
    (frontal, "_frontal"),
    (top, "_top"),
    (three_quarters_camera, "_three_quarters_camera"),
]

    if multi:
        for cam, sufijo in capturas_config:
            scene = graph(points, camera_data=cam, interactive=False, mode=mode, print_camera=False)
            scene.screenshot(mode + sufijo, scale=5)
            scene.close()
    # Posible llamada extra con sagital para refrescar, si es necesaria
            scene = graph(points, camera_data=sagital, interactive=False, mode=mode, print_camera=False)
            scene.close()
    else:
        scene = graph(points, camera_data=load_previous_camera(), interactive=False, mode=mode, print_camera=False)
        scene.screenshot(mode, scale=5)
        scene.close()

        scene = graph(points, camera_data=sagital, interactive=False, mode=mode, print_camera=False)
        scene.close()

def get_point_locations(region_name, base_dir="npy_points", output_file=None):
    """
    Carga los puntos de una región y devuelve la ubicación anatómica de cada punto usando el atlas 'whs_sd_rat_39um'.
    
    :param region_name: Nombre de la región (por ejemplo, "CA1v").
    :param base_dir: Directorio donde están los archivos .npy.
    :param output_file: Si se proporciona, guarda los resultados en este archivo.
    :return: Lista de tuplas (nombre, coordenadas, estructura).
    """
    # Inicializar el atlas
    bg_atlas = BrainGlobeAtlas("whs_sd_rat_39um", check_latest=False)

    # Construir rutas de archivos
    coords_file = os.path.join(base_dir, f"{region_name}_offsets_coords.npy")
    ID_file = os.path.join(base_dir, f"{region_name}_offsets_ids.npy")

    # Verificar archivos
    if not os.path.exists(coords_file):
        raise FileNotFoundError(f"No se encontró el archivo de coordenadas: {coords_file}")
    if not os.path.exists(ID_file):
        raise FileNotFoundError(f"No se encontró el archivo de nombres: {ID_file}")

    # Cargar los archivos
    loaded_coords = np.load(coords_file)
    loaded_ID = np.load(ID_file, allow_pickle=True)

    # Obtener las estructuras para cada punto
    results = []
    for ID, coords in zip(loaded_ID, loaded_coords):
        structure = bg_atlas.structure_from_coords(coords, as_acronym=True, microns=True)
        matching_region = {v["name"] 
            for v in bg_atlas.structures.values()
            if structure in v["acronym"]
        }
        
        # Formatear las coordenadas como decimales normales
        coords_formatted = tuple(round(float(c), 2) for c in coords)
        
        # Verificar si el punto está fuera del atlas
        if structure == "Outside atlas":
            print(f"⚠️ Advertencia: El punto '{ID}' está fuera del atlas.")
        
        results.append((ID, coords_formatted, structure, matching_region))

    # Imprimir resultados o guardarlos en archivo
    if output_file:
        with open(output_file, "w") as f:
            f.write("ID\tCoordenadas\tAcron\tEstructura")
            for ID, coords, structure, matching_region in results:
                f.write(f"{ID}\t{coords}\t{structure}\t{matching_region}\n")
        print(f"Resultados guardados en: {output_file}")
    else:
        print("ID\tCoordenadas\tAcron\tEstructura")
        for ID, coords, structure, matching_region in results:
            print(f"{ID}\t{coords}\t{structure}\t{matching_region}")
            #print(matching_regions)

    return results