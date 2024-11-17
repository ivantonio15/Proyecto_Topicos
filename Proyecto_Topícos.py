import os
import requests
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pyttsx3
import gradio as gr
import threading

# URLs y archivos necesarios
arch = 'resnet18'
model_file = f'{arch}_places365.pth.tar'
model_url = f'http://places2.csail.mit.edu/models_places365/{model_file}'
categories_file = 'categories_places365.txt'
categories_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
indoor_outdoor_file = 'IO_places365.txt'
indoor_outdoor_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'

# Función para descargar archivos
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f'Descargando {filename}...')
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f'{filename} descargado.')

# Descargar archivos necesarios
download_file(model_url, model_file)
download_file(categories_url, categories_file)
download_file(indoor_outdoor_url, indoor_outdoor_file)

# Cargar modelo preentrenado
model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Transformaciones para la imagen
transform = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Cargar las etiquetas de categorías
categories = []
with open(categories_file) as f:
    for line in f:
        category = line.strip().split(' ')[0][3:]
        categories.append(category)

# Cargar indoor/outdoor labels correctamente
io_mapping = {}
with open(indoor_outdoor_file) as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            # El último número es 0 para indoor, 1 para outdoor
            category = parts[0]
            is_outdoor = int(parts[-1])
            io_mapping[category] = is_outdoor

import os
import requests
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pyttsx3
import gradio as gr
import threading

# URLs y archivos necesarios
arch = 'resnet18'
model_file = f'{arch}_places365.pth.tar'
model_url = f'http://places2.csail.mit.edu/models_places365/{model_file}'
indoor_outdoor_file = 'IO_places365.txt'
indoor_outdoor_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'

# Función para descargar archivos
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f'Descargando {filename}...')
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f'{filename} descargado.')

# Descargar archivos necesarios
download_file(model_url, model_file)
download_file(indoor_outdoor_url, indoor_outdoor_file)

# Categorías clasificadas en Indoor y Outdoor
indoor_categories = [
    "airplane_cabin", "alcove", "amusement_arcade", "aquarium", "arena/hockey", "arena/performance", "arena/rodeo",
    "art_gallery", "art_school", "art_studio", "artists_loft", "attic", "auditorium", "auto_factory", "auto_showroom",
    "bakery/shop", "ball_pit", "ballroom", "bank_vault", "banquet_hall", "bar", "barn", "basketball_court/indoor",
    "bathroom", "bazaar/indoor", "bedchamber", "bedroom", "biology_laboratory", "booth/indoor", "bow_window/indoor",
    "bowling_alley", "boxing_ring", "bus_interior", "bus_station/indoor", "cafeteria", "chemistry_lab", "childs_room",
    "church/indoor", "classroom", "clean_room", "closet", "coffee_shop", "computer_room", "conference_center",
    "conference_room", "dining_hall", "dining_room", "discotheque", "dorm_room", "dressing_room", "drugstore",
    "elevator/door", "elevator_lobby", "fabric_store", "fastfood_restaurant", "flea_market/indoor",
    "florist_shop/indoor", "food_court", "formal_garden", "garage/indoor", "general_store/indoor", "gift_shop",
    "greenhouse/indoor", "gymnasium/indoor", "hospital", "hospital_room", "ice_cream_parlor", "ice_skating_rink/indoor",
    "jacuzzi/indoor", "jail_cell", "jewelry_shop", "kitchen", "laundromat", "lecture_room", "legislative_chamber",
    "library/indoor", "living_room", "lobby", "locker_room", "market/indoor", "martial_arts_gym", "movie_theater/indoor",
    "museum/indoor", "music_studio", "natural_history_museum", "nursery", "nursing_home", "office", "office_cubicles",
    "operating_room", "pharmacy", "physics_laboratory", "playroom", "pub/indoor", "reception", "recreation_room",
    "repair_shop", "restaurant", "restaurant_kitchen", "sauna", "science_museum", "server_room", "shoe_shop",
    "shopping_mall/indoor", "shower", "spa", "staircase", "storage_room", "subway_station/platform", "supermarket",
    "sushi_bar", "swimming_pool/indoor", "television_room", "television_studio", "temple/asia", "throne_room",
    "toyshop", "train_interior", "waiting_room", "wet_bar", "youth_hostel"
]

outdoor_categories = [
    "airfield", "airport_terminal", "alley", "amphitheater", "amusement_park", "apartment_building/outdoor",
    "aqueduct", "arch", "archaeological_excavation", "athletic_field/outdoor", "atrium/public", "badlands",
    "balcony/exterior", "beach", "beach_house", "beer_garden", "bridge", "building_facade", "bullring",
    "cabin/outdoor", "campsite", "campus", "canal/natural", "canal/urban", "canyon", "cemetery", "chalet", "cliff",
    "clothing_store", "coast", "corn_field", "corral", "courtyard", "creek", "crevasse", "crosswalk", "dam",
    "delicatessen", "department_store", "desert/sand", "desert/vegetation", "desert_road", "diner/outdoor",
    "downtown", "driveway", "fire_escape", "fire_station", "fishpond", "football_field", "forest/broadleaf",
    "forest_path", "forest_road", "fountain", "garage/outdoor", "gazebo/exterior", "general_store/outdoor", "glacier",
    "golf_course", "greenhouse/outdoor", "grotto", "hangar/outdoor", "harbor", "hayfield", "heliport", "highway",
    "hot_spring", "hotel/outdoor", "house", "hunting_lodge/outdoor", "ice_floe", "ice_shelf",
    "ice_skating_rink/outdoor", "iceberg", "igloo", "industrial_area", "inn/outdoor", "islet", "kasbah",
    "kennel/outdoor", "lagoon", "lake/natural", "landfill", "lawn", "lighthouse", "market/outdoor", "marsh",
    "meadow", "medina", "moat/water", "mosque/outdoor", "mountain", "mountain_path", "mountain_snowy", "park",
    "parking_garage/outdoor", "parking_lot", "pasture", "patio", "pavilion", "pier", "playground", "plaza", "pond",
    "porch", "promenade", "railroad_track", "rainforest", "residential_neighborhood", "restaurant_patio",
    "rice_paddy", "river", "rock_arch", "roof_garden", "rope_bridge", "ruin", "runway", "sandbox", "schoolhouse",
    "shed", "shopping_mall/outdoor", "ski_resort", "ski_slope", "sky", "skyscraper", "slum", "snowfield",
    "soccer_field", "stable", "stadium/baseball", "stadium/football", "stadium/soccer", "stage/outdoor", "street",
    "swamp", "swimming_hole", "swimming_pool/outdoor", "synagogue/outdoor", "temple/outdoor", "ticket_booth",
    "topiary_garden", "tower", "trench", "tundra", "underwater/ocean_deep", "utility_room", "valley",
    "vegetable_garden", "viaduct", "village", "vineyard", "volcano", "volleyball_court/outdoor", "water_park",
    "water_tower", "waterfall", "watering_hole", "wave", "wheat_field", "wind_farm", "windmill", "yard",
    "zen_garden"
]

# Crear un mapeo basado en estas listas
io_mapping = {category: 0 for category in indoor_categories}
io_mapping.update({category: 1 for category in outdoor_categories})

# El resto del código (como las predicciones y transformaciones) permanece igual.


# Actualizar el mapeo con las categorías conocidas
for category in indoor_categories:
    if category in io_mapping:
        io_mapping[category] = 0  # 0 para indoor
    else:
        print(f"Advertencia: {category} no encontrado en las categorías de IO.")

for category in outdoor_categories:
    if category in io_mapping:
        io_mapping[category] = 1  # 1 para outdoor
    else:
        print(f"Advertencia: {category} no encontrado en las categorías de IO.")

# Verificar si todas las categorías están asignadas
faltantes = [cat for cat in categories if cat not in io_mapping]
if faltantes:
    print(f"Advertencia: Las siguientes categorías no tienen mapeo definido: {faltantes}")
else:
    print("Todas las categorías tienen un mapeo asignado correctamente.")

def speak_text(text):
    """Función para manejar el texto a voz con un nuevo engine cada vez"""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Error en la síntesis de voz: {str(e)}")
    finally:
        try:
            engine.stop()
        except:
            pass

def classify_image(image):
    try:
        # Convertir a imagen RGB
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            raise ValueError("Formato de imagen no soportado")

        # Predicción
        input_img = V(transform(img).unsqueeze(0))
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # Obtener las top 5 predicciones
        output_str = []
        for i in range(5):
            category = categories[idx[i].item()]
            probability = probs[i].item() * 100
            is_indoor = io_mapping.get(category, -1) == 0
            io_text = "Indoor" if is_indoor else "Outdoor"
            output_str.append(f"{i+1}. {category} ({io_text}) - {probability:.2f}%")

        # Determinar si es Indoor/Outdoor basado en la predicción más probable
        top_category = categories[idx[0].item()]
        is_indoor = io_mapping.get(top_category, -1) == 0
        main_label = "Indoor" if is_indoor else "Outdoor"
        
        # Iniciar la voz en un hilo separado
        voice_thread = threading.Thread(
            target=speak_text,
            args=(main_label,),
            daemon=True
        )
        voice_thread.start()
        
        # Crear salida detallada
        detailed_output = f"Predicción principal: {main_label}\n"
        detailed_output += f"Categoría específica: {top_category}\n"
        detailed_output += "\nTop 5 predicciones:\n"
        detailed_output += "\n".join(output_str)
        
        return detailed_output
    
    except Exception as e:
        print(f"Error al procesar la imagen: {str(e)}")
        return f"Error al procesar la imagen: {str(e)}"

# Configuración de la interfaz de Gradio
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Textbox(label="Resultado", lines=8),
    title="Scene Classification (Indoor/Outdoor)",
    description="Sube una imagen para clasificar si la escena es de interior o exterior."
)

# Iniciar la interfaz
if __name__ == "__main__":
    interface.launch()
