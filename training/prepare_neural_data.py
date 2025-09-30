# training/prepare_neural_data.py
import json
import re
import unicodedata
import logging
from typing import List, Dict, Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize(s: str) -> str:
    """Normaliza texto igual que el sistema actual."""
    if not s:
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
    return s

def load_training_data(key_words_path: str) -> Tuple[List[str], List[str]]:
    """
    Carga datos de entrenamiento desde key_words_labels.json
    
    Returns:
        texts: Lista de textos normalizados
        labels: Lista de etiquetas correspondientes
    """
    with open(key_words_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    
    for field, examples in data['key_fields'].items():
        for example in examples:
            text = example['text']
            label = example['label']
            
            # Normalizar y agregar
            normalized_text = normalize(text)
            if normalized_text:
                texts.append(normalized_text)
                labels.append(label)
    
    logger.info(f"Datos cargados: {len(texts)} ejemplos")
    return texts, labels

def generate_negative_samples(positive_texts: List[str], num_samples: int = 100) -> List[str]:
    """
    Genera ejemplos negativos (texto que NO pertenece a ningún campo).
    """
    negative_samples = [
        "lorem ipsum", "xyz", "abcdefg", "random text", "noise",
        "zzz", "qwerty", "asdfgh", "123456", "test test", "blank"
        "dummy", "sample", "example", "placeholder", "unknown",
        "other", "misc", "various", "generic", "puntuacion",
        "puntualidad", "estudiante", 
    ]
    
    # Generar variaciones aleatorias
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    for _ in range(num_samples - len(negative_samples)):
        length = random.randint(3, 15)
        random_text = ''.join(random.choice(chars) for _ in range(length))
        negative_samples.append(random_text)
    
    return negative_samples[:num_samples]

def augment_data(texts: List[str], labels: List[str], augmentation_factor: int = 3) -> Tuple[List[str], List[str]]:
    """
    Aumenta los datos generando variaciones con errores simulados de OCR.
    """
    augmented_texts = list(texts)
    augmented_labels = list(labels)
    
    # Diccionario de sustituciones comunes en OCR
    ocr_substitutions = {
        'a': ['@', 'á', 'à'],
        'e': [ 'é', 'è'],
        'i': ['í', 'ì', '|'],
        'o': ['ó', 'ò'],
        't': ['+'],
        'u': ['ú', 'ù'],
        'n': ['ñ'],
    }
    
    for text, label in zip(texts, labels):
        for _ in range(augmentation_factor):
            augmented = apply_random_augmentation(text, ocr_substitutions)
            if augmented and augmented != text:
                augmented_texts.append(augmented)
                augmented_labels.append(label)
    
    logger.info(f"Datos aumentados: {len(texts)} -> {len(augmented_texts)} ejemplos")
    return augmented_texts, augmented_labels

def apply_random_augmentation(text: str, substitutions: Dict[str, List[str]]) -> str:
    """Aplica una augmentación aleatoria al texto."""
    if not text or len(text) < 2:
        return text
    
    operations = ['substitute', 'delete', 'swap', 'insert']
    operation = random.choice(operations)
    text_list = list(text)
    
    if operation == 'substitute' and len(text_list) > 0:
        # Sustituir un carácter
        idx = random.randint(0, len(text_list) - 1)
        char = text_list[idx].lower()
        if char in substitutions:
            text_list[idx] = random.choice(substitutions[char])
    
    elif operation == 'delete' and len(text_list) > 2:
        # Borrar un carácter
        idx = random.randint(0, len(text_list) - 1)
        text_list.pop(idx)
    
    elif operation == 'swap' and len(text_list) > 1:
        # Intercambiar dos caracteres adyacentes
        idx = random.randint(0, len(text_list) - 2)
        text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
    
    elif operation == 'insert' and len(text_list) > 0:
        # Insertar un carácter aleatorio
        idx = random.randint(0, len(text_list))
        text_list.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz0123456789'))
    
    return ''.join(text_list)

def prepare_training_data(key_words_path: str, output_path: str, 
                         augment: bool = True, augmentation_factor: int = 3,
                         negative_ratio: float = 0.2):
    """
    Prepara datos de entrenamiento completos y los guarda en JSON.
    
    Args:
        key_words_path: Ruta a key_words_labels.json
        output_path: Ruta donde guardar los datos preparados
        augment: Si aplicar data augmentation
        augmentation_factor: Factor de aumento de datos
        negative_ratio: Proporción de ejemplos negativos
    """
    # Cargar datos positivos
    texts, labels = load_training_data(key_words_path)
    
    # Augmentar datos si se solicita
    if augment:
        texts, labels = augment_data(texts, labels, augmentation_factor)
    
    # Generar ejemplos negativos
    num_negatives = int(len(texts) * negative_ratio)
    negative_texts = generate_negative_samples(texts, num_negatives)
    negative_labels = ['NO_MATCH'] * len(negative_texts)
    
    # Combinar todos los datos
    all_texts = texts + negative_texts
    all_labels = labels + negative_labels
    
    # Crear mapeo de etiquetas
    unique_labels = sorted(set(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Convertir etiquetas a índices
    label_indices = [label_to_idx[label] for label in all_labels]
    
    # Crear dataset
    dataset = {
        'texts': all_texts,
        'labels': all_labels,
        'label_indices': label_indices,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'num_classes': len(unique_labels),
        'num_samples': len(all_texts),
        'augmented': augment,
        'augmentation_factor': augmentation_factor if augment else 0
    }
    
    # Guardar
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Datos preparados guardados en: {output_path}")
    logger.info(f"Total ejemplos: {len(all_texts)}")
    logger.info(f"Ejemplos positivos: {len(texts)}")
    logger.info(f"Ejemplos negativos: {len(negative_texts)}")
    logger.info(f"Clases: {len(unique_labels)}")
    logger.info(f"Distribución por clase:")
    
    for label in unique_labels:
        count = all_labels.count(label)
        percentage = 100.0 * count / len(all_labels)
        logger.info(f"{label}: {count} ({percentage:.1f}%)")
    
    return dataset

if __name__ == "__main__":
    import os
    
    # Rutas
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    key_words_path = os.path.join(project_root, "training", "key_words_labels.json")
    output_path = os.path.join(project_root, "training", "neural_training_data.json")
    
    # Preparar datos
    dataset = prepare_training_data(
        key_words_path=key_words_path,
        output_path=output_path,
        augment=True,
        augmentation_factor=3,
        negative_ratio=0.2
    )
    
    logger.info("Preparación de datos completada exitosamente")
