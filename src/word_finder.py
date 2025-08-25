import os
import pickle
import numpy as np
from typing import List, Any, Dict
from sklearn.metrics.pairwise import cosine_similarity
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../data/word_finder_model.pkl')
        
        self.model = self._load_model(model_path)
        self.vectorizer = self.model['vectorizer']
        self.threshold = self.model['params']['threshold_similarity']
        
    def _load_model(self, model_path):
        """Carga el modelo pre-entrenado"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Modelo cargado: {model['total_words']} palabras, {model['vocabulario_size']} n-gramas")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Ejecuta generate_model.py primero.")
    
    def find_keywords(self, text: str):
        """
        Busca campos clave en el texto del polígono
        Args:
            texto_poligono (str): Texto extraído del polígono
        Returns:
            list: Lista de coincidencias encontradas
        """
        if not text.strip():
            return []
        
        # Vectorizar texto del polígono
        vectorized_text = self.vectorizer.transform([text])
        
        # Vectorizar todas las palabras clave
        all_words: List[Any] = []
        mapping_field: Dict[str, Any] = {}
        
        for field, variants in self.model['key_fields'].items():
            for variant in variants:
                all_words.append(variant)
                mapping_field[variant] = field
        
        vectorized_words = self.vectorizer.transform(all_words)
        
        # Calcular similitudes
        similities: np.ndarray[Any, Any] = cosine_similarity(vectorized_text, vectorized_words)[0]
        
        # Encontrar coincidencias
        coincidences: List[Any] = []
        for i, simility in enumerate(similities):
            if simility > self.threshold:
                word = vectorized_words[i]
                field = mapping_field[word]
                coincidences.append({
                    'field': field,
                    'word_finded': word,
                    'simility': float(simility),
                    'original_text': text
                })
        
        # Ordenar por similitud descendente
        coincidences.sort(key=lambda x: x['simility'], reverse=True)
        
        return coincidences
        
    def find_headers(self, text):
        return
    
    def get_model_info(self):
        """Retorna información del modelo cargado"""
        return {
            'total_words': self.model['total_words'],
            'vocabulario_size': self.model['vocabulario_size'],
            'threshold_similarity': self.threshold,
            'campos_disponibles': list(self.model['key_fields'].keys())
        }