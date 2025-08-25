import os
import pickle
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

class WordFinder:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../data/word_finder_model.pkl')
        
        self.model = self._load_model(model_path)
        self.vectorizer = self.model['vectorizer']
        self.umbral = self.model['config']['umbral_similitud']
        
    def _load_model(self, model_path):
        """Carga el modelo pre-entrenado"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Modelo cargado: {model['total_palabras']} palabras, {model['vocabulario_size']} n-gramas")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Ejecuta generate_model.py primero.")
    
    def seek_field(self, poligon_text):
        """
        Busca campos clave en el texto del polígono
        
        Args:
            texto_poligono (str): Texto extraído del polígono
            
        Returns:
            list: Lista de coincidencias encontradas
        """
        if not poligon_text.strip():
            return []
        
        # Vectorizar texto del polígono
        texto_vectorizado = self.vectorizer.transform([poligon_text])
        
        # Vectorizar todas las palabras clave
        all_words = []
        mapping_field = {}
        
        for field, variantes in self.model['campos_clave'].items():
            for variante in variantes:
                all_words.append(variante)
                mapping_field[variante] = field
        
        vectorized_words = self.vectorizer.transform(all_words)
        
        # Calcular similitudes
        similitudes = cosine_similarity(texto_vectorizado, vectorized_words)[0]
        
        # Encontrar coincidencias
        coincidences = []
        for i, similitud in enumerate(similitudes):
            if similitud > self.umbral:
                word = vectorized_words[i]
                field = mapping_field[word]
                coincidences.append({
                    'field': field,
                    'word_finded': word,
                    'similitud': float(similitud),
                    'original_text': poligon_text
                })
        
        # Ordenar por similitud descendente
        coincidences.sort(key=lambda x: x['similitud'], reverse=True)
        
        return coincidences
    
    def get_info_modelo(self):
        """Retorna información del modelo cargado"""
        return {
            'total_words': self.model['total_words'],
            'vocabulario_size': self.model['vocabulario_size'],
            'umbral_similitud': self.umbral,
            'campos_disponibles': list(self.model['key_fields'].keys())
        }