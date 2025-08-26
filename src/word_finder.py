import os
import pickle
import numpy as np
from typing import List, Any, Dict, Literal
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

model_path = os.path.join(os.path.dirname(__file__), '../data/word_finder_model.pkl')

class WordFinder:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.vectorizer = self.model['vectorizer']
        self.threshold = self.model['params']['threshold_similarity']

    def _load_model(self, model_path: str):
        """Carga el modelo pre-entrenado"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Modelo cargado: {model['total_words']} palabras, {model['vocabulario_size']} n-gramas")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Ejecuta generate_model.py primero.")

    def find_keywords(self, text: List[str], search_type: Literal["global", "headers"] = "global") -> List[Dict[str, Any]]:
        """
        Busca coincidencias en el texto según el tipo de búsqueda.
        Args:
            text: Lista de strings a analizar (cada elemento es una línea o bloque).
            search_type: "global" para campos clave, "headers" para encabezados de tabla.
        Returns:
            Lista de coincidencias ordenadas por similitud desc.
        """
        if not text:
            return []

        # aceptar string único también
        single_string = False
        if isinstance(text, str):
            text = [text]
            single_string = True

        try:
            X: spmatrix = self.vectorizer.transform(text)
        except Exception as e:
            logger.exception("Error transformando texto con vectorizer; devolviendo lista vacía.")
            return []

        if search_type == "global":
            candidates = self.model.get('global_words', [])
            Y = self.model.get('Y_global', None)
            mapping_field = self.model.get('variant_to_field', {})
            label_for = lambda w: {'type': 'global', 'field': mapping_field.get(w, None), 'word_found': w}
        elif search_type == "headers":
            candidates = self.model.get('header_words', [])
            Y: spmatrix = self.model.get('Y_headers', None)
            table_headers = self.model.get('table_headers', {})
            label_for = lambda w: {'type': 'header', 'group': table_headers.get(w, None), 'header_found': w}
        else:
            raise ValueError(f"search_type debe ser 'global' o 'headers', no '{search_type}'")

        if not Y or not candidates:
            logger.warning(f"No hay candidatos precomputados para search_type={search_type}")
            return []

        sims: np.ndarray[np.float64, Any] = cosine_similarity(X, Y)
        results: List[Dict[str, Any]] = []
        for row_idx, row in enumerate(sims):
            for col_idx, s in enumerate(row):
                if s > self.threshold:
                    w = candidates[col_idx]
                    base = label_for(w)
                    base['similarity'] = float(s)
                    base['original_text'] = text[row_idx]
                    results.append(base)

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def get_model_info(self):
        """Retorna información del modelo cargado"""
        return {
            'total_words': self.model['total_words'],
            'vocabulario_size': self.model['vocabulario_size'],
            'threshold_similarity': self.threshold,
            'global_words': len(self.model.get('global_words', [])),
            'header_words': len(self.model.get('header_words', [])),
            'campos_disponibles': list(self.model.get('key_fields', {}).keys())
        }