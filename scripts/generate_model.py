import os
import yaml
import pickle
from typing import List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class ModelGenerator(TransformerMixin, BaseEstimator):
    def __init__(self, data_path: str, project_root: str, param=1):
        self.project_root = project_root 
        self.data_path = data_path
        self.param = param
        
    def fit(self, all_words: List[Any], y=None):
        return self
        
    def transformer_list(self, all_words: List[Any]):
        return np.full(shape=len(all_words), fill_value=self.param)    
        
    def generate_model(self, data_path: str, project_root: str):
        self.data_path = data_path
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Extraer todas las palabras
        all_words: List[Any] = []
        for field, variants in config['key_fields'].items():
            all_words.extend(variants)
        
        logger.info(f"Generando modelo con {len(all_words)} palabras clave...")
        
        # Crear y entrenar vectorizador (híbrido: word + char_wb)
        # parámetros principales (se usan params.ngram_range como word_ngram por compatibilidad)
        word_ngram = tuple(config['params'].get('ngram_range', [1, 1]))
        char_ngram = tuple(config['params'].get('char_ngram_range', [3, 5]))

        word_vect = TfidfVectorizer(
            analyzer='word',
            strip_accents='ascii',
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b',
            ngram_range=word_ngram,
            sublinear_tf=True,
            norm='l2',
            max_df=0.95,
            min_df=1
        )

        char_vect = TfidfVectorizer(
            analyzer='char_wb',
            strip_accents='ascii',
            lowercase=True,
            ngram_range=char_ngram,
            sublinear_tf=True,
            norm='l2',
            max_df=0.99,
            min_df=1
        )

        # unir y entrenar
        vectorizer_union = FeatureUnion([('word', word_vect), ('char', char_vect)])
        vectorizer_union.fit(all_words)

        # vocab size sumando vocabularios de cada sub-vectorizer
        try:
            vocab_size = sum(len(t[1].vocabulary_) for t in vectorizer_union.transformer_list)
        except Exception:
            vocab_size = 0
         
         # Preparar modelo para guardar
        model = {
            'vectorizer': vectorizer_union,
            'key_fields': config['key_fields'],
            'params': config['params'],
            'vocabulario_size': vocab_size,
            'total_palabras': len(all_words)
        }
        
        # Guardar modelo
        output_path = os.path.join(project_root, 'data', 'word_finder_model.pkl')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"✓ Modelo generado exitosamente:")
        logger.info(f"  - Vocabulario: {model['vocabulario_size']} n-gramas")
        logger.info(f"  - Palabras clave: {model['key_fields']}")
        logger.info(f"  - Guardado en: {output_path}")
