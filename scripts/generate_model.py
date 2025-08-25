import os
import yaml
import pickle
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import spmatrix
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

class ModelGenerator(TransformerMixin, BaseEstimator):
    def __init__(self, config_path: str, project_root: str, param: int=1):
        self.project_root = project_root
        self.config = config_path
        self.param = param

    def fit(self, X=None, y=None):
        logger.debug("Llamada a fit() con X=%s, y=%s", type(X), type(y))
        # compatibilidad sklearn; no hay estado entrenable aquí
        return self

    def transform(self, X: List[str] | str) -> spmatrix:
        """
        X: iterable de strings (o un único string).
        Si el vectorizador ya fue entrenado, devuelve la matriz TF-IDF (sparse),
        si no, devuelve fallback numérico.
        """
        logger.debug("Llamada a transform() con X de tipo %s", type(X))
        if not X :
            logger.warning("transform() recibió X=None, devolviendo matriz vacía.")
            return np.empty((0, 0))
        # permitir string único
        single_string = False
        if isinstance(X, str):
            logger.debug("transform() recibió un string único, convirtiendo a lista.")
            X = [X]
            single_string = True

        vec = getattr(self, "vectorizer_union", None)
        if vec is not None:
            try:
                logger.debug("Usando vectorizer_union para transformar X.")
                out = vec.transform(X)
                logger.debug("Transformación exitosa. Shape: %s", out.shape)
                return out[0] if single_string else out
            except Exception as e:
                logger.error("Error al transformar con vectorizer_union: %s", e, exc_info=True)
                # fallback si algo falla
                pass
        logger.warning("No se pudo usar vectorizer_union, usando fallback numérico.")
        arr = np.full((len(X), 1), fill_value=self.param)
        return arr[0] if single_string else arr
         
    def generate_model(self, config_path: str, project_root: str):
        logger.info("Iniciando generación de modelo.")
        self.config = config_path
        self.project_root = project_root

        logger.info(f"Cargando configuración desde: {config_path}")
        # cargar YAML con validaciones básicas
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        key_fields: Dict[str, Any] = config.get('key_fields', {})
        params: Dict[str, Any] = config.get('params', {})

        logger.debug(f"Campos clave encontrados: {list(key_fields.keys())}")
        logger.debug(f"Parámetros de configuración: {params}")

        # 1) Construcción de listas de candidatos
        global_words: List[str] = []
        variant_to_field: Dict[str, str] = {}
        seen: set[str] = set()

        for field, variants in key_fields.items():
            logger.debug(f"Procesando campo: {field} con variantes: {variants}")
            if variants is None:
                logger.warning(f"Campo '{field}' sin variantes (None) — se omite.")
                continue
            if isinstance(variants, str):
                variants = [variants]
            if not isinstance(variants, (list, tuple)):
                logger.warning(f"Formato inesperado para '{field}': {type(variants).__name__} — se omite.")
                continue

            cleaned: List[str] = []
            for variant in variants:
                if not isinstance(variant, str):
                    logger.warning(f"Variante no string encontrada en '{field}': {variant} — se omite.")
                    continue
                s = variant.strip()
                if not s:
                    continue
                key_norm = s.lower()
                if key_norm in seen:
                    logger.debug(f"Variante normalizada repetida '{key_norm}' en '{field}' — se omite.")
                    continue
                seen.add(key_norm)
                cleaned.append(s)
                variant_to_field[s] = field
            if cleaned:
                logger.debug(f"Variantes limpias para '{field}': {cleaned}")
                global_words.extend(cleaned)

        logger.info("Procesando encabezados de tabla...")
        header_sections: Dict[str, List[str]] = {
            'descriptivo': config.get('descriptivo', []) or [],
            'cuantitativo': config.get('cuantitativo', []) or [],
            'identificador': config.get('identificador', []) or [],
        }
        header_words: List[str] = []
        header_to_group: Dict[str, str] = {}
        seen_headers: set[str] = set()
        for group, headers in header_sections.items():
            for h in headers:
                if not isinstance(h, str):
                    continue
                s = h.strip()
                if not s:
                    continue
                key_norm = s.lower()
                if key_norm in seen_headers:
                    continue
                seen_headers.add(key_norm)
                header_words.append(s)
                header_to_group[s] = group

        # Unión total solo para entrenar el vectorizador
        all_words: List[str] = global_words + header_words
        logger.info(f"Generando modelo con {len(all_words)} palabras clave únicas.")
        logger.debug(f"Palabras clave finales: {all_words}")

        # 2) Vectorizador híbrido
        word_ngram = tuple(params.get('ngram_range', [1, 2]))
        char_ngram = tuple(params.get('char_ngram_range', [3, 5]))
        logger.info(f"Parámetros de ngrama: word_ngram={word_ngram}, char_ngram={char_ngram}")

        logger.info("Inicializando TfidfVectorizer para palabras.")
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

        logger.info("Inicializando TfidfVectorizer para caracteres (char_wb).")
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

        logger.info("Combinando y entrenando vectorizadores.")
        vectorizer_union = FeatureUnion([('word', word_vect), ('char', char_vect)])
        vectorizer_union.fit(all_words)
        logger.info("Vectorizadores entrenados correctamente.")
        self.vectorizer_union = vectorizer_union

        # 3) Vocab size
        try:
            vocab_size: int = 0
            for _name, tr in getattr(vectorizer_union, "transformer_list", []):
                vocab = getattr(tr, "vocabulary_", None)
                if vocab is not None:
                    logger.debug(f"Vocabulario para '{vocab}': {len(vocab)} términos.")
                    vocab_size += len(vocab)
            logger.info(f"Tamaño total del vocabulario: {vocab_size}")
        except Exception as e:
            logger.error(f"Error al calcular el tamaño del vocabulario: {e}", exc_info=True)
            vocab_size = 0

        # 4) Precomputar matrices Y por tipo
        # Nota: se almacenan como matrices sparse; pickle las maneja bien.
        Y_global: spmatrix = vectorizer_union.transform(global_words)
        Y_headers: spmatrix = vectorizer_union.transform(header_words)

        # 5) Guardar modelo
        model = {
            'vectorizer': vectorizer_union,
            'params': params,
            'key_fields': key_fields,              # para mapping/inspección
            'global_words': global_words,          # lista candidatos globales
            'variant_to_field': variant_to_field,  # mapea variante -> campo
            'header_words': header_words,          # lista candidatos encabezados
            'table_headers': header_to_group,      # mapea encabezado -> grupo
            'Y_global': Y_global,                  # matriz candidatos globales
            'Y_headers': Y_headers,                # matriz candidatos encabezados
            'total_words': len(all_words),
            'vocabulario_size': vocab_size,
        }

        output_path = os.path.join(project_root, 'data', 'word_finder_model.pkl')
        logger.info(f"Guardando modelo en: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info("Modelo guardado correctamente.")
        logger.info(f"✓ Modelo generado exitosamente:")
        logger.info(f"  - Vocabulario: {model['vocabulario_size']} n-gramas")
        logger.info(f"  - Palabras clave globales: {len(global_words)} | encabezados: {len(header_words)}")
        logger.info(f"  - Guardado en: {output_path}")
