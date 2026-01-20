import re
import numpy as np
import logging
import unicodedata
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.params = config
        self.ngrams: Tuple[int, int] = self.params["char_ngrams"]
        self.top_ngrams_fraction: int = self.params.get("top_ngrams_fraction", 0)
        
    def train_all_vectorizers(self, key_words: Dict[str, Any], noise_words: List[str]):
        # Construir vocabulario normalizado y mapeo variant_to_field
        field_conversion_map_list: List[Dict[str, int]] = self.params["field_conversion_map"]
        field_conversion_map = {k: v for d in field_conversion_map_list for k, v in d.items()}
        
        global_words: List[str] = []
        variant_to_field: Dict[str, int] = {}
        all_ngrams: Dict[str, Dict[int, List[str]]] = {}

        for field, variants in key_words.items():
            if not variants:
                continue
            if isinstance(variants, str):
                variants = [variants]
            if not isinstance(variants, (list, tuple)):
                continue

            field_id = field_conversion_map.get(field)
            if field_id is None:
                logger.warning(f"El campo '{field}' no se encontró en 'field_conversion_map' y será omitido.")
                continue

            for v in variants:
                if not isinstance(v, str):
                    continue
                s = self._normalize(v)
                if not s:
                    continue
                ngrams_structure: Dict[int, List[str]] = {}
                
                # Iterar sobre el rango de N definido en config (ej: 2 a 3)
                for n in range(self.ngrams[0], self.ngrams[1] + 1):
                    # Generar n-gramas usando el método existente _ngrams
                    # IMPORTANTE: Se mantiene como LISTA (no set) para permitir 
                    # el algoritmo de "Greedy Unique Match" (conteo de repeticiones).
                    grams_list = self._ngrams(s, n)
                    ngrams_structure[n] = grams_list
                
                # Asignar al diccionario maestro
                all_ngrams[s] = ngrams_structure

                global_words.append(s)
                variant_to_field[s] = field_id
        
        # logger.info(f"All ngrams: {all_ngrams}")
        global_filter = self._train_global(global_words)
        noise_filter = self._train_noise_filter(noise_words)
        logger.debug(f"Rango n-gramas elegido: {self.ngrams}")

        return global_filter, noise_filter, global_words, variant_to_field, all_ngrams
    
    def _train_global(self, global_words: List[str]):
        try:
            global_vocab: List[str] = []
            for word in global_words:
                for n in range(self.ngrams[0], self.ngrams[1] + 1):
                    global_vocab.extend(self._ngrams(word, n))
            
            if global_vocab:
                global_counter = CountVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    ngram_range=(self.ngrams[0], self.ngrams[1]),
                    analyzer="char",
                    binary=True,
                    dtype=np.float32
                )
                
                Xg_counts: np.ndarray[str, np.dtype[np.float32]] = global_counter.fit_transform(global_vocab)
                gngrams = list(global_counter.get_feature_names_out(str(Xg_counts)))

                freqs: np.ndarray[Any, np.dtype[np.float32]] = Xg_counts.sum(axis=0).A1

                scaler = MaxAbsScaler()
                freq_array = freqs.reshape(-1, 1)
                scaled_freqs = scaler.fit_transform(freq_array).ravel().astype(np.float32)
                
                gngram_scaled: Dict[str, float] = {
                    ngram: float(s)
                    for ngram, s in zip(gngrams, scaled_freqs)
                }

                gngrams_scaled: List[Tuple[str, float]] = sorted(gngram_scaled.items(), key=lambda x: x[1], reverse=True)
                
                # 1. Calcular tamaño máximo usando TODOS los ngramas de TODAS las longitudes
                min_n = self.ngrams[0]
                total_ngrams_all_sizes = len(gngrams)  # Todos los ngramas sin filtrar
                filas_max_n = int(total_ngrams_all_sizes / self.top_ngrams_fraction)

                logger.info(f"Numero máximo de filas: {filas_max_n}")
                
                # 2. Filtrar solo ngramas con frecuencia absoluta > 2 para poblar la matriz
                gngrams_scaled = [
                    (ng, score) for ng, score in gngrams_scaled
                    if freqs[gngrams.index(ng)] > 2
                ]
                
                global_matrices: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
                for n in range(self.ngrams[0], self.ngrams[1] + 1):
                    ngrams_of_size = [ng for ng, score in gngrams_scaled if len(ng) == n]
                    # Calcula el número de filas máximas inversamente proporcional a n
                    filas_n = max(1, int(filas_max_n * min_n / n))
                    global_matrices[n] = self._generate_matrix(n, ngrams_of_size, filas_n)

                for matrix in global_matrices.values():
                    logger.info(f"{matrix.shape}")
                                    
                return {
                    "global_matrices": global_matrices,
                }

        except Exception as e:
            logger.error(f"Error entreando global: {e}", exc_info=True)

    def _train_noise_filter(self, noise_words: List[str]):
        try:
            noise_grams_per_word: List[Dict[int, set[str]]] = []

            for word in noise_words:
                # Nos aseguramos de usar la misma normalización
                if not word:
                    noise_grams_per_word.append({})
                    continue
                
                word_grams_dict: Dict[int, set[str]] = {}
                
                for n in range(self.ngrams[0], self.ngrams[1] + 1):
                    grams = self._ngrams(word, n)
                    if grams:
                        word_grams_dict[n] = set(grams)
                
                noise_grams_per_word.append(word_grams_dict)

            noise_filter: Dict[str, Any] = {
                "noise_grams": noise_grams_per_word
            }

            logger.warning(f"NOISE FILTER generado: {len(noise_grams_per_word)} perfiles de ruido pre-calculados.")
            return noise_filter

        except Exception as e:
            logger.error(f"Error entreando global: {e}", exc_info=True)

    def _normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
            # Normaliza y elimina acentos
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()
            # Deja solo letras a-z y espacios
            q = re.sub(r"[^a-z\s]+", " ", q)
            # Elimina espacios extra y extremos
            q = re.sub(r"\s+", " ", q).strip()
            # Filtra cualquier caracter fuera del rango ASCII seguro (32-126)
            q = ''.join(c for c in q if 32 <= ord(c) <= 126)
            return q
        except Exception as e:
            logger.error(msg=f"Error limpiando texto: {e}", exc_info=True)
        return ""
    
    def _ngrams(self, s: str, n: int) -> List[str]:
        if n <= 0 or not s:
            return []
        if len(s) < n:
            return []
        return [s[i:i+n] for i in range(len(s) - n + 1)]
    
    def _generate_matrix(self, size: int, ngrams: List[str], max_rows: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Genera una matriz de tamaño fijo (max_rows x size) usando uint8."""
        # Inicializamos matriz de ceros (espacio vacío) para asegurar el tamaño exacto
        matrix = np.zeros((max_rows, size), dtype=np.uint8)
        
        for i, ng in enumerate(ngrams[:max_rows]):
            # Convertimos cada caracter a su valor numérico uint8
            matrix[i] = [ord(char) for char in ng]
            
        return matrix