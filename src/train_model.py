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
                global_words.append(s)
                variant_to_field[s] = field_id
        
        global_filter = self._train_global(global_words)
        noise_filter = self._train_noise_filter(noise_words)
        logger.debug(f"Rango n-gramas elegido: {self.ngrams}")

        return global_filter, noise_filter, global_words, variant_to_field
    
    def _train_global(self, global_words: List[str]):
        try:
            global_vocab: List[str] = []
            for word in global_words:
                for n in range(self.ngrams[0], self.ngrams[1] + 1):
                    global_vocab.extend(self._ngrams(word, n))
            
            if global_vocab:
                global_counter = CountVectorizer(
                    strip_accents="unicode",
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
                top_grams_count: int = int(len(gngrams_scaled) / self.top_ngrams_fraction)
                
                # Extraer solo los strings de los n-gramas, descartando el score/frecuencia
                global_ngrams_list: List[str] = [ngram for ngram, _ in gngrams_scaled[:top_grams_count]]

                global_matrices: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
                for n in range(self.ngrams[0], self.ngrams[1] + 1):
                    # Seleccionamos los top N n-gramas EXCLUSIVAMENTE de este tamaño 'n'
                    ngrams_of_size = [ng for ng, score in gngrams_scaled if len(ng) == n]
                    
                    # Generamos la matriz con un número de filas estandarizado (N)
                    global_matrices[n] = self._generate_matrix(n, ngrams_of_size, top_grams_count)

                for ngrams, matrix in global_matrices.items():
                    logger.info(f"{matrix.shape}")
                    
                return {
                    "global_matrices": global_matrices,
                    "global_ngrams": global_ngrams_list
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
            
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()

            q = re.sub(r"[^a-z\s]+", " ", q)

            # dejar solo un espacio entre palabras y quitar extremos
            return re.sub(r"\s+", " ", q).strip()
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