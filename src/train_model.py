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
        self.gngr: Tuple[int, int] = self.params["char_ngram_global"]
        self.ngr: Tuple[int, int] = self.params["char_ngram_noise"]
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
        logger.debug(f"Rango global elegido: {self.gngr}")
        logger.debug(f"Rango noise elegido: {self.ngr}")

        return global_filter, noise_filter, global_words, variant_to_field
    
    def _train_global(self, global_words: List[str]):
        try:
            global_vocab: List[str] = []
            for word in global_words:
                for n in range(self.gngr[0], self.gngr[1] + 1):
                    global_vocab.extend(self._ngrams(word, n))
            
            if global_vocab:
                global_counter = CountVectorizer(
                    strip_accents="unicode",
                    ngram_range=(self.gngr[0], self.gngr[1]),
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

                logger.info(f"TOP GLOBAL (solo strings): {global_ngrams_list}")

                global_filter: Dict[str, Any] = {
                    "global_counter": global_counter,
                    "global_ngrams": set(global_ngrams_list),
                }

                logger.warning("GLOBAL FILTER generado")
                return global_filter

        except Exception as e:
            logger.error(f"Error entreando global: {e}", exc_info=True)

    def _train_noise_filter(self, noise_words: List[str]):
        try:
            
            noise_vocab: List[str] = []
            all_noise: List[List[str]] = [] 

            for word in noise_words:
                all_noise.append(word)
                per_word_ngrams: List[str] = []

                for n in range(self.ngr[0], self.ngr[1] + 1):
                    noise_ngrams = self._ngrams(word, n)

                    noise_vocab.extend(noise_ngrams)
                    
                    per_word_ngrams.extend(noise_ngrams)

                per_word_ngrams.append(noise_ngrams)

                all_noise.append(per_word_ngrams)
            
            # logger.info(f"ALL; {all_noise}")

            # for i in all_noise:
                # logger.info(f"ALL; {all_noise[i]}")

            if noise_vocab:
                noise_counter = CountVectorizer(
                    strip_accents="unicode",
                    ngram_range=(self.ngr[0], self.ngr[1]),
                    analyzer="char",
                    binary=True,
                    dtype=np.float32
                )
                count_matrix: np.ndarray[str, np.dtype[np.float32]] = noise_counter.fit_transform(noise_vocab)
                nngrams = list(noise_counter.get_feature_names_out(str(count_matrix)))
                freqs: np.ndarray[Any, np.dtype[np.float32]] = count_matrix.sum(axis=0).A1

                scaler = MaxAbsScaler()
                freq_array: np.ndarray[Any, np.dtype[np.float32]] = freqs.reshape(-1, 1)
                scaled_freqs: np.ndarray[Any, np.dtype[np.float32]] = scaler.fit_transform(freq_array).ravel().astype(np.float32)

                nngram_scaled: Dict[str, float] = {
                    ngram: float(s)
                    for ngram, s in zip(nngrams, scaled_freqs)
                }

                noise_grams: List[Tuple[str, float]] = sorted(nngram_scaled.items(), key=lambda x: x[1], reverse=True)
                logger.warning(f"SCALED_NOISE: {np.array(noise_grams).shape}")
                # logger.info(f"Frecuencias ruidosas: {noise_vocab}")

                noise_filter: Dict[str, Any] = {
                    "noise_counter": noise_counter,
                    "noise_grams": noise_grams,
                    "nngrams": nngrams
                }

                logger.warning("NOISE FILTER generado")
                return noise_filter

        except Exception as e:
            logger.error(f"Error entreando global: {e}", exc_info=True)

    def _normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
            
            # Normalizar para separar tildes y convertir a minúsculas
            # NFKD separa letras de sus acentos; luego 'ignore' los elimina al codificar a ASCII
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()

            # eliminar todo lo que no sean letras a-z y espacios
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
