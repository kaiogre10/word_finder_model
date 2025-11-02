import re
import numpy as np
import logging
import unicodedata
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import spmatrix #type: ignore

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.params = config
        self.gngr: Tuple[int, int] = self.params["char_ngram_global"]
        self.ngr: Tuple[int, int] = self.params["char_ngram_noise"]
        self.top_ngrams_fraction: int = self.params.get("top_ngrams_fraction", {})
        
    def train_all_vectorizers(self, key_words: Dict[str, Any], noise_words: List[str]):
        # Construir vocabulario normalizado y mapeo variant_to_field
        field_conversion_map_list: List[Dict[str, int]] = self.params.get("field_conversion_map", [])
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
        logger.info(f"Rango global elegido: {self.gngr}")
        logger.info(f"Rango noise elegido: {self.ngr}")

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
                Xg_counts: spmatrix = global_counter.fit_transform(global_vocab)
                gngrams: List[str] = list(global_counter.get_feature_names_out(Xg_counts))

                freqs: np.ndarray[Any, np.dtype[np.float32]] = Xg_counts.sum(axis=0).A1

                scaler = MaxAbsScaler()
                freq_array: np.ndarray[Any, np.dtype[np.float32]] = freqs.reshape(-1, 1)
                scaled_freqs: np.ndarray[Any, np.dtype[np.float32]] = scaler.fit_transform(freq_array).ravel().astype(np.float32)

                gngram_scaled: Dict[str, float] = {
                    ngram: float(s)
                    for ngram, s in zip(gngrams, scaled_freqs)
                }

                gngrams_scaled: List[Tuple[str, float]] = sorted(gngram_scaled.items(), key=lambda x: x[1], reverse=True)
                top_grams: int = int(len(gngrams_scaled)/self.top_ngrams_fraction)

                global_ngrams: List[Tuple[str, float]] = gngrams_scaled[:top_grams]

                logger.info(f"TOP_GLOBAL: {np.array(global_ngrams).shape}")
                
                logger.debug(f"Frecuencias globales ordenadas: {gngrams_scaled}")

                W = np.array(gngrams_scaled)
                Z = np.array(global_ngrams)
                logger.debug(f"TOP GLOBAL: {np.array(Z)}, SORTED GLOBAL: {np.array(W)}")
                logger.debug(f"TOP GLOBAL: {global_ngrams}")

                global_filter: Dict[str, Any] = {
                    "global_counter": global_counter,
                    "global_ngrams": global_ngrams,
                    "all_ngrams_scaled": gngrams_scaled,
                    "global_vocab": global_vocab
                }

                logger.info("GLOBAL FILTER generado")
                return global_filter

        except Exception as e:
            logger.error(f"Error entreando global: {e}", exc_info=True)

    def _train_noise_filter(self, noise_words: List[str]):
        try:
            noise_vocab: List[str] = []
            for word in noise_words:
                for n in range(self.ngr[0], self.ngr[1] + 1):
                    noise_vocab.extend(self._ngrams(word, n))

            if noise_vocab:
                noise_counter = CountVectorizer(
                    strip_accents="unicode",
                    ngram_range=(self.ngr[0], self.ngr[1]),
                    analyzer="char",
                    binary=True,
                    dtype=np.float32
                )
                count_matrix: spmatrix = noise_counter.fit_transform(noise_vocab)
                nngrams: List[str] = noise_counter.get_feature_names_out(count_matrix)
                freqs: np.ndarray[Any, np.dtype[np.float32]] = count_matrix.sum(axis=0).A1

                scaler = MaxAbsScaler()
                freq_array: np.ndarray[Any, np.dtype[np.float32]] = freqs.reshape(-1, 1)
                scaled_freqs: np.ndarray[Any, np.dtype[np.float32]] = scaler.fit_transform(freq_array).ravel().astype(np.float32)

                nngram_scaled: Dict[str, float] = {
                    ngram: float(s)
                    for ngram, s in zip(nngrams, scaled_freqs)
                }

                noise_grams: List[Tuple[str, float]] = sorted(nngram_scaled.items(), key=lambda x: x[1], reverse=True)

                logger.debug(f"SCALED_NOISE: {np.array(noise_grams).shape}")
                logger.debug(f"Frecuencias ruidosas: {noise_grams}")

                noise_filter: Dict[str, Any] = {
                    "noise_counter": noise_counter,
                    "noise_grams": noise_grams,
                    "nngrams": nngrams
                }

                logger.info("NOISE FILTER generado")
                return noise_filter

        except Exception as e:
            logger.error(f"Error entreando global: {e}", exc_info=True)

    def _normalize(self, s: str) -> str:            
        s = s.strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
        # s = re.sub(r"[^a-zA-Z0-9]", "", s)
        return s
    
    def _ngrams(self, s: str, n: int) -> List[str]:
        if n <= 0 or not s:
            return []
        if len(s) < n:
            return []
        return [s[i:i+n] for i in range(len(s) - n + 1)]
