import re
import numpy as np
import logging
import unicodedata
from typing import List, Dict, Any, Tuple, Set
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        self.params = config.get("params", {})
        
    def train_all_vectorizers(self, key_words: Dict[str, List[str]], global_words: List[str], noise_words: List[str]):
        self.ngr: Tuple[int, int]  = tuple(self.params.get("char_ngram_range", []))
        self.gngr: Tuple[int, int] = tuple(self.params.get("char_ngram_global", []))
        self.top_ngrams = self.params.get("top_ngrams", [])
        self.top_ngrams_fraction = self.params.get("top_ngrams_fraction", [])

        all_vectorizers = self._train_tfidftransformer(key_words)
        global_filter = self._train_global(global_words)
        noise_filter = self._train_noise_filter(noise_words)

        return all_vectorizers, global_filter, noise_filter

    def _train_tfidftransformer(self, key_words: Dict[str, List[str]]) -> Dict[Any, Any]:
        try:
            all_vectorizers = {}
            for field, variants in key_words.items():
                if not variants:
                    continue
            
                field_variants_normalized = [self._normalize(v) for v in variants if isinstance(v, str)]
                field_variants_normalized = [s for s in field_variants_normalized if s]

                if field_variants_normalized:
                    vocabulary: Set[str] = set()
                    for variant in field_variants_normalized:
                        for n in range(self.ngr[0], self.ngr[1] + 1):
                            vocabulary.update(self._ngrams(variant, n))

                    if not vocabulary:
                        continue

                    counter = CountVectorizer(
                        strip_accents="unicode",
                        ngram_range=(self.ngr[0], self.ngr[1]),
                        analyzer="char",
                        binary=True,
                        dtype=np.float32
                    )
                    X = counter.fit_transform(vocabulary).astype(np.float32)
                    ngrams: List[str] = counter.get_feature_names_out(X)

                    x_array = X.toarray()

                    logger.info(f"Tamaño del array: {x_array.shape}")

                    freqs = np.asarray(x_array.sum(axis=0)).ravel()
                    ngram_freqs: Dict[str, float] = {
                        ngram: float(freq)
                        for ngram, freq in zip(ngrams, freqs)
                    }

                    field_ngrams: List[Tuple[str, float]] = sorted(ngram_freqs.items(), key=lambda x: x[1], reverse=True)

                    # Normalizar frecuencias con MinMaxScaler (espera entrada 2D)
                    scaler = MaxAbsScaler()
                    freq_array = freqs.reshape(-1, 1)                    # -> shape (n_ngrams, 1)
                    scaled_freqs = scaler.fit_transform(freq_array).ravel()  # -> 1D array de long n_ngrams

                    ngram_scaled: Dict[str, float] = {
                        ngram: float(s)
                        for ngram, s in zip(ngrams, scaled_freqs)
                    }

                    ngrams_scaled: List[Tuple[str, float]] = sorted(ngram_scaled.items(), key=lambda x: x[1], reverse=True)

                    #logger.info(f"COUNTER: {X}")
                    
                    # logger.info(f"SORTED_COUNTER para {field}: {field_ngrams}")

                    # logger.info(f"SCALED_COUNTER para {field}: {ngrams_scaled}")


                    #logger.info(f"COUNTER SHAPE por {field}: {X.toarray().shape[1]}")

                    all_vectorizers[field] = {
                        "counter": counter,
                        # "field_ngrams": field_ngrams,
                        "ngrams_scaled": ngrams_scaled,
                        "ngram_range": (self.ngr[0], self.ngr[1]),
                    }
            logger.info("TRANSFORMER generad")
            return all_vectorizers
        
        except Exception as e:
            logger.error(f"Error entrenando tramsformer: {e}", exc_info=True)
            return {}
    
    def _train_global(self, global_words: List[str]) -> Dict[Any, Any]:
        NP_MODE: str = "shape"
        def nparray(array, mode=NP_MODE) -> str:
            if mode == "shape":
                return array.shape
            elif mode == "size":
                return array.size
            elif mode == "dtype":
                return array.dtype
            elif mode == "ndim":
                return array.ndim
            else:
                raise ValueError(f"Modo desconocido: {mode}")
    
        try:
            global_vocab: Set[List[str]] = set()
            for word in global_words:
                for n in range(self.gngr[0], self.gngr[1] + 1):
                    global_vocab.update(self._ngrams(word, n))
            
            if global_vocab:
                global_counter = CountVectorizer(
                    strip_accents="unicode",
                    ngram_range=(self.gngr[0], self.gngr[1]),
                    analyzer="char",
                    binary=True,
                    dtype=np.float32
                )
                Xg_counts = global_counter.fit_transform(global_vocab).astype(np.float32)
                gngrams: List[str] = global_counter.get_feature_names_out(Xg_counts)

                freqs = Xg_counts.sum(axis=0).A1
                ngram_freqs: Dict[str, float] = {
                    ngram: float(freq)
                    for ngram, freq in zip(gngrams, freqs)
                }
                sorted_ngrams: List[Tuple[str, float]] = sorted(ngram_freqs.items(), key=lambda x: x[1], reverse=True)

                scaler = MaxAbsScaler()
                freq_array = freqs.reshape(-1, 1)                    
                scaled_freqs = scaler.fit_transform(freq_array).ravel()  # -> 1D array de long n_ngrams

                gngram_scaled: Dict[str, float] = {
                    ngram: float(s)
                    for ngram, s in zip(gngrams, scaled_freqs)
                }

                gngrams_scaled: List[Tuple[str, float]] = sorted(gngram_scaled.items(), key=lambda x: x[1], reverse=True)
                top_grams = int(len(gngrams_scaled)/(self.top_ngrams_fraction))

                global_ngrams: List[Tuple[str, float]] = gngrams_scaled[:top_grams]
                
                logger.debug(f"Frecuencias globales ordenadas: {gngrams_scaled}")

                W = np.array(gngrams_scaled)
                Z = np.array(global_ngrams)
                logger.info(f"TOP GLOBAL: {nparray(Z)}, SORTED GLOBAL: {nparray(W)}")
                logger.debug(f"TOP GLOBAL: {global_ngrams}")

                global_filter: Dict[str, Any] = {
                    "global_counter": global_counter, 
                    "char_ngram_global": [self.gngr[0], self.gngr[1]],
                    "global_ngrams": global_ngrams,
                    "all_ngrams_scaled": gngrams_scaled,
                    "global_vocab": global_vocab
                }

            logger.info("GLOBAL FILTER generado")
            return global_filter

        except Exception as e:
            logger.error(f"Error entreando global: {e}", exc_info=True)

    def _train_noise_filter(self, noise_words: List[str]) -> Dict[Any, Any]:

        try:
            noise_vocab: Set[List[str]] = set()
            for word in noise_words:
                for n in range(self.gngr[0], self.gngr[1] + 1):
                    noise_vocab.update(self._ngrams(word, n))

            if noise_vocab:
                noise_counter = CountVectorizer(
                    strip_accents="unicode",
                    ngram_range=(self.gngr[0], self.gngr[1]),
                    analyzer="char",
                    binary=True,
                    dtype=np.float32
                )
                Xn_counts = noise_counter.fit_transform(noise_vocab).astype(np.float32)
                nngrams: List[str] = noise_counter.get_feature_names_out(Xn_counts)

                freqs = Xn_counts.sum(axis=0).A1
                ngram_freqs: Dict[str, float] = {
                    ngram: float(freq)
                    for ngram, freq in zip(nngrams, freqs)
                }
                sorted_ngrams: List[Tuple[str, float]] = sorted(ngram_freqs.items(), key=lambda x: x[1],
                                                                reverse=True)

                scaler = MaxAbsScaler()
                freq_array = freqs.reshape(-1, 1)
                scaled_freqs = scaler.fit_transform(freq_array).ravel()  # -> 1D array de long n_ngrams

                nngram_scaled: Dict[str, float] = {
                    ngram: float(s)
                    for ngram, s in zip(nngrams, scaled_freqs)
                }

                noise_grams: List[Tuple[str, float]] = sorted(nngram_scaled.items(), key=lambda x: x[1], reverse=True)

                logger.debug(f"Frecuencias ruidosas: {noise_grams}")

                noise_filter: Dict[str, Any] = {
                    "noise_counter": noise_counter,
                    "noise_grams": noise_grams,
                }

            logger.info("NOISE FILTER generado")
            return noise_filter

        except Exception as e:
            logger.error(f"Error entreando global: {e}", exc_info=True)

    def _normalize(self, s: str) -> str:
        if not s:
            return ""
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