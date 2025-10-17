import re
import numpy as np
import logging
import unicodedata
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
import scipy as sp #type: ignore

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        self.params = config.get("params", {})
        
    def train_all_vectorizers(self, key_words: Dict[str, List[str]], global_words: List[str], noise_words: List[str]):

        self.gngr: Tuple[int, int] = tuple(self.params.get("char_ngram_global", []))
        all_vectorizers = self._train_tfidftransformer(key_words)
        global_filter = self._train_global(global_words)
        noise_filter = self._train_noise_filter(noise_words)

        return all_vectorizers, global_filter, noise_filter

    def _train_tfidftransformer(self, key_words: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        ngr: Tuple[int, int] = tuple(self.params.get("char_ngram_range", []))
        try:
            all_vectorizers = {}
            for field, variants in key_words.items():
                if not variants:
                    continue
            
                field_variants_normalized = [self._normalize(v) for v in variants if isinstance(v, str)] # type: ignore
                field_variants_normalized = [s for s in field_variants_normalized if s]

                if field_variants_normalized:
                    vocabulary: List[str] = []
                    for variant in field_variants_normalized:
                        for n in range(ngr[0], ngr[1] + 1):
                            vocabulary.extend(self._ngrams(variant, n))

                    if not vocabulary:
                        continue

                    counter: CountVectorizer = CountVectorizer(
                        strip_accents="unicode",
                        ngram_range=(ngr[0], ngr[1]),
                        analyzer="char",
                        binary=True,
                        dtype=np.float32
                    )
                    X: sp.sparse.spmatrix = counter.fit_transform(vocabulary)
                    
                    ngrams: List[str] = list(counter.get_feature_names_out(X))

                    x_array: np.ndarray[Any, np.dtype[np.float32]] = X.toarray()

                    logger.info(f"Tamaño del array: {x_array.shape}")

                    freqs: np.ndarray[Any, np.dtype[np.float32]] = np.asarray(x_array.sum(axis=0)).ravel().astype(np.float32)
                    # ngram_freqs: Dict[str, float] = {
                    #     ngram: float(freq)
                    #     for ngram, freq in zip(ngrams, freqs)
                    # }

                    # field_ngrams: List[Tuple[str, float]] = sorted(ngram_freqs.items(), key=lambda x: x[1], reverse=True)

                    # Normalizar frecuencias con MinMaxScaler (espera entrada 2D)
                    scaler = MaxAbsScaler()
                    freq_array: np.ndarray[Any, np.dtype[np.float32]] = freqs.reshape(-1, 1) # -> shape (n_ngrams, 1)
                    scaled_freqs: np.ndarray[Any, np.dtype[np.float32]] = scaler.fit_transform(freq_array).ravel().astype(np.float32)  # -> 1D array de long n_ngrams

                    ngram_scaled: Dict[str, float] = {
                        ngram: float(s)
                        for ngram, s in zip(ngrams, scaled_freqs)
                    }

                    ngrams_scaled: List[Tuple[str, float]] = sorted(ngram_scaled.items(), key=lambda x: x[1], reverse=True)
                    #logger.info(f"COUNTER: {X}")
                    # logger.info(f"SORTED_COUNTER para {field}: {field_ngrams}")
                    # logger.info(f"SCALED_COUNTER para {field}: {np.array(ngrams_scaled).shape}")
                    #logger.info(f"COUNTER SHAPE por {field}: {X.toarray().shape[1]}")

                    all_vectorizers[field] = {
                        "counter": counter,
                        "ngrams_scaled": ngrams_scaled,
                        "ngram_range": (ngr[0], ngr[1]),
                    }
                    
            logger.info("TRANSFORMER generad")
            return all_vectorizers
        
        except Exception as e:
            logger.error(f"Error entrenando tramsformer: {e}", exc_info=True)
            return {}
    
    def _train_global(self, global_words: List[str]):
        top_ngrams_fraction: int = self.params.get("top_ngrams_fraction")
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
                Xg_counts: sp.sparse.spmatrix = global_counter.fit_transform(global_vocab)
                gngrams: List[str] = list(global_counter.get_feature_names_out(Xg_counts))

                freqs: np.ndarray[Any, np.dtype[np.float32]] = Xg_counts.sum(axis=0).A1
                # ngram_freqs: Dict[str, float] = {
                #     ngram: float(freq)
                #     for ngram, freq in zip(gngrams, freqs)
                # }
                # sorted_ngrams: List[Tuple[str, float]] = sorted(ngram_freqs.items(), key=lambda x: x[1], reverse=True)

                scaler = MaxAbsScaler()
                freq_array: np.ndarray[Any, np.dtype[np.float32]] = freqs.reshape(-1, 1)                    
                scaled_freqs: np.ndarray[Any, np.dtype[np.float32]] = scaler.fit_transform(freq_array).ravel().astype(np.float32)

                gngram_scaled: Dict[str, float] = {
                    ngram: float(s)
                    for ngram, s in zip(gngrams, scaled_freqs)
                }

                gngrams_scaled: List[Tuple[str, float]] = sorted(gngram_scaled.items(), key=lambda x: x[1], reverse=True)
                top_grams: int = int(len(gngrams_scaled)/(top_ngrams_fraction))

                global_ngrams: List[Tuple[str, float]] = gngrams_scaled[:top_grams]
                
                logger.debug(f"Frecuencias globales ordenadas: {gngrams_scaled}")

                W = np.array(gngrams_scaled)
                Z = np.array(global_ngrams)
                logger.debug(f"TOP GLOBAL: {np.array(Z)}, SORTED GLOBAL: {np.array(W)}")
                # logger.debug(f"TOP GLOBAL: {global_ngrams}")

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

    def _train_noise_filter(self, noise_words: List[str]):

        try:
            noise_vocab: List[str] = []
            for word in noise_words:
                for n in range(self.gngr[0], self.gngr[1] + 1):
                    noise_vocab.extend(self._ngrams(word, n))

            if noise_vocab:
                noise_counter = CountVectorizer(
                    strip_accents="unicode",
                    ngram_range=(self.gngr[0], self.gngr[1]),
                    analyzer="char",
                    binary=True,
                    dtype=np.float32
                )
                Xn_counts: sp.sparse.spmatrix = noise_counter.fit_transform(noise_vocab)
                nngrams: List[str] = list(noise_counter.get_feature_names_out(Xn_counts))

                freqs: np.ndarray[Any, np.dtype[np.float32]] = Xn_counts.sum(axis=0).A1
                # ngram_freqs: Dict[str, float] = {
                #     ngram: float(freq)
                #     for ngram, freq in zip(nngrams, freqs)
                # }
                # sorted_ngrams: List[Tuple[str, float]] = sorted(ngram_freqs.items(), key=lambda x: x[1], reverse=True)

                scaler = MaxAbsScaler()
                freq_array: np.ndarray[Any, np.dtype[np.float32]] = freqs.reshape(-1, 1)
                scaled_freqs: np.ndarray[Any, np.dtype[np.float32]] = scaler.fit_transform(freq_array).ravel().astype(np.float32)  # -> 1D array de long n_ngrams

                nngram_scaled: Dict[str, float] = {
                    ngram: float(s)
                    for ngram, s in zip(nngrams, scaled_freqs)
                }

                noise_grams: List[Tuple[str, float]] = sorted(nngram_scaled.items(), key=lambda x: x[1], reverse=True)

                logger.info(f"SCALED_NOISE: {np.array(noise_grams).shape}")

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
        if s is None:
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
