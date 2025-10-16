import re
import numpy as np
import logging
import unicodedata
from typing import List, Dict, Any, Tuple, Set
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import scipy as sp

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        self.params = config.get("params", {})
        
    def train_all_vectorizers(self, key_words: Dict[str, List[str]], global_words: List[str], noise_words: List[str]) -> Dict[str, Any]:
        self.ngr: Tuple[int, int]  = self.params.get("char_ngram_range", [])
        self.gngr: Tuple[int, int] = self.params.get("char_ngram_global", [])
        self.top_ngrams = self.params.get("top_ngrams", [])
        self.top_ngrams_fraction = self.params.get("top_ngrams_fraction", [])
        self.key_words = key_words
        self.global_words = global_words

        all_vectorizers = self._train_tfidftransformer(key_words)
        global_filter = self._train_global(global_words)
        all_maps = self._train_map_vectorizar(key_words)

        return all_vectorizers, global_filter, all_maps

    def _train_tfidftransformer(self, key_words: Dict[str, List[str]]) -> Dict[str, Any]:
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
                        analyzer="char_wb",
                        binary=False,
                        dtype=np.float32
                    )
                    X: np.ndarray[Any, np.float32] = counter.fit_transform(vocabulary).astype(np.float32)
                    ngrams: List[str] = counter.get_feature_names_out()
                    
                    ngram_freqs = X.sum(axis=1).A1
                    ngram_freqs: Dict[str, float] = {
                        ngram: ngram_freqs[idx]
                        for ngram, idx in ngrams
                        }
                    sorted_ngrams: List[Tuple[str, float]] = sorted(ngram_freqs.items(), key=lambda x: x[1], reverse=True)
                        

                    logger.info(f"COUNTER: {X}")
                    
                    logger.info(f"SORTED_COUNTER: {sorted_ngrams}")
                    # logger.info(f"FEATURES: {ngrams}")
                    # logger.info(f"COUNTER SHAPE por {field}: {X.toarray().shape[1]}")
                    # tfidf_tr = TfidfTransformer(
                    #     norm="l1", 
                    #     use_idf=True,
                    #     smooth_idf=True, 
                    #     sublinear_tf=False
                    # )

                    # X_counts: np.ndarray[Any, np.dtype[np.uint32]] = counter.fit_transform(field_variants_normalized).astype(np.uint32)
                    # X_tfidf: np.ndarray[Any, np.dtype[np.float32]] = tfidf_tr.fit_transform(X_counts).astype(np.float32)
                    # N_tfidf = np.array(X_tfidf.shape, dtype=np.float32)

                    # # logger.info(f" Numero de features: {N_tfidf[1]}")
                    
                    # lengths = np.array([[len(w)] for w in field_variants_normalized], dtype=np.float32)

                    # X_features = sp.sparse.hstack([X_tfidf, lengths])
                    # # logger.info(f"Features fusionados1: {X_features}")
                    
                    # X_featuresarr = np.asarray(X_features)
                    
                    # logger.info(f"TRANSFORMER: '{field}', features: {np.array(X_tfidf.shape[:2])}")
                    # # logger.info(f"Features fusionados: {X_featuresarr}")

                    all_vectorizers[field] = {
                        "counter": counter,
                        # "tfidf": tfidf_tr,
                        # "feature_names": tfidf_tr.get_feature_names_out().tolist() + ["length"],
                        # "n_features": N_tfidf[1],
                    }

            # logger.info(f"N_features: {N_tfidf[1]}")
            # logger.debug(f"Features puros completos: {np.array(X_tfidf.size)}")
            # logger.debug(f"Features fusionados completos: {np.array(X_counts.shape)}")
            logger.info("TRANSFORMER generado")
            return all_vectorizers
        
        except Exception as e:
            logger.error(f"Error entrenando tramsformer: {e}", exc_info=True)
            return {}
    
    def _train_global(self, global_words: List[str]) -> Dict[str, Any]:
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
                    strip_accents="ascii",
                    ngram_range=(self.gngr[0], self.gngr[1]),
                    analyzer="char_wb",
                    binary=True,
                    vocabulary=list(global_vocab),
                    dtype=np.float32
                )
                Xg_counts: np.ndarray[Any, np.float32] = global_counter.fit_transform(global_words).astype(np.float32)

                ngram_freqs = Xg_counts.sum(axis=0).A1
                ngram_freqs: Dict[str, float] = {
                    ngram: ngram_freqs[idx]
                    for ngram, idx in global_counter.vocabulary_.items()
                }
                sorted_ngrams: List[Tuple[str, float]] = sorted(ngram_freqs.items(), key=lambda x: x[1], reverse=True)        
                top_grams = int(len(sorted_ngrams)/(self.top_ngrams_fraction))
                logger.debug(f"TOP GLOBAL {sorted_ngrams}")

                global_ngrams: List[Tuple[str, float]] = sorted_ngrams[:top_grams]
                
                logger.debug(f"Frecuencias globales ordenadas: {sorted_ngrams}")

                W = np.array(sorted_ngrams)
                Z = np.array(global_ngrams)
                logger.info(f"TOP GLOBAL: {nparray(Z)}, SORTED GLOBAL: {nparray(W)}")
                logger.debug(f"TOP GLOBAL: {global_ngrams}")

                global_filter: Dict[str, Any] = {
                    "global_counter": global_counter, 
                    "char_ngram_global": [self.gngr[0], self.gngr[1]],
                    "global_ngrams": global_ngrams
                }

            logger.info("GLOBAL FILTER generado")
            return global_filter

        except Exception as e:
            logger.error(f"Error entreando global: {e}", exc_info=True)

    def _train_map_vectorizar(self, key_words: Dict[str, List[str]]) -> Dict[str, Any]:
        try:
            all_mappers = {}
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

                    counter_map = CountVectorizer(
                        strip_accents="ascii",
                        ngram_range=(self.ngr[0], self.ngr[1]),
                        analyzer="char_wb",
                        binary=True,
                        vocabulary=list(vocabulary),
                    )
                    
                    tfidf_map = TfidfTransformer(
                        norm="l1", 
                        use_idf=False, 
                        smooth_idf=True, 
                        sublinear_tf=False
                    )

                    map_counts = counter_map.fit_transform(field_variants_normalized)
                    X_tfidfmap = tfidf_map.fit_transform(map_counts)
                    N_tfidfmap = np.array(X_tfidfmap.shape)

                    # logger.info(f"MAPPER: '{field}', features: {np.array(X_tfidfmap.shape)}")
                    # logger.info(f"Features fusionados: {np.array(X_tfidf.shape)}")

                    all_mappers[field] = {
                        "counter_map": counter_map,
                        "tfidf_map": tfidf_map,
                        "n_feat_maps": N_tfidfmap[1],
                    }

            # logger.info(f"N_features: {N_tfidf[1]}")
            # logger.debug(f"Features puros completos: {np.array(X_tfidf.size)}")
            # logger.debug(f"Features fusionados completos: {np.array(X_counts.shape)}")
            # logger.info("MAPPER generado")
            return all_mappers
        
        except Exception as e:
            logger.error(f"Error entrenando mapper: {e}", exc_info=True)
    
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