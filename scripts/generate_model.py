import os
import re
import numpy as np
import logging
import pickle
import unicodedata
import yaml
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
logger = logging.getLogger(__name__)

class ModelGenerator:
    def __init__(self, config_file: str, project_root: str):
        self.project_root = project_root
        self.config_file = config_file
            
    def _normalize(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[^a-zA-Z0-9]", "", s)  # quita espacios, signos y también _        
        return s

    def _ngrams(self, s: str, n: int) -> List[str]:
        if n <= 0 or not s:
            return []
        if len(s) < n:
            return []
        return [s[i:i+n] for i in range(len(s) - n + 1)]
        
    def calculate_stats(self, word: str) -> Dict[str, float]:
        chars = list(word)
        values = [self.density_encoder.get(c, 0) for c in chars]
        char_count = float(len(word))
        mean_val = sum(values) / char_count if char_count else 0.0
        variance = sum((v - mean_val) ** 2 for v in values) / char_count if char_count else 0.0
        std_dev = variance ** 0.5
        return {"char_count": char_count, "mean": mean_val, "variance": variance, "std_dev": std_dev}
        
    def _train_all_vectorizers(self, key_words: Dict[str, list[str]], global_words: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Entrena CountVectorizer + TfidfTransformer por field y filtro global."""
        all_vectorizers = {}

        # 1. Vectorizers por field (CountVectorizer + TfidfTransformer)
        for field, variants in key_words.items():
            if not variants:
                continue
            if isinstance(variants, str):
                variants = [variants]

            variants_normalized = [self._normalize(v) for v in variants if isinstance(v, str)]
            variants_normalized = [s for s in variants_normalized if s]
            if not variants_normalized:
                continue

            # construir vocabulario de n-gramas por field
            vocabulary = set()
            for variant in variants_normalized:
                for n in range(self.ngr[0], self.ngr[1] + 1):
                    vocabulary.update(self._ngrams(variant, n))
            if not vocabulary:
                continue

            # CountVectorizer con vocab fijo (genera matriz de conteos)
            counter = CountVectorizer(
                strip_accents="ascii",
                ngram_range=(self.ngr[0], self.ngr[1]),
                analyzer="char_wb",
                vocabulary=list(vocabulary),
            )
            X_counts = counter.transform(variants_normalized)

            # TfidfTransformer ajustado sobre los conteos.
            # use_idf=False produce TF normalizado; cambiar a True si tienes corpus amplio.
            tfidf_tr = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
            X_tfidf = tfidf_tr.fit_transform(X_counts)

            all_vectorizers[field] = {
                "counter": counter,
                "tfidf": tfidf_tr,
                "tfidf_vectors": X_tfidf,
                "variants": variants_normalized,
            }
            logger.info(
                f"Vectorizador por campo '{field}': counts={X_counts.shape}, tfidf_vectors={X_tfidf.shape}"
            )

        # 2. Filtro global (extraer idf/weights si se desea)
        global_vocab = set()
        for word in global_words:
            for n in range(self.ngr[0], self.ngr[1] + 1):
                global_vocab.update(self._ngrams(word, n))

        if global_vocab:
            global_counter = CountVectorizer(
                strip_accents="ascii",
                ngram_range=(self.ngr[0], self.ngr[1]),
                analyzer="char_wb",
                vocabulary=list(global_vocab),
            )
            Xg_counts = global_counter.transform(global_words)
            logger.info("Global counts shape: %s (n_docs=%d, n_features=%d)", Xg_counts.shape, Xg_counts.shape[0], Xg_counts.shape[1])

            # ajustar TF-IDF sobre esos conteos (no modifica Xg_counts)
            global_tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
            global_tfidf.fit(Xg_counts)

            # loguear información relevante del tfidf
            logger.info("Global TF-IDF fitted: idf_ shape=%s, sample idf[:5]=%s", global_tfidf.idf_.shape, global_tfidf.idf_[:5])

            ngram_weights = {ngram: global_tfidf.idf_[idx] for ngram, idx in global_counter.vocabulary_.items()}
        else:
            ngram_weights = {}

        global_filter = {"ngram_weights": ngram_weights, "char_ngram_range": [self.ngr[0], self.ngr[1]]}
        return all_vectorizers, global_filter
            
    def generate_model(self) -> Dict[str, Any]:
        """
        Lee YAML, normaliza variantes, precomputa n-gramas 2-5
        y guarda un pickle con toda la info necesaria para WordFinder.
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"No existe config: {self.config_file}")
            
        with open(self.config_file, "r", encoding="utf-8") as f:
            if self.config_file:
                self.config: Dict[str, Any] = yaml.safe_load(f)
            
        self.params: Dict[str, Any] = self.config.get("params", {})
        # rango n-gramas
        self.ngr: Tuple[int, int] = tuple(self.params.get("char_ngram_range", [2, 5]))
        logger.info(f"Rango elegido: {self.ngr}")
        try:
            n_min, n_max = int(self.ngr[0]), int(self.ngr[1])
        except Exception:
            n_min, n_max = 2, 5
        if n_min < 1 or n_max < n_min:
            n_min, n_max = 2, 5
        self.params["char_ngram_range"] = [n_min, n_max]

        key_words: Dict[str, list[str]] | str = self.config.get("key_words", {}) 
        self.density_encoder: Dict[str, float] = self.config.get("density_encoder", {})
                
        # Construir vocabulario normalizado
        global_words: List[str] = []
        variant_to_field: Dict[str, str] = {}
        
        for field, variants in key_words.items():
            if variants is None:
                continue
            if isinstance(variants, str):
                variants = [variants]
            if not isinstance(variants, (list, tuple)):
                continue
            for v in variants:
                if not isinstance(v, str):
                    continue
                s = self._normalize(v)
                if not s:
                    continue
                global_words.append(s)
                variant_to_field[s] = field
                
            variants_normalized = [self._normalize(v) for v in variants if isinstance(v, str)]
            variants_normalized = [s for s in variants_normalized if s]
        # Líneas 118-120 reemplazar por:
        all_vectorizers, global_filter = self._train_all_vectorizers(key_words, global_words)
                        
        # Calcular estadísticas por campo
        field_stats: Dict[str, Dict[str, List[float]]] = {}
        word_field_stats: Dict[str, Dict[str, float]] = {}

        # Para key_words
        for field, variants in key_words.items():
            field_stats[field] = {"char_count": [], "mean": [], "variance": [], "std_dev": []}
            if variants:
                for v in variants:
                    s = self._normalize(v)
                    if s and s in variant_to_field:
                        stats = self.calculate_stats(s)
                        word_field_stats[s] = stats
                        for stat_name, stat_val in stats.items():
                            field_stats[field][stat_name].append(stat_val)

        # Normalizar estadísticas por campo
        MAX_CHAR_COUNT = 20.0
        MAX_MEAN = 113.0
        normalized_stats: Dict[str, Dict[str, float]] = {}

        for word, stats in word_field_stats.items():
            field = variant_to_field.get(word)
            if field and field in field_stats:
                norm_stats: Dict[str, float] = {}
                norm_stats["char_count_n"] = min(stats["char_count"] / MAX_CHAR_COUNT, 20.0)
                norm_stats["mean_n"] = stats["mean"] / MAX_MEAN
                var_min, var_max = min(field_stats[field]["variance"]), max(field_stats[field]["variance"])
                std_min, std_max = min(field_stats[field]["std_dev"]), max(field_stats[field]["std_dev"])
                norm_stats["variance_n"] = (stats["variance"] - var_min) / (var_max - var_min) if var_max != var_min else 1.0
                norm_stats["std_dev_n"] = (stats["std_dev"] - std_min) / (std_max - std_min) if std_max != std_min else 1.0
                normalized_stats[word] = norm_stats
        
        # Unificar y deduplicar preservando orden
        main_words: List[str] = global_words

        # Precomputar n-gramas y buckets por longitud
        grams_index: List[Dict[str, Any]] = []
        buckets_by_len: Dict[int, List[int]] = {}

        # Para estadísticas
        ngram_stats: Dict[int, List[int]] = {n: [] for n in range(n_min, n_max + 1)}
        total_ngrams_all = 0  # Contador total de n-gramas

        for i, w in enumerate(main_words):
            length = len(w)
            buckets_by_len.setdefault(length, []).append(i)

            gmap: Dict[int, List[str]] = {}
            for n in range(n_min, n_max + 1):
                ngrams = self._ngrams(w, n)
                gmap[n] = ngrams
                ngram_stats[n].append(len(ngrams))  
                total_ngrams_all += len(ngrams)

            grams_index.append({
                "len": length,
                "grams": gmap
            })

        # Mostrar estadísticas de n-gramas por palabra y por n
        for n in range(n_min, n_max + 1):
            total_ngrams = sum(ngram_stats[n])
            logger.info(f"n={n}: total de n-gramas generados={total_ngrams}, palabras={len(ngram_stats[n])}, promedio por palabra={total_ngrams/len(ngram_stats[n]) if ngram_stats[n] else 0:.2f}")
            for idx, count in enumerate(ngram_stats[n]):
                logger.debug(f"Palabra #{idx+1} ({main_words[idx]}): {count} n-gramas de tamaño {n}")

        logger.info(f"Cantidad total de palabras en key_words: {len(main_words)}")
        logger.info(f"Cantidad total de n-gramas generados (todos los tamaños): {total_ngrams_all}")

        model: Dict[str, Any] = {
            "params": self.params,
            "global_filter": global_filter,
            "all_vectorizers": all_vectorizers,
            "variant_to_field": variant_to_field,
            "key_words": main_words,
            "grams_index": grams_index,
            "density_encoder": self.density_encoder,
            "field_stats": field_stats,
            "normalized_stats": normalized_stats,
        }

        output_path = os.path.join(self.project_root, "data", "word_finder_model.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "wb") as f:
                pickle.dump(model, f)
                
            logger.info("Modelo (n-gramas binarios) guardado en: %s", output_path)
            return model
            
        except AttributeError as e:
            logger.info(f"Error costruyendo Modelo: {e}", exc_info=True)
        
if __name__ == "__main__":
    generator = ModelGenerator("data/config.yaml", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    generator.generate_model()