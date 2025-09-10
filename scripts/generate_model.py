import os
import re
import logging
import pickle
import unicodedata
import yaml
from typing import List, Dict, Any, Tuple

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
        ngr: Tuple[int, int] = self.params.get("char_ngram_range", [2, 5])
        logger.info(f"Rango elegido: {ngr}")
        try:
            n_min, n_max = int(ngr[0]), int(ngr[1])
        except Exception:
            n_min, n_max = 2, 5
        if n_min < 1 or n_max < n_min:
            n_min, n_max = 2, 5
        self.params["char_ngram_range"] = [n_min, n_max]

        key_words: Dict[str, Any] = self.config.get("key_words", {}) 
        self.density_encoder: Dict[str, float] = self.config.get("density_encoder", {})
        header_words_config: Dict[str, List[str]] = self.config.get("header_words", {})
                
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

        header_words: List[str] = []
        variant_to_header_category: Dict[str, str] = {}
        
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
                norm_stats = {}
                # char_count y mean: normalización absoluta
                norm_stats["char_count_n"] = min(stats["char_count"] / MAX_CHAR_COUNT, 1.0)
                norm_stats["mean_n"] = stats["mean"] / MAX_MEAN
                # variance y std_dev: normalización por campo
                var_min, var_max = min(field_stats[field]["variance"]), max(field_stats[field]["variance"])
                std_min, std_max = min(field_stats[field]["std_dev"]), max(field_stats[field]["std_dev"])
                norm_stats["variance_n"] = (stats["variance"] - var_min) / (var_max - var_min) if var_max != var_min else 1.0
                norm_stats["std_dev_n"] = (stats["std_dev"] - std_min) / (std_max - std_min) if std_max != std_min else 1.0
                normalized_stats[word] = norm_stats
        
        for category, words_list in header_words_config.items():
            if not isinstance(words_list, list):
                continue
            for word in words_list:
                if not isinstance(word, str):
                    continue
                s: str = self._normalize(word)
                if not s:
                    continue
                header_words.append(s)
                variant_to_header_category[s] = category

        # Unificar y deduplicar preservando orden
        combined_words: List[str] = list(dict.fromkeys(global_words + header_words))

        # Precomputar n-gramas y buckets por longitud
        grams_index: List[Dict[str, Any]] = []
        buckets_by_len: Dict[int, List[int]] = {}

        # Para estadísticas
        ngram_stats: Dict[int, List[int]] = {n: [] for n in range(n_min, n_max + 1)}
        total_ngrams_all = 0  # Contador total de n-gramas

        for i, w in enumerate(combined_words):
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
                logger.debug(f"  Palabra #{idx+1} ({combined_words[idx]}): {count} n-gramas de tamaño {n}")

        logger.info(f"Cantidad total de palabras en combined_words: {len(combined_words)}")
        logger.info(f"Cantidad total de n-gramas generados (todos los tamaños): {total_ngrams_all}")

        model: Dict[str, Any] = {
            "params": self.params,
            "variant_to_header_category": variant_to_header_category,
            "variant_to_field": variant_to_field,
            "combined_words": combined_words,
            "grams_index": grams_index,
            "density_encoder": self.density_encoder,
            "field_stats": field_stats,
            "normalized_stats": normalized_stats,
        }

        output_path = os.path.join(self.project_root, "data", "word_finder_model.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(model, f)
            
        logger.info("Modelo (n-gramas binarios) guardado en: %s", output_path)
        return model

if __name__ == "__main__":
    # Uso de ejemplo:
    generator = ModelGenerator("data/config.yaml", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    generator.generate_model()