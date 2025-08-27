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

    def generate_model(self) -> Dict[str, Any]:
        """
        Lee YAML, normaliza variantes, precomputa n-gramas 2-5
        y guarda un pickle con toda la info necesaria para WordFinder.
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"No existe config: {self.config_file}")
            
        with open(self.config_file, "r", encoding="utf-8") as f:
            if self.config_file:

                self.cfg: Dict[str, Any] = yaml.safe_load(f)
            
        self.params: Dict[str, Any] = self.cfg.get("params", {})
        # rango n-gramas
        ngr: Tuple[int, int] = self.params.get("char_ngram_range", [2, 5])
        try:
            n_min, n_max = int(ngr[0]), int(ngr[1])
        except Exception:
            n_min, n_max = 2, 5
        if n_min < 1 or n_max < n_min:
            n_min, n_max = 2, 5
        self.params["char_ngram_range"] = [n_min, n_max]

        key_fields: Dict[str, Any] = self.cfg.get("key_fields", {}) 
        descriptivo: List[str] = self.cfg.get("descriptivo", []) 
        cuantitativo: List[str] = self.cfg.get("cuantitativo", []) 
        identificador: List[str] = self.cfg.get("identificador", []) 

        # Construir vocabulario normalizado
        global_words: List[str] = []
        variant_to_field: Dict[str, str] = {}

        for field, variants in key_fields.items():
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
        for h in (descriptivo + cuantitativo + identificador):
            if not isinstance(h, str):
                continue
            s = self._normalize(h)
            if not s:
                continue
            header_words.append(s)

        # Unificar y deduplicar preservando orden
        combined_words: List[str] = list(dict.fromkeys(global_words + header_words))

        # Precomputar n-gramas y buckets por longitud
        grams_index: List[Dict[str, Any]] = []
        buckets_by_len: Dict[int, List[int]] = {}
        
        for i, w in enumerate(combined_words):
            length = len(w)
            buckets_by_len.setdefault(length, []).append(i)
            
            gmap: Dict[int, List[str]] = {}
            for n in range(n_min, n_max + 1):
                gmap[n] = self._ngrams(w, n)
                
            grams_index.append({
                "len": length,
                "grams": gmap  # dict n -> lista de n-gramas
            })

        model: Dict[str, Any] = {
            "params": self.params,
            "key_fields": key_fields,
            "global_words": global_words,
            "header_words": header_words,
            "variant_to_field": variant_to_field,
            "combined_words": combined_words,
            "grams_index": grams_index,
            "buckets_by_len": {int(k): v for k, v in buckets_by_len.items()},
            # campos legacy (compatibilidad): quedan como None
            "vectorizer": None,
            "Y_global": None,
            "Y_headers": None,
            "meta": {},
            "total_words": len(combined_words),
            "vocabulario_size": None,
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