import os
import re
import logging
import pickle
import unicodedata
import json
import yaml
import time
from typing import List, Dict, Any, Tuple, Optional
from src.train_model import TrainModel

logger = logging.getLogger(__name__)

class ModelGenerator:
    def __init__(self, config_file: str, project_root: str, key_words_file: str):
        time0 = time.perf_counter()
        self.project_root = project_root
        self.config_file = config_file
        self.key_words_file = key_words_file
        logger.info(f"Iniciación en: {time.perf_counter() - time0} s")
            
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
            
    def generate_model(self, config_file: str, key_words_file: str) -> Optional[Dict[str, Any]]:
        """Lee YAML, normaliza variantes, precomputa n-gramas 2-5y guarda un pickle con toda la info necesaria para WordFinder."""
        time1 = time.perf_counter()
        self.config_dict: Dict[str, Any] = {}
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"No existe config: {self.config_file}")
            with open(config_file, "r", encoding="utf-8") as f:
                if config_file:
                    self.config_dict = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}", exc_info=True)
            return None 
        
        self.key_words: Dict[str, List[str]] = {}
        try:
            if not os.path.exists(self.key_words_file):
                raise FileNotFoundError(f"No existe config: {self.key_words_file}")
            with open(key_words_file, "r", encoding="utf-8") as f:
                if key_words_file:
                    self.key_words_dict = json.load(f)
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}", exc_info=True)
            return None
        
        self._train = TrainModel(config=self.config_dict, project_root=self.project_root)
        self.params: Dict[str, Any] = self.config_dict.get("params", {})
        self.ngr: Tuple[int, int] = self.params.get("char_ngram_range", [])
        self.gngr: Tuple[int, int]= self.params.get("char_ngram_global", [])
        self.top_ngrams = self.params.get("top_ngrams", [])
        self.top_ngrams_fraction = self.params.get("top_ngrams_fraction", [])
        key_words: Dict[str, List[Dict[str, str]]] = self.key_words_dict.get("key_words", {})
        noise_words = self.key_words_dict.get("noise_words", [])

        logger.info(f"Rango elegido: {self.ngr}")
        try:
            n_min, n_max = int(self.ngr[0]), int(self.ngr[1])
        except Exception as e:
            logger.info(f"Error en rango: {e}", exc_info=True)
                
        # Construir vocabulario normalizado
        global_words: List[str] = []
        variant_to_field: Dict[str, str] = {}
        
        for field, variants in key_words.items():
            if not variants:
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
                
        all_vectorizers, global_filter = self._train.train_all_vectorizers(key_words, global_words)
                                                
        grams_index: List[Dict[str, Any]] = []
        buckets_by_len: Dict[int, List[int]] = {}
        total_ngrams_all = 0 
            
        for i, w in enumerate(global_words):
            length = len(w)
            buckets_by_len.setdefault(length, []).append(i)

            gmap: Dict[int, List[str]] = {}
            for n in range(n_min, n_max + 1):
                ngrams = self._ngrams(w, n)
                gmap[n] = ngrams
                total_ngrams_all += len(ngrams)

            grams_index.append({
                "len": length,
                "grams": gmap
            })
            
        # Generar n-gramas para noise_words
        noise_grams_index: List[Dict[str, Any]] = []
        for noise_word in noise_words:
            length = len(noise_word)
            gmap: Dict[int, List[str]] = {}
            for n in range(n_min, n_max + 1):
                ngrams = self._ngrams(noise_word, n)
                gmap[n] = ngrams
            
            noise_grams_index.append({
                "len": length,
                "grams": gmap,
                "word": noise_word
            })

        model: Dict[str, Any] = {
            "params": self.params,
            "global_filter": global_filter,
            "all_vectorizers": all_vectorizers,
            "variant_to_field": variant_to_field,
            "global_words": global_words,
            "noise_words": noise_words,
            "noise_grams_index": noise_grams_index,
            "grams_index": grams_index,
            }
            
        logger.info(f"Entradas guardadas en el modelo: {list(model.keys())}")
        logger.info(f"Modelo generado en: {time.perf_counter() - time1} s")

        output_path = os.path.join(self.project_root, "data", "word_finder_model.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "wb") as f:
                pickle.dump(model, f)
                
            logger.info("Modelo (n-gramas binarios) guardado en: %s", output_path)
            return model
            
        except AttributeError as e:
            logger.info(f"Error costruyendo Modelo: {e}", exc_info=True)
        