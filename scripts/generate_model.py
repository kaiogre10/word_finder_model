import os
import re
import logging
import pickle
import unicodedata
import json
import yaml
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
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
        self.config_file = config_file
        self.config_dict: Dict[str, Any] = {}
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"No existe config: {self.config_file}")
            with open(self.config_file, "r", encoding="utf-8") as f:
                if self.config_file:
                    self.config_dict = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}", exc_info=True)
            return None 
        
        self.key_words_dict: Dict[str, List[str]] = {}
        self.key_words_file = key_words_file
        try:
            if not os.path.exists(self.key_words_file):
                raise FileNotFoundError(f"No existe config: {self.key_words_file}")
            with open(self.key_words_file, "r", encoding="utf-8") as f:
                if self.key_words_file:
                    self.key_words_dict = json.load(f)
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}", exc_info=True)
            return None
        
        self._train = TrainModel(config=self.config_dict, project_root=self.project_root)
        self.params: Dict[str, Any] = self.config_dict.get("params", {})
        key_words: Dict[str, List[str]] = self.key_words_dict.get("key_words", {})
        noise_words = self.key_words_dict.get("noise_words", [])

        # Construir vocabulario normalizado
        global_words: List[str] = []
        variant_to_field: Dict[str, str] = {}
        
        for field, variants in key_words.items():
            if not variants:
                continue
            if isinstance(variants, str):
                variants = [variants]
            if not isinstance(variants, (list, tuple)): # type: ignore
                continue
            for v in variants:
                if not isinstance(v, str): # type: ignore
                    continue
                s = self._normalize(v)
                if not s:
                    continue
                global_words.append(s)
                variant_to_field[s] = field
                
        global_filter, noise_filter = self._train.train_all_vectorizers(global_words, noise_words)

        now = datetime.now()
        model_time = now.isoformat()
                            
        model: Dict[str, Any] = {
            "params": self.params,
            "noise_filter": noise_filter,
            "global_filter": global_filter,
            "variant_to_field": variant_to_field,
            "noise_words": noise_words,
            "global_words": global_words,
            "model_time": model_time,
        }

        logger.info(f"Modelo generado en: {time.perf_counter()-time1}s")

        output_path = os.path.join(self.project_root, "models", "wf_model.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, "wb") as f:
                pickle.dump(model, f)
                
            logger.critical(f"Modelo 'WORD_FINDER' generado el {model_time} guardado en: %s", output_path)
            return model
            
        except AttributeError as e:
            logger.info(f"Error costruyendo Modelo: {e}", exc_info=True)
