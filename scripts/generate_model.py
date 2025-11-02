import os
import logging
import pickle
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
        self.config_dict = self._load_params(config_file=self.config_file)
        self.key_words_dict = self._load_keywords(self.key_words_file)
        logger.info(f"Iniciación en: {time.perf_counter() - time0}s")
        self.generate_model()

    def _load_params(self, config_file: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"No existe config: {config_file}")
            with open(config_file, "r", encoding="utf-8") as f:
                if config_file:
                    config_dict = yaml.safe_load(f)
                    return config_dict
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}", exc_info=True)
        return {}
        
    def _load_keywords(self, key_words_file: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(key_words_file):
                raise FileNotFoundError(f"No existe config: {key_words_file}")
            with open(key_words_file, "r", encoding="utf-8") as f:
                if key_words_file:
                    key_words_dict = json.load(f)
                    return key_words_dict
                
        except Exception as e:
            logger.error(f"Error cargando el modelo: {e}", exc_info=True)
        return {}
            
    def generate_model(self) -> Optional[Dict[str, Any]]:
        """Lee YAML, normaliza variantes, precomputa n-gramas y guarda un pickle con toda la info necesaria para WordFinder."""
        time1 = time.perf_counter()
        key_words: Dict[str, Dict[str, List[str]]] = self.key_words_dict.get("key_words", {})
        noise_words: List[str] = self.key_words_dict["noise_words"]
        params: Dict[str, Any] = self.config_dict.get("params", {})
        self._train = TrainModel(config=params, project_root=self.project_root)
                
        global_filter, noise_filter, global_words, variant_to_field = self._train.train_all_vectorizers(key_words, noise_words)

        now = datetime.now()
        model_time = now.isoformat()
                            
        model: Dict[str, Any] = {
            "params": params,
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
