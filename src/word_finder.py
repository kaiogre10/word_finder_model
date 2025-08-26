import os
import re
import logging
import pickle
from typing import List, Any, Dict, Literal, Optional
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.vectorizer = self.model.get("vectorizer")
        self.Y_global = self.model.get("Y_global")
        self.global_words = self.model.get("global_words", [])
        self.variant_to_field = {k.lower(): v for k, v in self.model.get("variant_to_field", {}).items()} if self.model.get("variant_to_field") else {}
        self.threshold = float(self.model.get("params", {}).get("threshold_similarity", 0.75))

    def _normalize(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip().lower()
        # quitar puntuación redundante pero mantener espacios
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        vect = model.get("vectorizer")
        if vect is None:
            raise KeyError("El pickle no contiene 'vectorizer'")

        # aplicar transformer_weights si están guardados (reconstruir FeatureUnion)
        weights = model.get("meta", {}).get("transformer_weights")
        if weights and hasattr(vect, "transformer_list"):
            try:
                from sklearn.pipeline import FeatureUnion
                transformer_list = getattr(vect, "transformer_list")
                vect = FeatureUnion(transformer_list, transformer_weights=weights)
                model["vectorizer"] = vect
                logger.info("Vectorizer reconstruido con transformer_weights: %s", weights)
            except Exception:
                logger.exception("No se pudo aplicar transformer_weights; usando vectorizer original")

        # Normalizar y asegurar listas y matrices precomputadas
        key_fields = model.get("key_fields", {})
        if "global_words" not in model or not model.get("global_words"):
            gw = []
            for fld, variants in key_fields.items():
                if isinstance(variants, str):
                    variants = [variants]
                for v in variants or []:
                    if isinstance(v, str) and v.strip():
                        gw.append(v.strip())
            model["global_words"] = gw

        # precompute Y_global si falta
        try:
            if model.get("global_words") and model.get("Y_global") is None:
                model["Y_global"] = vect.transform(model["global_words"])
        except Exception:
            logger.exception("No se pudo transformar global_words con el vectorizer; Y_global queda None")
            model["Y_global"] = model.get("Y_global", None)

        # garantizar threshold en params
        params = model.get("params", {})
        params.setdefault("threshold_similarity", params.get("threshold_similarity", 0.75))
        model["params"] = params

        # mapping variantes normalizado
        if "variant_to_field" not in model:
            vt = {}
            for fld, variants in key_fields.items():
                if isinstance(variants, str):
                    variants = [variants]
                for v in variants or []:
                    if isinstance(v, str) and v.strip():
                        vt[v.strip().lower()] = fld
            model["variant_to_field"] = vt

        return model

    def find_keywords(self, text: str | List[str], search_type: Literal["global", "headers"] = "global") -> List[Dict[str, Any]]:
        """
        Busca coincidencias. No lanza excepciones por input desconocido:
        - si vectorizer falla o no hay matches por encima del threshold, devuelve [].
        - acepta string único o lista de strings.
        """
        if not text:
            return []

        single = False
        if isinstance(text, str):
            text = [text]
            single = True

        # normalizar inputs
        queries = [self._normalize(s) for s in text]

        # rechazar queries vacías
        queries = [q for q in queries if q]
        if not queries:
            return []

        if self.vectorizer is None or self.Y_global is None:
            logger.warning("Vectorizer o Y_global no disponibles; no se realizan búsquedas.")
            return []

        try:
            Xq: spmatrix = self.vectorizer.transform(queries)
        except Exception:
            logger.exception("Error transformando la consulta con vectorizer; devolviendo lista vacía.")
            return []

        try:
            sims = cosine_similarity(Xq, self.Y_global)
        except Exception:
            logger.exception("Error calculando similitudes; devolviendo lista vacía.")
            return []

        results: List[Dict[str, Any]] = []
        for qi, row in enumerate(sims):
            best_idx = int(row.argmax())
            best_score = float(row[best_idx])
            if best_score >= self.threshold:
                candidate = self.global_words[best_idx]
                results.append({
                    "field": self.variant_to_field.get(candidate.lower()),
                    "word_found": candidate,
                    "similarity": best_score,
                    "query": queries[qi]
                })
            # si no supera umbral, no incluir (es aceptable no encontrar)
        return results if not single else (results[0:1] if results else [])

    def get_model_info(self):
        """Retorna información del modelo cargado"""
        return {
            'total_words': self.model['total_words'],
            'vocabulario_size': self.model['vocabulario_size'],
            'threshold_similarity': self.threshold,
            'global_words': len(self.model.get('global_words', [])),
            'header_words': len(self.model.get('header_words', [])),
            'campos_disponibles': list(self.model.get('key_fields', {}).keys())
        }