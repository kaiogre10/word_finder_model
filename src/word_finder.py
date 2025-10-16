import os
import logging
import pickle
from typing import List, Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path: str, project_root: str):
        self.model: Dict[str, Any] = self._load_model(model_path)
        self.project_root = project_root
        self._apply_active_model()
        self.global_words: List[str] = self.model.get("global_words", [])
        self.noise_filter = self.model.get("noise_filter", {})
        self.global_filter = self.model.get("global_filter", {})
        self.inverted_grams_index = self.model.get("inverted_grams_index", {})
        self.all_vectorizers = self.model.get("all_vectorizers", {})
        self.variant_to_field = self.model.get("variant_to_field", {})
        self.noise_words = self.model.get("noise_words", [])

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            if not isinstance(self.model, dict):
                raise ValueError("El pickle no tiene el formato esperado (dict).")
            return self.model
        except Exception as e:
            logger.error(f"Error al cargar el modelo {e}", exc_info=True)
            raise

    def _apply_active_model(self):
        try:
            self.params: Dict[str, Any] = self.model.get("params", {})
        except Exception as e:
            logger.warning(f"Error en APPLY_MODEL: {e}", exc_info=True)
            return None

    def find_keywords(self, text: List[str] | str, threshold: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
        try:
            self.threshold: float = self.params.get("threshold_similarity")
            self.gngr: Tuple[int, int] = self.params.get("char_ngram_global", [])
            self.ngr: Tuple[int, int] = self.params.get("char_ngram_range", [])
            self.thresholds_by_len: List[Tuple[int, int, float]] = [tuple(item) for item in self.params.get("thresholds_by_len", [])]
            self.weights_by_n: List[Tuple[int, int, float]] = [tuple(item) for item in self.params.get("weights_by_n", [])]
            self.window_flex = self.params.get("window_flexibility", 3)
            self.global_filter_threshold = float(self.params.get("global_filter_threshold", 0.0))
            
            self.global_ngrams: List[Tuple[str, float]] = self.global_filter.get("global_ngrams", [])
            self.noisy_grams: List[Tuple[str, float]] = self.global_filter.get("noisy_grams", [])
            self.global_ngrams: List[Tuple[str, float]] = self.global_filter.get("global_ngrams", [])
            
            self.global_ngrams = (
                dict(self.global_filter.get("global_ngrams", []))
                if isinstance(self.global_filter.get("global_ngrams"), list)
                else self.global_filter.get("global_ngrams", {})
            )

            final_threshold = threshold if threshold is not None else self.threshold

            single = False
            if isinstance(text, str):
                text = [text]
                single = True

            results: List[Dict[str, Any]] = []
            for q in text:
                if not q:
                    continue

                if self._is_forbidden_text(q, final_threshold):
                    continue

                if not self._is_potential_keyword(q):
                    continue

                candidate_words = self._get_candidates_from_inverted_index(q)
                if not candidate_words:
                    continue

                for cand in candidate_words:
                    cand_len = len(cand)
                    min_w = max(1, cand_len - self.window_flex)
                    max_w = min(len(q), cand_len + self.window_flex)

                    grams_cand = {n: set(self._ngrams(cand, n)) for n in range(self.ngr[0], self.ngr[1] + 1)}

                    for w in range(min_w, max_w + 1):
                        if w > len(q):
                            break
                        for j in range(0, len(q) - w + 1):
                            sub = q[j:j + w]
                            
                            if sub == cand and len(sub) == cand_len:
                                ngram_score = 1.0
                            else:
                                grams_sub = self._build_query_grams(sub)
                                ngram_score = self._score_binary_cosine_multi_n(grams_cand, grams_sub)
                                len_ratio = max(len(sub), cand_len) / max(1, min(len(sub), cand_len))
                                if len_ratio >= 2.0:
                                    penalizacion = min(len(sub), cand_len) / max(len(sub), cand_len)
                                    ngram_score *= penalizacion

                            if ngram_score >= final_threshold:
                                if self._check_if_text_is_forbidden(sub, final_threshold):
                                    continue
                                
                                key_field = self.variant_to_field.get(cand)
                                results.append({
                                    "key_field": key_field,
                                    "word_found": cand,
                                    "similarity": float(ngram_score),
                                    "query": q
                                })
            if single:
                return results if results else []
            return results

        except Exception as e:
            logger.error(f"Error en find_keywords (nuevo flujo): {e}", exc_info=True)
            return None

    def _build_query_grams(self, q: str) -> Dict[int, set[str]]:
        gq: Dict[int, set[str]] = {}
        for n in range(self.ngr[0], self.ngr[1] + 1):
            gq[n] = set(self._ngrams(q, n))
        return gq

    def _ngrams(self, q: str, n: int) -> List[str]:
        try:
            if n <= 0 or not q or len(q) < n:
                return []
            return [q[i:i+n] for i in range(len(q) - n + 1)]
        except Exception as e:
            logger.error(f"Error construyendo n-gramas: {e}", exc_info=True)
            return []

    def _ngram_similarity(self, a: str, b: str) -> float:
        try:
            matches = float(sum(1 for x, y in zip(a, b) if x == y))
            return matches / float(max(len(a), len(b)))
        except Exception as e:
            logger.warning(f"Error comprobando texto: {e}", exc_info=True)
            return 0.0

    def _binary_cosine(self, size_a: int, size_b: int, soft_intersection: float) -> float:
        if size_a == 0 or size_b == 0:
            return 0.0
        return soft_intersection / float((size_a * size_b) ** 0.5)

    def _score_binary_cosine_multi_n(self, grams_a: Dict[int, set[str]], grams_b: Dict[int, set[str]]) -> float:
        try: 
            num = 0.0
            den = 0.0

            for n in range(self.ngr[0], self.ngr[1] + 1):
                A = grams_a.get(n, set())
                B = grams_b.get(n, set())
                w: float = self._get_weight_by_n(n)

                if not A or not B:
                    den += w
                    continue

                soft_intersection = 0.0
                for gram_a in A:
                    max_sim = 0.0
                    for gram_b in B:
                        sim = self._ngram_similarity(gram_a, gram_b)
                        if sim > max_sim:
                            max_sim = sim
                    soft_intersection += max_sim

                num += w * self._binary_cosine(len(A), len(B), soft_intersection)
                den += w

            if den <= 0.0:
                return 0.0
            return num / den
            
        except Exception as e:
            logger.error(f"Error calculando coseno suave: {e}" , exc_info=True)
            return 0.0
    
    def _get_weight_by_n(self, n: int, default: float = 1.0) -> float:
        for start, end, value in self.weights_by_n:
            if start <= n <= end:
                return value
        return default
        
    def _is_potential_keyword(self, q: str) -> bool:
        try:
            if not q:
                return False
            grams: Set[str] = set()
            for n in range(self.gngr[0], self.gngr[1] + 1):
                grams.update(self._ngrams(q, n))
            if not grams:
                return False

            total = 0.0
            weights_sum = 0.0
            for g in grams:
                freq = self.global_ngrams.get(g, 0.0)
                if freq > 0:
                    total += freq * freq
                    weights_sum += freq

            if weights_sum == 0:
                return False
            
            score = total / weights_sum
            return score >= self.global_filter_threshold
        except Exception as e:
            logger.error("Error en filtro global: %s", e, exc_info=True)
            return False

    def _check_if_text_is_forbidden(self, text_to_check: str, threshold: float) -> bool:
        try:
            if not text_to_check:
                return False
                
            for noise_word in self.noise_words:
                if not noise_word:
                    continue
                
                noise_len = len(noise_word)
                grams_noise = {n: set(self._ngrams(noise_word, n)) for n in range(self.ngr[0], self.ngr[1] + 1)}
                grams_sub = self._build_query_grams(text_to_check)
                
                if text_to_check == noise_word and len(text_to_check) == noise_len:
                    return True
                
                similarity = self._score_binary_cosine_multi_n(grams_noise, grams_sub)
                len_ratio = max(len(text_to_check), noise_len) / max(1, min(len(text_to_check), noise_len))
                if len_ratio >= 2.0:
                    similarity *= min(len(text_to_check), noise_len) / max(len(text_to_check), noise_len)
                
                if similarity >= threshold:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error verificando si el texto es prohibido: {e}", exc_info=True)
            return False
            
    def _get_candidates_from_inverted_index(self, q: str) -> Set[str]:
        if not hasattr(self, "inverted_grams_index") or not self.inverted_grams_index:
            return set(self.global_words)

        grams_q: Set[str] = set()
        for n in range(self.ngr[0], self.ngr[1] + 1):
            grams_q.update(self._ngrams(q, n))

        candidates: Set[str] = set()
        for g in grams_q:
            words_with_g = self.inverted_grams_index.get(g)
            if words_with_g:
                candidates.update(words_with_g)
        return candidates
        
    def _is_forbidden_text(self, q: str, threshold: float) -> bool:
        try:
            if not q or not self.noise_words:
                return False

            if hasattr(self, "inverted_noise_grams_index") and self.inverted_noise_grams_index:
                grams_q: Set[str] = set()
                for n in range(self.ngr[0], self.ngr[1] + 1):
                    grams_q.update(self._ngrams(q, n))
                hits = 0
                for g in grams_q:
                    if g in self.inverted_noise_grams_index:
                        hits += 1
                        if hits >= 2:
                            return True

            for noise_word in self.noise_words:
                if not noise_word:
                    continue
                noise_len = len(noise_word)
                min_w = max(1, noise_len - self.window_flex)
                max_w = min(len(q), noise_len + self.window_flex)

                grams_noise = {n: set(self._ngrams(noise_word, n)) for n in range(self.ngr[0], self.ngr[1] + 1)}

                for w in range(min_w, max_w + 1):
                    if w > len(q):
                        break
                    for j in range(0, len(q) - w + 1):
                        sub = q[j:j + w]
                        if sub == noise_word and len(sub) == noise_len:
                            return True
                        grams_sub = self._build_query_grams(sub)
                        similarity = self._score_binary_cosine_multi_n(grams_noise, grams_sub)
                        len_ratio = max(len(sub), noise_len) / max(1, min(len(sub), noise_len))
                        if len_ratio >= 2.0:
                            similarity *= min(len(sub), noise_len) / max(len(sub), noise_len)
                        if similarity >= threshold:
                            return True
            return False
        except Exception as e:
            logger.error(f"Error en _is_forbidden_text: {e}", exc_info=True)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return{
        "total_words": len(self.global_words),
        "threshold_similarity": self.params.get("threshold_similarity"),
        "char_ngram_range": self.params.get("char_ngram_range"),
        "weights_by_n": self.params.get("weights_by_n"),
        "noise_words": self.noise_words,
        "noise_filter": self.noise_filter,
    }