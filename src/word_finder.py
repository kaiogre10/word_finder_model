import os
import logging
import numpy as np
import pickle
import re
import unicodedata
import time
from typing import List, Any, Dict, Tuple, Set

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path: str, set_params: bool):
        model: Dict[str, Any] = self._load_model(model_path)
        if set_params:
            """Aquí va una función que obtendría los parametros de configuración del master_config
            pero me da flojera escribirla así que solo dejaré un log y no cambiaré el parametro "set_params"""
            logger.info(f"Parametros establecidos y cargados manualmente")

        params: Dict[str, Any] = model.get("params", {})
        noise_filter: Dict[str, Any] = model.get("noise_filter", {})
        global_filter = model.get("global_filter", {})

        self.all_ngrams: Dict[str, Tuple[int, Dict[int, List[str]]]] = model.get("all_ngrams", {})
        self.global_words: List[str] = model["global_words"]
        self.noise_words: Set[str] = set(model["noise_words"])
        self.global_filter_threshold: float = params.get("global_filter_threshold", {})
        self.noise_grams: List[Dict[int, List[str]]] = noise_filter["noise_grams"]
        self.noise_array: List[np.ndarray[Any, np.dtype[np.uint8]]] = noise_filter["noise_array"]
        self.threshold: float = params.get("threshold_similarity", {})
        self.ngrams: Tuple[int, int] = params["char_ngrams"]
        self.window_flex: int = params.get("window_flexibility", {})
        self.forb_match: float = params.get("forb_match", {})
        self.min_diff: float = params.get("min_diff", {})
        self.global_matrices: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = global_filter.get("global_matrices", {})

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
            with open(model_path, "rb") as f:
                self.model: Dict[str, Any] = pickle.load(f)
            if not isinstance(self.model, dict):  # type: ignore
                raise ValueError("El pickle no tiene el formato esperado (dict).")
            logger.info(f"Modelo cargado en: '{time.perf_counter() - t0:.6f}s'")
            return self.model
        except Exception as e:
            logger.error(f"Error al cargar el modelo {e}", exc_info=True)
            raise

    def find_keywords(self, text: List[str] | str) -> List[Dict[str, Any]]:
        try:
            if not text:
                return []

            single = False
            if isinstance(text, str):
                # Si es un string, lo dividimos en "palabras" respetando espacios y símbolos
                words = re.findall(r"[\w']+|[.,!?;]", text)
                queue = words
                single = True
            else:
                queue = list(text)

            results: List[Dict[str, Any]] = []
            
            # Procesamos cada palabra o token de la cola
            for s in queue:
                if not s:
                    continue

                q = self._normalize(s)
                
                # Si la palabra normalizada está vacía (ej. un punto), la tratamos como no-keyword
                if not q:
                    results.append({
                        "text": s,
                        "norm_text": "",
                        "key_field": None,
                        "key_word": "",
                        "similarity": 0.0,
                        "start": 0,
                        "end": len(s)
                    })
                    continue

                # --- Lógica de búsqueda de keywords existente ---
                if q in self.noise_words:
                    logger.debug(f"Ruido temprano: '{s}'")
                    # Aunque sea ruido, la incluimos en el output como no-keyword
                    results.append({
                        "text": s,
                        "norm_text": q,
                        "key_field": None,
                        "key_word": "noise",
                        "similarity": 0.0,
                        "start": 0,
                        "end": len(s)
                    })
                    continue

                if not self._is_potential_keyword(q):
                    results.append({
                        "text": s,
                        "norm_text": q,
                        "key_field": None,
                        "key_word": "",
                        "similarity": 0.0,
                        "start": 0,
                        "end": len(s)
                    })
                    continue
                
                q_cleaned, _ = self._remove_noise_substrings(q)
                if q_cleaned != q:
                    q = q_cleaned

                if self.check_full_word(text=q, place="noise"):
                    if not q:
                        continue
                
                found_matches_for_s: List[Dict[str, Any]] = []
                q_grams_idx: Dict[str, List[int]] = {}
                for n in range(self.ngrams[0], self.ngrams[1] + 1):
                    for idx, gram in enumerate(self._ngrams(q, n)):
                        q_grams_idx.setdefault(gram, []).append(idx)

                for cand, (key_field, grams_cand) in self.all_ngrams.items():
                    cand_len = len(cand)
                    hit_positions: List[int] = []
                    for n, grams in grams_cand.items():
                        for g in grams:
                            if g in q_grams_idx:
                                hit_positions.extend(q_grams_idx[g])

                    if not hit_positions:
                        continue

                    best_score_for_cand: float = 0.1
                    best_sub_details: Dict[str, int] = {}
                    sorted_unique_hits = sorted(list(set(hit_positions)))
                    min_w = max(1, cand_len - self.window_flex)
                    max_w = cand_len + self.window_flex

                    for hit_start_pos in sorted_unique_hits:
                        for w in range(min_w, max_w + 1):
                            for offset in range(-self.window_flex, 1):
                                start = hit_start_pos + offset
                                end = start + w
                                if start < 0 or end > len(q):
                                    continue
                                sub = q[start:end]
                                if not sub:
                                    continue
                                if sub == cand:
                                    final_score = 1.0 * self._length_penalty(sub, cand)
                                else:
                                    grams_sub = self._build_query_grams(sub)
                                    final_score = self._score_hybrid_greedy(grams_cand, grams_sub)
                                    final_score *= self._length_penalty(sub, cand)
                                if final_score > best_score_for_cand:
                                    best_score_for_cand = final_score
                                    best_sub_details = {"start": start, "end": end}

                    if best_score_for_cand > self.threshold:
                        found_matches_for_s.append({
                            "key_field": key_field,
                            "key_word": cand,
                            "similarity": best_score_for_cand,
                            "text": s,
                            "norm_text": q,
                            "start": best_sub_details.get("start", 0),
                            "end": best_sub_details.get("end", len(s))
                        })
                
                if found_matches_for_s:
                    # Usamos la lógica de desambiguación para encontrar el mejor match para la palabra 's'
                    best_matches = self._resolve_ambiguity_by_full_word(found_matches_for_s)
                    if best_matches:
                        # Tomamos el mejor y lo añadimos a los resultados
                        best_match = best_matches[0]
                        # Nos aseguramos que el texto original 's' esté en el resultado
                        best_match["text"] = s
                        best_match["norm_text"] = q
                        results.append(best_match)
                    else:
                        # Si después de resolver la ambigüedad no queda nada, es un no-match
                        results.append({
                            "text": s, "norm_text": q, "key_field": None, "key_word": "", 
                            "similarity": 0.0, "start": 0, "end": len(s)
                        })
                else:
                    # Si no hubo ningún match potencial, es un no-match
                    results.append({
                        "text": s, "norm_text": q, "key_field": None, "key_word": "", 
                        "similarity": 0.0, "start": 0, "end": len(s)
                    })

            # Re-calculamos start y end para que sean consecutivos en el output final
            current_pos = 0
            for res in results:
                text_len = len(res["text"])
                res["start"] = current_pos
                res["end"] = current_pos + text_len
                # Añadimos un espacio para el siguiente token, simulando el texto original
                current_pos += text_len + 1 

            if single:
                logger.info(f"RESULTS: {results}")
            return results

        except Exception as e:
            logger.error(f"Error buscando palabras clave: '{e}'", exc_info=True)
            return []

    def _resolve_ambiguity_by_full_word(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not matches:
            return []

        if len(matches) == 1:
            return matches

        for i, match in enumerate(matches):
            norm_ocr_text = match['norm_ocr_text']
            word_found = match['key_word']
            grams_text = self._build_query_grams(norm_ocr_text)

            if word_found in self.all_ngrams:
                _, grams_word = self.all_ngrams[word_found]
            else:
                grams_word = self._build_query_grams(word_found)

            # Calcular similitud base
            base_similarity = self._score_hybrid_greedy(grams_word, grams_text)

            # Penalización simétrica: min/max siempre da un valor entre 0 y 1 no importa cuál sea más largo, el resultado es el mismo
            length_penalty = self._length_penalty(norm_ocr_text, word_found)

            # Score final = similitud base * penalización por longitud
            match['score_final'] = base_similarity * length_penalty

            logger.debug(
                "EMPATE: Match #%d: campo: %s, palabra: '%s' | score de desempate: %.6f | texto: '%s'",
                i, match.get("key_field"), word_found, match['score_final'], norm_ocr_text
            )

        # Encontrar el mejor match usando max() en lugar de sort()
        best_match = max(matches, key=lambda x: (x['score_final'], len(x['key_word'])))

        logger.debug(
            "DESEMPATE: texto '%s': campo: %s, palabra: '%s', score_final: %.6f",
            best_match.get("text"), best_match.get("key_field"), best_match.get("key_word"),
            best_match.get("score_final")
        )
        return [best_match]

    def _build_query_grams(self, q: str) -> Dict[int, List[str]]:
        """Construye n-gramas de la consulta retornando LISTAS (Duplicados permitidos)"""
        gq: Dict[int, List[str]] = {}
        for n in range(self.ngrams[0], self.ngrams[1] + 1):
            gq[n] = self._ngrams(q, n)
        return gq

    def _ngrams(self, q: str, n: int) -> List[str]:
        try:
            if n <= 0 or not q:
                return []
            if len(q) < n:
                return []
            return [q[i:i + n] for i in range(len(q) - n + 1)]
        except Exception as e:
            logger.error(f"Error construyendo n-gramas: {e}", exc_info=True)
            return []

    def _ngram_similarity(self, a: str, b: str) -> float:
        """Calcula la similitud entre dos n-gramas."""
        if not a or not b: return 0.0
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / float(max(len(a), len(b)))

    def _score_hybrid_greedy(self, grams_cand: Dict[int, List[str]], grams_sub: Dict[int, List[str]]) -> float:
        """
        Calcula similitud híbrida "Greedy Unique Match" usando listas.
        No usa pesos por longitud de n-grama.
        """
        total_score = 0.0
        total_ngrams_cand = 0.0

        for n, cand_list in grams_cand.items():
            if not cand_list:
                continue

            num_cand = len(cand_list)
            total_ngrams_cand += num_cand

            sub_list = grams_sub.get(n, [])
            if not sub_list:
                continue

            # 1. Calcular todas las similitudes cruzadas posibles > 0
            possible_matches: List[Tuple[float, int, int]] = []
            for i, gc in enumerate(cand_list):
                for j, gs in enumerate(sub_list):
                    # gc y gs tienen garantizado tener la misma longitud 'n' aquí
                    if gc == gs:
                        sim = 1.0
                    else:
                        sim = self._ngram_similarity(gc, gs)
                    # Penalización simétrica
                    sim *= self._length_penalty(gc, gs)

                    if sim > 0.0:
                        possible_matches.append((sim, i, j))

            # 2. Ordenar por score descendente (voraz)
            possible_matches.sort(key=lambda x: x[0], reverse=True)

            # 3. Asignar asegurando unicidad de índices
            used_cand: Set[int] = set()
            used_sub: Set[int] = set()
            section_score = 0.0

            for score, i, j in possible_matches:
                if i not in used_cand and j not in used_sub:
                    section_score += score
                    used_cand.add(i)
                    used_sub.add(j)
                    if len(used_cand) == num_cand:
                        break

            total_score += section_score

        if total_ngrams_cand == 0.0:
            return 0.0

        return total_score / total_ngrams_cand

    def _is_potential_keyword(self, q: str) -> bool:
        try:
            if not q:
                return False

            if self.check_full_word(text=q, place="global"):
                return True

            # OPTIMIZACIÓN: Convertir string completo a integers UNA VEZ
            q_int = [ord(c) for c in q]

            total_soft_score = 0.0
            total_input_ngrams = 0

            for n, matrix_slice in self.global_matrices.items():
                # Generar n-gramas por slicing (sin ord)
                input_ngrams_int = [q_int[i:i + n] for i in range(len(q_int) - n + 1)]

                if not input_ngrams_int:
                    continue

                num_input = len(input_ngrams_int)
                total_input_ngrams += num_input

                matrix_input: np.ndarray[Any, np.dtype[np.uint8]] = np.array(input_ngrams_int, dtype=np.uint8)

                matches = (matrix_input[:, np.newaxis, :] == matrix_slice[np.newaxis, :, :])
                sim_matrix = matches.sum(axis=2) / n

                rows, cols = np.where(sim_matrix > 0)
                if rows.size > 0:
                    scores = sim_matrix[rows, cols]
                    prio_indices = np.argsort(-scores)

                    used_in: Set[int] = set()
                    used_gl: Set[int] = set()

                    for idx in prio_indices:
                        r, c = rows[idx], cols[idx]
                        if r not in used_in and c not in used_gl:
                            total_soft_score += float(scores[idx])
                            used_in.add(r)
                            used_gl.add(c)
                            if len(used_in) == num_input:
                                break

            if total_input_ngrams == 0:
                return False

            soft_coverage = total_soft_score / total_input_ngrams
            is_valid = soft_coverage > self.global_filter_threshold

            return is_valid

        except Exception as e:
            logger.error(f"Error en filtro matricial único: {e}")
            return False

    def _remove_noise_substrings(self, text: str) -> Tuple[str, List[str]]:
        cleaned = text
        removed_noise: List[str] = []
        try:
            candidates: List[Tuple[str, Dict[int, List[str]]]] = []
            for i, word in enumerate(self.noise_words):
                if word and i < len(self.noise_grams):
                    candidates.append((word, self.noise_grams[i]))

            for noise_word, grams_forbidden in candidates:
                noise_len = len(noise_word)
                min_w = max(1, noise_len - self.window_flex)

                found_any = True
                while found_any:
                    found_any = False
                    current_max_w = min(len(cleaned), noise_len + self.window_flex)

                    for w in range(current_max_w, min_w - 1, -1):
                        if w > len(cleaned):
                            continue
                        for j in range(0, len(cleaned) - w + 1):
                            sub = cleaned[j:j + w]

                            if sub == noise_word:
                                similarity = 1.0
                            else:
                                grams_sub = self._build_query_grams(sub)
                                similarity = self._score_hybrid_greedy(grams_forbidden, grams_sub)
                            # Penalización simétrica
                            similarity *= self._length_penalty(sub, noise_word)

                            if similarity > self.forb_match:
                                cleaned = (cleaned[:j] + " " + cleaned[j + w:]).strip()
                                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                                removed_noise.append(sub)
                                logger.debug(
                                    f"SUBSTRING ELIMINADO: '{sub}' | Similitud: {similarity:.4f} | RUIDO ORIG: '{noise_word}'")
                                found_any = True
                                break
                        if found_any:
                            break
            return cleaned, removed_noise

        except Exception as e:
            logger.error(f"Error eliminando substrings de ruido: {e}", exc_info=True)
            return text, []

    def check_full_word(self, text: str, place: str) -> bool:
        if place == "global":
            return text in set(self.global_words)
        elif place == "noise":
            return text in set(self.noise_words)
        else:
            logger.warning(f"Error en parámetro 'place': {place}, se retornará True intencionalmente")
            return True

    def _normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()
            # Convertir cualquier cosa que NO sea letra o espacio en un ESPACIO
            q = re.sub(r"[^a-z\s]+", " ", q)
            # Limpiar espacios múltiples / extremos
            q = re.sub(r"\s+", " ", q).strip()
            return q
        except Exception as e:
            logger.error(msg=f"Error limpiando texto: {e}", exc_info=True)
        return ""

    def _update_best_match(self, current_best: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide si el nuevo match es mejor que el actual según las reglas de similitud y longitud.
        """
        if match["similarity"] > current_best["similarity"]:
            return match
        elif abs(match["similarity"] - current_best["similarity"]) < self.min_diff:
            if len(match["key_word"]) > len(current_best["key_word"]):
                return match
        return current_best

    def _length_penalty(self, a: str, b: str) -> float:
        """Penalización simétrica por diferencia de longitud."""
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0.0
        if la == lb:
            return 1.0
        return min(la, lb) / max(la, lb)

    def vectorice_word(self, text: str) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return np.array([ord(char) for char in text], dtype=np.uint8)

    def check_full_vectors(self, text: str) -> bool:
        vect_text = self.vectorice_word(text)
        vec_len = len(vect_text)

        lens = [len(array) for array in self.noise_array]
        if not vec_len in lens:
            return False
        
        mask = lens.index(vec_len) 
        if not mask:
            logger.info(f"Texto no coincide en largo")
            return False
        
        cand = self.noise_array[mask]  # Sin list comprehension
        sims = np.sum(vect_text == cand)
        similarity = sims / vec_len
        # revect_text = "".join(chr(sim) for sim in sims)

        # logger.info(f"SIMS: {sims}")

        similarity = sims.size / vec_len if sims.size > 0 else 0

        if similarity > self.forb_match:
            cand_str = "".join(chr(c) for c in cand)
            logger.debug(f"Similitud '{similarity}' para '{text}' con: '{cand_str}'")
            return True
        else:
            logger.debug(f"Similitud insufuciente: '{similarity}' para '{text}'")

            return False
