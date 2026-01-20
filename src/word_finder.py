import os
# import datetime
import logging
import numpy as np
import pickle
import re
import unicodedata
import time
# from datetime import datetime
from typing import List, Any, Dict, Optional, Tuple, Set

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path: str):
        self.model: Dict[str, Any] = self._load_model(model_path)
        self.wf_path: str = "C:/word_finder_model/src/word_finder.py"
        self.params = self.model.get("params", {})
        self.all_ngrams = self.model.get("all_ngrams", {})
        self.global_words: List[str] = self.model["global_words"]
        self.variant_to_field = self.model.get("variant_to_field", {})
        self.noise_words = self.model["noise_words"]
        noise_filter = self.model.get("noise_filter", {})
        global_filter = self.model.get("global_filter", {})
        self.global_filter_threshold = float(self.params.get("global_filter_threshold"))
        self.noise_grams: List[Tuple[str, float]] = noise_filter["noise_grams"]
        self.threshold: float = self.params.get("threshold_similarity")
        self.ngrams: Tuple[int, int] = self.params["char_ngrams"]
        # self.thresholds_by_len: List[Tuple[int, int, float]] = [tuple(item) for item in self.params["thresholds_by_len"]]
        self.weights_by_n: List[Tuple[int, int, float]] = [tuple(item) for item in self.params["weights_by_n"]]
        self.window_flex = self.params.get("window_flexibility")
        self.forb_match: float = self.params.get("forb_match")
        self.global_matrices: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = global_filter.get("global_matrices", {})
        # self.model_time = self.model.get("model_time")
     #   timestamp_model = os.path.getmtime(self.wf_path)
       # fecha_wf = datetime.fromtimestamp(timestamp_model).isoformat()
      #p  logger.critical(f"FECHA DE GENERACIÓN DEL MODELO: {self.model_time}, FECHA DEL SCRIPT WORD_FINDER.PY: {fecha_wf}")

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            if not isinstance(self.model, dict):  # type: ignore
                raise ValueError("El pickle no tiene el formato esperado (dict).")
            return self.model
        except Exception as e:
            logger.error(f"Error al cargar el modelo {e}", exc_info=True)
            raise

    def find_keywords(self, text: List[str] | str, threshold: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
        t0 = time.perf_counter()
        try:
            final_threshold = threshold if threshold is not None else self.threshold

            single = False
            if isinstance(text, str):
                queue = [text]
                single = True
            else:
                queue = list(text)

            results: List[Dict[str, Any]] = []
            
            while queue:
                s = queue.pop(0)

                if not s:
                    continue

                q = self._normalize(s)
                if not q:
                    continue 

                if not self._is_potential_keyword(q):
                    continue
                
                q_cleaned, removed_noise = self._remove_noise_substrings(q)
                if removed_noise:
                    logger.debug(f"Ruido eliminado: '{removed_noise}' | Texto Limpio: '{q_cleaned}'")
                    q = q_cleaned

                if self.check_full_word(text=q, place="noise"):
                    logger.info(f"Input completo es ruido, ignorando: '{q}'")
                    q = q_cleaned
                    if not q:
                        continue

                # Lista para guardar todos los matches de este string 's'
                found_matches_for_s: List[Dict[str, Any]] = []

                for i in range(len(self.global_words)):
                    cand = self.global_words[i]
                    cand_len = len(cand)
                    min_w = max(1, cand_len - self.window_flex)
                    size = len(q) 
                    if min_w > size or size == 1:
                        continue

                    max_w = min(len(q), cand_len + self.window_flex)
                    grams_cand = self._build_query_grams(cand)

                    try:
                        for w in range(min_w, max_w + 1):
                            if w > len(q):
                                break
                            for j in range(0, len(q) - w + 1):
                                sub = q[j:j + w]
                                grams_sub = self._build_query_grams(sub)
                                # Solo asignar similitud perfecta si es la misma palabra completa (misma longitud)
                                if sub == cand and len(sub) == cand_len:
                                    ngram_score: float = 1.0
                                else:
                                    ngram_score: float = self._score_binary_cosine_multi_n(grams_cand, grams_sub)
                                    len_ratio = max(len(sub), cand_len) / max(1, min(len(sub), cand_len))
                                    if len_ratio >= 2.0:
                                        penalty = min(len(sub), cand_len) / max(len(sub), cand_len)
                                        ngram_score *= penalty
                                if ngram_score > final_threshold:

                                    key_field = self.variant_to_field.get(cand)
                                    # Añadir el match a la lista temporal en lugar de retornar
                                    found_matches_for_s.append({
                                        "key_field": key_field,
                                        "word_found": cand,
                                        "similarity": float(ngram_score),
                                        "text": q,
                                        "start": j,
                                        "end": j + w
                                    })

                    except Exception as e:
                        logger.error(f"Error en el bucle de búsqueda de find_keywords: {e}", exc_info=True)
            
                # Después de comprobar todos los candidatos, agrupar y seleccionar el mejor por campo
                if found_matches_for_s:
                    best_match_by_field: Dict[str, Dict[str, Any]] = {}
                
                    for match in found_matches_for_s:
                        field = match["key_field"]
                    
                    # Si es el primer match para este campo, o si es mejor que el guardado
                        if field not in best_match_by_field or match["similarity"] > best_match_by_field[field]["similarity"]:
                            best_match_by_field[field] = match
                  
                # Desempatar usando la similitud de palabra completa
                    final_matches = self._resolve_ambiguity_by_full_word(list(best_match_by_field.values()))
                    
                    if final_matches:
                        best_match = final_matches[0]
                        results.append(best_match)
                        
                        # Extraer la palabra encontrada y procesar el resto
                        start = best_match.get("start")
                        end = best_match.get("end")
                        
                        if start is not None and end is not None:
                            left_part = q[:start].strip()
                            right_part = q[end:].strip()
                            
                            if left_part:
                                queue.append(left_part)
                            if right_part:
                                queue.append(right_part)
                            
                            logger.debug(f"Extracted '{best_match['word_found']}' from '{q}'. Remaining: '{left_part}', '{right_part}'")
            if single:
                logger.info(f"RESULTS: {results}")
                return results if results else []
            return results
        except Exception as e:
            logger.error(f"Error buscando palabras clave: '{e}'", exc_info=True)

    def _resolve_ambiguity_by_full_word(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dada una lista con el mejor match por campo, resuelve empates o ambigüedades
        calculando la similitud de n-gramas entre el texto original y la palabra clave completa.
        Agrega logs útiles sobre el proceso de desempate y puntajes.
        """

        if not matches:
            return []
        
        if len(matches) == 1:
            return matches

        # Calcular un "score de desempate" para cada match
        for i, match in enumerate(matches):
            original_text = match['text']
            keyword_found = self._normalize(match['word_found'])

            grams_text = self._build_query_grams(original_text)
            grams_keyword = self._build_query_grams(keyword_found)

            tiebreaker_score = self._score_binary_cosine_multi_n(grams_keyword, grams_text)
            match['score_final'] = tiebreaker_score

            logger.debug(
                "EMPATE: Match #%d: campo: %s, palabra: '%s' | score de desempate: %.4f | texto: '%s'",
                i,
                match.get("key_field"),
                match.get("word_found"),
                tiebreaker_score,
                original_text
            )

        # Ordenar por el score de desempate
        matches.sort(key=lambda x: x['score_final'], reverse=True)

        logger.debug(
            "DESEMPATE: texto '%s': campo: %s, palabra: '%s', score_final: %.4f \n"
            "====================================================================================================================================================================================================",
            matches[0].get("text"),
            matches[0].get("key_field"),
            matches[0].get("word_found"),
            matches[0].get("score_final"),
        )
        return [matches[0]]

    def _build_query_grams(self, q: str) -> Dict[int, set[str]]:
        """Construye n-gramas de la consulta"""
        gq: Dict[int, set[str]] = {}
        for n in range(self.ngrams[0], self.ngrams[1] + 1):
            gq[n] = set(self._ngrams(q, n))
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
        """Calcula la similitud entre dos n-gramas como la proporción de caracteres iguales."""
        try:
            matches = float(sum(1 for x, y in zip(a, b) if x == y))
            # Normaliza por la longitud máxima para manejar n-gramas de longitudes distintas si fuera el caso.
            return matches / float(max(len(a), len(b)))
        except Exception as e:
            logger.warning(f"Error comprobando texto: {e}", exc_info=True)
            return 0.0

    def _binary_cosine(self, size_a: int, size_b: int, soft_intersection: float) -> float:
        """Calcula coseno binario entre dos conjuntos"""
        if size_a == 0 or size_b == 0:
            return 0.0
        return soft_intersection / float((size_a * size_b) ** 0.5)

    def _score_binary_cosine_multi_n(self, grams_a: Dict[int, set[str]], grams_b: Dict[int, set[str]]) -> float:
        """Calcula score ponderado por coseno binario multi-n-grama"""
        num = 0.0
        den = 0.0
        for n in range(self.ngrams[0], self.ngrams[1] + 1):
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

    def _get_weight_by_n(self, n: int, default: float = 1.0) -> float:
        for start, end, value in self.weights_by_n:
            if start <= n <= end:
                return value
        return default

    def _is_potential_keyword(self, q: str) -> bool:
        try:
            if not q or not self.global_matrices:
                return False
            
            if self.check_full_word(text=q, place="global"):
                return True

            total_soft_score = 0.0
            total_input_ngrams = 0

            for n, matrix_slice in self.global_matrices.items():
                input_ngrams = self._ngrams(q, n)
                if not input_ngrams: 
                    continue
                
                num_input = len(input_ngrams)
                total_input_ngrams += num_input
                
                # Matriz del Input (M x n) en uint8
                matrix_input: np.ndarray[Any, np.dtype[np.uint8]] = np.array([[ord(c) for c in ng] for ng in input_ngrams], dtype=np.uint8)
                
                # Comparación posicional: [M_input, N_global, n]
                matches = (matrix_input[:, np.newaxis, :] == matrix_slice[np.newaxis, :, :])
                
                # Similitud real (0.0 a 1.0) entre cada n-grama input y cada n-grama global
                sim_matrix = matches.sum(axis=2) / n
                
                # --- Lógica de Emparejamiento por Prioridad (Unicidad) ---
                # Identificamos todos los pares con similitud > 0
                rows, cols = np.where(sim_matrix > 0)
                if rows.size > 0:
                    # Obtenemos los scores y ordenamos de mayor a menor (1.0 > N)
                    # Esto garantiza que un match de 100% se asigne y bloquee sus índices
                    # antes de que un match parcial pueda reclamarlos.
                    scores = sim_matrix[rows, cols]
                    prio_indices = np.argsort(-scores)
                    
                    used_in: Set[np.ndarray[Any, np.dtype[np.intp]]] = set()
                    used_gl: Set[np.ndarray[Any, np.dtype[np.intp]]] = set()
                    
                    for idx in prio_indices:
                        r, c = rows[idx], cols[idx]
                        # Si ni el n-grama del input ni el de referencia han sido bloqueados:
                        if r not in used_in and c not in used_gl:
                            total_soft_score += float(scores[idx])
                            used_in.add(r)
                            used_gl.add(c)
                            
                            # Optimización: si ya asignamos todos los n-gramas del input, salimos
                            if len(used_in) == num_input:
                                break

            if total_input_ngrams == 0: 
                return False
            
            # El score final es la media de las mejores similitudes únicas encontradas
            soft_coverage = total_soft_score / total_input_ngrams
            
            is_valid = soft_coverage > self.global_filter_threshold
            
            if is_valid:
                logger.debug(f"Filtro global aprobado: '{q}' | Score Unívoco: {soft_coverage:.4f}")
            else:
                logger.info(f"Filtro global rechazado: '{q}' | Score Unívoco: {soft_coverage:.4f}")
                
            return is_valid

        except Exception as e:
            logger.error(f"Error en filtro matricial único: {e}")
            return False

    def _normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
            # Normaliza y elimina acentos
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()
            # Deja solo letras a-z y espacios
            q = re.sub(r"[^a-z\s]+", " ", q)
            # Elimina espacios extra y extremos
            q = re.sub(r"\s+", " ", q).strip()
            # Filtra cualquier caracter fuera del rango ASCII seguro (32-126)
            q = ''.join(c for c in q if 32 <= ord(c) <= 126)
            return q
        except Exception as e:
            logger.error(msg=f"Error limpiando texto: {e}", exc_info=True)
        return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "params": self.params
        }

    def _remove_noise_substrings(self, text: str) -> Tuple[str, List[str]]:
        """
        Elimina todos los substrings que coincidan con palabras prohibidas.
        Retorna: (texto_limpio, lista_de_ruidos_eliminados)
        """
        cleaned = text
        removed_noise: List[str] = []
        try:
            # Emparejar palabras con sus perfiles de n-gramas precalculados
            candidates: List[Tuple[str, Any]] = []
            for i, word in enumerate(self.noise_words):
                if word and i < len(self.noise_grams):
                    candidates.append((word, self.noise_grams[i]))
            
            # PASO 2: Ordenar por longitud descendente. 
            candidates.sort(key=lambda x: len(x[0]), reverse=True)

            for noise_word, grams_forbidden_tuple in candidates:
                noise_len = len(noise_word)
                min_w = max(1, noise_len - self.window_flex)

                if isinstance(grams_forbidden_tuple, dict):
                    grams_forbidden = grams_forbidden_tuple
                elif isinstance(grams_forbidden_tuple, tuple) and isinstance(grams_forbidden_tuple[0], dict):
                    grams_forbidden = grams_forbidden_tuple[0]
                else:
                    grams_forbidden: Dict[int, Set[str]] = {}

                # Buscar coincidencias (múltiples pases para eliminar repeticiones de la misma palabra)
                found_any = True
                while found_any:
                    found_any = False
                    
                    # Calculamos el max_w dinámicamente sobre el texto que se va reduciendo
                    current_max_w = min(len(cleaned), noise_len + self.window_flex)
                    
                    # Iterar de ventana GRANDE a ventana PEQUEÑA
                    for w in range(current_max_w, min_w - 1, -1):
                        if w > len(cleaned):
                            continue
                        for j in range(0, len(cleaned) - w + 1):
                            sub = cleaned[j:j + w]

                            # Comparación rápida primero
                            if sub == noise_word:
                                similarity = 1.0
                            else:
                                grams_sub = self._build_query_grams(sub)
                                # grams_forbidden ya es el diccionario óptimo
                                similarity = self._score_binary_cosine_multi_n(grams_forbidden, grams_sub)
                                
                                len_ratio = max(len(sub), noise_len) / max(1, min(len(sub), noise_len))
                                if len_ratio >= 2.0:
                                    penalty = min(len(sub), noise_len) / max(len(sub), noise_len)
                                    similarity *= penalty

                            if similarity > self.forb_match:
                                # Eliminar el substring agregando un espacio de seguridad para no fusionar palabras
                                cleaned = (cleaned[:j] + " " + cleaned[j + w:]).strip()
                                # Limpiar dobles espacios generados
                                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                                
                                removed_noise.append(sub)
                                logger.debug(f"SUBSTRING ELIMINADO: '{sub}' | Similitud: {similarity:.4f} | RUIDO ORIG: '{noise_word}'")
                                found_any = True
                                break # Romper bucle interno j para reiniciar escaneo
                        if found_any:
                            break # Romper bucle de ventana para reiniciar

            return cleaned, removed_noise

        except Exception as e:
            logger.error(f"Error eliminando substrings de ruido: {e}", exc_info=True)
            return text, []
  
    def check_full_word(self, text: str, place: str) -> bool:
        try:
            if place == "global":
                if text in set(self.global_words):
                    # logger.info(f"String global completo encontrado: '{text}'")
                    return True
                else:
                    return False
            if place == "noise":
                if text in set(self.noise_words):
                    # logger.info(f"String ruidoso completo encontrado: '{text}'")
                    return True
                else:
                    return False
            
        except Exception as e:
            logger.info(f"Error buscando string inmediatos: {e}")
        return False