import re
import numpy as np
# import time
import logging
import unicodedata
from typing import List, Dict, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.ngrams: Tuple[int, int] = config["char_ngrams"]
        self.min_ngram_size = self.ngrams[0]
        self.max_ngram_size = (self.min_ngram_size + 1) if self.ngrams[1] == self.min_ngram_size else self.ngrams[1]
        self.top_ngrams_fraction: int = config.get("top_ngrams_fraction", {})
        
    def train_all_vectorizers(self, key_words: Dict[str, Dict[str, List[str]]], noise_words: List[str]):
        
        global_filter = self._train_global(key_words)
        noise_filter = self._train_noise_filter(noise_words)
        return global_filter, noise_filter
    
    def _train_global(self, key_words: Dict[str, Dict[str, List[str]]]):
        global_vocab: Dict[Tuple[int, int], Dict[str, List[str]]] = {}
        all_ngrams: List[str] = []
        map_ngrams: Dict[str, List[List[int]]] = {}
        for field_id, (_, words) in enumerate(key_words.items(), 1):
            for id, (word, words_list) in enumerate(words.items(), 1):
                norm_word = self._normalize(word)
                index = (field_id, id)
                # array_index = np.array(index)
                for_matrixes: List[List[int]] = []
                for_matrixes.append(list(index))
                for n in range(self.min_ngram_size, self.max_ngram_size + 1):
                    n_gramas = self._ngrams(norm_word, n)
                    for_matrixes.extend([[ord(char) for char in ng] for ng in n_gramas])
                    words_list.extend(n_gramas)
                    all_ngrams.extend(n_gramas)
                    # for_matrixes.append(list(index))
                    map_ngrams[norm_word] = for_matrixes
                # logger.info(f"SHAPE INDES: {map_ngrams}")
                    
                global_vocab[index] = {norm_word: words_list}
        
        # logger.info("GLOBQL:\n"f"{map_ngrams}")
            
        all_words = [list(w.keys())[0] for w in global_vocab.values()]
        counts = Counter(all_ngrams)
        gngrams = list(counts.keys())
        
        maped_matrix: Dict[Tuple[int, int], np.ndarray[Any, np.dtype[np.uint8]]] = {}
        for _, matrixes in map_ngrams.items():
            index: Tuple[int, int] = (matrixes[0][0], matrixes[0][1])
            cols = self.max_ngram_size
            vals_fill = cols - 2
            index_array_pad = np.concatenate([np.array(index), np.zeros(vals_fill, np.uint8)])
            # logger.info(f"MAPPED: {index_array_pad.shape}")
            # Crear matriz donde cada fila es un n-grama de la palabra
            matrixes.remove(matrixes[0])
            rows = len(matrixes)
            matrix_n = np.zeros((rows, cols), dtype=np.uint8)
            for i, mat in enumerate(matrixes):
                matrix_n[i, :len(mat)] = np.array(mat, dtype=np.uint8)
            
            # Diccionario con palabra como clave y matriz como valor
            maped_matrix[index] = matrix_n
        # logger.info(f"MAPPED: {maped_matrix}")
        
        # 1. Calcular tamaño máximo usando TODOS los ngramas de TODAS las longitudes
        min_n = self.min_ngram_size
        total_ngrams_all_sizes = sum(counts.values()) # Todos los ngramas sin filtrar
        filas_max_n = int(total_ngrams_all_sizes / self.top_ngrams_fraction)
        
        # array_map = np.array([w for w in global_vocab.keys()])
        
        global_matrices: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        for n in range(self.min_ngram_size, self.max_ngram_size + 1):
            # Base: ngramas frecuentes del tamaño n (limitados a filas_n)
            ngrams_of_size = [ng for ng in gngrams if len(ng) == n]
            filas_n = max(1, int(filas_max_n * min_n / n))
            base_ngrams = ngrams_of_size[:filas_n]

            # Extras: keywords de longitud exacta n que no estén ya en la base
            short_keywords = sorted([w for w in all_words if len(w)==n])
            base_set = set(base_ngrams)

            extras = [w for w in short_keywords if w not in base_set]

            # if extras:
            #     logger.info(f"Inyectando keywords cortas en matriz n={n}: {extras}")

            base_matrix = self._generate_matrix(n, base_ngrams)

            if extras:
                extras_matrix = self._generate_matrix(n, extras)
                global_matrices[n] = np.vstack([base_matrix, extras_matrix], dtype=np.uint8)
            else:
                global_matrices[n] = base_matrix

        # for matrix in global_matrices.values():
            # logger.info(f"Tamaño de la matriz: {matrix.shape}")

        # logger.info("Matriz:\n"f"{global_matrices.get(2).shape}, }")
        return global_vocab, global_matrices, maped_matrix

    def _train_noise_filter(self, noise_words: List[str]):
        # timen = time.perf_counter()
        noise_words_sorted = sorted(noise_words, key=len, reverse=True)
        noise_filter: Dict[int, Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]] = {}

        for i, noise_word in enumerate(noise_words_sorted):

            word_grams_dict: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
            for n in range(self.min_ngram_size, self.max_ngram_size + 1):
                int_grams = np.array([[ord(char) for char in ng] for ng in self._ngrams(noise_word, n)])
                if int_grams.size > 0:
                    word_grams_dict[n] = np.stack(int_grams)
                    
                # logger.info(f"{word_grams_dict}")
                
            noise_filter[i] = {
                "noise_words": noise_word,
                "noise_grams": word_grams_dict,
            }        
        # logger.info(f"NOISE FLTER ACABADO EN {time.perf_counter()- timen:.6f}'s")
        return noise_filter

    def _normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
            # Eliminar espacios al borde y convertir puntos
            s = s.lower()
            s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
            s = re.sub(r"(?<=[a-zA-Z])[^\w\s]+(?=[a-zA-Z])", "", s)
            s = re.sub(r"[^a-z\s]+", " ", s)
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
            return re.sub(r"\s+", " ", q).strip()
        except UnicodeError as e:
            logger.warning(f"ERROR codificando: {e}", exc_info=True)
        return ""

    def _ngrams(self, s: str, n: int) -> List[str]:
        if n <= 0 or not s:
            return []
        if len(s) < n:
            return []
        return [s[i:i+n] for i in range(len(s) - n + 1)]

    def _generate_matrix(self, size: int, ngrams: List[str]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Genera una matriz (len(ngrams) x size) usando uint8, sin padding."""
        if not ngrams:
            return np.zeros((0, size), dtype=np.uint8)

        matrix = np.empty((len(ngrams), size), dtype=np.uint8)
        for i, ng in enumerate(ngrams):
            matrix[i] = [ord(char) for char in ng]
        return matrix