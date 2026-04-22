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
        self.prime = config["primes"]
        
    def train_all_vectorizers(self, key_words: Dict[str, Dict[str, List[str]]], noise_words: List[str]):
        global_filter = self._train_global(key_words)
        noise_filter = self._train_noise_filter(noise_words)
        return global_filter, noise_filter
    
    def _train_global(self, key_words: Dict[str, Dict[str, List[str]]]):
        all_ngrams: List[str] = []
        map_ngrams: Dict[str, List[List[int]]] = {}
        global_vocab: Dict[Tuple[int, int], Dict[str, List[str]]] = {}
        for field_id, (_, words) in enumerate(key_words.items(), 1):
            for id, (word, words_list) in enumerate(words.items(), 1):
                norm_word = self._normalize(word)
                index = (field_id, id)
                # array_index = np.array(index)
                for_matrixes: List[List[int]] = []
                for_matrixes.append(list(index))
                for n in range(self.min_ngram_size, self.max_ngram_size + 1):
                    # if n == 2:
                    #     temp_word = norm_word.replace(" ", "")
                    # else:
                    #     temp_word = norm_word
                    n_gramas = self._ngrams(norm_word, n)
                    for_matrixes.extend([[ord(char) for char in ng] for ng in n_gramas])
                    words_list.extend(n_gramas)
                    all_ngrams.extend(n_gramas)
                    map_ngrams[norm_word] = for_matrixes
                global_vocab[index] = {norm_word: words_list}
            
        all_words = [list(w.keys())[0] for w in global_vocab.values()]
        counts = Counter(all_ngrams)
        gngrams = list(counts.keys())
        logger.info("COUNTS:\n"f"{sum(counts.values())}")
        
        mapped_matrix: Dict[int, Any] = {n: [] for n in range(self.min_ngram_size, self.max_ngram_size + 1)}
        hash_i: Dict[int, List[int]] = {n: [] for n in range(self.min_ngram_size, self.max_ngram_size + 1)}
        
        for ngrams in map_ngrams.values():
            concatenated_index = (ngrams[0][0] * self.prime[0]) + ngrams[0][1]
            
            rows = ngrams[1:]  # omite el índice [field_id, id]
            for row in rows:
                size = len(row)
                if self.min_ngram_size <= size <= self.max_ngram_size:
                    mapped_matrix[size].append(row)
                    hash_i[size].append(concatenated_index)
        
        hash_index: Dict[int, np.ndarray[Any, np.dtype[np.uint32]]] = {n: np.array(hash_i[n], dtype=np.uint32) for n in hash_i}
        
        for n in range(self.min_ngram_size, self.max_ngram_size + 1):
            if mapped_matrix[n]:
                mapped_matrix[n] = np.array(mapped_matrix[n], dtype=np.uint8)
            else:
                mapped_matrix[n] = np.zeros((0, n), dtype=np.uint8)
            # logger.info(f"{mapped_matrix.get(n).shape}")
        # 1. Calcular tamaño máximo usando TODOS los ngramas de TODAS las longitudes
        min_n = self.min_ngram_size
        total_ngrams_all_sizes = len(gngrams) # Todos los ngramas sin filtrar
        filas_max_n = int(total_ngrams_all_sizes / self.top_ngrams_fraction)
                
        global_matrices: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        for n in range(self.min_ngram_size, self.max_ngram_size + 1):
            # Base: ngramas frecuentes del tamaño n (limitados a filas_n)
            ngrams_of_size = [ng for ng in gngrams if len(ng) == n]
            filas_n = int(filas_max_n * min_n / n)
            base_ngrams = ngrams_of_size[:filas_n]

            # Extras: keywords de longitud exacta n que no estén ya en la base
            short_keywords = sorted([w for w in all_words if len(w)==n])
            base_set = set(base_ngrams)

            extras = [w for w in short_keywords if w not in base_set]

            base_matrix = self._generate_matrix(n, base_ngrams)

            if extras:
                extras_matrix = self._generate_matrix(n, extras)
                global_matrices[n] = np.vstack([base_matrix, extras_matrix], dtype=np.uint8)
                # logger.info(f"Inyectando keywords cortas en matriz n={n}: {extras}")
            else:
                global_matrices[n] = base_matrix

        for matrix in global_matrices.values():
            logger.info(f"Tamaño de la matriz: {matrix.shape}")

        # logger.info("Matriz:\n"f"{global_matrices.get(2).shape}, }")
        return global_vocab, global_matrices, mapped_matrix, hash_index

    def _train_noise_filter(self, noise_words: List[str]) -> Dict[int, Dict[str, str | Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]]]:
        # timen = time.perf_counter()
        noise_words_sorted = sorted(noise_words, key=len, reverse=True)
        noise_filter: Dict[str, Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]] = {}

        for i, noise_word in enumerate(noise_words_sorted):

            word_grams_dict: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
            for n in range(self.min_ngram_size, self.max_ngram_size + 1):
                int_grams = np.array([[ord(char) for char in ng] for ng in self._ngrams(noise_word, n)])
                if int_grams.size > 0:
                    word_grams_dict[n] = int_grams
                    
            
                
            noise_filter[noise_word] = word_grams_dict
        logger.info(f"{noise_filter}")
        # logger.info(f"NOISE FLTER ACABADO EN {time.perf_counter()- timen:.6f}'s")
        return noise_filter

    def _normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
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
        # Solo acepta ngramas que NO inician ni terminan con espacio
        return [s[i:i+n] for i in range(len(s) - n + 1)]

    def _generate_matrix(self, size: int, ngrams: List[str]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Genera una matriz (len(ngrams) x size) usando uint8, sin padding."""
        if not ngrams:
            return np.zeros((0, size), dtype=np.uint8)

        matrix = np.empty((len(ngrams), size), dtype=np.uint8)
        for i, ng in enumerate(ngrams):
            matrix[i] = [ord(char) for char in ng]
        return matrix