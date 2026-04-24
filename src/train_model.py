import re
import numpy as np
# import time
import logging
import unicodedata
from typing import List, Dict, Any, Tuple, Set
from collections import Counter

logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        ngrams_range: Tuple[int, int] = config["char_ngrams"]
        self.min_ngram_size = ngrams_range[0]
        self.max_ngram_size = (self.min_ngram_size + 1) if ngrams_range[1] == self.min_ngram_size else ngrams_range[1]
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
                for_matrixes: List[List[int]] = []
                for_matrixes.append(list(index))
                for n in range(self.min_ngram_size, self.max_ngram_size + 1):
                    n_gramas = self._ngrams(norm_word, n)
                    for_matrixes.extend([[ord(char) for char in ng] for ng in n_gramas])
                    words_list.extend(n_gramas)
                    all_ngrams.extend(n_gramas)
                    map_ngrams[norm_word] = for_matrixes
                global_vocab[index] = {norm_word: words_list}
            
        all_words = [list(w.keys())[0] for w in global_vocab.values()]
        counts = Counter(all_ngrams)    # Conteo de N gramas
        gngrams: Set[str] = set(all_ngrams)       # Ejemplar de cada ngrama presente
        short_key = [w for w in all_words if self.max_ngram_size >= len(w)]
        
        unique_count = len([k for k, v in counts.items() if v == 1])    # Cantidad de n gramas con frecuencia mayor a 1
        total_count = len(all_ngrams)                                   # Cantidad total de ngramas
        max_rows_n = int((total_count - unique_count) / self.top_ngrams_fraction)
           
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
                continue
        # logger.info(f"{mapped_matrix}")
        
        inver_index: Dict[Tuple, np.ndarray[Any, np.dtype[np.uint32]]] = {}
        unique_list = [[ord(char) for char in ng] for ng in gngrams]    # [[1, 2], [4, 3]]
        
        for unique in unique_list:                                      # [1, 2] -> [4, 3]
            tuple_grma = tuple(unique)
            for n, grams in map_ngrams.items():
                # idx = grams[0]
                idx = (grams[0][0] * self.prime[0]) + grams[0][1]       # [1743]
                for gram in grams:                                      # [1, 2] -> [4, 3]
                    list_idx: List[int] = []
                    if gram != unique:                                  # [1, 2] == [1, 2]
                        continue
                    else:
                        list_idx.append(idx)
                        
                    inver_index[tuple_grma] = np.array(list_idx, np.uint32)
                    
        # logger.info(f"{inver_index}")
        min_grams_dict = {}
        for n in range(self.min_ngram_size, self.max_ngram_size + 1):
            list_n = []
            for w in short_key:                    
                list_n.extend([[ord(char) for char in ng] for ng in self._ngrams(w, n)])
            min_gramas = np.array(list_n)
            min_grams_dict[n] = min_gramas
        
        global_matrices: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        for n, ngrmas in mapped_matrix.items():
            min_grams = min_grams_dict[n]
            ngrmas = np.concatenate([ngrmas, min_grams])
            unique_array, unique_counts = np.unique(ngrmas, return_counts=True, axis=0)
            unique_id = np.where(unique_counts > 1)[0]
            unique_array = unique_array[unique_id]
            global_matrices[n] = unique_array
            logger.info(f"Tamaño de la global matriz: {unique_array.shape}")
            
            # logger.info("Matriz:\n"f"{global_matrices[n]}")
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
        # logger.info(f"{noise_filter}")
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
