import os
import sys

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scipy.sparse as sp
import numpy as np
from functools import cached_property
from typing import List, Dict, Any, Tuple
import logging
import time
from collections import Counter

dict_path: List[str] = ["/media/kaiogre05/DATA/tensor/diccionario/diccionary/0_palabras_normalizadas.txt"]

logger = logging.getLogger(__name__)

class PrecomputedMatrix:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.matrix_size: int = config.get("matrix_size", 0)
        self.dict_path_full = os.path.join(*dict_path)
        ngrams_range: Tuple[int, int] = config["char_ngrams"]
        self.min_ngram_size = ngrams_range[0]
        self.ngrams_name = config.get("ngrams_name", "")
        self.max_ngram_size = (self.min_ngram_size + 1) if ngrams_range[1] == self.min_ngram_size else ngrams_range[1]
        output_path = config["output_path"]
        self.output_path = os.path.join(self.project_root, *output_path)

    @cached_property
    def all_words(self) -> List[str]:
        """
        Extrae n-gramas que nunca aparecen y los menos frecuentes.
        ruta_archivo_salida: Archivo de salida con n-gramas raros
        """
        palabras: List[str] = []
        with open(file=self.dict_path_full, mode="r", encoding="utf-8") as file:
            for i, palabra in enumerate(file):
                palabra = palabra.strip()
                if palabra:
                    palabras.append(palabra)
        logger.info(f"RUTA ALL WORDS: '{self.dict_path_full}'")
        return palabras

    def precompute_similarites(self):
        ngrams_path = self.generte_ngrams()
        ngramas_matrixes = np.load(ngrams_path)
        sparce_blocks: Dict[str, Any] = {}
        try:
            t0 = time.perf_counter()
            for key in ngramas_matrixes.files:
                A = ngramas_matrixes[key]
                M, N = A.shape
                total_cols = (N * 256)
                pos = np.arange(N, dtype=np.uint32) * 256
                chars_idx = A + pos

                rows = np.repeat(np.arange(M, dtype=np.uint32), N)
                cols = chars_idx.flatten()
                data = np.ones(M * N, dtype=np.float32)
                matrix_csr = sp.csr_matrix((data, (rows, cols)), shape=(M, total_cols))

                sparce_matches = matrix_csr @ matrix_csr.T

                sparce_blocks[f"{key}_data"] = np.divide(sparce_matches.data, N, dtype=np.float32)
                sparce_blocks[f"{key}_indices"] = sparce_matches.indices
                sparce_blocks[f"{key}_indptr"] = sparce_matches.indptr
                sparce_blocks[f"{key}_shape"] = np.array(matrix_csr.shape, dtype=np.uint32)

            output_dir = os.path.join(self.output_path, "sparced_matrixes.npz")
            np.savez_compressed(output_dir, **sparce_blocks)
            logger.info(f"Matrices precomputadas en: {time.perf_counter() - t0}'s, guardadas en: '{output_dir}'")
        except Exception as e:
            logger.info(f"ERROR GENERANDO MATRIZ DE SIMILITUDES: {e}", exc_info=True)
        return ""

    def generte_ngrams(self) -> str:
        all_words = self.all_words
        ngrams_name = self.ngrams_name
        all_ngrams_dict: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        for n in range(self.min_ngram_size, (self.max_ngram_size + 1)):
            all_ngrams_array = self.vectorice_all_ngrams(all_words, n)
            all_ngrams_dict[rf"{n}{ngrams_name}"] = all_ngrams_array

        output_dir = os.path.join(self.output_path, "all_ngrams.npz")
        try:
            np.savez_compressed(output_dir, **all_ngrams_dict)
            logger.info(f"N GRAMAS ORDENADO GURADADOS EN: '{output_dir}', '{len(all_ngrams_dict.keys())}' LONGITUDES DE N-GRAMAS")
            return output_dir
        except OSError as e:
            logger.error(f"NO SE PUDO GENERAR '{output_dir}' BINARIO DE MATRICES: {e}", exc_info=True)
        return ""

    def vectorice_all_ngrams(self, all_words: List[str], n: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        contador_ngramas = Counter()
        total_ngrams = 0
        t0 = time.perf_counter()
        for palabra in all_words:
            len_word = len(palabra)
            if len_word == n:
                total_ngrams += 1
                ngram = palabra
                contador_ngramas[ngram] += 1
            elif len_word > n:
                for i in range(len(palabra) - n + 1):
                    ngram = palabra[i:i + n]
                    contador_ngramas[ngram] += 1
                    total_ngrams += 1
            else:
                continue

        all_ngrams_count = len(sorted(contador_ngramas))
        all_top_ngrams = [ngram[0] for ngram in contador_ngramas.most_common(all_ngrams_count)]

        all_ngrams_arr = np.array(all_top_ngrams, dtype=f'S{n}')
        all_ngrams_array = all_ngrams_arr.view(np.uint8).reshape(all_ngrams_count, -1)

        logger.info(f"ALL NGRAMS ARRAY TIME: {time.perf_counter() - t0}'s,  SHAPE: {all_ngrams_array.shape}")
        return all_ngrams_array
