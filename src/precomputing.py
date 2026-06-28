import os
import scipy.sparse as sp # type: ignore
import numpy as np
import pandas as pd
from functools import cached_property
from typing import List, Dict, Any, Tuple
import logging
import time
from collections import Counter

logger = logging.getLogger(__name__)

class PrecomputedMatrix:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.matrix_size: int = config.get("matrix_size", 0)
        self.dict_path_full = os.path.join(*["/media/kaiogre05/DATA/tensor/diccionario/diccionary/0_palabras_normalizadas.txt"])
        ngrams_range: Tuple[int, int] = config["char_ngrams"]
        self.min_ngram_size = ngrams_range[0]
        self.max_ngram_size = (self.min_ngram_size + 1) if ngrams_range[1] == self.min_ngram_size else ngrams_range[1]
        output_path = config["data_path"]

        self.data = config.get("data", "")
        self.indices = config.get("indices", "")
        self.indptr = config.get("indptr", "")
        self.mtx_shape = config.get("mtx_shape", "")

        self.matrix_folders = config.get("matrix_path", "")
        
        self.matrix_path = os.path.join(self.project_root, *output_path)

        self.ngrams_name = config.get("ngrams_name", "")
        ngrams_path = config["ngrams_path"]
        self.ngrams_path = os.path.join(self.project_root, output_path[0], *ngrams_path)
        os.makedirs(self.ngrams_path, exist_ok=True)


    @cached_property
    def all_words(self) -> List[str]:
        """
        Extrae n-gramas que nunca aparecen y los menos frecuentes.
        ruta_archivo_salida: Archivo de salida con n-gramas raros
        """
        return pd.read_csv(
            self.dict_path_full, 
            header=None, 
            names=["words"], 
            dtype=str, sep=",", 
            skip_blank_lines=True, 
            quoting=3, 
            keep_default_na=False
            )["words"].to_list()

    def precompute_similarites(self):
        ngrams_path = self.generte_ngrams()
        ngramas_matrixes = np.load(ngrams_path)
        try:
            t0 = time.perf_counter()
            for _, key in enumerate(ngramas_matrixes.files):
                A = ngramas_matrixes[key]
                M, N = A.shape
                total_cols = (N * 256)
                pos = np.arange(N, dtype=np.uint32) * 256
                chars_idx = A + pos

                rows = np.repeat(np.arange(M, dtype=np.uint32), N)
                cols = chars_idx.flatten()
                data = np.ones(M * N, dtype=np.float32)
                matrix_csr = sp.csr_matrix((data, (rows, cols)), shape=(M, total_cols), dtype=np.float32)

                sparce_matches: sp.csr_matrix = matrix_csr @ matrix_csr.T

                data = np.divide(sparce_matches.data, N, dtype=np.float32)
                indices = np.asarray(sparce_matches.indices, dtype=np.uint32)
                indptr = np.asarray(sparce_matches.indptr, dtype=np.uint32)
                shape = np.array(sparce_matches.shape, dtype=np.uint32)

                output_folder = os.path.join(*[self.matrix_path, rf"{key[:1]}_{self.matrix_folders}"])
                os.makedirs(output_folder, exist_ok=True)

                np.save(os.path.join(output_folder, self.data), data)
                np.save(os.path.join(output_folder, self.indices), indices)
                np.save(os.path.join(output_folder, self.indptr), indptr)
                np.save(os.path.join(output_folder, self.mtx_shape), shape)

            logger.info(f"Matrices precomputadas en: {time.perf_counter() - t0}'s, guardadas en: '{self.matrix_path}'")
        except Exception as e:
            logger.info(f"ERROR GENERANDO MATRIZ DE SIMILITUDES: {e}", exc_info=True)
        return ""

    def generte_ngrams(self) -> str:
        ngrams_name = self.ngrams_name
        all_ngrams_dict: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        for n in range(self.min_ngram_size, (self.max_ngram_size + 1)):
            all_ngrams_array = self.vectorice_all_ngrams(self.all_words, n)
            all_ngrams_dict[rf"{n}{ngrams_name}"] = all_ngrams_array
                                    
        output_dir = os.path.join(self.ngrams_path, "all_ngrams.npz")
        try:
            np.savez_compressed(output_dir, **all_ngrams_dict)
            logger.info(f"N GRAMAS ORDENADO GURADADOS EN: '{output_dir}', '{len(all_ngrams_dict.keys())}' LONGITUDES DE N-GRAMAS")
            return output_dir
        except OSError as e:
            logger.error(f"NO SE PUDO GENERAR '{output_dir}' BINARIO DE MATRICES: {e}", exc_info=True)
        return ""

    def vectorice_all_ngrams(self, all_words: List[str], n: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        contador_ngramas = Counter() # type: ignore
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
        all_top_ngrams: List[str] = [ngram[0] for ngram in contador_ngramas.most_common(all_ngrams_count)]

        all_ngrams_arr = np.array(all_top_ngrams, dtype=f'S{n}')
        all_ngrams_array = all_ngrams_arr.view(np.uint8).reshape(all_ngrams_count, n)

        logger.info(f"ALL NGRAMS ARRAY TIME: {time.perf_counter() - t0}'s,  SHAPE: {all_ngrams_array.shape}")
        return all_ngrams_array
