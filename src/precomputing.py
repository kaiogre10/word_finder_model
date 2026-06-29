import os
import scipy.sparse as sp # type: ignore
import numpy as np
import pandas as pd
from functools import cached_property
from typing import List, Dict, Any, Tuple
import logging
import time
from collections import Counter
import pdfplumber
from utils.utils import normalize, clean_pattern, space_pattern, format_elapsed_time

_clean_pattern = clean_pattern 
_space_pattern = space_pattern

logger = logging.getLogger(__name__)

class PrecomputedMatrix:
    def __init__(self, config: Dict[str, Any], project_root: str, key_words_dict: Dict[str, Any]):
        self.project_root = project_root
        key_words: Dict[str, Dict[str, List[str]]] = key_words_dict.get("key_words", {})
        self.noise_words: List[str] = key_words_dict["noise_words"]
        self.key_words = key_words
        self.matrix_size: int = config.get("matrix_size", 0)
        self.dict_path_full = os.path.join(*["/media/kaiogre05/DATA/data/text_data/dictionary/0_palabras_normalizadas.txt"])
        self.books_path_full = os.path.join(*["/media/kaiogre05/DATA/data/text_data/books"])
        ngrams_range: Tuple[int, int] = config["char_ngrams"]
        self.min_ngram_size = ngrams_range[0]
        self.max_ngram_size = (self.min_ngram_size + 1) if ngrams_range[1] == self.min_ngram_size else ngrams_range[1]
        output_path = config["data_path"]
        self.precompute = config.get("precompute", False)

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

    def get_all_key_words(self) -> List[str]: 
        all_key_words: List[str] = []
        for key_words_dict in self.key_words.values():
            for key_word in key_words_dict.keys():
                key_word = normalize(key_word)
                all_key_words.append(key_word)

        for noise_word in self.noise_words:
            noise_word = normalize(noise_word)
            all_key_words.append(noise_word)
        # logger.info(f"KEY_WORD: {all_key_words}")
        if self.precompute:
            all_key_words.append(self.procesar_pagina_pdf())
        return all_key_words

    @cached_property
    def all_words(self) -> List[str]:
        """
        Extrae n-gramas que nunca aparecen y los menos frecuentes.
        ruta_archivo_salida: Archivo de salida con n-gramas raros
        """
        data_words = pd.read_csv(
            self.dict_path_full, 
            header=None, 
            names=["words"], 
            dtype=str, sep=",", 
            skip_blank_lines=True, 
            quoting=3, 
            keep_default_na=False
            )["words"].to_list()
        
        all_key_words = self.get_all_key_words()
        return data_words + all_key_words

    def precompute_similarites(self):
        time0 = time.perf_counter()
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
                matrix_csr = sp.csr_matrix((data, (rows, cols)), shape=(M, total_cols))

                sparce_matches: sp.csr_matrix = matrix_csr @ matrix_csr.T

                data = np.divide(sparce_matches.data, N, dtype=np.float16)
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
            logger.info(f"PROCESO TOTAL ACABADO EN: {format_elapsed_time(time.perf_counter() - time0)}", )
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
        contador_ngramas: Dict[str, int] = Counter()
        t0 = time.perf_counter()
        for palabra in all_words:
            len_word = len(palabra)
            if len_word == n:
                contador_ngramas[palabra] += 1
            elif len_word > n:
                for i in range(len(palabra) - n + 1):
                    ngram = palabra[i:i + n]
                    contador_ngramas[ngram] += 1
            else:
                continue

        all_ngrams_count = len(sorted(contador_ngramas))
        all_top_ngrams: List[str] = [ngram[0] for ngram in contador_ngramas.most_common(all_ngrams_count)] 

        all_ngrams_arr = np.array(all_top_ngrams, dtype=f'S{n}')
        all_ngrams_array = all_ngrams_arr.view(np.uint8).reshape(all_ngrams_count, n)

        logger.info(f"ALL NGRAMS ARRAY TIME: {time.perf_counter() - t0}'s,  SHAPE: {all_ngrams_array.shape}")
        return all_ngrams_array
    
    def procesar_pagina_pdf(self) -> str:
    #     # 1. INTENTO DE SUPLANTACIÓN (Costo de CPU/RAM: Casi 0)
        try:
            pdf_path = [file for _, _, files in os.walk(self.books_path_full) for file in files if file.endswith(".pdf")]
            if not os.path.isdir(self.books_path_full):
                logger.info(f"NO EXISTE ruta: {pdf_path}")
            for pat in pdf_path:
                file_path = os.path.join(self.books_path_full, pat)
                if not os.path.isfile(file_path):
                    logger.info(f"NO EXISTE PDF: {file_path}")
                
                with pdfplumber.open(file_path, unicode_norm='NFD') as pdf:
                    texto_vectorial: List[str] = []
                    num_pages: int = 0
                    for _, pagina in enumerate(pdf.pages):
                        num_pages += 1
                        logger.info(f"Pages: {num_pages}")
                        text_page = pagina.extract_text_simple()
                        if text_page:
                            texto_vectorial.append(text_page)
                        continue
                    
                plain_text = " ".join(text for text in texto_vectorial if self.min_ngram_size >= len(text))
                plain_text = _clean_pattern.sub("", plain_text).lower()
                plain_text = _space_pattern.sub(" ", plain_text)
                plain_text.encode('ascii', 'ignore').decode('utf-8').strip()
                # logger.info(f"TEXTO VECTORIAL: '{plain_text}'")
                self.save_plain_text(plain_text=plain_text, pat=pat)
                return plain_text
        except FileNotFoundError as e:
            logger.warning(f"ERROR BUSCANDO PDF: {e}", exc_info=True)
        return ""

    def save_plain_text(self, plain_text: str, pat: str):
        file = f"{pat[:-4]}_plain_pdf.txt"
        if not os.path.exists(file):
            with open(file, 'w', encoding="utf-8") as f:
                f.write(plain_text)
        else:
            with open(file, 'r', encoding="utf-8") as f:
                f.read()
        