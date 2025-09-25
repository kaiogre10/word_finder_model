import os
import random
import sys
import pickle
from typing import List, Dict, Any, Tuple, Optional
import logging
import time
import json
import glob

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cache_service import cleanup_project_cache
cleanup_project_cache(PROJECT_ROOT)

from src.word_finder import WordFinder
MODEL_STD = os.path.join(PROJECT_ROOT, "data", "word_finder_model.pkl")
DATA_FOLDER = os.path.join("C:/PerfectOCR/output/fragmented")
try:    
    wf: WordFinder = WordFinder(MODEL_STD, PROJECT_ROOT)
except Exception as e:
    logger.info(f"Error estableciendo root: {e}", exc_info=True)

try:
    with open(MODEL_STD, "rb") as f:
        model = pickle.load(f)
    logger.info("'grams_index' en model: %s", "grams_index" in model)
except Exception as e:
    logger.info(f"Error: {e}", exc_info=True)
base_queries: List[str] = [
    "ticketrazon", "preciocantidad", "detalleconcepto", "referenciaproducto",
    "servicioprecio", "subtotalhora", "cantidadsku", "importemodelo",
    "totalservicio", "fechadescripcion", "articulofolio", "marcaconcepto",
    "ivaarticulo", "razonsocialticket", "preciounitariocodigo", "ticketxyz",
    "totalxpto", "fechaclienteabc", "importetest", "sku", "modeloprueba",
    "preciounitario", "cantidadbar", "serviciolorem", "detalletest",
    "code"
]
# Perturbaciones
def delete_char(s: str) -> str:
    if len(s) <= 2: return s
    i = random.randrange(len(s))
    return s[:i] + s[i+1:]

def swap_chars(s: str) -> str:
    if len(s) <= 2: return s
    i = random.randrange(len(s)-1)
    lst = list(s)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return "".join(lst)

def replace_char(s: str) -> str:
    if not s: return s
    i = random.randrange(len(s))
    return s[:i] + random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+-=[]{}|;':\",./<>?`~áéíóúàèìòùâêîôûäëïöüñçÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÑÇ") + s[i+1:]

def remove_spaces(s: str) -> str:
    return s.replace(" ", "")

def perturb(s: str) -> str:
    ops = [delete_char, swap_chars, replace_char, remove_spaces]
    f = random.choice(ops)
    return f(s)

# Generar queries con ruido
random.seed(42)
text: List[str] = []
for q in base_queries:
    text.append(q)
    for _ in range(3):
        text.append(perturb(q))

def log_model_summary(wf: WordFinder):
    try:
        logger.info("Resumen del modelo WordFinder")
        logger.info(f"Total de palabras clave: {len(wf.key_words)}")
        logger.info(f"Rango de n-gramas: {wf.ngr}")
        logger.info(f"Umbral de similitud: {wf.threshold}")
    except Exception as e:
        logger.error(f"Error en logg: {e}", exc_info=True)

def log_search_results(res: Optional[List[Dict[str, Any]]], q: List[str], params: Dict[str, Any]):
    time0 = time.perf_counter()
    try:
        if not res:
            return None
        for i, res in enumerate(res):
            logger.info(f"Campo: {res.get('key_field')}, Palabra: '{res.get('word_found')}', Similitud: {res.get('similarity'):.6}, Query: '{q}")
        tiempo = time.perf_counter() - time0
        logger.debug(f"Tiempo total del tester: {tiempo}")
    except Exception as e:
        logger.info(f"Error en log: {e}", exc_info=True)

def run_queries(text: List[str], wf: WordFinder, show_no_match: bool = True, show_dudosos: bool = True):
    num_matches = 0
    num_no_matches = 0
    matches: List[Tuple[str, Dict[str, Any]]] = []
    no_matches: List[str] = []
    dudosos: List[Tuple[str, Dict[str, Any]]] = []

    time0 = time.perf_counter()
    try:
        for q in text:
            res: Optional[List[Dict[str, Any]]] = wf.find_keywords(q)
            used = wf._active
            # if not res and "standard" in wf.available_models() and wf._active != "standard":
            #     used = "standard"
            #     wf.set_active_model("standard")
            #     res = wf.find_keywords(q)
            if res:
                num_matches += 1
                for r in (res if isinstance(res, list) else [res]):
                    key_field = r.get("key_field")
                    params = r.get("params")
                    word_found = r.get("word_found")
                    score = r.get("similarity")
                    matches.append((q, r))
                    thr = wf._len_threshold(len(word_found))
                    if show_dudosos and (score < thr + 0.05 and score > thr - 0.05):
                        dudosos.append((q, r))
                log_search_results(res, q, params)
            else:
                num_no_matches += 1
                no_matches.append(q)
                if show_no_match:
                    logger.info(f"No match para: '{q}'")
    except Exception as e:
        logger.error(f"Error en RUN QUERIES: {e}", exc_info=True)

    porcentaje: float = (100.00/len(text)) * num_matches
    logger.info(f"Resumen final: {num_matches}/{len(text)} matches")
    logger.info(f"Porcentaje de coincidencia de palabras: {porcentaje:.2f}%")
    logger.info(f"Total dudosos: {len(dudosos)}")
    logger.info(f"Total sin match: {num_no_matches}")
    logger.info(f"Tiempo total: {time.perf_counter()-time0:.2f}s")

def _test_json(wf: WordFinder, DATA_FOLDER: str,):
    time0 = time.perf_counter()
    logger.info(f"\nBuscando archivos JSON en la carpeta {DATA_FOLDER}...")
    json_files = glob.glob(os.path.join(DATA_FOLDER, '*.json'))

    if not json_files:
        logger.error(f"No se encontraron archivos JSON en {DATA_FOLDER}.")
        exit()

    for file_path in json_files:
        logger.info(f"Cargando archivo: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            polygons:List[List[str]]  = json.load(f)

            all_text: List[str] = []
            for poly_data in polygons:
                text = poly_data.get("text", "")
                if text:
                    all_text.append(text)

                results = wf.find_keywords(all_text)
                logger.info(f"Total archivos procesados: {len(json_files)}: {results}")
                logger.info(f"Tiempo total: {time.perf_counter()-time0:.4f}s")

if __name__ == "__main__":
    time0 = time.perf_counter()
    wf = WordFinder(MODEL_STD, PROJECT_ROOT) 
    log_model_summary(wf)
    # _test_json(wf, DATA_FOLDER)
    run_queries(text, wf)

    # # Prueba con diferentes inputs
    # text = ["total", "iva", "rfc", "folio", "cliente", "fecha", "subtotal", "encabezados"]
    # for q in text:
    #     try:
    #         result = wf.find_keywords(q)
    #         if result is None:
    #             logger.warning("error")
    #         logger.debug(f"Query: '{q}'"
    #         f"Result: {result}")
    #     except Exception as e:
    #         logger.error(f"Error en Test: {e}", exc_info=True)

    # # logger.debug("TESTING EXACT RETURN VALUES FROM WordFinder.find_keywords()")

    # # Test detallado de estructura completa
    # detailed_result = wf.find_keywords(base_queries)
    # logger.debug(
    #     f"\n\n4. DETAILED STRUCTURE ANALYSIS:\nBase queries: {base_queries}"
    #     f"\nResult for 'total': {detailed_result}"
    # )
    # if detailed_result:
    #     item = detailed_result[0]
    #     logger.debug(
    #         f"\nDetailed analysis of first result item:\n"
    #         f"Keys count: {len(item.keys())}\n"
    #         f"Keys: {list(item.keys())}\n"
    #         f"Items:"
    #     )

    # logger.debug("\n" + "="*80 + "\nEND OF RETURN VALUE TESTING\n" + "="*80)
    logger.info(f"Testing acabado en: {time.perf_counter()-time0:.6f}s")
    # logger.debug("\n5. USING DEBUG METHOD:\n" + "-" * 60)
    # cleanup_project_cache(PROJECT_ROOT)
