import os
import random
import sys
import pickle
from typing import List, Dict, Any, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.word_finder import WordFinder

try:
    MODEL_STD = os.path.join(ROOT, "data", "word_finder_model.pkl")
    wf = WordFinder(MODEL_STD, ROOT)
except Exception as e:
    logger.info(f"Error estableciendo root: {e}", exc_info=True)


logging.info("Threshold: %s", getattr(wf, "threshold", "N/A"))
logging.info("Rango de n-ggramas: %s", getattr(wf, "ngr", "N/A"))

with open(MODEL_STD, "rb") as f:
    model = pickle.load(f)
logging.info("'grams_index' en model: %s", "grams_index" in model)
logging.info("Longitud de grams_index: %d", len(model.get("grams_index", [])))

# Queries base
base_queries = [
    "total", "importe total", "total a pagar", "subtotal", "iva",
    "rfc", "folio", "ticket no.", "fecha", "hora", "cliente",
    "total de piezas", "total de articulos", "fecha",
    "descripcion", "cantidad", "precio unitario", "precio", "importe",
    "articulo", "producto", "servicio", "concepto", "detalle",
    "codigo", "sku", "referencia", "marca", "modelo", "razon social",
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
queries: List[str] = []
for q in base_queries:
    queries.append(q)
    for _ in range(3):
        queries.append(perturb(q))

def run_queries(queries: List[str], wf: WordFinder, show_no_match: bool = True, show_dudosos: bool = True, top_n: int = 10):
    num_matches = 0
    num_no_matches = 0
    matches: List[Tuple[str, Dict[str, Any]]] = []
    no_matches: List[str] = []
    dudosos: List[Tuple[str, Dict[str, Any]]] = []

    for q in queries:
        res: Optional[List[Dict[str, Any]]] = wf.find_keywords(q)
        used = wf._active
        if not res and "standard" in wf.available_models() and wf._active != "standard":
            wf.set_active_model("standard")
            res = wf.find_keywords(q)
            used = "standard"
        if res:
            num_matches += 1
            for r in (res if isinstance(res, list) else [res]):
                key_field = r.get("key_field")
                word_found = r.get("word_found")
                score = r.get("similarity")
                matches.append((q, r))
                # Considera dudoso si el score está cerca del umbral
                thr = wf._len_threshold(len(word_found))
                if show_dudosos and (score < thr + 0.05 and score > thr - 0.05):
                    dudosos.append((q, r))
        else:
            num_no_matches += 1
            no_matches.append(q)

    porcentaje: float = (100.00/len(queries)) * num_matches
    logger.info(f"{res}")
    logger.info(f"Resumen: {num_matches}/{len(queries)} matches")
    logger.info(f"Porcentaje de coincidencia de palabras: {porcentaje}%")
    
if __name__ == "__main__":
    run_queries(queries, wf)

# Test simple de estructura de retorno
logger.debug("\n=== Estructura de retorno ===")
time0 = time.perf_counter()
sample = wf.find_keywords("total")
if sample:
    logger.debug(f"Ejemplo: {sample[0]}")

# Prueba con diferentes inputs
test_queries = ["total", "iva", "rfc", "folio", "cliente", "fecha", "subtotal", "encabezados"]

for q in queries:
    result = wf.find_keywords(q)
    logger.debug(f"Query: '{q}'"
    f"Result: {result}")

logger.debug("TESTING EXACT RETURN VALUES FROM WordFinder.find_keywords()")

# Test con queries individuales
individual_test_queries = ["total", "iva", "rfc", "folio", "cliente", "fecha", "subtotal", "encabezados"]

logger.debug("\n1. TESTING INDIVIDUAL QUERIES (string input):")
for query in individual_test_queries:
    result = wf.find_keywords(query)
    logger.debug(f"\nInput: '{query}' | Return value: {result} | Return length: {len(result)}")
    if result:
        logger.debug(f"First item keys: {list(result[0].keys())}")
        logger.debug(f"First item values: {list(result[0].values())}")

# Test con lista de queries
list_queries = ["total", "iva", "rfc", "folio", "cliente", "fecha", "subtotal", "encabezados"]
logger.debug(
    "\n\n2. TESTING LIST INPUT:\n" +
    "-" * 60 +
    f"\nInput: {list_queries}\n" +
    f"Return value: {(result_list := wf.find_keywords(list_queries))}\n" +
    f"Return length: {len(result_list)}"
)

# Test detallado de estructura completa
logger.debug(
    "\n\n4. DETAILED STRUCTURE ANALYSIS:\n" +
    "-" * 60 +
    f"\nResult for 'total': {(detailed_result := wf.find_keywords('total'))}"
)
if detailed_result:
    item = detailed_result[0]
    logger.debug(
        f"\nDetailed analysis of first result item:\n"
        f"Keys count: {len(item.keys())}\n"
        f"Keys: {list(item.keys())}\n"
        f"Items:"
    )

logger.debug("\n" + "="*80 + "\nEND OF RETURN VALUE TESTING\n" + "="*80)
logger.info(f"Testing acabado en: {time.perf_counter()-time0:.6f}s")
logger.debug("\n5. USING DEBUG METHOD:\n" + "-" * 60)
debug_result = wf.debug_find_keywords("total")