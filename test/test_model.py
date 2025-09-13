import os
import random
import sys
import pickle
from typing import List, Dict, Any, Tuple, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.word_finder import WordFinder

MODEL_STD = os.path.join(ROOT, "data", "word_finder_model.pkl")
wf = WordFinder(MODEL_STD)

print("Threshold:", wf.threshold)
print(f"Rango de n-ggramas:", wf.ngr)
print()

with open(MODEL_STD, "rb") as f:
    model = pickle.load(f)
print("grams_index" in model)
print(len(model.get("grams_index", [])))
print()

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
    print(f"{res}")
    print(f"Resumen: {num_matches}/{len(queries)} matches")
    print(f"Porcentaje de coincidencia de palabras: {porcentaje}%")
    
if __name__ == "__main__":
    run_queries(queries, wf)

# Test simple de estructura de retorno
print("\n=== Estructura de retorno ===")
sample = wf.find_keywords("total")
if sample:
    print(f"Ejemplo: {sample[0]}")

# Prueba con diferentes inputs
test_queries = ["total", "iva", "rfc", "folio", "cliente", "fecha", "subtotal", "encabezados"]

for q in queries:
    result = wf.find_keywords(q)
    print(f"Query: '{q}'")
    print(f"Result: {result}")
    if result:
        print(f"First item keys: {list(result[0].keys())}")
    print("-" * 50)

print("="*80)
print("TESTING EXACT RETURN VALUES FROM WordFinder.find_keywords()")
print("="*80)

# Test con queries individuales
individual_test_queries = ["total", "iva", "rfc", "folio", "cliente", "fecha", "subtotal", "encabezados"]

print("\n1. TESTING INDIVIDUAL QUERIES (string input):")
print("-" * 60)
for query in individual_test_queries:
    result = wf.find_keywords(query)
    print(f"\nInput: '{query}")
    print(f"Return value: {result}")
    print(f"Return length: {len(result)}")
    if result:
        print(f"First item keys: {list(result[0].keys())}")
        print(f"First item values: {list(result[0].values())}")

# Test con lista de queries
print("\n\n2. TESTING LIST INPUT:")
print("-" * 60)
list_queries = ["total", "iva", "rfc", "folio", "cliente", "fecha", "subtotal", "encabezados"]
result_list = wf.find_keywords(list_queries)
print(f"\nInput: {list_queries}")
print(f"Return value: {result_list}")
print(f"Return length: {len(result_list)}")

# Test detallado de estructura completa
print("\n\n4. DETAILED STRUCTURE ANALYSIS:")
print("-" * 60)
detailed_result = wf.find_keywords("total")
print(f"Result for 'total': {detailed_result}")
if detailed_result:
    item = detailed_result[0]
    print(f"\nDetailed analysis of first result item:")
    print(f"Is dict: {isinstance(item, dict)}")
    print(f"Keys count: {len(item.keys())}")
    print(f"Keys: {list(item.keys())}")
    print(f"Items:")

print("\n" + "="*80)
print("END OF RETURN VALUE TESTING")
print("="*80)
# Usar el método debug
print("\n5. USING DEBUG METHOD:")
print("-" * 60)
debug_result = wf.debug_find_keywords("total")