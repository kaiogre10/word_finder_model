import os
import random
import sys
import pickle
from typing import List, Dict, Any, Tuple

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
    "total de piezas", "total de articulos", "fecha"
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
    return s[:i] + random.choice("abcdefghijklmnopqrstuvwxyz1234567890") + s[i+1:]

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

def run_queries(queries: List[str], wf: WordFinder, show_no_match: bool = True, show_dudosos: bool = True, top_n: int = 5):
    num_matches = 0
    num_no_matches = 0
    matches: List[Tuple[str, Dict[str, Any]]] = []
    no_matches: List[str] = []
    dudosos: List[Tuple[str, Dict[str, Any]]] = []

    for q in queries:
        res: List[Dict[str, Any]] = wf.find_keywords(q)
        used = wf._active
        if not res and "standard" in wf.available_models() and wf._active != "standard":
            wf.set_active_model("standard")
            res = wf.find_keywords(q)
            used = "standard"
        if res:
            num_matches += 1
            for r in (res if isinstance(res, list) else [res]):
                key_field = r.get("key_field")
                header_category = r.get("header_category")
                word_found = r.get("word_found")
                score = r.get("similarity")
                tipo = "KEY_FIELD" if key_field else ("HEADER" if header_category else "NO CLASIFICACIÓN")
                matches.append((q, r))
                # Considera dudoso si el score está cerca del umbral
                thr = wf._len_threshold(len(word_found))
                if show_dudosos and (score < thr + 0.05 and score > thr - 0.05):
                    dudosos.append((q, r))
        else:
            num_no_matches += 1
            no_matches.append(q)

    print(f"\nResumen:")
    print(f"  Total de queries: {len(queries)}")
    print(f"  Coincidencias encontradas: {num_matches}")
    print(f"  Sin coincidencia: {num_no_matches}")
    print()

    print("Ejemplos de matches:")
    for q, r in matches[:top_n]:
        key_field = r.get("key_field")
        header_category = r.get("header_category")
        word_found = r.get("word_found")
        score = r.get("similarity")
        tipo = f"KEY_FIELD={key_field}" if key_field else (f"HEADER={header_category}" if header_category else "NO CLASIFICACIÓN")
        print(f"Q: '{q}' -> {tipo:20} | word_found='{word_found}' | score={score:.4f}")

    if show_dudosos and dudosos:
        print("\nMatches dudosos (score cerca del umbral):")
        for q, r in dudosos[:top_n]:
            key_field = r.get("key_field")
            header_category = r.get("header_category")
            word_found = r.get("word_found")
            score = r.get("similarity")
            tipo = f"KEY_FIELD={key_field}" if key_field else (f"HEADER={header_category}" if header_category else "NO CLASIFICACIÓN")
            print(f"Q: '{q}' -> {tipo:20} | word_found='{word_found}' | score={score:.4f}")

    if show_no_match and no_matches:
        print("\nQueries sin match:")
        for q in no_matches[:top_n]:
            print(f"  - {q}")

    # Top mejores y peores scores
    if matches:
        sorted_matches = sorted(matches, key=lambda x: x[1].get("similarity", 0))
        print("\nTop 3 peores scores:")
        for q, r in sorted_matches[:3]:
            print(f"Q: '{q}' | score={r.get('similarity'):.4f} | word_found='{r.get('word_found')}'")
        print("\nTop 3 mejores scores:")
        for q, r in sorted_matches[-3:]:
            print(f"Q: '{q}' | score={r.get('similarity'):.4f} | word_found='{r.get('word_found')}'")

if __name__ == "__main__":
    run_queries(queries, wf)


# Prueba con diferentes inputs
test_queries = ["total", "iva", "rfc", "folio", "cliente"]

for query in test_queries:
    result = wf.find_keywords(query)
    print(f"Query: '{query}'")
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
    if result:
        print(f"First item keys: {list(result[0].keys())}")
    print("-" * 50)

print("="*80)
print("TESTING EXACT RETURN VALUES FROM WordFinder.find_keywords()")
print("="*80)

# Test con queries individuales
individual_test_queries = ["total", "iva", "rfc", "folio", "cliente", "fecha", "subtotal"]

print("\n1. TESTING INDIVIDUAL QUERIES (string input):")
print("-" * 60)
for query in individual_test_queries:
    result = wf.find_keywords(query)
    print(f"\nInput: '{query}' (type: {type(query)})")
    print(f"Return type: {type(result)}")
    print(f"Return value: {result}")
    print(f"Return length: {len(result)}")
    if result:
        print(f"First item type: {type(result[0])}")
        print(f"First item keys: {list(result[0].keys())}")
        print(f"First item values: {list(result[0].values())}")
        for key, value in result[0].items():
            print(f"  {key}: {value} (type: {type(value)})")

# Test con lista de queries
print("\n\n2. TESTING LIST INPUT:")
print("-" * 60)
list_queries = ["total", "iva", "rfc"]
result_list = wf.find_keywords(list_queries)
print(f"\nInput: {list_queries} (type: {type(list_queries)})")
print(f"Return type: {type(result_list)}")
print(f"Return value: {result_list}")
print(f"Return length: {len(result_list)}")
for i, item in enumerate(result_list):
    print(f"Item {i}: {item} (type: {type(item)})")

# Test con queries que no tienen match
print("\n\n3. TESTING NO MATCH CASES:")
print("-" * 60)
no_match_queries = ["xyz123", "asdflkj", ""]
for query in no_match_queries:
    result = wf.find_keywords(query)
    print(f"\nInput: '{query}'")
    print(f"Return: {result} (type: {type(result)}, length: {len(result)})")

# Test detallado de estructura completa
print("\n\n4. DETAILED STRUCTURE ANALYSIS:")
print("-" * 60)
detailed_result = wf.find_keywords("total")
print(f"Result for 'total': {detailed_result}")
if detailed_result:
    item = detailed_result[0]
    print(f"\nDetailed analysis of first result item:")
    print(f"  Type: {type(item)}")
    print(f"  Is dict: {isinstance(item, dict)}")
    print(f"  Keys count: {len(item.keys())}")
    print(f"  Keys: {list(item.keys())}")
    print(f"  Items:")
    for k, v in item.items():
        print(f"    '{k}': {repr(v)} (type: {type(v).__name__})")

print("\n" + "="*80)
print("END OF RETURN VALUE TESTING")
print("="*80)
# Usar el método debug
print("\n5. USING DEBUG METHOD:")
print("-" * 60)
debug_result = wf.debug_find_keywords("total")