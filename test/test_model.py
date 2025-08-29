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