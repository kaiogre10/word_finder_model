import os
import random
import sys
import pickle
# resolver modelos relativos al proyecto (usa only os)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# asegurar que la raíz del proyecto esté en sys.path para poder importar `src` como paquete
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.word_finder import WordFinder
MODEL_STD = os.path.join(ROOT, "data", "word_finder_model.pkl")

# elige weighted si existe, y pasa la ruta standard como fallback
wf = WordFinder(MODEL_STD)

print("Threshold:", wf.threshold)
print()

with open(MODEL_STD, "rb") as f:
    model = pickle.load(f)
print("grams_index" in model)
print(len(model.get("grams_index", [])))

# lista base de palabras (ajusta según tu modelo)
base_queries = [
    "total", "importe total", "total a pagar", "subtotal", "iva",
    "rfc", "folio", "ticket no.", "fecha", "hora", "cliente", "usted debe",
    "total de piezas", "total de articulos", "fecha"
]

# funciones de perturbación simples
def delete_char(s):
    if len(s) <= 2: return s
    i = random.randrange(len(s))
    return s[:i] + s[i+1:]

def swap_chars(s):
    if len(s) <= 2: return s
    i = random.randrange(len(s)-1)
    lst = list(s)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return "".join(lst)

def replace_char(s):
    if not s: return s
    i = random.randrange(len(s))
    return s[:i] + random.choice("abcdefghijklmnopqrstuvwxyz1234567890") + s[i+1:]

def remove_spaces(s):
    return s.replace(" ", "")

def perturb(s):
    ops = [delete_char, swap_chars, replace_char, remove_spaces]
    f = random.choice(ops)
    return f(s)

# construir conjunto de queries con ruido
random.seed(42)
queries = []
for q in base_queries:
    queries.append(q)
    for _ in range(3):  # 3 variantes ruidosas por palabra
        queries.append(perturb(q))

# test runner: intenta con active (weighted) y si no hay resultado hace fallback a standard
def run_queries(queries, wf):
    for q in queries:
        res = wf.find_keywords(q)
        used = wf._active
        # fallback simple: si no hay resultado y existe modelo 'standard', pruebo con standard
        if not res and "standard" in wf.available_models() and wf._active != "standard":
            wf.set_active_model("standard")
            res = wf.find_keywords(q)
            used = "standard"
        print(f"Q: '{q}'  (probado con {used})")
        if res:
            for r in (res if isinstance(res, list) else [res]):
                print(f"  -> field={r.get('field')} word_found='{r.get('word_found')}' score={r.get('similarity'):.6f}")
        else:
            print("  -> NO MATCH")
        print("-" * 60)

if __name__ == "__main__":
    run_queries(queries, wf)