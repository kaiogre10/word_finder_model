import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ruta al modelo (resuelta con os)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "word_finder_model.pkl"))

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

# cargar pickle correctamente
with open(model_path, "rb") as f:
    m = pickle.load(f)

vect_union = m.get("vectorizer")
if vect_union is None:
    raise KeyError("El pickle no contiene 'vectorizer'")

global_words = m.get("global_words") or []
# construir mapping variante->campo si no existe
variant_to_field = {k.lower(): v for k, v in m.get("variant_to_field", {}).items()} if m.get("variant_to_field") else {}
if not variant_to_field and m.get("key_fields"):
    for fld, vars in m["key_fields"].items():
        if isinstance(vars, str):
            vars = [vars]
        for v in vars or []:
            if isinstance(v, str) and v.strip():
                variant_to_field[v.strip().lower()] = fld

# obtener sub-vectorizadores (FeatureUnion -> lista de (name, transformer))
subs = dict(getattr(vect_union, "transformer_list", []))
word_vect = subs.get("word")
char_vect = subs.get("char")

def eval_vect(vect, name):
    try:
        X = vect.transform(global_words)
    except Exception as e:
        print(f"{name}: error al transformar global_words: {e}")
        return
    sims = cosine_similarity(X, X)
    top1_correct = 0
    scores = []
    for i, row in enumerate(sims):
        idx = int(np.argmax(row))
        pred = global_words[idx]
        if variant_to_field.get(pred.lower()) == variant_to_field.get(global_words[i].lower()):
            top1_correct += 1
        scores.append(row[idx])
    print(f"{name}: top1={top1_correct}/{len(global_words)} ({100*top1_correct/len(global_words):.1f}%) mean_score={np.mean(scores):.3f}")

# ejecutar evaluaciones
if word_vect is not None:
    eval_vect(word_vect, "WORD_ONLY")
if char_vect is not None:
    eval_vect(char_vect, "CHAR_ONLY")
eval_vect(vect_union, "UNION")