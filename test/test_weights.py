# prueba de pesos: test_weights.py
import numpy as np
import pickle
import os
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "word_finder_model.pkl"))

with open(model_path, "rb") as f:
    m = pickle.load(f)

vect_union = m["vectorizer"]
global_words = m["global_words"]
variant_to_field = {k.lower(): v for k, v in m.get("variant_to_field", {}).items()}
if not variant_to_field and m.get("key_fields"):
    for fld, vars in m["key_fields"].items():
        if isinstance(vars, str):
            vars = [vars]
        for v in vars or []:
            if isinstance(v, str) and v.strip():
                variant_to_field[v.strip().lower()] = fld

transformer_list = getattr(vect_union, "transformer_list", [])
names = [n for n, _ in transformer_list]
print("Sub-transformers:", names)

def loo_score(fu):
    X = fu.transform(global_words)
    sims = cosine_similarity(X, X)
    n = len(global_words)
    correct = 0
    for i in range(n):
        row = sims[i].copy()
        row[i] = -1.0
        idx = int(row.argmax())
        if variant_to_field.get(global_words[idx].lower()) == variant_to_field.get(global_words[i].lower()):
            correct += 1
    return correct, n, correct / n

# grid search simple en peso_char de 0.0 a 1.0
best = (None, -1.0)
for wc in np.linspace(0.0, 1.0, 11):
    weights = {}
    for name, _ in transformer_list:
        if "word" in name.lower():
            weights[name] = 1.0
        elif "char" in name.lower():
            weights[name] = float(wc)
        else:
            weights[name] = 1.0
    fu = FeatureUnion(transformer_list, transformer_weights=weights)
    correct, n, ratio = loo_score(fu)
    print(f"weight_char={wc:.2f} -> LOO top1 {correct}/{n} ({100*ratio:.1f}%)")
    if ratio > best[1]:
        best = (weights.copy(), ratio)

print("Mejor peso encontrado:", best[0], "ratio:", best[1])