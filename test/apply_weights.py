import os
import pickle
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity

MODEL_IN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "word_finder_model.pkl"))
MODEL_OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "word_finder_model_weighted.pkl"))

with open(MODEL_IN, "rb") as f:
    m = pickle.load(f)

vect_union = m.get("vectorizer")
global_words = m.get("global_words") or []
variant_to_field = {k.lower(): v for k, v in m.get("variant_to_field", {}).items()}
if not variant_to_field and m.get("key_fields"):
    for fld, vars in m["key_fields"].items():
        if isinstance(vars, str):
            vars = [vars]
        for v in vars or []:
            if isinstance(v, str) and v.strip():
                variant_to_field[v.strip().lower()] = fld

# obtener transformadores ya entrenados
transformer_list = getattr(vect_union, "transformer_list", [])
print("Sub-transformers:", [n for n,_ in transformer_list])

# ajustar peso recomendado (word=1.0, char=0.1)
weights = {}
for name, _ in transformer_list:
    if "word" in name.lower():
        weights[name] = 1.0
    elif "char" in name.lower():
        weights[name] = 0.1
    else:
        weights[name] = 1.0

# reconstruir FeatureUnion con los mismos transformadores + pesos (no re-fit necesario)
weighted_fu = FeatureUnion(transformer_list, transformer_weights=weights)

# evaluar LOO
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

correct, n, ratio = loo_score(weighted_fu)
print("Weighted LOO top1: {}/{} ({:.1%})".format(correct, n, ratio))

# guardar modelo nuevo (no sobreescribe el original)
m_weighted = m.copy()
m_weighted["vectorizer"] = weighted_fu
m_weighted.setdefault("meta", {})["transformer_weights"] = weights

with open(MODEL_OUT, "wb") as f:
    pickle.dump(m_weighted, f)

print("Guardado modelo ponderado en:", MODEL_OUT)