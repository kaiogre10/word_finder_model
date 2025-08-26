import os
import pickle
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "word_finder_model.pkl"))
if not os.path.exists(model_path):
    raise FileNotFoundError(model_path)

with open(model_path, "rb") as f:
    m = pickle.load(f)

vect_union = m.get("vectorizer")
global_words = m.get("global_words") or []
variant_to_field = {k.lower(): v for k, v in m.get("variant_to_field", {}).items()} if m.get("variant_to_field") else {}
if not variant_to_field and m.get("key_fields"):
    for fld, vars in m["key_fields"].items():
        if isinstance(vars, str):
            vars = [vars]
        for v in vars or []:
            if isinstance(v, str) and v.strip():
                variant_to_field[v.strip().lower()] = fld

subs = dict(getattr(vect_union, "transformer_list", []))
word_vect = subs.get("word")
char_vect = subs.get("char")

def loo_eval(vect, name):
    X = vect.transform(global_words)
    sims = cosine_similarity(X, X)
    n = len(global_words)
    correct = 0
    scores = []
    for i in range(n):
        row = sims[i].copy()
        row[i] = -1.0  # excluir self
        idx = int(np.argmax(row))
        pred = global_words[idx]
        if variant_to_field.get(pred.lower()) == variant_to_field.get(global_words[i].lower()):
            correct += 1
        scores.append(row[idx])
    print(f"{name} LOO top1: {correct}/{n} ({100*correct/n:.1f}%) mean_score_ex_self={np.mean(scores):.3f}")

def noisy_tests(vect, name, samples=10):
    def perturb(s):
        s = s.strip()
        if len(s) <= 3:
            return s  # evitar romper demasiado
        i = random.randrange(len(s))
        return s[:i] + s[i+1:]  # simple deletion typo
    queries = [perturb(w) for w in random.sample(global_words, min(samples, len(global_words)))]
    Xq = vect.transform(queries)
    Y = vect.transform(global_words)
    sims = cosine_similarity(Xq, Y)
    print(f"\n{name} noisy tests:")
    for qi, q in enumerate(queries):
        row = sims[qi]
        idx = int(np.argmax(row))
        pred = global_words[idx]
        score = row[idx]
        print(f"  q='{q}' -> pred='{pred}' field_pred={variant_to_field.get(pred.lower())} score={score:.3f}")

# Ejecutar LOO
if word_vect is not None:
    loo_eval(word_vect, "WORD_ONLY")
if char_vect is not None:
    loo_eval(char_vect, "CHAR_ONLY")
loo_eval(vect_union, "UNION")

# Ejecutar pruebas con ruido
random.seed(42)
if word_vect is not None:
    noisy_tests(word_vect, "WORD_ONLY")
if char_vect is not None:
    noisy_tests(char_vect, "CHAR_ONLY")
noisy_tests(vect_union, "UNION")