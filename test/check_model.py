import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = Path("data/word_finder_model.pkl")

def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def iter_vocab_info(vectorizer):
    # soporta FeatureUnion o TfidfVectorizer simple
    infos = []
    if hasattr(vectorizer, "transformer_list"):
        for name, tr in vectorizer.transformer_list:
            vocab = getattr(tr, "vocabulary_", None)
            try:
                feats = tr.get_feature_names_out()
            except Exception:
                feats = list(vocab.keys()) if vocab else []
            infos.append((name, vocab, feats))
    else:
        vocab = getattr(vectorizer, "vocabulary_", None)
        try:
            feats = vectorizer.get_feature_names_out()
        except Exception:
            feats = list(vocab.keys()) if vocab else []
        infos.append(("vectorizer", vocab, feats))
    return infos

def main():
    if not MODEL_PATH.exists():
        print("Modelo no encontrado en", MODEL_PATH)
        return 1

    m = load_model(MODEL_PATH)

    print("Keys en el modelo:")
    for k, v in sorted(m.items()):
        print(f" - {k}: {type(v)}")

    vocab_size = m.get("vocabulario_size")
    print("\nvocabulario_size (guardado):", vocab_size)
    params = m.get("params", {})
    print("params:", params)

    vect = m.get("vectorizer")
    if vect is None:
        print("No hay 'vectorizer' en el pickle. Abortando checks.")
        return 1

    print("\nSub-vectorizadores / vocab info:")
    infos = iter_vocab_info(vect)
    total_calc = 0
    for name, vocab, feats in infos:
        n = len(vocab) if vocab is not None else len(feats)
        total_calc += n
        print(f"  - {name}: vocab_size={n}, sample_features={list(feats)[:10]}")

    print("vocab size (calc sum sub-vectorizers):", total_calc)

    # palabras candidatas
    global_words = m.get("global_words")
    if not global_words:
        key_fields = m.get("key_fields", {})
        gw = []
        for fld, vars in key_fields.items():
            if isinstance(vars, str):
                vars = [vars]
            for v in vars or []:
                if isinstance(v, str) and v.strip():
                    gw.append(v.strip())
        global_words = gw
    print("\nTotal global_words (candidatos):", len(global_words))
    print("Ejemplo global_words (primers 20):", global_words[:20])

    # matrices precomputadas
    Y_global = m.get("Y_global")
    if Y_global is None and global_words:
        print("Y_global no estaba en el pickle: la calculo ahora usando vectorizer...")
        try:
            Y_global = vect.transform(global_words)
            print("  - Y_global shape:", Y_global.shape, "nnz:", getattr(Y_global, "nnz", None))
        except Exception as e:
            print("  - fallo al transformar global_words:", e)
            Y_global = None
    else:
        if Y_global is not None:
            print("Y_global (precomputed) shape:", getattr(Y_global, "shape", None), "nnz:", getattr(Y_global, "nnz", None))
        else:
            print("Y_global: None")

    # headers
    header_words = m.get("header_words")
    if header_words:
        print("\nTotal header_words:", len(header_words))
        Y_headers = m.get("Y_headers")
        if Y_headers is None:
            try:
                Y_headers = vect.transform(header_words)
                print("  - Y_headers shape:", Y_headers.shape, "nnz:", getattr(Y_headers, "nnz", None))
            except Exception as e:
                print("  - fallo al transformar header_words:", e)
                Y_headers = None
        else:
            print("  - Y_headers (precomputed) shape:", getattr(Y_headers, "shape", None))
    else:
        print("\nNo header_words en el modelo.")

    # mapping variante -> campo
    variant_to_field = m.get("variant_to_field") or m.get("variant_map") or {}
    if not variant_to_field and global_words:
        # construir mapping heurístico (lower -> campo) desde key_fields si existe
        for fld, vars in m.get("key_fields", {}).items():
            if isinstance(vars, str):
                vars = [vars]
            for v in vars or []:
                if isinstance(v, str) and v.strip():
                    variant_to_field[v.strip().lower()] = fld
    print("\nMapping variantes->campo (ejemplo 10):", dict(list(variant_to_field.items())[:10]))

    # prueba top-k para algunas queries
    if Y_global is not None and global_words:
        print("\nPrueba top-k similitud (queries = primeras 5 global_words):")
        queries = global_words[:5]
        Xq = vect.transform(queries)
        sims = cosine_similarity(Xq, Y_global)
        for qi, q in enumerate(queries):
            row = sims[qi]
            top_idx = np.argsort(row)[-5:][::-1]
            print(f"\nQuery: '{q}'")
            for rank, idx in enumerate(top_idx, start=1):
                cand = global_words[idx]
                field = variant_to_field.get(cand.lower(), "<unknown>")
                print(f"  {rank}. {cand} (field={field}) score={row[idx]:.4f}")
    else:
        print("\nNo es posible probar top-k (Y_global o global_words faltan).")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())