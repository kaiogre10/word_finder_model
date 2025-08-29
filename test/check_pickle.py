import os
import pickle

# resolver ruta relativa al proyecto (directorio padre de esta prueba)
MODEL1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "word_finder_model.pkl"))

print("CWD:", os.getcwd())
print("Script __file__ dir:", os.path.dirname(__file__))
print("Comprobando rutas:")
for p in (MODEL1):
    print(" -", p, "->", "EXISTE" if os.path.exists(p) else "NO EXISTE")

# opcional: cargar si existe
for p in (MODEL1):
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                m = pickle.load(f)
            print(f"  OK: cargado {p}, keys: {sorted(m.keys())}")
        except Exception as e:
            print(f"  ERROR al cargar {p}: {e}")