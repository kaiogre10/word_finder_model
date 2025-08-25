import os
from scripts.generate_model import _generate_model

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "palabras_clave.yaml")
    project_root = PROJECT_ROOT
    data_path = DATA_PATH
    _generate_model(data_path, project_root)