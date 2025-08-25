import os
from scripts.generate_model import ModelGenerator


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    CONFIG_FILE = os.path.join(PROJECT_ROOT, "data", "config.yaml")
    project_root = PROJECT_ROOT
    config_path = CONFIG_FILE
    vectorizer_union = ModelGenerator(config_path, project_root)
    vectorizer = vectorizer_union.generate_model(config_path, project_root) 