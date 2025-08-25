import os
from scripts.generate_model import ModelGenerator

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "global_keywords.yaml")
    project_root = PROJECT_ROOT
    data_path = DATA_PATH
    generator = ModelGenerator(data_path, project_root)
    generator.generate_model(data_path, project_root) 