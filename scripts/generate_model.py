import os
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def _generate_model(data_path: str, project_root: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extraer todas las palabras
    all_words = []
    for field, variants in config['key_fields'].items():
        all_words.extend(variants)
    
    logger.info(f"Generando modelo con {len(all_words)} palabras clave...")
    
    # Crear y entrenar vectorizador
    ngram_range = tuple(config['config']['ngram_range'])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    vectorizer.fit(all_words)
    
    # Preparar modelo para guardar
    model = {
        'vectorizer': vectorizer,
        'key_fields': config['key_fields'],
        'config': config['config'],
        'vocabulario_size': len(vectorizer.vocabulary_),
        'total_palabras': len(all_words)
    }
    
    # Guardar modelo
    output_path = os.path.join(project_root, 'data', 'word_finder_model.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"✓ Modelo generado exitosamente:")
    logger.info(f"  - Vocabulario: {model['vocabulario_size']} n-gramas")
    logger.info(f"  - Palabras clave: {model['key_fields']}")
    logger.info(f"  - Guardado en: {output_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'palabras_clave.yaml')
    project_root = PROJECT_ROOT
    data_path = DATA_PATH
    _generate_model(data_path, project_root)