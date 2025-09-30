# scripts/train_neural_model.py
import os
import sys
import json
import yaml
import logging
import torch
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(filename)s:%(lineno)d %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar rutas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.neural_classifier import OCRKeywordClassifier
from src.neural_trainer import NeuralTrainer, OCRDataset, build_char_vocab
from torch.utils.data import DataLoader

def load_config(config_path):
    """Carga configuración desde YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_training_data(data_path):
    """Carga datos de entrenamiento preparados."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def train_neural_model():
    """Entrena el modelo neuronal completo."""
    
    # Rutas
    config_path = os.path.join(PROJECT_ROOT, "training", "neural_config.yaml")
    data_path = os.path.join(PROJECT_ROOT, "training", "neural_training_data.json")
    output_model_path = os.path.join(PROJECT_ROOT, "data", "neural_model.pth")
    
    logger.info("="*80)
    logger.info("ENTRENAMIENTO DE MODELO NEURONAL OCR")
    logger.info("="*80)
    
    # Cargar configuración
    if not os.path.exists(config_path):
        logger.error(f"Archivo de configuración no encontrado: {config_path}")
        return
    
    config = load_config(config_path)
    logger.info(f"Configuración cargada desde: {config_path}")
    
    # Preparar datos si no existen
    if not os.path.exists(data_path):
        logger.info("Datos de entrenamiento no encontrados. Preparando datos...")
        from training.prepare_neural_data import prepare_training_data
        
        key_words_path = os.path.join(PROJECT_ROOT, "training", "key_words_labels.json")
        prepare_training_data(
            key_words_path=key_words_path,
            output_path=data_path,
            augment=config['data']['augment'],
            augmentation_factor=config['data']['augmentation_factor'],
            negative_ratio=config['data']['negative_ratio']
        )
    
    # Cargar datos
    logger.info(f"Cargando datos desde: {data_path}")
    dataset = load_training_data(data_path)
    
    texts = dataset['texts']
    label_indices = dataset['label_indices']
    label_to_idx = dataset['label_to_idx']
    idx_to_label = dataset['idx_to_label']
    
    logger.info(f"Datos cargados: {len(texts)} ejemplos, {dataset['num_classes']} clases")
    
    # Construir vocabulario de caracteres
    logger.info("Construyendo vocabulario de caracteres...")
    char_to_idx, idx_to_char = build_char_vocab(texts)
    vocab_size = len(char_to_idx)
    
    # Dividir datos
    val_split = config['training']['val_split']
    test_split = config['training']['test_split']
    
    # Train/Temp split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, label_indices, test_size=(val_split + test_split), random_state=42, stratify=label_indices
    )
    
    # Val/Test split
    val_size = val_split / (val_split + test_split)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=(1 - val_size), random_state=42, stratify=temp_labels
    )
    
    logger.info(f"División de datos:")
    logger.info(f"  Train: {len(train_texts)} ejemplos")
    logger.info(f"  Val:   {len(val_texts)} ejemplos")
    logger.info(f"  Test:  {len(test_texts)} ejemplos")
    
    # Crear datasets
    max_len = config['model']['max_len']
    train_dataset = OCRDataset(train_texts, train_labels, char_to_idx, max_len)
    val_dataset = OCRDataset(val_texts, val_labels, char_to_idx, max_len)
    test_dataset = OCRDataset(test_texts, test_labels, char_to_idx, max_len)
    
    # Crear dataloaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo
    logger.info("Inicializando modelo neuronal...")
    num_classes = dataset['num_classes']
    model = OCRKeywordClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout']
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parámetros del modelo:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Entrenables: {trainable_params:,}")
    
    # Entrenar
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")
    
    trainer = NeuralTrainer(model, device=device)
    
    best_val_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['learning_rate'],
        patience=config['training']['patience'],
        save_path=output_model_path + '.temp'
    )
    
    # Evaluar en test
    logger.info("Evaluando en conjunto de test...")
    test_loss, test_acc = trainer.validate(test_loader, torch.nn.CrossEntropyLoss())
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Guardar modelo final con todos los metadatos
    logger.info(f"Guardando modelo final en: {output_model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'max_len': max_len,
        'embedding_dim': config['model']['embedding_dim'],
        'hidden_dim': config['model']['hidden_dim'],
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'config': config
    }, output_model_path)
    
    logger.info("="*80)
    logger.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info(f"Modelo guardado en: {output_model_path}")
    logger.info(f"Precisión en test: {test_acc:.2f}%")
    logger.info("="*80)

if __name__ == "__main__":
    try:
        train_neural_model()
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}", exc_info=True)
        sys.exit(1)