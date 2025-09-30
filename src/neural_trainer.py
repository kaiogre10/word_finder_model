# src/neural_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import time

logger = logging.getLogger(__name__)

class OCRDataset(Dataset):
    """Dataset para entrenamiento de la red neuronal."""
    
    def __init__(self, texts: List[str], labels: List[int], char_to_idx: Dict[str, int], max_len: int = 50):
        self.texts = texts
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convertir texto a índices de caracteres
        char_indices = [self.char_to_idx.get(c, self.char_to_idx.get('<UNK>', 0)) for c in text[:self.max_len]]
        
        # Padding
        if len(char_indices) < self.max_len:
            char_indices += [0] * (self.max_len - len(char_indices))
        
        return torch.tensor(char_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class NeuralTrainer:
    """Clase para entrenar el modelo neuronal."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Valida el modelo."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001, patience=10, save_path='best_model.pth'):
        """
        Entrena el modelo con early stopping.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Iniciando entrenamiento...")
        logger.info(f"Epochs: {num_epochs}, LR: {lr}, Patience: {patience}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Entrenar
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validar
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Scheduler
            scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s): '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            
            # Early stopping y guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, save_path)
                logger.info(f'  ✓ Mejor modelo guardado (val_loss: {val_loss:.4f})')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping activado después de {epoch+1} epochs')
                    break
        
        logger.info(f'Entrenamiento completado. Mejor val_loss: {best_val_loss:.4f}')
        
        # Cargar mejor modelo
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return best_val_loss

def build_char_vocab(texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Construye vocabulario de caracteres desde los textos.
    """
    chars = set()
    for text in texts:
        chars.update(text)
    
    # Crear mapeos
    char_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, char in enumerate(sorted(chars), start=2):
        char_to_idx[char] = idx
    
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    logger.info(f"Vocabulario construido: {len(char_to_idx)} caracteres únicos")
    
    return char_to_idx, idx_to_char
