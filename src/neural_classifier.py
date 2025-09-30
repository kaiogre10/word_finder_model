#src/neural_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class OCRKeywordClassifier(nn.Module):
    """
    Red neuronal para clasificar palabras clave en texto OCR.
    Arquitectura: Embedding + CNN + LSTM + Attention + Classifier
    """
    def __init__(self, vocab_size, num_classes, embedding_dim=64, hidden_dim=128, dropout=0.3):
        super(OCRKeywordClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 1. Embedding de caracteres
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Convoluciones para capturar patrones de diferentes longitudes
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=4, padding=2)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 3. LSTM bidireccional
        self.lstm = nn.LSTM(
            hidden_dim * 3,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # 4. Atención multi-cabeza
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 5. Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: tensor de shape (batch_size, seq_len) con índices de caracteres
        Returns:
            logits: tensor de shape (batch_size, num_classes)
        """
        # 1. Embedding
        embedded = self.char_embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        # 2. Convoluciones paralelas
        conv1_out = F.relu(self.bn1(self.conv1(embedded)))
        conv2_out = F.relu(self.bn2(self.conv2(embedded)))
        conv3_out = F.relu(self.bn3(self.conv3(embedded)))
        
        # Encontrar la longitud mínima para concatenar
        min_len = min(conv1_out.size(2), conv2_out.size(2), conv3_out.size(2))
        conv1_out = conv1_out[:, :, :min_len]
        conv2_out = conv2_out[:, :, :min_len]
        conv3_out = conv3_out[:, :, :min_len]
        
        # Concatenar salidas
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len, hidden_dim*3)
        
        # 3. LSTM
        lstm_out, (hidden, cell) = self.lstm(conv_out)
        
        # 4. Atención
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 5. Pooling: tomar el último estado
        pooled = attn_out[:, -1, :]  # (batch, hidden_dim*2)
        
        # 6. Clasificación
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return logits
    
    def predict(self, x, return_probabilities=False):
        """
        Predice la clase y retorna probabilidades.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            if return_probabilities:
                return predicted_classes, probabilities
            return predicted_classes