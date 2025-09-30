# src/hybrid_finder.py
import os
import torch
import pickle
import logging
from typing import List, Dict, Any, Optional
from src.word_finder import WordFinder
from src.neural_classifier import OCRKeywordClassifier

logger = logging.getLogger(__name__)

class HybridWordFinder(WordFinder):
    """
    Sistema híbrido que combina el sistema tradicional de n-gramas
    con una red neuronal para mayor robustez ante errores de OCR.
    """
    
    def __init__(self, model_path: str, project_root: str, neural_model_path: Optional[str] = None, 
                 use_neural: bool = True, neural_threshold: float = 0.7):
        """
        Args:
            model_path: Ruta al modelo tradicional (pickle)
            project_root: Ruta raíz del proyecto
            neural_model_path: Ruta al modelo neuronal (opcional)
            use_neural: Si usar la red neuronal
            neural_threshold: Umbral de confianza para predicciones neuronales
        """
        super().__init__(model_path, project_root)
        
        self.use_neural = use_neural
        self.neural_threshold = neural_threshold
        self.neural_model = None
        self.char_to_idx = None
        self.idx_to_label = None
        self.max_len = 50
        
        # Cargar modelo neuronal si está disponible
        if use_neural and neural_model_path and os.path.exists(neural_model_path):
            self._load_neural_model(neural_model_path)
        else:
            if use_neural:
                logger.warning(f"Modelo neuronal no encontrado en {neural_model_path}. Usando solo sistema tradicional.")
            self.use_neural = False
    
    def _load_neural_model(self, neural_model_path: str):
        """Carga el modelo neuronal y sus metadatos."""
        try:
            checkpoint = torch.load(neural_model_path, map_location='cpu')
            
            # Cargar metadatos
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_label = checkpoint['idx_to_label']
            vocab_size = checkpoint['vocab_size']
            num_classes = checkpoint['num_classes']
            self.max_len = checkpoint.get('max_len', 50)
            
            # Inicializar y cargar modelo
            self.neural_model = OCRKeywordClassifier(
                vocab_size=vocab_size,
                num_classes=num_classes
            )
            self.neural_model.load_state_dict(checkpoint['model_state_dict'])
            self.neural_model.eval()
            
            logger.info(f"Modelo neuronal cargado exitosamente desde {neural_model_path}")
            logger.info(f"  Vocab size: {vocab_size}, Num classes: {num_classes}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo neuronal: {e}", exc_info=True)
            self.use_neural = False
    
    def _neural_predict(self, text: str) -> Optional[Dict[str, Any]]:
        """Predice usando la red neuronal."""
        if not self.use_neural or self.neural_model is None:
            return None
        
        try:
            # Normalizar texto
            normalized = self._normalize(text)
            if not normalized:
                return None
            
            # Convertir a índices
            char_indices = [self.char_to_idx.get(c, self.char_to_idx.get('<UNK>', 1)) 
                          for c in normalized[:self.max_len]]
            
            # Padding
            if len(char_indices) < self.max_len:
                char_indices += [0] * (self.max_len - len(char_indices))
            
            # Inferencia
            x = torch.tensor([char_indices], dtype=torch.long)
            with torch.no_grad():
                logits = self.neural_model(x)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Verificar umbral
            if confidence < self.neural_threshold:
                return None
            
            # Obtener label
            key_field = self.idx_to_label.get(predicted_class)
            if key_field == "NO_MATCH":
                return None
            
            return {
                "key_field": key_field,
                "word_found": normalized,
                "similarity": confidence,
                "query": text,
                "method": "neural"
            }
            
        except Exception as e:
            logger.error(f"Error en predicción neuronal: {e}", exc_info=True)
            return None
    
    def find_keywords(self, text: List[str] | str, threshold: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Busca palabras clave usando sistema híbrido.
        
        Estrategia:
        1. Intenta con sistema tradicional (rápido)
        2. Si confianza alta, devuelve resultado tradicional
        3. Si no hay resultado o confianza baja, usa red neuronal
        4. Devuelve el mejor resultado
        """
        # Sistema tradicional
        traditional_results = super().find_keywords(text, threshold)
        
        # Si es batch, procesar cada texto
        if isinstance(text, list):
            results = []
            for i, single_text in enumerate(text):
                trad_result = traditional_results[i] if traditional_results and i < len(traditional_results) else None
                hybrid_result = self._hybrid_decision(single_text, trad_result)
                results.append(hybrid_result if hybrid_result else [])
            return results
        
        # Procesamiento individual
        return self._hybrid_decision(text, traditional_results)
    
    def _hybrid_decision(self, text: str, traditional_result: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Decide qué resultado usar: tradicional o neuronal."""
        
        # Si hay resultado tradicional con alta confianza, usarlo
        if traditional_result and len(traditional_result) > 0:
            if traditional_result[0].get('similarity', 0) > 0.9:
                return traditional_result
        
        # Si no hay red neuronal, devolver resultado tradicional
        if not self.use_neural:
            return traditional_result if traditional_result else []
        
        # Intentar con red neuronal
        neural_result = self._neural_predict(text)
        
        # Decidir cuál usar
        if neural_result:
            neural_conf = neural_result.get('similarity', 0)
            trad_conf = traditional_result[0].get('similarity', 0) if traditional_result else 0
            
            # Usar el de mayor confianza
            if neural_conf > trad_conf:
                return [neural_result]
            elif traditional_result:
                return traditional_result
            else:
                return [neural_result]
        
        return traditional_result if traditional_result else []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Información del sistema híbrido."""
        info = super().get_model_info()
        info['neural_enabled'] = self.use_neural
        info['neural_threshold'] = self.neural_threshold
        if self.use_neural and self.neural_model:
            info['neural_vocab_size'] = len(self.char_to_idx)
            info['neural_num_classes'] = len(self.idx_to_label)
        return info