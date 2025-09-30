# test/test_neural.py
import os
import sys
import logging
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(filename)s:%(lineno)d %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hybrid_finder import HybridWordFinder

def test_neural_model():
    """Prueba el modelo neuronal con casos de prueba."""

    model_path = os.path.join(PROJECT_ROOT, "data", "word_finder_model.pkl")
    neural_model_path = os.path.join(PROJECT_ROOT, "data", "neural_model.pth")
    
    logger.info("="*80)
    logger.info("TEST DEL MODELO NEURONAL")
    logger.info("="*80)
    
    # Verificar que existe el modelo neuronal
    if not os.path.exists(neural_model_path):
        logger.error(f"Modelo neuronal no encontrado: {neural_model_path}")
        logger.info("Ejecuta primero: python scripts/train_neural_model.py")
        return
    
    # Cargar sistema híbrido
    logger.info("Cargando sistema híbrido...")
    hwf = HybridWordFinder(
        model_path=model_path,
        project_root=PROJECT_ROOT,
        neural_model_path=neural_model_path,
        use_neural=True,
        neural_threshold=0.7
    )
    
    # Información del modelo
    info = hwf.get_model_info()
    logger.info("\nInformación del sistema:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # Casos de prueba
    test_cases = [
        # Casos exactos
        ("total", "MontoTotalDocumento"),
        ("iva", "MontoIVAGeneral"),
        ("fecha", "FechaDocumento"),
        ("cliente", "NombreCliente"),
        ("folio", "FolioDocumento"),
        
        # Casos con errores de OCR
        ("totai", "MontoTotalDocumento"),
        ("t0tal", "MontoTotalDocumento"),
        ("tota1", "MontoTotalDocumento"),
        ("fecna", "FechaDocumento"),
        ("c1iente", "NombreCliente"),
        ("fo1io", "FolioDocumento"),
        
        # Casos con ruido
        ("total123", "MontoTotalDocumento"),
        ("iva...", "MontoIVAGeneral"),
        
        # Casos que NO deberían matchear
        ("xyz123", None),
        ("random", None),
        ("lorem ipsum", None),
    ]
    
    logger.info("\n" + "="*80)
    logger.info("RESULTADOS DE PRUEBAS")
    logger.info("="*80)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected_field in test_cases:
        results = hwf.find_keywords(query)
        
        if results and len(results) > 0:
            result = results[0]
            predicted_field = result.get('key_field')
            confidence = result.get('similarity', 0)
            method = result.get('method', 'traditional')
            
            is_correct = (predicted_field == expected_field)
            status = "✓" if is_correct else "✗"
            
            if is_correct:
                correct += 1
            
            logger.info(
                f"{status} Query: '{query:20s}' -> "
                f"Predicho: {predicted_field:25s} "
                f"(conf: {confidence:.3f}, método: {method:10s}) "
                f"Esperado: {expected_field}"
            )
        else:
            # No se encontró match
            is_correct = (expected_field is None)
            status = "✓" if is_correct else "✗"
            
            if is_correct:
                correct += 1
            
            logger.info(
                f"{status} Query: '{query:20s}' -> "
                f"Predicho: {'NO_MATCH':25s} "
                f"Esperado: {expected_field}"
            )
    
    accuracy = 100.0 * correct / total
    logger.info("\n" + "="*80)
    logger.info(f"RESULTADO FINAL: {correct}/{total} correctos ({accuracy:.1f}%)")
    logger.info("="*80)

if __name__ == "__main__":
    try:
        test_neural_model()
    except Exception as e:
        logger.error(f"Error en test: {e}", exc_info=True)
        sys.exit(1)
