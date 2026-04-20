import os
import sys
import logging
from scripts.generate_model import ModelGenerator
from cache_service import  cleanup_project_cache

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

logger = logging.getLogger(__name__)

CONSOLE_LEVEL = "INFO"
FILE_LEVEL = "INFO"
CONSOLE_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(message)s"
FILE_FORMAT = "%(asctime)s - %(module)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%H:%M:%S"  # Solo horas:minutos:segundos en formato 00:00:00

logger_root = logging.getLogger()
logger_root.setLevel(logging.DEBUG)

if logger_root.hasHandlers():
    logger_root.handlers.clear()
file_formatter = logging.Formatter(
    fmt=FILE_FORMAT,
    datefmt=DATE_FORMAT
)
console_formatter = logging.Formatter(
    fmt=CONSOLE_FORMAT,
    datefmt=DATE_FORMAT
)
try:
    LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "wfm.txt")
    if os.path.exists(LOG_FILE_PATH):    
        file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(FILE_LEVEL.upper())
        logger_root.addHandler(file_handler)
        
except FileNotFoundError as e:
    logger.warning(f"Error generando archivo log: '{e}'", exc_info=True)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(CONSOLE_LEVEL.upper())
logger_root.addHandler(console_handler)
    
if __name__ == "__main__":
    
    CONFIG_FILE = os.path.join(PROJECT_ROOT, "data", "config.yaml")
    KEY_WORDS = os.path.join(PROJECT_ROOT, "data", "key_words.json")
    cleanup_project_cache(PROJECT_ROOT)
    try:
        config_file = CONFIG_FILE
        key_words_file = KEY_WORDS
        generator = ModelGenerator(config_file, PROJECT_ROOT, key_words_file)
        cleanup_project_cache(PROJECT_ROOT)
        logger.info("Proceso terminado correctamente.")
    except Exception as e:
        logger.error("Error en el proceso de generación del modelo: {e}", exc_info=True)