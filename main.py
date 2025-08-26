import os
import logging
from logging.handlers import RotatingFileHandler
from scripts.generate_model import ModelGenerator

def configure_logging():
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter("%(levelname)s %(name)s: %(message)s")

    root = logging.getLogger()
    # evitar añadir handlers duplicados si se llama varias veces
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        root.addHandler(sh)

        log_dir = os.path.join(os.path.dirname(__file__), "data", "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(os.path.join(log_dir, "app.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        root.addHandler(fh)

    root.setLevel(level)

if __name__ == "__main__":
    configure_logging()
    logger = logging.getLogger(__name__)
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    CONFIG_FILE = os.path.join(PROJECT_ROOT, "data", "config.yaml")
    try:
        generator = ModelGenerator(CONFIG_FILE, PROJECT_ROOT)
        generator.generate_model(CONFIG_FILE, PROJECT_ROOT)
        logger.info("Proceso terminado correctamente.")
    except Exception:
        logger.exception("Error en el proceso de generación del modelo.")