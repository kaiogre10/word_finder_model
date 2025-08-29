import yaml
import math
from typing import Dict, List, Any
import unicodedata
import re 
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

config_file = os.path.join(ROOT, "data", "config.yaml")

with open(config_file, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
MAX_CHAR_COUNT = 20.0
MAX_MEAN = 113.0
    
def correct(palabra: str) -> str:
    if not palabra:
        return ""
    palabra = palabra.strip().lower()
    palabra = unicodedata.normalize("NFKD", palabra)
    palabra = "".join(ch for ch in palabra if not unicodedata.combining(ch))
    word = re.sub(r"[^a-zA-Z0-9]", "", palabra)
    return word
    
# Normalización global
def normaliza(val: float, minv: float, maxv: float) -> float:
    if maxv == minv:
        return 1.0
    return (val - minv) / (maxv - minv)

key_words = config["key_words"]
density_encoder = config["density_encoder"]
try:
# Recolectar todos los valores para normalización global
    field_stats: Dict[str, Dict[str, List[float]]] = {}
    word_stats: Dict[str, Dict[str, Any]] = {}

    for field, variants in key_words.items():
        field_stats[field] = {"char_count": [], "mean": [], "var": [], "DE": []}
        for palabra in variants:
            word = correct(palabra)
            if word in word_stats:
                continue
            logger.debug(f"Palabra normalizada-> {palabra} -> {word}")
            chars = list(word)
            values = [density_encoder.get(c, 0) for c in chars]
            char_count = len(word)
            char_count = float(char_count)
            addition = float(sum(values))
            mean = addition / char_count if char_count else 0
            var = sum((v - mean) ** 2 for v in values) / char_count if char_count else 0
            DE = math.sqrt(var)
            word_stats[word] = {
                "field": field,
                "palabra": word,
                "char_count": char_count,
                "mean": mean,
                "var": var,
                "DE": DE
            }
            
            field_stats[field]["char_count"].append(char_count)
            field_stats[field]["mean"].append(mean)
            field_stats[field]["var"].append(var)
            field_stats[field]["DE"].append(DE)
                
    print(f"{'Campo':20} | {'Palabra':25} | char_count | promedio | varianza | DE | char_count_n | promedio_n | varianza_n | DE_n")
    print("-"*140)
    for stat in word_stats.values():
        field = stat["field"]
        # Para cada feature, usa el min y max del campo correspondiente
        cc_min, cc_max = 0, MAX_CHAR_COUNT
        prom_min, prom_max = 0, MAX_MEAN
        var_min, var_max = min(field_stats[field]["var"]), max(field_stats[field]["var"])
        de_min, de_max = min(field_stats[field]["DE"]), max(field_stats[field]["DE"])
        cc_n = normaliza(stat["char_count"], cc_min, cc_max)
        prom_n = normaliza(stat["mean"], prom_min, prom_max)
        var_n = normaliza(stat["var"], var_min, var_max)
        de_n = normaliza(stat["DE"], de_min, de_max)
        print(f"{stat['field'][:20]:20} | {stat['palabra'][:25]:25} | {stat['char_count']:10} | {stat['mean']:8.2f} | {stat['var']:8.2f} | {stat['DE']:6.2f} | {cc_n:11.4f} | {prom_n:10.4f} | {var_n:10.2f} | {de_n:6.2f}")
        
except Exception as e:
    logger.info(f" Error{e}", exc_info=True)        
    