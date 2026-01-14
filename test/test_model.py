import os
import random
import sys
import pickle
import logging
import time
import json
import glob
from typing import List, Dict, Any, Tuple, Optional

logging.basicConfig(
    level=logging.INFO, 
    format='%(filename)s:%(lineno)d %(message)s'
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cache_service import cleanup_project_cache
cleanup_project_cache(PROJECT_ROOT)

from src.word_finder import WordFinder
MODEL_STD = os.path.join(PROJECT_ROOT, "models", "wf_model.pkl")
DATA_FOLDER = os.path.join(PROJECT_ROOT, "input")
DATA_FOLDER2 = os.path.join(PROJECT_ROOT, "input2")

try:    
    wf: WordFinder = WordFinder(MODEL_STD)
except Exception as e:
    logger.info(f"Error estableciendo root: {e}", exc_info=True)

try:
    with open(MODEL_STD, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    logger.info(f"Error: {e}", exc_info=True)
base_queries: List[str] = [
    "ticketrazon", "preciocantidad", "detalleconcepto", "referenciaproducto",
    "servicioprecio", "subtotalhora", "cantidadsku", "importemodelo",
    "totalservicio", "fechadescripcion", "articulofolio", "marcaconcepto",
    "ivaarticulo", "razonsocialticket", "preciounitariocodigo", "ticketxyz",
    "totalxpto", "fechaclienteabc", "importetest", "sku", "modeloprueba",
    "preciounitario", "cantidadbar", "serviciolorem", "detalletest",
    "code", "puntuacion", "puntualidad", "estudiante", "italiano", "punt",
    "puntillas", "pun", "puesto", "amarillo", "punct", "punto", "ano", "titulo"
]

base_queries2: List[str] = [
    "ticket razon", "precio cantidad", "detalle concepto", "referencia producto",
    "servicio precio", "subtotal hora", "cantidad sku", "importe modelo",
    "total servicio", "fecha descripcion", "articulo folio", "marca concepto",
    "iva articulo", "razon social ticket", "precio unitario codigo", "ticket xyz",
    "total xpto", "fecha cliente abc", "importe test", "sku", "modelo prueba",
    "precio unitario", "cantidad bar", "servicio lorem", "detalle test",
    "code", "puntuacion", "puntualidad", "estudiante", "italiano", "punt",
    "puntillas", "pun", "puesto", "amarillo", "punct", "punto", "ano", "titulo"
]

# Perturbaciones
def delete_char(s: str) -> str:
    if len(s) <= 2: return s
    i = random.randrange(len(s))
    return s[:i] + s[i+1:]

def swap_chars(s: str) -> str:
    if len(s) <= 2: return s
    i = random.randrange(len(s)-1)
    lst = list(s)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return "".join(lst)

def replace_char(s: str) -> str:
    if not s: return s
    i = random.randrange(len(s))
    return s[:i] + random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+-=[]{}|;':\",./<>?`~áéíóúàèìòùâêîôûäëïöüñçÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÑÇ") + s[i+1:]

def perturb(s: str) -> str:
    ops = [delete_char, swap_chars, replace_char]
    f = random.choice(ops)
    return f(s)

# Generar queries con ruido
random.seed(42)
text: List[str] = []
for q in base_queries and base_queries2:
    text.append(q)
    for _ in range(3):
        text.append(perturb(q))

def log_model_summary(wf: WordFinder):
    try:
        logger.info("Resumen del modelo WordFinder")
        logger.info(f"Total de palabras clave: {len(wf.global_words)}")
        logger.info(f"Rango de n-gramas: {wf.ngrams}")
        logger.info(f"Umbral de similitud: {wf.threshold}")
    except Exception as e:
        logger.error(f"Error en logg: {e}", exc_info=True)

def log_search_results(res: Optional[List[Dict[str, Any]]], q: List[str], params: Dict[str, Any]):
    time0 = time.perf_counter()
    try:
        if not res:
            return None
        for i, res in enumerate(res):
            logger.debug(f"Campo: {res.get('key_field')}, Palabra: '{res.get('word_found')}', Similitud: {res.get('similarity'):.6}, Query: '{q}'")
        tiempo = time.perf_counter() - time0
        logger.debug(f"Tiempo total del tester: {tiempo}")
    except Exception as e:
        logger.info(f"Error en log: {e}", exc_info=True)

def run_queries(base_queries2: List[str], wf: WordFinder, show_no_match: bool = True):
    num_matches = 0
    num_no_matches = 0
    matches: List[Tuple[str, Dict[str, Any]]] = []
    no_matches: List[str] = []

    time0 = time.perf_counter()
    try:
        for q in base_queries2:
            res: Optional[List[Dict[str, Any]]] = wf.find_keywords(q)
            if res:
                num_matches += 1
                for r in (res if isinstance(res, list) else [res]):
                    key_field = r.get("key_field")
                    params = r.get("params")
                    word_found = r.get("word_found")
                    score = r.get("similarity")
                    matches.append((q, r))
                log_search_results(res, q, params)
            else:
                num_no_matches += 1
                no_matches.append(q)
                if show_no_match:
                    logger.info(f"QUERIES1: No match para: '{q}'")
    except Exception as e:
        logger.error(f"Error en RUN QUERIES1: : {e}", exc_info=True)

    porcentaje: float = (100.00/len(base_queries)) * num_matches
    logger.info(f"QUERIES1: Resumen final: {num_matches}/{len(base_queries)} matches")
    logger.info(f"QUERIES1: Porcentaje de coincidencia de palabras: {porcentaje:.2f}%")
    logger.info(f"QUERIES1: Total sin match: {num_no_matches}")
    logger.info(f"QUERIES1: Tiempo total: {time.perf_counter()-time0:.2f}s")

def run_queries2(base_queries2: List[str], wf: WordFinder, show_no_match: bool = True):
    num_matches = 0
    num_no_matches = 0
    matches: List[Tuple[str, Dict[str, Any]]] = []
    no_matches: List[str] = []

    time0 = time.perf_counter()
    try:
        for q in base_queries2:
            res: Optional[List[Dict[str, Any]]] = wf.find_keywords(q)
            if res:
                num_matches += 1
                for r in (res if isinstance(res, list) else [res]):
                    key_field = r.get("key_field")
                    params = r.get("params")
                    word_found = r.get("word_found")
                    score = r.get("similarity")
                    matches.append((q, r))

                log_search_results(res, q, params)
            else:
                num_no_matches += 1
                no_matches.append(q)
                if show_no_match:
                    logger.debug(f"No match QUERIES2: '{q}'")
    except Exception as e:
        logger.error(f"Error en RUN QUERIES: {e}", exc_info=True)

    porcentaje: float = (100.00/len(base_queries2)) * num_matches
    logger.info(f"QUERIES2: Resumen final: {num_matches}/{len(base_queries2)}matches")
    logger.info(f"QUERIES2: Porcentaje de coincidencia de palabras: {porcentaje:.2f}%")
    logger.info(f"QUERIES2: Total sin match: {num_no_matches}")
    logger.info(f"QUERIES: Tiempo total: {time.perf_counter()-time0:.4f}s")
    
def test_json_poligons(wf: WordFinder, DATA_FOLDER2: str):
    logger.info(f"\nBuscando archivos JSON individuales en la carpeta {DATA_FOLDER2}...")
    json_files = glob.glob(os.path.join(DATA_FOLDER2, '*.json'))

    if not json_files:
        logger.error(f"No se encontraron archivos JSON en {DATA_FOLDER2}.")
        return

    total_matches = 0
    total_words_processed = 0
    proceced_files = 0
    time0 = time.perf_counter()
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        logger.info(f"\n--- Procesando archivo: {file_name} ---")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            polygons: Dict[str, Dict[str, Any]] = json.load(f)
            matches_in_doc = 0
            time1 = time.perf_counter()
            for poly_id, poly_data in polygons.items():
                if not poly_data:
                    continue
                text = poly_data.get("text", "")
                if text and text.strip():
                    results = wf.find_keywords(text.strip())
                    if results:
                        matches_in_doc += len([r for r in results if r])
                        for result in results:
                            if result and len(result) > 0:
                                logger.info(f"MATCH: {poly_id} '{text}' | {result}")

            logger.info(f"Total de matches por documento: {matches_in_doc} / {len(polygons.items())} en {time.perf_counter() - time1:.6f}s")

            if matches_in_doc == 0:
                logger.info("No se encontraron coincidencias en este documento")

        total_matches += matches_in_doc
        total_words_processed += len(polygons.items())
        proceced_files +=1
        # Resumen final
    logger.info(f"\n=== RESUMEN FINAL ===")
    logger.info(f"Archivos procesados: {proceced_files}")
    logger.info(f"Total coincidencias: {total_matches} / {total_words_processed}")
    if total_words_processed > 0:
        porcentaje = (total_matches / total_words_processed) * 100
        logger.info(f"Porcentaje de coincidencias: {porcentaje:.2f}%")
    total_time = time.perf_counter() - time0
    logger.info(f"Tiempo total: {total_time:.4}s, tiempo promedio: {total_time/proceced_files:.4f}s")
        
def test_json_lines(wf: WordFinder, DATA_FOLDER2: str):
    try: 
        time0 = time.perf_counter()
        logger.info(f"\nBuscando archivos JSON en la carpeta {DATA_FOLDER2}...")
        json_files = glob.glob(os.path.join(DATA_FOLDER2, '*.json'))

        if not json_files:
            logger.error(f"No se encontraron archivos JSON en {DATA_FOLDER2}.")
            return

        total_matches = 0
        total_lines_processed = 0
        
        for file_path in json_files:
            logger.info(f"\n--- Procesando archivo: {os.path.basename(file_path)} ---")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines: Dict[str, Dict[str, Any]] = json.load(f)

                # Extraer todos los textos del documento
                all_text: List[str] = []
                line_ids: List[str] = []
                
                for line_id, line_data in lines.items():
                    text = line_data.get("text", "")
                    if text and text.strip():  # Solo textos no vacíos
                        all_text.append(text.strip())
                        line_ids.append(line_id)
                
                if not all_text:
                    logger.warning(f"No se encontró texto válido en {os.path.basename(file_path)}")
                    continue
                
                logger.info(f"Líneas con texto: {len(all_text)}")
                
                time_doc = time.perf_counter()
                results = wf.find_keywords(all_text)
                time_doc_end = time.perf_counter()
                
                # Contar y mostrar resultados
                if results:
                    matches_in_doc = len([r for r in results if r])
                    total_matches += matches_in_doc
                    
                    logger.debug(f"Coincidencias encontradas: {matches_in_doc}/{len(all_text)}")
                    
                    # Mostrar solo las coincidencias más relevantes
                    try:
                        for i, result in enumerate(results):
                            if result and len(result) > 0:
                                logger.info(f"Resultado: {line_ids[i]}: {result}")
                                # logger.info(f"{line_ids[i]}: '{all_text[i][:30]}...' -> {result.get('field', 'N/A')} ({result.get('similarity', 0):.3f})")
                                
                    except Exception as e:
                        logger.error(f"Errpr buscando mejor match: {e}", exc_info=True)
                else:
                    logger.info("No se encontraron coincidencias en este documento")
                
                total_lines_processed += len(all_text)
        
        # Resumen final
        logger.info(f"\n=== RESUMEN FINAL JSON ===")
        logger.info(f"Archivos procesados: {len(json_files)}")
        logger.info(f"Total líneas procesadas: {total_lines_processed}")
        logger.info(f"Total coincidencias: {total_matches}")
        if total_lines_processed > 0:
            porcentaje = (total_matches / total_lines_processed) * 100
            logger.info(f"Porcentaje de coincidencias: {porcentaje:.2f}%")
        logger.info(f"Tiempo total: {time.perf_counter() - time0:.4f}s")
        
    except Exception as e:
        logger.error(f"Error testeando: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    time0 = time.perf_counter()
    wf = WordFinder(MODEL_STD)
    # log_model_summary(wf)
    #try:
     #    test_json_lines(wf, DATA_FOLDER)
    #except Exception as e:
     #    logger.error(f"Error testeando: {e}", exc_info=True)

#    try:
 #      test_json_poligons(wf, DATA_FOLDER2)
  #  except Exception as e:
   #     logger.error(f"Error testeando: {e}", exc_info=True)

    # logger.info("=====TEST DE QUERIES SIN ESPACIAR INCIADO=====")
    # run_queries(base_queries, wf)
    # logger.info(f"TIEMPO TEST 1: {time.perf_counter()-time0:.6f}")
    logger.info("=====TEST DE QUERIES2 CON ESPACIOS INCIADO=====")
    run_queries2(base_queries2, wf)
    logger.info(f"TIEMPO TEST 2: {time.perf_counter()-time0:.6f}")

        # Prueba con diferentes inputs
#    text = ["total", "iva", "rfc", "folio", "cliente", "fecha", "subtotal", "encabezados"]
        # total_r = 0
        # for q in text:
        #     result = wf.find_keywords(q)
        #     total_r += 1
        #     if result:
        #         logger.debug(f"Texto'{q}': {result}")
        # logger.info(f"Total: {total_r}")
        # logger.warning("error")

        # # logger.debug("TESTING EXACT RETURN VALUES FROM WordFinder.find_keywords()")

        # # Test detallado de estructura completa
        # detailed_result = wf.find_keywords(base_queries)
        # logger.debug(
        #     f"\n\n4. DETAILED STRUCTURE ANALYSIS:\nBase queries: {base_queries}"
        #     f"\nResult for 'total': {detailed_result}"
        # )
        # if detailed_result:
        #     item = detailed_result[0]
        #     logger.debug(
        #         f"\nDetailed analysis of first result item:\n"
        #         f"Keys count: {len(item.keys())}\n"
        #         f"Keys: {list(item.keys())}\n"
        #         f"Items:"
        #     )

        # logger.debug("\n" + "="*80 + "\nEND OF RETURN VALUE TESTING\n" + "="*80)
    logger.info(f"Testing acabado en: {time.perf_counter()-time0:.6f}s")
        # logger.debug("\n5. USING DEBUG METHOD:\n" + "-" * 60)
        # cleanup_project_cache(PROJECT_ROOT)
