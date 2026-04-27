# ================================================================================
# ARQUITECTURA DEL PROYECTO CON RED NEURONAL
# ================================================================================

# ESTRUCTURA DE ARCHIVOS:
# -----------------------

# word_finder_model/
# │
# ├── src/                              [Código fuente]
# │   ├── word_finder.py                
# │   └── train_model.py                
# │
# ├── data/                             [Datos y modelos]
# │   └── word_finder_model.pkl         
# │
# ├── training/                         [Datos de entrenamiento]
# │   ├── key_words_labels.json         
# │   └── word_finder_training_v1.json  
# │
# ├── scripts/                          [Scripts de utilidad]
# │   └── generate_model.py             
# │
# ├── test/                             [Tests]
# │   └── test_model.py                
# │
# └── [otros archivos sin cambios]
