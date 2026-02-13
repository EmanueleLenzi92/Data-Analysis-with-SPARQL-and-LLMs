import os
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

#Query da valuatre
QID= "Q4"
#cartella modello
cartellaMod= "ragcontesto/"


# === CONFIGURAZIONE ===
# cartella con i gold standard
GOLD_FOLDER = "goldStandardQueries"


# scegli quale gold usare (cambia il nome qui)
GOLD_FILE = QID + ".json"  # <-- puoi cambiare questo nome per scegliere un altro gold
# cartella con i modelli
MODELS_FOLDER = cartellaMod + QID

# === FUNZIONI ===
def load_ids_from_json(file_path):
    """Carica gli id da un file json, gestendo sia liste che dict."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []
    if isinstance(data, list):
        for elem in data:
            if isinstance(elem, dict) and "id" in elem:
                ids.append(str(elem["id"]))
            elif isinstance(elem, list):
                for sub in elem:
                    if isinstance(sub, dict) and "id" in sub:
                        ids.append(str(sub["id"]))
    elif isinstance(data, dict):
        if "id" in data:
            ids.append(str(data["id"]))
    return ids

# === CARICO GOLD ===
gold_path = os.path.join(GOLD_FOLDER, GOLD_FILE)
gold_ids = set(load_ids_from_json(gold_path))

# === SCORRO I MODELLI ===
results = []
for model_name in os.listdir(MODELS_FOLDER):
    model_path = os.path.join(MODELS_FOLDER, model_name, "a.json")
    if not os.path.exists(model_path):
        continue

    pred_ids = set(load_ids_from_json(model_path))

    # creo vettori binari per sklearn
    y_true = [1 if x in gold_ids else 0 for x in gold_ids.union(pred_ids)]
    y_pred = [1 if x in pred_ids else 0 for x in gold_ids.union(pred_ids)]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    results.append({
        "modello": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# === MOSTRO RISULTATI ===
df = pd.DataFrame(results)
print(df)








