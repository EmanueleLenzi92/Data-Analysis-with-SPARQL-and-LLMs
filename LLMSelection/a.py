import os
import re
import json
from collections import defaultdict

GOLD_PATH = "gold.json"
PRED_FILENAME = "a.json"
QUERY_DIR_RE = re.compile(r"^Q[0-9A-Za-z]+$")  # es. Q1, Q3, Q23, Q6A, ecc.

def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  File non trovato: {path}")
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON malformato: {path} -> {e}")
    except Exception as e:
        print(f"⚠️  Errore leggendo {path}: {e}")
    return None

def validate_preds(obj, path):
    """
    accetta o corregge parzialmente:
    - deve essere una lista
    - ogni item deve essere un dict con chiave 'id' (stringa)
    - scarta gli item non validi
    """
    valid_ids = []
    if not isinstance(obj, list):
        print(f"⚠️  {path}: atteso array JSON, trovato {type(obj).__name__}. Ignoro file.")
        return set()
    for i, it in enumerate(obj):
        if not isinstance(it, dict):
            print(f"⚠️  {path}: elemento #{i} non è un oggetto. Scartato.")
            continue
        if "id" not in it:
            print(f"⚠️  {path}: elemento #{i} senza chiave 'id'. Scartato.")
            continue
        _id = it["id"]
        if not isinstance(_id, str):
            print(f"⚠️  {path}: 'id' non stringa in elemento #{i}. Convertito a stringa.")
            _id = str(_id)
        valid_ids.append(_id)
    return set(valid_ids)

def build_gold_by_query(gold, restrict_to_queries=None):
    """
    Crea mapping {query: set(ids)}.
    - Se restrict_to_queries è fornito, include solo quelle query.
    - Altrimenti include TUTTE le query presenti nel gold.
    - Gli item senza 'query' valida vengono ignorati.
    """
    gold_by_q = defaultdict(set)
    for i, item in enumerate(gold):
        if not isinstance(item, dict):
            continue
        _id = item.get("id")
        if _id is None:
            continue
        if not isinstance(_id, str):
            _id = str(_id)
        qlist = item.get("query", [])
        if not isinstance(qlist, list):
            continue
        for q in qlist:
            if not isinstance(q, str):
                continue
            if restrict_to_queries is not None and q not in restrict_to_queries:
                continue
            gold_by_q[q].add(_id)
    return dict(gold_by_q)

def compute_metrics(pred_ids, gold_ids):
    tp = len(pred_ids & gold_ids)
    fp = len(pred_ids - gold_ids)
    fn = len(gold_ids - pred_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "TP": tp, "FP": fp, "FN": fn,
        "pred_size": len(pred_ids),
        "gold_size": len(gold_ids),
    }

def discover_query_model_paths(root="."):
    """
    Rileva tutte le directory Q*/<model>/a.json.
    Ritorna: dict { query: { model: path_to_a_json } }
    """
    mapping = defaultdict(dict)
    for entry in os.listdir(root):
        qpath = os.path.join(root, entry)
        if not os.path.isdir(qpath):
            continue
        if not QUERY_DIR_RE.match(entry):  # deve essere Q*
            continue
        query = entry
        # sottocartelle = modelli
        for model in os.listdir(qpath):
            mpath = os.path.join(qpath, model)
            if not os.path.isdir(mpath):
                continue
            apath = os.path.join(mpath, PRED_FILENAME)
            if os.path.exists(apath):
                mapping[query][model] = apath
    return dict(mapping)

def main():
    # 1) Scopri (query, modello) dal filesystem
    q_model_paths = discover_query_model_paths(".")
    if not q_model_paths:
        print("⚠️  Nessun path trovato del tipo Q*/<modello>/a.json.")
    all_queries_fs = set(q_model_paths.keys())

    # 2) Carica gold
    gold = load_json_safe(GOLD_PATH)
    if not isinstance(gold, list):
        print("❌  gold.json mancante o non valido. Interrompo.")
        return

    # 3) Costruisci gold_by_query (solo per query viste sul FS o tutte? -> usiamo le viste sul FS)
    gold_by_query = build_gold_by_query(gold, restrict_to_queries=all_queries_fs)

    # 4) Valuta per ogni (query, modello)
    per_query_model = []
    model_agg_micro = defaultdict(lambda: {"TP":0,"FP":0,"FN":0})
    model_queries_seen = defaultdict(list)  # per macro-avg

    for query, model_paths in sorted(q_model_paths.items()):
        gold_ids = gold_by_query.get(query, set())
        for model, apath in sorted(model_paths.items()):
            preds_raw = load_json_safe(apath)
            pred_ids = validate_preds(preds_raw, apath)

            # NB: recall è solo verso gold_ids di quella query
            m = compute_metrics(pred_ids, gold_ids)
            per_query_model.append({
                "query": query,
                "model": model,
                **m
            })

            # micro-agg per modello (sommo TP/FP/FN su tutte le query viste per quel modello)
            model_agg_micro[model]["TP"] += m["TP"]
            model_agg_micro[model]["FP"] += m["FP"]
            model_agg_micro[model]["FN"] += m["FN"]
            model_queries_seen[model].append((query, m))

    # 5) Calcola aggregati per modello
    per_model_micro = []
    per_model_macro = []
    for model, agg in model_agg_micro.items():
        TP, FP, FN = agg["TP"], agg["FP"], agg["FN"]
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall    = TP / (TP + FN) if (TP + FN) else 0.0
        f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
        per_model_micro.append({
            "model": model,
            "precision": round(precision,6),
            "recall": round(recall,6),
            "f1": round(f1,6),
            "TP": TP, "FP": FP, "FN": FN
        })

        # macro-avg (media semplice delle metriche sulle query valutate per quel modello)
        if model_queries_seen[model]:
            p_list = [x[1]["precision"] for x in model_queries_seen[model]]
            r_list = [x[1]["recall"]    for x in model_queries_seen[model]]
            f_list = [x[1]["f1"]        for x in model_queries_seen[model]]
            per_model_macro.append({
                "model": model,
                "precision_macro": round(sum(p_list)/len(p_list),6),
                "recall_macro": round(sum(r_list)/len(r_list),6),
                "f1_macro": round(sum(f_list)/len(f_list),6),
                "queries_count": len(p_list)
            })

    # 6) Salva risultati
    out = {
        "per_query_model": per_query_model,
        "per_model_micro": per_model_micro,
        "per_model_macro": per_model_macro,
        "info": {
            "gold_path": GOLD_PATH,
            "pred_filename": PRED_FILENAME,
            "discovered_queries": sorted(list(all_queries_fs)),
        }
    }
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 7) Stampa riepilogo leggibile

    print("\n📊 RISULTATI PER QUERY & MODELLO (ordinati per F1 decrescente)")
    for r in sorted(per_query_model, key=lambda x: x["f1"], reverse=True):
        print(f"{r['query']:<6} | {r['model']:<20} → "
              f"P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  "
              f"(TP={r['TP']}, FP={r['FP']}, FN={r['FN']}, "
              f"pred={r['pred_size']}, gold={r['gold_size']})")

    print("\n🧮 AGGREGATI PER MODELLO (micro-average, ordinati per F1 decrescente)")
    for r in sorted(per_model_micro, key=lambda x: x["f1"], reverse=True):
        print(f"{r['model']:<20} → P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  "
              f"(TP={r['TP']}, FP={r['FP']}, FN={r['FN']})")

    print("\n📐 AGGREGATI PER MODELLO (macro-average, ordinati per F1̄ decrescente)")
    for r in sorted(per_model_macro, key=lambda x: x["f1_macro"], reverse=True):
        print(f"{r['model']:<20} → P̄={r['precision_macro']:.3f}  R̄={r['recall_macro']:.3f}  "
              f"F1̄={r['f1_macro']:.3f}  (queries={r['queries_count']})")


    print("\n✅ Salvato: evaluation_results.json")


main()
