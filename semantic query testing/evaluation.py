#env -> eccolo, dentro chatgpt auto

import os
import json
import pandas as pd

base_dir = "movingJson/experiments"

records = []

for exp in os.listdir(base_dir):
    exp_path = os.path.join(base_dir, exp)
    report_path = os.path.join(exp_path, "report.json")
    if not os.path.exists(report_path):
        continue

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    params = report.get("params", {})
    for result in report.get("results", []):
        prompt = result.get("prompt")
        retrieved_ids = result.get("retrieved_ids", [])
        shortlist_size = result.get("shortlist_size", 0)

        records.append({
            "experiment": exp,
            "prompt": prompt,
            "retrieved_count": len(retrieved_ids),
            "shortlist_size": shortlist_size,
            "search_type": params.get("search_type"),
            "fetch_k": params.get("fetch_k"),
            "chunked": params.get("chunked"),
            "chunk_size": params.get("chunk_size"),
            "chunk_overlap": params.get("chunk_overlap"),
            "embed_model": params.get("embed_model"),
            "time_sec": report.get("time_sec"),
        })

df = pd.DataFrame(records)
print(f"📊 {len(df)} righe caricate da {len(df['experiment'].unique())} esperimenti.")

if df.empty:
    print("⚠️ Nessun dato trovato. Controlla che i report.json siano nel formato previsto.")
    exit()

# Raggruppa i risultati medi per esperimento e prompt
summary = (
    df.groupby(["experiment", "prompt"])
    .agg(
        avg_retrieved=("retrieved_count", "mean"),
        max_retrieved=("retrieved_count", "max"),
        search_type=("search_type", "first"),
        chunked=("chunked", "first"),
        embed_model=("embed_model", "first")
    )
    .reset_index()
)

# Ordina per performance media
summary = summary.sort_values(by="avg_retrieved", ascending=False)
print("\n=== 🔝 Esperimenti più performanti (media documenti recuperati per prompt) ===")
print(summary.head(10).to_string(index=False))

# Salva il riassunto in CSV
summary.to_csv("movingJson/experiments_summary.csv", index=False)
print("\n📁 Riassunto salvato in: movingJson/experiments_summary.csv")


# === 🔍 Analisi di coverage rispetto al gold standard ===

gold_path = "gold_standard.json"
if not os.path.exists(gold_path):
    print(f"\n⚠️ Nessun gold standard trovato in {gold_path}. Salto l'analisi di coverage.")
else:
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_standard = json.load(f)

    coverage_records = []

    for exp in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp)
        report_path = os.path.join(exp_path, "report.json")
        if not os.path.exists(report_path):
            continue

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        params = report.get("params", {})
        for result in report.get("results", []):
            prompt = result.get("prompt")
            retrieved_ids = set(result.get("retrieved_ids", []))
            gold_ids = set(gold_standard.get(prompt, []))

            if not gold_ids:
                continue

            matches = retrieved_ids.intersection(gold_ids)
            recall = len(matches) / len(gold_ids) * 100

            coverage_records.append({
                "experiment": exp,
                "prompt": prompt,
                "search_type": params.get("search_type"),
                "chunked": params.get("chunked"),
                "embed_model": params.get("embed_model"),
                "retrieved_count": len(retrieved_ids),
                "gold_count": len(gold_ids),
                "matches": len(matches),
                "recall_percent": round(recall, 2)
            })

    if not coverage_records:
        print("\n⚠️ Nessun dato di coverage trovato (controlla che i prompt del gold standard combacino).")
    else:
        df_cov = pd.DataFrame(coverage_records)
        print("\n=== 🧠 RISULTATI COVERAGE / RECALL ===")
        print(df_cov.sort_values(by="recall_percent", ascending=False).head(110).to_string(index=False))

        df_cov.to_csv("movingJson/experiments_recall.csv", index=False)
        print("\n📁 File di coverage salvato in: movingJson/experiments_recall.csv")
