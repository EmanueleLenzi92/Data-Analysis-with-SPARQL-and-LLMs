import os
import shutil
import time
import gc
import re
import json
import subprocess
from typing import List
from contextlib import suppress

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

# RAG deps
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from chromadb.config import Settings

from pathlib import Path
HOME = str(Path.home())
DB_DIR = os.path.join(HOME, ".cache", "chroma", "movingjson_db")

# ============= CONFIG =============
TXT_PATH = "../MovingNarratives.txt"       # <-- file TXT già generato
BASE_OUTPUT_DIR = "../movingJson"           # cartella di output per JSON e log
DB_DIR = os.path.join(BASE_OUTPUT_DIR, "chroma_db")  # persistenza Chroma
NUM_CTX = 4096                           # contesto per Ollama
TEMPERATURE = 0.01                       # temperatura bassa per risposte più stabili

# --- RAG classico ON/OFF ---
USE_RAG_CONTEXT = False     # metti True per usare il Rag classico (contesto combinato in una sola chiamata)
RAG_MAX_DOCS = 7           # quanti documenti mettere nel contesto (taglia se serve dalla shortlisted)

# Modelli Ollama da iterare (modifica con i modelli disponibili su 'ollama list')
listllms = [
    "gemma3:12b-it-q8_0", "phi4:14b-q8_0", 
    "deepseek-r1:14b-qwen-distill-q8_0", "llama3.1:8b", "llama3:8b-instruct-q8_0",
    "gemma2:9b-instruct-q8_0",  "gemma3:4b-it-q8_0",
     "gemma2:2b", "deepseek-r1:1.5b-qwen-distill-fp16"
]


# Embedding model (via Ollama)
EMBED_MODEL = "nomic-embed-text" # es "nomic-embed-text"  "mxbai-embed-large" o "snowflake-arctic-embed" "embeddinggemma"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 8  # quanti chunk recuperare dal RAG

# ========== SYSTEM PROMPTS ==========


listPrompts = [
    # Prompt originale (PDO)
    """Given a list of narratives about mountain value chains, recognize if the mountain value chains involve cheese with Protected Designation of Origin (PDO) certification. If the value chains involve cheese with Protected Designation of Origin certification, write a json containing the list of the "ids" of the narratives. If the value chains don't involve cheese with Protected Designation of Origin certification, write an empty json.

### Example of Json (if the value chains involve cheese with Protected Designation of Origin certification)
{"ids": ["<ID_OF_THE_NARRATIVE>", "<ID_OF_THE_NARRATIVE>", ... ]}

### Example of Json (if the value chains don't involve cheese with Protected Designation of Origin certification)
{}""",

    # Prompt 1 ristrutturato
    """Given a list of narratives about mountain value chains, recognize if the mountain value chains are involved in producing cheese made with cow milk, goat milk, or a mix of both. If the value chains are involved in producing cheese made with cow milk, goat milk, or a mix of both, write a JSON containing the list of the "ids" of the narratives. If the value chains are not involved in producing cheese made with cow milk, goat milk, or a mix of both, write an empty JSON.

### Example of JSON (if the value chains are involved in producing cheese made with cow milk, goat milk, or a mix of both)
{"ids": ["<ID_OF_THE_NARRATIVE>", "<ID_OF_THE_NARRATIVE>", ... ]}

### Example of JSON (if the value chains are not involved in producing cheese made with cow milk, goat milk, or a mix of both)
{}""",

    # Prompt 2 ristrutturato
    """Given a list of narratives about mountain value chains, recognize if the mountain value chains are involved in using sheep to produce cheese. If the value chains are involved in using sheep to produce cheese, write a JSON containing the list of the "ids" of the narratives. If the value chains are not involved in using sheep to produce cheese, write an empty JSON.

### Example of JSON (if the value chains are involved in using sheep to produce cheese)
{"ids": ["<ID_OF_THE_NARRATIVE>", "<ID_OF_THE_NARRATIVE>", ... ]}

### Example of JSON (if the value chains are not involved in using sheep to produce cheese)
{}""",

    # Prompt 3 ristrutturato
    """Given a list of narratives about mountain value chains, recognize if the mountain value chains are involved in using sheep to produce wool. If the value chains are involved in using sheep to produce wool, write a JSON containing the list of the "ids" of the narratives. If the value chains are not involved in using sheep to produce wool, write an empty JSON.

### Example of JSON (if the value chains are involved in using sheep to produce wool)
{"ids": ["<ID_OF_THE_NARRATIVE>", "<ID_OF_THE_NARRATIVE>", ... ]}

### Example of JSON (if the value chains are not involved in using sheep to produce wool)
{}"""
]


listPrompts = [
"""Given a narrative about a mountain value chain, recognize if the mountain value chain involves cheese with Protected Designation of Origin certification (PDO) certification. If the value chain involves cheese with Protected Designation of Origin certification, write a JSON containing the "id" of the narrative. If the value chain doesn't involve cheese with Protected Designation of Origin certification, write an empty JSON.  

### Example of JSON (if the value chain involves cheese with Protected Designation of Origin certification)
{"id": "<ID_OF_THE_NARRATIVE>"}

### Example of JSON (if the value chain doesn't involve cheese with Protected Designation of Origin certification)
{}
""",
    """Given a narrative about a mountain value chain, recognize if the mountain value chain is involved in producing cheese made with cow milk, goat milk, or a mix of both. If the value chain is involved in producing cheese made with cow milk, goat milk, or a mix of both, write a JSON containing the "id" of the narrative. If the value chain is not involved in producing cheese made with cow milk, goat milk, or a mix of both, write an empty JSON.  

### Example of JSON (if the value chain is involved in producing cheese made with cow milk, goat milk, or a mix of both)
{"id": "<ID_OF_THE_NARRATIVE>"}

### Example of JSON (if the value chain is not involved in producing cheese made with cow milk, goat milk, or a mix of both)
{}
""",
    """Given a narrative about a mountain value chain, recognize if the mountain value chain is involved in using sheep to produce cheese. If the value chain is involved in using sheep to produce cheese, write a JSON containing the "id" of the narrative. If the value chain is not involved in using sheep to produce cheese, write an empty JSON.  

### Example of JSON (if the value chain is involved in using sheep to produce cheese)
{"id": "<ID_OF_THE_NARRATIVE>"}

### Example of JSON (if the value chain is not involved in using sheep to produce cheese)
{}
""",
    """Given a narrative about a mountain value chain, recognize if the mountain value chain is involved in using sheep to produce wool. If the value chain is involved in using sheep to produce wool, write a JSON containing the "id" of the narrative. If the value chain is not involved in using sheep to produce wool, write an empty JSON.  

### Example of JSON (if the value chain is involved in using sheep to produce wool)
{"id": "<ID_OF_THE_NARRATIVE>"}

### Example of JSON (if the value chain is not involved in using sheep to produce wool)
{}
"""

    
]




# ============= UTIL =============

# Regex: prende "ID NARRAZIONE: <id>" + corpo fino alla prossima "ID NARRAZIONE:" (preceduta da >= 2 newline) o fine file
NARRA_RE = re.compile(
    r'^\s*ID\s*NARRATIVE:\s*(?P<id>[^\r\n]+)\s*[\r\n]+(?P<body>.*?)(?=(?:\r?\n){2,}ID\s*NARRATIVE:|\Z)',
    flags=re.DOTALL | re.IGNORECASE | re.MULTILINE
)

def normalize_newlines(s: str) -> str:
    # Uniforma CRLF->LF, rimuove eventuale BOM
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    if s.startswith('\ufeff'):
        s = s.lstrip('\ufeff')
    return s

def parse_narratives(text: str):
    """
    Ritorna una lista di Document, uno per narrazione.
    page_content contiene sia la riga ID che il corpo (aiuta il retriever).
    metadata['id'] è l'ID della narrazione.
    """
    text = normalize_newlines(text)
    docs = []
    for m in NARRA_RE.finditer(text.strip()):
        _id = m.group("id").strip()
        body = m.group("body").strip()
        content = f"ID NARRATIVE: {_id}\n{body}"
        docs.append(Document(page_content=content, metadata={"id": _id}))
    return docs

def ensure_writable_dir(path: str):
    os.makedirs(path, exist_ok=True)
    testfile = os.path.join(path, ".__write_test__")
    with open(testfile, "w", encoding="utf-8") as f:
        f.write("ok")
    os.remove(testfile)

def safe_reset_chroma(db_dir: str):
    """Chiude connessioni, resetta il DB e ricrea la cartella senza lock residui."""
    with suppress(Exception):
        # Prova a chiudere/azzerare il client persistente
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=db_dir, settings=Settings(anonymized_telemetry=False))
        with suppress(Exception):
            client.reset()   # svuota collezioni
        with suppress(Exception):
            client.stop()    # best-effort (alcune versioni non lo hanno)
    # aiuta il GC a rilasciare file handle
    del client
    gc.collect()
    time.sleep(0.2)

    # ripulisci directory
    shutil.rmtree(db_dir, ignore_errors=True)
    os.makedirs(db_dir, exist_ok=True)
    ensure_writable_dir(db_dir)

def carica_testo(txt_path: str) -> str:
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"TXT non trovato: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def estrai_json_da_stringa(s: str):
    risultati = []
    for m in re.finditer(r'\{[^{}]*\}', s, flags=re.DOTALL):
        blocco = m.group(0).strip()
        try:
            obj = json.loads(blocco)
            if isinstance(obj, dict):
                if "id" in obj and obj["id"]:
                    risultati.append({"id": str(obj["id"])})
        except Exception:
            pass
    return risultati

def aggiorna_file_json(nuovi: list, path_file: str):
    esistenti = []
    if os.path.exists(path_file):
        try:
            with open(path_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    esistenti = data
        except Exception:
            esistenti = []

    id_presenti = {str(x.get("id")) for x in esistenti if isinstance(x, dict) and "id" in x}

    for obj in nuovi:
        _id = str(obj.get("id"))
        if _id and _id not in id_presenti:
            esistenti.append({"id": _id})
            id_presenti.add(_id)

    os.makedirs(os.path.dirname(path_file), exist_ok=True)
    with open(path_file, "w", encoding="utf-8") as f:
        json.dump(esistenti, f, ensure_ascii=False, indent=2)

def scrivi_log(base_dir: str, prompt: str, risposta: str):
    os.makedirs(base_dir, exist_ok=True)
    log_path = os.path.join(base_dir, "log.txt")
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f'Prompt:\n{prompt}\n\n')
        logf.write(f'Risposta:\n{risposta}\n')
        logf.write("\n\n-----------------------------\n\n")

# ============= RAG =============
def chunk_text(s: str, size=2000, overlap=250):
    out = []
    i = 0
    n = len(s)
    while i < n:
        out.append(s[i:i+size])
        i += max(1, size - overlap)
    return out

def build_retriever_from_txt(
    txt_path: str,
    embed_model: str,
    base_cache_dir: str = None,
    fresh: bool = False,            # True => DB nuovo (eviti lock in Jupyter); False => fisso con reset
    k_shortlist: int = 200,
    chunk_size: int = 2000,
    chunk_overlap: int = 250,
    chunked: bool = False,           # 👈 NOVITÀ: scegli se usare chunk oppure no
):
    """
    Crea indicizzazione Chroma (persistente) e ritorna (retriever, db_dir).
    - chunked=True: indicizza CHUNK per narrazione (ID propagato in ogni chunk).
    - chunked=False: indicizza 1 Document = 1 narrazione intera (ID sempre presente).
    - fresh=True: usa DB con timestamp; False: DB fisso con reset sicuro.
    Fallback automatico: MMR -> similarity se 'fetch_k'/'lambda_mult' non sono supportati.
    """
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from chromadb.config import Settings

    # 1) Path DB
    HOME = str(Path.home())
    base_cache_dir = base_cache_dir or os.path.join(HOME, ".cache", "chroma")
    os.makedirs(base_cache_dir, exist_ok=True)

    if fresh:
        db_dir = os.path.join(base_cache_dir, f"movingjson_db_{int(time.time())}")
        os.makedirs(db_dir, exist_ok=True)
        ensure_writable_dir(db_dir)
        print(f"[INFO] DB_DIR (fresh): {db_dir}")
    else:
        db_dir = os.path.join(base_cache_dir, "movingjson_db")
        safe_reset_chroma(db_dir)  # reset sicuro
        print(f"[INFO] DB_DIR (persistent): {db_dir} (reset eseguito)")

    # 2) Carica testo e costruisci Document
    text = carica_testo(txt_path)
    if chunked:
        docs = build_docs_from_output(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"[INFO] Indicizzazione CHUNKED: {len(docs)} chunk")
    else:
        docs = parse_narratives(text)  # 1 doc = 1 narrazione, "ID NARRAZIONE: <id>\n<body>"
        print(f"[INFO] Indicizzazione NON chunked: {len(docs)} narrazioni")

    # 3) Embeddings e Vector Store
    embeddings = OllamaEmbeddings(model=embed_model)
    client_settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=db_dir,
        allow_reset=True,
    )
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_dir,
        client_settings=client_settings,
    )
    vs.persist()

    # 4) Retriever: prova MMR, fallback similarity
    try:
        retriever = vs.as_retriever(
            search_type= "mmr",
            search_kwargs={
                "k": k_shortlist,
                "fetch_k": max(k_shortlist * 2, 200),
                "lambda_mult": 0.5,
            },
        )
        _ = retriever.get_relevant_documents("smoke test")
        print("[INFO] Retriever: MMR")
    except Exception as e:
        print(f"[WARN] MMR non disponibile ({e}). Fallback a 'similarity'.")
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k_shortlist})
        _ = retriever.get_relevant_documents("smoke test")
        print("[INFO] Retriever: similarity")

    return retriever, db_dir




def make_validation_prompt(narrative_text: str, nid: str) -> str:
    return (
        #"You must analyze ONLY the following NARRATIVE CHUNK (part of a single narrative) and apply the system instructions. "
        #"If it matches, return exactly one JSON object with its id; otherwise return {}.\n\n"
        #f"NARRATIVE CHUNK:\n{narrative_text}"
        f"ID Narrative: {nid}\n"
        f"{narrative_text}\n\n"
    )


def retrieve_context(retriever, query: str) -> str:
    """
    Recupera TOP_K documenti rilevanti per 'query' e li concatena come contesto.
    Qui usiamo il system prompt come proxy della query, per ancorare semanticamente.
    """
    results = retriever.get_relevant_documents(query)
    # concatenazione semplice
    context_chunks: List[str] = []
    for i, d in enumerate(results, start=1):
        # ogni d.page_content è già un chunk del TXT
        context_chunks.append(f"[DOC {i}]\n{d.page_content}")
    return "\n\n".join(context_chunks)

def make_rag_prompt(context: str) -> str:
    """
    Prompt utente passato all'LLM (il system rimane il tuo systemPrompti).
    Non ripetiamo le istruzioni: le hai nel system. Qui forniamo solo il contesto e vincoli di output.
    """
    return (
        #"Use ONLY the following CONTEXT to apply the system instructions and produce the answer.\n"
        #"If nothing in the context supports a positive match, output exactly an empty JSON object {}.\n"
        #"Return ONLY a single JSON object (no prose, no code fences).\n\n"
        #f"CONTEXT:\n{context}"
        context
    )

def build_docs_from_output(text: str, chunk_size=2000, chunk_overlap=250):
    docs = []
    for nar in parse_narratives(text):
        nid = nar.metadata["id"]
        full = nar.page_content  # "ID NARRAZIONE: <id>\n<body>"
        # separa header (riga ID) e body
        if "\n" in full:
            header, body = full.split("\n", 1)
        else:
            header, body = full, ""
        # Propaga anche le prime ~300 battute del body per non perdere segnali chiave
        header_tail = body[:300]

        chunks = chunk_text(body, size=chunk_size, overlap=chunk_overlap)
        for j, ch in enumerate(chunks):
            # PREPENDO L’ID (in italiano corretto) + una fetta di header_tail a ogni chunk
            content = f"{header}\n{header_tail}\n{ch}"  # es: "ID NARRAZIONE: <id>\n<prime frasi>\n<chunk>"
            docs.append(Document(
                page_content=content,
                metadata={"id": nid, "chunk_index": j}
            ))
    return docs

# ============= ESECUZIONE =============
# ============= ESECUZIONE (JUPYTER) =============

# SCEGLI tu qui:
USE_CHUNKS = True   # 👈 metti False per indicizzare 1 narrazione = 1 documento



# ============================
# 🚀 LOOP AUTOMATICO ESPERIMENTI
# ============================

EXPERIMENTS = [
    # (nome, search_type, fetch_k, chunked, chunk_size, chunk_overlap, embed_model)

    ("exp1_mmr_chunkFalse", "mmr", 200, False, 2000, 250, "nomic-embed-text"),
    ("exp2_mmr_fetch1k", "mmr", 1000, False, 2000, 250, "nomic-embed-text"),
    ("exp3_sim_fetch1k", "similarity", 1000, False, 2000, 250, "nomic-embed-text"),
    ("exp4_sim_chunkTrue", "similarity", 1000, True, 1000, 150, "nomic-embed-text"),
    ("exp5_sim_chunkMed", "similarity", 1000, True, 1500, 200, "nomic-embed-text"),
    ("exp6_sim_chunkLarge", "similarity", 1000, True, 2500, 300, "nomic-embed-text"),
    ("exp7_sim_chunkMed_mxbai", "similarity", 1000, True, 1500, 200, "mxbai-embed-large"),
    ("exp8_mmr_chunkMed_mxbai", "mmr", 1000, True, 1500, 200, "mxbai-embed-large"),
    ("exp9_sim_chunkOverlap400", "similarity", 1000, True, 1500, 400, "nomic-embed-text"),
    ("exp10_mmr_lambda03", "mmr", 1000, True, 1500, 200, "nomic-embed-text"),  # servirà parametro extra
    ("exp11_mmr_lambda08", "mmr", 1000, True, 1500, 200, "nomic-embed-text"),
    ("exp12_query3lines", "similarity", 1000, False, 2000, 250, "nomic-embed-text"),  # con query estesa
    ("exp13_queryExpanded", "mmr", 1000, True, 1500, 200, "nomic-embed-text"),
    ("exp14_embed_mxbai_queryExpanded", "mmr", 1000, True, 1500, 200, "mxbai-embed-large"),
    ("exp16_mmr_lambdaLow", "mmr", 1000, True, 1500, 200, "nomic-embed-text"),
    ("exp17_chunkSmall", "mmr", 1000, True, 800, 100, "nomic-embed-text"),
    ("exp18_queryLocalLang", "mmr", 1000, True, 1500, 200, "nomic-embed-text")






]
query_expansions = {
    "Q1": "cheese PDO DOP formaggio certificato denominazione origine protetta",
    "Q3": "sheep cheese pecorino latte di pecora formaggio ovino"
}
K_VALUES = [200, 150, 100]



for exp_name, search_type, fetch_k, chunked, chunk_size, chunk_overlap, embed_model in EXPERIMENTS:
    for k_val in K_VALUES:
        exp_name_k = f"{exp_name}_k{k_val}"
        print(f"\n\n==============================")
        print(f"🚀 INIZIO ESPERIMENTO: {exp_name_k}")
        print(f"==============================\n")

        t_start = time.time()
        EXP_BASE_DIR = os.path.join(BASE_OUTPUT_DIR, "experiments", exp_name_k)
        os.makedirs(EXP_BASE_DIR, exist_ok=True)

        # === Costruzione retriever / indicizzazione ===
        retriever, DB_DIR = build_retriever_from_txt(
            txt_path=TXT_PATH,
            embed_model=embed_model,
            fresh=True,
            k_shortlist=k_val,                  # << usa k corrente
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunked=chunked,
        )

        # === Configura il retriever per questo esperimento + k ===
        try:
            if search_type == "mmr":
                retriever = retriever.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k_val,
                        "fetch_k": max(5 * k_val, 1000),  # pool ampio per MMR
                        "lambda_mult": 0.5,               # o il tuo valore per exp10/11
                    },
                )
            else:
                # Chroma in similarity potrebbe non supportare fetch_k: non passarlo.
                retriever = retriever.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k_val},
                )
            print(f"[INFO] Retriever: {search_type}, k={k_val}")
        except Exception as e:
            print(f"[WARN] Riconfigurazione retriever fallita: {e}")

        # diagnostica parsing (come prima)
        text_dbg = carica_testo(TXT_PATH)
        _docs_raw = parse_narratives(text_dbg)
        print(f"[DBG] Narrazioni parse: {len(_docs_raw)}")
        if chunked:
            _docs_chunks = build_docs_from_output(text_dbg, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            print(f"[DBG] Chunk indicizzati: {len(_docs_chunks)}")

        # === Report header per questo exp+k ===
        report_data = {
            "experiment": exp_name_k,
            "params": {
                "search_type": search_type,
                "k": k_val,                         # << salva k nel report
                "fetch_k": fetch_k,
                "chunked": chunked,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embed_model": embed_model,
            },
            "results": [],
        }

        # === CICLO PROMPTS (come già fai) ===
        for idx, systemPrompti in enumerate(listPrompts, start=1):
            prompt_folder = f"Q{idx}"
            base_prompt_dir = os.path.join(EXP_BASE_DIR, prompt_folder)
            os.makedirs(base_prompt_dir, exist_ok=True)

            query_for_retriever = systemPrompti.split("\n", 1)[0].strip()

            try:
                candidates = retriever.get_relevant_documents(query_for_retriever)
            except Exception as e:
                print(f"[ERRORE] RAG retrieval fallito: {e}")
                candidates = []

            # Dedup
            seen = set()
            shortlisted = []
            for d in candidates:
                key = (d.metadata.get("id"), d.metadata.get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    shortlisted.append(d)

            # (opzionale) tronca comunque a k_val se per qualche motivo supera
            shortlisted = shortlisted[:k_val]

            # Salva shortlist
            shortlist_path = os.path.join(base_prompt_dir, "shortlist.txt")
            with open(shortlist_path, "w", encoding="utf-8") as sf:
                for i, d in enumerate(shortlisted, 1):
                    nid = d.metadata.get("id")
                    chx = d.metadata.get("chunk_index")
                    preview = (d.page_content[:800] + "…") if len(d.page_content) > 800 else d.page_content
                    sf.write(f"[CAND {i}] ID={nid} CHUNK={chx}\n{preview}\n\n")

            # Solo retrieval: registra ids per coverage
            prompt_result = {
                "prompt": prompt_folder,
                "shortlist_size": len(shortlisted),
                "retrieved_ids": list({d.metadata.get("id") for d in shortlisted if d.metadata.get("id")}),
            }
            report_data["results"].append(prompt_result)
            print(f"[OK] {len(shortlisted)} documenti recuperati per {prompt_folder} (k={k_val})")

        # chiusura e salvataggio report
        t_elapsed = time.time() - t_start
        report_data["time_sec"] = round(t_elapsed, 2)
        report_path = os.path.join(EXP_BASE_DIR, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Esperimento {exp_name_k} completato in {t_elapsed/60:.1f} min")
        print(f"📊 Report salvato in: {report_path}")