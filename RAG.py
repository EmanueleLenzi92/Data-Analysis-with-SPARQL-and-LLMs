# -*- coding: utf-8 -*-
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
TXT_PATH = "MovingNarratives.txt"       # <-- file TXT già generato
BASE_OUTPUT_DIR = "movingJson"           # cartella di output per JSON e log
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
EMBED_MODEL = "mxbai-embed-large" # es "nomic-embed-text"  "mxbai-embed-large" o "snowflake-arctic-embed" "embeddinggemma"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 8  # quanti chunk recuperare dal RAG

# ========== 4 SYSTEM PROMPTS ==========



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
                "fetch_k": 1000, #max(k_shortlist * 2, 200),
                "lambda_mult": 0.5,
            },
        )
        #retriever = vs.as_retriever(
        #    search_type="similarity",
        #    search_kwargs={"k": 200} # per PDO
            
        #)
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
"""You must use ONLY the CONTENT given in this user message as CONTEXT of the narrative.
Do not use any external or real-world knowledge.

"""
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

retriever, DB_DIR = build_retriever_from_txt(
    txt_path=TXT_PATH,
    embed_model=EMBED_MODEL,
    fresh=True,            # DB nuovo per evitare lock in Jupyter
    k_shortlist=200, #numero massimo di narrazioni simili da trovare su Chroma (db vettorizzato con rag)
    chunk_size=1500, #2000,
    chunk_overlap= 200, #250,
    chunked=USE_CHUNKS,    # 👈 scelta
)

# diagnostica parsing
text_dbg = carica_testo(TXT_PATH)
_docs_raw = parse_narratives(text_dbg)
print(f"[DBG] Narrazioni parse: {len(_docs_raw)} (es. primi 3 ID: {[d.metadata['id'] for d in _docs_raw[:3]]})")

if USE_CHUNKS:
    _docs_chunks = build_docs_from_output(text_dbg, chunk_size=2500, chunk_overlap=300)
    print(f"[DBG] Chunk indicizzati: {len(_docs_chunks)} "
          f"(media chunk/narrazione: {len(_docs_chunks)/max(1,len(_docs_raw)):.2f})")


for idx, systemPrompti in enumerate(listPrompts, start=1):
    prompt_folder = f"Q{idx}"

    # 2) RAG shortlist: recupera i chunk candidati più pertinenti al system prompt

    #  Usa solo la prima riga del system prompt per il retrieval semantico
    query_for_retriever = systemPrompti.split("\n", 1)[0].strip()
    
    try:
        candidates = retriever.get_relevant_documents(query_for_retriever)
    except TypeError as e:
        # es. "fetch_k" non supportato → ricrea retriever in "similarity"
        print(f"[WARN] get_relevant_documents ha fallito con TypeError: {e}. Fallback a 'similarity'.")
        try:
            # ricrea retriever in similarity
            # (riusa lo stesso vectorstore dentro build_retriever_from_txt, oppure ricrealo qui se lo hai a portata)
            retriever = retriever.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 200})
            candidates = retriever.get_relevant_documents(query_for_retriever)
        except Exception as e2:
            print(f"[ERRORE] RAG retrieval fallito anche in fallback: {e2}")
            candidates = []
    except Exception as e:
        print(f"[ERRORE] RAG retrieval fallito per {prompt_folder}: {e}")
        candidates = []


    # Dedup su (id, chunk_index) per evitare doppioni identici
    seen = set()
    shortlisted = []
    for d in candidates:
        key = (d.metadata.get("id"), d.metadata.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            shortlisted.append(d)

    if USE_RAG_CONTEXT:
        context_blocks = []
        for i, d in enumerate(shortlisted[:RAG_MAX_DOCS], start=1):
            nid = d.metadata.get("id")
            context_blocks.append(f"[DOC {i}] ID={nid}\n{d.page_content}")
        user_prompt_rag = make_rag_prompt("\n\n".join(context_blocks))

    # (Debug) salva la shortlist per audit
    base_dir_prompt = os.path.join(BASE_OUTPUT_DIR, prompt_folder)
    os.makedirs(base_dir_prompt, exist_ok=True)
    with open(os.path.join(base_dir_prompt, "shortlist.txt"), "w", encoding="utf-8") as sf:
        for i, d in enumerate(shortlisted, 1):
            nid = d.metadata.get("id")
            chx = d.metadata.get("chunk_index")
            # mostriamo solo i primi caratteri per non esplodere i file di log
            preview = (d.page_content[:800] + "…") if len(d.page_content) > 800 else d.page_content
            sf.write(f"[CAND {i}] ID={nid} CHUNK={chx}\n{preview}\n\n")

    # 3) Validazione per-chunk: se un chunk matcha → salvi l'ID (dedup)
    for llmModel in listllms:
        print(f"[INFO] Esecuzione (RAG validate): {prompt_folder} | Modello: {llmModel}")

        llm = Ollama(
            model=llmModel,
            system=systemPrompti,
            num_ctx=NUM_CTX,
            temperature=TEMPERATURE,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

        base_dir = os.path.join(BASE_OUTPUT_DIR, prompt_folder, llmModel)
        json_path = os.path.join(base_dir, "a.json")
        os.makedirs(base_dir, exist_ok=True)

        hits_ids = set()

        if USE_RAG_CONTEXT:
        # 🔹 Rag calssico: UNA SOLA INFERENZA con contesto combinato passato 1 volta sola
            try:
                resp = llm(user_prompt_rag)
                scrivi_log(base_dir, user_prompt_rag, resp)
                estratti = estrai_json_da_stringa(resp)
                if estratti:
                    aggiorna_file_json(estratti, json_path)
                print(f"[OK] RAG classico: {len(estratti)} ID trovati")
            except Exception as e:
                print(f"[ERRORE] Chiamata a {llmModel} fallita (rag): {e}")
                with suppress(Exception):
                    subprocess.run(["ollama", "stop", llmModel], check=False)
                continue
    
            scrivi_log(base_dir, user_prompt_rag, resp)
            estratti = estrai_json_da_stringa(resp)
            if estratti:
                for obj in estratti:
                    hits_ids.add(obj["id"])
                aggiorna_file_json(estratti, json_path)
            print(f"[OK] (rag) {len(hits_ids)} ID → {json_path}")

        else:
         # 1 prompt per ogni narrazione (delle top ritrovate da Chroma)
            for d in shortlisted:
                nid = d.metadata.get("id")
                if nid in hits_ids:
                    continue
    
                # 👇 NUOVO: prompt compatto che include SEMPRE l'ID
                user_prompt = make_validation_prompt(d.page_content, nid)
               
    
                try:
                    resp = llm(user_prompt)
                except Exception as e:
                    print(f"[ERRORE] Chiamata a {llmModel} fallita su ID={nid}: {e}")
                    try:
                        subprocess.run(["ollama", "stop", llmModel], check=False)
                    except Exception:
                        pass
                    continue
    
                scrivi_log(base_dir, user_prompt, resp)
    
                estratti = estrai_json_da_stringa(resp)
                if estratti:
                    hits_ids.add(estratti[0]["id"])
                    aggiorna_file_json(estratti, json_path)
    
            print(f"[OK] {len(hits_ids)} ID unici aggiunti → {json_path}")

        try:
            subprocess.run(["ollama", "stop", llmModel], check=True)
            print(f"[INFO] Modello {llmModel} scaricato dalla GPU.")
        except Exception as e:
            print(f"[WARN] Impossibile scaricare {llmModel}: {e}")