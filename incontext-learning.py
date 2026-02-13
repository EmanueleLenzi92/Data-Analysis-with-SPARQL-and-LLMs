from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

import json
from difflib import SequenceMatcher
import difflib

import time
import os
import re
import logging

import csv
import subprocess

# Configura il logger
logging.basicConfig(filename='error_log_moving.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def estrai_json_da_stringa(stringa):
    # Usa una regex per estrarre il JSON dalla stringa
    match = re.search(r'\{.*\}', stringa, re.DOTALL)
    if match:
        json_string = match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            # Registra l'errore e la stringa che ha causato l'errore
            logging.error(f"Errore nel parsing del JSON: {e}")
            logging.error(f"Stringa problematica: {json_string}")
            return None
    else:
        logging.error(f"Nessun JSON valido trovato nella stringa: {stringa}")
        return None

def aggiorna_file_json(nuovi_dati, percorso_file):
    # Se il file esiste, carica i dati esistenti
    if os.path.exists(percorso_file):
        with open(percorso_file, 'r', encoding='utf-8') as file:
            dati_esistenti = json.load(file)
    else:
        dati_esistenti = []

    # Aggiungi i nuovi dati ai dati esistenti
    dati_esistenti.append(nuovi_dati)

    # Salva il file aggiornato
    with open(percorso_file, 'w', encoding='utf-8') as file:
        json.dump(dati_esistenti, file, ensure_ascii=False, indent=4)


directory= "movingNarratives_CSV/"

listllms= ["gemma3:12b-it-q8_0", "phi4:14b-q8_0", "llama2:13b-chat-q8_0", "deepseek-r1:14b-qwen-distill-q8_0", "llama3.1:8b",  "llama3:8b-instruct-q8_0", "gemma2:9b-instruct-q8_0", "mistral:7b-instruct-q8_0",  "gemma3:4b-it-q8_0", "llama3.2:3b", "gemma2:2b", "deepseek-r1:1.5b-qwen-distill-fp16", "phi3.5:latest" ]



listPrompts = [

"""Given a narrative about a mountain value chain, recognize if the mountain value chain is involved in the production of cheese with Protected Designation of Origin (PDO) certification. If the value chain is involved in the production of cheese with PDO certification, write a JSON containing the "id" of the narrative. If the value chain is not involved in the production of cheese with PDO certification, write an empty JSON.  

### Example of JSON (if the value chain is involved in the production of cheese with PDO certification)
{"id": "<ID_OF_THE_NARRATIVE>"}

### Example of JSON (if the value chain is not involved  in the production of cheese with PDO certification)
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

for idx, systemPrompti in enumerate(listPrompts, start=1):
    prompt_folder = f"Q{idx}"   # Q1, Q2, Q3, ...
    
    for llmModel in listllms:
        # Cicla tutti i file nella cartella
        for filename in os.listdir(directory):
            # Controlla se il file ha estensione .csv
            if filename.endswith(".csv"):
                filepath = os.path.join(directory, filename)
                #print(f"Processing file: {filename}")
                
                    
                # Apri il file CSV
                with open(filepath, newline='', encoding='utf-8') as csvfile:
                    csvreader = csv.reader(csvfile)
                    #csvreader = csv.reader(csvfile, delimiter=';', quotechar='"')
                        
                    # Salta la prima riga (i titoli)
                    next(csvreader)
    
                     # === ricava ID dal nome file ===
                    base = os.path.splitext(filename)[0]           # es: export_public_2974moving_52-q22
                    parts = base.split('_')                        # ["export","public","2974moving","52-q22"]
                    token = parts[2] if len(parts) >= 3 else ""    # "2974moving"
                    m = re.search(r'\d+', token)                   # prende solo le cifre iniziali
                    nar_id = m.group(0) if m else token 
                        
                    # Processa ogni riga
                    for row in csvreader:
                        # Stampa il valore della seconda colonna (indice 1)
                        if len(row) > 1:
                            #print(row[1])
        
                            sen = f"ID Narrative: {nar_id}\n{row[1]}"
                            print(sen)
                
                            
                            llm = Ollama(
                                model=llmModel, 
                                system=systemPrompti, 
                                num_ctx=4096, 
                                temperature=0.01, 
                                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                            )
                            
                            events = llm(sen)
        
                            # Percorso del file JSON dove salvare i dati
                            base_dir = os.path.join("movingJson", prompt_folder, llmModel)
                            os.makedirs(base_dir, exist_ok=True)
                            percorso_file_json = os.path.join(base_dir, "a.json")

                            # === LOG ===
                            log_path = os.path.join(base_dir, "log.txt")
                            with open(log_path, "a", encoding="utf-8") as logf:
                                logf.write(f'Promp: "{sen}"\n\n')
                                logf.write(f'Risposta: "{events}"\n')
                                logf.write("\n\n\n-----------------------------\n\n\n")  # 3 a capo + separatore
                            
                            # Estrai il JSON dalla stringa
                            json_estratto = estrai_json_da_stringa(events, percorso_file_json)
    
                            print()
                            
                            if json_estratto:
                                
                                # Aggiorna il file JSON con i nuovi dati
                                aggiorna_file_json(json_estratto, percorso_file_json)
                            else:
                                print("Nessun JSON trovato nella stringa.")
        
        # 🟢 fine ciclo CSV → scarica il modello dalla GPU
        try:
            subprocess.run(["ollama", "stop", llmModel], check=True)
            print(f"[INFO] Modello {llmModel} scaricato dalla GPU.")
        except Exception as e:
            print(f"[WARN] Impossibile scaricare {llmModel}: {e}")