import os
import jsonlines
import numpy as np
import requests
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import sys
import readline
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit import print_formatted_text, HTML
import time
import threading
import argparse

DATA_PATH = 'db.jsonl'
OLLAMA_EMBED_MODEL = 'nomic-embed-text'
OLLAMA_RAG_MODEL = 'dolphin-mistral:7b-v2-q4_0'
OLLAMA_URL = 'http://localhost:11434/api/generate'
EMBED_URL = 'http://localhost:11434/api/embeddings'
CHROMA_DB_PATH = 'db'
COLLECTION_NAME = 'rabids'
TOP_K = 5

def load_documents(path, max_docs=1000):
    docs = []
    with jsonlines.open(path) as reader:
        for i, obj in enumerate(reader):
            if i >= max_docs:
                break
            if isinstance(obj, dict) and 'text' in obj:
                docs.append(obj['text'])
            else:
                docs.append(str(obj))
    return docs

def get_embedding(text):
    resp = requests.post(EMBED_URL, json={"model": OLLAMA_EMBED_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()['embedding']

def get_chroma_collection(docs, show_progress=False):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(allow_reset=True))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return collection
        else:
            client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)
    iterator = tqdm(docs, desc='ChromaDB loading', unit='doc') if show_progress else docs
    for i, doc in enumerate(iterator):
        emb = get_embedding(doc)
        collection.add(
            documents=[doc],
            embeddings=[emb],
            ids=[str(i)]
        )
    return collection

def print_ascii_art():
    art = r'''
     []  ,----.___
   __||_/___      '.
  / O||    /|       )
 /   ""   / /   =._/
/________/ /
|________|/   dew
'''
    print_formatted_text(HTML(f'<ansired>{art}</ansired>'))

def print_error(msg):
    print_formatted_text(HTML(f'<ansired>{msg}</ansired>'))

def print_system(msg):
    print_formatted_text(HTML(f'<ansigreen>{msg}</ansigreen>'))

def print_success(msg):
    print_formatted_text(HTML(f'<ansigreen>{msg}</ansigreen>'))

def print_status(msg, color='green'):
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'cyan': '\033[96m',
        'reset': '\033[0m',
    }
    color_code = colors.get(color, colors['green'])
    reset_code = colors['reset']
    sys.stdout.write('\r\033[2K')
    sys.stdout.write(f'{color_code}{msg}{reset_code}')
    sys.stdout.flush()

def clear_status_line():
    sys.stdout.write('\r\033[2K')
    sys.stdout.flush()
    print()

def retrieve(query, collection, k=TOP_K):
    q_emb = get_embedding(query)
    if not isinstance(q_emb, list) or len(q_emb) == 0 or (isinstance(q_emb[0], list) and len(q_emb[0]) == 0):
        return []
    try:
        results = collection.query(query_embeddings=[q_emb], n_results=k)
        docs = results['documents'][0] if results['documents'] else []
        return docs
    except Exception as e:
        return []

import sys

def generate_rag(query, context, loading_stop_event=None):
    system_prompt = (
        "You are a helpful AI assistant. If the following context contains code, "
        "adapt the code to improve your answer, but do not copy it verbatim. "
        "You only genrate complete code"
        "Use the code's functionality if it makes the generated code better.\n\n"
    )
    prompt_text = f"{system_prompt}Context:\n{chr(10).join(context)}\n\nQuestion: {query}\nAnswer:"
    resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_RAG_MODEL, "prompt": prompt_text, "stream": True}, stream=True)
    first_chunk = True
    print("\n[Answer]: ", end="", flush=True)
    for line in resp.iter_lines():
        if line:
            if first_chunk and loading_stop_event is not None:
                loading_stop_event.set()
                first_chunk = False
            try:
                data = line.decode('utf-8')
                import json
                chunk = json.loads(data)
                if 'response' in chunk:
                    print(chunk['response'], end="", flush=True)
            except Exception:
                continue
    print("\n")
    return None

def loading_animation(stop_event, message="generating"):
    import itertools
    import time
    spinner = itertools.cycle(['|', '/', '-', '\\', '-'])
    while not stop_event.is_set():
        frame = next(spinner)
        sys.stdout.write(f'\r{message} {frame}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * (len(message) + 2) + '\r')
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RABIDS CLI")
    parser.add_argument('--model', type=str, help='Ollama model to use for RAG')
    parser.add_argument('--create-db', action='store_true', help='Force recreate the ChromaDB vector DB')
    parser.add_argument('--size', type=int, default=1000, help='Number of samples to use when creating the DB (default: 1000)')
    args = parser.parse_args()
    if args.model:
        OLLAMA_RAG_MODEL = args.model
    print_ascii_art()
    docs = load_documents(DATA_PATH, max_docs=args.size)
    if args.create_db:
        import shutil
        import os
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
        collection = get_chroma_collection(docs, show_progress=True)
    else:
        collection = get_chroma_collection(docs, show_progress=False)
    style = Style.from_dict({
        'prompt': '#ffffff',
        'placeholder': '#8A8A8A',
    })
    while True:
        try:
            query = prompt(
                [('class:prompt', '>>> ')],
                style=style,
                placeholder='Send a message (/? for help)',
                include_default_pygments_style=False
            )
        except KeyboardInterrupt:
            break
        if query.lower() == 'exit':
            break
        context = retrieve(query, collection)
        stop_event = threading.Event()
        anim_thread = threading.Thread(target=loading_animation, args=(stop_event, "generating"))
        anim_thread.start()
        generate_rag(query, context, loading_stop_event=stop_event)
        stop_event.set()
        anim_thread.join()
