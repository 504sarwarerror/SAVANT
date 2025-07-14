import warnings
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass
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
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion

DATA_PATH = 'db.jsonl'
OLLAMA_EMBED_MODEL = 'nomic-embed-text'
OLLAMA_RAG_MODEL = 'qwen2.5-coder:7b-instruct-q4_0'
OLLAMA_URL = 'http://localhost:11434/api/generate'
EMBED_URL = 'http://localhost:11434/api/embeddings'
CHROMA_DB_PATH = 'db'
COLLECTION_NAME = 'rabids'
TOP_K = 5

HISTORY_FILE = 'rabids_history.txt'
COMMANDS = {
    '/help': 'Show this help message',
    '/clear': 'Clear the terminal',
    '/history': 'Show previous queries',
    '/reset': 'Reset the conversation/history',
    '/config': 'Show current config',
    '/exit': 'Exit the program',
}

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

class PersistentHistory(InMemoryHistory):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                for line in f:
                    self.append_string(line.rstrip())
    def save(self):
        with open(self.filename, 'w') as f:
            for item in self.get_strings():
                f.write(item + '\n')

class CommandOnlyCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith('/'):
            for cmd in COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))

command_completer = CommandOnlyCompleter()

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
    history = PersistentHistory(HISTORY_FILE)
    last_message = None

    kb = KeyBindings()

    @kb.add('enter')
    def _(event):
        event.current_buffer.validate_and_handle()

    try:
        @kb.add('c-enter')
        def _(event):
            event.current_buffer.insert_text('\n')
    except Exception:
        pass

    def show_help():
        print_system('Available commands:')
        for cmd, desc in COMMANDS.items():
            print_formatted_text(HTML(f'<ansicyan>{cmd}</ansicyan>: {desc}'))

    def show_history():
        print_system('Query history:')
        for i, item in enumerate(history.get_strings()):
            print_formatted_text(HTML(f'<ansiyellow>{i+1}:</ansiyellow> {item}'))

    def show_config():
        print_system('Current config:')
        print_formatted_text(HTML(f'<ansicyan>Model:</ansicyan> {OLLAMA_RAG_MODEL}'))
        print_formatted_text(HTML(f'<ansicyan>Top K:</ansicyan> {TOP_K}'))
        print_formatted_text(HTML(f'<ansicyan>DB Path:</ansicyan> {CHROMA_DB_PATH}'))
        print_formatted_text(HTML(f'<ansicyan>Collection:</ansicyan> {COLLECTION_NAME}'))

    def clear_terminal():
        os.system('clear' if os.name == 'posix' else 'cls')
        print_ascii_art()

    def reset_history():
        history.strings = []
        history.save()
        print_success('History reset.')

    while True:
        try:
            query = prompt(
                [('class:prompt', '>>> ')],
                style=style,
                placeholder='Send a message (/? for help)',
                include_default_pygments_style=False,
                history=history,
                key_bindings=kb,
                completer=CommandOnlyCompleter(),
                complete_while_typing=True
            )
        except KeyboardInterrupt:
            break
        if not query.strip():
            continue
        if query.startswith('/'):
            cmd = query.strip().split()[0].lower()
            if cmd == '/help':
                show_help()
            elif cmd == '/clear':
                clear_terminal()
            elif cmd == '/history':
                show_history()
            elif cmd == '/reset':
                confirm = input('Are you sure you want to reset history? (y/n): ')
                if confirm.lower() == 'y':
                    reset_history()
            elif cmd == '/config':
                show_config()
            elif cmd == '/exit':
                break
            else:
                print_error(f'Unknown command: {cmd}. Type /help for a list of commands.')
            continue
        if query.lower() == 'exit':
            break
        context = retrieve(query, collection)
        stop_event = threading.Event()
        anim_thread = threading.Thread(target=loading_animation, args=(stop_event, "generating"))
        anim_thread.start()
        start_time = time.time()
        generate_rag(query, context, loading_stop_event=stop_event)
        stop_event.set()
        anim_thread.join()
        elapsed = time.time() - start_time
        print_status(f"[Done in {elapsed:.2f}s]", color='cyan')
        print()
        history.append_string(query)
        history.save()
