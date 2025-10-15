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
from prompt_toolkit.formatted_text import FormattedText
import time
import threading
import argparse
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion
from datetime import datetime

DATA_PATH = 'db.jsonl'
OLLAMA_EMBED_MODEL = 'nomic-embed-text'
OLLAMA_RAG_MODEL = 'qwen2.5-coder:7b-instruct-q4_0'
OLLAMA_URL = 'http://localhost:11434/api/generate'
EMBED_URL = 'http://localhost:11434/api/embeddings'
CHROMA_DB_PATH = 'db'
COLLECTION_NAME = 'savant'
TOP_K = 5

HISTORY_FILE = 'savant_history.txt'
COMMANDS = {
    '/help': 'Show available commands',
    '/clear': 'Clear the screen',
    '/history': 'Show conversation history',
    '/reset': 'Reset conversation history',
    '/config': 'Show current configuration',
    '/exit': 'Exit SAVANT',
}

COLORS = {
    'primary': '#FF6B35',      # Orange accent
    'secondary': '#7B68EE',    # Purple
    'success': '#00D9A3',      # Teal
    'muted': '#8B8B8B',        # Gray
    'text': '#E8E8E8',         # Light gray
    'dim': '#606060',          # Dim gray
    'error': '#FF4444',        # Red
    'warning': '#FFB84D',      # Yellow
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
    # Ollama embeddings endpoint expects the input under the key 'input'.
    # Add retries, timeout and better error reporting to avoid unhandled 500s.
    payload = {"model": OLLAMA_EMBED_MODEL, "input": text}
    retries = 3
    backoff = 0.5
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(EMBED_URL, json=payload, timeout=30)
            if resp.status_code == 200:
                # Try to parse JSON but fall back to raw text on failure for debug
                try:
                    data = resp.json()
                except Exception:
                    raise ValueError(f"Embedding endpoint returned non-json response: {resp.text}")

                # Support different possible shapes returned by embedding services
                # 1) {"embedding": [...]} or {"embeddings": [[...]]}
                if isinstance(data, dict):
                    if 'embedding' in data and isinstance(data['embedding'], (list, tuple)):
                        return data['embedding']
                    if 'embeddings' in data and isinstance(data['embeddings'], list):
                        emb = data['embeddings']
                        if len(emb) == 1:
                            return emb[0]
                        return emb
                    # 2) {"data": [{"embedding": [...]}]}
                    if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                        first = data['data'][0]
                        if isinstance(first, dict) and 'embedding' in first:
                            return first['embedding']
                        # some services nest the vector under 'embedding' key inside 'data'
                        if isinstance(first, list):
                            return first
                    # 3) some services return {'results': [{'vector': [...]}, ...]}
                    if 'results' in data and isinstance(data['results'], list) and len(data['results']) > 0:
                        first = data['results'][0]
                        if isinstance(first, dict):
                            for key in ('embedding', 'vector', 'emb'):
                                if key in first and isinstance(first[key], (list, tuple)):
                                    return first[key]
                # 4) top-level list e.g., [[...]] or [ {...} ]
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, (list, tuple)):
                        return first
                    if isinstance(first, dict) and 'embedding' in first:
                        return first['embedding']

                # unexpected shape
                raise ValueError(f"Unexpected embedding response shape: {json.dumps(data) if isinstance(data, (dict, list)) else str(data)}")
            else:
                # For 5xx errors, retry with backoff. For 4xx, raise immediately.
                body = resp.text
                if 500 <= resp.status_code < 600 and attempt < retries:
                    time.sleep(backoff * (2 ** (attempt - 1)))
                    continue
                resp.raise_for_status()
        except requests.RequestException as e:
            last_exc = e
            # Retry for transient network / server issues
            if attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            # no more retries
            raise
    # If we exit the loop without returning, raise a helpful error
    raise RuntimeError(f"Failed to get embedding after {retries} attempts")

def get_chroma_collection(docs, show_progress=False):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(allow_reset=True))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return collection
        else:
            client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)
    iterator = tqdm(docs, desc='Building vector database', unit='doc', 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') if show_progress else docs
    id_counter = 0
    skipped_count = 0
    for i, doc in enumerate(iterator):
        try:
            emb = get_embedding(doc)
        except Exception:
            # Increment counter and silently skip documents that fail to embed
            skipped_count += 1
            continue
        # Normalize numpy arrays
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        # Flatten single-item nested lists (e.g., [[...]] -> [...])
        if isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list):
            emb = emb[0]
        # Validate embedding is a non-empty 1D list/iterable of numbers
        valid = True
        if not isinstance(emb, (list, tuple)) or len(emb) == 0:
            valid = False
        else:
            for v in emb:
                if not isinstance(v, (int, float)):
                    valid = False
                    break
        if not valid:
            # Invalid embedding; count and skip silently
            skipped_count += 1
            continue
        collection.add(
            documents=[doc],
            embeddings=[emb],
            ids=[str(id_counter)]
        )
        id_counter += 1
    # Print a single summary instead of per-document warnings
    if skipped_count > 0:
        print_formatted_text(HTML(f'<style fg="{COLORS["warning"]}">⚠ Indexed {id_counter} documents, skipped {skipped_count} invalid/failed embeddings</style>'))
    else:
        print_formatted_text(HTML(f'<style fg="{COLORS["success"]}">✓ Indexed {id_counter} documents</style>'))

    return collection

def print_header():
    """Print Claude Code style header"""
    header = f""""""
    print_formatted_text(HTML(f'<style fg="{COLORS["primary"]}">{header}</style>'))
    print_formatted_text(HTML(f'<style fg="{COLORS["muted"]}">  Model: {OLLAMA_RAG_MODEL}</style>'))
    print_formatted_text(HTML(f'<style fg="{COLORS["muted"]}">  Type /help for commands</style>\n'))

def print_divider(char='─', color='dim'):
    """Print a subtle divider"""
    width = os.get_terminal_size().columns if hasattr(os, 'get_terminal_size') else 80
    print_formatted_text(HTML(f'<style fg="{COLORS[color]}">{char * width}</style>'))

def print_message(role, content, color='text'):
    """Print a message with role prefix"""
    timestamp = datetime.now().strftime('%H:%M')
    if role == 'user':
        icon = '❯'
        role_color = COLORS['primary']
    else:
        icon = '●'
        role_color = COLORS['success']
    
    print_formatted_text(HTML(f'<style fg="{role_color}">{icon} {role.upper()}</style> <style fg="{COLORS["muted"]}">{timestamp}</style>'))
    print_formatted_text(HTML(f'<style fg="{COLORS[color]}">{content}</style>\n'))

def print_status(msg, status='info'):
    """Print status message"""
    icons = {
        'info': 'ℹ',
        'success': '✓',
        'error': '✗',
        'warning': '⚠'
    }
    colors = {
        'info': COLORS['secondary'],
        'success': COLORS['success'],
        'error': COLORS['error'],
        'warning': COLORS['warning']
    }
    icon = icons.get(status, 'ℹ')
    color = colors.get(status, COLORS['muted'])
    print_formatted_text(HTML(f'<style fg="{color}">{icon} {msg}</style>'))

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

def generate_rag(query, context, loading_stop_event=None):
    system_prompt = (
        "You are a helpful AI assistant. If the following context contains code, "
        "adapt the code to improve your answer, but do not copy it verbatim. "
        "You only generate complete code. "
        "Use the code's functionality if it makes the generated code better.\n\n"
    )
    prompt_text = f"{system_prompt}Context:\n{chr(10).join(context)}\n\nQuestion: {query}\nAnswer:"
    resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_RAG_MODEL, "prompt": prompt_text, "stream": True}, stream=True)
    
    first_chunk = True
    full_response = []
    
    for line in resp.iter_lines():
        if line:
            if first_chunk and loading_stop_event is not None:
                loading_stop_event.set()
                sys.stdout.write('\r\033[2K')
                sys.stdout.flush()
                timestamp = datetime.now().strftime('%H:%M')
                print_formatted_text(HTML(f'<style fg="{COLORS["success"]}">● ASSISTANT</style> <style fg="{COLORS["muted"]}">{timestamp}</style>'))
                first_chunk = False
            try:
                data = line.decode('utf-8')
                import json
                chunk = json.loads(data)
                if 'response' in chunk:
                    text = chunk['response']
                    full_response.append(text)
                    print(text, end="", flush=True)
            except Exception:
                continue
    print("\n")
    return ''.join(full_response)

def loading_animation(stop_event, message="Thinking"):
    """Claude Code style loading animation"""
    import itertools
    frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner = itertools.cycle(frames)
    
    while not stop_event.is_set():
        frame = next(spinner)
        sys.stdout.write(f'\r\033[38;2;123;104;238m{frame}\033[0m \033[38;2;139;139;139m{message}...\033[0m')
        sys.stdout.flush()
        time.sleep(0.08)
    
    sys.stdout.write('\r\033[2K')
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

class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith('/'):
            for cmd in COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text), display_meta=COMMANDS[cmd])

def show_help():
    print_divider()
    print_formatted_text(HTML(f'<style fg="{COLORS["primary"]}">Available Commands</style>\n'))
    for cmd, desc in COMMANDS.items():
        print_formatted_text(HTML(f'  <style fg="{COLORS["secondary"]}">{cmd:12}</style> <style fg="{COLORS["muted"]}">{desc}</style>'))
    print_divider()

def show_history(history):
    print_divider()
    print_formatted_text(HTML(f'<style fg="{COLORS["primary"]}">Conversation History</style>\n'))
    items = list(history.get_strings())
    if not items:
        print_formatted_text(HTML(f'<style fg="{COLORS["muted"]}">  No history yet</style>'))
    else:
        for i, item in enumerate(items[-10:], 1):  # Show last 10
            preview = item[:60] + '...' if len(item) > 60 else item
            print_formatted_text(HTML(f'  <style fg="{COLORS["muted"]}">{i:2}.</style> <style fg="{COLORS["text"]}">{preview}</style>'))
    print_divider()

def show_config():
    print_divider()
    print_formatted_text(HTML(f'<style fg="{COLORS["primary"]}">Current Configuration</style>\n'))
    config = [
        ('Model', OLLAMA_RAG_MODEL),
        ('Top K Results', str(TOP_K)),
        ('Database Path', CHROMA_DB_PATH),
        ('Collection', COLLECTION_NAME),
        ('Embed Model', OLLAMA_EMBED_MODEL),
    ]
    for key, value in config:
        print_formatted_text(HTML(f'  <style fg="{COLORS["secondary"]}">{key:16}</style> <style fg="{COLORS["text"]}">{value}</style>'))
    print_divider()

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')
    print_header()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAVANT - Smart AI Assistant for Variable and Novel Tasks")
    parser.add_argument('--model', type=str, help='Ollama model to use for RAG')
    parser.add_argument('--create-db', action='store_true', help='Force recreate the vector database')
    parser.add_argument('--size', type=int, default=1000, help='Number of documents to index (default: 1000)')
    args = parser.parse_args()
    
    if args.model:
        OLLAMA_RAG_MODEL = args.model
    
    clear_screen()
    
    # Initialize database
    print_status('Initializing vector database...', 'info')
    docs = load_documents(DATA_PATH, max_docs=args.size)
    
    if args.create_db:
        import shutil
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
        collection = get_chroma_collection(docs, show_progress=True)
        print_status(f'Database created with {len(docs)} documents', 'success')
    else:
        collection = get_chroma_collection(docs, show_progress=False)
        print_status(f'Database loaded with {collection.count()} documents', 'success')
    
    print()
    
    # Setup prompt style
    style = Style.from_dict({
        'prompt': f'{COLORS["primary"]}',
        'placeholder': f'{COLORS["dim"]}',
    })
    
    history = PersistentHistory(HISTORY_FILE)
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
    
    # Main loop
    while True:
        try:
            query = prompt(
                FormattedText([
                    (f'fg:{COLORS["primary"]}', '❯ '),
                ]),
                style=style,
                placeholder='Ask anything... (type /help for commands)',
                multiline=False,
                history=history,
                key_bindings=kb,
                completer=CommandCompleter(),
                complete_while_typing=True
            )
        except (KeyboardInterrupt, EOFError):
            print_formatted_text(HTML(f'\n<style fg="{COLORS["muted"]}">Goodbye! 👋</style>'))
            break
        
        if not query.strip():
            continue
        
        # Handle commands
        if query.startswith('/'):
            cmd = query.strip().split()[0].lower()
            if cmd == '/help':
                show_help()
            elif cmd == '/clear':
                clear_screen()
            elif cmd == '/history':
                show_history(history)
            elif cmd == '/reset':
                print_status('Reset conversation history? (y/n): ', 'warning')
                confirm = input().strip().lower()
                if confirm == 'y':
                    history.strings = []
                    history.save()
                    print_status('History cleared', 'success')
            elif cmd == '/config':
                show_config()
            elif cmd == '/exit':
                print_formatted_text(HTML(f'<style fg="{COLORS["muted"]}">Goodbye! 👋</style>'))
                break
            else:
                print_status(f'Unknown command: {cmd}', 'error')
            continue
        
        if query.lower() == 'exit':
            break
        
        # Display user message
        print_divider('─', 'dim')
        print_message('user', query)
        
        # Retrieve context
        context = retrieve(query, collection)
        
        # Generate response with loading animation
        stop_event = threading.Event()
        anim_thread = threading.Thread(target=loading_animation, args=(stop_event, "Thinking"))
        anim_thread.start()
        
        start_time = time.time()
        response = generate_rag(query, context, loading_stop_event=stop_event)
        
        stop_event.set()
        anim_thread.join()
        
        elapsed = time.time() - start_time
        print_formatted_text(HTML(f'<style fg="{COLORS["muted"]}">  Generated in {elapsed:.2f}s</style>'))
        print_divider('─', 'dim')
        print()
        
        # Save to history
        history.append_string(query)
        history.save()