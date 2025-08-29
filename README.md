SAVANT is a Retrieval-Augmented Generation (RAG) system for malware code generation, powered by a dataset of 50,000 malware samples including stealers, worms, keyloggers, and more. Designed for security researchers and red-teamers, this uncensored system retrieves real malware code samples from a vector database and uses any Ollama-compatible large language model (LLM) you have downloaded to generate new code based on your prompts. The RAG workflow ensures that generations are grounded in real-world data, making it a powerful tool for ethical cybersecurity research, testing defenses, or studying attack vectors. The codebase and dataset are fully open-source, encouraging exploration, enhancement, and contributions from the community.

> **Use responsibly and ethically. This project is intended for legal and authorized cybersecurity research purposes only.**

## Features

- **Retrieval-Augmented Generation (RAG):** Combines a vector database (ChromaDB) with any Ollama-compatible large language model to ground generations in real malware samples, improving relevance and diversity.
- **Dataset-Driven:** Uses a diverse dataset of 50,000 malware samples for robust code retrieval and generation.
- **Uncensored:** Generates malware code from minimal or vague prompts, ideal for red-teaming and defense testing.
- **Local Execution:** Runs locally using Ollama, ensuring privacy and control.
- **Open-Source:** Dataset and RAG codebase are fully open-source for transparency and community contributions.
- **Ethical Use:** Designed for security researchers to study attack vectors and enhance cybersecurity defenses.

## Installation

### Prerequisites
- **Ollama**: Install [Ollama](https://ollama.com/) to run the language model locally. You can use any Ollama-compatible model you have downloaded. 
- **Python 3.8+**: For the RAG system and vector database.
- **Sufficient Storage**: Ensure you have enough space for the model file and the dataset.

### Download the Model and Dataset
- **Model File:** Download Ollama-compatible model(make sure is an uncencored model).
- **Embed Model File:** Download `nomic-embed-text`
- **Dataset:** `db.jsonl` (provided in the repo)

## RAG System Setup

1. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare the dataset:** Place `db.jsonl` in the project directory.
3. **Create or update the vector database:**
   ```sh
   python main.py --create-db --size 1000
   ```
   - Use `--size` to specify how many samples to index (e.g., 1000, 5000, 50000).
   - The script uses ChromaDB for fast vector search.


## Usage

### Start the RAG CLI
```sh
python main.py
```

- **Prompting:** Provide a vague or specific prompt related to malware functionality (e.g., `create a keylogger in Python`). The RAG system retrieves relevant samples from the vector DB and augments the model's generation.
- **Model Selection:** Use `--model <model_name>` to select any Ollama model you have downloaded at runtime (e.g., `--model mistral`, `--model llama2`, or any other model you prefer`).
- **Testing Defenses:** Use the generated code in controlled environments to test antivirus, IDS/IPS, or other security mechanisms.
- **Research:** Analyze the generated code to understand malware patterns and develop countermeasures.


## Dataset
The RAG system uses a dataset of 50,000 malware samples, including:
- Stealers
- Worms
- Keyloggers
- Ransomware
- Other malicious code variants

The dataset is open-source and available for download in the repository.

## Ethical Considerations

- **Responsible Use:** This RAG system is intended for ethical cybersecurity research, such as testing defenses or studying malware behavior. Misuse for malicious purposes is strictly prohibited.
- **Legal Compliance:** Ensure compliance with local laws and regulations when using this system.
- **Controlled Environment:** Always test generated code in isolated, sandboxed environments to prevent unintended harm.


## Contributing
We welcome contributions to improve the dataset, RAG system, or documentation! To contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a clear description of your contribution.

Please follow the Contributor Covenant Code of Conduct.


## License
This project is licensed under the MIT License. See the LICENSE file for details.
