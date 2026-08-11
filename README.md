# Agentic RAG System

A local RAG (Retrieval-Augmented Generation) system that agentically answeres questions and can generate podcasts and quizzes from ingested documentation using locally ran, open-source LLMs.

## 🎯 Features

- **Local & Free**: Runs entirely on your machine with no API costs
- **Semantic Search**: Vector-based document retrieval using sqlite-vec
- **ReAct Agent**: Iterative Reasoning + Acting loop with tool calling
- **Educational Content**: Generate quizzes and podcasts from documentation
- **Multiple Backends**: Support for Ollama, llama.cpp, and MLX (Apple Silicon)
- **Optimized for Apple Silicon**: Fast performance on M-series Macs with Metal/MLX acceleration

## 📁 Project Structure

```
agentic-rag/
├── src/                        # Source code modules
│   ├── __init__.py             # Package initialization
│   ├── agent.py                # ReAct agent with tool calling
│   ├── config.py               # Configuration loader
│   ├── langchain_agent.py      # Optional LangChain-based agent
│   ├── podcast_generator.py    # Podcast script + audio generation
│   ├── quiz_generator.py       # Quiz generation
│   ├── rag.py                  # SimpleRAG class for vector search
│   ├── tools.py                # Tool definitions for the agent
│   └── tts_engine.py           # Text-to-speech engine (Piper TTS)
├── main.py                     # CLI interface
├── benchmark.py                # Benchmarking utilities
├── visualize_benchmarks.py     # Benchmark visualisation
├── config.yaml                 # Project configuration
├── data/
│   ├── docling_docs/           # Documentation to ingest
│   └── vectors.db              # Vector database
├── voices/                     # Piper TTS voice models
├── outputs/                    # Generated content
│   ├── quizzes/                # Quiz JSON files
│   └── podcasts/               # Podcast MP3 files
├── tests/                      # Test suite
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** (if not already installed):
   ```bash
   brew install ollama
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull a supported model** (choose one or more):
   ```bash
   # IBM Granite 30B (recommended for M4 MacBook Pro)
   ollama pull granite4.1:30b

   # IBM Granite 3B (lightweight, faster)
   ollama pull granite4.1:3b

   # Google Gemma 26B
   ollama pull gemma4:26b
   ```

4. **Pull the embedding model**:
   ```bash
   ollama pull nomic-embed-text
   ```

### Supported Models

**Ollama Models:**
- `granite4.1:30b` — IBM Granite 30B (best balance of quality and speed)
- `granite4.1:3b` — IBM Granite 3B (fastest, lowest memory)
- `gemma4:26b` — Google Gemma 26B

**llama.cpp Models:**
- `unsloth/granite-4.1-30b-GGUF:Q4_K_M` — Granite 30B quantized
- `bartowski/google_gemma-4-26B-A4B-it-GGUF:Q4_K_M` — Gemma 26B quantized

**MLX Models** (Apple Silicon only):
- Any HuggingFace MLX model ID or local path to an MLX model directory
- Example: `mlx-community/granite-4.1-30b-nvfp4`

### Performance Notes (M4 MacBook Pro, 36GB RAM)

| Model | Backend | Tokens/sec | Load Time | Memory Usage |
|-------|---------|------------|-----------|--------------|
| granite4.1:30b | Ollama | ~45–50 | ~15s | ~18GB |
| granite4.1:3b | Ollama | ~75–80 | ~3s | ~2GB |
| gemma4:26b | Ollama | ~40–45 | ~14s | ~16GB |

*Note: Performance varies based on prompt complexity and context length. Metal acceleration is automatically enabled on Apple Silicon Macs.*

### Installation

```bash
# Clone/navigate to project directory
cd agentic-rag

# Install dependencies
uv sync
```

## 📖 Usage

### 1. Ingest Documentation

Load documents into the vector database:

```bash
python main.py ingest data/docling_docs/
```

### 2. Query the RAG System

Search for relevant information:

```bash
# Basic query
python main.py query "How do I extract tables from documents?"

# Get more results
python main.py query "PDF parsing" --k 10
```

### 3. Run the Agent

Run the ReAct agent with tool calling:

```bash
# Default (Ollama backend)
python main.py agent "How does Docling handle PDF parsing?"

# Show step-by-step reasoning and tool calls
python main.py agent "document parsing" --show-trace

# Use a different model or backend
python main.py agent "explain table extraction" --backend ollama --model gemma4:26b
python main.py agent "explain table extraction" --backend mlx
```

### 4. Generate a Quiz

Generate a multiple-choice quiz from the ingested documentation:

```bash
# Basic quiz
python main.py quiz "document parsing"

# Custom number of questions and difficulty
python main.py quiz "PDF extraction" --num 10 --difficulty hard

# Show answers and explanations
python main.py quiz "Docling models" --show-answers

# Use LangChain implementation
python main.py quiz "document parsing" --use-langchain
```

### 5. Generate a Podcast

Generate a conversational podcast script (and optional audio):

```bash
# Script only
python main.py podcast "how Docling processes documents"

# Include MP3 audio generation
python main.py podcast "document AI" --audio

# Custom duration
python main.py podcast "table extraction" --duration 10

# Use LangChain implementation
python main.py podcast "Docling overview" --use-langchain
```

### 6. Interactive Chat

Start an interactive chat session with full tool access:

```bash
# Default Ollama backend
python main.py chat

# Different model
python main.py chat --backend ollama --model gemma4:26b

# MLX backend
python main.py chat --backend mlx

# Show agent reasoning traces
python main.py chat --show-trace
```

## 🏗️ Architecture

### Components

1. **SimpleRAG** (`src/rag.py`)
   - Vector database management with sqlite-vec
   - Document ingestion using Docling
   - Semantic search with cosine similarity
   - Embedding generation via Ollama

2. **Agent** (`src/agent.py`)
   - ReAct (Reasoning + Acting) iterative loop
   - Tool calling with JSON-structured actions
   - Supports Ollama, llama.cpp, and MLX backends
   - Optional trace callback for streaming reasoning steps

3. **LangChainAgent** (`src/langchain_agent.py`)
   - Optional LangChain-based agent with the same interface
   - Enable with `--use-langchain` flag or `use_langchain: true` in `config.yaml`

4. **Tools** (`src/tools.py`)
   - `search_documents` — semantic search over the vector database
   - `calculator` — safe mathematical expression evaluator
   - `generate_quiz` — quiz generation via `QuizGenerator`
   - `generate_podcast` — podcast script/audio generation via `PodcastGenerator`
   - `final_answer` — signals the agent to return its final response

5. **QuizGenerator** (`src/quiz_generator.py`) / **PodcastGenerator** (`src/podcast_generator.py`)
   - Use the agent and RAG to produce structured educational content
   - Saves output to `outputs/quizzes/` and `outputs/podcasts/`

6. **TTSEngine** (`src/tts_engine.py`)
   - Text-to-speech using Piper TTS
   - HOST and EXPERT voices from `voices/`

7. **Config** (`src/config.py`)
   - Loads `config.yaml` with dot-notation access (e.g. `config.get("llm.ollama_model")`)

### ReAct Agent Loop

The agent follows the Reasoning + Acting pattern:

```
1. Reason  → LLM decides what to do next (JSON with thought + action)
2. Act     → Execute the chosen tool
3. Observe → Add the tool result back into the conversation
4. Repeat  → Until final_answer is called or max_iterations reached
```

### Programmatic Usage

```python
from src import SimpleRAG, Agent
from src.tools import create_rag_tools, create_final_answer_tool

# Set up RAG
rag = SimpleRAG()
rag.ingest("data/docling_docs/")

# Set up agent with tools
tools = create_rag_tools(rag)
tools.append(create_final_answer_tool())
agent = Agent(model="granite4.1:30b", backend="ollama", tools=tools)

# Run a query
response = agent.run("How does Docling handle table extraction?", verbose=True)
print(response)

rag.close()
```

## 🔧 Configuration

Edit `config.yaml` to change defaults without touching code:

```yaml
llm:
  use_langchain: false
  ollama_model: "granite4.1:30b"
  mlx_model: "/path/to/local/mlx-model"
  llamacpp_path: "/path/to/model.gguf"
  temperature: 0.2
  max_tokens: 2000

rag:
  db_path: "data/vectors.db"
  embedding_model: "nomic-embed-text"
  chunk_size: 500
  chunk_overlap: 50
  top_k: 5

podcast:
  default_duration: 5
  host_voice: "voices/en_US-lessac-medium.onnx"
  expert_voice: "voices/en_US-libritts-high.onnx"

quiz:
  default_questions: 5
  difficulty: "medium"
```

## 🐛 Troubleshooting

### Ollama Issues

```bash
# Check if Ollama is running
ollama list

# Restart Ollama
killall ollama && ollama serve

# Re-pull model if corrupted
ollama rm granite4.1:30b && ollama pull granite4.1:30b
```

### Import Errors

Make sure you're running commands from the project root:

```bash
cd agentic-rag
python main.py --help
```

### Database Issues

If you encounter database errors, recreate it:

```bash
rm data/vectors.db
python main.py ingest data/docling_docs/
```

## 📚 Resources

- **Ollama**: https://ollama.ai/
- **Docling**: https://github.com/DS4SD/docling
- **sqlite-vec**: https://github.com/asg017/sqlite-vec
- **MLX**: https://github.com/ml-explore/mlx
- **Piper TTS**: https://github.com/rhasspy/piper
- **UV**: https://github.com/astral-sh/uv
