# RAG Core Integration

A powerful integration combining **LightRAG** and **RAG-Anything** - bringing together knowledge graph-based retrieval and multimodal document processing capabilities.

## 🚀 Overview

This repository contains the core code from two complementary RAG systems:

- **LightRAG**: Fast and lightweight retrieval-augmented generation with graph-based knowledge representation
- **RAG-Anything**: Multimodal RAG system with advanced document parsing (PDF, images, tables, equations)

## 📦 Project Structure

```
rag-core-integration/
├── lightrag/              # LightRAG core library (Backend)
│   ├── lightrag.py        # Main RAG class
│   ├── operate.py         # Core operations
│   ├── kg/                # Knowledge graph storage
│   ├── llm/               # LLM integrations
│   ├── api/               # FastAPI server
│   └── utils/             # Utilities
├── lightrag_webui/        # LightRAG WebUI (Frontend)
│   ├── src/               # React + TypeScript source
│   │   ├── components/    # UI components
│   │   ├── features/      # Feature modules
│   │   ├── api/           # API client
│   │   └── locales/       # i18n translations
│   ├── public/            # Static assets
│   └── package.json       # Node dependencies
├── raganything/           # RAG-Anything core library
│   ├── api/               # API server
│   ├── parser/            # Document parsers
│   ├── modalprocessors.py # Multimodal processors
│   └── utils/             # Utilities
├── docs/                  # Documentation
│   ├── LightRAG_README.md
│   └── RAG-Anything_README.md
└── examples/              # Usage examples
```

## ✨ Key Features

### LightRAG Features
- 🔍 **Graph-based RAG**: Knowledge graph representation with Neo4j/NetworkX
- ⚡ **Multiple Query Modes**: local, global, hybrid, naive, mix
- 💾 **Flexible Storage**: PostgreSQL, Redis, MongoDB, Milvus, Qdrant
- 🔌 **LLM Provider Support**: OpenAI, Ollama, DeepSeek, Claude, and more
- 🌐 **API Server**: FastAPI with WebUI and Ollama-compatible endpoints

### RAG-Anything Features
- 📄 **Advanced PDF Parsing**: MinerU-based PDF processing
- 🖼️ **Multimodal Support**: Images, tables, equations extraction
- 🧠 **Context-Aware**: Page-level context extraction
- 🔄 **Flexible Pipeline**: Customizable processing workflows
- 📊 **Rich Formats**: Support for Office documents, images, and more

## 🛠️ Installation

### Prerequisites

**Backend:**
- Python 3.10+
- PostgreSQL (for LightRAG storage)
- Neo4j (optional, for graph storage)

**Frontend:**
- Node.js 18+
- npm or bun

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/blingtien/rag-core-integration.git
cd rag-core-integration
```

### Backend Setup (LightRAG)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install LightRAG dependencies
pip install -e ".[api]"

# Configure environment
cp env.example .env
# Edit .env with your configuration
```

### Frontend Setup (LightRAG WebUI)

```bash
# Navigate to frontend directory
cd lightrag_webui

# Install dependencies
npm install
# or using bun: bun install

# Configure environment
cp env.local.sample .env.local
# Edit .env.local with your API endpoint

# Development mode
npm run dev

# Production build
npm run build
```

### RAG-Anything Setup

```bash
# Install RAG-Anything dependencies
cd raganything
pip install -e .

# Configure environment
cp env.example .env
# Edit .env with your configuration
```

## 📖 Usage

### LightRAG Example

```python
import asyncio
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status

async def main():
    # Initialize LightRAG
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=your_llm_func,
        embedding_func=your_embedding_func
    )

    # Required initialization
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Insert documents
    await rag.ainsert("Your document text here")

    # Query
    result = await rag.aquery(
        "Your question",
        param=QueryParam(mode="hybrid")
    )
    print(result)

asyncio.run(main())
```

### RAG-Anything Example

```python
from raganything import RAGAnything

# Initialize RAG-Anything
rag = RAGAnything(
    parser_config={"use_mineru": True},
    embedding_config=your_embedding_config
)

# Process multimodal document
result = rag.process_document("path/to/document.pdf")

# Query across modalities
answer = rag.query("Your question about the document")
```

## 🏗️ Architecture

### LightRAG Architecture
- **4-tier Storage**: KV, Vector, Graph, DocStatus
- **Query Modes**: Multiple retrieval strategies
- **API Layer**: FastAPI with authentication
- **WebUI**: React-based management interface

### RAG-Anything Architecture
- **Parser Layer**: MinerU for advanced parsing
- **Modal Processors**: Specialized handlers for different content types
- **RAG Layer**: Integration with various RAG backends
- **API Layer**: RESTful endpoints for all operations

## 📚 Documentation

Detailed documentation for each component:
- [LightRAG Documentation](docs/LightRAG_README.md)
- [RAG-Anything Documentation](docs/RAG-Anything_README.md)

## 🤝 Integration

These systems can be used together for enhanced capabilities:

```python
# Example: Using RAG-Anything for parsing with LightRAG for retrieval
from raganything import DocumentParser
from lightrag import LightRAG

# Parse multimodal document
parser = DocumentParser(use_mineru=True)
parsed_content = parser.parse("document.pdf")

# Index in LightRAG
rag = LightRAG(...)
await rag.ainsert(parsed_content.text)

# Query with graph-based retrieval
result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))
```

## 📄 License

This integration repository contains code from:
- LightRAG: See [LICENSE_LightRAG](LICENSE_LightRAG)
- RAG-Anything: See [LICENSE_RAG-Anything](LICENSE_RAG-Anything)

## 🙏 Credits

This repository integrates core components from:
- **LightRAG**: [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- **RAG-Anything**: [jina-ai/RAG-Anything](https://github.com/jina-ai/RAG-Anything)

Please refer to the original repositories for full documentation, issues, and contributions.

## ⚠️ Note

This is a core code integration for learning and development purposes. For production use, please refer to the original repositories for the latest updates and official support.

## 🔗 Links

- [LightRAG Original Repository](https://github.com/HKUDS/LightRAG)
- [RAG-Anything Original Repository](https://github.com/jina-ai/RAG-Anything)
- [Integration Repository](https://github.com/blingtien/rag-core-integration)
