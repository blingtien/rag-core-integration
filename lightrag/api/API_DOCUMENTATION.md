# LightRAG Backend API Reference

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Response Formats](#common-response-formats)
4. [Core API Endpoints](#core-api-endpoints)
   - [System & Health](#system--health)
   - [Authentication Endpoints](#authentication-endpoints)
   - [Document Management](#document-management)
   - [Query Operations](#query-operations)
   - [Graph Operations](#graph-operations)
   - [Ollama API Emulation](#ollama-api-emulation)
5. [Error Handling](#error-handling)
6. [Environment Configuration](#environment-configuration)

## Overview

LightRAG is a Fast, Lightweight RAG (Retrieval-Augmented Generation) server that provides comprehensive document management, query processing, and knowledge graph capabilities. The API supports multiple LLM backends (OpenAI, Ollama, Azure OpenAI, AWS Bedrock) and various storage implementations.

**Base URL**: `http://localhost:9621` (default)
**API Version**: Variable (returned in health endpoint)
**Authentication**: API Key, JWT, or Whitelist-based

## Authentication

The LightRAG API supports three authentication methods:

### 1. No Authentication (Default)
By default, all endpoints are accessible without authentication.

### 2. API Key Authentication
Include the API key in request headers:
```http
X-API-Key: your-secure-api-key-here
```

### 3. JWT Authentication
1. Login to get a token
2. Include in Authorization header:
```http
Authorization: Bearer <jwt-token>
```

### 4. Whitelist Paths
Certain paths are whitelisted by default (configurable):
- `/health` - Health check endpoint
- `/api/*` - Ollama API emulation endpoints

## Common Response Formats

### Success Response
```json
{
  "status": "success",
  "message": "Operation completed successfully",
  "data": {}
}
```

### Error Response
```json
{
  "detail": "Error description"
}
```

### Pipeline Status Response
```json
{
  "autoscanned": false,
  "busy": false,
  "job_name": "Default Job",
  "job_start": "2025-03-31T12:34:56Z",
  "docs": 0,
  "batchs": 0,
  "cur_batch": 0,
  "request_pending": false,
  "latest_message": "",
  "history_messages": []
}
```

## Core API Endpoints

### System & Health

#### GET `/health`
Get current system status and configuration.

**Authentication**: Required (unless whitelisted)

**Response**:
```json
{
  "status": "healthy",
  "working_directory": "/path/to/rag_storage",
  "input_directory": "/path/to/inputs", 
  "configuration": {
    "llm_binding": "openai",
    "llm_binding_host": "https://api.openai.com/v1",
    "llm_model": "gpt-4o",
    "embedding_binding": "ollama",
    "embedding_binding_host": "http://localhost:11434",
    "embedding_model": "bge-m3:latest",
    "max_tokens": 32768,
    "kv_storage": "JsonKVStorage",
    "doc_status_storage": "JsonDocStatusStorage",
    "graph_storage": "NetworkXStorage",
    "vector_storage": "NanoVectorDBStorage",
    "enable_llm_cache_for_extract": true,
    "enable_llm_cache": true,
    "workspace": "default",
    "max_graph_nodes": 1000,
    "enable_rerank": true,
    "rerank_binding": "cohere",
    "rerank_model": "rerank-english-v3.0",
    "summary_language": "English",
    "force_llm_summary_on_merge": 0,
    "max_parallel_insert": 2,
    "cosine_threshold": 0.2,
    "min_rerank_score": 0.3,
    "related_chunk_number": 30,
    "max_async": 4,
    "embedding_func_max_async": 16,
    "embedding_batch_num": 32
  },
  "auth_mode": "disabled",
  "pipeline_busy": false,
  "keyed_locks": {},
  "core_version": "0.9.8",
  "api_version": "0.9.8",
  "webui_title": "LightRAG WebUI",
  "webui_description": "A lightweight RAG interface"
}
```

#### GET `/`
Redirects to `/webui` for the web interface.

#### GET `/auth-status`
Get authentication configuration and guest token if applicable.

**Response**:
```json
{
  "auth_configured": false,
  "access_token": "guest-token-if-no-auth",
  "token_type": "bearer",
  "auth_mode": "disabled",
  "message": "Authentication is disabled. Using guest access.",
  "core_version": "0.9.8",
  "api_version": "0.9.8",
  "webui_title": "LightRAG WebUI",
  "webui_description": "A lightweight RAG interface"
}
```

### Authentication Endpoints

#### POST `/login`
Authenticate and receive JWT token.

**Request Body**:
```json
{
  "username": "admin",
  "password": "password"
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "auth_mode": "enabled",
  "core_version": "0.9.8",
  "api_version": "0.9.8",
  "webui_title": "LightRAG WebUI",
  "webui_description": "A lightweight RAG interface"
}
```

### Document Management

#### POST `/documents/upload`
Upload a file for document indexing.

**Authentication**: Required
**Content-Type**: `multipart/form-data`

**Parameters**:
- `file` (required): File to upload
- `use_multimodal` (optional, default=true): Whether to use RAG-Anything for multimodal processing

**Response**:
```json
{
  "status": "success",
  "message": "File 'document.pdf' uploaded successfully. Processing will continue in background.",
  "track_id": "upload_20250729_170612_abc123"
}
```

**Supported File Types**:
- Text: `.txt`, `.md`, `.html`, `.htm`, `.tex`, `.json`, `.xml`, `.yaml`, `.yml`, `.rtf`, `.odt`, `.epub`, `.csv`, `.log`, `.conf`, `.ini`, `.properties`, `.sql`, `.bat`, `.sh`
- Office: `.pdf`, `.docx`, `.pptx`, `.xlsx`
- Code: `.c`, `.cpp`, `.py`, `.java`, `.js`, `.ts`, `.swift`, `.go`, `.rb`, `.php`, `.css`, `.scss`, `.less`

#### POST `/documents/upload-multimodal`
Upload a document for advanced multimodal processing using RAG-Anything with MinerU parsing.

**Authentication**: Required
**Content-Type**: `multipart/form-data`

**Parameters**:
- `file` (required): File to upload and process

**Response**:
```json
{
  "status": "success", 
  "message": "File 'document.pdf' uploaded successfully for multimodal processing. MinerU parsing will continue in background.",
  "track_id": "multimodal_upload_20250729_170612_abc123"
}
```

#### POST `/documents/text`
Insert a single text document.

**Authentication**: Required
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "text": "This is a sample text to be inserted into the RAG system.",
  "file_source": "Source of the text (optional)"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Text successfully received. Processing will continue in background.",
  "track_id": "insert_20250729_170612_abc123"
}
```

#### POST `/documents/texts`
Insert multiple text documents.

**Authentication**: Required
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "texts": [
    "This is the first text to be inserted.",
    "This is the second text to be inserted."
  ],
  "file_sources": [
    "First file source (optional)"
  ]
}
```

#### POST `/documents/scan`
Trigger scanning for new documents in the input directory.

**Authentication**: Required

**Response**:
```json
{
  "status": "scanning_started",
  "message": "Scanning process has been initiated in the background",
  "track_id": "scan_20250729_170612_abc123"
}
```

#### GET `/documents`
Get all documents grouped by status.

**Authentication**: Required

**Response**:
```json
{
  "statuses": {
    "PENDING": [
      {
        "id": "doc_123",
        "content_summary": "Pending document",
        "content_length": 5000,
        "status": "PENDING",
        "created_at": "2025-03-31T10:00:00Z",
        "updated_at": "2025-03-31T10:00:00Z",
        "track_id": "upload_20250331_100000_abc123",
        "chunks_count": null,
        "error_msg": null,
        "metadata": null,
        "file_path": "pending_doc.pdf"
      }
    ],
    "PROCESSED": [
      {
        "id": "doc_456",
        "content_summary": "Processed document",
        "content_length": 8000,
        "status": "PROCESSED",
        "created_at": "2025-03-31T09:00:00Z",
        "updated_at": "2025-03-31T09:05:00Z",
        "track_id": "insert_20250331_090000_def456",
        "chunks_count": 8,
        "error_msg": null,
        "metadata": {"author": "John Doe"},
        "file_path": "processed_doc.pdf"
      }
    ]
  }
}
```

#### POST `/documents/paginated`
Get documents with pagination, filtering, and sorting.

**Authentication**: Required

**Request Body**:
```json
{
  "status_filter": "PROCESSED",
  "page": 1,
  "page_size": 50,
  "sort_field": "updated_at",
  "sort_direction": "desc"
}
```

**Response**:
```json
{
  "documents": [
    {
      "id": "doc_123456",
      "content_summary": "Research paper on machine learning",
      "content_length": 15240,
      "status": "PROCESSED",
      "created_at": "2025-03-31T12:34:56Z",
      "updated_at": "2025-03-31T12:35:30Z",
      "track_id": "upload_20250729_170612_abc123",
      "chunks_count": 12,
      "error_msg": null,
      "metadata": {"author": "John Doe", "year": 2025},
      "file_path": "research_paper.pdf"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "total_count": 150,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false
  },
  "status_counts": {
    "PENDING": 10,
    "PROCESSING": 5,
    "PROCESSED": 130,
    "FAILED": 5
  }
}
```

#### GET `/documents/status_counts`
Get count of documents by status.

**Authentication**: Required

**Response**:
```json
{
  "status_counts": {
    "PENDING": 10,
    "PROCESSING": 5,
    "PROCESSED": 130,
    "FAILED": 5
  }
}
```

#### GET `/documents/track_status/{track_id}`
Get processing status of documents by tracking ID.

**Authentication**: Required

**Path Parameters**:
- `track_id` (required): Tracking ID returned from upload/text operations

**Response**:
```json
{
  "track_id": "upload_20250729_170612_abc123",
  "documents": [
    {
      "id": "doc_123456",
      "content_summary": "Research paper on machine learning",
      "content_length": 15240,
      "status": "PROCESSED",
      "created_at": "2025-03-31T12:34:56Z",
      "updated_at": "2025-03-31T12:35:30Z",
      "track_id": "upload_20250729_170612_abc123",
      "chunks_count": 12,
      "error_msg": null,
      "metadata": {"author": "John Doe", "year": 2025},
      "file_path": "research_paper.pdf"
    }
  ],
  "total_count": 1,
  "status_summary": {"PROCESSED": 1}
}
```

#### GET `/documents/pipeline_status`
Get current document processing pipeline status.

**Authentication**: Required

**Response**:
```json
{
  "autoscanned": false,
  "busy": false,
  "job_name": "Default Job",
  "job_start": "2025-03-31T12:34:56Z",
  "docs": 0,
  "batchs": 0,
  "cur_batch": 0,
  "request_pending": false,
  "latest_message": "",
  "history_messages": [],
  "update_status": {}
}
```

#### DELETE `/documents`
Clear all documents from the system.

**Authentication**: Required

**Response**:
```json
{
  "status": "success",
  "message": "All documents cleared successfully. Deleted 15 files."
}
```

#### DELETE `/documents/delete_document`
Delete specific documents by their IDs.

**Authentication**: Required

**Request Body**:
```json
{
  "doc_ids": ["doc_123", "doc_456"],
  "delete_file": false
}
```

**Response**:
```json
{
  "status": "deletion_started",
  "message": "Document deletion for 2 documents has been initiated. Processing will continue in background.",
  "doc_id": "doc_123, doc_456"
}
```

#### DELETE `/documents/delete_entity`
Delete an entity and all its relationships.

**Authentication**: Required

**Request Body**:
```json
{
  "entity_name": "John Doe"
}
```

#### DELETE `/documents/delete_relation`
Delete a relationship between two entities.

**Authentication**: Required

**Request Body**:
```json
{
  "source_entity": "John Doe",
  "target_entity": "Company XYZ"
}
```

#### POST `/documents/clear_cache`
Clear LLM response cache.

**Authentication**: Required

**Request Body**:
```json
{}
```

**Response**:
```json
{
  "status": "success",
  "message": "Successfully cleared all cache"
}
```

### Query Operations

#### POST `/query`
Execute a RAG query and get a complete response.

**Authentication**: Required
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "query": "What is LightRAG?",
  "mode": "hybrid",
  "only_need_context": false,
  "only_need_prompt": false,
  "response_type": "Multiple Paragraphs",
  "top_k": 60,
  "chunk_top_k": 30,
  "max_entity_tokens": 2000,
  "max_relation_tokens": 2000,
  "max_total_tokens": 8000,
  "conversation_history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous response"}
  ],
  "history_turns": 3,
  "ids": ["doc_123", "doc_456"],
  "user_prompt": "Please be concise",
  "enable_rerank": true,
  "multimodal_content": [
    {
      "type": "image",
      "image_data": "base64-encoded-image-data"
    },
    {
      "type": "table",
      "table_data": "CSV or structured table data",
      "table_caption": "Sales data for Q1"
    },
    {
      "type": "equation",
      "latex": "E = mc^2",
      "equation_caption": "Einstein's mass-energy equivalence"
    }
  ]
}
```

**Query Modes**:
- `local`: Entity-based local search
- `global`: Community-based global search  
- `hybrid`: Combination of local and global
- `naive`: Simple vector similarity search
- `mix`: Adaptive mode selection
- `bypass`: Direct LLM query without RAG

**Response**:
```json
{
  "response": "LightRAG is a fast, lightweight Retrieval-Augmented Generation system..."
}
```

#### POST `/query/stream`
Execute a RAG query and stream the response.

**Authentication**: Required
**Content-Type**: `application/json`

**Request Body**: Same as `/query`

**Response**: Server-Sent Events stream
```
{"response": "LightRAG"}\n
{"response": " is a"}\n
{"response": " fast"}\n
...
```

#### POST `/query/multimodal`
Execute a multimodal query with enhanced RAG-Anything support.

**Authentication**: Required
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "query": "Analyze this image and related documents",
  "mode": "hybrid",
  "multimodal_content": [
    {
      "type": "image",
      "image_data": "base64-encoded-image-data"
    }
  ],
  "context_window": 1,
  "max_context_tokens": 2000,
  "enable_image_processing": true,
  "enable_table_processing": true,
  "enable_equation_processing": true
}
```

**Response**:
```json
{
  "response": "Based on the image analysis and related documents..."
}
```

### Graph Operations

#### GET `/graphs`
Retrieve knowledge graph data for a specific label.

**Authentication**: Required

**Query Parameters**:
- `label` (required): Label to get knowledge graph for
- `max_depth` (default=3): Maximum depth of graph
- `max_nodes` (default=1000): Maximum nodes to return

**Response**:
```json
{
  "nodes": [
    {
      "id": "entity_1",
      "label": "Person", 
      "properties": {"name": "John Doe", "age": 30}
    }
  ],
  "edges": [
    {
      "source": "entity_1",
      "target": "entity_2", 
      "relationship": "WORKS_FOR",
      "properties": {"since": "2020"}
    }
  ]
}
```

#### GET `/graph/label/list`
Get all available graph labels.

**Authentication**: Required

**Response**:
```json
[
  "Person",
  "Company", 
  "Technology",
  "Project"
]
```

#### GET `/graph/entity/exists`
Check if an entity exists in the knowledge graph.

**Authentication**: Required

**Query Parameters**:
- `name` (required): Entity name to check

**Response**:
```json
{
  "exists": true
}
```

#### POST `/graph/entity/edit`
Update an entity's properties.

**Authentication**: Required

**Request Body**:
```json
{
  "entity_name": "John Doe",
  "updated_data": {
    "age": 31,
    "title": "Senior Developer"
  },
  "allow_rename": false
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Entity updated successfully",
  "data": {}
}
```

#### POST `/graph/relation/edit`
Update a relationship's properties.

**Authentication**: Required

**Request Body**:
```json
{
  "source_id": "John Doe",
  "target_id": "Company XYZ",
  "updated_data": {
    "role": "Lead Developer",
    "start_date": "2020-01-01"
  }
}
```

### Ollama API Emulation

The LightRAG server provides Ollama-compatible endpoints for integration with Ollama-based applications like Open WebUI.

#### GET `/api/version`
Get Ollama version information.

**Authentication**: Required (unless whitelisted)

**Response**:
```json
{
  "version": "0.9.3"
}
```

#### GET `/api/tags`
List available models (returns LightRAG as an Ollama model).

**Authentication**: Required (unless whitelisted)

**Response**:
```json
{
  "models": [
    {
      "name": "lightrag:latest",
      "model": "lightrag:latest", 
      "modified_at": "2025-03-31T12:00:00Z",
      "size": 1073741824,
      "digest": "sha256:abc123...",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "lightrag",
        "families": ["lightrag"],
        "parameter_size": "13B",
        "quantization_level": "Q4_0"
      }
    }
  ]
}
```

#### GET `/api/ps`
List running models.

**Authentication**: Required (unless whitelisted)

**Response**:
```json
{
  "models": [
    {
      "name": "lightrag:latest",
      "model": "lightrag:latest",
      "size": 1073741824,
      "digest": "sha256:abc123...",
      "details": {
        "parent_model": "",
        "format": "gguf", 
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "7.2B",
        "quantization_level": "Q4_0"
      },
      "expires_at": "2050-12-31T14:38:31Z",
      "size_vram": 1073741824
    }
  ]
}
```

#### POST `/api/generate`
Generate completion (Ollama-compatible endpoint).

**Authentication**: Required (unless whitelisted)

**Request Body**:
```json
{
  "model": "lightrag:latest",
  "prompt": "What is artificial intelligence?",
  "system": "You are a helpful AI assistant.",
  "stream": false,
  "options": {}
}
```

**Response**:
```json
{
  "model": "lightrag:latest",
  "created_at": "2025-03-31T12:34:56Z", 
  "response": "Artificial intelligence (AI) refers to...",
  "done": true,
  "context": [],
  "total_duration": 1500000000,
  "load_duration": 0,
  "prompt_eval_count": 15,
  "prompt_eval_duration": 500000000,
  "eval_count": 50,
  "eval_duration": 1000000000
}
```

#### POST `/api/chat`
Chat completion (Ollama-compatible endpoint with RAG capabilities).

**Authentication**: Required (unless whitelisted)

**Request Body**:
```json
{
  "model": "lightrag:latest",
  "messages": [
    {"role": "user", "content": "What is LightRAG?"}
  ],
  "stream": false,
  "options": {},
  "system": "You are a helpful AI assistant."
}
```

**Special Query Prefixes for RAG Mode Control**:
- `/local` - Entity-based local search
- `/global` - Community-based global search  
- `/hybrid` - Combination mode (default)
- `/naive` - Simple vector search
- `/mix` - Adaptive mode selection
- `/bypass` - Direct LLM query without RAG
- `/context` - Return context only
- `/[prompt] query` - Add custom user prompt

**Examples**:
```json
{"role": "user", "content": "/local What is the main character?"}
{"role": "user", "content": "/[use bullet points] Summarize the key findings"}
{"role": "user", "content": "/bypass What's the weather today?"}
```

**Response**:
```json
{
  "model": "lightrag:latest",
  "created_at": "2025-03-31T12:34:56Z",
  "message": {
    "role": "assistant",
    "content": "LightRAG is a fast, lightweight...",
    "images": null
  },
  "done": true,
  "total_duration": 1500000000,
  "load_duration": 0,
  "prompt_eval_count": 15,
  "prompt_eval_duration": 500000000,
  "eval_count": 50,
  "eval_duration": 1000000000
}
```

## Error Handling

### Standard HTTP Status Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required/failed
- `403 Forbidden` - Access denied (invalid API key)
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

### Common Error Responses

#### Authentication Error
```json
{
  "detail": "Invalid API Key"
}
```

#### Validation Error
```json
{
  "detail": "Query cannot be empty"
}
```

#### Pipeline Busy Error
```json
{
  "detail": "Cannot perform operation while pipeline is busy"
}
```

#### File Upload Error
```json
{
  "status": "failure",
  "message": "Unsupported file type. Supported types: ['.txt', '.pdf', ...]",
  "track_id": ""
}
```

## Environment Configuration

### Core Settings
```bash
# Server Configuration
HOST=0.0.0.0
PORT=9621
WORKERS=2
TIMEOUT=150

# Authentication
LIGHTRAG_API_KEY=your-secure-api-key-here
AUTH_ACCOUNTS='admin:admin123,user1:pass456'  
TOKEN_SECRET=your-jwt-secret
TOKEN_EXPIRE_HOURS=48
WHITELIST_PATHS=/health,/api/*

# LLM Configuration
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your-api-key

# Embedding Configuration  
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
EMBEDDING_BINDING_HOST=http://localhost:11434

# RAG Configuration
WORKING_DIR=./rag_storage
INPUT_DIR=./inputs
WORKSPACE=default
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
MAX_ASYNC=4
MAX_PARALLEL_INSERT=2
TOP_K=60
CHUNK_TOP_K=30
MAX_ENTITY_TOKENS=2000
MAX_RELATION_TOKENS=2000
MAX_TOTAL_TOKENS=8000
COSINE_THRESHOLD=0.2
SUMMARY_LANGUAGE=English
ENABLE_LLM_CACHE=true
ENABLE_LLM_CACHE_FOR_EXTRACT=true

# Storage Configuration
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage  
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage
LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage

# Rerank Configuration
RERANK_BINDING=cohere
RERANK_MODEL=rerank-english-v3.0
RERANK_BINDING_HOST=https://api.cohere.ai/v1
RERANK_BINDING_API_KEY=your-rerank-api-key
MIN_RERANK_SCORE=0.3

# Multimodal Processing (RAG-Anything)
PARSE_METHOD=auto
PARSER=mineru
ENABLE_IMAGE_PROCESSING=true
ENABLE_TABLE_PROCESSING=true
ENABLE_EQUATION_PROCESSING=true
CONTEXT_WINDOW=1
MAX_CONTEXT_TOKENS=2000
```

### Storage Options

**KV Storage**: `JsonKVStorage`, `RedisKVStorage`, `PGKVStorage`, `MongoKVStorage`

**Vector Storage**: `NanoVectorDBStorage`, `FaissVectorDBStorage`, `MilvusVectorDBStorage`, `QdrantVectorDBStorage`, `PGVectorStorage`, `MongoVectorDBStorage`

**Graph Storage**: `NetworkXStorage`, `Neo4JStorage`, `MemgraphStorage`, `PGGraphStorage`, `MongoGraphStorage`

**Document Status Storage**: `JsonDocStatusStorage`, `RedisDocStatusStorage`, `PGDocStatusStorage`, `MongoDocStatusStorage`

---

This documentation covers all major API endpoints and configuration options for the LightRAG backend system. For the most up-to-date information, consult the interactive API documentation at `/docs` or `/redoc` when the server is running.