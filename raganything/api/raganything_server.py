"""
RAGAnything FastAPI Server - Complete LightRAG UI Integration

This server integrates the LightRAG Web UI with RAG-Anything backend,
providing full multimodal document processing and query capabilities.

Based on LightRAG's lightrag_server.py with RAG-Anything backend integration.
"""

import os
import sys
import asyncio
import logging
import uvicorn
import signal
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import tempfile
import shutil

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from project root
load_dotenv(dotenv_path=project_root / ".env", override=False)

# Configure logging early
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RAGAnything and related modules
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from raganything.modalprocessors import (
        ImageModalProcessor,
        TableModalProcessor,
        EquationModalProcessor,
        ContextExtractor,
        ContextConfig,
    )
    logger.info("RAGAnything modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import RAGAnything modules: {e}")
    raise

# Import required LightRAG utilities for compatibility
try:
    from lightrag.utils import EmbeddingFunc
    logger.info("LightRAG utilities imported successfully")
except ImportError as e:
    logger.warning(f"LightRAG utilities not available: {e}")
    # Create a minimal EmbeddingFunc if not available
    class EmbeddingFunc:
        def __init__(self, embedding_dim, func):
            self.embedding_dim = embedding_dim
            self.func = func

# ===== MODEL FUNCTION IMPLEMENTATIONS =====

def create_deepseek_chat_model_func():
    """Create DeepSeek chat model function"""
    
    async def chat_model_func(prompt: str, system_prompt: str = None, history_messages: List = None, **kwargs):
        """DeepSeek chat model function"""
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=os.getenv("LLM_BINDING_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com")
        )
        
        try:
            msgs = []
            
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            
            if history_messages:
                msgs.extend(history_messages)
                
            msgs.append({"role": "user", "content": prompt})
            
            response = await client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "deepseek-chat"),
                messages=msgs,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"DeepSeek chat model call failed: {e}")
            raise HTTPException(status_code=500, detail=f"Chat model error: {str(e)}")
    
    return chat_model_func


def create_deepseek_vision_model_func():
    """Create DeepSeek vision model function with proper multimodal support"""
    
    def vision_model_func(prompt: str, system_prompt: str = None, history_messages: List = None, 
                         image_data=None, messages=None, **kwargs):
        """DeepSeek VL model function that handles both text and vision"""
        import openai
        
        # Initialize OpenAI client for DeepSeek
        client = openai.OpenAI(
            api_key=os.getenv("VISION_MODEL_API_KEY", os.getenv("LLM_BINDING_API_KEY")),
            base_url=os.getenv("VISION_MODEL_BASE_URL", "https://api.deepseek.com")
        )
        
        try:
            # If messages format is provided directly, use it
            if messages:
                response = client.chat.completions.create(
                    model=os.getenv("VISION_MODEL", "deepseek-vl"),
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 4096),
                    temperature=kwargs.get("temperature", 0.7),
                )
                return response.choices[0].message.content
            
            # Build messages manually
            msgs = []
            
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            
            if history_messages:
                msgs.extend(history_messages)
            
            # Handle multimodal content with proper format
            if image_data:
                user_content = []
                
                # Add text part
                if prompt:
                    user_content.append({"type": "text", "text": prompt})
                
                # Add images using DeepSeek VL format
                if isinstance(image_data, str):
                    # Single image as base64
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
                elif isinstance(image_data, list):
                    # Multiple images
                    for img in image_data:
                        user_content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                        })
                
                msgs.append({"role": "user", "content": user_content})
            else:
                # Text only
                msgs.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=os.getenv("VISION_MODEL", "deepseek-vl"),
                messages=msgs,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"DeepSeek VL model call failed: {e}")
            raise HTTPException(status_code=500, detail=f"Vision model error: {str(e)}")
    
    return vision_model_func


def create_qwen_embedding_func():
    """Create local Qwen3 embedding function"""
    
    # Global variables to store model and tokenizer
    _qwen_tokenizer = None
    _qwen_model = None
    
    def load_qwen_model():
        """Load Qwen embedding model locally"""
        nonlocal _qwen_tokenizer, _qwen_model
        
        if _qwen_tokenizer is not None and _qwen_model is not None:
            return _qwen_tokenizer, _qwen_model
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Configuration from environment
            model_name = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
            device_config = os.getenv("EMBEDDING_DEVICE", "auto")
            cache_dir = os.getenv("MODEL_CACHE_DIR", "./models")
            
            # Handle device configuration
            if device_config == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = device_config
            
            logger.info(f"Loading Qwen3 embedding model: {model_name}")
            logger.info(f"Device: {device}, Cache dir: {cache_dir}")
            
            # Load tokenizer
            _qwen_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            
            # Load model
            _qwen_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            logger.info(f"✅ Qwen3 embedding model loaded successfully")
            return _qwen_tokenizer, _qwen_model
            
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen3 embedding model: {e}")
            raise
    
    async def embedding_func_impl(texts: List[str]):
        """Local Qwen3 embedding function implementation"""
        try:
            import torch
            
            tokenizer, model = load_qwen_model()
            device = next(model.parameters()).device
            
            # Batch processing configuration
            batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
            max_length = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
            all_embeddings = []
            
            logger.debug(f"Processing {len(texts)} texts in batches of {batch_size}")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize input
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Average pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    # L2 normalization
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.extend(embeddings.cpu().numpy().tolist())
            
            logger.debug(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ Local Qwen embedding failed: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")
    
    # Return EmbeddingFunc wrapper
    return EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        func=embedding_func_impl
    )


# ===== GLOBAL VARIABLES =====

rag_anything: Optional[RAGAnything] = None


# ===== PYDANTIC MODELS =====

class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str
    mode: str = "mix"
    multimodal_content: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    only_need_context: Optional[bool] = None
    only_need_prompt: Optional[bool] = None
    response_type: Optional[str] = None
    top_k: Optional[int] = None
    chunk_top_k: Optional[int] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    working_directory: str
    input_directory: str
    configuration: Dict[str, Any]
    auth_mode: str = "disabled"
    core_version: str
    api_version: str
    webui_title: str
    webui_description: str


# ===== DOCUMENT MANAGEMENT SIMULATION =====

class DocumentTracker:
    """Simple document tracking for LightRAG compatibility"""
    
    def __init__(self):
        self._documents = {}
        self._counter = 0
    
    def add_document(self, filename: str, file_path: str, result: Dict) -> str:
        """Add a document to tracking"""
        self._counter += 1
        doc_id = f"doc_{self._counter}"
        
        doc_info = {
            "id": doc_id,
            "filename": filename,
            "file_path": file_path,
            "status": "processed",
            "created_at": "2025-01-01T00:00:00Z",
            "file_size": Path(file_path).stat().st_size if Path(file_path).exists() else 0,
            "result": result
        }
        
        self._documents[doc_id] = doc_info
        return doc_id
    
    def get_documents(self, page: int = 1, page_size: int = 10) -> Dict:
        """Get paginated documents"""
        documents = list(self._documents.values())
        total = len(documents)
        start = (page - 1) * page_size
        end = start + page_size
        
        paginated_docs = documents[start:end]
        total_pages = (total + page_size - 1) // page_size
        
        status_counts = {
            "all": total,
            "processed": len([d for d in documents if d.get("status") == "processed"]),
            "processing": 0,
            "pending": 0,
            "failed": 0
        }
        
        return {
            "documents": paginated_docs,
            "pagination": {
                "total_count": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "status_counts": status_counts
        }
    
    def clear_documents(self):
        """Clear all documents"""
        self._documents.clear()
        self._counter = 0


# Global document tracker
doc_tracker = DocumentTracker()


# ===== INITIALIZATION FUNCTIONS =====

async def initialize_rag_anything():
    """Initialize RAGAnything instance with proper configuration"""
    global rag_anything
    
    try:
        logger.info("Initializing RAGAnything with configuration from environment...")
        
        # Create configuration based on .env file
        config = RAGAnythingConfig(
            working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
            parse_method=os.getenv("PARSE_METHOD", "auto"),
            parser=os.getenv("PARSER", "mineru"),
            parser_output_dir=os.getenv("OUTPUT_DIR", os.getenv("INPUT_DIR", "./inputs")),
            display_content_stats=os.getenv("DISPLAY_CONTENT_STATS", "true").lower() == "true",
            
            # Multimodal Processing
            enable_image_processing=os.getenv("ENABLE_IMAGE_PROCESSING", "true").lower() == "true",
            enable_table_processing=os.getenv("ENABLE_TABLE_PROCESSING", "true").lower() == "true",
            enable_equation_processing=os.getenv("ENABLE_EQUATION_PROCESSING", "true").lower() == "true",
            
            # Batch Processing
            max_concurrent_files=int(os.getenv("MAX_CONCURRENT_FILES", "4")),
            recursive_folder_processing=os.getenv("RECURSIVE_FOLDER_PROCESSING", "true").lower() == "true",
            
            # Context Extraction
            context_window=int(os.getenv("CONTEXT_WINDOW", "1")),
            context_mode=os.getenv("CONTEXT_MODE", "page"),
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "2000")),
            include_headers=os.getenv("INCLUDE_HEADERS", "true").lower() == "true",
            include_captions=os.getenv("INCLUDE_CAPTIONS", "true").lower() == "true",
            content_format=os.getenv("CONTENT_FORMAT", "minerU"),
        )
        
        # Create model functions
        chat_model_func = create_deepseek_chat_model_func()
        vision_model_func = create_deepseek_vision_model_func()
        embedding_func = create_qwen_embedding_func()
        
        # LightRAG configuration for backend storage
        lightrag_kwargs = {
            "working_dir": config.working_dir,
            "workspace": os.getenv("WORKSPACE"),
            
            # Storage configuration - use environment variables or defaults
            "kv_storage": os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
            "graph_storage": os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),  
            "vector_storage": os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
            "doc_status_storage": os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
            
            # Processing configuration
            "chunk_token_size": int(os.getenv("CHUNK_SIZE", "1200")),
            "chunk_overlap_token_size": int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
            "llm_model_name": os.getenv("LLM_MODEL", "deepseek-chat"),
            "llm_model_max_async": int(os.getenv("MAX_ASYNC", "4")),
            "summary_max_tokens": int(os.getenv("SUMMARY_MAX_TOKENS", "30000")),
            "max_parallel_insert": int(os.getenv("MAX_PARALLEL_INSERT", "2")),
            "max_graph_nodes": int(os.getenv("MAX_GRAPH_NODES", "1000")),
            "addon_params": {"language": os.getenv("SUMMARY_LANGUAGE", "English")},
            "enable_llm_cache": os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true",
            "enable_llm_cache_for_entity_extract": os.getenv("ENABLE_LLM_CACHE_FOR_EXTRACT", "true").lower() == "true",
        }
        
        logger.info(f"RAGAnything config: working_dir={config.working_dir}, parser={config.parser}")
        logger.info(f"Storage config: kv={lightrag_kwargs['kv_storage']}, graph={lightrag_kwargs['graph_storage']}")
        
        # Create RAGAnything instance with proper parameters
        rag_anything = RAGAnything(
            config=config,
            llm_model_func=chat_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs=lightrag_kwargs
        )
        
        # Ensure LightRAG is properly initialized
        await rag_anything._ensure_lightrag_initialized()
        
        logger.info("✅ RAGAnything initialized successfully")
        logger.info(f"Working directory: {rag_anything.working_dir}")
        
        return rag_anything
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAGAnything: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    
    def signal_handler(sig, frame):
        print(f"\n\nReceived signal {sig}, shutting down gracefully...")
        print(f"Process ID: {os.getpid()}")
        
        # Cleanup
        if rag_anything:
            try:
                import asyncio
                if asyncio.get_event_loop().is_running():
                    asyncio.create_task(rag_anything.finalize_storages())
                else:
                    asyncio.run(rag_anything.finalize_storages())
            except Exception as e:
                print(f"Warning: Failed to finalize RAGAnything storages: {e}")
        
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command


# ===== FASTAPI APPLICATION =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    global rag_anything
    
    try:
        logger.info("🚀 Starting RAGAnything server initialization...")
        
        # Initialize RAGAnything
        rag_anything = await initialize_rag_anything()
        
        logger.info("✅ Server is ready to accept connections! 🚀")
        yield
        
    finally:
        # Cleanup
        if rag_anything:
            try:
                await rag_anything.finalize_storages()
                logger.info("✅ RAGAnything storages finalized")
            except Exception as e:
                logger.error(f"❌ Error finalizing storages: {e}")


def create_app():
    """Create FastAPI application"""
    
    app = FastAPI(
        title="RAGAnything Server API",
        description="Multimodal RAG system with LightRAG UI - DeepSeek & Qwen3 Integration",
        version="1.0.0",
        lifespan=lifespan,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Get CORS origins
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    if cors_origins == "*":
        origins = ["*"]
    else:
        origins = [origin.strip() for origin in cors_origins.split(",")]
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ===== CORE ROUTES =====
    
    @app.get("/")
    async def redirect_to_webui():
        """Redirect root path to static files"""
        return RedirectResponse(url="/static/")
    
    @app.get("/webui")
    async def redirect_to_static():
        """Redirect /webui to static files"""
        return RedirectResponse(url="/static/")
    
    @app.get("/auth-status")
    async def get_auth_status():
        """Get authentication status - currently disabled for simplicity"""
        return {
            "auth_configured": False,
            "access_token": "guest_token",
            "token_type": "bearer", 
            "auth_mode": "disabled",
            "message": "Authentication is disabled. Using guest access.",
            "core_version": "1.0.0",
            "api_version": "1.0.0",
            "webui_title": os.getenv("WEBUI_TITLE", "RAGAnything"),
            "webui_description": os.getenv("WEBUI_DESCRIPTION", "Multimodal RAG System"),
        }
    
    @app.post("/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        """Login endpoint - returns guest token since auth is disabled"""
        return {
            "access_token": "guest_token",
            "token_type": "bearer",
            "auth_mode": "disabled",
            "message": "Authentication is disabled. Using guest access.",
            "core_version": "1.0.0",
            "api_version": "1.0.0",
            "webui_title": os.getenv("WEBUI_TITLE", "RAGAnything"),
            "webui_description": os.getenv("WEBUI_DESCRIPTION", "Multimodal RAG System"),
        }
    
    @app.get("/health") 
    async def get_health() -> HealthResponse:
        """Get system health status"""
        if not rag_anything:
            raise HTTPException(status_code=503, detail="RAGAnything not initialized")
        
        try:
            config_info = rag_anything.get_config_info()
            
            return HealthResponse(
                status="healthy",
                working_directory=str(rag_anything.working_dir),
                input_directory=config_info["directory"]["parser_output_dir"],
                auth_mode="disabled",
                core_version="1.0.0",
                api_version="1.0.0", 
                webui_title=os.getenv("WEBUI_TITLE", "RAGAnything"),
                webui_description=os.getenv("WEBUI_DESCRIPTION", "Multimodal RAG System"),
                configuration={
                    # LLM configuration
                    "llm_binding": os.getenv("LLM_BINDING", "openai"),
                    "llm_model": os.getenv("LLM_MODEL", "deepseek-chat"),
                    "llm_binding_host": os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
                    
                    # Embedding configuration  
                    "embedding_binding": os.getenv("EMBEDDING_BINDING", "openai"),
                    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                    "embedding_binding_host": os.getenv("EMBEDDING_BINDING_HOST"),
                    "embedding_dim": int(os.getenv("EMBEDDING_DIM", "3072")),
                    
                    # Storage configuration from LightRAG kwargs
                    "kv_storage": os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
                    "graph_storage": os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
                    "vector_storage": os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
                    "doc_status_storage": os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
                    
                    # RAGAnything specific
                    "parser": config_info["parsing"]["parser"],
                    "parse_method": config_info["parsing"]["parse_method"],
                    "multimodal_processing": config_info["multimodal_processing"],
                    "context_extraction": config_info["context_extraction"],
                    "batch_processing": config_info["batch_processing"],
                    
                    # Processing settings
                    "max_tokens": int(os.getenv("SUMMARY_MAX_TOKENS", "30000")),
                    "chunk_token_size": int(os.getenv("CHUNK_SIZE", "1200")),
                    "chunk_overlap_token_size": int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
                    "workspace": os.getenv("WORKSPACE"),
                    "max_graph_nodes": int(os.getenv("MAX_GRAPH_NODES", "1000")),
                    "enable_llm_cache": os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true",
                    "enable_llm_cache_for_extract": os.getenv("ENABLE_LLM_CACHE_FOR_EXTRACT", "true").lower() == "true",
                    "summary_language": os.getenv("SUMMARY_LANGUAGE", "English"),
                    "max_parallel_insert": int(os.getenv("MAX_PARALLEL_INSERT", "2")),
                    "max_async": int(os.getenv("MAX_ASYNC", "4")),
                }
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")
    
    # ===== DOCUMENT MANAGEMENT ROUTES =====
    
    @app.post("/documents/upload")
    async def upload_document(file: UploadFile = File(...)):
        """Upload and process document using RAGAnything backend"""
        if not rag_anything:
            raise HTTPException(status_code=503, detail="RAGAnything not initialized")
        
        try:
            logger.info(f"📄 Processing uploaded file: {file.filename}")
            
            # Create temporary directory for uploads
            temp_dir = Path("./temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            temp_file_path = temp_dir / file.filename
            
            # Save uploaded file
            with open(temp_file_path, "wb") as buffer:
                content_bytes = await file.read()
                buffer.write(content_bytes)
                logger.info(f"💾 Saved file to: {temp_file_path} ({len(content_bytes)} bytes)")
            
            # Process through RAGAnything
            logger.info("🔄 Starting document processing...")
            result = await rag_anything.process_document_complete(str(temp_file_path))
            
            # Track document
            doc_id = doc_tracker.add_document(file.filename, str(temp_file_path), result)
            
            # Clean up temporary file
            temp_file_path.unlink(missing_ok=True)
            
            logger.info(f"✅ Document processed successfully: {doc_id}")
            
            return {
                "status": "success",
                "message": f"Document {file.filename} processed successfully",
                "doc_id": doc_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"❌ Document upload failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/insert")
    async def insert_document_api(
        file: Optional[UploadFile] = File(None),
        content: Optional[str] = Form(None),
        file_path: Optional[str] = Form(None)
    ):
        """API insert endpoint compatible with LightRAG frontend"""
        if file:
            return await upload_document(file)
        elif file_path:
            if not rag_anything:
                raise HTTPException(status_code=503, detail="RAGAnything not initialized")
            
            path = Path(file_path)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
            try:
                logger.info(f"📄 Processing file path: {file_path}")
                result = await rag_anything.process_document_complete(str(path))
                doc_id = doc_tracker.add_document(path.name, str(path), result)
                
                return {
                    "status": "success",
                    "message": f"Document processed successfully",
                    "doc_id": doc_id,
                    "result": result
                }
            except Exception as e:
                logger.error(f"❌ File path processing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        elif content:
            if not rag_anything:
                raise HTTPException(status_code=503, detail="RAGAnything not initialized")
            
            try:
                # For direct text, we need to use LightRAG directly
                logger.info("📝 Processing direct text content...")
                lightrag_instance = rag_anything.lightrag
                if lightrag_instance:
                    await lightrag_instance.ainsert(content)
                    
                    doc_id = doc_tracker.add_document("direct_text.txt", "content", {"chunks_inserted": 1})
                    
                    return {
                        "status": "success",
                        "message": "Text content processed successfully",
                        "doc_id": doc_id
                    }
                else:
                    raise Exception("LightRAG instance not available")
                    
            except Exception as e:
                logger.error(f"❌ Text content processing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either file upload, file_path, or content must be provided"
            )
    
    @app.get("/documents")
    async def get_documents():
        """Get document list for LightRAG frontend compatibility"""
        try:
            result = doc_tracker.get_documents()
            return result
        except Exception as e:
            logger.error(f"❌ Failed to get documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/documents/paginated")
    async def get_documents_paginated(request: dict):
        """Get paginated document list"""
        try:
            page = request.get("page", 1)
            page_size = request.get("page_size", 10)
            result = doc_tracker.get_documents(page=page, page_size=page_size)
            return result
        except Exception as e:
            logger.error(f"❌ Failed to get paginated documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/documents")
    async def clear_documents():
        """Clear all documents"""
        try:
            doc_tracker.clear_documents()
            return {"status": "success", "message": "All documents cleared"}
        except Exception as e:
            logger.error(f"❌ Failed to clear documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/documents/scan")
    async def scan_documents():
        """Document scanning placeholder"""
        try:
            documents = doc_tracker.get_documents()
            return {
                "status": "success", 
                "scanned": documents["pagination"]["total_count"],
                "message": f"Found {documents['pagination']['total_count']} documents"
            }
        except Exception as e:
            logger.error(f"❌ Failed to scan documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ===== QUERY ROUTES =====
    
    @app.post("/query")
    async def query_documents(request: QueryRequest):
        """Query documents using RAGAnything backend"""
        if not rag_anything:
            raise HTTPException(status_code=503, detail="RAGAnything not initialized")
        
        try:
            logger.info(f"🔍 Processing query: {request.query[:100]}...")
            logger.info(f"Query mode: {request.mode}")
            
            # Use RAGAnything's multimodal query capability
            # Only pass core parameters that RAGAnything expects
            try:
                result = await rag_anything.aquery_with_multimodal(
                    query=request.query,
                    multimodal_content=request.multimodal_content or [],
                    mode=request.mode
                )
            except Exception as e:
                logger.error(f"RAGAnything query failed, trying direct LightRAG query: {e}")
                # Fallback to direct LightRAG query if RAGAnything method fails
                lightrag_instance = rag_anything.lightrag
                if lightrag_instance:
                    # Build LightRAG query parameters
                    lightrag_params = {
                        "query": request.query,
                        "param": {
                            "mode": request.mode,
                        }
                    }
                    
                    # Add LightRAG-specific parameters if provided
                    if request.only_need_context is not None:
                        lightrag_params["param"]["only_need_context"] = request.only_need_context
                    if request.only_need_prompt is not None:
                        lightrag_params["param"]["only_need_prompt"] = request.only_need_prompt
                    if request.response_type is not None:
                        lightrag_params["param"]["response_type"] = request.response_type
                    if request.top_k is not None:
                        lightrag_params["param"]["top_k"] = request.top_k
                    if request.chunk_top_k is not None:
                        lightrag_params["param"]["chunk_top_k"] = request.chunk_top_k
                    
                    logger.debug(f"LightRAG query parameters: {lightrag_params}")
                    result = await lightrag_instance.aquery(**lightrag_params)
                else:
                    raise e
            
            logger.info(f"✅ Query completed successfully")
            
            return {
                "status": "success",
                "result": result,
                "query": request.query,
                "mode": request.mode,
                "multimodal_content_count": len(request.multimodal_content) if request.multimodal_content else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/query")
    async def query_documents_api(request: QueryRequest):
        """API query endpoint - delegates to main query"""
        return await query_documents(request)
    
    @app.post("/query/stream")
    async def query_stream(request: QueryRequest):
        """Stream query endpoint (non-streaming for now)"""
        # For simplicity, return non-streaming result
        return await query_documents(request)
    
    # ===== GRAPH ROUTES =====
    
    @app.get("/graphs")
    async def get_graphs(label: str = "", max_depth: int = 3, max_nodes: int = 1000):
        """Get knowledge graph data"""
        if not rag_anything:
            raise HTTPException(status_code=503, detail="RAGAnything not initialized")
        
        try:
            logger.info(f"🕸️ Retrieving knowledge graph: label={label}, max_nodes={max_nodes}")
            
            # Access the internal LightRAG instance
            lightrag_instance = rag_anything.lightrag
            if lightrag_instance and hasattr(lightrag_instance, 'get_knowledge_graph'):
                result = await lightrag_instance.get_knowledge_graph(node_label=label or "*")
                logger.info(f"✅ Graph retrieved: {len(result.get('nodes', []))} nodes, {len(result.get('edges', []))} edges")
                return result
            else:
                # Return empty graph structure
                return {
                    "nodes": [],
                    "edges": [],
                    "metadata": {
                        "total_nodes": 0,
                        "total_edges": 0,
                        "max_depth": max_depth,
                        "max_nodes": max_nodes,
                        "label": label or "*"
                    }
                }
        except Exception as e:
            logger.error(f"❌ Failed to get graphs: {e}")
            return {
                "nodes": [],
                "edges": [], 
                "metadata": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "max_depth": max_depth,
                    "max_nodes": max_nodes,
                    "label": label or "*",
                    "error": str(e)
                }
            }
    
    @app.get("/graph/label/list")
    async def get_graph_labels():
        """Get available graph labels"""
        try:
            # Access the internal LightRAG instance
            if rag_anything and rag_anything.lightrag:
                lightrag_instance = rag_anything.lightrag
                if hasattr(lightrag_instance, 'get_graph_labels'):
                    result = await lightrag_instance.get_graph_labels()
                    return result
            
            # Return default labels
            return ["*", "person", "organization", "location", "concept"]
            
        except Exception as e:
            logger.error(f"❌ Failed to get graph labels: {e}")
            return ["*"]
    
    # Additional graph endpoints for LightRAG compatibility
    @app.post("/graph/entity/edit")
    async def edit_graph_entity(request: dict):
        """Edit graph entity (placeholder)"""
        return {"status": "success", "message": "Entity edit not implemented"}
    
    @app.post("/graph/relation/edit")
    async def edit_graph_relation(request: dict):
        """Edit graph relation (placeholder)"""
        return {"status": "success", "message": "Relation edit not implemented"}
    
    @app.get("/graph/entity/exists")
    async def check_entity_exists(name: str):
        """Check if entity exists (placeholder)"""
        return {"exists": False}
    
    # ===== STATIC FILE SERVING =====
    
    class SmartStaticFiles(StaticFiles):
        """Custom StaticFiles class for smart caching from LightRAG"""
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            
            if path.endswith(".html"):
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            elif "/assets/" in path:
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            
            # Ensure correct Content-Type
            if path.endswith(".js"):
                response.headers["Content-Type"] = "application/javascript"
            elif path.endswith(".css"):
                response.headers["Content-Type"] = "text/css"
            
            return response
    
    # Mount static files (LightRAG UI)
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", SmartStaticFiles(directory=static_path, html=True), name="static")
        logger.info(f"✅ Static files mounted from: {static_path}")
    else:
        logger.error(f"❌ Static files directory not found: {static_path}")
        # This is a critical error since we need the UI
        raise FileNotFoundError(f"Static files directory missing: {static_path}")
    
    return app


# Create the app instance
app = create_app()


def main():
    """Main function to run the server"""
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    logger.info("🚀 Starting RAGAnything Server with LightRAG UI Integration...")
    logger.info(f"📁 Working directory: {os.getenv('WORKING_DIR', './rag_storage')}")
    logger.info(f"📁 Input directory: {os.getenv('INPUT_DIR', './inputs')}")
    logger.info(f"⚙️ Parser: {os.getenv('PARSER', 'mineru')}")
    logger.info(f"🤖 LLM Model: {os.getenv('LLM_MODEL', 'deepseek-chat')}")
    logger.info(f"🔤 Embedding Model: {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')}")
    
    # Check critical environment variables
    if not os.getenv("LLM_BINDING_API_KEY"):
        logger.warning("⚠️ LLM_BINDING_API_KEY not set - model calls may fail")
    if not os.getenv("EMBEDDING_BINDING_API_KEY"):
        logger.warning("⚠️ EMBEDDING_BINDING_API_KEY not set - embedding calls may fail")
    
    # Run the server
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "9000")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    main()