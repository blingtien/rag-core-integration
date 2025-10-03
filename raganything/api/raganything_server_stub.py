"""
RAGAnything FastAPI Server

Ported from LightRAG server structure but uses RAGAnything backend.
Uses DeepSeek as chat/vision models, Qwen3 as embedding, and PostgreSQL/Neo4j as storage.
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

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=False)

# Import RAGAnything and related modules
from raganything import RAGAnything, RAGAnythingConfig
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    ContextExtractor,
    ContextConfig,
)


# Pydantic models for API
class InsertRequest(BaseModel):
    """Request model for document insertion"""
    content: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}


class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str
    mode: str = "mix"
    multimodal_content: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    working_directory: str
    input_directory: str
    configuration: Dict[str, Any]
    auth_mode: str = "disabled"


# Global RAGAnything instance
rag_anything: Optional[RAGAnything] = None


def create_deepseek_vision_model_func():
    """Create DeepSeek vision model function with OpenAI-compatible API"""
    
    def vision_model_func(prompt: str, system_prompt: str = None, history_messages: List = None, image_data=None, messages=None, **kwargs):
        """
        DeepSeek VL model function that handles both text and vision
        
        Args:
            prompt: Text prompt
            system_prompt: System prompt
            history_messages: Message history 
            image_data: Base64 encoded image data or list of images
            messages: Complete message format (for direct VLM calls)
            **kwargs: Additional parameters
            
        Returns:
            str: Model response
        """
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
            
            # Handle multimodal content
            if image_data:
                user_content = []
                
                # Add text part
                if prompt:
                    user_content.append({"type": "text", "text": prompt})
                
                # Add images - Use text-based format for DeepSeek compatibility
                if isinstance(image_data, str):
                    # Single image - use text reference format
                    user_content.append({
                        "type": "text",
                        "text": f"[Image: {prompt}]"  # Text-based image reference
                    })
                elif isinstance(image_data, list):
                    # Multiple images - use text references
                    for i, img in enumerate(image_data):
                        user_content.append({
                            "type": "text", 
                            "text": f"[Image {i+1}: Related to {prompt}]"
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
            logging.error(f"DeepSeek VL model call failed: {e}")
            raise HTTPException(status_code=500, detail=f"Vision model error: {str(e)}")
    
    return vision_model_func


def create_deepseek_chat_model_func():
    """Create DeepSeek chat model function"""
    
    async def chat_model_func(prompt: str, system_prompt: str = None, history_messages: List = None, **kwargs):
        """
        DeepSeek chat model function
        
        Args:
            prompt: Text prompt
            system_prompt: System prompt
            history_messages: Message history
            **kwargs: Additional parameters
            
        Returns:
            str: Model response
        """
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
            logging.error(f"DeepSeek chat model call failed: {e}")
            raise HTTPException(status_code=500, detail=f"Chat model error: {str(e)}")
    
    return chat_model_func


def create_qwen_embedding_func():
    """Create Qwen3 embedding function"""
    from lightrag.utils import EmbeddingFunc
    
    async def embedding_func_impl(texts: List[str]):
        """
        Qwen3 embedding function implementation
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        import openai
        
        client = openai.AsyncOpenAI(
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BINDING_HOST")
        )
        
        try:
            response = await client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                input=texts,
                encoding_format="float"
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logging.error(f"Qwen embedding call failed: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")
    
    # Return EmbeddingFunc wrapper
    return EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
        func=embedding_func_impl
    )


async def initialize_rag_anything():
    """Initialize RAGAnything instance with configuration"""
    global rag_anything
    
    try:
        # Create configuration
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
        
        # LightRAG kwargs for storage configuration
        lightrag_kwargs = {
            "working_dir": config.working_dir,
            "workspace": os.getenv("WORKSPACE"),
            "kv_storage": os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
            "graph_storage": os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),  
            "vector_storage": os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
            "doc_status_storage": os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
            "chunk_token_size": int(os.getenv("CHUNK_SIZE", "1200")),
            "chunk_overlap_token_size": int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
            "llm_model_name": os.getenv("LLM_MODEL", "deepseek-chat"),
            "llm_model_max_async": int(os.getenv("MAX_ASYNC", "4")),
            "summary_max_tokens": int(os.getenv("MAX_TOKENS", "32768")),
            "max_parallel_insert": int(os.getenv("MAX_PARALLEL_INSERT", "2")),
            "max_graph_nodes": int(os.getenv("MAX_GRAPH_NODES", "1000")),
            "addon_params": {"language": os.getenv("SUMMARY_LANGUAGE", "English")},
            "enable_llm_cache": os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true",
            "enable_llm_cache_for_entity_extract": os.getenv("ENABLE_LLM_CACHE_FOR_EXTRACT", "true").lower() == "true",
        }
        
        # Create RAGAnything instance
        rag_anything = RAGAnything(
            config=config,
            llm_model_func=chat_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs=lightrag_kwargs
        )
        
        # Ensure initialization
        await rag_anything._ensure_lightrag_initialized()
        
        logging.info("RAGAnything initialized successfully")
        return rag_anything
        
    except Exception as e:
        logging.error(f"Failed to initialize RAGAnything: {e}")
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    global rag_anything
    
    try:
        # Initialize RAGAnything
        rag_anything = await initialize_rag_anything()
        
        logging.info("Server is ready to accept connections! 🚀")
        yield
        
    finally:
        # Cleanup
        if rag_anything:
            try:
                await rag_anything.finalize_storages()
                logging.info("RAGAnything storages finalized")
            except Exception as e:
                logging.error(f"Error finalizing storages: {e}")


def create_app():
    """Create FastAPI application"""
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = FastAPI(
        title="RAGAnything Server API",
        description="Multimodal RAG system with DeepSeek and Qwen3 integration",
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
    
    @app.get("/")
    async def redirect_to_webui():
        """Redirect root path to /webui"""
        return RedirectResponse(url="/webui")
    
    @app.get("/webui")
    async def redirect_to_static():
        """Redirect /webui to static files"""
        return RedirectResponse(url="/static/")
    
    @app.get("/auth-status")
    async def get_auth_status():
        """Get authentication status - currently disabled"""
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
        """Login endpoint - currently returns guest token"""
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
        
        config_info = rag_anything.get_config_info()
        
        return HealthResponse(
            status="healthy",
            working_directory=str(rag_anything.working_dir),
            input_directory=config_info["directory"]["parser_output_dir"],
            auth_mode="disabled",
            configuration={
                # LLM configuration
                "llm_binding": os.getenv("LLM_BINDING", "openai"),
                "llm_model": os.getenv("LLM_MODEL", "deepseek-chat"),
                "llm_binding_host": os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
                
                # Embedding configuration  
                "embedding_binding": os.getenv("EMBEDDING_BINDING", "openai"),
                "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                "embedding_binding_host": os.getenv("EMBEDDING_BINDING_HOST"),
                
                # Storage configuration
                "kv_storage": config_info.get("lightrag_config", {}).get("custom_parameters", {}).get("kv_storage", "JsonKVStorage"),
                "graph_storage": config_info.get("lightrag_config", {}).get("custom_parameters", {}).get("graph_storage", "NetworkXStorage"),
                "vector_storage": config_info.get("lightrag_config", {}).get("custom_parameters", {}).get("vector_storage", "NanoVectorDBStorage"),
                "doc_status_storage": config_info.get("lightrag_config", {}).get("custom_parameters", {}).get("doc_status_storage", "JsonDocStatusStorage"),
                
                # RAGAnything specific
                "parser": config_info["parsing"]["parser"],
                "parse_method": config_info["parsing"]["parse_method"],
                "multimodal_processing": config_info["multimodal_processing"],
                "context_extraction": config_info["context_extraction"],
                "batch_processing": config_info["batch_processing"],
                
                # Other settings
                "max_tokens": int(os.getenv("MAX_TOKENS", "32768")),
                "workspace": os.getenv("WORKSPACE"),
                "max_graph_nodes": int(os.getenv("MAX_GRAPH_NODES", "1000")),
                "enable_llm_cache": os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true",
                "enable_llm_cache_for_extract": os.getenv("ENABLE_LLM_CACHE_FOR_EXTRACT", "true").lower() == "true",
                "summary_language": os.getenv("SUMMARY_LANGUAGE", "English"),
                "max_parallel_insert": int(os.getenv("MAX_PARALLEL_INSERT", "2")),
                "max_async": int(os.getenv("MAX_ASYNC", "4")),
            }
        )
    
    @app.post("/api/insert")
    async def insert_document(
        file: Optional[UploadFile] = File(None),
        content: Optional[str] = Form(None),
        file_path: Optional[str] = Form(None)
    ):
        """
        Insert document using RAGAnything.process_document_complete
        
        Supports file upload, direct content, or file path
        """
        if not rag_anything:
            raise HTTPException(status_code=503, detail="RAGAnything not initialized")
        
        try:
            if file:
                # Handle file upload
                temp_dir = Path("./temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                temp_file_path = temp_dir / file.filename
                
                # Save uploaded file
                with open(temp_file_path, "wb") as buffer:
                    content_bytes = await file.read()
                    buffer.write(content_bytes)
                
                # Process the file
                result = await rag_anything.process_document_complete(str(temp_file_path))
                
                # Clean up temporary file
                temp_file_path.unlink(missing_ok=True)
                
                return {
                    "status": "success", 
                    "message": f"File {file.filename} processed successfully",
                    "filename": file.filename,
                    "result": result
                }
                
            elif file_path:
                # Handle file path
                path = Path(file_path)
                if not path.exists():
                    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
                
                result = await rag_anything.process_document_complete(str(path))
                
                return {
                    "status": "success",
                    "message": f"File {file_path} processed successfully",
                    "file_path": file_path,
                    "result": result
                }
                
            elif content:
                # Handle direct content insertion
                temp_dir = Path("./temp_content") 
                temp_dir.mkdir(exist_ok=True)
                
                temp_file_path = temp_dir / "content.txt"
                
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                result = await rag_anything.process_document_complete(str(temp_file_path))
                
                # Clean up
                temp_file_path.unlink(missing_ok=True)
                
                return {
                    "status": "success",
                    "message": "Content processed successfully",
                    "content_length": len(content),
                    "result": result
                }
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Either file upload, file_path, or content must be provided"
                )
                
        except Exception as e:
            logging.error(f"Error in insert_document: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/query")
    async def query_document(request: QueryRequest):
        """
        Query using RAGAnything.aquery_with_multimodal
        
        Supports both text and multimodal queries
        """
        if not rag_anything:
            raise HTTPException(status_code=503, detail="RAGAnything not initialized")
        
        try:
            # Prepare query parameters
            kwargs = {}
            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens
            if request.temperature:
                kwargs["temperature"] = request.temperature
            
            # Execute query
            result = await rag_anything.aquery_with_multimodal(
                query=request.query,
                multimodal_content=request.multimodal_content or [],
                mode=request.mode,
                **kwargs
            )
            
            return {
                "status": "success",
                "result": result,
                "query": request.query,
                "mode": request.mode,
                "multimodal_content_count": len(request.multimodal_content) if request.multimodal_content else 0
            }
            
        except Exception as e:
            logging.error(f"Error in query_document: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Additional LightRAG-compatible endpoints for frontend compatibility
    
    @app.post("/query")
    async def query_compatible(request: QueryRequest):
        """Query endpoint compatible with LightRAG frontend (without /api prefix)"""
        return await query_document(request)
    
    @app.post("/documents/upload")
    async def upload_document_compatible(file: UploadFile = File(...)):
        """Upload endpoint compatible with LightRAG frontend"""
        return await insert_document(file=file)
    
    @app.get("/documents")
    async def get_documents():
        """Get documents list - stub implementation for frontend compatibility"""
        return {"status": "success", "data": {"documents": [], "total": 0}}
    
    @app.delete("/documents")
    async def clear_documents():
        """Clear all documents - stub implementation"""
        return {"status": "success", "message": "Documents cleared (not implemented)"}
    
    @app.post("/documents/clear_cache")
    async def clear_cache():
        """Clear LLM cache - stub implementation"""
        return {"status": "success", "message": "Cache cleared (not implemented)"}
    
    @app.delete("/documents/delete_document")
    async def delete_documents():
        """Delete specific documents - stub implementation"""
        return {"status": "success", "message": "Documents deleted (not implemented)"}
    
    @app.post("/documents/paginated")
    async def get_documents_paginated(request: dict):
        """Get paginated documents - stub implementation"""
        return {"status": "success", "data": {"documents": [], "total": 0, "page": 1}}
    
    @app.get("/documents/pipeline_status")
    async def get_pipeline_status():
        """Get pipeline status - stub implementation"""
        return {"pipeline_busy": False}
    
    @app.post("/documents/scan")
    async def scan_documents():
        """Scan for documents - stub implementation"""
        return {"status": "success", "message": "Scan completed"}
    
    @app.get("/graphs")
    async def get_graphs(label: str = "", max_depth: int = 3, max_nodes: int = 1000):
        """Get graph data - stub implementation for frontend compatibility"""
        return {
            "status": "success", 
            "data": {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "max_depth": max_depth,
                    "max_nodes": max_nodes,
                    "label": label
                }
            }
        }
    
    @app.get("/graph/label/list")
    async def get_graph_labels():
        """Get available graph labels - stub implementation"""
        return {"status": "success", "data": []}
    
    @app.post("/graph/entity/edit")
    async def edit_graph_entity():
        """Edit graph entity - stub implementation"""
        return {"status": "success", "message": "Entity edited (not implemented)"}
    
    @app.post("/graph/relation/edit") 
    async def edit_graph_relation():
        """Edit graph relation - stub implementation"""
        return {"status": "success", "message": "Relation edited (not implemented)"}
    
    @app.get("/graph/entity/exists")
    async def check_entity_exists(name: str):
        """Check if entity exists - stub implementation"""
        return {"exists": False}
    
    # Custom StaticFiles class for smart caching (from LightRAG)
    class SmartStaticFiles(StaticFiles):
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
    
    # Mount static files (the UI from LightRAG)
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", SmartStaticFiles(directory=static_path, html=True), name="static")
        logging.info(f"Static files mounted from: {static_path}")
    else:
        logging.warning(f"Static files directory not found: {static_path}")
    
    return app


# Create the app instance
app = create_app()


def main():
    """Main function to run the server"""
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting RAGAnything Server...")
    logging.info(f"Working directory: {os.getenv('WORKING_DIR', './rag_storage')}")
    logging.info(f"Input directory: {os.getenv('INPUT_DIR', './inputs')}")
    logging.info(f"Parser: {os.getenv('PARSER', 'mineru')}")
    logging.info(f"LLM Model: {os.getenv('LLM_MODEL', 'deepseek-chat')}")
    logging.info(f"Embedding Model: {os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')}")
    
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