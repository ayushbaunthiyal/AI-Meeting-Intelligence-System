"""
FastAPI Application Entry Point

This is the main entry point for the backend API server.
It creates the FastAPI application, configures middleware, and includes routes.

=============================================================================
APPLICATION STARTUP FLOW:
=============================================================================

    1. Python loads this module
    2. FastAPI app is created with metadata
    3. CORS middleware is configured
    4. API routes are registered
    5. Uvicorn starts the server (when run via `uvicorn src.main:app`)

=============================================================================
KEY COMPONENTS:
=============================================================================

FastAPI:
    - Modern, fast Python web framework
    - Automatic OpenAPI documentation
    - Async support out of the box
    - Pydantic integration for validation

CORS Middleware:
    - Allows frontend to make requests to backend
    - Required for Streamlit UI running on different port

Router:
    - API routes defined in src/api/routes.py
    - All routes prefixed with /api/v1

=============================================================================
RUNNING THE SERVER:
=============================================================================

Development:
    uvicorn src.main:app --reload --port 8001

Production (Docker):
    uvicorn src.main:app --host 0.0.0.0 --port 8001

=============================================================================
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import our API routes
from .api.routes import router
from .config import get_settings

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Set up logging to see what's happening in the application.
# In production, you'd want structured logging (JSON format).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout for Docker
    ],
)

logger = logging.getLogger(__name__)


# =============================================================================
# CREATE FASTAPI APPLICATION
# =============================================================================
# The FastAPI instance is the core of our API server.
# Metadata here appears in the auto-generated API documentation.

app = FastAPI(
    title="AI Meeting Intelligence API",
    description=(
        "API for analyzing meeting transcripts using AI. "
        "Features include transcript parsing, summarization, "
        "decision extraction, action item identification, and Q&A."
    ),
    version="0.1.0",
    # Prefix for API docs URLs
    docs_url="/docs",           # Swagger UI at /docs
    redoc_url="/redoc",         # ReDoc at /redoc
    openapi_url="/openapi.json",
)


# =============================================================================
# CORS (Cross-Origin Resource Sharing) MIDDLEWARE
# =============================================================================
# CORS is required because the Streamlit UI runs on a different port (8504)
# than the API (8001). Without CORS, browsers block cross-origin requests.
#
# SECURITY NOTE: 
# - In production, restrict origins to your actual frontend domains
# - "*" allows any origin (fine for development, risky in production)

app.add_middleware(
    CORSMiddleware,
    # Which origins can make requests
    allow_origins=["*"],  # TODO: Restrict in production!
    # Allow cookies and authentication headers
    allow_credentials=True,
    # Which HTTP methods are allowed
    allow_methods=["*"],  # GET, POST, PUT, DELETE, etc.
    # Which request headers are allowed
    allow_headers=["*"],
)


# =============================================================================
# REGISTER API ROUTES
# =============================================================================
# All our API endpoints are defined in src/api/routes.py
# They're mounted with a version prefix for future API versioning

app.include_router(
    router,
    prefix="/api/v1",  # All routes will be /api/v1/...
    tags=["api"],
)


# =============================================================================
# LIFECYCLE EVENTS
# =============================================================================
# These hooks run at application startup and shutdown.
# Useful for initializing/cleaning up resources.

@app.on_event("startup")
async def startup_event():
    """
    Called when the application starts.
    
    Use this to:
    - Initialize database connections
    - Load ML models (Whisper, embeddings)
    - Validate configuration
    """
    logger.info("🚀 Starting Meeting Intelligence API...")
    
    # Load settings to validate configuration early
    settings = get_settings()
    
    # Log configuration (hide sensitive values)
    logger.info(f"Configuration loaded:")
    logger.info(f"  - OpenAI Model: {settings.openai_model}")
    logger.info(f"  - Embedding Model: {settings.openai_embedding_model}")
    logger.info(f"  - Whisper Model: {settings.whisper_model}")
    logger.info(f"  - Log Level: {settings.log_level}")
    
    # Verify OpenAI API key is set
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        logger.warning("⚠️ OPENAI_API_KEY is not set! LLM features will not work.")
    else:
        logger.info("  - OpenAI API Key: ****" + settings.openai_api_key[-4:])
    
    logger.info("✅ Meeting Intelligence API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Called when the application shuts down.
    
    Use this to:
    - Close database connections
    - Save state
    - Clean up resources
    """
    logger.info("👋 Shutting down Meeting Intelligence API...")


# =============================================================================
# HEALTH CHECK (Root Endpoint)
# =============================================================================
# Simple root endpoint for basic connectivity testing

@app.get("/")
async def root():
    """Root endpoint - basic API info."""
    return {
        "name": "AI Meeting Intelligence API",
        "version": "0.1.0",
        "docs": "/docs",
    }
