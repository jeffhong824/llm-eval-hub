"""FastAPI main application."""

# Import pydantic patch FIRST, before any other imports
# This fixes Python 3.12 compatibility issues with langsmith/pydantic v1
import ai.core.pydantic_patch  # noqa: F401

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import structlog

from api.routes import evaluation, testset, health
from api.middleware import LoggingMiddleware, MetricsMiddleware
from configs.settings import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LLM Evaluation Hub", version=settings.app_version)
    
    # Create necessary directories
    import os
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.results_dir, exist_ok=True)
    os.makedirs(settings.artifacts_dir, exist_ok=True)
    
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Application shutdown initiated")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Universal LLM Evaluation Platform with RAGAS and LangSmith",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Mount static files first (order matters in FastAPI)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["evaluation"])
app.include_router(testset.router, prefix="/api/v1/testset", tags=["testset"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.get("/")
async def root():
    """Root endpoint - serve frontend directly."""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@app.get("/info")
async def info():
    """Application information endpoint."""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "debug": settings.debug,
        "available_metrics": settings.available_metrics,
        "agent_metrics": settings.agent_metrics
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
