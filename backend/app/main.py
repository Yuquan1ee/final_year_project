"""
FastAPI backend for Diffusion Models Image Editing application.
Provides REST API endpoints for inpainting, image editing, and style transfer.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import inpainting, editing, style_transfer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("Starting up Diffusion Image Editor API...")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Diffusion Image Editor API",
    description="API for intelligent image editing using diffusion models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inpainting.router, prefix="/api/inpainting", tags=["Inpainting"])
app.include_router(editing.router, prefix="/api/editing", tags=["Image Editing"])
app.include_router(style_transfer.router, prefix="/api/style", tags=["Style Transfer"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Diffusion Image Editor API is running",
        "endpoints": {
            "inpainting": "/api/inpainting",
            "editing": "/api/editing",
            "style_transfer": "/api/style",
        }
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "services": {
            "api": True,
            "huggingface": True,  # TODO: Add actual HF API check
        }
    }
