"""
FastAPI backend for Diffusion Models Image Editing application.
Provides REST API endpoints for inpainting, image editing, and style transfer.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import inpainting, style_transfer, restoration


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

# CORS middleware - allow all origins for cloud deployment
# For production, you may want to restrict this to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for cloud deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inpainting.router, prefix="/api/inpainting", tags=["Inpainting"])
app.include_router(style_transfer.router, prefix="/api/style", tags=["Style Transfer"])
app.include_router(restoration.router, prefix="/api/restoration", tags=["Restoration"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Diffusion Image Editor API is running",
        "endpoints": {
            "inpainting": "/api/inpainting",
            "style_transfer": "/api/style",
            "restoration": "/api/restoration",
        }
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check with GPU info."""
    from app.services.diffusion import get_diffusion_service

    service = get_diffusion_service()
    gpu_info = service.get_gpu_info()

    return {
        "status": "healthy",
        "gpu": gpu_info,
        "loaded_pipelines": list(service._pipelines.keys()),
    }
