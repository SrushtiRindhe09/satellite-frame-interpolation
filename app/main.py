import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.services.raft_inference import load_raft_model
from app.services.rife_inference import load_rife_model
from app.routes.inference import router as inference_router
from app.routes.evaluation import router as evaluation_router

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load AI models once at server startup.
    Models are stored in app.state so they are reused across all requests.
    """
    print("\n" + "=" * 60)
    print("  ISRO Satellite Frame Interpolation — Starting Server")
    print("=" * 60 + "\n")

    # Load models once
    app.state.raft_model = load_raft_model()
    app.state.rife_model = load_rife_model()

    print("\n" + "=" * 60)
    print("  [OK] All models loaded. Server is ready.")
    print("=" * 60 + "\n")

    yield

    # Cleanup on shutdown
    print("\n[STOP] Server shutting down. Releasing model resources.")


app = FastAPI(
    title="ISRO Satellite Frame Interpolation API",
    description=(
        "AI-powered backend for generating optical flow visualization "
        "and intermediate satellite frames using RAFT and RIFE models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# CORS — allow frontend to connect from any origin during development
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Register API routes
# ------------------------------------------------------------------
app.include_router(inference_router)
app.include_router(evaluation_router)

# Mount static directory for UI assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def root():
    """
    Serve the Web UI at the root URL.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "raft_loaded": hasattr(app.state, "raft_model"),
        "rife_loaded": hasattr(app.state, "rife_model"),
    }
