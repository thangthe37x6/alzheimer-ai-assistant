from fastapi import APIRouter, HTTPException
from app.services.vit_service import vit_model
from app.config import VIT_MODEL_PATH, CHROMA_PATH, IMAGE_SIZE, collection
from app.services.state import get_active_sessions_count

router = APIRouter()

@router.get("/health")
async def health():
    return {
        "status":          "ok",
        "vit_loaded":      vit_model is not None,
        "vit_path":        VIT_MODEL_PATH,
        "chroma":          CHROMA_PATH,
        "image_size":      IMAGE_SIZE,
        "active_sessions": get_active_sessions_count(),
    }

@router.get("/ready")
async def ready():
    checks = {
        "vit_loaded": vit_model is not None,
        "chroma": False,
    }
    try:
        collection.count()
        checks["chroma"] = True
    except Exception:
        checks["chroma"] = False

    if not all(checks.values()):
        raise HTTPException(503, {"status": "not_ready", "checks": checks})

    return {"status": "ready", "checks": checks}
