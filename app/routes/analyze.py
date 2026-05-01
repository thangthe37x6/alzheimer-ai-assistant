import os

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.schemas.models import AnalyzeResponse
from app.services.vit_service import predict, vit_model
from app.services.gpt_service import analyze_with_gpt4v
from app.services.state import save_vit_session
from app.config import CLASS_META

router = APIRouter()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file:       UploadFile = File(...),
    session_id: str        = Form(default=""),
):
    if vit_model is None:
        raise HTTPException(503, "Model chưa được load. Kiểm tra VIT_MODEL_PATH trong .env")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Chỉ chấp nhận file ảnh (jpg, png...)")
    try:
        img_bytes  = await file.read()
        if len(img_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File quá lớn. Giới hạn hiện tại là {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.")
        vit_result = predict(img_bytes)

        if session_id:
            save_vit_session(session_id, vit_result)
            print(f"[session={session_id}] ViT result saved to Redis/State: {vit_result['top_class']}")

        gpt_text   = analyze_with_gpt4v(img_bytes, vit_result)
        top        = vit_result["top_class"]
        label, color = CLASS_META.get(top, (top, "#64748b"))
        return AnalyzeResponse(
            top_class=top, label=label, color=color,
            scores=vit_result["scores"], gpt_analysis=gpt_text,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
