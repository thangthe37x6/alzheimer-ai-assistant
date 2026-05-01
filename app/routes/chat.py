from fastapi import APIRouter, HTTPException
from app.schemas.models import ChatRequest, ChatResponse, ResetRequest
from app.services.rag_service import query_rag
from app.services.state import clear_session

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Câu hỏi không được để trống.")
    try:
        answer, pages = query_rag(req.question, req.session_id)
        return ChatResponse(answer=answer, pages=pages)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/reset")
async def reset(req: ResetRequest):
    clear_session(req.session_id)
    return {"message": "Đã xóa lịch sử hội thoại."}
