from fastapi import APIRouter, HTTPException
from models import ChatRequest, ChatResponse
from services.chat_service import process_chat

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for frontend integration"""
    try:
        result = await process_chat(request.message, request.history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))