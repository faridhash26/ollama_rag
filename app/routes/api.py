from fastapi import APIRouter, UploadFile, File
from app.models.schemas import ChatRequest
from app.controllers.upload_controller import handle_upload
from app.controllers.chat_controller import ChatRAGService

router = APIRouter()
chat_service = ChatRAGService()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return await handle_upload(file)

@router.post("/chat")
async def chat(req: ChatRequest):
    question = req.question
    return await chat_service.chat(question)

@router.post("/chatstream")
async def chatstream(req: ChatRequest):
    question = req.question
    async for chunk in chat_service.chat_stream(question):
        yield chunk

