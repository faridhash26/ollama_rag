from fastapi import APIRouter, UploadFile, File
from app.models.schemas import ChatRequest
from app.controllers.upload_controller import handle_upload
from app.controllers.chat_controller import chat_handler, chat_stream_handler

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return await handle_upload(file)

@router.post("/chat")
async def chat(req: ChatRequest):
    return chat_handler(req.question)

@router.post("/chatstream")
async def chatstream(req: ChatRequest):
    return await chat_stream_handler(req.question)

