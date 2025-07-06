import os
from fastapi import UploadFile
from app.services.pdf_service import extract_docs_from_pdf
from app.services.rag_service import create_vectorstore

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def handle_upload(file: UploadFile):
    content = await file.read()
    docs = extract_docs_from_pdf(content, file.filename)
    create_vectorstore(docs)
    return {"message": "successful"}
