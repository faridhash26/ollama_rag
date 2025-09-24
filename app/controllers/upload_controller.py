import os
from fastapi import UploadFile
from app.services.pdf_service import extract_docs_from_pdf
from app.services.rag_service import create_vectorstore
import pdfplumber
import io
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def handle_upload(file: UploadFile):
    content = await file.read()
    docs = extract_docs_from_pdf(content, file.filename)
    create_vectorstore(docs)
    return {"message": "successful"}



# async def handle_upload(file: UploadFile):
#     # خواندن محتوای فایل
#     content = await file.read()

#     # تبدیل به فایل در حافظه (BytesIO) برای استفاده با pdfplumber
#     with pdfplumber.open(io.BytesIO(content)) as pdf:
#         full_text = ""
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text()
#             if text:
#                 full_text += f"\n--- Page {i + 1} ---\n{text}"
#             else:
#                 full_text += f"\n--- Page {i + 1} ---\n[No text found]"

#     # نمایش خروجی در کنسول (برای تست)
#     print("\n" + "="*50)
#     print("Extracted Text from PDF:")
#     print("="*50)
#     print(full_text)
#     print("="*50 + "\n")

#     # فعلا فقط یک پیام موفقیت برمی‌گردانیم
#     return {"message": "PDF processed successfully", "filename": file.filename}
