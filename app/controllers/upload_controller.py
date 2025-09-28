from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageOps
import io
import numpy as np
from paddleocr import PaddleOCR
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import os
from app.controllers.chat_controller import ChatRAGService

from app.services.vector_store import ChromaVectorService

# تردپول برای عملیات سنگین
executor = ThreadPoolExecutor(max_workers=4)
vector_service = ChromaVectorService(executor=executor)
rag_service = ChatRAGService()
# ————————————————————————————————
# 🖼️ تبدیل صفحه PDF به تصویر
# ————————————————————————————————

def pdf_page_to_pil_safe(doc: fitz.Document, page_idx: int, zoom: float = 2.5, max_dim: int = 3000) -> Image.Image:
    page = doc.load_page(page_idx)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes(output="png")
    img = Image.open(io.BytesIO(png_bytes))
    img = img.convert("RGB")
    img.load()
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img

# ————————————————————————————————
# 🧹 پیش‌پردازش تصویر
# ————————————————————————————————

def preprocess_for_ocr(pil_img: Image.Image, max_dim: int = 3000) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) > max_dim:
        pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    
    # تبدیل به خاکستری
    g = pil_img.convert("L")
    
    # افزایش کنتراست
    try:
        enhancer = ImageEnhance.Contrast(g)
        g = enhancer.enhance(2.0)
    except Exception:
        pass
        
    return g.convert("RGB")

# ————————————————————————————————
# 💾 ذخیره تصاویر دیباگ (اصلی + پیش‌پردازش‌شده)
# ————————————————————————————————

def save_sample_pages(images: List[Image.Image], out_prefix: str = "page_debug"):
    os.makedirs("ocr_debug", exist_ok=True)
    for i, img in enumerate(images[:20]):
        try:
            # تصویر اصلی
            img.save(f"ocr_debug/{out_prefix}_{i}.png")
            # تصویر پیش‌پردازش‌شده
            pre = preprocess_for_ocr(img)
            pre.save(f"ocr_debug/{out_prefix}_{i}_preprocessed.png")
        except Exception as e:
            print(f"❌ failed to save debug images for page {i}: {e}")

# ————————————————————————————————
# 🚀 تابع اصلی پردازش آپلود
# ————————————————————————————————

async def handle_upload(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="فایل باید PDF باشد.")
    
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="فایل خالی است.")

    # ایجاد موتور OCR (در اولین فراخوانی)
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang="fa",
 
    )

    loop = asyncio.get_running_loop()
    pages_out = []
    skipped = []
    images_for_debug = []

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطا در باز کردن PDF: {e}")

    total = len(doc)
    for i in range(total):
        try:
            # تبدیل صفحه به تصویر
            img = await loop.run_in_executor(executor, pdf_page_to_pil_safe, doc, i)
            images_for_debug.append(img)
            
            # اول چک کن که آیا PDF متنی هست (نه اسکن)
            page_text = doc[i].get_text()
            if page_text.strip():
                # PDF متنی هست - مستقیماً استفاده کن
                lines = []
                for line_text in page_text.split('\n'):
                    line_text = line_text.strip()
                    if line_text:
                        lines.append({
                            "text": line_text,
                            "confidence": 1.0,
                            "box": [],
                            "type": "text"
                        })
                full_text = "\n".join([ln["text"] for ln in lines])
                pages_out.append({
                    "page_index": i,
                    "lines": lines,
                    "full_text": full_text,
                    "total_lines": len(lines)
                })
                continue

            # اگر PDF اسکن شده بود، از OCR استفاده کن
            pre = await loop.run_in_executor(executor, preprocess_for_ocr, img)
            img_array = np.array(pre)
            result = ocr_engine.ocr(img_array)  # بدون cls=True
            
            lines = []
            # بررسی ایمن خروجی OCR
            if result is not None and isinstance(result, list) and len(result) > 0:
                page_results = result[0]
                if page_results is not None and isinstance(page_results, list):
                    for line in page_results:
                        if not isinstance(line, (list, tuple)) or len(line) < 2:
                            continue
                            
                        box = line[0]
                        text_conf = line[1]
                        
                        if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                            continue
                            
                        text = str(text_conf[0]).strip()
                        confidence = float(text_conf[1])
                        
                        if text:  # فقط متن‌های غیرخالی
                            lines.append({
                                "text": text,
                                "confidence": confidence,
                                "box": box,
                                "type": "text"
                            })
            
            full_text = "\n".join([ln["text"] for ln in lines])
            pages_out.append({
                "page_index": i,
                "lines": lines,
                "full_text": full_text,
                "total_lines": len(lines)
            })
            
        except Exception as e:
            skipped.append({"page_index": i, "reason": f"ocr_error: {repr(e)}"})
            continue

    doc.close()

    # ذخیره تصاویر دیباگ
    if images_for_debug:
        try:
            save_sample_pages(images_for_debug, out_prefix="page_debug")
        except Exception as e:
            print("⚠️ failed saving debug images:", e)
            
    combined_text = "\n\n".join(page["full_text"] for page in pages_out if page["full_text"].strip())
    enhanced_text = await rag_service.enhance_document_text(combined_text)

    # بررسی اینکه آیا حداقل یک صفحه پردازش شده
    if len(pages_out) == 0 and len(skipped) > 0:
        raise HTTPException(status_code=500, detail={
            "message": "هیچ صفحه‌ای پردازش نشد",
            "errors": skipped
        })
    
    try:
        index_ref = await vector_service.index_single_document(
            filename=file.filename,
            content=enhanced_text
        )
    except Exception as e:
        print("⚠️ Vector indexing failed:", e)
        index_ref = ""
    return JSONResponse({
        "filename": file.filename,
        "total_pages": total,
        "pages": pages_out,
        "skipped_pages": skipped,
        "model_used": "PaddleOCR (lang=fa)"
    })