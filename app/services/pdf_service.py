import pdfplumber
import io
from langchain_core.documents import Document
import re

def is_rtl_text(text: str) -> bool:
    rtl_chars = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u0590-\u05FF\uFE70-\uFEFF]')
    return bool(rtl_chars.search(text))

def reverse_rtl_text(text: str) -> str:
    words = text.split(' ')
    reversed_words = []
    for word in words:
        
        reversed_words.append(word[::-1])
        
    return ' '.join(reversed_words)

def extract_docs_from_pdf(content: bytes, filename: str):
    docs = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1

            # 1️⃣ استخراج متن اصلی (با پشتیبانی RTL)
            text_content = extract_clean_rtl_text(page)
            print('text_content' ,text_content)
            if text_content.strip():
                docs.append(
                    Document(
                        page_content=text_content,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "type": "text",
                            "language": "fa",
                            "direction": "rtl"
                        }
                    )
                )

            # 2️⃣ استخراج جداول
            tables = extract_tables_from_page(page)
            for j, table in enumerate(tables):
                # تبدیل جدول به متن خوانا (مثلاً CSV-like یا Markdown)
                table_text = table_to_markdown(table)
                print('table_text' ,table_text)
                if table_text.strip():
                    docs.append(
                        Document(
                            page_content=table_text,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "type": "table",
                                "table_index": j + 1,
                                "language": "fa"
                            }
                        )
                    )

            # 3️⃣ (اختیاری) استخراج سایر عناصر — مثل تصاویر، خطوط، متادیتا
            # می‌توانید در آینده اضافه کنید — مثلاً با PyMuPDF برای عکس‌ها

    return docs

def extract_clean_rtl_text(page):
    """
    استخراج متن با حفظ ترتیب RTL — بدون نیاز به reverse!
    """
    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
        line_dir="ttb",   # خطوط از بالا به پایین
        char_dir="rtl",   # کاراکترها از راست به چپ 👈 کلید اصلی
    )

    if not words:
        return ""

    # گروه‌بندی کلمات بر اساس خط
    lines = {}
    for word in words:
        top = round(word['top'], 1)
        lines.setdefault(top, []).append(word)

    full_text = ""
    for top in sorted(lines.keys()):  # از بالا به پایین
        # مرتب‌سازی کلمات در هر خط از راست به چپ
        line_words = sorted(lines[top], key=lambda w: w['x0'], reverse=True)
        line_text = " ".join(w['text'] for w in line_words)
        full_text += line_text + "\n"

    return full_text.strip()


def extract_tables_from_page(page):
    """
    استخراج تمام جداول از یک صفحه
    """
    try:
        tables = page.extract_tables(
            table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 5,
                "join_tolerance": 5,
            }
        )
        return tables or []
    except Exception:
        return []
    

def table_to_markdown(table):
    """
    تبدیل جدول (لیست لیست) به متن Markdown-like برای ذخیره در Document
    """
    if not table or not any(row for row in table):
        return ""

    # پیدا کردن بیشترین تعداد ستون
    max_cols = max(len(row) for row in table)

    lines = []
    for row in table:
        # پر کردن ردیف‌های کوتاه با خالی
        padded_row = (row + [""] * max_cols)[:max_cols]
        # تبدیل هر سلول به str و حذف None
        clean_row = [str(cell).strip() if cell is not None else "" for cell in padded_row]
        # ادغام با جداکننده |
        lines.append(" | ".join(clean_row))

    return "\n".join(lines)