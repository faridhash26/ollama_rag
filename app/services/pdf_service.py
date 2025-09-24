import pdfplumber
import io
from langchain_core.documents import Document
import re
from .embedding_service import normalize_text_for_persian
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
    """
    خروجی: لیستی از Document که هر Document محتوای یک صفحه است.
    metadata شامل: source (filename) و page_num
    """
    docs = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            raw_text = extract_clean_rtl_text(page)
            if not raw_text.strip():
                raw_text = extract_text_fallback(page)
            text = normalize_text_for_persian(raw_text)
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page_num": page_num
                }
            ))

            # استخراج جداول (اختیاری) — اگر می‌خواهی نگه داری کن:
            try:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                }) or []
            except Exception:
                tables = []
            for j, table in enumerate(tables):
                # تبدیل جدول به متن ساده
                max_cols = max((len(r) for r in table), default=0)
                rows = []
                for row in table:
                    padded = (row + [""] * max_cols)[:max_cols]
                    clean_row = [str(c).strip() if c is not None else "" for c in padded]
                    rows.append(" | ".join(clean_row))
                table_text = "\n".join(rows)
                docs.append(Document(
                    page_content=table_text,
                    metadata={
                        "source": filename,
                        "page_num": page_num,
                        "type": "table",
                        "table_index": j+1
                    }
                ))

    return docs

def extract_clean_rtl_text(page):
    """
    تلاش اولیه با extract_words+char_dir برای نگه داشتن ترتیب RTL.
    اگر خروجی خالی بود، fallback به extract_text.
    """
    try:
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
            # تنظیمات جهت‌ها ممکن است در نسخه‌های pdfplumber متفاوت باشد
            # اگر محیطت با char_dir کار نکرد، این قسمت را کم کن یا حذفش کن
            line_dir="ttb",
            char_dir="rtl",
        )
    except Exception:
        words = None

    if not words:
        # fallback
        text = extract_text_fallback(page)
        return text or ""

    # گروه‌بندی بر اساس top (سطرها)
    lines = {}
    for w in words:
        top = round(w.get('top', 0), 1)
        lines.setdefault(top, []).append(w)

    full_text = ""
    for top in sorted(lines.keys()):
        line_words = lines[top]
        # برای RTL: مرتب‌سازی نزولی x0
        line_sorted = sorted(line_words, key=lambda w: w.get('x0', 0), reverse=True)
        line_text = " ".join(w.get('text', '') for w in line_sorted)
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


def extract_text_fallback(page):
    """فانکشن fallback: اگر extract_words خالی داد از extract_text استفاده کن."""
    try:
        t = page.extract_text() or ""
        return t
    except Exception:
        return ""
    