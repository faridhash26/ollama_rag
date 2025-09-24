import os, re, unicodedata, math, hashlib
import requests
import numpy as np
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from parsivar import Normalizer, Tokenizer
from typing import List, Dict,Any
import json

OLLAMA_URL = os.getenv("Embeding_OLLAMA_URL", "http://localhost:11434/api/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "aligh4699/heydariAI-persian-embeddings:latest")
AR_YEH = "\u064A"; FA_YEH = "\u06CC"
AR_KEH = "\u0643"; FA_KEH = "\u06A9"
KASHIDA = "\u0640"
ZWNJ = "\u200c"

normalizer = Normalizer()
tokenizer = Tokenizer()
_url_re = re.compile(r"https?://\S+")
_email_re = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_phone_re = re.compile(r"\+?\d[\d\-\s]{6,}\d")
_SENT_END_RE = re.compile(r'[\.!\?؟؛…]$')

def sanitize_metadata_value(v: Any) -> Any:
    """
    تبدیل مقدار متادیتا به یکی از انواع مجاز (str, int, float, bool, None).
    - لیست‌هایی که شامل مقادیر ساده هستند -> به رشته تبدیل می‌شوند (","-joined).
    - dict یا پیچیده‌ها -> json.dumps
    - None حفظ می‌شود.
    - سایر موارد به str تبدیل می‌شوند.
    """
    # مجازها را همان‌طور برگردان
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v

    # لیست از مقادیر ساده
    if isinstance(v, list):
        # فیلتر Noneها
        cleaned = [x for x in v if x is not None]
        # اگر هیچ‌چیز نماند -> None
        if not cleaned:
            return None
        # اگر همهٔ اعضا از نوع ساده‌اند -> join کن
        if all(isinstance(x, (str, int, float, bool)) for x in cleaned):
            # تبدیل همه به str و کاما-جین
            return ",".join(str(x) for x in cleaned)
        # در غیر این صورت json.dumps
        try:
            return json.dumps(cleaned, ensure_ascii=False)
        except Exception:
            return str(cleaned)

    # dict -> json
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    # بقیه انواع (مثلاً custom objects) -> str
    try:
        return str(v)
    except Exception:
        return None
def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (metadata or {}).items():
        clean_v = sanitize_metadata_value(v)
        # اگر تمایل ندارید کلیدهای None حذف شوند می‌توانید آن‌ها را نگه دارید
        if clean_v is not None:
            out[k] = clean_v
        else:
            # اگر می‌خوای کلید با مقدار None بمونه، uncomment کن:
            # out[k] = None
            pass
    return out

# استفاده هنگام ساخت داکیومنت‌ها (مثال در pipeline خودت)
def sanitize_documents_for_chroma(docs):
    sanitized = []
    for d in docs:
        md = d.metadata if hasattr(d, "metadata") and d.metadata else {}
        md_clean = sanitize_metadata(md)
        sanitized.append(Document(page_content=d.page_content, metadata=md_clean))
    return sanitized
def merge_pages_into_full_sentences(docs):
    """
    هر داکیومنت (صفحه) رو به لیست خطوط تبدیل می‌کنه، 
    خطوط رو با منطق فارسی ادغام می‌کنه، 
    و یه لیست از داکیومنت‌های جدید با محتوای کامل برمی‌گردونه.
    """
    merged_docs = []
    
    for doc in docs:
        page_text = doc.page_content
        lines = page_text.splitlines()
        
        # ادغام خطوط داخل یک صفحه با منطق فارسی
        merged_lines = merge_lines_by_context([page_text])  # این تابع شماست
        
        # هر خط ادغام‌شده یه داکیومنت جدید می‌شه
        for merged_line in merged_lines:
            if merged_line.strip():  # حذف خطوط خالی
                merged_docs.append(
                    Document(
                        page_content=merged_line.strip(),
                        metadata=doc.metadata.copy()  # حفظ متادیتا مثل source, page_num
                    )
                )
    
    return merged_docs
def ends_with_sentence_terminator(text: str) -> bool:
    if not text:
        return False
    # اگر خط یا پاراگراف با نقطه یا علامت سؤال یا «؛» یا علامت فارسی ختم شده باشد
    return bool(_SENT_END_RE.search(text.strip()))


def merge_pages_into_full_sentences(docs: List[Document]) -> List[Document]:
    """
    ورودی: لیست صفحات (Document) با metadata شامل 'source' و 'page_num' (ترجیحا).
    خروجی: لیستی از Document که هرکدام یک یا چند جملهٔ کامل فارسی دارند.
    """
    # گروه‌بندی بر اساس منبع (مثلا نام فایل)
    groups: Dict[str, List[Document]] = {}
    for d in docs:
        source = d.metadata.get('source') or d.metadata.get('file_name') or 'UNKNOWN_SOURCE'
        groups.setdefault(source, []).append(d)

    merged_docs: List[Document] = []

    for source, pages in groups.items():
        # ترتیب‌دهی صفحات براساس page_num در متادیتا (اگر موجود باشد)
        try:
            pages_sorted = sorted(pages, key=lambda x: int(x.metadata.get('page_num', 0)))
        except Exception:
            pages_sorted = pages

        buffer_text = ""
        buffer_metadata = {}

        for page in pages_sorted:
            page_text = page.page_content or ""
            page_text = normalize_text_for_persian(page_text)

            if buffer_text == "":
                buffer_text = page_text
                buffer_metadata = page.metadata.copy()
                buffer_metadata['_pages'] = [page.metadata.get('page_num')]
                continue

            # اگر با پایان جمله خاتمه نیافته، به صفحهٔ بعدی الصاق کن
            if not ends_with_sentence_terminator(buffer_text):
                buffer_text = buffer_text + " " + page_text
                pages_list = buffer_metadata.get('_pages', [])
                pages_list.append(page.metadata.get('page_num'))
                buffer_metadata['_pages'] = pages_list
            else:
                # buffer_text را به جملات تقسیم کن با tokenizer پارسی‌وار
                try:
                    sentences = tokenizer.tokenize_sentences(buffer_text)
                except Exception:
                    # fallback ساده (regex)
                    sentences = re.split(r'(?<=[\.!\?؟؛…])\s+', buffer_text)

                for s in sentences:
                    s_clean = s.strip()
                    if s_clean:
                        md = buffer_metadata.copy()
                        merged_docs.append(Document(page_content=s_clean, metadata=md.copy()))

                # reset buffer با متن صفحهٔ جاری
                buffer_text = page_text
                buffer_metadata = page.metadata.copy()
                buffer_metadata['_pages'] = [page.metadata.get('page_num')]

        # پس از اتمام صفحات یک منبع، هر آنچه در buffer مانده را هم بخش‌بندی کن
        if buffer_text:
            try:
                sentences = tokenizer.tokenize_sentences(buffer_text)
            except Exception:
                sentences = re.split(r'(?<=[\.!\?؟؛…])\s+', buffer_text)

            for s in sentences:
                s_clean = s.strip()
                if s_clean:
                    md = buffer_metadata.copy()
                    merged_docs.append(Document(page_content=s_clean, metadata=md.copy()))

    return merged_docs


def normalize_text_for_persian(text: str) -> str:
    if not text:
        return ""
    # حذف شکست هیفون صفحه
    text = re.sub(r'-\s*\n\s*', '', text)
    # جایگزینی newline های میانی با فاصله (تا پاراگراف‌ها حفظ شوند)
    text = re.sub(r'\s*\n+\s*', '\n', text)
    # حذف فاصله اضافی
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()
    try:
        text = normalizer.normalize(text)
    except Exception:
        pass
    return text
def merge_lines_by_context(pages_text_list):
    """
    ادغام خطوط متوالی که نشانه‌های ادامه جمله دارند (مثلاً خط اخر صفحه بدون نقطه)
    و خط بعدی با حروف کوچک شروع شده باشد.
    """
    merged = []
    current_line = ""

    for page_text in pages_text_list:
        lines = page_text.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # اگر خط فعلی با حرف کوچک شروع شود → احتمالاً ادامهٔ خط قبلی است
            if current_line and line[0].islower() and len(line) > 1:
                current_line += " " + line
            else:
                # اگر خط قبلی به نقطه/علامت پایان جمله ختم شده بود → ذخیره کن
                if current_line and re.search(r'[.؛!؟]\s*$', current_line):
                    merged.append(current_line)
                    current_line = line
                else:
                    # اگر خط قبلی ادامه داشت، ادغام کن
                    if current_line:
                        current_line += " " + line
                    else:
                        current_line = line
        
        # پس از پایان هر صفحه، اگر خط فعلی خالی نبود، ادامه بده
        if current_line and not re.search(r'[.؛!؟]\s*$', current_line):
            # نگه دار تا در صفحه بعد ادغام شود
            pass
        elif current_line:
            merged.append(current_line)
            current_line = ""

    # اضافه کردن آخرین خط
    if current_line:
        merged.append(current_line)

    return merged
def normalize_persian(text: str) -> str:
    if not text:
        return text
    # فقط NFC normalization + حذف فضاهای اضافی
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- HTTP client with retry ---
def _new_session():
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

_session = _new_session()

# --- simple in-memory cache (swap with Redis if needed) ---
_cache = {}

def _cache_key(model: str, text: str) -> str:
    h = hashlib.sha256(f"{model}||{text}".encode("utf-8")).hexdigest()
    return h

def _unit_norm(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32, copy=False)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def _embed_once(text: str, timeout: float = 20.0):
    payload = {"model": EMBEDDING_MODEL, "prompt": text}
    r = _session.post(OLLAMA_URL, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama embed failed: {r.status_code} - {r.text[:200]}")
    data = r.json()
    emb = np.array(data["embedding"], dtype=np.float32)
    return _unit_norm(emb)
class OllamaEmbeddings:
    def __init__(self, model: str = EMBEDDING_MODEL, timeout: float = 20.0):
        self.model = model
        self.timeout = timeout
        # warm-up (optional): discover dim
        try:
            v = _embed_once("ping", timeout=self.timeout)
            self.dim = int(v.shape[0])
        except Exception:
            self.dim = None

    def embed_query(self, text: str):
        t = normalize_persian(text or "")
        key = _cache_key(self.model, t)
        if key in _cache:
            return _cache[key]
        v = _embed_once(t, timeout=self.timeout)
        _cache[key] = v
        return v

    def embed_documents(self, texts, batch_size: int = 32, max_workers: int = 8):
        # normalize & cache-aware
        items = [(i, normalize_persian(t or "")) for i, t in enumerate(texts)]
        results = [None] * len(items)

        def work(idx_text):
            i, t = idx_text
            key = _cache_key(self.model, t)
            if key in _cache:
                return i, _cache[key]
            v = _embed_once(t, timeout=self.timeout)
            _cache[key] = v
            return i, v

        # chunking for better memory control
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(work, it) for it in chunk]
                for f in as_completed(futures):
                    i, v = f.result()
                    results[i] = v

        # sanity check on dims
        if any(r is None for r in results):
            raise RuntimeError("Embedding failed for some documents.")
        if self.dim is not None:
            for r in results:
                if r.shape[0] != self.dim:
                    raise RuntimeError(f"Inconsistent embedding dim: expected {self.dim}, got {r.shape[0]}")
        return results
    


def merge_pages_smart(docs: List[Document], strategy: str = "smart") -> List[Document]:
    """
    strategy:
      - "per_page": هر صفحه جدا بمونه (بدون ادغام)
      - "merge_if_no_terminator": اگر صفحه با علامت پایان جمله ختم نشده بود، به صفحه بعد الصاق شود
      - "smart": مشابه merge_if_no_terminator ولی با محافظت از طول خیلی بزرگ (مثلاً هر merged chunk بیشتر از N کاراکتر نباشد)
    """
    # گروه‌بندی بر اساس source
    groups: Dict[str, List[Document]] = {}
    for d in docs:
        md = d.metadata or {}
        src = md.get("source") or md.get("file_name") or "UNKNOWN_SOURCE"
        groups.setdefault(src, []).append(d)

    out: List[Document] = []

    MAX_CHUNK_CHARS = 2000  # اگر خیلی طولانی شد، با این حد تقسیم می‌کنیم (قابل تنظیم)

    for src, pages in groups.items():
        # مرتب سازی بر اساس page_num
        try:
            pages_sorted = sorted(pages, key=lambda x: int(x.metadata.get("page_num", 0)))
        except Exception:
            pages_sorted = pages

        if strategy == "per_page":
            # فقط پاک‌سازی و عبور
            for p in pages_sorted:
                txt = (p.page_content or "").strip()
                if txt:
                    m = p.metadata.copy() if p.metadata else {}
                    out.append(Document(page_content=txt, metadata=m))
            continue

        # حالت merge_if_no_terminator یا smart
        buffer_text = ""
        buffer_md = None
        buffer_pages = []

        for p in pages_sorted:
            text = (p.page_content or "").strip()
            if text == "":
                continue

            if buffer_text == "":
                buffer_text = text
                buffer_md = p.metadata.copy() if p.metadata else {}
                buffer_pages = [p.metadata.get("page_num")]
                continue

            # اگر صفحهٔ قبل با terminator ختم نشده بود -> الصاق کن
            if not ends_with_sentence_terminator(buffer_text):
                buffer_text = buffer_text + " " + text
                buffer_pages.append(p.metadata.get("page_num"))
            else:
                # اگر buffer خیلی بزرگ شده و استراتژی smart هست، آن را قطعه قطعه کن
                if strategy == "smart" and len(buffer_text) > MAX_CHUNK_CHARS:
                    # تقسیم ساده بر اساس فاصله‌ها (می‌توان بهتر کرد)
                    parts = []
                    s = buffer_text
                    while len(s) > MAX_CHUNK_CHARS:
                        cut = s.rfind(" ", 0, MAX_CHUNK_CHARS)
                        if cut <= 0:
                            cut = MAX_CHUNK_CHARS
                        parts.append(s[:cut].strip())
                        s = s[cut:].strip()
                    if s:
                        parts.append(s)
                    for part in parts:
                        md = buffer_md.copy() if buffer_md else {}
                        md["_pages"] = buffer_pages.copy()
                        out.append(Document(page_content=part, metadata=md))
                else:
                    md = buffer_md.copy() if buffer_md else {}
                    md["_pages"] = buffer_pages.copy()
                    out.append(Document(page_content=buffer_text.strip(), metadata=md))

                # reset buffer با صفحهٔ فعلی
                buffer_text = text
                buffer_md = p.metadata.copy() if p.metadata else {}
                buffer_pages = [p.metadata.get("page_num")]

        # پس از loop صفحات، buffer مانده را اضافه کن
        if buffer_text:
            md = buffer_md.copy() if buffer_md else {}
            md["_pages"] = buffer_pages.copy()
            out.append(Document(page_content=buffer_text.strip(), metadata=md))

    return out    