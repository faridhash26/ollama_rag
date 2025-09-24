from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .embedding_service import merge_pages_into_full_sentences ,sanitize_documents_for_chroma ,merge_pages_smart
from langchain_ollama import OllamaEmbeddings
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document


text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=1200,
    chunk_overlap=120,
    separators=[
             "\n\n",                    # پاراگراف
        "\n• ",                    # ⭐️ خط جدید + نقطه‌چین + فاصله (مهم!)
        "\n•",                     # ⭐️ خط جدید + نقطه‌چین (بدون فاصله)
        "\n",                      # خط جدید (حالا مهمه!)
        ". ",                      # نقطه + فاصله
        "، ",                      # ویرگول فارسی + فاصله
        "؛ ",                      # نیمه‌فاصله فارسی
        "—",                       # خط تیره
        "–",                       # خط تیره کوتاهchunk_overlap
        ")",                       # بستن پرانتز
        "(",                       # باز کردن پرانتز
        "«",                       # نقل قول فارسی
        "»",                       # نقل قول فارسی
        ":",                       # دو نقطه
        "…",                       # سه نقطه
    ],
   keep_separator=True,           # ⚠️ حیاتی: علائم رو حفظ کن
    strip_whitespace=True,
    )
embeddings = OllamaEmbeddings(
    model = 'aligh4699/heydariAI-persian-embeddings:latest',
    base_url = 'http://localhost:11434'
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen3-30B-A3B")
CHAT_MODEL_URL = os.getenv("CHAT_MODEL_URL", "https://arvancloudai.ir/gateway/models/Qwen3-30B-A3B/pWcs1fv2u5siLJch1cCHnvyIw6oXDx2eLDK6Ae33icWThd_VdXz4KdurkdFGy0svJZcIeuthMT0Av8a1JIs0etwIBWSq7rjKf5-dH2LbTcKma67lrdfzavueqtuahIMr0ZBmz1gVH18ZaulBMREVy91ZKHa-Kl4_CMe7xAoN0yKD-5GkxTgoVntJK2Qsq3-YlXoB9_nPmPvTgIR0d7rzu8gMcV66H3HuAB_7F6u8uOPcw_p-4rywgwhOl6BITL0j/v1")
API_KEY =os.getenv("OLLAMA_API_KEY")
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key='b9e54544-2d8e-573a-8b14-ec8f498c735d',           # می‌توانی حذف کنی اگر OPENAI_API_KEY در env ست شده
    base_url=CHAT_MODEL_URL,         # فقط تا /v1
    timeout=60,
    max_retries=2,
    temperature= 0.2

)
prompt = ChatPromptTemplate.from_template(
    """
    شما یک کمک‌کننده هوشمند برای پاسخ به سوالات دربارهٔ سامانه ساجد هستید. لطفاً فقط اطلاعات موجود در متن مرجع را مبنای کار خود قرار دهید.
    اگر سوال شما در متن ارائه‌شده پاسخ داده نشده باشد،."

متن مرجع:
{context}

سوال:
{question}

/no_think
"""
)

def create_vectorstore(docs, persist_dir=".chroma_db", merge_strategy="smart"):
    print("input docs:", len(docs))
    merged_docs = merge_pages_smart(docs, strategy=merge_strategy)
    print("after merge:", len(merged_docs))

    safe_docs = sanitize_documents_for_chroma(merged_docs)
    splits = text_splitter.split_documents(safe_docs)
    print("after split:", len(splits))

    # 👇 این بخش جدید است: اضافه کردن chunk_index بر اساس ترتیب ظاهر شدن
    for i, doc in enumerate(splits):
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["chunk_index"] = i  # شماره ترتیبی کلی
        # یا اگر می‌خواهید بر اساس source جدا کنید:
        # doc.metadata["chunk_seq"] = i  # می‌توانید بعداً بهترش کنید

    return Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)

def get_retriever(persist_dir=".chroma_db"):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return ContextualRetriever(vectorstore=vectorstore, k=5, fetch_k=50)

def get_rag_chain(retriever):
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )



class ContextualRetriever(BaseRetriever):
    vectorstore: Chroma
    k: int = 5
    fetch_k: int = 50
    include_neighbors: bool = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # مرحله ۱: بازیابی اولیه با MMR
        base_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.fetch_k}
        )
        initial_docs = base_retriever.invoke(query)

        if not self.include_neighbors:
            return initial_docs

        # مرحله ۲: جمع‌آوری chunk_index و source هر سند
        expanded_docs = []
        seen = set()  # برای جلوگیری از تکرار

        for doc in initial_docs:
            # اضافه کردن خود سند
            key = (doc.metadata.get("source"), doc.metadata.get("chunk_index"))
            if key not in seen:
                expanded_docs.append(doc)
                seen.add(key)

            # پیدا کردن chunk قبلی و بعدی (در همان source)
            current_index = doc.metadata.get("chunk_index")
            source = doc.metadata.get("source")

            if current_index is None or source is None:
                continue

            # جستجوی chunk قبلی
            prev_doc = self._find_chunk_by_index(source, current_index - 1)
            if prev_doc and (source, current_index - 1) not in seen:
                expanded_docs.append(prev_doc)
                seen.add((source, current_index - 1))

            # جستجوی chunk بعدی
            next_doc = self._find_chunk_by_index(source, current_index + 1)
            if next_doc and (source, current_index + 1) not in seen:
                expanded_docs.append(next_doc)
                seen.add((source, current_index + 1))

        # مرتب‌سازی بر اساس chunk_index برای حفظ ترتیب
        expanded_docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))

        return expanded_docs

    def _find_chunk_by_index(self, source: str, index: int) -> Document | None:
        # جستجو در vectorstore با فیلتر
        try:
            results = self.vectorstore.similarity_search(
                "",  # جستجوی خالی — فقط فیلتر مهم است
                k=1,
                filter={"source": source, "chunk_index": index}
            )
            return results[0] if results else None
        except Exception:
            return None