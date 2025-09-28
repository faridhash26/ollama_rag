# vector_service.py
import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain.schema import BaseRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------
# تنظیمات پیش‌فرض
# ----------------------------
DEFAULT_CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_store")
DEFAULT_COLLECTION = "pdf_docs"
DEFAULT_EMBED_MODEL = os.environ.get(
    "EMBED_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ----------------------------
# کلاس سرویس برداری (Chroma)
# ----------------------------
class ChromaVectorService:
    def __init__(
        self,
        chroma_path: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        chroma_path: مسیر persistent chroma (پوشه)
        collection_name: نام کالکشن در chroma
        embed_model_name: مدل sentence-transformers برای امبدینگ
        executor: ThreadPoolExecutor برای اجرای blocking tasks
        """
        self.chroma_path = chroma_path or DEFAULT_CHROMA_DIR
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self.executor = executor or ThreadPoolExecutor(max_workers=4)

        # init chroma client (PersistentClient معمولی)
        try:
            self.client = chromadb.PersistentClient(path=self.chroma_path)
        except Exception:
            # fallback به Client اگر PersistentClient در نسخه شما متفاوت باشه
            self.client = chromadb.Client()

        # ایجاد یا واکشی کالکشن
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        # مدل امبدینگ (بارگذاری در ترد اصلی؛ encode در threadpool اجرا می‌کنیم)
        self.embedder = SentenceTransformer(self.embed_model_name)

    # ----------------------------
    # helper: chunk متن با overlap
    # ----------------------------
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        L = len(text)
        while start < L:
            end = min(start + chunk_size, L)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == L:
                break
            start = end - overlap if (end - overlap) > start else end
        return chunks

    # ----------------------------
    # helper: synchronous embedding (ممکنه blocking) - اجرا در executor توصیه میشه
    # ----------------------------
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        embs = self.embedder.encode(texts, show_progress_bar=False)
        # tolist if numpy
        if hasattr(embs, "tolist"):
            return embs.tolist()
        return list(map(list, embs))

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._embed_sync, texts)

    # ----------------------------
    # index کردن pages_out (همزمانی async)
    # pages_out: لیست دیکشنری‌هایی مثل {"page_index": i, "full_text": "..."}
    # filename: برای ساخت id منحصر به فرد
    # returns: index_ref (str) یا "" اگر چیزی ایندکس نشد
    # ----------------------------
    async def index_pages(
        self,
        pages_out: List[Dict[str, Any]],
        filename: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        persist: bool = True,
    ) -> str:
        docs: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for p in pages_out:
            page_idx = p.get("page_index")
            full_text = (p.get("full_text") or "").strip()
            if not full_text:
                continue
            chunks = self.chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
            for ci, ch in enumerate(chunks):
                doc_id = f"{filename}::p{page_idx}::c{ci}"
                docs.append(ch)
                ids.append(doc_id)
                metadatas.append({"filename": filename, "page_index": page_idx, "chunk_index": ci})

        if not docs:
            return ""

        # ساخت امبدینگ‌ها (async)
        embeddings = await self.embed_texts(docs)

        # افزودن به کالکشن
        self.collection.add(documents=docs, ids=ids, metadatas=metadatas, embeddings=embeddings)

        # persist اگر ممکنه
        try:
            if hasattr(self.client, "persist"):
                self.client.persist()
        except Exception:
            pass

        index_ref = f"chroma://{filename}::{int(time.time())}"
        return index_ref

    # ----------------------------
    # بازیابی
    # ----------------------------
    async def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        برمی‌گرداند ساختاری مثل chroma.query:
        {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
        """
        loop = asyncio.get_running_loop()
        # chroma.query بلوکینگ نیست معمولاً، اما برای یکنواختی از executor استفاده کن
        return await loop.run_in_executor(self.executor, lambda: self.collection.query(query_texts=[query_text], n_results=top_k))

    # ----------------------------
    # حذف بر اساس filename (تمام داکیومنت‌هایی که metadata filename==filename دارند)
    # Chroma ممکن است api برای delete وجود داشته باشد: delete(where={"filename": filename})
    # ----------------------------
    async def delete_by_filename(self, filename: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        def _delete():
            # برخی نسخه‌های chroma از delete_documents(ids=[]) پشتیبانی دارند.
            # ولی راحت‌ترین راه استفاده از delete(where=...) است (در صورت پشتیبانی)
            try:
                # try where-based deletion
                self.collection.delete(where={"filename": filename})
                if hasattr(self.client, "persist"):
                    self.client.persist()
                return {"deleted_by": "where", "filename": filename}
            except Exception:
                # fallback: حذف تک‌تک ids که با filename شروع می‌شوند
                try:
                    # لیست همه متادیتا را بگیر و فیلتر کن
                    # توجه: get may be heavy on large collections
                    # در بعضی نسخه‌ها collection.get() نیاز به پارامترها دارد
                    all_meta = self.collection.get(include=["ids", "metadatas"])
                    ids = all_meta.get("ids", [])
                    metadatas = all_meta.get("metadatas", [])
                    # flatten if nested lists
                    if ids and isinstance(ids[0], list):
                        ids = ids[0]
                    if metadatas and isinstance(metadatas[0], list):
                        metadatas = metadatas[0]
                    to_delete = [i for i, m in zip(ids, metadatas) if m.get("filename") == filename]
                    if to_delete:
                        self.collection.delete(ids=to_delete)
                        if hasattr(self.client, "persist"):
                            self.client.persist()
                    return {"deleted_by": "ids", "count": len(to_delete)}
                except Exception as e:
                    return {"error": str(e)}
        return await loop.run_in_executor(self.executor, _delete)

    # ----------------------------
    # حذف کل کالکشن (ریست)
    # ----------------------------
    async def reset_collection(self) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        def _reset():
            try:
                # در chroma ممکنه api remove_collection داشته باشه
                if hasattr(self.client, "delete_collection"):
                    self.client.delete_collection(self.collection_name)
                elif hasattr(self.client, "delete_collection"):
                    self.client.delete_collection(name=self.collection_name)
                else:
                    # fallback: recreate same name by dropping and creating anew
                    self.client.get_or_create_collection(name=self.collection_name).delete()
                # recreate
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                if hasattr(self.client, "persist"):
                    self.client.persist()
                return {"status": "reset"}
            except Exception as e:
                return {"error": str(e)}
        return await loop.run_in_executor(self.executor, _reset)

    # ----------------------------
    # متد کمکی: تعداد آیتم‌ها / آمار
    # ----------------------------
    async def stats(self) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        def _stats():
            try:
                info = self.collection.count()
                return {"count": info}
            except Exception:
                # fallback: get size by getting ids
                try:
                    all_meta = self.collection.get(include=["ids"])
                    ids = all_meta.get("ids", [])
                    if ids and isinstance(ids[0], list):
                        return {"count": len(ids[0])}
                    return {"count": 0}
                except Exception as e:
                    return {"error": str(e)}
        return await loop.run_in_executor(self.executor, _stats)
    def get_retriever(self, search_kwargs: dict = None):
        search_kwargs = search_kwargs or {"k": 3}

        # embedding درست
        embed = SentenceTransformerEmbeddings(model_name=self.embed_model_name)

        # persist_directory مستقیم داده شود
        retriever = Chroma(
            collection_name=self.collection_name,
            embedding_function=embed,
            persist_directory=self.chroma_path,  # ← مستقیم بده
            client_settings=None                  # ← هیچ dict نده
        ).as_retriever(search_kwargs=search_kwargs)

        return retriever
    async def index_single_document(self, filename: str, content: str):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(content)

            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
            ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

            # ذخیره در کروما (همان روش قبلی شما)
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            return {"status": "indexed", "chunks": len(chunks)}