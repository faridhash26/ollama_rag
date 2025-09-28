from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from app.services.vector_store import ChromaVectorService
import os
import asyncio
from langchain.schema import HumanMessage, SystemMessage
from typing import List

class ChatRAGService:
    def __init__(self):
        # تنظیمات مدل
        self.CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen3-30B-A3B")
        self.CHAT_MODEL_URL = os.getenv(
            "CHAT_MODEL_URL",
            "https://arvancloudai.ir/gateway/models/Qwen3-30B-A3B/pWcs1fv2u5siLJch1cCHnvyIw6oXDx2eLDK6Ae33icWThd_VdXz4KdurkdFGy0svJZcIeuthMT0Av8a1JIs0etwIBWSq7rjKf5-dH2LbTcKma67lrdfzavueqtuahIMr0ZBmz1gVH18ZaulBMREVy91ZKHa-Kl4_CMe7xAoN0yKD-5GkxTgoVntJK2Qsq3-YlXoB9_nPmPvTgIR0d7rzu8gMcV66H3HuAB_7F6u8uOPcw_p-4rywgwhOl6BITL0j/v1"
        )

        # LLM async
        self.llm = ChatOpenAI(
            model=self.CHAT_MODEL,
            base_url=self.CHAT_MODEL_URL,
            temperature=0.2,
            timeout=60,
            max_retries=2,
            streaming=False,
            openai_api_key=os.getenv("OPENAI_API_KEY", "b9e54544-2d8e-573a-8b14-ec8f498c735d")
        )

        # سرویس برداری
        self.vector_service = ChromaVectorService()
        self.retriever = self._get_retriever()

        # prompt سفارشی
        self.prompt = ChatPromptTemplate.from_template(
            """
            شما یک کمک‌کننده هوشمند برای پاسخ به سوالات دربارهٔ سامانه ساجد هستید. لطفاً فقط اطلاعات موجود در متن مرجع را مبنای کار خود قرار دهید.
            اگر سوال شما در متن ارائه‌شده پاسخ داده نشده باشد، پاسخ ندهید.

            متن مرجع:
            {context}

            سوال:
            {question}

            /no_think
            """
        )

        # chain RAG
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def _get_retriever(self, search_kwargs: dict = None):
        search_kwargs = search_kwargs or {"k": 3}

        embed = SentenceTransformerEmbeddings(
            model_name=self.vector_service.embed_model_name
        )

        retriever = Chroma(
            collection_name=self.vector_service.collection_name,
            embedding_function=embed,
            persist_directory=self.vector_service.chroma_path,
            client_settings=None
        ).as_retriever(search_kwargs=search_kwargs)

        return retriever

    async def chat(self, question: str):
        if not question.strip():
            return {"answer": "", "sources": []}

        res = await self.rag_chain.acall({"query": question})
        answer = res['result']
        sources = [doc.metadata.get('source', '') for doc in res.get('source_documents', [])]

        return {"answer": answer, "sources": sources}
    async def enhance_document_text(self, raw_text: str, chunk_size: int = 2500) -> str:
        if not raw_text or not raw_text.strip():
            return raw_text

        # اگر متن کوتاه است، یک‌جا پردازش شود
        if len(raw_text) <= chunk_size:
            return await self._enhance_chunk(raw_text)

        # تقسیم به چانک‌های هوشمند (از خط جدید شروع شود)
        chunks = self._smart_split_text(raw_text, max_length=chunk_size)
        enhanced_chunks = []

        for chunk in chunks:
            try:
                enhanced = await self._enhance_chunk(chunk)
                print('enhanced length:', len(enhanced))
                print('enhanced preview:', repr(enhanced[:200]))
                enhanced_chunks.append(enhanced)
            except Exception as e:
                print(f"⚠️ Failed to enhance chunk: {e}")
                enhanced_chunks.append(chunk)  # fallback

        return "\n\n".join(enhanced_chunks)

    def _smart_split_text(self, text: str, max_length: int) -> List[str]:
        """تقسیم متن به چانک‌هایی که از خط جدید شروع/پایان می‌شوند."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_length
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            # سعی کن از خط جدید قطع کنی
            split_point = text.rfind('\n\n', start, end)
            if split_point == -1:
                split_point = text.rfind('\n', start, end)
            if split_point == -1:
                split_point = end
            chunks.append(text[start:split_point].strip())
            start = split_point + 1
        return chunks

    async def _enhance_chunk(self, chunk: str, max_retries: int = 2) -> str:
        system_message = SystemMessage(
            content=(
                "You are an expert Persian document processor. "
                "Correct OCR errors and return ONLY the cleaned text. "
                "Do not add any extra text, explanations, or formatting."
                "/no_think"
            )
        )
        human_message = HumanMessage(content=chunk)

        for attempt in range(max_retries + 1):
            try:
                # ⬇️ کلید اصلی: محدود کردن خروجی
                response =  self.llm.invoke(
                    [system_message, human_message],
                )
                dataresponse = response.content.strip() or chunk

                print("dataresponse" , response )
                print("human_message" ,human_message )
                return dataresponse
            except Exception as e:
                if attempt == max_retries:
                    print(f"⚠️ Chunk enhancement failed: {e}")
                    return chunk
                await asyncio.sleep(2 ** attempt)
        return chunk