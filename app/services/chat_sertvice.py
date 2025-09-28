from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from app.services.vector_store import ChromaVectorService
import os
import asyncio

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
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
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

        res = await self.rag_chain.arun(question)
        answer = res.get("result", "")
        sources = [doc.metadata.get("source", "") for doc in res.get("source_documents", [])]

        return {"answer": answer, "sources": sources}
