from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .embedding_service import merge_pages_into_full_sentences ,sanitize_documents_for_chroma
from langchain_ollama import OllamaEmbeddings


text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=600,
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
        "–",                       # خط تیره کوتاه
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

)
prompt = ChatPromptTemplate.from_template(
    """
    شما یک کمک‌کننده هوشمند برای پاسخ به سوالات دربارهٔ سامانه ساجد هستید. لطفاً فقط اطلاعات موجود در متن مرجع را مبنای کار خود قرار دهید.
    اگر سوال شما در متن ارائه‌شده پاسخ داده نشده باشد، بنویسید: "فقط به سوال ها در مورد سمانه ساجد میتوانم پاسخ بدم."

متن مرجع:
{context}

سوال:
{question}

/no_think
"""
)

def create_vectorstore(docs, persist_dir=".chroma_db"):
    merged_docs = merge_pages_into_full_sentences(docs)
    print('merged_docs count:', len(merged_docs))

    # مهم: metadata ها را sanitize کن
    safe_docs = sanitize_documents_for_chroma(merged_docs)

    splits = text_splitter.split_documents(safe_docs)
    print('splits count:', len(splits))

    return Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)


def get_retriever(persist_dir=".chroma_db"):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def get_rag_chain(retriever):
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
