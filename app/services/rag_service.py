from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .embedding_service import OllamaEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
embeddings = OllamaEmbeddings()
llm = OllamaLLM(model="qwen3:latest", base_url="http://ollama:11434")

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
    print(docs)
    splits = text_splitter.split_documents(docs)
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
