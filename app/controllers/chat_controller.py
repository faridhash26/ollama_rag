from fastapi.responses import StreamingResponse
from app.services.rag_service import get_retriever, get_rag_chain

def chat_handler(question: str):
    retriever = get_retriever()
    rag_chain = get_rag_chain(retriever)
    result = rag_chain.invoke(question)
    return {"response": result.strip()}

async def chat_stream_handler(question: str):
    retriever = get_retriever()
    rag_chain = get_rag_chain(retriever)

    async def generate():
        async for chunk in rag_chain.astream(question):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
