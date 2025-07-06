from io import BytesIO
from pypdf import PdfReader
from langchain.docstore.document import Document

def extract_docs_from_pdf(content: bytes, filename: str):
    pdf_reader = PdfReader(BytesIO(content))
    docs = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs
