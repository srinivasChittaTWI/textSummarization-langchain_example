import os
from typing import List


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from pydantic import field_validator
from langchain_core.documents import Document


# 🔹 Set Groq API Key
os.environ["GROQ_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SERPAPI_API_KEY"] = "26a1c3fd42e8a6165c513f59ee1fa5510d198473ce2a962a649a618d6d464601"

# 🔹 Load and Extract Text from PDF
pdf_files = ["resources/rag_explained.pdf", "resources/LLM overview.pdf", "resources/rise-of-llm.pdf"]
docs = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())

# 🔹 Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 🔹 Store in Vector Database (ChromaDB)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
vector_db = Chroma.from_documents(splits, embedding_model)
pdf_retriever = vector_db.as_retriever(search_kwargs={"k": 2})

google_search = SerpAPIWrapper()
search_tool = Tool(
    name="Web Search",
    func=google_search.run,
    description="Perform a Google search to get the latest information."
)


# 🔹 Fusion Retriever
class HybridRetriever(BaseRetriever):
    pdf_retriever: BaseRetriever
    web_search_tool: Tool

    @field_validator("pdf_retriever", "web_search_tool", mode="before")
    @classmethod
    def validate_fields(cls, value):
        if value is None:
            raise ValueError("Both pdf_retriever and web_search_tool must be provided")
        return value

    def _get_relevant_documents(self, query: str) -> List[Document]:
        pdf_docs = self.pdf_retriever.invoke(query)
        web_results = self.web_search_tool.func(query)
        web_doc = Document(page_content=web_results)  # Convert web text into Document format
        return pdf_docs + [web_doc]


# Instantiate HybridRetriever properly
hybrid_retriever = HybridRetriever(pdf_retriever=pdf_retriever, web_search_tool=search_tool)

# 🔹 Create RAG Pipeline with Groq
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=hybrid_retriever, chain_type="stuff")

# 🔹 Query and Answer
queries = [
    "what is diff b/w naive RAG and advanced RAG?",
    "what is LLMOps and how important is it?",
    "summarize about Evaluation of Datasets for LLMs",
    "how to implement Agentic RAG?",
]
for query in queries:
    response = qa_chain.invoke(query)
    print(f"🔹 Answer: {response}")
    print("\n")

# Evaluate the RAG Pipeline
