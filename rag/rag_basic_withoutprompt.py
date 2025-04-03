import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from evaluation.rag_eval_ragas import evaluate_rag_pipeline

# 🔹 Set Groq API Key
os.environ["GROQ_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "textSummarization-langchain"

# 🔹 Load and Extract Text from PDF
file_path = "../resources/14 may office drop.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# 🔹 Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 🔹 Store in Vector Database (ChromaDB)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
vector_db = Chroma.from_documents(splits, embedding_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 🔹 Create RAG Pipeline with Groq
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# 🔹 Query and Answer
query = "when was the ride taken and where the source of the ride?"
response = qa_chain.invoke(query)

print(f"🔹 Answer: {response}")

#Evaluate the RAG Pipeline
test_queries = [
    "when was the ride taken?",
    "where the source of the ride."
]
expected_responses = [
    "The ride was taken on May 14, 2024, at 3:18 PM",
    "The source of the ride was 59, Sector 63 Rd, D Block, Sector 63, Noida, Uttar Pradesh 201307, India."
]
result = evaluate_rag_pipeline(llm, qa_chain, retriever, embedding_model, test_queries, expected_responses)
print("\n🔹 Ragas Evaluation Results:")
print(result)
result.to_pandas().to_csv("ragas_evaluation_new.csv", index=False)
