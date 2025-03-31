import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from rag_eval_ragas_refactor import evaluate_rag_pipeline_dict

# 🔹 Set Groq API Key
os.environ["GROQ_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 🔹 Load and Extract Text from PDF
pdf_files = ["example_data/rag_explained.pdf", "example_data/LLM overview.pdf", "example_data/rise-of-llm.pdf"]
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
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 🔹 Create RAG Pipeline with Groq
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, question_answer_chain)
#qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# 🔹 Query and Answer
test_queries = [
    "what is diff b/w naive RAG and advanced RAG?",
    "what is LLMOps and how important is it?",
    "summarize about Evaluation of Datasets for LLMs",
    "how to implement Agentic RAG?"
]
test_responses = []
for query in test_queries:
    response = qa_chain.invoke({"input": query})
    print( response["answer"])
    print("\n")
    test_responses.append(response)

print(f"🔹 Answer: {len(test_responses)}")
#Evaluate the RAG Pipeline
# expected_responses = [
#     "Use Naïve RAG if you need a quick, simple RAG system.Use Advanced RAG for real-world applications where accuracy, relevance, and reducing hallucinations matter.",
#     "LLMOps is a methodology for managing the entire LLM lifecycle, addressing challenges such as managing large volumes of data, scaling computational resources, monitoring and maintenance, versioning, and reproducibility.",
#     "Evaluation of Datasets for LLMs involves assessing the quality of the data, understanding the data distribution, and ensuring that the data is representative of the real-world scenarios the LLM will encounter.",
#    # "To implement Agentic RAG, you need to define the agent's goals, capabilities, and constraints, and then train the agent to achieve those goals using reinforcement learning."
# ]
# result = evaluate_rag_pipeline_dict(llm,qa_chain, retriever, embedding_model, test_queries, expected_responses,test_responses)
# print("\n🔹 Ragas Evaluation Results:")
# print(result)
