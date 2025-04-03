from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os


os.environ["GROQ_API_KEY"] = "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"
chat = ChatGroq(model_name="mistral-saba-24b", temperature=0)

parser = StrOutputParser()
file_path = "../resources/14 may office drop.pdf"


def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def document_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
    vector_db = Chroma.from_documents(splits, embedding_model)

    return vector_db.as_retriever(search_kwargs={"k": 3})


docs = document_loader(file_path)
retriever = document_splitter(docs)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# system_prompt = (
#     "your task is to extract and summarize the content. "
#     "Use the following pieces of retrieved context. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "what is the cost of the ride?"})
print(results["answer"])
