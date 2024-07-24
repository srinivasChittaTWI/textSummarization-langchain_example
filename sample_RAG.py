from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    groq_api_key="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"
    # Optional if not set as an environment variable
)

parser = StrOutputParser()
file_path = "example_data/14 may office drop.pdf"


def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def document_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vectorstore.as_retriever()


docs = document_loader(file_path)
retriever = document_splitter(docs)

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

system_prompt = (
    "your task is to extract and summarize the content. "
    "Use the following pieces of retrieved context. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": ""})
print(results["answer"])
