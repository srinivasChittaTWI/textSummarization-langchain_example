import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

os.environ["GROQ_API_KEY"] = "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"
eval_llm = ChatGroq(model_name="mistral-saba-24b", temperature=0)
eval_llm = LangchainLLMWrapper(eval_llm)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
os.environ["RAGAS_APP_TOKEN"] = "apt.4278-6bf0c1789f57-30c1-9949-75a9bee2-0397a"

path = "../resources/"
loader = DirectoryLoader(path, glob="**/14 may office drop.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

llm = ChatGroq(model_name="mistral-saba-24b", temperature=0)
generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

dataset.to_pandas().to_csv('../resources/output_export/autotestdata.csv', index=False)
dataset.upload()
