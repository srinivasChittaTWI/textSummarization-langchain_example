from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

path = "example_data/"
loader = DirectoryLoader(path, glob="**/14 may office drop.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

llm = ChatGroq(model_name="mistral-saba-24b", temperature=0)
generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

dataset.to_pandas().to_csv('autoqueries.csv', index=False)
dataset.upload()
