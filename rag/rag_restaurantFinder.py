import json
import os

import streamlit as st
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 🔹 Set Groq API Key
os.environ["GROQ_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "textSummarization-langchain"
os.environ["RAGAS_APP_TOKEN"] = "apt.4278-6bf0c1789f57-30c1-9949-75a9bee2-0397a"


file_path = "/Users/srinivas.chitta/PycharmProjects/textSummarization-langchain/resources/restaurant_data.json"

# 🔹 Load and store data from json
with open(file_path, "r") as file:
    restaurant_data = json.load(file)

# Convert to a format suitable for retrieval
docs = []
for entry in restaurant_data:
    menu_items = ", ".join([f"{item['item']} (₹{item['price']})" for item in entry["menu"]])  # Fix f-string issue
    document = (
        f"Restaurant: {entry['restaurant']}\n"
        f"Cuisine: {entry['cuisine']}\n"
        f"Type: {entry['type']}\n"
        f"Menu: {menu_items}"
    )
    docs.append(document)

print(f"Loaded {len(docs)} restaurant entries!")


embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
# Store documents in ChromaDB
vector_store = Chroma.from_texts(docs, embedding_model, persist_directory="./chroma_db")

print("✅ Embeddings stored in ChromaDB")
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 🔹 Create RAG Pipeline with Groq
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.3)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# 🔹 Query and Answer
# query = "What are some Indian vegetarian dishes available?"
#are there any vegan restaurants in the area?
#which restaurant serves the Chicken Biryani with lower price?

# response = qa_chain.invoke(query)

st.title("🍽️ Restaurant Finder Chatbot")
st.write("Ask questions about nearby restaurants and their menus!")

user_query = st.text_input("🔍 Enter your query:")
response = None

if user_query:
    with st.spinner("Fetching answer..."):
        response = qa_chain.invoke({"query": user_query})
        st.write("### ✅ Answer:")
        st.write(response["result"])  # Display the retrieved answer


print(f"🔹 Answer: {response}")

