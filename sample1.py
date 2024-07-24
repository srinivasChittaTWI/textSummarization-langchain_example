from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    groq_api_key="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"
    # Optional if not set as an environment variable
)

system = ("You are a helpful assistant in summarizing the given text. Please summarize the text below in a few "
          "sentences or less")
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
parser = StrOutputParser()

chain = prompt | chat | parser
result = chain.invoke({"text": "Explain the importance of RAG for LLMs."})
print(result)
