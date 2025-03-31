import json

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate, RunConfig
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy

from sample_RAG_restaurant import qa_chain, retriever


# Load the dataset
with open("example_data/restaurant_goldendataset.json", "r") as f:
    data = json.load(f)

# Convert to RAGAS Dataset format
dataset = Dataset.from_list(data)
print(dataset)

test_responses = []
contexts = []
for sample in data:
    query = sample["user_input"]
    retrieved_docs = retriever.invoke(query)
    print(retrieved_docs)
    response = qa_chain.invoke({"query": query})["result"]
    test_responses.append(response)
    #contexts.append([" ".join([doc[0].page_content for doc in docs]) if docs else "" for docs in retrieved_docs])
    contexts.append([" ".join(doc.page_content for doc in retrieved_docs)] if retrieved_docs else "")
# Add generated responses to dataset
dataset_list = dataset.to_list()

# Add responses to each entry
for i in range(len(dataset_list)):
    dataset_list[i]["response"] = test_responses[i]  # Ensure `test_responses` has correct length
    dataset_list[i]["retrieved_contexts"] = contexts[i]

# Recreate the dataset with the new responses
dataset_with_responses = Dataset.from_list(dataset_list)
print(dataset_with_responses)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0)
evaluator_llm = LangchainLLMWrapper(llm)
config = RunConfig(
    timeout=180,
    max_retries=3,
)

results = evaluate(
    dataset_with_responses,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    llm=evaluator_llm,
    embeddings=embedding_model,
    run_config=config
)
df = results.to_pandas()
df.to_csv('score.csv', index=False)

results.upload()

print(results)
