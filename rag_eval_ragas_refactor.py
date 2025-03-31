from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy


def evaluate_rag_pipeline_common(llm, qa_chain, retriever, embedding_model, test_queries, expected_responses,test_responses,
                                 use_dict=False):
    test_responses = test_responses
    contexts = []

    for query in test_queries:
        # if use_dict:
        #     test_responses.append(qa_chain.invoke({"input": query}))
        # else:
        #     test_responses.append(qa_chain.invoke(query))
        retrieved_docs = [retriever.invoke(q) for q in test_queries]

        # Convert Contexts to List of Lists
        contexts.append([" ".join([doc.page_content for doc in docs]) if docs else "" for docs in retrieved_docs])
        responses = [r["response"] if isinstance(r, dict) and "response" in r else str(r) for r in test_responses]

    data = {
        "user_input": test_queries,
        "retrieved_contexts": contexts,
        "response": responses,
        "reference": expected_responses
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)
    evaluator_llm = LangchainLLMWrapper(llm)

    # Compute RAG Metrics
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=evaluator_llm,
        embeddings=embedding_model
    )
    return result


def evaluate_rag_pipeline(llm, qa_chain, retriever, embedding_model, test_queries, expected_responses, test_responses):
    return evaluate_rag_pipeline_common(llm, qa_chain, retriever, embedding_model, test_queries, expected_responses, test_responses,
                                        use_dict=False)


def evaluate_rag_pipeline_dict(llm, qa_chain, retriever, embedding_model, test_queries, expected_responses, test_responses):
    return evaluate_rag_pipeline_common(llm, qa_chain, retriever, embedding_model, test_queries, expected_responses, test_responses,
                                        use_dict=True)
