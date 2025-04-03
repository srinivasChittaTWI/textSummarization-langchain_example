import os

import pytest
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextRecall, Faithfulness, ResponseRelevancy

from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "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"
eval_llm = ChatGroq(model_name="mistral-saba-24b", temperature=0)
eval_llm = LangchainLLMWrapper(eval_llm)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")



@pytest.mark.asyncio
async def test_llm_context_precision_without_reference():
    context_precision = LLMContextPrecisionWithoutReference(llm=eval_llm)

    sample = SingleTurnSample(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["The Eiffel Tower is located in Paris."],
    )

    result = await context_precision.single_turn_ascore(sample)
    assert result > 0.9


@pytest.mark.asyncio
async def test_llm_context_recall():
    context_recall = LLMContextRecall(llm=eval_llm)

    sample = SingleTurnSample(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["Paris is the capital of France."],
    )

    result = await context_recall.single_turn_ascore(sample)
    assert result == 0.0


@pytest.mark.asyncio
async def test_llm_faithfulness():
    faithfulness = Faithfulness(llm=eval_llm)

    sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )

    result = await faithfulness.single_turn_ascore(sample)
    assert result == 1.0


@pytest.mark.asyncio
async def test_llm_responseRelevancy():
    response_relevancy = ResponseRelevancy(llm=eval_llm, embeddings=embedding_model)

    sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )

    result = await response_relevancy.single_turn_ascore(sample)
    assert result >0.9