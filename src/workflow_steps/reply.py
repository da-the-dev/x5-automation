from llama_index.core.workflow import Context, StopEvent
from src.workflow_events import HasQAExamplesEvent

import json
from os import getenv
import openai

async def reply(query_clean: str, qa: list[tuple[str, str]]) -> str:
    # Initialize AsyncOpenAI client
    llm = openai.AsyncOpenAI(
        base_url=getenv("VLLM_LLM_BASE_API"),
        api_key=getenv("VLLM_LLM_API_KEY"),
    )

    # Format QA pairs as documents
    documents = []
    for idx, (q, a) in enumerate(qa):
        documents.append({"doc_id": idx, "question": q, "answer": a})

    # Create chat messages
    system_prompt = (
        "Ты помощник, который дает ответы в том же стиле, что и представленные примеры. "
        "Используй предоставленные вопросы и ответы как образец стиля и уровня детализации."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "documents", "content": json.dumps(documents, ensure_ascii=False)},
        {"role": "user", "content": f"Запрос пользователя: {query_clean}"},
    ]

    # Make async API call
    response = await llm.chat.completions.create(
        model=getenv("VLLM_LLM_MODEL"),
        messages=messages,
        max_tokens=512,
        temperature=0.5,
    )

    # Return just the response content
    return response.choices[0].message.content

async def reply_step(ev: HasQAExamplesEvent, ctx: Context) -> StopEvent:
    qa = ev.qa
    query_clean = await ctx.get("query_clean")

    result = await reply(query_clean, qa)
    return StopEvent(result=result)
