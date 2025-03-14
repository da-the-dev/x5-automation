from llama_index.core.workflow import Context, StopEvent
from src.workflow_events import HasQAExamplesEvent
from src.settings import settings

import json
import openai

async def reply(query_clean: str, qa: list[tuple[str, str]]) -> str:
    # Initialize AsyncOpenAI client
    llm = openai.AsyncOpenAI(
        base_url=settings.llm.BASE_API,
        api_key=settings.llm.API_KEY,
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

    if settings.llm.MODEL == "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "documents", "content": json.dumps(documents, ensure_ascii=False)},
            {"role": "user", "content": f"Запрос пользователя: {query_clean}"},
        ]
    else:
        # For other models, format documents in prompt
        examples_text = ""
        for idx, doc in enumerate(documents):
            examples_text += f"Пример {idx+1}:\nВопрос: {doc['title']}\nОтвет: {doc['content']}\n\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": f"Примеры вопросов и ответов:\n\n{examples_text}\n\nЗапрос пользователя: {query_clean}\n\nОтветь на запрос пользователя в том же стиле, что и приведенные примеры."
            }
        ]

    # Make async API call
    response = await llm.chat.completions.create(
        model=settings.llm.MODEL,
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
