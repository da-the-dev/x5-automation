from llama_index.core.workflow import Context, StopEvent
from src.workflow_events import HasQAExamplesEvent
from src.settings import settings

import json
import openai


async def reply(query_clean: str, qa: list[tuple[str, str]], clear_history: list[dict] = None) -> str:
    # Initialize AsyncOpenAI client
    llm = openai.AsyncOpenAI(
        base_url=settings.llm.BASE_API,
        api_key=settings.llm.API_KEY,
    )

    # Create chat messages
    system_prompt = (
        "Ты помощник, который дает ответы на основе предоставленных примеров вопросов и ответов."
        "Используй предоставленные вопросы и ответы как образец стиля и уровня детализации."
        "Обращай внимание на прошлые сообщения для ответа на запрос пользователя."
        "Не задавай уточняющих вопросов."
        "Если примеры вопросов и ответов не содержат релевантной для запроса информации, не придумывай ответ, а дай знать пользователю."
    )

    if settings.llm.MODEL == "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24":
        # Format QA pairs as documents
        documents = []
        for idx, (q, a) in enumerate(qa):
            documents.append({"doc_id": idx, "question": q, "answer": a})

        # Base messages with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add filtered chat history
        messages.extend(clear_history)

        # Add current documents and query
        messages.extend([
            {"role": "documents", "content": json.dumps(documents, ensure_ascii=False)},
            {"role": "user", "content": query_clean},
        ])

    elif settings.llm.MODEL == "google/gemma-2-9b-it":
        # For google/gemma-2-9b-it
        examples_text = ""
        for idx, doc in enumerate(documents):
            examples_text += f"Пример {idx+1}:\nВопрос: {doc['question']}\nОтвет: {doc['answer']}\n\n"

        # Add all messages and add system prompt to the first message and examples to last message
        messages = clear_history + [{"role": "user", "content": query_clean}]
        messages[-1]["content"] = f"Примеры вопросов и ответов:\n{examples_text}\n\nЗапрос пользователя: {messages[-1]['content']}"
        messages[0]["content"] = f"{system_prompt}\n\n{messages[0]['content']}"

    else:
        # For other models, format documents in prompt
        examples_text = ""
        for idx, doc in enumerate(documents):
            examples_text += f"Пример {idx+1}:\nВопрос: {doc['question']}\nОтвет: {doc['answer']}\n\n"
        
        # Base messages with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add filtered chat history
        messages.extend(clear_history)

        # Add current documents and query
        messages.extend([
            {"role": "user", "content": f"Примеры вопросов и ответов:\n{examples_text}\n\nЗапрос пользователя: {query_clean}"},
        ])

    print("\n--- Messages ---")
    for i, msg in enumerate(messages):
        print(f"{i}: {msg['role']} - {msg['content']}")

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
    clear_history = await ctx.get("clear_history")
    
    result = await reply(query_clean, qa, clear_history)
    return StopEvent(result=(result, query_clean))
