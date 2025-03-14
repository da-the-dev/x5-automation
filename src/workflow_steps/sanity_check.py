from llama_index.core.workflow import Context
from src.workflow_events import DeduplicateEvent, SanityCheckEvent
from src.settings import settings

import json
import openai
import asyncio

async def process_batch(
    llm, query_clean: str, batch: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Process a single batch of QA pairs and return the relevant ones."""
    # System prompt for grounded responses
    system_prompt = (
        "Твоя задача - определить, релевантны ли предоставленные документы запросу пользователя. "
        "Релевантым считай тот документ, в котором тема хотя бы смежно связана с запросом. "
        "Верни ровно один массив из строк '0' или '1', где '1' означает, что документ релевантен запросу, "
        "а '0' - что нерелевантен. Массив должен иметь ровно столько элементов, сколько документов в запросе."
    )

    # If Vikhr model with documents role
    if settings.llm.MODEL == "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24":
        # Format QA pairs as documents
        documents = []
        for idx, (q, a) in enumerate(batch):
            documents.append({"doc_id": idx, "title": q, "content": a})

        # Create chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "documents", "content": json.dumps(documents, ensure_ascii=False)},
            {
                "role": "user",
                "content": f"Запрос: '{query_clean}'. Оцени релевантность каждого документа к этому запросу и верни массив из {len(documents)} элементов, где каждый элемент - '0' или '1'.",
            },
        ]
    elif settings.llm.MODEL == "google/gemma-2-9b-it":
        # For google/gemma-2-9b-it
        documents_text = ""
        for idx, (q, a) in enumerate(batch):
            documents_text += f"Документ {idx}:\nВопрос: {q}\nОтвет: {a}\n\n"
        
        messages = [
            {
                "role": "user",
                "content": f"{system_prompt}\n\nЗапрос: '{query_clean}'\n\nДокументы:\n{documents_text}\n\nОцени релевантность каждого документа к этому запросу и верни массив из {len(batch)} элементов, где каждый элемент - '0' или '1'."
            }
        ]
    else:
        # For other models, format documents in prompt
        documents_text = ""
        for idx, (q, a) in enumerate(batch):
            documents_text += f"Документ {idx}:\nВопрос: {q}\nОтвет: {a}\n\n"
        
        messages = [
            {"role": "user", "content": system_prompt},
            {
                "role": "user",
                "content": f"Запрос: '{query_clean}'\n\nДокументы:\n{documents_text}\n\nОцени релевантность каждого документа к этому запросу и верни массив из {len(batch)} элементов, где каждый элемент - '0' или '1'."
            }
        ]

    print("MESSAGES")
    print(messages)

    # Call the API with guided_json in extra_body
    response = await llm.chat.completions.create(
        model=settings.llm.MODEL,
        messages=messages,
        temperature=0.0,
        extra_body={
            "guided_json": {
                "type": "array",
                "items": {"type": "number", "enum": [0, 1]},
            }
        },
    )

    # Extract response and parse scores
    response_text = response.choices[0].message.content
    scores = list(map(int, json.loads(response_text)))

    # Ensure the length is correct
    if len(scores) != len(batch):
        if len(scores) < len(batch):
            # If too short, extend with zeros
            scores.extend([0] * (len(batch) - len(scores)))
        else:
            # If too long, truncate
            scores = scores[: len(batch)]

    # Add relevant QA pairs to results
    filtered_qa = []
    for (q, a), score in zip(batch, scores):
        if score == 1:  # Check for integer value
            filtered_qa.append((q, a))

    return filtered_qa

async def sanity_check(
    query_clean: str, qa_pairs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    # Initialize LLM with OpenAI interface
    llm = openai.AsyncOpenAI(
        base_url=settings.llm.BASE_API,
        api_key=settings.llm.API_KEY,
    )

    # Process QA pairs in batches
    batch_size = 10  # Adjust if needed

    # Prepare batches
    batches = [
        qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)
    ]

    # Process all batches concurrently
    tasks = [process_batch(llm, query_clean, batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    # Flatten the results
    filtered_qa = [item for sublist in results for item in sublist]

    return filtered_qa

async def sanity_check_step(ev: DeduplicateEvent, ctx: Context) -> SanityCheckEvent:
    qa = ev.qa
    query_clean = await ctx.get("query_clean")

    sane_qa = await sanity_check(query_clean, qa)
    return SanityCheckEvent(qa=sane_qa)
