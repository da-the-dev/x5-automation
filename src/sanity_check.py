from llama_index.llms.openai_like import OpenAILike
import json


def sanity_check(query_clean: str, qa_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    # Initialize LLM with vLLM backend
    llm = OpenAILike(
        api_base="http://localhost:8000/v1",
        api_key="token-123",
        model="Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct",
    )

    # Process QA pairs in batches
    batch_size = 5  # Adjust if needed
    filtered_qa = []

    for i in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[i : i + batch_size]

        # Russian prompt
        prompt = (
            f"У нас есть запрос: '{query_clean}'\n\n"
            f"Ниже дан набор Q&A пар (в количестве {len(batch)}). Для каждой пары ответь, "
            f"является ли она релевантной запросу. Нужно вернуть ровно один список JSON, "
            f"где каждый элемент — строка '1' или '0'. '1', если пара релевантна, "
            f"'0' — если нерелевантна.\n\n"
        )
        index = 1
        for q, a in batch:
            prompt += f"Пара {index}:\nВопрос: {q}\nОтвет: {a}\n\n"
            index += 1
        prompt += "Ответ должен быть в формате: [\"0\" или \"1\", \"0\" или \"1\", ...]."

        # Use guided_json to produce array of strings (each '0' or '1')
        # enum: ["0", "1"] ensures only '0' or '1'
        response = llm.complete(
            prompt,
            extra_body={
                "guided_json": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["0", "1"]
                    }
                }
            }
        )
        # response.text should be a JSON array of 0/1 strings
        try:
            scores = json.loads(response.text)
            # Double-check length. If it's not correct, skip or handle gracefully
            if len(scores) == len(batch):
                for (q, a), score in zip(batch, scores):
                    if score == "1":
                        filtered_qa.append((q, a))
        except json.JSONDecodeError:
            # If we can't parse it, we skip or handle differently
            pass

    return filtered_qa
