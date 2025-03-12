# from llama_index.llms.openai_like import OpenAILike
# import openai
# import json

# from src.config import config


# llm: openai.OpenAI = openai.OpenAI(
#     base_url=config["api_base"],
#     api_key=config["api_key"],
# )


# GROUNDED_SYSTEM_PROMPT = "Your task is to answer the user's questions using only the information from the provided documents. Give two answers to each question: one with a list of relevant document identifiers and the second with the answer to the question itself, using documents with these identifiers."

# documents = [
#   {
#     "doc_id": 0,
#     "title": "Глобальное потепление: ледники",
#     "content": "За последние 50 лет объем ледников в мире уменьшился на 30%"
#   },
#   {
#     "doc_id": 1,
#     "title": "Глобальное потепление: Уровень моря",
#     "content": "Уровень мирового океана повысился на 20 см с 1880 года и продолжает расти на 3,3 мм в год"
#   }
# ]
# sample_history = [
#     {'role': 'system', 'content': GROUNDED_SYSTEM_PROMPT}, 
#     {'role': 'documents', 'content': json.dumps(documents, ensure_ascii=False)},
#     {'role': 'user', 'content': 'Глоабльное потепление'}
# ]
# relevant_indexes = llm.chat.completions.create(
#     model=config["llm"], 
#     messages=sample_history, 
#     temperature=0.0, 
#     max_tokens=2048
# )

# print(relevant_indexes)

# # print('Using documents: ' + relevant_indexes + '\n----')
# # final_answer = llm_client.chat.completions.create(
# #     model=llm_model,
# #     messages=sample_history + [{'role': 'assistant', 'content': relevant_indexes}],
# #     temperature=0.3,
# #     max_tokens=2048
# # ).choices[0].message.content

# # print(final_answer)

from llama_index.llms.openai_like import OpenAILike
import openai
import json

from src.config import config

def sanity_check(
    query_clean: str, qa_pairs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    # Initialize LLM with OpenAI interface
    llm = openai.OpenAI(
        base_url=config["api_base"],
        api_key=config["api_key"],
    )

    # Process QA pairs in batches
    batch_size = 2  # Adjust if needed
    filtered_qa = []

    for i in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[i : i + batch_size]
        
        # System prompt for grounded responses
        system_prompt = (
            "Твоя задача - определить, релевантны ли предоставленные документы запросу пользователя. "
            "Верни ровно один массив из строк '0' или '1', где '1' означает, что документ релевантен запросу, "
            "а '0' - что нерелевантен. Массив должен иметь ровно столько элементов, сколько документов в запросе."
        )
        
        # Format QA pairs as documents
        documents = []
        for idx, (q, a) in enumerate(batch):
            documents.append({
                "doc_id": idx,
                "title": q,
                "content": a
            })
        
        # Create chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "documents", "content": json.dumps(documents, ensure_ascii=False)},
            {"role": "user", "content": f"Запрос: '{query_clean}'. Оцени релевантность каждого документа к этому запросу и верни массив из {len(documents)} элементов, где каждый элемент - '0' или '1'."}
        ]
        
        print("\n---- Processing batch ----")
        for doc in documents:
            print(f"Doc {doc['doc_id']}: {doc['title']} - {doc['content']}")
        
        try:
            # Call the API with guided_json in extra_body
            response = llm.chat.completions.create(
                model=config["llm"],
                messages=messages,
                temperature=0.0,
                extra_body={
                    "guided_json": {
                        "type": "array",
                        "items": {"type": "number", "enum": [0, 1]},
                    }
                }
            )
            
            # Extract response
            response_text = response.choices[0].message.content
            print("\n---- Response ----")
            print(response_text)
            
            # Parse the response as a JSON array and ensure all elements are integers
            scores = list(map(int, json.loads(response_text)))
            
            # Ensure the length is correct
            if len(scores) != len(batch):
                print(f"Warning: Expected {len(batch)} scores but got {len(scores)}. Adjusting...")
                if len(scores) < len(batch):
                    # If too short, extend with zeros
                    scores.extend([0] * (len(batch) - len(scores)))
                else:
                    # If too long, truncate
                    scores = scores[:len(batch)]
            
            # Add relevant QA pairs to results
            for (q, a), score in zip(batch, scores):
                if score == 1:  # Check for integer value
                    filtered_qa.append((q, a))
                    
        except Exception as e:
            print(f"Error processing batch: {e}")
    
    return filtered_qa

# Test scenario
if __name__ == "__main__":
    # Test query about global warming
    query = "Влияние глобального потепления на океаны"
    
    # Sample QA pairs - some relevant, some not
    qa_pairs = [
        ("Как глобальное потепление влияет на уровень мирового океана?", 
         "Уровень мирового океана повысился на 20 см с 1880 года и продолжает расти на 3,3 мм в год."),
        
        ("Сколько видов рыб обитает в Тихом океане?", 
         "В Тихом океане обитает более 20000 видов рыб."),
        
        ("Как изменилась температура океанов в последние десятилетия?", 
         "За последние 30 лет температура верхних слоев океана повысилась в среднем на 0.8°C."),
        
        ("Кто был первым человеком, переплывшим Атлантический океан?", 
         "Первым человеком, переплывшим Атлантический океан в одиночку, был Ален Бомбар в 1952 году."),
        
        ("Какое влияние оказывает потепление на коралловые рифы?", 
         "Повышение температуры воды приводит к обесцвечиванию кораллов и массовой гибели коралловых рифов."),
    ]
    
    print(f"Testing sanity check with query: '{query}'")
    print(f"Testing with {len(qa_pairs)} QA pairs")
    
    # Run the sanity check
    filtered_results = sanity_check(query, qa_pairs)
    
    # Display results
    print("\n===== RESULTS =====")
    print(f"Original QA pairs: {len(qa_pairs)}")
    print(f"Filtered QA pairs: {len(filtered_results)}")
    print("\nRelevant QA pairs:")
    for i, (q, a) in enumerate(filtered_results, 1):
        print(f"{i}. Q: {q}")
        print(f"   A: {a}")
        print()