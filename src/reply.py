async def reply(query_clean: str, qa: list[tuple[str, str]]) -> str:
    from llama_index.llms.openai_like import OpenAILike

    llm = OpenAILike(
        api_base="http://localhost:8000/v1",
        api_key="token-123",
        model="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24",
    )
    return await llm.acomplete(
        (
            "Запрос пользователя следующий:\n",
            f"{query_clean}\n",
            "Вот примеры вопрос и ответов:\n",
            f"{"\n".join([q + "->" + a for q, a in qa])}\n",
            "Дай ответ на первоначальный запрос"
        )
    )
