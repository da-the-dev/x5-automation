from src.config import config


async def reply(query_clean: str, qa: list[tuple[str, str]]) -> str:
    from llama_index.llms.openai_like import OpenAILike

    llm = OpenAILike(
        api_base=config["api_base"],
        api_key=config["api_key"],
        model=config["llm"],
        max_tokens=512,
        temperature=0.5,
    )

    query = (
        "Запрос пользователя:\n"
        f"{query_clean}\n"
        "Вот примеры вопрос и ответов:\n"
        f"{"\n".join([q + "->" + a for q, a in qa])}\n"
        "Дай ответ на запрос пользователя в том же стиле, как в примерах"
    )

    return await llm.acomplete(query)
