from llama_index.utils.workflow import draw_most_recent_execution

from src.workflow import AssistantFlow


async def main():
    w = AssistantFlow(timeout=60, verbose=False)

    print("You:")

    query = input()

    result = await w.run(query=query)

    print("Result:")
    print(result)

    draw_most_recent_execution(w, filename="assistant_flow_recent.html")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
