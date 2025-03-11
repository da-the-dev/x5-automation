from src.workflow import AssistantFlow


async def main():
    print("You:")

    query = input()

    w = AssistantFlow(timeout=60, verbose=False)
    result = await w.run(query=query)

    print("Result:")
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
