from llama_index.utils.workflow import draw_most_recent_execution

from src.workflow_with_tracing import run_workflow_with_tracing


async def main():
    print("You:")

    query = input()

    result = await run_workflow_with_tracing(query)

    print("Result:")
    print(result)

    # Optionally draw the workflow
    # draw_most_recent_execution(w, filename="assistant_flow_recent.html")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
