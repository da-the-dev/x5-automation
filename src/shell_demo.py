import asyncio

from agentic.graph import graph


async def main() -> None:
    global graph

    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        pass

    while True:
        # tred = input("Thread ID: ")
        tred = "1"
        request = input("You: ")

        if request == "exit":
            break
        inputs = {"messages": [("user", request)]}

        async for event in graph.astream(inputs, {"configurable": {"thread_id": tred}}, stream_mode="values"):
            message = event["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
                print("\n")


if __name__ == "__main__":
    asyncio.run(main())