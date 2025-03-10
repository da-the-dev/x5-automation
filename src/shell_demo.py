from agentic.graph import graph


def main() -> None:
    global graph

    while True:
        # tred = input("Thread ID: ")
        tred = "1"
        request = input("You: ")

        if request == "exit":
            break
        inputs = {"messages": [("user", request)]}

        for event in graph.stream(inputs, {"configurable": {"thread_id": tred}}, stream_mode="values"):
            message = event["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
                print("\n")


if __name__ == "__main__":
    main()
