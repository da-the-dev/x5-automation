from langfuse.llama_index import LlamaIndexInstrumentor

from src.config import config
from src.workflow import AssistantFlow


# Initialize the Langfuse instrumentor
instrumentor = LlamaIndexInstrumentor(
    public_key=config["public_key"],
    secret_key=config["secret_key"],
    host=config["host"],
)


# Example of how to use the workflow with Langfuse tracing
async def run_workflow_with_tracing(
    query: str, session_id: str = None, user_id: str = None
):
    try:
        # Start the instrumentation
        instrumentor.start()
    
        # Or use the context manager for more control over tracing parameters
        with instrumentor.observe(
            trace_id=f"assistant-flow-{query[:10]}",  # Optional custom trace ID
            session_id=session_id,
            user_id=user_id,
            metadata={
                "query": query,
                "llm": config["llm"],
            },
        ) as trace:
                # Run your workflow
                workflow = AssistantFlow(timeout=3 * 60)
                result = await workflow.run(query=query)

    except Exception as e:
        print(f"Error: {e}")
        result = "An error occurred"

    finally:
        # Make sure to flush before the application exits
        instrumentor.flush()

    return result
