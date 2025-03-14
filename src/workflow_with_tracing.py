from langfuse.llama_index import LlamaIndexInstrumentor

from src.settings import settings
from src.workflow import AssistantFlow


instrumentor = LlamaIndexInstrumentor(
    public_key=settings.langfuse.PUBLIC_KEY,
    secret_key=settings.langfuse.SECRET_KEY,
    host=settings.langfuse.HOST,
)


async def run_workflow_with_tracing(
    query: str, session_id: str = None, user_id: str = None
):
    try:
        instrumentor.start()

        # Use the context manager for tracing parameters
        with instrumentor.observe(
            trace_id=f"assistant-flow-{query[:10]}",
            session_id=session_id,
            user_id=user_id,
            metadata={
                "query": query,
                "llm": settings.llm.MODEL,
            },
        ):
            workflow = AssistantFlow(timeout=3 * 60)
            return await workflow.run(query=query)
    finally:
        # Make sure to flush before the application exits
        instrumentor.flush()
