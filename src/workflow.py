from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
from langfuse.llama_index import LlamaIndexInstrumentor # https://langfuse.com/docs/integrations/llama-index/get-started


class PreprocessEvent(Event):
    query_clean: str


class RetrieveEvent(Event):
    qa: list[tuple[str, str]]


class DeduplicateEvent(Event):
    qa: list[tuple[str, str]]


class SanityCheckEvent(Event):
    qa: list[tuple[str, str]]


# Initialize the Langfuse instrumentor
instrumentor = LlamaIndexInstrumentor()


class AssistantFlow(Workflow):
    # Addressing LLM via VLLM (for reference)
    #
    # from llama_index.llms.openai_like import OpenAILike
    # llm = OpenAILike(
    #     api_base="http://localhost:8000/v1",
    #     api_key="token-123",
    #     model="Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct",
    # )
    # llm.complete("what is an atom")

    @step
    async def preprocess(self, ev: StartEvent) -> PreprocessEvent:
        query = ev.query

        # TODO preprocess

        return PreprocessEvent(query_clean="query_clean")

    @step
    async def retrieve(self, ev: PreprocessEvent, ctx: Context) -> RetrieveEvent:
        query_clean = ev.query_clean
        await ctx.set("query_clean", query_clean)  # Saving clean query for use later

        # TODO retrieve

        return RetrieveEvent(qa=[("q", "a")])

    @step
    async def deduplicate(self, ev: RetrieveEvent, ctx: Context) -> DeduplicateEvent:
        qa = ev.qa

        # TODO deduplicate

        return DeduplicateEvent(qa=[("q", "a")])

    @step
    async def sanity_check(
        self, ev: DeduplicateEvent, ctx: Context
    ) -> SanityCheckEvent:
        qa = ev.qa
        query_clean = await ctx.get("query_clean")

        # TODO sanity check

        return SanityCheckEvent(qa=[("q", "a")])

    @step
    async def reply(self, ev: SanityCheckEvent, ctx: Context) -> StopEvent:
        qa = ev.qa
        query_clean = await ctx.get("query_clean")

        # TODO reply

        return StopEvent(result="result generator")


# Example of how to use the workflow with Langfuse tracing
async def run_workflow_with_tracing(query: str, session_id: str = None, user_id: str = None):
    # Start the instrumentation
    instrumentor.start()
    
    # Or use the context manager for more control over tracing parameters
    with instrumentor.observe(
        trace_id=f"assistant-flow-{query[:10]}",  # Optional custom trace ID
        session_id=session_id,
        user_id=user_id,
        name="AssistantFlow",
        metadata={"original_query": query}
    ) as trace:
        # Run your workflow
        workflow = AssistantFlow()
        result = await workflow.run(query=query)
        
        # Optionally add a score or update the trace
        trace.score(name="workflow_completed", value=1.0)
        
    # Make sure to flush before the application exits
    instrumentor.flush()
    
    return result
