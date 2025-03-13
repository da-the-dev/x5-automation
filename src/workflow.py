from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
from langfuse.llama_index import LlamaIndexInstrumentor


from src.config import config


class PreprocessEvent(Event):
    query_clean: str


class RetrieveEvent(Event):
    qa: list[tuple[str, str]]


class DeduplicateEvent(Event):
    qa: list[tuple[str, str]]


class SanityCheckEvent(Event):
    qa: list[tuple[str, str]]


# Initialize the Langfuse instrumentor
instrumentor = LlamaIndexInstrumentor(
    public_key=config["public_key"],
    secret_key=config["secret_key"],
    host=config["host"],
)


class AssistantFlow(Workflow):
    @step
    async def preprocess(self, ev: StartEvent) -> PreprocessEvent:
        query = ev.query

        from src.preprocess import preprocess

        query_clean = preprocess()

        return PreprocessEvent(query_clean=query_clean)

    @step
    async def retrieve(self, ev: PreprocessEvent, ctx: Context) -> RetrieveEvent:
        query_clean = ev.query_clean
        await ctx.set("query_clean", query_clean)  # Saving clean query for use later

        from src.retriever import retriever

        qa = retriever(query_clean)

        return RetrieveEvent(qa=qa)

    @step
    async def deduplicate(self, ev: RetrieveEvent) -> DeduplicateEvent:
        qa = ev.qa
        unique_answers = set()
        unique_qa_pairs = []
        for pair in qa:
            question, answer = pair[0], pair[1]
            if answer in unique_answers:
                continue
            else:
                unique_answers.add(answer)
                unique_qa_pairs.append(tuple([question, answer]))

        return DeduplicateEvent(qa=unique_qa_pairs)

    @step
    async def sanity_check(
        self, ev: DeduplicateEvent, ctx: Context
    ) -> SanityCheckEvent:
        qa = ev.qa
        query_clean = await ctx.get("query_clean")

        from src.sanity_check import sanity_check

        sane_qa = await sanity_check(query_clean, qa)

        return SanityCheckEvent(qa=sane_qa)

    @step
    async def reply(self, ev: SanityCheckEvent, ctx: Context) -> StopEvent:
        qa = ev.qa
        query_clean = await ctx.get("query_clean")

        from src.reply import reply

        result = await reply(query_clean, qa)

        return StopEvent(result=result)


# Example of how to use the workflow with Langfuse tracing
async def run_workflow_with_tracing(
    query: str, session_id: str = None, user_id: str = None
):
    # Start the instrumentation
    instrumentor.start()

    # Or use the context manager for more control over tracing parameters
    with instrumentor.observe(
        trace_id=f"assistant-flow-{query[:10]}",  # Optional custom trace ID
        session_id=session_id,
        user_id=user_id,
        metadata={"original_query": query},
    ) as trace:
        # Run your workflow
        workflow = AssistantFlow(timeout=3 * 60)
        result = await workflow.run(query=query)

        # Optionally add a score or update the trace
        trace.score(name="workflow_completed", value=1.0)

    # Make sure to flush before the application exits
    instrumentor.flush()

    return result
