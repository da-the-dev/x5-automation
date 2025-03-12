from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)


class PreprocessEvent(Event):
    query_clean: str


class RetrieveEvent(Event):
    qa: list[tuple[str, str]]


class DeduplicateEvent(Event):
    qa: list[tuple[str, str]]


class SanityCheckEvent(Event):
    qa: list[tuple[str, str]]


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
        query_clean = await ctx.get("query_clean")

        # TODO deduplicate

        return DeduplicateEvent(qa=[("q", "a")])

    @step
    async def sanity_check(
        self, ev: DeduplicateEvent, ctx: Context
    ) -> SanityCheckEvent:
        qa = ev.qa
        query_clean = await ctx.get("query_clean")

        from sanity_check import sanity_check

        sane_qa = sanity_check(query_clean, qa)

        return SanityCheckEvent(qa=sane_qa)

    @step
    async def reply(self, ev: SanityCheckEvent, ctx: Context) -> StopEvent:
        qa = ev.qa
        query_clean = await ctx.get("query_clean")

        # TODO reply

        return StopEvent(result="result generator")
