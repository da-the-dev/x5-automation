from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


class PreprocessEvent(Event):
    query_clean: str


class RetrieveEvent(Event):
    qa: list[tuple[str, str]]


class SanityCheckEvent(Event):
    sane_qa: list[tuple[str, str]]




class AssistantFlow(Workflow):
    # Addressing LLM via VLLM (for reference)
    #
    # from llama_index.llms.openai_like import OpenAILike
    # llm = OpenAILike(
    #     api_base="http://localhost:8000/v1",
    #     api_key="token-123",
    #     model="Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct",
    # )

    @step
    async def preprocess(self, ev: StartEvent) -> PreprocessEvent:
        query = ev.query

        # TODO preprocess

        return PreprocessEvent(query_clean="query_clean")

    @step
    async def retrieve(self, ev: PreprocessEvent) -> RetrieveEvent:
        query_clean = ev.query_clean

        # TODO retrieve

        return RetrieveEvent(qa=[("q", "a")])

    @step
    async def retrieve(self, ev: PreprocessEvent) -> RetrieveEvent:
        query_clean = ev.query_clean

        # TODO retrieve

        return RetrieveEvent(qa=[("q", "a")])

    @step
    async def sanity_check(self, ev: RetrieveEvent) -> SanityCheckEvent:
        qa = ev.qa

        # TODO sanity check

        return SanityCheckEvent(sane_qa=[("q", "a")])
    
    @step
    async def reply(self, ev: SanityCheckEvent) -> StopEvent:
        sane_qa = ev.sane_qa
        
        # TODO reply
        
        return StopEvent(result="result generator")

