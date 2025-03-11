from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
import requests
import json



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
        '''
        We need to specify endpoints for:
        - embedder model
        - qdrant database
        '''
        query_clean = ev.query_clean
        await ctx.set("query_clean", query_clean)  # Saving clean query for use later

        # TODO retrieve
        # mock values
        VLLM_HOST = "http://localhost:8000"
        embedder_endpoint = f"{VLLM_HOST}/v1/embeddings"

        headers = {
            "Content-Type": "application/json"
        }

        # Define the data payload
        data = {
            "model": "elderberry17/USER-bge-m3-x5",
            "input": [query_clean]
        }

        response = requests.post(embedder_endpoint, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            query_embedding = response.json().get('data', [])[0].get('embedding', [])
            print(f"embedding len: {len(query_embedding)}")
        else:
            print(response.status_code)
    
        # TODO search similar queries from qdrant

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