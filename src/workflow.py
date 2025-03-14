from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)

from src.workflow_steps.preprocess import preprocess_step
from src.workflow_steps.retrieve import retrieve_step
from src.workflow_steps.deduplicate import deduplicate_step
from src.workflow_steps.sanity_check import sanity_check_step
from src.workflow_steps.qa_examples import is_there_qa_examples_step
from src.workflow_steps.reply import reply_step
from src.workflow_steps.gala_otmena import gala_otmena_step

# Re-export Event types for backward compatibility
from src.workflow_events import (
    PreprocessEvent,
    RetrieveEvent,
    DeduplicateEvent,
    SanityCheckEvent,
    IsThereQAExamplesEvent,
    HasQAExamplesEvent,
    GalaOtmenaEvent,
)

class AssistantFlow(Workflow):
    @step
    async def preprocess(self, ev: StartEvent) -> PreprocessEvent:
        return await preprocess_step(ev)

    @step
    async def retrieve(self, ev: PreprocessEvent, ctx: Context) -> RetrieveEvent:
        return await retrieve_step(ev, ctx)

    @step
    async def deduplicate(self, ev: RetrieveEvent) -> DeduplicateEvent:
        return await deduplicate_step(ev)

    @step
    async def sanity_check(
        self, ev: DeduplicateEvent, ctx: Context
    ) -> SanityCheckEvent:
        return await sanity_check_step(ev, ctx)

    @step
    async def is_there_qa_examples(
        self, ev: SanityCheckEvent
    ) -> HasQAExamplesEvent | GalaOtmenaEvent:
        return await is_there_qa_examples_step(ev)

    @step
    async def reply(self, ev: HasQAExamplesEvent, ctx: Context) -> StopEvent:
        return await reply_step(ev, ctx)

    @step
    async def gala_otmena(self, ev: GalaOtmenaEvent) -> StopEvent:
        return await gala_otmena_step(ev)
