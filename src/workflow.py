# Third-party libraries
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)

# Local imports - event types
from src.workflow_events import (
    PreprocessEvent,
    RetrieveEvent,
    DeduplicateEvent,
    SanityCheckEvent,
    HasQAExamplesEvent,
)

# Local imports - workflow steps
from src.workflow_steps.preprocess import preprocess_step
from src.workflow_steps.retrieve import retrieve_step
from src.workflow_steps.deduplicate import deduplicate_step
from src.workflow_steps.sanity_check import sanity_check_step
from src.workflow_steps.qa_examples import is_there_qa_examples_step
from src.workflow_steps.reply import reply_step


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
    ) -> HasQAExamplesEvent | StopEvent:
        return await is_there_qa_examples_step(ev)

    @step
    async def reply(self, ev: HasQAExamplesEvent, ctx: Context) -> StopEvent:
        return await reply_step(ev, ctx)
