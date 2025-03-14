from src.workflow_events import (
    SanityCheckEvent,
    HasQAExamplesEvent,
    GalaOtmenaEvent
)

async def is_there_qa_examples_step(
    ev: SanityCheckEvent
) -> HasQAExamplesEvent | GalaOtmenaEvent:
    # Check if there are any QA examples
    qa = ev.qa
    # If there are QA examples, continue
    if len(qa) == 0:
        return GalaOtmenaEvent(qa=qa)
    # Else return GalaOtmena
    else:
        return HasQAExamplesEvent(qa=qa)
