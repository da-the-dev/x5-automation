from src.workflow_events import (
    SanityCheckEvent,
    HasQAExamplesEvent,
)

from llama_index.core.workflow import StopEvent

async def is_there_qa_examples_step(
    ev: SanityCheckEvent
) -> HasQAExamplesEvent | StopEvent:
    # Check if there are any QA examples
    qa = ev.qa
    # If there are QA examples, continue
    if len(qa) == 0:
        return StopEvent("К сожалению, у меня недостаточно информации, чтобы ответить на ваш запрос. Переключаю на оператора...")
    # Else return GalaOtmena
    else:
        return HasQAExamplesEvent(qa=qa)
