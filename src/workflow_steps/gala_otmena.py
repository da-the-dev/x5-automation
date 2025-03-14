from llama_index.core.workflow import StopEvent
from src.workflow_events import GalaOtmenaEvent

async def gala_otmena_step(ev: GalaOtmenaEvent) -> StopEvent:
    return StopEvent(
        result="К сожалению, у меня недостаточно информации, чтобы ответить на ваш запрос. Переключаю на оператора..."
    )
