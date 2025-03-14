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


class IsThereQAExamplesEvent(Event):
    qa: list[tuple[str, str]]


class HasQAExamplesEvent(Event):
    qa: list[tuple[str, str]]


class AssistantFlow(Workflow):
    @step
    async def preprocess(self, ev: StartEvent) -> PreprocessEvent | StopEvent:
        query = ev.query

        from src.preprocess import preprocess

        query_clean = preprocess(query)

        if len(query_clean) <= 0:
            return StopEvent(
                "К сожалению, у меня недостаточно информации, чтобы ответить на ваш запрос. Переключаю на оператора..."
            )

        return PreprocessEvent(query_clean=query_clean)

    @step
    async def retrieve(self, ev: PreprocessEvent, ctx: Context) -> RetrieveEvent:
        query_clean = ev.query_clean
        await ctx.set("query_clean", query_clean)  # Saving clean query for use later

        from src.retriever import retriever

        qa = await retriever(query_clean)

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
    async def is_there_qa_examples(
        self, ev: SanityCheckEvent
    ) -> HasQAExamplesEvent | StopEvent:
        qa = ev.qa
        if len(qa) == 0:
            return StopEvent(
                "К сожалению, у меня недостаточно информации, чтобы ответить на ваш запрос. Переключаю на оператора..."
            )
        else:
            return HasQAExamplesEvent(qa=qa)

    @step
    async def reply(self, ev: HasQAExamplesEvent, ctx: Context) -> StopEvent:
        qa = ev.qa
        query_clean = await ctx.get("query_clean")

        from src.reply import reply

        result = await reply(query_clean, qa)

        return StopEvent(result)
