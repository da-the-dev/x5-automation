# Third-party libraries
from llama_index.core.workflow import Event

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

class GalaOtmenaEvent(Event):
    qa: list[tuple[str, str]]
