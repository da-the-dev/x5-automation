from src.workflow_events import DeduplicateEvent, RetrieveEvent

async def deduplicate_step(ev: RetrieveEvent) -> DeduplicateEvent:
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
