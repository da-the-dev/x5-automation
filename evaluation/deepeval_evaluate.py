from dotenv import find_dotenv, load_dotenv
from langfuse import Langfuse
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric
from datetime import datetime
from os import getenv
import os

 
if not os.getenv("PROD"):
   load_dotenv(find_dotenv())

 
os.environ['OPENAI_API_KEY'] = ""


def fetch_traces(langfuse, batch_size, from_timestamp):
    traces_batch = langfuse.fetch_traces(
        limit=batch_size,
        from_timestamp=from_timestamp,
    ).data
    return traces_batch

def process_context(qa_pairs):
    if len(qa_pairs) == 0:
        return None
    
    context = []
    for pair in qa_pairs:
        context.append(f"Вопрос: {pair[0]}\nОтвет:{pair[1]}\n")

    return context


def process_trace(langfuse, trace):
    trace_data = {}
    trace_data['output'] = trace.output
    
    for observation_id in trace.observations:
        if 'preprocess' in observation_id:
            observation = langfuse.fetch_observation(observation_id).data
            trace_data['query_clean'] = observation.output['query_clean']
        elif 'reply' in observation_id:
            observation = langfuse.fetch_observation(observation_id).data
            qa_pairs = observation.input.get('ev', {}).get('qa')
            context = process_context(qa_pairs)
            trace_data['context'] = context

    return trace_data


def evaluate_trace(langfuse, trace):
    trace_data = process_trace(langfuse, trace)
    query = trace_data['query_clean']
    answer = trace_data['output']
    context = trace_data['context']
    
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context
    )
    
    contextual_relevancy_metric = ContextualRelevancyMetric(model="gpt-4o")
    
    # TODO: улучшить промпт (?)
    evaluate(
        test_cases=[test_case],
        metrics=[contextual_relevancy_metric]
    )
    
    score = contextual_relevancy_metric.score
    return score

def write_score(langfuse, trace_id, score):
    langfuse.score(
        trace_id=trace_id,
        name="contextual_relevancy",
        value=score
    )
    

if __name__ == "__main__":
    langfuse = Langfuse(
        public_key=getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=getenv("LANGFUSE_SECRET_KEY"),
        host=getenv("LANGFUSE_HOST"),
    )

    now = datetime.now()
    from_timestamp = datetime(now.year, now.month, now.day, 8, 0)
    batch_size = 10

    traces_batch = fetch_traces(langfuse, batch_size, from_timestamp)

    for trace in traces_batch:
        score = evaluate_trace(langfuse, trace)
        write_score(langfuse, trace_id=trace.id, score=score) 
