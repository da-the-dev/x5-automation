from langchain_openai import ChatOpenAI

# from .settings import agents_settings


# For now, we are using OpenAI's ChatGPT model, but in the future, we need to make model selection more configurable.
llm: ChatOpenAI = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-123",
    model="Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"
)
