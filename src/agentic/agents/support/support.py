from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledGraph

from ...llm import llm
from .tools import (
    find_relevant_qa_examples_tool,
    сall_human_support_tool
)


support_agent: CompiledGraph = create_react_agent(
    model=llm,
    tools=[
        find_relevant_qa_examples_tool,
        сall_human_support_tool
    ],
    prompt=(
"""
Ты агент клиентской поддержки. 

Твоя задача отвечать пользователю используя ТОЛЬКО доступную тебе информацию, основываясь на релевантных примерах QA (вопросы и ответы).

Для получения релевантных примеров QA используй соответствующий tool. Если пользователь не предоставил достаточно полезной информации для поиска релевантных примеров QA - ты можешь задавать пользователю уточняющие вопросы.

Если ты не можешь ответить на запрос пользователя или пользователь об этом просит - переключи чат на человека, используя соответствующий tool.

Будь вежлив, корректно отрабатывай недовольство пользователя.
"""
    )
)


print("Compiled support agent subgraph with this structure:")
try:
    print(support_agent.get_graph().draw_ascii())
except Exception as e:
    print(e)
    pass
