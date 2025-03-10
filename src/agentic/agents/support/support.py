from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledGraph

from ...llm import llm
from ...memory import memory
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
    checkpointer=memory,
    prompt=(
"""
Ты агент клиентской поддержки. 

Твоя задача отвечать пользователю используя ТОЛЬКО доступную тебе информацию, основываясь на релевантных примерах QA (вопросы и ответы).

Для получения релевантных примеров QA используй соответствующий tool.

Если ты не можешь ответить на запрос пользователя или пользователь об этом просит - позови на помощь человека, используя соответствующий tool.
"""
    )
)
