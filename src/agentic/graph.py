import logging

from langgraph.graph import MessagesState, StateGraph, START
from langgraph.graph.state import CompiledGraph

from .agents import support_agent
from .memory import memory


# Create a state graph for the agents
graph_builder = StateGraph(MessagesState)

# Add manager_agent to the graph
graph_builder.add_node("support", support_agent)
# Add entry point to the graph
graph_builder.add_edge(START, "support")

# Compile the graph with a checkpointer
graph: CompiledGraph = graph_builder.compile(checkpointer=memory)
