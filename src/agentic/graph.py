from langgraph.graph.state import CompiledGraph

from .agents import support_agent


# For now we can set compiled support_agent as whole graph
graph: CompiledGraph = support_agent
