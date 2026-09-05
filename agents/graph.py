from langgraph.graph import StateGraph, END
from agents.nodes import (
    AgentState,
    run_exception_explainer,
    run_adversarial_auditor,
    run_assembler
)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("explainer", run_exception_explainer)
    graph.add_node("auditor",   run_adversarial_auditor)
    graph.add_node("assembler", run_assembler)

    graph.set_entry_point("explainer")
    graph.add_edge("explainer", "auditor")
    graph.add_edge("auditor",   "assembler")
    graph.add_edge("assembler", END)

    return graph.compile()


app = build_graph()