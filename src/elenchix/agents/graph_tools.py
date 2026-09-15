"""Optional, case-scoped graph context for tutoring and assessment."""

from elenchix.graph.base import GraphStore
from elenchix.llm import FunctionTool


def case_graph_tool(graph: GraphStore, case_id: str) -> FunctionTool:
    def retrieve() -> dict:
        return graph.case_subgraph(case_id)

    return FunctionTool(
        name="get_case_subgraph",
        description=(
            "Optionally retrieve the current case's knowledge points, ability anchors, "
            "and immediate neighbours with directed, weighted relations. Call only when "
            "graph relationships are needed. Neighbours provide context and do not expand "
            "the eligible assessment targets or count as learner evidence."
        ),
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=retrieve,
    )
