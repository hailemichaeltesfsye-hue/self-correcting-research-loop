from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import ResearchState
from .nodes import (
    research_node,
    judge_node,
    refiner_node,
    human_review_node,
    publish_node,
)


def route_after_judge(state: ResearchState) -> str:
    """
    Conditional routing function evaluating Judge score and iteration safety limits.
    """
    score = state.get("score", 0.0)
    threshold = state.get("quality_threshold", 8.0)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if score >= threshold:
        # High quality: proceed directly to human review gate
        return "human_review"
    elif iteration_count < max_iterations:
        # Below threshold and iterations remaining: loop back to refiner
        return "refiner"
    else:
        # Max iteration safety cutoff triggered: route to human review for manual decision (prevents infinite loops)
        return "human_review"


def route_after_human(state: ResearchState) -> str:
    """
    Conditional routing function following Human-in-the-Loop decision.
    """
    approved = state.get("approved", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if approved:
        return "publish"
    elif iteration_count < max_iterations:
        return "refiner"
    else:
        # If rejected after max iterations, terminate graph safely
        return END


def save_graph_visualization(graph, output_path: str = "graph.png") -> str:
    """
    Renders the graph as a Mermaid PNG and writes it to disk.

    Args:
        graph: The compiled LangGraph graph object.
        output_path: File path to save the PNG to (default: "graph.png").

    Returns:
        The path the image was saved to.
    """
    png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print(f"[✓] Graph visualization saved to: {output_path}")
    return output_path


def build_research_graph(checkpointer=None):
    """
    Assembles and compiles the self-correcting research StateGraph.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("research", research_node)
    builder.add_node("judge", judge_node)
    builder.add_node("refiner", refiner_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("publish", publish_node)

    # Define graph connectivity
    builder.add_edge(START, "research")
    builder.add_edge("research", "judge")

    # Conditional routing after Judge evaluation
    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "human_review": "human_review",
            "refiner": "refiner",
        },
    )

    # Return loop edge from refiner back to judge
    builder.add_edge("refiner", "judge")

    # Conditional routing after Human Review
    builder.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "publish": "publish",
            "refiner": "refiner",
            END: END,
        },
    )

    builder.add_edge("publish", END)

    # Compile graph with MemorySaver checkpointer to enable HITL & Time Travel
    graph = builder.compile(checkpointer=checkpointer)
    return graph