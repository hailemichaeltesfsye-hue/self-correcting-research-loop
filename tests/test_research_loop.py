import pytest
from langgraph.types import Command
from langgraph_research_loop.graph import build_research_graph
from langgraph_research_loop.time_travel import get_history, find_checkpoint_by_node, fork_and_modify_state


def test_self_correction_and_approval_flow():
    graph = build_research_graph()
    config = {"configurable": {"thread_id": "test_thread_1"}}

    # Initial invocation - expected to run research -> judge -> refiner -> judge -> HITL interrupt
    initial_state = graph.invoke(
        {
            "topic": "LangGraph Checkpointing",
            "quality_threshold": 8.0,
            "max_iterations": 3,
        },
        config=config,
    )

    # Graph should pause at human_review interrupt
    state_info = graph.get_state(config)
    assert len(state_info.next) > 0
    assert "human_review" in state_info.next
    assert state_info.values["score"] >= 8.0
    assert state_info.values["iteration_count"] >= 1

    # Resume graph execution with approval
    final_output = graph.invoke(Command(resume="approve"), config=config)

    assert final_output["status"] == "published"
    assert final_output["approved"] is True
    assert "final_report" in final_output


def test_max_iteration_safety_cutoff():
    graph = build_research_graph()
    config = {"configurable": {"thread_id": "test_thread_max_iter"}}

    # Force threshold higher than maximum achievable score to trigger loop boundary
    _ = graph.invoke(
        {
            "topic": "Quantum Computing",
            "quality_threshold": 99.0,  # Unreachable threshold
            "max_iterations": 2,
        },
        config=config,
    )

    # Graph must pause at human_review instead of looping indefinitely
    state_info = graph.get_state(config)
    assert "human_review" in state_info.next
    assert state_info.values["iteration_count"] == 2

    # Reject at human review
    final_output = graph.invoke(Command(resume="reject"), config=config)
    assert final_output["approved"] is False


def test_hitl_direct_edit():
    graph = build_research_graph()
    config = {"configurable": {"thread_id": "test_thread_edit"}}

    # Initial run hits interrupt
    graph.invoke({"topic": "Agentic Workflows", "quality_threshold": 7.0}, config=config)

    # Resume with direct user edit
    edited_draft = "# Manually Reviewed & Edited Report on Agentic Workflows\nVerified content."
    final_output = graph.invoke(
        Command(resume={"action": "edit", "draft": edited_draft}),
        config=config,
    )

    assert final_output["status"] == "published"
    assert final_output["draft"] == edited_draft


def test_time_travel_inspection_and_forking():
    graph = build_research_graph()
    config = {"configurable": {"thread_id": "test_thread_time_travel"}}

    # Run graph until interrupt
    graph.invoke({"topic": "Time Travel in Graphs", "quality_threshold": 8.0}, config=config)

    # Inspect execution history
    history = get_history(graph, config)
    assert len(history) > 1

    # Find historical checkpoint before human_review (e.g. at judge or refiner node)
    judge_checkpoint = find_checkpoint_by_node(graph, config, "human_review")
    assert judge_checkpoint is not None

    # Fork graph from that checkpoint with modified topic/threshold
    forked_output = fork_and_modify_state(
        graph,
        checkpoint_config=judge_checkpoint,
        state_updates={"quality_threshold": 10.0},  # Change requirement on forked branch
        input_data=Command(resume="approve"),
    )

    assert forked_output is not None
