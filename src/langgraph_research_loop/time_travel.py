from typing import List, Dict, Any, Optional
from langgraph.types import Command


def get_history(graph, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieves and summarizes the complete execution state history for a thread.
    """
    history = []
    for idx, state in enumerate(graph.get_state_history(config)):
        checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id", "initial")
        history.append({
            "step_index": idx,
            "checkpoint_id": checkpoint_id,
            "next_nodes": state.next,
            "values": state.values,
            "config": state.config,
            "tasks": [
                {
                    "name": t.name,
                    "interrupts": [i.value for i in t.interrupts] if hasattr(t, "interrupts") and t.interrupts else [],
                }
                for t in state.tasks
            ] if hasattr(state, "tasks") and state.tasks else [],
        })
    return history


def find_checkpoint_by_node(graph, config: Dict[str, Any], target_node: str) -> Optional[Dict[str, Any]]:
    """
    Finds the most recent state checkpoint where `next` includes target_node.
    """
    for state in graph.get_state_history(config):
        if target_node in state.next:
            return state.config
    return None


def rewind_to_step(graph, checkpoint_config: Dict[str, Any], input_data: Any = None):
    """
    Replays graph execution starting from a designated historical checkpoint configuration.
    """
    return graph.invoke(input_data, config=checkpoint_config)


def fork_and_modify_state(
    graph,
    checkpoint_config: Dict[str, Any],
    state_updates: Dict[str, Any],
    input_data: Any = None,
):
    """
    Forks graph execution by updating state variables at a target historical checkpoint
    and resuming execution down the new branch trajectory.
    """
    # Create forked checkpoint config with updated state values
    forked_config = graph.update_state(checkpoint_config, state_updates)
    
    # Resume graph execution from forked state
    return graph.invoke(input_data, config=forked_config)
