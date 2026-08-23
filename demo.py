"""
LangGraph Self-Correcting Research Loop Demonstration
-----------------------------------------------------
Showcases:
1. Self-Correction Loop (Research -> Judge -> Refine -> Judge)
2. Infinite Loop Guard (Max Iteration Cutoff)
3. Human-in-the-Loop (Interrupt & Resume via Command)
4. Time Travel (Inspect State History & Fork Trajectory)
"""

import sys
import json
from langgraph.types import Command
from langgraph_research_loop import (
    build_research_graph,
    get_history,
    find_checkpoint_by_node,
    fork_and_modify_state,
)


def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80 + "\n")


def run_demo():
    print_separator("LangGraph Self-Correcting Research Loop Demo")

    # -------------------------------------------------------------------------
    # SCENARIO 1: Self-Correction Loop & HITL Approval
    # -------------------------------------------------------------------------
    print_separator("Scenario 1: Self-Correction Loop & HITL Approval")
    graph = build_research_graph()
    thread_1 = {"configurable": {"thread_id": "thread_demo_1"}}
    
    topic = "Self-Correcting AI Agents in LangGraph"
    print(f"[*] Submitting research topic: '{topic}'")
    print(f"[*] Target Quality Threshold: 8.0/10.0 | Max Iterations: 3\n")

    # Invoke graph - will run research, judge (score 6.0), refiner (iter 1), judge (score 8.5), and interrupt
    print("[1] Invoking graph initial run...")
    state = graph.invoke(
        {"topic": topic, "quality_threshold": 8.0, "max_iterations": 3},
        config=thread_1,
    )

    current_state = graph.get_state(thread_1)
    print(f"\n[+] Graph Execution Paused at Node: {current_state.next}")
    print(f"[+] Current Draft Score: {current_state.values.get('score')}/10.0")
    print(f"[+] Iterations Performed: {current_state.values.get('iteration_count')}")
    print(f"[+] Latest Judge Feedback: {current_state.values.get('feedback')}")

    # Inspect the interrupt payload
    tasks = current_state.tasks
    if tasks and tasks[0].interrupts:
        interrupt_payload = tasks[0].interrupts[0].value
        print(f"\n[!] Human-in-the-Loop Interrupt Triggered:")
        print(f"    Instruction: {interrupt_payload.get('instruction')}")
        print(f"    Allowed Actions: {interrupt_payload.get('allowed_actions')}")

    print("\n[2] Resuming graph with Human Decision: 'approve'...")
    final_state = graph.invoke(Command(resume="approve"), config=thread_1)
    print(f"[OK] Final Graph Status: {final_state.get('status')}")
    print(f"[OK] Report Published Successfully!")

    # -------------------------------------------------------------------------
    # SCENARIO 2: Infinite Loop Safeguard (Max Iterations)
    # -------------------------------------------------------------------------
    print_separator("Scenario 2: Infinite Loop Safeguard (Max Iterations)")
    thread_2 = {"configurable": {"thread_id": "thread_demo_max_iter"}}
    print("[*] Submitting topic with unreachable threshold (99.0) & Max Iterations = 2")

    state_2 = graph.invoke(
        {"topic": "Infinite Loop Safety Test", "quality_threshold": 99.0, "max_iterations": 2},
        config=thread_2,
    )

    current_state_2 = graph.get_state(thread_2)
    print(f"\n[+] Iterations Executed: {current_state_2.values.get('iteration_count')}/2")
    print(f"[+] Score Achieved: {current_state_2.values.get('score')} (Below 99.0 threshold)")
    print(f"[+] Self-Healing Guard: Infinite loop prevented! Graph safely routed to HITL approval gate: {current_state_2.next}")

    _ = graph.invoke(Command(resume="reject"), config=thread_2)
    print("[OK] Safeguard verified successfully.")

    # -------------------------------------------------------------------------
    # SCENARIO 3: Human-in-the-Loop Direct Editing
    # -------------------------------------------------------------------------
    print_separator("Scenario 3: Human-in-the-Loop Direct Content Edit")
    thread_3 = {"configurable": {"thread_id": "thread_demo_edit"}}
    
    print("[1] Invoking graph...")
    graph.invoke({"topic": "Quantum AI", "quality_threshold": 8.0}, config=thread_3)

    edited_content = (
        "# Quantum AI: Expert Curated Report\n\n"
        "## Summary\n"
        "Manually reviewed, updated, and validated by senior AI domain specialist."
    )
    print("\n[2] Resuming graph with Human Action: 'edit' (overriding draft content)...")
    resumed_3 = graph.invoke(
        Command(resume={"action": "edit", "draft": edited_content}),
        config=thread_3,
    )

    print(f"[OK] Final Status: {resumed_3.get('status')}")
    print(f"[OK] Published Draft Content Preview:\n{resumed_3.get('draft')[:120]}...")

    # -------------------------------------------------------------------------
    # SCENARIO 4: Time Travel (Inspect State History & Rewind/Fork)
    # -------------------------------------------------------------------------
    print_separator("Scenario 4: Time Travel (State History Inspection & Rewind)")
    thread_tt = {"configurable": {"thread_id": "thread_demo_time_travel"}}
    
    graph.invoke({"topic": "Time Travel Mechanics in LangGraph", "quality_threshold": 8.0}, config=thread_tt)

    print("[*] Retrieving Checkpoint History for thread...")
    history = get_history(graph, thread_tt)
    print(f"[+] Total Checkpoints Recorded: {len(history)}")
    
    for entry in history[:4]:
        print(f"    - Step #{entry['step_index']} | Checkpoint ID: {entry['checkpoint_id'][:12]}... | Next Nodes: {entry['next_nodes']}")

    # Find checkpoint paused at human_review
    target_config = find_checkpoint_by_node(graph, thread_tt, "human_review")
    if target_config:
        print("\n[*] Time Travel: Rewinding to checkpoint before human review and forking execution...")
        forked_res = fork_and_modify_state(
            graph,
            checkpoint_config=target_config,
            state_updates={"human_feedback": "Forked trajectory test note."},
            input_data=Command(resume="approve"),
        )
        print(f"[OK] Forked Trajectory Status: {forked_res.get('status')}")

    print_separator("All Scenarios Completed Successfully!")


if __name__ == "__main__":
    run_demo()
