"""
LangGraph Self-Correcting Research Loop Interactive CLI
"""

import os
import sys
import uuid
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_research_loop import (
    build_research_graph,
    get_history,
    find_checkpoint_by_node,
    fork_and_modify_state,
    save_graph_visualization,
)

load_dotenv()


def get_configured_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def print_banner():
    model_name = get_configured_model()
    print(f"Using Groq model: {model_name}\n")
    print("-" * 50)
    print("\nLangGraph Self-Correcting Research Loop CLI")
    print("1. Start a new research session")
    print("2. Resume an existing session")
    print("3. Inspect checkpoint history of a session")
    print("4. Time Travel (Rewind or Fork from history)")
    print("5. Inspect the graph structure and node information")
    print("6. Save graph visualization as PNG")
    print("7. Exit\n")


def start_new_session(graph):
    print("\n--- Start New Research Session ---")
    topic = input("Enter research topic: ").strip()
    if not topic:
        topic = "Self-Correcting AI Agents in LangGraph"

    threshold_str = input("Enter target quality threshold (default: 8.0): ").strip()
    quality_threshold = float(threshold_str) if threshold_str else 8.0

    max_iter_str = input("Enter max iterations limit (default: 3): ").strip()
    max_iterations = int(max_iter_str) if max_iter_str else 3

    thread_id = f"session-{str(uuid.uuid4())[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n[*] Starting session: {thread_id}")
    print(f"[*] Topic: '{topic}' | Target Threshold: {quality_threshold} | Max Iterations: {max_iterations}\n")

    state = graph.invoke(
        {
            "topic": topic,
            "quality_threshold": quality_threshold,
            "max_iterations": max_iterations,
        },
        config=config,
    )

    handle_graph_state(graph, config)


def handle_graph_state(graph, config: Dict[str, Any]):
    thread_id = config["configurable"]["thread_id"]
    current_state = graph.get_state(config)

    while current_state.next:
        print(f"\n[+] Execution paused at node: {current_state.next}")
        print(f"[+] Current Draft Score: {current_state.values.get('score', 0.0)}/10.0")
        print(f"[+] Iterations Completed: {current_state.values.get('iteration_count', 0)}")
        if current_state.values.get("feedback"):
            print(f"[+] Judge Feedback: {current_state.values.get('feedback')}")

        tasks = current_state.tasks
        if tasks and hasattr(tasks[0], "interrupts") and tasks[0].interrupts:
            interrupt_payload = tasks[0].interrupts[0].value
            print("\n" + "!" * 50)
            print("HUMAN-IN-THE-LOOP INTERRUPT DETECTED")
            print(f"Instruction: {interrupt_payload.get('instruction')}")
            print(f"\nCurrent Draft (full):\n{'-' * 50}")
            print(current_state.values.get("draft", ""))
            print("-" * 50)
            print("!" * 50)

            print("\nApproval Options:")
            print("1. Approve report for publication")
            print("2. Reject report with feedback")
            print("3. Edit draft directly")
            choice = input("Select decision (1-3): ").strip()

            if choice == "1":
                resume_cmd = Command(resume="approve")
            elif choice == "2":
                fb = input("Enter rejection feedback: ").strip()
                resume_cmd = Command(resume={"action": "reject", "feedback": fb})
            elif choice == "3":
                print("Enter new draft text (end with single line 'END'):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                new_draft = "\n".join(lines)
                resume_cmd = Command(resume={"action": "edit", "draft": new_draft})
            else:
                print("Invalid choice, approving by default.")
                resume_cmd = Command(resume="approve")

            state = graph.invoke(resume_cmd, config=config)
            current_state = graph.get_state(config)
        else:
            break

    final_values = graph.get_state(config).values
    status = final_values.get("status")
    print(f"\n[✓] Session {thread_id} completed!")
    print(f"[✓] Final Status: {status}")

    final_report = final_values.get("final_report")
    if final_report:
        print(f"\n[✓] Final Report (full):\n{'=' * 50}")
        print(final_report)
        print("=" * 50)

        report_path = f"report_{thread_id}.md"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"\n[✓] Full report also saved to: {report_path}")
        except Exception as e:
            print(f"[!] Could not save report to file: {e}")


def resume_session(graph):
    print("\n--- Resume Existing Session ---")
    thread_id = input("Enter Thread ID to resume: ").strip()
    if not thread_id:
        print("Thread ID cannot be empty.")
        return

    config = {"configurable": {"thread_id": thread_id}}
    state_info = graph.get_state(config)

    if not state_info.values:
        print(f"No existing session found for Thread ID: {thread_id}")
        return

    print(f"[*] Found session {thread_id}. Next nodes: {state_info.next}")
    handle_graph_state(graph, config)


def inspect_checkpoint_history(graph):
    print("\n--- Inspect Checkpoint History ---")
    thread_id = input("Enter Thread ID to inspect: ").strip()
    if not thread_id:
        return

    config = {"configurable": {"thread_id": thread_id}}
    history = get_history(graph, config)

    if not history:
        print(f"No execution history found for Thread ID: {thread_id}")
        return

    print(f"\nCheckpoint History for {thread_id} ({len(history)} checkpoints):")
    print("-" * 75)
    for entry in history:
        print(f"Step #{entry['step_index']} | Checkpoint ID: {entry['checkpoint_id']}")
        print(f"  Next Nodes: {entry['next_nodes']}")
        print(f"  Score: {entry['values'].get('score')} | Iterations: {entry['values'].get('iteration_count')}")
        if entry['tasks']:
            print(f"  Pending Tasks/Interrupts: {entry['tasks']}")
        print("-" * 75)


def time_travel_session(graph):
    print("\n--- Time Travel (Rewind or Fork) ---")
    thread_id = input("Enter Thread ID: ").strip()
    if not thread_id:
        return

    config = {"configurable": {"thread_id": thread_id}}
    history = get_history(graph, config)

    if not history:
        print(f"No history found for Thread ID: {thread_id}")
        return

    print("\nAvailable Checkpoints:")
    for idx, entry in enumerate(history):
        print(f"[{idx}] Step #{entry['step_index']} | Checkpoint ID: {entry['checkpoint_id']} | Next: {entry['next_nodes']}")

    cp_choice = input("\nSelect checkpoint index to rewind to: ").strip()
    if not cp_choice.isdigit() or int(cp_choice) >= len(history):
        print("Invalid checkpoint index.")
        return

    selected_cp = history[int(cp_choice)]
    cp_config = selected_cp["config"]

    print("\nTime Travel Actions:")
    print("1. Rewind and execute as-is")
    print("2. Fork state with modified values")
    act_choice = input("Select action (1-2): ").strip()

    if act_choice == "2":
        new_score = input("Enter new score (leave empty to keep current): ").strip()
        new_feedback = input("Enter new feedback (leave empty to keep current): ").strip()
        updates = {}
        if new_score:
            updates["score"] = float(new_score)
        if new_feedback:
            updates["feedback"] = new_feedback

        print("[*] Forking state and resuming execution...")
        forked_res = fork_and_modify_state(
            graph,
            checkpoint_config=cp_config,
            state_updates=updates,
            input_data=Command(resume="approve") if selected_cp["next_nodes"] and "human_review" in selected_cp["next_nodes"] else None,
        )
        print(f"[✓] Forked Execution Completed! Final Status: {forked_res.get('status')}")
    else:
        print("[*] Rewinding to checkpoint...")
        resumed = graph.invoke(
            Command(resume="approve") if selected_cp["next_nodes"] and "human_review" in selected_cp["next_nodes"] else None,
            config=cp_config,
        )
        print(f"[✓] Rewound Execution Completed! Final Status: {resumed.get('status') if isinstance(resumed, dict) else 'done'}")


def inspect_graph_structure(graph):
    print("\n" + "-" * 50)
    try:
        ascii_graph = graph.get_graph().draw_ascii()
        print(ascii_graph)
    except Exception as e:
        print(f"Could not render ASCII graph: {e}")

    print("\n")
    try:
        nodes = graph.get_graph().nodes
        for node_id, node in nodes.items():
            print(f"Node: {node_id}")
            node_type = type(node.data) if hasattr(node, "data") else type(node)
            print(f"Node Type: {node_type}\n")
    except Exception as e:
        print(f"Could not list node details: {e}")


def save_graph_image(graph):
    print("\n--- Save Graph Visualization ---")
    path = input("Enter output file path (default: graph.png): ").strip()
    if not path:
        path = "graph.png"
    try:
        save_graph_visualization(graph, path)
    except Exception as e:
        print(f"[!] Failed to save graph image: {e}")


def run_cli(graph):
    """
    Runs the main interactive menu loop. Separated from main() so the
    SqliteSaver connection (opened in main()) stays open for the
    entire duration of the CLI session.
    """
    while True:
        print_banner()
        choice = input("Select an option (1-7): ").strip()

        if choice == "1":
            start_new_session(graph)
        elif choice == "2":
            resume_session(graph)
        elif choice == "3":
            inspect_checkpoint_history(graph)
        elif choice == "4":
            time_travel_session(graph)
        elif choice == "5":
            inspect_graph_structure(graph)
        elif choice == "6":
            save_graph_image(graph)
        elif choice == "7":
            print("\nExiting CLI. Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please select 1-7.\n")


def main():
    # SqliteSaver persists checkpoints to disk (checkpoints.db) so sessions
    # can be resumed even after the program is closed and reopened,
    # unlike MemorySaver which only keeps state in RAM for the current run.
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = build_research_graph(checkpointer=checkpointer)
        run_cli(graph)


if __name__ == "__main__":
    main()