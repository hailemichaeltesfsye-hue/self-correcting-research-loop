# Agent Guidelines for LangGraph Self-Correcting Research Loop

## Workspace Overview

This repository contains a modular Python implementation of a **Self-Correcting Research Loop** using **LangGraph**.
It features autonomous research drafting, Judge evaluation & scoring, iterative refinement loops, maximum iteration safeguards, MemorySaver checkpointing, Human-in-the-Loop (HITL) `interrupt()`, and Time-Travel state history inspection/forking.

## Development & Dependency Rules

- **Package Management**: Always use `uv` for dependency management (`uv add <package>`, `uv run <command>`).
- **Python Package Layout**: Code is organized inside `src/langgraph_research_loop/`. Keep modules focused and decoupled:
  - `state.py`: `TypedDict` graph state definitions.
  - `nodes.py`: Autonomous nodes (research, judge, refiner, human_review, publish).
  - `graph.py`: `StateGraph` construction, conditional routing (`route_after_judge`, `route_after_human`), and compilation.
  - `time_travel.py`: State history inspection (`get_history`), rewinding, and state forking (`fork_and_modify_state`).
- **LLM Integration**:
  - Primary provider: **Groq** (`ChatGroq`) using Llama 70B (`llama-3.3-70b-versatile`) or 8B (`llama-3.1-8b-instant`) models configured via `.env` (`GROQ_API_KEY`, `GROQ_MODEL`).
  - Always maintain deterministic synthetic fallback behavior when `GROQ_API_KEY` is not present, ensuring tests run offline out-of-the-box.
- **LangGraph Best Practices**:
  - Use modern LangGraph 0.2+ functional `interrupt()` and `Command(resume=...)`.
  - Always bound iterative loops using `iteration_count` and `max_iterations` to prevent infinite loops.
  - Always attach a `checkpointer` (e.g. `MemorySaver`) when compiling graph builders to maintain state persistence across interrupts and enable time travel.

## Verification Instructions

- Run unit tests: `uv run pytest`
- Run demonstration CLI: `uv run python demo.py`

## commit messages

-commits should be in conventional commits format.
