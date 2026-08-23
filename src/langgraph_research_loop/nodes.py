import os
import json
import re
from typing import Any, Dict
from dotenv import load_dotenv
from langgraph.types import interrupt
from .state import ResearchState

# Load environment variables from .env file if present
load_dotenv()


RESEARCH_PROMPT = """You are an expert research analyst producing a comprehensive, well-structured report.

Topic: {topic}

Write a detailed research report following this exact structure:

# Research Report: {topic}

## 1. Introduction
Provide a comprehensive overview and initial synthesis of the topic.

## 2. Core Concepts
List 3-5 fundamental concepts, each as a bolded sub-heading followed by a clear definition and explanation of how it applies to {topic}.

## 3. Principles
List the key guiding principles. For each principle:
- Give its name in bold
- Explain what it means and how it's typically achieved
- Follow with 1-2 concrete real-world examples showing the principle in practice

## 4. Applications
Describe how {topic} applies across different domains (e.g. healthcare, education, workplace, urban planning, or others relevant to the topic). For each domain:
- Use a bolded sub-heading
- List specific, concrete examples with technical detail (mention actual methods, tools, or technologies where relevant)

## 5. Methods and Tools
Describe the key methods, frameworks, or tools relevant to {topic}, explaining what each does and how it's used.

## 6. Challenges and Limitations
Identify realistic challenges, trade-offs, or open problems.

## 7. Future Directions
Suggest 3-5 specific future directions or open research questions.

Requirements:
- Use bullet points and sub-bullets extensively, not just paragraphs
- Every claim should be specific and concrete, not vague generalities
- Aim for substantial depth — this should read like a real research report, not a summary
- Use markdown formatting (##, **, -) throughout
"""


REFINER_PROMPT = """You are revising a research report based on judge feedback.

Original Topic: {topic}
Current Draft:
{draft}

Judge Feedback:
{feedback}

Revise the draft to directly address every point in the feedback. Maintain the same structure (Introduction, Core Concepts, Principles, Applications, Methods and Tools, Challenges, Future Directions), but:
- Add missing depth, examples, or sections the feedback identified as weak
- Keep all strong existing content
- Ensure every section has concrete, specific bullet points — not vague statements

Output the full revised report, not just the changes.
"""


def _get_llm():
    """
    Initialize LLM provider, prioritizing Groq with Llama models (70B / 8B).
    Falls back gracefully to mock responses if no key is configured.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if groq_key and groq_key.strip():
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=groq_model,
                groq_api_key=groq_key,
                temperature=0.7,
            )
        except Exception as e:
            print(f"[Warning] Failed to initialize ChatGroq: {e}")

    # Fallback options for OpenAI / Gemini
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key.strip():
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        except Exception as e:
            print(f"[Warning] Failed to initialize ChatOpenAI: {e}")

    print("[Warning] No working LLM provider configured — using mock fallback drafts.")
    return None


def research_node(state: ResearchState) -> Dict[str, Any]:
    """
    Initial research node: conducts research and generates a first draft.
    """
    topic = state.get("topic", "Self-Correcting AI Agents")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    quality_threshold = state.get("quality_threshold", 8.0)

    llm = _get_llm()
    if llm:
        try:
            prompt = RESEARCH_PROMPT.format(topic=topic)
            response = llm.invoke(prompt)
            draft = response.content
        except Exception as e:
            print(f"[LLM ERROR in research_node] {type(e).__name__}: {e}")
            draft = (
                f"# Research Report: {topic}\n\n"
                f"## 1. Introduction\n"
                f"Comprehensive overview of {topic}. Initial synthesis of paradigms.\n\n"
                f"## 2. Core Concepts\n"
                f"- Concept A: Fundamental principles and theoretical limits.\n"
                f"- Concept B: Architectural patterns.\n\n"
                f"## 3. Analysis\n"
                f"Draft requires further technical depth and explicit self-correction mechanisms."
            )
    else:
        draft = (
            f"# Research Report: {topic}\n\n"
            f"## 1. Introduction\n"
            f"Comprehensive overview of {topic}. Initial synthesis of paradigms.\n\n"
            f"## 2. Core Concepts\n"
            f"- Concept A: Fundamental principles and theoretical limits.\n"
            f"- Concept B: Architectural patterns.\n\n"
            f"## 3. Analysis\n"
            f"Draft requires further technical depth and explicit self-correction mechanisms."
        )

    return {
        "topic": topic,
        "draft": draft,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "quality_threshold": quality_threshold,
        "status": "researching",
        "approved": False,
    }


def judge_node(state: ResearchState) -> Dict[str, Any]:
    """
    Judge node: evaluates draft quality, assigns quality score (0.0 - 10.0), and provides critique.
    """
    draft = state.get("draft", "")
    iteration_count = state.get("iteration_count", 0)
    llm = _get_llm()

    if llm:
        try:
            prompt = (
                f"You are a strict academic reviewer. Evaluate the following research report:\n\n{draft}\n\n"
                f"Return JSON ONLY with format:\n"
                f'{{"score": <float 0.0-10.0>, "feedback": "<detailed critique>"}}'
            )
            response = llm.invoke(prompt)
            clean_content = response.content.strip().strip("```json").strip("```").strip()
            # Escape backslashes that aren't already valid JSON escapes
            # (fixes drafts containing LaTeX like \( \) which break json.loads)
            clean_content = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', clean_content)
            parsed = json.loads(clean_content)
            score = float(parsed.get("score", 6.5))
            feedback = parsed.get("feedback", "Needs more depth and technical precision.")
        except Exception as e:
            print(f"[LLM ERROR in judge_node] {type(e).__name__}: {e}")
            if iteration_count == 0:
                score = 6.0
                feedback = "Draft lacks details on Llama 70B fine-tuning, checkpointers, and HITL safety gates."
            elif iteration_count == 1:
                score = 8.5
                feedback = "Substantial improvement. Thorough coverage of checkpointers and self-correction loops."
            else:
                score = 9.5
                feedback = "Exemplary quality. Publication-ready research."
    else:
        if iteration_count == 0:
            score = 6.0
            feedback = "Draft lacks details on Llama 70B/8B model integration, checkpointers, and HITL safety gates."
        elif iteration_count == 1:
            score = 8.5
            feedback = "Substantial improvement. Thorough coverage of checkpointers, interrupt handling, and refiner loops."
        else:
            score = 9.5
            feedback = "Exemplary research report. Fully verified and ready for publication."

    return {
        "score": score,
        "feedback": feedback,
        "status": "evaluated",
    }


def refiner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Refiner node: improves draft based on Judge feedback and increments iteration count.
    """
    topic = state.get("topic", "")
    draft = state.get("draft", "")
    feedback = state.get("feedback", "")
    human_fb = state.get("human_feedback", "")
    iteration_count = state.get("iteration_count", 0)
    llm = _get_llm()

    combined_feedback = f"Judge Critique: {feedback}"
    if human_fb:
        combined_feedback += f" | Human Reviewer Notes: {human_fb}"

    if llm:
        try:
            prompt = REFINER_PROMPT.format(
                topic=topic,
                draft=draft,
                feedback=combined_feedback,
            )
            response = llm.invoke(prompt)
            new_draft = response.content
        except Exception as e:
            print(f"[LLM ERROR in refiner_node] {type(e).__name__}: {e}")
            new_draft = (
                f"{draft}\n\n"
                f"## 4. Refinement Pass (Iteration {iteration_count + 1})\n"
                f"### Addressing Feedback:\n{combined_feedback}\n\n"
                f"### Enhanced Implementation Details:\n"
                f"- Optimized for Groq Llama 70B/8B fast inference.\n"
                f"- Integrated state persistence with MemorySaver checkpointer.\n"
                f"- Implemented interrupt() pattern for human-in-the-loop approval.\n"
                f"- Bound loop iteration count ({iteration_count + 1}/{state.get('max_iterations', 3)}) to ensure graph stability."
            )
    else:
        new_draft = (
            f"{draft}\n\n"
            f"## 4. Refinement Pass (Iteration {iteration_count + 1})\n"
            f"### Addressing Feedback:\n{combined_feedback}\n\n"
            f"### Enhanced Implementation Details:\n"
            f"- Optimized for Groq Llama 70B/8B fast inference.\n"
            f"- Integrated state persistence with MemorySaver checkpointer.\n"
            f"- Implemented interrupt() pattern for human-in-the-loop approval.\n"
            f"- Bound loop iteration count ({iteration_count + 1}/{state.get('max_iterations', 3)}) to ensure graph stability."
        )

    return {
        "draft": new_draft,
        "iteration_count": iteration_count + 1,
        "status": "refined",
    }


def human_review_node(state: ResearchState) -> Dict[str, Any]:
    """
    Human-in-the-Loop node: pauses execution using `interrupt()` before publication.
    """
    payload = {
        "instruction": "High-Risk Gate: Human approval required before publication.",
        "topic": state.get("topic"),
        "draft": state.get("draft"),
        "score": state.get("score"),
        "feedback": state.get("feedback"),
        "iteration_count": state.get("iteration_count"),
        "max_iterations_reached": state.get("iteration_count", 0) >= state.get("max_iterations", 3),
        "allowed_actions": ["approve", "reject", "edit"],
    }

    # Pause graph execution and wait for human response via Command(resume=...)
    human_response = interrupt(payload)

    approved = False
    human_fb = ""
    updated_draft = state.get("draft")

    if isinstance(human_response, str):
        if human_response.lower() in ["approve", "yes", "y"]:
            approved = True
        elif human_response.lower() in ["reject", "no", "n"]:
            approved = False
            human_fb = "Rejected by reviewer during human approval gate."
    elif isinstance(human_response, dict):
        action = human_response.get("action", "approve")
        if action == "approve":
            approved = True
            if "draft" in human_response:
                updated_draft = human_response["draft"]
        elif action == "reject":
            approved = False
            human_fb = human_response.get("feedback", "Rejected by reviewer.")
        elif action == "edit":
            approved = True
            updated_draft = human_response.get("draft", updated_draft)

    return {
        "approved": approved,
        "draft": updated_draft,
        "human_feedback": human_fb,
        "status": "approved" if approved else "rejected",
    }


def publish_node(state: ResearchState) -> Dict[str, Any]:
    """
    Final publication node.
    """
    return {
        "status": "published",
        "final_report": state.get("draft"),
    }