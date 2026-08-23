from typing import TypedDict, Optional


class ResearchState(TypedDict, total=False):
    """
    Graph state for the self-correcting research loop.
    """
    topic: str
    draft: str
    score: float
    quality_threshold: float
    feedback: str
    iteration_count: int
    max_iterations: int
    status: str
    approved: bool
    human_feedback: Optional[str]
    final_report: Optional[str]
