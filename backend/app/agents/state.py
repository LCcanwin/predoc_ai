"""Agent state definition for LangGraph consultation workflow."""

from typing import TypedDict, Literal, Union
from langchain_core.messages import HumanMessage, AIMessage


class AgentState(TypedDict):
    """State for the consultation agent."""

    messages: list[Union[HumanMessage, AIMessage]]
    """Conversation history."""

    symptoms_list: list[dict]
    """Collected symptoms in format {"dimension": "value"}."""

    is_complete: bool
    """Whether information gathering is complete."""

    reflection_count: int
    """Number of reflection cycles (max 3)."""

    thread_id: str
    """Session identifier."""

    current_phase: Literal["intention", "options", "confirm", "inquiry", "reflection", "generation", "complete"]
    """Current workflow phase."""

    # Intention understanding fields
    user_intention: str
    """Parsed user intention type."""

    intention_summary: str
    """Summary of user intention."""

    intention_info: dict
    """Additional information about user intention."""

    # RAG retrieval context
    retrieved_context: str
    """Retrieved knowledge base context for current turn."""

    memory_context: str
    """Short-term user memory context for current consultation."""

    memory_agent: dict
    """Structured memory and follow-up signals."""

    rule_context: dict
    """Rule-engine diagnosis hints."""

    validation_context: dict
    """Diagnosis validation output."""

    # Selected options from ten inquiry
    selected_dimensions: list[str]
    """User selected dimensions from ten inquiry options."""

    # Confirmation questions
    confirm_questions: list[str]
    """Two confirmation questions generated after options selection."""


# Ten inquiry dimensions (十问歌)
TEN_INQUIRY_DIMENSIONS = [
    "寒热",      # Cold/heat sensation
    "汗",        # Sweating
    "头身",      # Head/body
    "便溏",      # Bowel movements
    "饮食",      # Diet/appetite
    "胸腹",      # Chest/abdomen
    "耳目",      # Eyes/ears
    "口渴",      # Thirst
    "睡眠",      # Sleep
    "舌脉",      # Tongue/pulse
]

DIMENSION_DESCRIPTIONS = {
    "寒热": "畏寒/发热/寒热往来",
    "汗": "有汗/无汗/盗汗/自汗",
    "头身": "头痛/头晕/身痛/腰痛",
    "便溏": "大便形状、次数",
    "饮食": "食欲、口味偏好",
    "胸腹": "胸闷/腹胀/腹痛",
    "耳目": "视力/听力变化",
    "口渴": "口干/渴喜冷饮",
    "睡眠": "失眠/多梦/嗜睡",
    "舌脉": "舌象/脉象",
}
