"""LangGraph consultation workflow definition with intention understanding."""

from typing import Literal
from langgraph.graph import StateGraph, END

from ..agents.state import AgentState, TEN_INQUIRY_DIMENSIONS
from ..agents.intention_agent import intention_node, IntentionType
from ..agents.inquiry_node import inquiry_node
from ..agents.reflection_node import reflection_node
from ..agents.generator_node import generator_node
from ..agents.rag_retriever import RAGRetriever


def create_consultation_graph(retriever: RAGRetriever) -> StateGraph:
    """
    Create the consultation workflow graph.

    Flow:
    1. intention → Understand user intent
    2. inquiry → Collect symptoms (if needed)
    3. reflection → Assess completeness
    4. generate → Generate final output
    (with potential optimization loop)

    Args:
        retriever: RAG retriever for knowledge base access

    Returns:
        Compiled StateGraph
    """
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("intention", intention_node)
    graph.add_node("inquiry", lambda state: inquiry_node(state, retriever))
    graph.add_node("reflection", reflection_node)
    graph.add_node("generate", lambda state: generator_node(state, retriever))

    # Set entry point
    graph.set_entry_point("intention")

    # Edge: intention → inquiry (for most cases)
    graph.add_edge("intention", "inquiry")

    # Edge: inquiry → reflection
    graph.add_edge("inquiry", "reflection")

    # Conditional routing after reflection
    def route_after_reflection(state: AgentState) -> Literal["inquiry", "generate"]:
        """Route based on reflection result."""
        # Check if this is a quick/specific consultation that should skip additional inquiry
        user_intention = state.get("user_intention", IntentionType.FIRST_VISIT)

        if user_intention in [IntentionType.QUICK_CONSULT, IntentionType.SPECIFIC_SYMPTOM]:
            # For quick/specific consult, limit inquiry rounds
            if state.get("reflection_count", 0) >= 1:
                return "generate"

        if state.get("is_complete", False) or state.get("reflection_count", 0) >= 3:
            return "generate"
        return "inquiry"

    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "inquiry": "inquiry",
            "generate": "generate",
        }
    )

    # End after generation
    graph.add_edge("generate", END)

    return graph


def compile_consultation_graph(retriever: RAGRetriever) -> StateGraph:
    """
    Compile and return the consultation graph.

    Args:
        retriever: RAG retriever instance

    Returns:
        Compiled graph ready for execution
    """
    graph = create_consultation_graph(retriever)
    return graph.compile()


def run_consultation(
    retriever: RAGRetriever,
    thread_id: str,
    initial_message: str,
    user_name: str = "匿名",
) -> AgentState:
    """
    Run a complete consultation.

    Args:
        retriever: RAG retriever
        thread_id: Session identifier
        initial_message: First user message
        user_name: Optional patient name

    Returns:
        Final agent state
    """
    from langchain_core.messages import HumanMessage

    compiled_graph = compile_consultation_graph(retriever)

    # Extract potential first symptom from initial message
    symptoms_list = _extract_symptoms_from_message(initial_message)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=initial_message)],
        "symptoms_list": symptoms_list,
        "is_complete": False,
        "reflection_count": 0,
        "thread_id": thread_id,
        "current_phase": "intention",
        "user_intention": IntentionType.OTHER,
        "intention_summary": "",
        "intention_info": {},
        "retrieved_context": "",
        "memory_context": "",
        "memory_agent": {},
        "rule_context": {},
        "validation_context": {},
    }

    result = compiled_graph.invoke(initial_state)
    return result


def _extract_symptoms_from_message(message: str, existing_symptoms: list[dict] = None) -> list[dict]:
    """
    Simple symptom extraction from user message.

    This is a basic implementation. In production, use NER or LLM.
    """
    symptoms = []
    message_lower = message.lower()

    # Negation keywords
    negation_keywords = ["没有", "无", "不曾", "不会", "不有", "否认", "不是", "没啥", "不清楚"]
    has_negation = any(neg in message_lower for neg in negation_keywords)

    # Simple keyword matching
    symptom_keywords = {
        "寒热": ["怕冷", "畏寒", "发热", "怕热", "寒热往来"],
        "汗": ["出汗", "盗汗", "自汗", "无汗"],
        "头身": ["头痛", "头晕", "腰痛", "身痛", "肩背痛"],
        "便溏": ["腹泻", "便秘", "大便溏", "大便稀"],
        "饮食": ["食欲不振", "食欲不佳", "厌食", "暴饮暴食"],
        "胸腹": ["胸闷", "胸痛", "腹胀", "腹痛", "胃痛"],
        "耳目": ["耳鸣", "听力下降", "视力模糊", "眼花"],
        "口渴": ["口干", "口渴", "喜冷饮", "喜热饮"],
        "睡眠": ["失眠", "多梦", "嗜睡", "难入睡", "易醒"],
    }

    # Get already collected dimensions to avoid duplicates
    existing_symptoms = existing_symptoms or []
    existing_dims = {s.get("dimension") for s in existing_symptoms if s.get("dimension")}

    # First pass: extract any positive symptoms mentioned
    for dimension, keywords in symptom_keywords.items():
        if dimension in existing_dims:
            continue
        for keyword in keywords:
            if keyword in message_lower:
                # Check if it's negated (e.g., "没有头痛")
                if has_negation:
                    # User explicitly denied this symptom - mark as "无"
                    symptoms.append({"dimension": dimension, "value": "无"})
                else:
                    symptoms.append({"dimension": dimension, "value": keyword})
                break

    # If no symptoms extracted and user gave a negation response,
    # mark one uncollected dimension as "未提及" to allow progression
    if not symptoms and has_negation:
        from ..agents.state import TEN_INQUIRY_DIMENSIONS
        uncollected = [d for d in TEN_INQUIRY_DIMENSIONS if d not in existing_dims]
        if uncollected:
            # Mark the first uncollected dimension as "未提及"
            symptoms.append({"dimension": uncollected[0], "value": "未提及"})

    return symptoms
