"""RAG symptom retrieval agent."""

from .intention_agent import IntentionType
from .rag_retriever import RAGRetriever


def build_symptom_query(state: dict, task: str = "inquiry") -> str:
    """Build a retrieval query from symptoms, intent, memory, and task."""
    symptoms_list = state.get("symptoms_list", [])
    user_intention = state.get("user_intention", "")
    intention_info = state.get("intention_info", {})

    task_terms = {
        "inquiry": "问诊 追问 十问歌 症状收集",
        "generation": "辨证论治 诊断 治则",
        "validation": "辨证校验 证型 鉴别",
    }.get(task, task)

    intent_terms = {
        IntentionType.SPECIFIC_SYMPTOM: "单一症状辨证",
        IntentionType.QUICK_CONSULT: "快速问诊 关键症状",
        IntentionType.GET_PRESCRIPTION: "处方 治法 用药前问诊",
        IntentionType.FOLLOW_UP: "复诊 用药反馈",
        IntentionType.CLARIFY_DOUBT: "解释 辨证依据",
    }.get(user_intention, "初诊 中医")

    symptom_terms = []
    for item in symptoms_list[-6:]:
        dim = item.get("dimension", "")
        value = item.get("value", "")
        if dim and value and value not in ["无", "未提及", "待确认", "待进一步确认"]:
            symptom_terms.append(f"{dim} {value}")

    mentioned_terms = intention_info.get("mentioned_symptoms", [])[:5]
    query_parts = ["中医", task_terms, intent_terms, *symptom_terms, *mentioned_terms]
    return " ".join(part for part in query_parts if part).strip()


def retrieve_symptom_context(state: dict, retriever: RAGRetriever, task: str = "inquiry", limit: int = 4) -> tuple[str, str]:
    """Retrieve and format context for a symptom-oriented task."""
    query = build_symptom_query(state, task)
    docs = retriever.retrieve(query)
    return query, retriever.format_retrieved_docs(docs[:limit])
