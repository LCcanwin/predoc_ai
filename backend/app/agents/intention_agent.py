"""Intention understanding agent for the consultation system."""

import json
import re
from typing import Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from .state import AgentState
from ..config import get_llm_config


# Intention types
class IntentionType:
    """User intention types."""
    FIRST_VISIT = "first_visit"           # First time consultation, needs full inquiry
    FOLLOW_UP = "follow_up"               # Follow-up visit
    SPECIFIC_SYMPTOM = "specific_symptom" # Asking about specific symptom
    QUICK_CONSULT = "quick_consult"       # Quick consultation, minimal questions
    GET_PRESCRIPTION = "get_prescription"  # Wants prescription directly
    CLARIFY_DOUBT = "clarify_doubt"       # Asking for clarification
    OTHER = "other"                        # Other intentions


INTENTION_DESCRIPTIONS = {
    IntentionType.FIRST_VISIT: "初诊问诊，需要全面收集症状信息",
    IntentionType.FOLLOW_UP: "复诊问诊，查看之前病例或反馈用药情况",
    IntentionType.SPECIFIC_SYMPTOM: "咨询特定症状或问题",
    IntentionType.QUICK_CONSULT: "快速咨询，只需回答几个关键问题",
    IntentionType.GET_PRESCRIPTION: "直接获取处方",
    IntentionType.CLARIFY_DOUBT: "对之前的诊断或处方有疑问需要澄清",
    IntentionType.OTHER: "其他意图",
}

VALID_INTENTIONS = set(INTENTION_DESCRIPTIONS.keys())


class IntentionAnalysis(BaseModel):
    """Structured result returned by the intention classifier."""

    intention: str = Field(description="One of the supported intention type values.")
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    mentioned_symptoms: list[str] = Field(default_factory=list)
    is_urgent: bool = False
    previous_case_exists: bool = False
    additional_notes: str = ""

    def normalized(self) -> tuple[str, str, dict]:
        intention = self.intention if self.intention in VALID_INTENTIONS else IntentionType.OTHER
        return intention, self.summary or INTENTION_DESCRIPTIONS.get(intention, "需要进一步了解"), {
            "mentioned_symptoms": self.mentioned_symptoms,
            "is_urgent": self.is_urgent,
            "previous_case_exists": self.previous_case_exists,
            "additional_notes": self.additional_notes,
            "confidence": self.confidence,
        }


def create_intention_llm():
    """Create LLM for intention understanding."""
    from langchain_openai import ChatOpenAI
    config = get_llm_config()
    return ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=0.0,
    )


def parse_user_intention(user_message: str, conversation_history: list) -> tuple[str, str, dict]:
    """
    Parse user intention from their message.

    Args:
        user_message: Current user input
        conversation_history: Previous conversation messages

    Returns:
        Tuple of (intention_type, summary, extracted_info)
    """
    # Build context from history
    history_context = ""
    for msg in conversation_history[-6:]:  # Last 6 messages
        if hasattr(msg, "content") and msg.content:
            role = "用户" if hasattr(msg, "type") and msg.type == "human" else "助手"
            history_context += f"{role}：{msg.content}\n"

    # Strong rules catch explicit product-flow intents. Ambiguous symptom text still
    # goes through the classifier; defaulting to first_visit too early causes
    # routing errors.
    strong_result = _strong_intention_detect(user_message)
    if strong_result is not None:
        return strong_result

    # Use LLM for complex cases
    try:
        llm = create_intention_llm()
        prompt = f"""你是一个中医问诊系统的意图理解专家。你的任务是根据用户的输入，准确判断用户的意图。

【最近对话】（可用于判断复诊、澄清、改需求等上下文；不要编造）
{history_context if history_context else "无"}

【当前用户输入】（必须重点依据这个输入）
"{user_message}"

【重要约束】
1. 只能根据用户明确表达和最近对话判断，不要推测或编造任何信息
2. 如果只是描述多个症状、希望系统问诊，通常是 first_visit
3. 如果用户明确说“只问/只想咨询/就这个症状”，才判为 specific_symptom
4. 如果用户明确要求少问、快点、直接说重点，判为 quick_consult
5. 如果用户明确要求开药、处方、吃什么药，判为 get_prescription，不要误判为 follow_up
6. extracted 字段中的内容必须来自用户输入或最近对话

【意图类型说明】
1. first_visit: 初诊问诊，用户想要进行完整的中医问诊，收集症状信息
2. follow_up: 复诊问诊，用户之前已经问诊过，现在是复诊
3. specific_symptom: 用户只想咨询某个特定症状，不是完整的问诊
4. quick_consult: 用户想要快速咨询，不想回答太多问题
5. get_prescription: 用户直接要求开处方
6. clarify_doubt: 用户对之前的诊断或处方有疑问需要澄清
7. other: 其他意图

请只返回JSON，不要Markdown，不要解释。格式如下：
{{
    "intention": "意图类型",
    "confidence": 0.0,
    "summary": "一句话总结用户意图",
    "mentioned_symptoms": ["从用户输入中直接提取的症状列表"],
    "is_urgent": false,
    "previous_case_exists": false,
    "additional_notes": "从用户输入或最近对话中直接提取的其他信息"
}}"""

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        result = _parse_json_object(content)
        if result:
            analysis = IntentionAnalysis.model_validate(result)
            return analysis.normalized()

    except Exception as e:
        print(f"Intention LLM error: {e}")

    # Fallback
    return _quick_intention_detect(user_message)


def should_reclassify_intention(message: str, current_intention: Optional[str] = None) -> bool:
    """Return True when a later user message clearly changes the consultation goal."""
    msg = message.lower()
    if not msg.strip():
        return False

    switch_keywords = [
        "不用问", "别问", "少问", "快点", "快速", "直接", "就说重点", "一句话",
        "开药", "处方", "吃什么药", "给我药",
        "复诊", "上次", "之前", "继续", "吃了药", "用药后",
        "为什么", "什么意思", "解释一下", "不懂",
        "只想问", "只问", "只咨询", "就这个",
    ]
    if any(kw in msg for kw in switch_keywords):
        return True

    urgent_keywords = ["紧急", "严重", "急性", "立刻", "马上"]
    return current_intention != IntentionType.FIRST_VISIT and any(kw in msg for kw in urgent_keywords)


def _parse_json_object(content: str) -> Optional[dict]:
    """Parse the first valid JSON object from an LLM response."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _strong_intention_detect(message: str) -> Optional[tuple[str, str, dict]]:
    """Detect high-precision explicit intents before calling the LLM."""
    msg = message.lower()
    mentioned = _extract_mentioned_symptoms(message)

    urgent_keywords = ["紧急", "严重", "急性", "立刻", "马上"]
    if any(kw in msg for kw in urgent_keywords):
        return IntentionType.FIRST_VISIT, "用户有紧急情况需要处理", {
            "mentioned_symptoms": mentioned,
            "is_urgent": True,
            "previous_case_exists": False,
            "additional_notes": "需要快速收集关键信息",
            "confidence": 0.9,
        }

    prescription_keywords = ["开药", "给我药", "吃什么药", "用什么药", "开个方", "开方子"]
    if any(kw in msg for kw in prescription_keywords):
        return IntentionType.GET_PRESCRIPTION, "用户直接要求用药或处方建议", {
            "mentioned_symptoms": mentioned,
            "is_urgent": False,
            "previous_case_exists": False,
            "additional_notes": "需要收集足够信息才能给出建议",
            "confidence": 0.95,
        }

    follow_up_keywords = ["复诊", "上次", "之前看过", "之前问过", "吃了药", "用药后", "药后", "没效", "好转"]
    if any(kw in msg for kw in follow_up_keywords):
        return IntentionType.FOLLOW_UP, "用户进行复诊或反馈用药情况", {
            "mentioned_symptoms": mentioned,
            "is_urgent": False,
            "previous_case_exists": True,
            "additional_notes": "需要结合之前的病例或用药反馈",
            "confidence": 0.9,
        }

    quick_keywords = ["快点", "快速", "简单", "直接说", "直接给", "就说重点", "一句话", "少问"]
    if any(kw in msg for kw in quick_keywords):
        return IntentionType.QUICK_CONSULT, "用户想要快速咨询", {
            "mentioned_symptoms": mentioned,
            "is_urgent": False,
            "previous_case_exists": False,
            "additional_notes": "尽量少问问题，聚焦关键信息",
            "confidence": 0.88,
        }

    clarify_keywords = ["为什么", "什么意思", "不懂", "解释一下", "疑惑"]
    if any(kw in msg for kw in clarify_keywords):
        return IntentionType.CLARIFY_DOUBT, "用户对之前的内容有疑问", {
            "mentioned_symptoms": mentioned,
            "is_urgent": False,
            "previous_case_exists": True,
            "additional_notes": "需要结合之前的对话内容解释",
            "confidence": 0.82,
        }

    specific_markers = ["只想问", "只问", "只咨询", "就这个", "单独问"]
    if any(marker in msg for marker in specific_markers) and mentioned:
        return IntentionType.SPECIFIC_SYMPTOM, "用户只想咨询特定症状", {
            "mentioned_symptoms": mentioned,
            "is_urgent": False,
            "previous_case_exists": False,
            "additional_notes": "围绕用户明确提到的症状咨询",
            "confidence": 0.86,
        }

    return None


def _quick_intention_detect(message: str) -> tuple[str, str, dict]:
    """Quick keyword-based intention detection."""
    strong_result = _strong_intention_detect(message)
    if strong_result is not None:
        return strong_result

    msg = message.lower()

    # Check for quick consultation
    quick_keywords = ["快点", "快速", "简单", "直接", "就说", "一句话"]
    if any(kw in msg for kw in quick_keywords):
        return IntentionType.QUICK_CONSULT, "用户想要快速咨询", {
            "mentioned_symptoms": _extract_mentioned_symptoms(message),
            "is_urgent": False,
            "previous_case_exists": False,
            "additional_notes": "尽量少问问题，直接给出建议",
            "confidence": 0.75,
        }

    # Check for specific symptom
    specific_keywords = ["头痛", "发烧", "咳嗽", "胃疼", "失眠", "便秘", "腹泻"]
    mentioned = [kw for kw in specific_keywords if kw in msg]
    specific_question_markers = ["怎么办", "怎么回事", "如何", "咨询", "问一下", "只想问", "只问"]
    if len(mentioned) == 1 and any(marker in msg for marker in specific_question_markers) and "问诊" not in msg:
        return IntentionType.SPECIFIC_SYMPTOM, f"用户只想咨询{mentioned[0]}相关问题", {
            "mentioned_symptoms": mentioned,
            "is_urgent": False,
            "previous_case_exists": False,
            "additional_notes": "针对单一症状进行咨询",
            "confidence": 0.7,
        }

    # Check for prescription request
    prescription_keywords = ["开药", "处方", "给我药", "吃什么药"]
    if any(kw in msg for kw in prescription_keywords):
        return IntentionType.GET_PRESCRIPTION, "用户直接要求开处方", {
            "mentioned_symptoms": _extract_mentioned_symptoms(message),
            "is_urgent": False,
            "previous_case_exists": False,
            "additional_notes": "注意：需要收集足够信息才能开处方",
            "confidence": 0.8,
        }

    # Check for clarification
    clarify_keywords = ["为什么", "什么意思", "不懂", "解释", "疑惑"]
    if any(kw in msg for kw in clarify_keywords):
        return IntentionType.CLARIFY_DOUBT, "用户对之前的内容有疑问", {
            "mentioned_symptoms": [],
            "is_urgent": False,
            "previous_case_exists": True,
            "additional_notes": "需要回顾之前的对话内容",
            "confidence": 0.7,
        }

    # Check for urgency
    urgent_keywords = ["紧急", "严重", "急性", "立刻"]
    if any(kw in msg for kw in urgent_keywords):
        return IntentionType.FIRST_VISIT, "用户有紧急情况需要处理", {
            "mentioned_symptoms": _extract_mentioned_symptoms(message),
            "is_urgent": True,
            "previous_case_exists": False,
            "additional_notes": "需要快速收集关键信息",
            "confidence": 0.8,
        }

    # Default: first visit
    return IntentionType.FIRST_VISIT, "用户进行初诊问诊", {
        "mentioned_symptoms": _extract_mentioned_symptoms(message),
        "is_urgent": False,
        "previous_case_exists": False,
        "additional_notes": "",
        "confidence": 0.5,
    }


def _extract_mentioned_symptoms(message: str) -> list:
    """Extract mentioned symptoms from message."""
    symptom_keywords = {
        "寒热": ["怕冷", "畏寒", "发热", "怕热", "寒热往来", "发烧"],
        "汗": ["出汗", "盗汗", "自汗", "无汗", "多汗"],
        "头身": ["头痛", "头晕", "身痛", "腰痛", "肩背痛", "乏力"],
        "便溏": ["腹泻", "便秘", "大便溏", "大便稀", "大便干", "便血"],
        "饮食": ["食欲不振", "食欲不佳", "厌食", "暴饮暴食", "恶心", "呕吐"],
        "胸腹": ["胸闷", "胸痛", "腹胀", "腹痛", "胃痛", "嗳气"],
        "耳目": ["耳鸣", "听力下降", "视力模糊", "眼花", "头晕目眩"],
        "口渴": ["口干", "口渴", "喜冷饮", "喜热饮", "口苦"],
        "睡眠": ["失眠", "多梦", "嗜睡", "难入睡", "易醒", "早醒"],
        "舌脉": ["舌苔", "舌色", "脉象", "脉滑", "脉细"],
    }

    mentioned = []
    for dimension, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in message:
                if dimension not in mentioned:
                    mentioned.append(dimension)
                break
    return mentioned


def format_intention_response(intention: str, summary: str, info: dict) -> str:
    """Format intention understanding result for display."""
    intention_desc = INTENTION_DESCRIPTIONS.get(intention, "未知意图")

    mentioned = info.get("mentioned_symptoms", [])
    mentioned_str = "、".join(mentioned) if mentioned else "无"

    response = f"""【意图理解】
意图类型：{intention_desc}
意图摘要：{summary}
已提及症状：{mentioned_str}
{"⚠️ 紧急情况" if info.get("is_urgent") else ""}
{f"备注：{info.get('additional_notes', '')}" if info.get('additional_notes') else ""}"""

    return response


def intention_node(state: AgentState) -> AgentState:
    """
    Intention understanding node: analyzes user input to understand their goal.

    Args:
        state: Current agent state

    Returns:
        Updated agent state with parsed intention
    """
    messages = state.get("messages", [])
    user_message = ""

    # Get latest user message
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_message = msg.content
            break

    if not user_message:
        # No user message yet, this is the first turn
        intention = IntentionType.FIRST_VISIT
        summary = "新用户开始问诊"
        info = {
            "mentioned_symptoms": [],
            "is_urgent": False,
            "previous_case_exists": False,
            "additional_notes": ""
        }
    else:
        # Parse intention
        intention, summary, info = parse_user_intention(
            user_message,
            messages[:-1] if len(messages) > 1 else []
        )

    # Format and add intention message
    intention_text = format_intention_response(intention, summary, info)
    intention_message = AIMessage(content=intention_text)

    new_messages = state.get("messages", []) + [intention_message]

    return {
        **state,
        "messages": new_messages,
        "current_phase": "intention",
        "user_intention": intention,
        "intention_summary": summary,
        "intention_info": info,
    }
