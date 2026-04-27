"""Inquiry node for the consultation agent."""

import asyncio
import re
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.callbacks import AsyncCallbackHandler

from .state import AgentState, TEN_INQUIRY_DIMENSIONS, DIMENSION_DESCRIPTIONS
from .rag_retriever import RAGRetriever
from .intention_agent import IntentionType
from .symptom_rag_agent import retrieve_symptom_context
from ..config import get_llm_config


def get_missing_dimensions(symptoms_list: list[dict]) -> list[str]:
    """Get dimensions that haven't been collected yet."""
    collected = {s.get("dimension") for s in symptoms_list if s.get("dimension")}
    return [d for d in TEN_INQUIRY_DIMENSIONS if d not in collected]


def format_collected_symptoms(symptoms_list: list[dict]) -> str:
    """Format collected symptoms for prompt."""
    if not symptoms_list:
        return "暂无"

    formatted = []
    for symptom in symptoms_list:
        dim = symptom.get("dimension", "未知")
        value = symptom.get("value", "未填写")
        formatted.append(f"- {dim}：{value}")
    return "\n".join(formatted)


def _generate_fallback_inquiry(symptoms_list: list[dict], missing: list[str], intention: str = None) -> AIMessage:
    """Generate fallback inquiry using rules, adapting to user intention."""
    inquiries = {
        "寒热": "请问您平时是否怕冷或怕热？有没有发热或寒热往来的情况？",
        "汗": "请问您平时出汗情况如何？有没有盗汗（睡觉时出汗）或自汗（不动也出汗）的情况？",
        "头身": "请问您有没有头痛、头晕，或者身体其他部位疼痛（如腰痛、肩背痛等）？",
        "便溏": "请问您的大便情况如何？是否成形？每天几次？有没有腹泻或便秘？",
        "饮食": "请问您最近的食欲如何？有没有口味偏好或厌恶某些食物的情况？",
        "胸腹": "请问您有没有胸闷、胸痛、腹胀、腹痛等不适感？",
        "耳目": "请问您的视力和听力有没有什么变化？有没有耳鸣或视力模糊的情况？",
        "口渴": "请问您口渴吗？有没有特别喜欢喝冷水或热水的情况？",
        "睡眠": "请问您最近睡眠质量如何？有没有失眠、多梦或嗜睡的情况？",
        "舌脉": "如果方便的话，能否描述一下您的舌象（舌色、舌苔）？如有脉象信息也请告知。",
    }

    # Adapt questions based on intention
    if intention == IntentionType.QUICK_CONSULT:
        # For quick consult, only ask about the most important dimensions
        priority_dims = ["头身", "寒热", "饮食", "睡眠"]
        missing = [d for d in priority_dims if d in missing]
        if missing:
            dim = missing[0]
            text = f"为了更好地帮助您，请问您{DIMENSION_DESCRIPTIONS.get(dim, dim)}方面的情况如何？"
            return AIMessage(content=text)
        else:
            return AIMessage(content="好的，您的情况我已经了解了。请问还有什么需要补充的吗？")

    if intention == IntentionType.SPECIFIC_SYMPTOM:
        # For specific symptom, skip routine questions
        mentioned = []
        for symptom in symptoms_list:
            dim = symptom.get("dimension", "")
            value = symptom.get("value", "")
            if value and value not in ["无", "未提及"]:
                mentioned.append(f"{dim}：{value}")

        text = f"好的，您提到的问题是：{'；'.join(mentioned) if mentioned else '这个症状'}。"
        text += "能否详细描述一下：这种情况持续多长时间了？有没有什么诱因或加重的因素？"
        return AIMessage(content=text)

    if missing:
        dim = missing[0]
        text = inquiries.get(dim, f"请告诉我您关于'{dim}'方面的情况。")
    else:
        text = "感谢您的配合，您的症状信息已收集完整。请问您还有其他需要补充的症状吗？"

    return AIMessage(content=text)


def generate_inquiry_message(state: AgentState, retriever: RAGRetriever) -> AIMessage:
    """
    Generate the next inquiry message based on current state using LLM.

    Args:
        state: Current agent state
        retriever: RAG retriever for knowledge base

    Returns:
        AIMessage with the inquiry question
    """
    symptoms_list = state.get("symptoms_list", [])
    messages = state.get("messages", [])
    user_intention = state.get("user_intention", IntentionType.FIRST_VISIT)
    intention_info = state.get("intention_info", {})
    intention_summary = state.get("intention_summary", "")
    memory_context = state.get("memory_context", "")
    collected = format_collected_symptoms(symptoms_list)
    missing = get_missing_dimensions(symptoms_list)

    # Get conversation history for context (strip think tags)
    conversation_context = ""
    for msg in messages[-6:]:
        if hasattr(msg, "content") and msg.content:
            content = msg.content
            # Remove think tags for context
            content = re.sub(r'<think>[\s\S]*?</think>', '', content)
            role = "患者" if hasattr(msg, "type") and msg.type == "human" else "助手"
            conversation_context += f"{role}：{content}\n"

    # Format missing dimensions
    missing_formatted = []
    for dim in missing:
        desc = DIMENSION_DESCRIPTIONS.get(dim, "")
        missing_formatted.append(f"- {dim}：{desc}")
    missing_str = "\n".join(missing_formatted) if missing_formatted else "所有维度已收集完毕"

    # Get relevant knowledge from the symptom RAG agent.
    _, knowledge_context = retrieve_symptom_context(state, retriever, task="inquiry", limit=3)

    # Adapt prompt based on intention
    intention_instruction = ""
    if user_intention == IntentionType.QUICK_CONSULT:
        intention_instruction = "【特殊要求】用户希望快速咨询，请只问2-3个最关键的问题，不要过多追问。"
    elif user_intention == IntentionType.SPECIFIC_SYMPTOM:
        mentioned = [s.get("dimension") for s in symptoms_list]
        intention_instruction = f"【特殊要求】用户只想咨询特定症状（{'、'.join(mentioned)}），请围绕这些症状深入询问。"
    elif user_intention == IntentionType.GET_PRESCRIPTION:
        intention_instruction = "【特殊要求】用户想要获取处方，请收集关键症状信息以便给出建议。"

    # Build prompt
    prompt = f"""你是一位专业的中医问诊AI助手，名为"华佗小助手"。

【核心原则】（必须严格遵守）
1. 只根据【当前已收集的症状】和【最近对话】中用户提供的信息进行回复
2. 绝对不要编造、推测或添加任何用户未提到的症状或信息
3. 如果某个症状没有在列表中显示为"未记录"，不要声称已经了解该症状

【用户意图理解】
{intention_instruction}
意图摘要：{intention_summary}

【当前已收集的症状】（只读取这些，不要编造）
{collected}

【尚未收集的维度】
{missing_str}

【最近对话】（用户明确说的内容）
{conversation_context if conversation_context else "（用户刚开始描述）"}

【用户短期记忆】（仅作为近期背景，不能当作本次明确症状）
{memory_context if memory_context else "无"}

【知识库参考】（仅供参考，不要直接引用）
{knowledge_context}

请作为华佗小助手，根据以上信息生成下一轮问诊内容。注意：
1. 每次只问1-2个问题
2. 要结合之前的对话上下文自然地追问
3. 语言要像和蔼的中医大夫一样，温和耐心
4. 如果所有维度都已收集，询问患者是否还有其他不适
5. 只询问尚未收集的症状维度，不要重复询问已收集的信息"""

    config = get_llm_config()

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=0.7,
        )
        response = llm.invoke(prompt)
        return AIMessage(content=response.content)
    except Exception as e:
        print(f"LLM error: {e}")
        return _generate_fallback_inquiry(symptoms_list, missing, user_intention)


def retrieve_knowledge_for_context(state: AgentState, retriever: RAGRetriever) -> str:
    """
    Retrieve relevant knowledge based on current context.

    Args:
        state: Current agent state
        retriever: RAG retriever

    Returns:
        Formatted knowledge context
    """
    _, context = retrieve_symptom_context(state, retriever, task="inquiry", limit=4)
    return context


def inquiry_node(state: AgentState, retriever: RAGRetriever) -> AgentState:
    """Inquiry node: generates questions to collect symptom information."""
    user_intention = state.get("user_intention", IntentionType.FIRST_VISIT)
    missing = get_missing_dimensions(state.get("symptoms_list", []))

    # Generate inquiry message using LLM with RAG context
    inquiry_message = generate_inquiry_message(state, retriever)

    # Also retrieve knowledge for context
    retrieved_context = retrieve_knowledge_for_context(state, retriever)

    new_messages = state.get("messages", []) + [inquiry_message]
    new_phase = "inquiry"

    return {
        **state,
        "messages": new_messages,
        "current_phase": new_phase,
        "retrieved_context": retrieved_context,
    }
