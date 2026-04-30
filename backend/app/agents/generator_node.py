"""Generator node for case output generation with optimization."""

import re

from langchain_core.messages import AIMessage

from .state import AgentState, TEN_INQUIRY_DIMENSIONS
from .rag_retriever import RAGRetriever
from .intention_agent import IntentionType
from .diagnosis_validator import append_validation_summary, validate_diagnosis_report
from .rules_engine import build_rule_context, format_rule_context, infer_diagnosis
from .symptom_rag_agent import retrieve_symptom_context
from ..config import get_llm_config


UN_COLLECTED_FIELD_TEXT = "本次预问诊未采集该信息。"
PAST_HISTORY_UNCOLLECTED_TEXT = "本次预问诊未采集既往史信息。"


# Standard TCM case template
CASE_TEMPLATE = """# 诊断结论

## 基本信息
姓名：{name}
性别：{gender}
年龄：{age}

## 主诉
{main_complaint}

## 现病史
{present_illness}

## 既往史
{past_history}

## 十问歌
{ten_inquiry}

## 辨证
{differentiation}

## 治则
{treatment_principle}
"""


def extract_symptom_value(symptoms_list: list[dict], dimension: str) -> str:
    """Extract symptom value for a specific dimension."""
    for symptom in symptoms_list:
        if symptom.get("dimension") == dimension:
            return symptom.get("value", "")
    return ""


def format_ten_inquiry_for_case(symptoms_list: list[dict]) -> str:
    """Format only collected symptoms into the ten inquiry format."""
    lines = []
    for dim in TEN_INQUIRY_DIMENSIONS:
        value = extract_symptom_value(symptoms_list, dim)
        if value and value not in ["未记录", "待确认", "待进一步确认", ""]:
            lines.append(f"- {dim}：{value}")
    return "\n".join(lines) if lines else "本次未采集到明确症状。"


def sanitize_case_text(case_text: str) -> str:
    """Remove unsupported clinical claims from fields this flow does not collect."""
    if not case_text:
        return case_text

    replacements = {
        "既往史": PAST_HISTORY_UNCOLLECTED_TEXT,
        "过敏史": "本次预问诊未采集过敏史信息。",
        "用药史": "本次预问诊未采集用药史信息。",
        "家族史": "本次预问诊未采集家族史信息。",
    }

    sanitized = case_text
    for heading, value in replacements.items():
        pattern = rf"(## {heading}\n)(.*?)(?=\n## |\Z)"
        if re.search(pattern, sanitized, flags=re.DOTALL):
            sanitized = re.sub(pattern, rf"\1{value}\n", sanitized, flags=re.DOTALL)

    return sanitized


def create_generator_llm(temperature: float = 0.5):
    """Create the LLM for case generation."""
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    return ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=temperature,
    )


def _format_recent_conversation(messages: list) -> str:
    lines = []
    for msg in messages:
        if hasattr(msg, "content") and msg.content:
            role = "患者" if hasattr(msg, "type") and msg.type == "human" else "助手"
            lines.append(f"{role}：{msg.content}")
    return "\n".join(lines)


def _build_generation_context(messages: list, session_summary: str = "", max_chars: int = 2000) -> str:
    """Combine compressed session history with the latest turns, preserving the tail."""
    recent_context = _format_recent_conversation(messages)
    sections = []
    if session_summary:
        sections.append(f"【本次早期会话摘要】\n{session_summary}")
    if recent_context:
        sections.append(f"【最近对话】\n{recent_context}")

    context = "\n\n".join(sections)
    if len(context) <= max_chars:
        return context

    if session_summary:
        summary_budget = min(len(session_summary), max_chars // 2)
        summary_part = session_summary[-summary_budget:].lstrip()
        recent_budget = max_chars - summary_budget - 18
        recent_part = recent_context[-recent_budget:].lstrip() if recent_budget > 0 else ""
        return f"【本次早期会话摘要】\n{summary_part}\n\n【最近对话】\n{recent_part}".strip()

    return context[-max_chars:].lstrip()


def generate_case_text(state: AgentState, retriever: RAGRetriever, user_name: str = "匿名") -> str:
    """
    Generate a structured TCM case from collected symptoms using LLM.

    Args:
        state: Current agent state
        retriever: RAG retriever for knowledge base context
        user_name: Optional patient name

    Returns:
        Formatted case text in markdown
    """
    symptoms_list = state.get("symptoms_list", [])
    messages = state.get("messages", [])
    user_intention = state.get("user_intention", IntentionType.FIRST_VISIT)
    intention_info = state.get("intention_info", {})
    memory_context = state.get("memory_context", "")
    session_summary = state.get("session_summary", "")
    retrieved_context = state.get("retrieved_context", "")
    rule_messages = messages
    if session_summary:
        rule_messages = [AIMessage(content=session_summary)] + messages
    rule_context = build_rule_context(symptoms_list, rule_messages)

    # Format ten inquiry section
    ten_inquiry = format_ten_inquiry_for_case(symptoms_list)

    conversation_context = _build_generation_context(messages, session_summary, max_chars=2200)

    # Get relevant knowledge from the symptom RAG agent.
    _, knowledge_context = retrieve_symptom_context(state, retriever, task="generation", limit=5)

    # Check if this is an optimization pass
    needs_optimization = state.get("needs_optimization", False)
    previous_case = ""
    if needs_optimization:
        # Find the previous case
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
                content = msg.content
                if "中医病例" in content or "辨证" in content:
                    previous_case = content
                    break

    try:
        llm = create_generator_llm()

        # Adapt generation based on intention
        if user_intention == IntentionType.QUICK_CONSULT:
            case_text = _generate_quick_consultation(symptoms_list, conversation_context, knowledge_context, user_name)
            return _finalize_case_text(case_text, symptoms_list, rule_messages)
        elif user_intention == IntentionType.SPECIFIC_SYMPTOM:
            case_text = _generate_specific_symptom_response(symptoms_list, conversation_context, knowledge_context)
            return _finalize_case_text(case_text, symptoms_list, rule_messages)
        elif user_intention == IntentionType.GET_PRESCRIPTION:
            case_text = _generate_prescription_suggestion(symptoms_list, conversation_context, knowledge_context)
            return _finalize_case_text(case_text, symptoms_list, rule_messages)
        elif needs_optimization:
            case_text = _optimize_case(symptoms_list, previous_case, knowledge_context)
            return _finalize_case_text(case_text, symptoms_list, rule_messages)

        # Standard full case generation
        prompt = f"""作为资深中医专家，请根据以下症状信息生成一份标准化的诊断结论。

【已采集症状】
{ten_inquiry}

【问诊对话摘要】
{_build_generation_context(messages, session_summary, max_chars=1500)}

【用户短期记忆】（仅作近期背景参考，不能替代本次问诊信息）
{memory_context if memory_context else "无"}

【规则引擎参考】（用于约束辨证方向，不能替代医生判断）
{format_rule_context(rule_context)}

【相关知识库参考】
{knowledge_context}

请生成包含以下部分的诊断结论（使用Markdown格式）：

# 诊断结论

## 基本信息
姓名：[仅当对话明确提供姓名时填写，否则写"匿名"]
性别：[仅当对话明确提供性别时填写，否则写"本次预问诊未采集该信息"]
年龄：[仅当对话明确提供年龄时填写，否则写"本次预问诊未采集该信息"]

## 主诉
[一句话概括最主要症状]

## 现病史
[详细描述发病过程、症状特点、伴随症状等]

## 既往史
本次预问诊未采集既往史信息。

## 十问歌
[只整理已采集到的症状维度，不要输出未采集维度，不要写"未记录"]

## 辨证
[根据症状推理出的证型，结合中医理论]

## 治则
[对应的治疗原则]

注意：
1. 只使用【已采集症状】和【问诊对话摘要】中用户明确提供的信息，不要编造
2. 严禁推断既往史、过敏史、用药史、家族史、性别、年龄等用户未明确提供的信息
3. 如果某字段未明确采集，必须写"本次预问诊未采集该信息"或对应的"本次预问诊未采集...信息"
4. 不要写"无特殊"、"否认"、"无明显"、"健康"等表示已确认排除的内容，除非用户原文明确说明
5. 辨证要结合中医理论，但必须基于已采集症状
6. 语言要专业准确
7. 不要输出 <think> 标签或任何思考过程"""

        response = llm.invoke(prompt)
        return _finalize_case_text(response.content, symptoms_list, rule_messages)

    except Exception as e:
        print(f"LLM error: {e}")
        # Fallback to rule-based generation
        case_text = _generate_fallback_case(symptoms_list, user_name)
        return _finalize_case_text(case_text, symptoms_list, rule_messages)


def _finalize_case_text(case_text: str, symptoms_list: list[dict], messages: list) -> str:
    """Sanitize and validate final report before returning it."""
    sanitized = sanitize_case_text(case_text)
    validation = validate_diagnosis_report(sanitized, symptoms_list, messages)
    return append_validation_summary(sanitized, validation)


def _generate_quick_consultation(symptoms_list: list[dict], context: str, knowledge: str, user_name: str) -> str:
    """Generate a quick consultation response."""
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    llm = ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=0.5,
    )

    # Extract key symptoms
    key_symptoms = []
    for s in symptoms_list:
        dim = s.get("dimension", "")
        value = s.get("value", "")
        if value and value not in ["无", "未记录"]:
            key_symptoms.append(f"{dim}：{value}")

    prompt = f"""作为中医专家，请根据以下信息给出快速的健康建议。

【关键症状】
{"；".join(key_symptoms)}

【对话摘要】
{context[:1000]}

【知识参考】
{knowledge}

请给出简洁的建议（Markdown格式）：

# 快速咨询建议

## 主要问题
[一句话概括]

## 可能的原因
[简要分析]

## 建议
1. [调理建议]
2. [注意事项]
3. [是否需要进一步检查]

## 食疗建议
[如有适合的食疗方案]

注意：只回答用户当前的问题，不要过度展开。"""

    response = llm.invoke(prompt)
    return response.content


def _generate_specific_symptom_response(symptoms_list: list[dict], context: str, knowledge: str) -> str:
    """Generate response for specific symptom consultation."""
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    llm = ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=0.5,
    )

    # Get the main symptom
    main_symptom = ""
    for s in symptoms_list:
        value = s.get("value", "")
        if value and value not in ["无", "未记录"]:
            main_symptom = f"{s.get('dimension')}：{value}"
            break

    prompt = f"""作为中医专家，请分析以下症状：

【主要症状】
{main_symptom}

【相关症状】
{context[:800]}

【知识参考】
{knowledge}

请给出专业分析（Markdown格式）：

# 症状分析

## 症状描述
[详细描述这个症状]

## 可能的中医理解
[从中医角度分析]

## 辨证要点
[需要鉴别的证型]

## 调理建议
1. [情志调节]
2. [饮食建议]
3. [作息调整]
4. [如需就医的情况]"""

    response = llm.invoke(prompt)
    return response.content


def _generate_prescription_suggestion(symptoms_list: list[dict], context: str, knowledge: str) -> str:
    """Generate prescription suggestion (advisory only)."""
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    llm = ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=0.5,
    )

    ten_inquiry = format_ten_inquiry_for_case(symptoms_list)

    prompt = f"""作为中医专家，请根据以下症状信息给出调理建议。

【症状信息】
{ten_inquiry}

【对话摘要】
{context[:1000]}

【知识参考】
{knowledge}

请注意：以下内容仅供参考，不能替代正式就医。

请生成调理建议（Markdown格式）：

# 中医调理建议

## 辨证分析
[根据症状分析证型]

## 调理原则
[对应的调理原则]

## 生活方式建议
1. 饮食：[具体建议]
2. 作息：[具体建议]
3. 情志：[具体建议]
4. 运动：[具体建议]

## 食疗方案
[适合的药食同源食材]

## 注意事项
[需要避免的事项]

⚠️ 重要提示：以上内容仅供参考，如有不适请及时就医。
具体用药需要在中医师面诊后确定。"""

    response = llm.invoke(prompt)
    return response.content


def _optimize_case(symptoms_list: list[dict], previous_case: str, knowledge: str) -> str:
    """Optimize the previously generated case."""
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    llm = ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=0.4,
    )

    ten_inquiry = format_ten_inquiry_for_case(symptoms_list)

    prompt = f"""作为资深中医专家，请根据反馈优化以下病例。

【原始症状】
{ten_inquiry}

【之前的病例】
{previous_case[:2000]}

【知识库参考】
{knowledge}

请优化病例，重点改进：
1. 辨证是否准确
2. 治则治法是否对应
3. 内容是否完整

请返回优化后的完整病例："""

    response = llm.invoke(prompt)
    return response.content


def _generate_fallback_case(symptoms_list: list[dict], user_name: str) -> str:
    """Generate case using rules when LLM fails."""
    main_complaint = extract_symptom_value(symptoms_list, "头身") or "见下述症状"

    present_illness_parts = []
    for dim in TEN_INQUIRY_DIMENSIONS:
        value = extract_symptom_value(symptoms_list, dim)
        if value and value != "未记录":
            present_illness_parts.append(f"{dim}：{value}")

    present_illness = "；".join(present_illness_parts) if present_illness_parts else "本次未采集到明确症状。"

    ten_inquiry = format_ten_inquiry_for_case(symptoms_list)

    differentiation, treatment_principle, _ = infer_diagnosis(symptoms_list)

    return CASE_TEMPLATE.format(
        name=user_name or "匿名",
        gender=UN_COLLECTED_FIELD_TEXT,
        age=UN_COLLECTED_FIELD_TEXT,
        main_complaint=main_complaint,
        present_illness=present_illness,
        past_history=PAST_HISTORY_UNCOLLECTED_TEXT,
        ten_inquiry=ten_inquiry,
        differentiation=differentiation,
        treatment_principle=treatment_principle,
    )


def generator_node(state: AgentState, retriever: RAGRetriever, user_name: str = "匿名") -> AgentState:
    """
    Generator node: produces the final structured case.

    Args:
        state: Current agent state
        retriever: RAG retriever for knowledge base context
        user_name: Optional patient name

    Returns:
        Updated agent state with generated case
    """
    needs_optimization = state.get("needs_optimization", False)

    # Generate or optimize case
    case_text = generate_case_text(state, retriever, user_name)

    generation_message = AIMessage(content=case_text)

    new_messages = state.get("messages", []) + [generation_message]

    return {
        **state,
        "messages": new_messages,
        "is_complete": True,
        "current_phase": "complete",
        "needs_optimization": False,  # Reset after use
    }
