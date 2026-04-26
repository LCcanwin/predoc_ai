"""Reflection node for the consultation agent with optimization capabilities."""

from typing import Literal

from langchain_core.messages import AIMessage

from .state import AgentState, TEN_INQUIRY_DIMENSIONS
from ..config import get_llm_config


def assess_symptom_completeness(symptoms_list: list[dict], intention: str = None) -> tuple[list[str], bool]:
    """
    Assess whether symptoms are complete enough to generate a case.

    Args:
        symptoms_list: List of collected symptoms
        intention: User intention type

    Returns:
        Tuple of (missing_dimensions, is_complete_enough)
    """
    from .intention_agent import IntentionType

    collected = {s.get("dimension") for s in symptoms_list if s.get("dimension")}

    # Adapt threshold based on intention
    if intention == IntentionType.QUICK_CONSULT:
        # For quick consult, only need a few key dimensions
        threshold = 6  # At least 4 dimensions
        key_dimensions = ["头身", "寒热", "饮食", "睡眠"]
        key_collected = sum(1 for d in key_dimensions if d in collected)
        is_complete = key_collected >= 3 and len(collected) >= threshold
    elif intention == IntentionType.SPECIFIC_SYMPTOM:
        # For specific symptom, need that symptom + related ones
        threshold = 3
        is_complete = len(collected) >= threshold
    elif intention == IntentionType.GET_PRESCRIPTION:
        # For prescription, need more complete info
        threshold = 6
        is_complete = len(collected) >= threshold
    else:
        # Standard: missing 2 or fewer
        threshold = 2
        is_complete = len([d for d in TEN_INQUIRY_DIMENSIONS if d not in collected]) <= threshold

    missing = [d for d in TEN_INQUIRY_DIMENSIONS if d not in collected]
    return missing, is_complete


def format_ten_inquiry_status(symptoms_list: list[dict]) -> str:
    """Format the status of all ten inquiry dimensions."""
    collected = {s.get("dimension"): s.get("value") for s in symptoms_list if s.get("dimension")}

    lines = []
    for dim in TEN_INQUIRY_DIMENSIONS:
        if dim in collected:
            lines.append(f"✓ {dim}：{collected[dim]}")
        else:
            lines.append(f"✗ {dim}：未收集")

    return "\n".join(lines)


def create_reflection_llm():
    """Create the LLM for reflection."""
    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    return ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=0.3,
    )


def reflect_on_generated_case(state: AgentState) -> tuple[str, bool]:
    """
    Reflect on the generated case to check quality and provide optimization.

    Args:
        state: Current agent state

    Returns:
        Tuple of (reflection_result, needs_optimization)
    """
    from .intention_agent import IntentionType

    messages = state.get("messages", [])
    symptoms_list = state.get("symptoms_list", [])
    user_intention = state.get("user_intention", IntentionType.FIRST_VISIT)

    # Get the generated case (last AI message)
    generated_case = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
            content = msg.content
            if "中医病例" in content or "辨证" in content or "治则" in content:
                generated_case = content
                break

    if not generated_case:
        return "", False

    # Format symptoms
    collected_str = "\n".join([
        f"- {s.get('dimension')}：{s.get('value')}"
        for s in symptoms_list
    ]) or "暂无"

    try:
        llm = create_reflection_llm()
        prompt = f"""作为资深中医专家，请评估以下生成的病例质量，并判断是否需要优化。

【原始症状信息】
{collected_str}

【用户意图】
{user_intention}

【生成的病例】
{generated_case[:2000]}

请从以下角度进行评估：

1. 【完整性检查】
   - 病例是否涵盖了主要症状？
   - 辨证论治是否与症状对应？
   - 治则治法是否合理？

2. 【准确性检查】
   - 辨证分型是否准确？
   - 治疗原则是否恰当？
   - 是否有逻辑矛盾？

3. 【针对性检查】
   - 是否针对用户的具体意图？
   - 对于快速咨询，内容是否过于冗长？
   - 对于需要处方的请求，处方建议是否完整？

请以JSON格式返回：
{{
    "quality_score": 1-10,
    "is_complete": true/false,
    "is_accurate": true/false,
    "needs_optimization": true/false,
    "issues": ["问题1", "问题2"],
    "optimization_suggestions": ["建议1", "建议2"]
}}"""

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # Try to parse JSON
        import json
        import re

        json_match = re.search(r'\{[^{}]*"quality_score"[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            needs_opt = result.get("needs_optimization", False)
            issues = result.get("issues", [])
            suggestions = result.get("optimization_suggestions", [])

            reflection = f"""【病例质量评估】
评分：{result.get('quality_score', 'N/A')}/10
完整性：{'✓ 完整' if result.get('is_complete') else '✗ 不完整'}
准确性：{'✓ 准确' if result.get('is_accurate') else '✗ 存疑'}
{f"发现问题：{'；'.join(issues)}" if issues else ""}
{f"优化建议：{'；'.join(suggestions)}" if suggestions else ""}"""

            return reflection, needs_opt

    except Exception as e:
        print(f"Reflection LLM error: {e}")

    return "", False


def reflection_node(state: AgentState) -> AgentState:
    """
    Reflection node: evaluates if collected information is sufficient.

    Routes to:
    - generation if sufficient (missing <= 2, or based on intention)
    - inquiry if insufficient (missing > 2)
    - optimization if case was generated and needs improvement

    Args:
        state: Current agent state

    Returns:
        Updated agent state with routing decision
    """
    from .intention_agent import IntentionType

    symptoms_list = state.get("symptoms_list", [])
    messages = state.get("messages", [])
    reflection_count = state.get("reflection_count", 0)
    user_intention = state.get("user_intention", IntentionType.FIRST_VISIT)

    # Check if we're reflecting on a generated case (for optimization)
    if state.get("current_phase") == "generation" and state.get("reflection_count", 0) > 0:
        reflection_result, needs_optimization = reflect_on_generated_case(state)
        if needs_optimization:
            # Route back to generation with optimization flag
            reflection_message = AIMessage(content=f"{reflection_result}\n\n需要进行优化...")
            new_messages = state.get("messages", []) + [reflection_message]
            return {
                **state,
                "messages": new_messages,
                "reflection_count": reflection_count + 1,
                "current_phase": "generation",
                "needs_optimization": True,
            }

    # Assess completeness
    missing, is_complete_enough = assess_symptom_completeness(symptoms_list, user_intention)

    # Check reflection limit
    if reflection_count >= 3:
        is_complete_enough = True

    # Format status
    collected_str = "\n".join([
        f"- {s.get('dimension')}：{s.get('value')}"
        for s in symptoms_list
    ]) or "暂无"

    status = format_ten_inquiry_status(symptoms_list)
    missing_str = "\n".join([f"- {m}" for m in missing]) if missing else "所有维度已收集"

    # Get last assistant message for context
    last_assistant_msg = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
            last_assistant_msg = msg.content
            break

    # Get conversation context
    conversation_context = ""
    for msg in messages[-4:]:
        if hasattr(msg, "content") and msg.content:
            role = "患者" if hasattr(msg, "type") and msg.type == "human" else "助手"
            conversation_context += f"{role}：{msg.content[:200]}...\n"

    try:
        llm = create_reflection_llm()

        # Adapt prompt based on intention
        intention_context = ""
        if user_intention == IntentionType.QUICK_CONSULT:
            intention_context = "用户希望快速咨询，标准可以适当放宽。"
        elif user_intention == IntentionType.SPECIFIC_SYMPTOM:
            intention_context = "用户只想咨询特定症状，重点关注提到的症状是否收集完整。"

        prompt = f"""作为中医专家，请评估当前收集的症状信息是否足以完成任务。

【用户意图】
{user_intention}
{intention_context}

已收集症状：
{collected_str}

十问歌状态：
{status}

缺少维度（{len(missing)}个）：
{missing_str}

最近一次问诊内容：
{last_assistant_msg[:500]}

【评估标准】
- 标准问诊：缺少维度 ≤ 2 个 → 可以生成病例
- 快速咨询：至少收集4个关键维度 → 可以生成简化病例
- 特定症状：收集到主要症状 + 2个相关维度 → 可以生成建议
- 反射次数已达上限 → 直接生成病例

请给出简洁的评估和建议，格式：
评估：[简短评估]
建议：继续问诊/生成病例
缺少项：[列出缺少的维度]"""

        response = llm.invoke(prompt)
        reflection_content = f"【症状收集评估】\n\n{response.content}"

    except Exception as e:
        print(f"LLM error: {e}")
        # Fallback to rule-based
        reflection_content = f"""【症状收集评估】

已收集症状：
{collected_str}

十问歌状态：
{status}

缺少维度（{len(missing)}个）：
{missing_str}

评估：{"信息基本完备，可以生成病例" if is_complete_enough else "需要继续问诊收集更多信息"}
建议：{"生成病例" if is_complete_enough else "继续问诊"}
"""

    reflection_message = AIMessage(content=reflection_content)

    # Update state
    new_messages = state.get("messages", []) + [reflection_message]
    new_phase: Literal["reflection", "generation", "inquiry"] = (
        "generation" if is_complete_enough else "inquiry"
    )

    return {
        **state,
        "messages": new_messages,
        "is_complete": is_complete_enough,
        "reflection_count": reflection_count + 1,
        "current_phase": new_phase,
        "missing_dimensions": missing,
    }
