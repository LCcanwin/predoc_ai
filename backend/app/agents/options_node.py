"""Options node for ten inquiry dimension selection."""

import re
from langchain_core.messages import AIMessage

from .state import AgentState, TEN_INQUIRY_DIMENSIONS, DIMENSION_DESCRIPTIONS
from ..config import get_llm_config


def format_ten_inquiry_options() -> str:
    """Format ten inquiry dimensions as selectable options."""
    lines = ["【中医十问歌】请选择您有的症状（可多选）：", ""]
    for i, dim in enumerate(TEN_INQUIRY_DIMENSIONS, 1):
        desc = DIMENSION_DESCRIPTIONS.get(dim, "")
        lines.append(f"{i}. {dim}（{desc}）")
    return "\n".join(lines)


def generate_confirm_questions(
    user_input: str,
    selected_dimensions: list[str],
    mentioned_symptoms: list[dict],
) -> list[str]:
    """
    Generate 2 confirmation questions based on selected dimensions and user input.

    Args:
        user_input: Original user input
        selected_dimensions: User selected dimensions
        mentioned_symptoms: Symptoms extracted from user input

    Returns:
        List of 2 confirmation questions
    """
    # Format mentioned symptoms
    symptom_str = "\n".join([
        f"- {s.get('dimension')}：{s.get('value')}"
        for s in mentioned_symptoms
    ]) if mentioned_symptoms else "无"

    # Format selected dimensions
    selected_str = "、".join(selected_dimensions) if selected_dimensions else "无"

    config = get_llm_config()

    prompt = f"""作为中医问诊助手，请根据用户选择的症状维度，生成2个确认性问题。

【用户原始描述】
{user_input}

【用户选择的症状维度】
{selected_str}

【从用户描述中提取的症状】
{symptom_str}

【要求】
1. 只生成2个问题，不要多也不要少
2. 问题要针对用户选择的最重要的维度
3. 用通俗易懂的语言，像中医大夫一样提问
4. 每个问题后跟选项让用户选择（如：是/否/不确定）
5. 不要编造任何用户未提及的信息

请以JSON格式返回：
{{
    "questions": [
        "问题1？[选项]",
        "问题2？[选项]"
    ]
}}"""

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=0.5,
        )
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON
        import json
        json_match = re.search(r'\{[^{}]*"questions"[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            questions = result.get("questions", [])
            if len(questions) >= 2:
                return questions[:2]

    except Exception as e:
        print(f"LLM error in generate_confirm_questions: {e}")

    # Fallback questions
    return _generate_fallback_questions(selected_dimensions, mentioned_symptoms)


def _generate_fallback_questions(
    selected_dimensions: list[str],
    mentioned_symptoms: list[dict],
) -> list[str]:
    """Generate fallback confirmation questions using rules."""
    questions = []

    # Get top 2 dimensions
    dims_to_ask = selected_dimensions[:2] if selected_dimensions else ["头身", "寒热"]

    for dim in dims_to_ask:
        if dim == "寒热":
            questions.append("您平时是怕冷还是怕热？怕冷/怕热/寒热往来/不确定")
        elif dim == "汗":
            questions.append("您平时出汗多吗？自汗/盗汗/正常/不确定")
        elif dim == "头身":
            questions.append("您有头痛或头晕吗？有/没有/不确定")
        elif dim == "便溏":
            questions.append("您的大便情况如何？正常/溏薄/便秘/不确定")
        elif dim == "饮食":
            questions.append("您最近食欲如何？正常/不振/亢进/不确定")
        elif dim == "胸腹":
            questions.append("您有胸闷或腹胀吗？有/没有/不确定")
        elif dim == "耳目":
            questions.append("您的视力和听力有变化吗？有/没有/不确定")
        elif dim == "口渴":
            questions.append("您口渴吗？口干/正常/不确定")
        elif dim == "睡眠":
            questions.append("您的睡眠质量如何？正常/失眠/嗜睡/不确定")
        elif dim == "舌脉":
            questions.append("您的舌象有明显异常吗？有/没有/不确定")
        else:
            questions.append(f"关于{dim}，您能详细描述一下吗？")

    return questions[:2]


def options_node(state: AgentState) -> AgentState:
    """
    Options node: generates ten inquiry options for user to select.

    This replaces the iterative inquiry approach with a selection-based approach.

    Args:
        state: Current agent state

    Returns:
        Updated agent state with options phase
    """
    messages = state.get("messages", [])
    symptoms_list = state.get("symptoms_list", [])
    user_intention = state.get("user_intention", "first_visit")

    # Get user's original input
    user_input = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_input = msg.content
            break

    # Format the ten inquiry options
    options_text = format_ten_inquiry_options()

    # Add extracted symptoms from user input
    extracted_info = ""
    if symptoms_list:
        extracted_info = "\n\n【从您的描述中已提取的症状】：\n"
        extracted_info += "\n".join([
            f"- {s.get('dimension')}：{s.get('value')}"
            for s in symptoms_list
        ])

    # Add user intention context
    intention_context = ""
    if user_intention == "specific_symptom":
        intention_context = "\n用户只想咨询特定症状，可以跳过一些不相关的维度。"
    elif user_intention == "quick_consult":
        intention_context = "\n用户希望快速咨询，建议选择最关键的2-3个维度。"

    response_content = f"""【已理解您的意图】{"咨询特定症状" if user_intention == "specific_symptom" else "初诊问诊"}

{options_text}
{extracted_info}
{intention_context}

请回复您有的症状编号（如：1,3,5），我会根据您选择的维度继续问诊。"""

    # Generate confirmation questions based on selection (will be shown after selection)
    # For now, store empty list - will be populated after user selects
    confirm_questions = []

    response_message = AIMessage(content=response_content)

    new_messages = state.get("messages", []) + [response_message]

    return {
        **state,
        "messages": new_messages,
        "current_phase": "options",
        "selected_dimensions": [],
        "confirm_questions": confirm_questions,
    }


def process_selection(
    selected_indices: list[int],
    state: AgentState,
) -> tuple[str, list[str], list[dict]]:
    """
    Process user's selection and generate confirmation questions.

    Args:
        selected_indices: User selected dimension indices (1-based)
        state: Current agent state

    Returns:
        Tuple of (response_content, confirm_questions, updated_symptoms)
    """
    messages = state.get("messages", [])
    symptoms_list = state.get("symptoms_list", [])
    user_intention = state.get("user_intention", "first_visit")

    # Get user's original input
    user_input = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_input = msg.content
            break

    # Convert indices to dimensions (1-based to 0-based)
    selected_dimensions = []
    for idx in selected_indices:
        if 1 <= idx <= len(TEN_INQUIRY_DIMENSIONS):
            selected_dimensions.append(TEN_INQUIRY_DIMENSIONS[idx - 1])

    # Also add symptoms already extracted from user input
    existing_dims = {s.get("dimension") for s in symptoms_list}
    all_symptoms = list(symptoms_list)

    # For dimensions not in user input but selected, mark as "待确认"
    for dim in selected_dimensions:
        if dim not in existing_dims:
            all_symptoms.append({"dimension": dim, "value": "待确认"})

    # Generate confirmation questions
    confirm_questions = generate_confirm_questions(
        user_input,
        selected_dimensions,
        symptoms_list,
    )

    # Format confirmation questions
    questions_text = "\n\n".join([
        f"{i+1}. {q}"
        for i, q in enumerate(confirm_questions)
    ])

    response_content = f"""好的，我已记录您选择的症状维度：{', '.join(selected_dimensions)}

请确认以下2个问题：

{questions_text}

请分别回复您的答案（如：1.是 2.否）。"""

    return response_content, confirm_questions, all_symptoms


def confirm_node(
    answers: list[str],
    state: AgentState,
) -> AgentState:
    """
    Process confirmation answers and update symptoms.

    Args:
        answers: User's answers to confirmation questions
        state: Current agent state

    Returns:
        Updated agent state ready for RAG and generation
    """
    messages = state.get("messages", [])
    symptoms_list = state.get("symptoms_list", [])
    confirm_questions = state.get("confirm_questions", [])
    selected_dimensions = state.get("selected_dimensions", [])

    # Update symptoms based on answers
    updated_symptoms = []
    for symptom in symptoms_list:
        dim = symptom.get("dimension")
        value = symptom.get("value", "")

        # Check if this dimension was confirmed/answered
        if dim in selected_dimensions and value == "待确认":
            # Try to parse answer
            answer_text = " ".join(answers).lower()
            if "不确定" in answer_text or "不知道" in answer_text:
                value = "待进一步确认"
            elif "否" in answer_text or "没有" in answer_text:
                value = "无"
            else:
                value = answer_text[:50] if answer_text else "待确认"

        updated_symptoms.append({"dimension": dim, "value": value})

    # Check if we have enough information
    existing_dims = {s.get("dimension") for s in updated_symptoms if s.get("value") not in ["待确认", "无", ""]}
    is_complete = len(existing_dims) >= 4  # At least 4 dimensions collected

    # If not complete, add fallback questions
    missing = [d for d in TEN_INQUIRY_DIMENSIONS if d not in existing_dims]

    if missing and len(missing) <= 2:
        # Only missing a few, add them
        for dim in missing:
            updated_symptoms.append({"dimension": dim, "value": "待补充"})
        is_complete = True

    response_content = f"""好的，已确认您的症状信息。

【已收集的症状】：
{chr(10).join([f"- {s.get('dimension')}：{s.get('value')}" for s in updated_symptoms if s.get('value') not in ['待确认', '待补充', '']])}

{"症状信息已足够，正在为您生成病例..." if is_complete else "感谢您的配合，我将根据这些信息为您分析。"}"""

    response_message = AIMessage(content=response_content)
    new_messages = state.get("messages", []) + [response_message]

    return {
        **state,
        "messages": new_messages,
        "symptoms_list": updated_symptoms,
        "current_phase": "generation" if is_complete else "inquiry",
        "is_complete": is_complete,
    }
