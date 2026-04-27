"""API routes for consultation endpoints."""

import uuid
import asyncio
import json
import re
from datetime import datetime
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse
import logging

from langchain_core.messages import AIMessage, HumanMessage

from ..config import VECTOR_STORE_PATH
from ..auth import (
    append_experience_event,
    append_memory_event,
    get_consultation_records,
    get_current_user,
)
from ..agents.message_queue import UserMessageQueue
from ..agents.memory_agent import (
    build_memory_message,
    enrich_intention_info,
    hydrate_symptoms_from_memory,
    load_user_memory,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/consultation", tags=["consultation"])

# Session storage
_sessions: Dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=4)
_retriever = None
_message_queue = UserMessageQueue()


def strip_think_tags(content: str) -> str:
    """Remove <think>...</think> tags from LLM response content."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    return content.strip()


def _get_retriever():
    """Load and cache the knowledge-base retriever."""
    global _retriever
    if _retriever is None:
        from ..knowledge_base.vector_store import VectorStore
        from ..agents.rag_retriever import RAGRetriever

        vector_store = VectorStore(persist_path=VECTOR_STORE_PATH)
        loaded = vector_store.load()
        if not loaded:
            logger.warning("Vector store not loaded from %s; RAG retrieval will return empty context", VECTOR_STORE_PATH)
        _retriever = RAGRetriever(vector_store)
    return _retriever


@router.post("/start", response_model=dict)
async def start_consultation(request: dict, current_user: dict = Depends(get_current_user)):
    """Create a new consultation session."""
    thread_id = str(uuid.uuid4())
    memory = load_user_memory(current_user["id"])
    memory_context = memory.get("memory_context", "")

    session = {
        "thread_id": thread_id,
        "user_id": current_user["id"],
        "user_name": request.get("user_name", "匿名"),
        "created_at": datetime.now().isoformat(),
        "messages": [],
        "symptoms_list": [],
        "is_complete": False,
        "user_intention": None,
        "intention_summary": "",
        "intention_info": {},
        "memory_context": memory_context,
        "memory_agent": memory,
    }

    _sessions[thread_id] = session

    return {
        "thread_id": thread_id,
        "created_at": session["created_at"],
        "memory_context": memory_context,
    }


@router.get("/history", response_model=dict)
async def get_history(current_user: dict = Depends(get_current_user)):
    """Return previous consultation records for the logged-in user."""
    return {"records": get_consultation_records(current_user["id"])}


def _process_turn_sync(messages: list, symptoms_list: list, thread_id: str, session_data: dict = None) -> tuple[str, str, bool, list]:
    """
    Process a consultation turn using the new options-based workflow.

    Flow:
    1. First message: Understand intention, show ten inquiry options
    2. User selects options (e.g., "1,3,5"): Generate confirmation questions
    3. User confirms: Update symptoms, check completeness
    4. If complete: Move to generation

    Returns:
        Tuple of (response_content, phase, is_complete, all_symptoms)
    """
    try:
        from ..agents.intention_agent import (
            parse_user_intention,
            should_reclassify_intention,
            IntentionType,
            INTENTION_DESCRIPTIONS,
        )
        from ..agents.options_node import format_ten_inquiry_options, generate_confirm_questions
        from ..agents.state import AgentState, TEN_INQUIRY_DIMENSIONS

        retriever = _get_retriever()

        # Initialize session_data if not provided
        if session_data is None:
            session_data = {}

        current_phase = session_data.get("current_phase", "intention")

        # Get user's latest message
        user_message = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_message = msg.content
                break

        # Step 1: Understand user intention (first turn only)
        memory_agent = session_data.get("memory_agent") or {}
        memory_context = session_data.get("memory_context", "")
        intention_history = messages[:-1]
        memory_message = build_memory_message(memory_agent)
        if memory_message is not None:
            intention_history = [memory_message] + intention_history

        intention = session_data.get("user_intention")
        if intention is None and user_message:
            intention, summary, info = parse_user_intention(user_message, intention_history)
            intention, info = enrich_intention_info(intention, info, memory_agent)
            session_data["user_intention"] = intention
            session_data["intention_summary"] = summary
            session_data["intention_info"] = info
            session_data["current_phase"] = "options"
        elif intention is not None and user_message and should_reclassify_intention(user_message, intention):
            updated_intention, summary, info = parse_user_intention(user_message, intention_history)
            updated_intention, info = enrich_intention_info(updated_intention, info, memory_agent)
            if info.get("confidence", 0) >= 0.65 and updated_intention != intention:
                intention = updated_intention
                session_data["user_intention"] = intention
                session_data["intention_summary"] = summary
                session_data["intention_info"] = info

        if intention is None:
            intention = IntentionType.FIRST_VISIT
            session_data["user_intention"] = intention
            session_data["current_phase"] = "options"

        current_phase = session_data.get("current_phase", current_phase)

        # Step 2: Extract symptoms from user's message
        new_symptoms = _extract_symptoms_from_text(user_message)
        existing_dims = {s.get("dimension") for s in symptoms_list if s.get("dimension")}
        for sym in new_symptoms:
            if sym.get("dimension") not in existing_dims:
                symptoms_list.append(sym)
                existing_dims.add(sym.get("dimension"))

        all_symptoms, memory_symptoms = hydrate_symptoms_from_memory(
            symptoms_list,
            memory_agent,
            intention,
        )
        if memory_symptoms:
            session_data["memory_hydrated_symptoms"] = memory_symptoms

        # Step 3: Determine response based on current phase
        response_content = ""
        new_phase = current_phase

        if current_phase == "options":
            # First try to parse as "dimension: value" format from new UI
            parsed_symptoms = _parse_symptom_selections(user_message)
            selected_indices = []

            if parsed_symptoms:
                # User selected options with actual values - process them directly
                for sym in parsed_symptoms:
                    if sym.get("dimension") not in existing_dims:
                        all_symptoms.append(sym)
                        existing_dims.add(sym.get("dimension"))

                # Mark selected dimensions as "待确认" in symptoms
                selected_dimensions = [s["dimension"] for s in parsed_symptoms]
                session_data["selected_dimensions"] = selected_dimensions

                # Generate confirmation questions
                confirm_questions = generate_confirm_questions(
                    user_message,
                    selected_dimensions,
                    all_symptoms
                )
                session_data["confirm_questions"] = confirm_questions
                session_data["current_phase"] = "confirm"

                # Format response
                questions_text = "\n\n".join([
                    f"{i+1}. {q}"
                    for i, q in enumerate(confirm_questions)
                ])
                response_content = f"""好的，我已记录您选择的症状：
{user_message}

请确认以下{len(confirm_questions)}个问题：

{questions_text}

请分别回复您的答案。"""

            else:
                # Check if user sent option numbers (e.g., "1,3,5" or "1 3 5") - legacy format
                selected_indices = _parse_option_selection(user_message)

            if not parsed_symptoms and selected_indices:
                # User selected options - generate confirmation questions
                selected_dimensions = []
                for idx in selected_indices:
                    if 1 <= idx <= len(TEN_INQUIRY_DIMENSIONS):
                        selected_dimensions.append(TEN_INQUIRY_DIMENSIONS[idx - 1])

                session_data["selected_dimensions"] = selected_dimensions

                # Mark selected dimensions as "待确认" in symptoms
                for dim in selected_dimensions:
                    if dim not in existing_dims:
                        all_symptoms.append({"dimension": dim, "value": "待确认"})

                # Generate confirmation questions
                confirm_questions = generate_confirm_questions(
                    user_message,
                    selected_dimensions,
                    symptoms_list
                )
                session_data["confirm_questions"] = confirm_questions
                session_data["current_phase"] = "confirm"

                # Format response
                questions_text = "\n\n".join([
                    f"{i+1}. {q}"
                    for i, q in enumerate(confirm_questions)
                ])
                response_content = f"""好的，我已记录您选择的症状维度：{', '.join(selected_dimensions)}

请确认以下2个问题：

{questions_text}

请分别回复您的答案。"""

            elif not parsed_symptoms:
                # Show ten inquiry options
                options_text = format_ten_inquiry_options()

                extracted_info = ""
                if all_symptoms:
                    extracted_info = "【从您的描述中已提取的症状】：\n"
                    extracted_info += "\n".join([
                        f"- {s.get('dimension')}：{s.get('value')}"
                        for s in all_symptoms
                    ])

                extracted_summary = extracted_info or "【从您的描述中已提取的症状】：\n- 暂未识别到明确症状，请在下方补充选择。"
                if memory_symptoms:
                    memory_summary = "\n".join([
                        f"- {item.get('dimension')}：{item.get('value')}"
                        for item in memory_symptoms
                    ])
                    extracted_summary += f"\n\n【从历史记忆中带入的相关症状】：\n{memory_summary}"
                response_content = f"""我先帮您记录当前描述，并做一次症状确认。

{extracted_summary}

{options_text}

请在下方补充或修正相关症状，确认后我会再问2个关键问题，然后直接生成诊断结论。"""

        elif current_phase == "confirm":
            # Process confirmation answers after LLM analysis
            selected_dimensions = session_data.get("selected_dimensions", [])
            confirm_questions = session_data.get("confirm_questions", [])

            # Parse user's answers - split by common delimiters
            raw_answers = user_message.strip()
            # Try to split by "，" (Chinese comma), ",", or "."
            import re
            answer_parts = re.split(r'[,，\.。\s]+', raw_answers)
            answer_parts = [a.strip() for a in answer_parts if a.strip()]

            # Update symptoms based on answers.
            for symptom in all_symptoms:
                dim = symptom.get("dimension")
                if dim in selected_dimensions:
                    # Find corresponding answer if available
                    idx = selected_dimensions.index(dim)
                    if idx < len(answer_parts):
                        answer = answer_parts[idx]
                        # Analyze answer and update symptom info
                        if "不确定" in answer or "不知道" in answer:
                            symptom["value"] = "待进一步确认"
                        elif "否" in answer or "没有" in answer or answer == "无":
                            symptom["value"] = "无"
                        else:
                            symptom["value"] = answer

            # After the user answers the two confirmation questions, move directly
            # to generation. The selected symptoms plus confirmation answers are
            # the scoped input for this interaction; do not restart ten inquiry.
            confirmed_dims = {
                s.get("dimension"): s.get("value")
                for s in all_symptoms
                if s.get("value") not in ["待确认", "待进一步确认", ""]
            }
            is_complete = True
            session_data["current_phase"] = "generation"

            # Format confirmed symptoms
            confirmed_text = "\n".join([
                f"- {dim}：{value}"
                for dim, value in confirmed_dims.items()
            ]) or "无"

            response_content = f"""好的，已确认您的症状信息。

【已收集的症状】：
{confirmed_text}

感谢您的配合，正在为您生成病例..."""

        elif current_phase == "inquiry":
            # Fallback to original inquiry for additional dimensions
            from ..agents.inquiry_node import generate_inquiry_message

            state: AgentState = {
                "messages": messages,
                "symptoms_list": all_symptoms,
                "is_complete": False,
                "reflection_count": 0,
                "thread_id": thread_id,
                "current_phase": "inquiry",
                "user_intention": intention,
                "intention_summary": session_data.get("intention_summary", ""),
                "intention_info": session_data.get("intention_info", {}),
                "memory_context": session_data.get("memory_context", ""),
                "memory_agent": session_data.get("memory_agent", {}),
                "retrieved_context": "",
            }

            response_msg = generate_inquiry_message(state, retriever)
            response_content = response_msg.content if hasattr(response_msg, 'content') else str(response_msg)

            # Check completion
            confirmed_dims = {
                s.get("dimension"): s.get("value")
                for s in all_symptoms
                if s.get("value") not in ["待确认", "待进一步确认", ""]
            }
            if len(confirmed_dims) >= 6:
                session_data["current_phase"] = "generation"

        elif current_phase == "generation":
            # Already in generation phase, just return a placeholder
            response_content = "正在生成病例，请稍候..."

        else:
            # Default: show options
            options_text = format_ten_inquiry_options()
            intention_desc = INTENTION_DESCRIPTIONS.get(intention, "初诊问诊")
            response_content = f"""【已理解您的意图】{intention_desc}

{options_text}

请回复您有的症状编号（如：1,3,5）。"""

        # Strip think tags
        response_content = strip_think_tags(response_content)

        # Determine final phase
        final_phase = session_data.get("current_phase", "options")
        is_complete = final_phase == "generation"

        logger.info(f"Processed turn for thread {thread_id}, phase: {final_phase}, intention: {intention}")

        return response_content, final_phase, is_complete, all_symptoms

    except Exception as e:
        logger.error(f"Processing error: {e}")
        import traceback
        traceback.print_exc()

        # Fallback
        from ..agents.options_node import format_ten_inquiry_options
        options_text = format_ten_inquiry_options()
        return f"""欢迎！请选择您有的症状：

{options_text}

请回复您有的症状编号。""", "options", False, symptoms_list


def _extract_symptoms_from_text(text: str) -> list[dict]:
    """Extract symptoms from text using keyword matching."""
    symptoms = []
    text_lower = text.lower()

    negation_keywords = ["没有", "无", "不曾", "不会", "不有", "否认", "不是", "没啥", "不清楚"]
    has_negation = any(neg in text_lower for neg in negation_keywords)

    symptom_keywords = {
        "寒热": ["怕冷", "畏寒", "发热", "怕热", "寒热往来", "发烧"],
        "汗": ["出汗", "盗汗", "自汗", "无汗", "多汗"],
        "头身": ["头痛", "头晕", "腰痛", "身痛", "肩背痛", "乏力"],
        "便溏": ["腹泻", "便秘", "大便溏", "大便稀", "大便干"],
        "饮食": ["食欲不振", "食欲不佳", "厌食", "恶心", "呕吐"],
        "胸腹": ["胸闷", "胸痛", "腹胀", "腹痛", "胃痛", "嗳气"],
        "耳目": ["耳鸣", "听力下降", "视力模糊", "眼花"],
        "口渴": ["口干", "口渴", "喜冷饮", "喜热饮", "口苦"],
        "睡眠": ["失眠", "多梦", "嗜睡", "难入睡", "易醒"],
        "舌脉": ["舌苔", "舌色"],
    }

    for dimension, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                if has_negation:
                    symptoms.append({"dimension": dimension, "value": "无"})
                else:
                    symptoms.append({"dimension": dimension, "value": keyword})
                break

    return symptoms


def _parse_symptom_selections(text: str) -> list[dict]:
    """Parse user's symptom selection in format 'dimension: value; dimension: value'.

    Examples:
    - "寒热: 畏寒; 头身: 头痛"
    - "寒热:畏寒;汗:盗汗"
    """
    import re
    text = text.strip()

    # Pattern to match "dimension: value" pairs
    # Dimensions: 寒热, 汗, 头身, 便溏, 饮食, 胸腹, 耳目, 口渴, 睡眠, 舌脉
    dimension_pattern = r'(寒热|汗|头身|便溏|饮食|胸腹|耳目|口渴|睡眠|舌脉)'
    separator_pattern = r'[,，;；\s]+'

    # Try to find dimension: value patterns
    pattern = rf'{dimension_pattern}\s*[:：]\s*([^;,；]+)'
    matches = re.findall(pattern, text)

    if not matches:
        # Fallback: try to parse option numbers
        return []

    symptoms = []
    for dim, value in matches:
        value = value.strip()
        if value:
            symptoms.append({"dimension": dim, "value": value})

    return symptoms


def _parse_option_selection(text: str) -> list[int]:
    """Parse user's option selection (e.g., '1,3,5' or '1 3 5' or '1,3, 5')."""
    import re
    text = text.strip()

    # Try to find number patterns
    # Match: 1,3,5 or 1 3 5 or 1.3.5 or 1，3，5 (Chinese comma)
    pattern = r'[\d,，\s\.]+'
    match = re.search(pattern, text)

    if not match:
        return []

    selection_str = match.group()
    # Extract numbers
    numbers = re.findall(r'\d+', selection_str)
    return [int(n) for n in numbers if 1 <= int(n) <= 10]


@router.post("/{thread_id}/message")
async def send_message(thread_id: str, request: dict, current_user: dict = Depends(get_current_user)):
    """Send message and stream response."""
    if thread_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[thread_id]
    if session.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    user_content = request.get("content", "")

    async def event_generator():
        try:
            queued_message = _message_queue.enqueue(thread_id, current_user["id"], user_content)
            next_message = _message_queue.dequeue(thread_id) or queued_message
            session["memory_agent"] = load_user_memory(current_user["id"], next_message.content)
            session["memory_context"] = session["memory_agent"].get("memory_context", "")

            # Add user message to session
            session["messages"] = session.get("messages", []) + [HumanMessage(content=next_message.content)]

            logger.info(f"Processing message for thread {thread_id}")

            # Send thinking status - now with intention understanding
            yield {
                "event": "thinking",
                "data": json.dumps({"status": "thinking", "message": "正在理解您的意图并分析症状..."})
            }

            # Run processing in thread pool
            loop = asyncio.get_event_loop()
            full_content, phase, is_complete, all_symptoms = await loop.run_in_executor(
                _executor,
                _process_turn_sync,
                session["messages"],
                session.get("symptoms_list", []),
                thread_id,
                session
            )

            # Update session with new symptoms (content already stripped of think tags)
            session["symptoms_list"] = all_symptoms
            session["messages"] = session["messages"] + [AIMessage(content=full_content)]
            append_memory_event(
                current_user["id"],
                thread_id,
                next_message.content,
                full_content,
                all_symptoms,
            )
            session["memory_agent"] = load_user_memory(current_user["id"], next_message.content)
            session["memory_context"] = session["memory_agent"].get("memory_context", "")

            # Clean previous message content while preserving human/AI roles.
            cleaned_messages = []
            for msg in session["messages"]:
                if not hasattr(msg, "content"):
                    cleaned_messages.append(msg)
                    continue
                content = strip_think_tags(msg.content)
                if getattr(msg, "type", None) == "human":
                    cleaned_messages.append(HumanMessage(content=content))
                elif getattr(msg, "type", None) == "ai":
                    cleaned_messages.append(AIMessage(content=content))
                else:
                    cleaned_messages.append(msg)
            session["messages"] = cleaned_messages

            session["is_complete"] = is_complete

            # Send message
            yield {
                "event": "message",
                "data": json.dumps({
                    "content": full_content,
                    "is_complete": is_complete,
                    "phase": phase
                })
            }

            yield {
                "event": "complete",
                "data": json.dumps({
                    "content": full_content,
                    "is_complete": is_complete,
                    "phase": phase
                })
            }

        except Exception as e:
            logger.error(f"Event generator error: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }

    return EventSourceResponse(event_generator())


@router.get("/{thread_id}/case")
async def get_case(thread_id: str, current_user: dict = Depends(get_current_user)):
    """Get generated case."""
    if thread_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[thread_id]
    if session.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Forbidden")

    if not session.get("is_complete"):
        raise HTTPException(status_code=400, detail="Consultation not yet complete")

    from ..agents.diagnosis_validator import validate_diagnosis_report
    from ..agents.generator_node import generate_case_text, sanitize_case_text
    from ..agents.intention_agent import IntentionType

    retriever = _get_retriever()

    state = {
        "messages": session.get("messages", []),
        "symptoms_list": session.get("symptoms_list", []),
        "is_complete": True,
        "reflection_count": 0,
        "thread_id": thread_id,
        "current_phase": "complete",
        "user_intention": session.get("user_intention", IntentionType.FIRST_VISIT),
        "intention_summary": session.get("intention_summary", ""),
        "intention_info": session.get("intention_info", {}),
        "memory_context": session.get("memory_context", ""),
        "memory_agent": session.get("memory_agent", {}),
        "retrieved_context": "",
    }

    case_text = sanitize_case_text(strip_think_tags(generate_case_text(state, retriever, session.get("user_name", "匿名"))))
    validation = validate_diagnosis_report(case_text, session.get("symptoms_list", []), session.get("messages", []))
    if not session.get("experience_saved"):
        append_experience_event(
            current_user["id"],
            thread_id,
            session.get("symptoms_list", []),
            case_text,
            validation,
        )
        session["experience_saved"] = True

    return {"case": case_text}
