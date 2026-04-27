"""Follow-up and history memory agent."""

import re
from typing import Optional

from langchain_core.messages import AIMessage

from ..auth import get_memory_context, get_memory_snapshot
from .intention_agent import IntentionType

FOLLOW_UP_TERMS = ["复诊", "上次", "之前", "继续", "还是", "又", "吃了药", "用药后", "没效", "好转"]
EMPTY_VALUES = {"", "无", "未提及", "待确认", "待进一步确认", "待补充"}


def load_user_memory(user_id: str, current_message: str = "") -> dict:
    """Load short-term and experience memory for the current user."""
    snapshot = get_memory_snapshot(user_id)
    memory_context = get_memory_context(user_id)
    relevant_events = _rank_relevant_events(snapshot.get("events", []), current_message)
    experience_context = _format_experiences(snapshot.get("experiences", [])[-5:])
    return {
        "memory_context": memory_context,
        "relevant_events": relevant_events,
        "suggested_symptoms": _extract_symptoms_from_events(relevant_events),
        "experience_context": experience_context,
        "follow_up_hint": _detect_follow_up_hint(current_message, relevant_events),
    }


def build_memory_message(memory: dict) -> Optional[AIMessage]:
    """Build an AIMessage carrying memory context for intention classification."""
    sections = []
    if memory.get("memory_context"):
        sections.append("【用户短期记忆】\n" + memory["memory_context"])
    if memory.get("experience_context"):
        sections.append("【历史经验摘要】\n" + memory["experience_context"])
    if memory.get("follow_up_hint"):
        sections.append("【复诊提示】\n" + memory["follow_up_hint"])
    if not sections:
        return None
    return AIMessage(content="\n\n".join(sections))


def enrich_intention_info(intention: str, info: dict, memory: dict) -> tuple[str, dict]:
    """Use memory signals to strengthen follow-up classification."""
    updated = dict(info or {})
    if memory.get("follow_up_hint"):
        updated["memory_follow_up_hint"] = memory["follow_up_hint"]
        if intention == IntentionType.FIRST_VISIT:
            intention = IntentionType.FOLLOW_UP
            updated["previous_case_exists"] = True
            updated["confidence"] = max(float(updated.get("confidence", 0.0)), 0.72)
    return intention, updated


def hydrate_symptoms_from_memory(
    current_symptoms: list[dict],
    memory: dict,
    intention: str,
) -> tuple[list[dict], list[dict]]:
    """Merge relevant historical symptoms into a follow-up consultation."""
    if intention != IntentionType.FOLLOW_UP and not memory.get("follow_up_hint"):
        return current_symptoms, []

    merged = list(current_symptoms)
    existing_dims = {
        item.get("dimension")
        for item in merged
        if item.get("dimension") and item.get("value") not in EMPTY_VALUES
    }

    added = []
    for symptom in memory.get("suggested_symptoms", []):
        dim = symptom.get("dimension")
        value = symptom.get("value")
        if not dim or value in EMPTY_VALUES or dim in existing_dims:
            continue
        hydrated = {
            "dimension": dim,
            "value": value,
            "source": "memory",
        }
        merged.append(hydrated)
        added.append(hydrated)
        existing_dims.add(dim)

    return merged, added


def _rank_relevant_events(events: list[dict], query: str) -> list[dict]:
    if not events:
        return []

    is_follow_up = any(term in query for term in FOLLOW_UP_TERMS)
    terms = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", query))
    if not terms:
        return events[-3:]

    scored = []
    for event in events:
        text = f"{event.get('user_message', '')} {event.get('assistant_message', '')}"
        for symptom in event.get("symptoms", []):
            text += f" {symptom.get('dimension', '')} {symptom.get('value', '')}"
        score = sum(1 for term in terms if term and term in text)
        if score:
            scored.append((score, event))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [event for _, event in scored[:3]]
    if is_follow_up:
        return list(reversed(events[-3:]))
    return []


def _extract_symptoms_from_events(events: list[dict]) -> list[dict]:
    symptoms = []
    seen_dims = set()
    for event in reversed(events):
        for item in reversed(event.get("symptoms", [])):
            dim = item.get("dimension")
            value = item.get("value")
            if not dim or value in EMPTY_VALUES or dim in seen_dims:
                continue
            symptoms.append({"dimension": dim, "value": value})
            seen_dims.add(dim)
    return symptoms[:8]


def _format_experiences(experiences: list[dict]) -> str:
    lines = []
    for item in experiences:
        diagnosis = item.get("diagnosis_summary", "")
        symptoms = item.get("symptom_summary", "")
        if diagnosis or symptoms:
            lines.append(f"症状：{symptoms[:120]}；结论：{diagnosis[:160]}")
    return "\n".join(lines)


def _detect_follow_up_hint(message: str, events: list[dict]) -> str:
    if any(term in message for term in FOLLOW_UP_TERMS) and events:
        latest = events[0]
        return f"用户表达复诊/延续咨询意图，最近相关记录：{latest.get('user_message', '')[:120]}"
    return ""
