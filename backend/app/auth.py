"""Local user auth and short-term memory storage."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import BASE_DIR


AUTH_STORE_PATH = Path(os.getenv("AUTH_STORE_PATH", str(BASE_DIR / "storage" / "auth_store.json")))
AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-auth-secret-change-me")
TOKEN_TTL_HOURS = int(os.getenv("AUTH_TOKEN_TTL_HOURS", "168"))
MAX_MEMORY_EVENTS = int(os.getenv("MAX_MEMORY_EVENTS", "30"))

_lock = threading.Lock()
_bearer = HTTPBearer(auto_error=False)


def _utc_now() -> datetime:
    return datetime.utcnow()


def _load_store() -> dict:
    if not AUTH_STORE_PATH.exists():
        return {"users": {}}
    try:
        return json.loads(AUTH_STORE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"users": {}}


def _save_store(store: dict) -> None:
    AUTH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = AUTH_STORE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(AUTH_STORE_PATH)


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return base64.urlsafe_b64encode(digest).decode("ascii")


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _sign(payload: str) -> str:
    return hmac.new(AUTH_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _encode_token(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{encoded}.{_sign(encoded)}"


def _decode_token(token: str) -> dict:
    try:
        encoded, signature = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    if not hmac.compare_digest(_sign(encoded), signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    expires_at = datetime.fromisoformat(payload.get("exp", "1970-01-01T00:00:00"))
    if expires_at < _utc_now():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    return payload


def create_user(username: str, password: str) -> dict:
    normalized = _normalize_username(username)
    if len(normalized) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    with _lock:
        store = _load_store()
        if normalized in store["users"]:
            raise HTTPException(status_code=409, detail="Username already exists")

        salt = secrets.token_urlsafe(16)
        user = {
            "id": str(secrets.token_hex(16)),
            "username": username.strip(),
            "username_normalized": normalized,
            "password_hash": _hash_password(password, salt),
            "salt": salt,
            "created_at": _utc_now().isoformat(),
            "memory": {"events": []},
        }
        store["users"][normalized] = user
        _save_store(store)
        return _public_user(user)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    normalized = _normalize_username(username)
    with _lock:
        user = _load_store()["users"].get(normalized)
    if not user:
        return None
    expected = _hash_password(password, user["salt"])
    if not hmac.compare_digest(expected, user["password_hash"]):
        return None
    return user


def create_access_token(user: dict) -> str:
    payload = {
        "sub": user["id"],
        "username": user["username"],
        "exp": (_utc_now() + timedelta(hours=TOKEN_TTL_HOURS)).isoformat(),
    }
    return _encode_token(payload)


def _public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "username": user["username"],
        "created_at": user.get("created_at"),
    }


def get_user_by_id(user_id: str) -> Optional[dict]:
    with _lock:
        users = _load_store()["users"].values()
        for user in users:
            if user.get("id") == user_id:
                return user
    return None


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = _decode_token(credentials.credentials)
    user = get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return _public_user(user)


def get_memory_context(user_id: str) -> str:
    user = get_user_by_id(user_id)
    if not user:
        return ""
    events = user.get("memory", {}).get("events", [])[-8:]
    lines = []
    for event in events:
        user_text = event.get("user_message", "")
        assistant_text = event.get("assistant_summary") or event.get("assistant_message", "")
        symptoms = event.get("symptoms", [])
        symptom_text = "、".join(
            f"{item.get('dimension')}:{item.get('value')}"
            for item in symptoms[-5:]
            if item.get("dimension") and item.get("value")
        )
        parts = []
        if user_text:
            parts.append(f"用户曾说：{user_text[:120]}")
        if symptom_text:
            parts.append(f"已记录症状：{symptom_text}")
        if assistant_text:
            parts.append(f"助手曾回复：{assistant_text[:160]}")
        if parts:
            lines.append("；".join(parts))
    return "\n".join(lines)


def get_memory_snapshot(user_id: str) -> dict:
    """Return raw short-term memory and experience items for memory agents."""
    user = get_user_by_id(user_id)
    if not user:
        return {"events": [], "experiences": []}
    memory = user.get("memory", {})
    return {
        "events": list(memory.get("events", [])),
        "experiences": list(memory.get("experiences", [])),
    }


def get_consultation_records(user_id: str, limit: int = 20) -> list[dict]:
    """Return user-facing previous consultation records."""
    snapshot = get_memory_snapshot(user_id)
    events = snapshot.get("events", [])
    experiences = snapshot.get("experiences", [])

    experience_by_thread = {
        item.get("thread_id"): item
        for item in experiences
        if item.get("thread_id")
    }

    grouped: dict[str, dict] = {}
    for event in events:
        thread_id = event.get("thread_id")
        if not thread_id:
            continue
        record = grouped.setdefault(thread_id, {
            "thread_id": thread_id,
            "created_at": event.get("created_at"),
            "updated_at": event.get("created_at"),
            "messages": [],
            "symptoms": [],
            "diagnosis_summary": "",
            "validation_confidence": None,
        })
        record["updated_at"] = event.get("created_at") or record["updated_at"]
        record["messages"].append({
            "user_message": event.get("user_message", ""),
            "assistant_summary": event.get("assistant_summary") or event.get("assistant_message", "")[:180],
            "created_at": event.get("created_at"),
        })
        for symptom in event.get("symptoms", []):
            dim = symptom.get("dimension")
            value = symptom.get("value")
            if not dim or not value:
                continue
            if not any(item.get("dimension") == dim and item.get("value") == value for item in record["symptoms"]):
                record["symptoms"].append({"dimension": dim, "value": value})

    for thread_id, experience in experience_by_thread.items():
        record = grouped.setdefault(thread_id, {
            "thread_id": thread_id,
            "created_at": experience.get("created_at"),
            "updated_at": experience.get("created_at"),
            "messages": [],
            "symptoms": [],
            "diagnosis_summary": "",
            "validation_confidence": None,
        })
        record["diagnosis_summary"] = experience.get("diagnosis_summary", "")
        record["validation_confidence"] = experience.get("validation_confidence")
        if experience.get("symptom_summary") and not record["symptoms"]:
            record["symptoms"] = [
                {"dimension": part.split("：", 1)[0], "value": part.split("：", 1)[1]}
                for part in experience["symptom_summary"].split("；")
                if "：" in part
            ]

    records = sorted(
        grouped.values(),
        key=lambda item: item.get("updated_at") or item.get("created_at") or "",
        reverse=True,
    )
    return records[:limit]


def append_memory_event(user_id: str, thread_id: str, user_message: str, assistant_message: str, symptoms: list[dict]) -> None:
    with _lock:
        store = _load_store()
        target_key = None
        for key, user in store["users"].items():
            if user.get("id") == user_id:
                target_key = key
                break
        if target_key is None:
            return

        user = store["users"][target_key]
        memory = user.setdefault("memory", {"events": []})
        events = memory.setdefault("events", [])
        events.append({
            "thread_id": thread_id,
            "created_at": _utc_now().isoformat(),
            "user_message": user_message,
            "assistant_message": assistant_message[:1200],
            "assistant_summary": _summarize_assistant_message(assistant_message),
            "symptoms": symptoms[-10:],
        })
        memory["events"] = events[-MAX_MEMORY_EVENTS:]
        _save_store(store)


def _summarize_assistant_message(message: str) -> str:
    """Keep memory compact and focused on reusable clinical context."""
    if not message:
        return ""
    lines = []
    for raw_line in message.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(marker in line for marker in ["已记录", "已收集", "诊断", "辨证", "治则", "建议", "确认"]):
            lines.append(line)
        if len("；".join(lines)) > 220:
            break
    return "；".join(lines)[:260] if lines else message[:180]


def append_experience_event(
    user_id: str,
    thread_id: str,
    symptoms: list[dict],
    case_text: str,
    validation: Optional[dict] = None,
) -> None:
    """Persist compact final-case experience for future follow-up retrieval."""
    symptom_summary = "；".join(
        f"{item.get('dimension')}：{item.get('value')}"
        for item in symptoms
        if item.get("dimension") and item.get("value") not in ["", "待确认", "待进一步确认"]
    )

    diagnosis_summary = ""
    for marker in ["## 辨证", "## 治则", "## 诊断验证"]:
        if marker in case_text:
            diagnosis_summary += case_text.split(marker, 1)[1][:240] + " "

    with _lock:
        store = _load_store()
        target_key = None
        for key, user in store["users"].items():
            if user.get("id") == user_id:
                target_key = key
                break
        if target_key is None:
            return

        user = store["users"][target_key]
        memory = user.setdefault("memory", {"events": []})
        experiences = memory.setdefault("experiences", [])
        experiences.append({
            "thread_id": thread_id,
            "created_at": _utc_now().isoformat(),
            "symptom_summary": symptom_summary,
            "diagnosis_summary": diagnosis_summary.strip(),
            "validation_confidence": (validation or {}).get("confidence"),
        })
        memory["experiences"] = experiences[-MAX_MEMORY_EVENTS:]
        _save_store(store)
