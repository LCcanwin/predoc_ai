from langchain_core.messages import AIMessage, HumanMessage

from app.api.routes import (
    SESSION_MESSAGE_WINDOW,
    _compact_session_context,
)


def test_compact_session_context_keeps_recent_window_and_summary():
    session = {
        "messages": [],
        "symptoms_list": [
            {"dimension": "胸腹", "value": "胸痛"},
            {"dimension": "睡眠", "value": "失眠"},
        ],
        "session_summary": "",
    }
    for index in range(10):
        session["messages"].append(HumanMessage(content=f"早期补充{index}，有胸痛和剧烈不适"))
        session["messages"].append(AIMessage(content=f"已记录第{index}轮症状，建议继续确认"))

    _compact_session_context(session)

    assert len(session["messages"]) == SESSION_MESSAGE_WINDOW
    assert session["messages"][0].content == "早期补充4，有胸痛和剧烈不适"
    assert "患者早期补充" in session["session_summary"]
    assert "当前已确认症状：胸腹:胸痛、睡眠:失眠" in session["session_summary"]
    assert "早期安全信号：胸痛、剧烈" in session["session_summary"]


def test_compact_session_context_does_not_touch_short_sessions():
    messages = [HumanMessage(content="头痛"), AIMessage(content="已记录头痛")]
    session = {
        "messages": list(messages),
        "symptoms_list": [{"dimension": "头身", "value": "头痛"}],
        "session_summary": "",
    }

    _compact_session_context(session)

    assert session["messages"] == messages
    assert session["session_summary"] == ""
