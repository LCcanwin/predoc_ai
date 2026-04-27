"""Rule engine used by diagnosis generation and validation."""

from typing import Optional

from .state import TEN_INQUIRY_DIMENSIONS


def symptom_text(symptoms_list: list[dict]) -> str:
    """Return a compact symptom string for matching and logging."""
    return " ".join(
        f"{item.get('dimension')}:{item.get('value')}"
        for item in symptoms_list
        if item.get("dimension") and item.get("value") and item.get("value") != "未记录"
    )


def infer_diagnosis(symptoms_list: list[dict]) -> tuple[str, str, list[str]]:
    """Infer a conservative TCM pattern from collected symptoms."""
    text = symptom_text(symptoms_list)
    evidence: list[str] = []

    if "寒热:畏寒" in text or "寒热:怕冷" in text:
        evidence.append("畏寒或怕冷")
        if "汗:自汗" in text:
            evidence.append("自汗")
            return "阳虚证", "温阳固表，益气助阳", evidence
        return "表寒证", "辛温解表", evidence

    if "寒热:发热" in text or "寒热:怕热" in text:
        evidence.append("发热或怕热")
        if "口渴:喜冷饮" in text:
            evidence.append("喜冷饮")
            return "实热证", "清热泻火", evidence
        return "阴虚证", "滋阴清热", evidence

    if "便溏:腹泻" in text or "便溏:大便溏" in text or "便溏:大便稀" in text:
        evidence.append("腹泻或便溏")
        if "饮食:食欲不振" in text or "饮食:食欲不佳" in text:
            evidence.append("食欲不振")
            return "脾虚湿盛证", "健脾祛湿", evidence
        return "寒湿证", "温中散寒，健脾化湿", evidence

    if "睡眠:失眠" in text or "睡眠:难入睡" in text:
        evidence.append("失眠或难入睡")
        if "口渴:口干" in text:
            evidence.append("口干")
            return "心肾不交证", "滋阴降火，交通心肾", evidence
        return "心脾两虚证", "补益心脾", evidence

    if "饮食:口味偏淡" in text and "睡眠:多梦" in text:
        evidence.extend(["口味偏淡", "多梦"])
        return "脾虚夹心神不宁倾向", "健脾和中，养心安神", evidence

    return "需进一步辨证", "根据补充问诊、舌象和脉象进一步确定治则", evidence


def find_missing_core_dimensions(symptoms_list: list[dict]) -> list[str]:
    """List core dimensions that have not been collected."""
    collected = {
        item.get("dimension")
        for item in symptoms_list
        if item.get("dimension") and item.get("value") not in ["", "待确认", "待进一步确认"]
    }
    return [dim for dim in TEN_INQUIRY_DIMENSIONS if dim not in collected]


def detect_red_flags(messages: list, symptoms_list: list[dict]) -> list[str]:
    """Detect safety signals that should be surfaced in validation."""
    text_parts = []
    for msg in messages:
        if hasattr(msg, "content"):
            text_parts.append(msg.content)
    text_parts.append(symptom_text(symptoms_list))
    text = " ".join(text_parts)

    red_flags = []
    red_flag_terms = {
        "呼吸困难": "出现呼吸困难应及时线下就医",
        "胸痛": "胸痛需要警惕急症风险",
        "高热": "持续高热需要及时就医",
        "便血": "便血需要进一步检查",
        "意识不清": "意识异常需要紧急处理",
        "剧烈": "剧烈疼痛或剧烈不适需要线下评估",
    }
    for term, note in red_flag_terms.items():
        if term in text:
            red_flags.append(note)
    return red_flags


def build_rule_context(symptoms_list: list[dict], messages: Optional[list] = None) -> dict:
    """Build structured rule-engine context for downstream agents."""
    pattern, treatment, evidence = infer_diagnosis(symptoms_list)
    return {
        "pattern": pattern,
        "treatment_principle": treatment,
        "evidence": evidence,
        "missing_dimensions": find_missing_core_dimensions(symptoms_list),
        "red_flags": detect_red_flags(messages or [], symptoms_list),
    }


def format_rule_context(rule_context: dict) -> str:
    """Format rule context for prompt injection."""
    evidence = "、".join(rule_context.get("evidence", [])) or "暂无明确规则证据"
    missing = "、".join(rule_context.get("missing_dimensions", [])[:6]) or "核心维度基本已覆盖"
    red_flags = "；".join(rule_context.get("red_flags", [])) or "未识别到明显急症风险表达"
    return f"""规则引擎初判：{rule_context.get("pattern", "需进一步辨证")}
建议治则：{rule_context.get("treatment_principle", "根据辨证结果确定治则")}
规则证据：{evidence}
仍缺维度：{missing}
安全提示：{red_flags}"""
