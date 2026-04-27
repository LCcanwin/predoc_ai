"""Diagnosis validation agent backed by deterministic rules."""

import re

from .rules_engine import build_rule_context


REQUIRED_SECTIONS = ["基本信息", "主诉", "现病史", "十问歌", "辨证", "治则"]


def validate_diagnosis_report(case_text: str, symptoms_list: list[dict], messages: list) -> dict:
    """Validate generated report against required structure and rule context."""
    missing_sections = [
        section
        for section in REQUIRED_SECTIONS
        if not re.search(rf"^##\s*{re.escape(section)}\s*$", case_text, flags=re.MULTILINE)
    ]
    rule_context = build_rule_context(symptoms_list, messages)

    unsupported_claims = []
    guarded_terms = ["无特殊", "否认", "无明显", "健康"]
    for term in guarded_terms:
        if term in case_text:
            unsupported_claims.append(f"报告包含可能表示已排除病史的表述：{term}")

    warnings = []
    if missing_sections:
        warnings.append(f"报告缺少结构化章节：{'、'.join(missing_sections)}")
    warnings.extend(unsupported_claims)
    warnings.extend(rule_context.get("red_flags", []))

    confidence = 0.82
    if missing_sections:
        confidence -= 0.2
    if unsupported_claims:
        confidence -= 0.15
    if rule_context.get("missing_dimensions"):
        confidence -= min(0.25, len(rule_context["missing_dimensions"]) * 0.03)
    confidence = max(0.2, round(confidence, 2))

    return {
        "is_valid": not missing_sections and not unsupported_claims,
        "confidence": confidence,
        "warnings": warnings,
        "rule_context": rule_context,
    }


def append_validation_summary(case_text: str, validation: dict) -> str:
    """Append concise validation output to the report."""
    warnings = validation.get("warnings", [])
    rule_context = validation.get("rule_context", {})
    warning_text = "\n".join(f"- {item}" for item in warnings) if warnings else "- 未发现明显结构化问题。"

    summary = f"""

## 诊断验证
规则引擎参考证型：{rule_context.get("pattern", "需进一步辨证")}
规则引擎参考治则：{rule_context.get("treatment_principle", "根据辨证结果确定治则")}
验证置信度：{validation.get("confidence", 0)}

验证提示：
{warning_text}
"""
    return case_text.rstrip() + "\n" + summary
