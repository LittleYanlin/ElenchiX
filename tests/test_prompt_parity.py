from hashlib import sha256

from elenchix.prompts import (
    EVALUATION_PROMPT_TEMPLATE,
    PLANNING_PROMPT_TEMPLATE,
    TEACHING_SYSTEM_PROMPT_TEMPLATE,
)


def test_finetuned_teaching_prompt_is_preserved():
    assert sha256(TEACHING_SYSTEM_PROMPT_TEMPLATE.encode()).hexdigest() == (
        "b06562178e1672e8ed9124b3edae0b5bf9d2be074678f5b1c490a5ab45bfbe22"
    )


def test_assessment_uses_paper_rubric_and_complete_context():
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        case_context="CASE_FULL",
        teaching_plan="PLAN_FULL",
        knowledge_points="TARGET_IDS",
        conversation_history="ALL_MESSAGES",
    )
    for required in [
        "CASE_FULL",
        "PLAN_FULL",
        "TARGET_IDS",
        "ALL_MESSAGES",
        "[0.80,1.00]",
        "[0.60,0.80)",
        "[0.35,0.60)",
        "[0.10,0.35)",
        "(0,0.10)",
        "[-0.10,0)",
        "[-0.35,-0.10)",
        "[-0.65,-0.35)",
        "[-1.00,-0.65)",
        "message_ids",
        "mechanical_repetition",
        "prompted_omission",
        "全体20维",
        "不用零分或负分代替缺失",
        "学生态度",
    ]:
        assert required in prompt
    for template in [EVALUATION_PROMPT_TEMPLATE, PLANNING_PROMPT_TEMPLATE]:
        assert "给药时机" in template and "禁忌证" in template
