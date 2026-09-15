from __future__ import annotations

from types import SimpleNamespace

from elenchix.llm import OpenAICompatibleLLM


def test_streamed_content_keeps_visible_answer_and_ignores_reasoning() -> None:
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, reasoning_content="internal")
                )
            ]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="请先", reasoning_content=None))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="说明依据。", reasoning_content=None))]
        ),
    ]
    assert OpenAICompatibleLLM._streamed_content(chunks) == "请先说明依据。"
