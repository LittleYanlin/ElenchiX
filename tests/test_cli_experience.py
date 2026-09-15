import json
import re
from pathlib import Path

import pytest

from elenchix.testing import main


def test_cli_defaults_to_brief_output_but_exports_the_full_result(monkeypatch, capsys, tmp_path):
    messages = iter(["我会先追问发病时间", "/finish", "/quit"])
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return next(messages)

    monkeypatch.setattr("builtins.input", answer)
    root = Path(__file__).resolve().parents[1]
    assert (
        main(
            [
                "--learner-id",
                "test_learner",
                "--config",
                str(root / "tests/fixtures/config.yaml"),
                "--offline",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert not re.search(r"[\u4e00-\u9fff]", output.out + output.err + " ".join(prompts))
    assert "Tutor:" in output.out and "Session feedback" in output.out
    assert "Assessed:" in output.out and "Scores:" in output.out
    assert "Tutor is thinking" in output.err and "Full-dialogue assessment" in output.err
    assert "planning_trace" not in output.out and "prediction_snapshot" not in output.out
    assert "Agent call" not in output.out and "Tutor[" not in output.out
    assert "── Round" not in output.out and "final answer" not in output.out
    assert not any("final answer" in prompt for prompt in prompts)
    payload = json.loads(next(tmp_path.glob("round-1-*.json")).read_text(encoding="utf-8"))
    assert len(payload["dialogue"]) == 3
    assert [m["content"] for m in payload["dialogue"] if m["role"] == "learner"] == [
        "我会先追问发病时间"
    ]
    assert all(event["message_ids"] == [2] for event in payload["assessment"]["events"])
    assert payload["planning_trace"] and payload["assessment"]["events"]
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize("status", ["failed", "awaiting_prior_sequences"])
def test_summary_does_not_claim_successful_training_when_no_fit_succeeded(status, capsys):
    from elenchix import ElenchiXSession
    from elenchix.testing import _print_result

    root = Path(__file__).resolve().parents[1]
    with ElenchiXSession.from_config(root / "tests/fixtures/config.yaml", offline=True) as session:
        result = session.run_round("u", 1, "回答")
    result.model_update = {"status": status, "error": "internal sensitive diagnostic"}
    result.assessment.feedback = "反馈" * 500
    _print_result(result)
    output = capsys.readouterr().out
    assert "Learner model updated" not in output
    assert "internal sensitive diagnostic" not in output
    assert len(output) < 450
    assert ("Model update failed" if status == "failed" else "later cases") in output
