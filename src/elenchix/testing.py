"""Interactive live-test launcher with credential-safe configuration checks."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

from elenchix.config import AppConfig, load_config
from elenchix.errors import AssessmentFailedError, NoAvailableCasesError
from elenchix.graph import create_graph_store
from elenchix.identifiers import english_topic
from elenchix.llm import OpenAICompatibleLLM
from elenchix.schemas import SessionResult
from elenchix.terminal import Spinner, TerminalProgress, print_planning
from elenchix.workflow import ElenchiXSession


def safe_error(exc: Exception) -> str:
    text = str(exc)
    for name, value in os.environ.items():
        if (
            value
            and len(value) >= 4
            and any(part in name.upper() for part in ("KEY", "PASSWORD", "TOKEN", "SECRET"))
        ):
            text = text.replace(value, "[redacted]")
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", text)
    return f"{type(exc).__name__}: {text[:800]}"


def preflight(config: AppConfig, *, topic: str | None = None, probe_api: bool = False) -> dict:
    from elenchix.agents.planning_tools import PlanningToolbox

    for role in (config.llm.planning, config.llm.teaching, config.llm.assessment):
        config.llm.credentials(role)
    graph = create_graph_store(config.graph)
    try:
        cases = list(graph.list_cases())
        abilities = list(graph.list_abilities())
        knowledge = list(graph.list_knowledge())
        matching = [
            case for case in cases if not topic or PlanningToolbox.matches_topic(case, topic)
        ]
        links = sum(len(graph.ability_knowledge(node.id)) for node in abilities)
        if not matching:
            raise ValueError("No cases match this topic. Check the topic or graph configuration.")
        if len(abilities) != 20 or links == 0:
            raise ValueError(
                "Live testing requires all 20 abilities and valid Ability-to-KP links."
            )
        if any(not graph.case_targets(case.id) for case in matching):
            raise ValueError("A candidate case has no teaching targets.")
        report = {
            "cases": len(cases),
            "knowledge": len(knowledge),
            "abilities": len(abilities),
            "ability_knowledge_links": links,
            "matching_cases": len(matching),
        }
    finally:
        graph.close()
    if probe_api:
        llm = OpenAICompatibleLLM(config.llm)
        checked = set()
        for role, settings in llm.roles.items():
            key, endpoint = config.llm.credentials(settings)
            identity = (endpoint, settings.model, key)
            if identity not in checked:
                request = {
                    "model": settings.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "user", "content": "Connection check. Reply only with OK."}
                    ],
                    "max_tokens": 32,
                }
                if settings.extra_body:
                    body = dict(settings.extra_body)
                    if "enable_thinking" in body:
                        body["enable_thinking"] = False
                    request["extra_body"] = body
                response = (
                    llm.clients[role]
                    .with_options(timeout=30, max_retries=0)
                    .chat.completions.create(**request)
                )
                if not response.choices or not (response.choices[0].message.content or "").strip():
                    raise RuntimeError(f"The {role} API returned an empty response.")
                checked.add(identity)
            report[role] = "API OK"
    return report


def _print_result(result: SessionResult) -> None:
    events = result.assessment.events
    knowledge = {event.target_id for event in events if event.target_type == "knowledge"}
    abilities = {event.target_id for event in events if event.target_type == "assessment_point"}
    print("\n── Session feedback ──")
    feedback = " ".join(result.assessment.feedback.split())
    print(feedback if len(feedback) <= 240 else feedback[:240] + "…")
    if events:
        positive = sum(event.score > 0 for event in events)
        zero = sum(event.score == 0 for event in events)
        negative = len(events) - positive - zero
        print(
            f"Assessed: {len(knowledge)} knowledge points / {len(abilities)} abilities. "
            f"Scores: {positive} positive / {zero} zero / {negative} negative."
        )
    else:
        print("No assessable evidence in this session.")
    status = result.model_update.get("status")
    messages = {
        "no_new_evidence": "No new score evidence; the model is unchanged.",
        "fitted": f"Learner model updated (version {result.model_update.get('model_version', '?')}).",
        "awaiting_prior_sequences": "Evidence recorded. Fitting will begin once later cases provide training sequences.",
        "disabled": "Evidence recorded; online fitting is disabled.",
        "failed": "Evidence recorded. Model update failed; it will be retried before the next case.",
    }
    print(messages.get(status, "Session complete."))


def _export_result(result: SessionResult, output_dir: Path | None, round_index: int) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"round-{round_index}-{uuid4().hex[:12]}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Full results: {path.resolve()}")


def _print_round_opening(active, *, show_planning: bool = False) -> None:
    if show_planning:
        print_planning(active.plan, active.planning_trace)
    print(f"\nTutor: {active.last_teaching.tutor_message}")


def _interact(config: AppConfig, args, learner: str, topic: str | None) -> int:
    progress = TerminalProgress()
    with Spinner("Loading the learner model"):
        session = ElenchiXSession(config, offline=args.offline, on_progress=progress)
    try:
        active = session.start_round(
            learner,
            topic=topic,
            group=args.group,
        )
        _print_round_opening(active, show_planning=args.show_planning)
        print("Reply to continue; /finish to end this case; /quit to exit.")
        while True:
            learner_message = input("\nYou: ").strip()
            if not learner_message:
                continue
            if learner_message == "/quit":
                return 0
            if learner_message.split(maxsplit=1)[0] == "/finish":
                if learner_message != "/finish":
                    print("Enter /finish on its own to end this case. Send answers separately.")
                    continue
                try:
                    result = session.finish(
                        active.learner_id,
                        active.round_index,
                        expected_message_count=active.message_count,
                    )
                except AssessmentFailedError:
                    print(
                        "Assessment could not be completed. Your dialogue is saved. Enter /finish to retry, or continue answering."
                    )
                    continue
                _print_result(result)
                _export_result(result, args.output_dir, active.round_index)
                next_action = input(
                    "\nPress Enter or type /next for another case; /quit to exit: "
                ).strip()
                if next_action == "/quit":
                    return 0
                if next_action not in {"", "/next"}:
                    print("Unrecognized command. Exiting; your completed session is saved.")
                    return 0
                active = session.start_round(
                    learner,
                    topic=topic,
                    group=args.group,
                )
                _print_round_opening(active, show_planning=args.show_planning)
                continue
            active = session.reply(
                active.learner_id,
                active.round_index,
                learner_message,
                expected_message_count=active.message_count,
            )
            print(f"\nTutor: {active.last_teaching.tutor_message}")
    finally:
        progress.close()
        session.close()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stdin, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="ElenchiX interactive agent testing")
    parser.add_argument("--config", type=Path, default=Path("configs/private.testing.yaml"))
    parser.add_argument("--learner-id")
    parser.add_argument("--topic")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--probe-api", action="store_true", help="send a short API connection check"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="check the workflow with synthetic scores; no API calls or persistent data",
    )
    parser.add_argument("--group", help="learner group; defaults to the configured study_group")
    parser.add_argument(
        "--show-planning",
        action="store_true",
        help="show the planning rationale and ordered tool calls",
    )
    parser.add_argument("--output-dir", type=Path, help="export complete session results as JSON")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        topic = args.topic or config.agents.experiment_topic
        if not args.check_only:
            learner = (
                args.learner_id or input("Learner ID [test_learner]: ").strip() or "test_learner"
            )
            if not args.offline:
                topic = (
                    args.topic
                    or input(f"Clinical topic [{english_topic(topic) or 'Any'}]: ").strip()
                    or english_topic(topic)
                )
        if args.offline and args.probe_api:
            raise ValueError("--offline cannot be combined with --probe-api.")
        if not args.offline:
            with Spinner(
                "Checking configuration and graph"
                + (" and API connections" if args.probe_api else "")
            ):
                report = preflight(config, topic=topic, probe_api=args.probe_api)
            if args.check_only:
                print(
                    f"Checks passed: {report['matching_cases']} matching cases / "
                    f"{report['knowledge']} knowledge points / {report['abilities']} abilities."
                )
                for role in ("planning", "teaching", "assessment"):
                    print(f"{role}: {report.get(role, 'configured')}")
        if args.check_only:
            if args.offline:
                with ElenchiXSession(config, offline=True):
                    print("Offline configuration checks passed.")
            return 0
        return _interact(config, args, learner, topic)
    except NoAvailableCasesError as exc:
        print(str(exc))
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nExited. Use the same learner ID to resume saved sessions.")
        return 0
    except Exception as exc:  # noqa: BLE001 - launcher prints only redacted error details
        print("Testing failed: " + safe_error(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
