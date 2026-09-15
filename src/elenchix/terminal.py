"""Terminal-only presentation; the public session API never uses these helpers."""

from __future__ import annotations

import itertools
import json
import sys
import time
import unicodedata
from threading import Event, Thread
from typing import ClassVar, Self, TextIO

from elenchix.progress import ProgressEvent


def print_planning(plan, trace) -> None:
    """Show ordered invocations; full results remain in the persisted round record."""
    if plan is None:
        print("Planning: rationale and tool trace were not saved in this older record.")
        return
    print(f"\nCase: {plan.case_id}\nPlanning rationale: {plan.rationale}")
    labels = {"react": "Agent call", "required_context": "System context", "offline": "Offline workflow"}
    for index, call in enumerate(trace, 1):
        arguments = json.dumps(call.arguments, ensure_ascii=False, default=str)
        result = call.result
        if isinstance(result, dict) and "total" in result:
            summary = f"{result['total']} items, page {result.get('page', 1)}"
            if "cases" in result:
                summary += "; cases " + ", ".join(row["case_id"] for row in result["cases"])
        elif isinstance(result, list):
            summary = f"{len(result)} records returned"
        elif isinstance(result, dict) and "case" in result:
            summary = f"Case {result['case']['id']}; {len(result.get('targets', []))} targets"
        else:
            summary = json.dumps(result, ensure_ascii=False, default=str)
            if len(summary) > 160:
                summary = summary[:160] + "…"
        status = "Failed" if getattr(call, "status", "success") == "error" else "Done"
        print(f"  {index}. [{labels.get(call.source, call.source)}] {call.tool_name}({arguments})")
        print(f"     {status} · {summary}")


class Spinner:
    """Animate a single waiting line on a TTY; emit one plain line when redirected."""

    def __init__(self, message: str, *, stream: TextIO | None = None):
        self.message = message
        self.stream = stream if stream is not None else sys.stderr
        self._stop = Event()
        self._thread: Thread | None = None
        self._width = 0

    def __enter__(self) -> Self:
        if not self.stream.isatty():
            self.stream.write(self.message + "...\n")
            self.stream.flush()
            return self
        self._started = time.monotonic()
        self._draw("◐")
        self._thread = Thread(target=self._animate, name="elenchix-spinner", daemon=True)
        self._thread.start()
        return self

    def _draw(self, frame: str) -> None:
        text = f"{frame} {self.message}  {int(time.monotonic() - self._started)}s"
        width = sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)
        self._width = max(self._width, width)
        self.stream.write("\r" + text)
        self.stream.flush()

    def _animate(self) -> None:
        for frame in itertools.cycle("◓◑◒◐"):
            if self._stop.wait(0.12):
                return
            self._draw(frame)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self.stream.write("\r" + " " * self._width + "\r")
            self.stream.flush()


class TerminalProgress:
    """Render actual workflow stages, retaining a short receipt for assessment and refits."""

    LABELS: ClassVar[dict[str, str]] = {
        "planning": "Planning the next case",
        "teaching": "Tutor is thinking",
        "assessment": "Full-dialogue assessment",
        "akt_refit": "AKT refitting",
        "adapter_refit": "Graph adapter fitting",
        "saving": "Saving session results",
    }
    SKIP_REASONS: ClassVar[dict[str, str]] = {
        "no_new_evidence": "no new score evidence; keeping the current model",
        "disabled": "online fitting is disabled",
        "awaiting_prior_sequences": "no training sequences across cases yet; more cases are needed",
        "no_knowledge_observations": "no knowledge-point scores yet; keeping initial parameters",
    }

    def __init__(self, *, stream: TextIO | None = None):
        self.stream = stream if stream is not None else sys.stderr
        self._spinner: Spinner | None = None

    def close(self) -> None:
        if self._spinner is not None:
            self._spinner.__exit__(None, None, None)
            self._spinner = None

    def __call__(self, event: ProgressEvent) -> None:
        self.close()
        label = self.LABELS[event.phase]
        if event.phase == "saving" and not event.details.get("persistent", True):
            label = "Recording results (in memory)"
        if event.status == "started":
            self._spinner = Spinner(label, stream=self.stream)
            self._spinner.__enter__()
        elif event.status == "skipped":
            reason = self.SKIP_REASONS.get(str(event.details.get("reason")), "not needed for this session")
            self.stream.write(f"– {label}: {reason}.\n")
        elif event.status == "failed":
            self.stream.write(f"! {label} failed ({event.elapsed_seconds:.1f}s).\n")
        elif event.phase not in {"planning", "teaching"}:
            self.stream.write(f"✓ {label} completed ({event.elapsed_seconds:.1f}s).\n")
        self.stream.flush()
