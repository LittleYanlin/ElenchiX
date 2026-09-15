"""Optional structured progress notifications, independent of terminal or web frameworks."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Phase = Literal["planning", "teaching", "assessment", "akt_refit", "adapter_refit", "saving"]


class ProgressEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: Phase
    status: Literal["started", "completed", "skipped", "failed"]
    elapsed_seconds: float = 0.0
    details: dict[str, int | str | bool] = Field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]


class ProgressReporter:
    def __init__(self, callback: ProgressCallback | None = None):
        self.callback = callback

    def emit(self, event: ProgressEvent) -> None:
        if self.callback is not None:
            try:
                self.callback(event)
            except Exception:
                # A disconnected presentation layer must not undo completed learning work.
                logging.getLogger(__name__).debug("Progress callback failed", exc_info=True)

    def skip(self, phase: Phase, reason: str) -> None:
        self.emit(ProgressEvent(phase=phase, status="skipped", details={"reason": reason}))

    @contextmanager
    def phase(self, phase: Phase, **details: int | str | bool) -> Iterator[None]:
        started = time.monotonic()
        self.emit(ProgressEvent(phase=phase, status="started", details=details))
        try:
            yield
        except BaseException:
            self.emit(ProgressEvent(
                phase=phase, status="failed", elapsed_seconds=time.monotonic() - started,
                details=details,
            ))
            raise
        else:
            self.emit(ProgressEvent(
                phase=phase, status="completed", elapsed_seconds=time.monotonic() - started,
                details=details,
            ))
