"""Minimal public implementation of the ElenchiX tutoring loop."""

from .errors import AssessmentFailedError, NoAvailableCasesError
from .progress import ProgressEvent
from .schemas import ActiveRound, RoundRecord, SessionResult
from .workflow import ElenchiXSession, RoundConflictError, RoundNotFoundError

__all__ = [
    "ActiveRound",
    "AssessmentFailedError",
    "ElenchiXSession",
    "NoAvailableCasesError",
    "ProgressEvent",
    "RoundConflictError",
    "RoundNotFoundError",
    "RoundRecord",
    "SessionResult",
]
__version__ = "0.1.0"
