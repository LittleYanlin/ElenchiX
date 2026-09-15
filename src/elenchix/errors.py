"""Recoverable errors exposed to application adapters."""


class NoAvailableCasesError(ValueError):
    """No unseen case remains in the learner's requested candidate set."""


class AssessmentFailedError(ValueError):
    """No assessment passed validation; the active round remains available to retry."""

    def __init__(self, attempts: int, validation_error: str | None = None):
        self.attempts = attempts
        self.validation_error = validation_error
        message = f"assessment did not pass validation after {attempts} attempts"
        if validation_error:
            message += ": " + validation_error
        super().__init__(message)
