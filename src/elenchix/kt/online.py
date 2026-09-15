"""Chronological online refitting and atomic local persistence.

Offline benchmark cross-validation remains in reliable_transfer.py. Online transfer
records retain predictions and source expectations made before observing outcomes.
"""

from __future__ import annotations

import pickle
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .reliable_transfer import attach_messages, fit_relation_reliability, fit_zero_anchor


def fit_online_adapter(rows: list[dict]) -> tuple[dict, list[float], list[float]]:
    if not rows:
        return {}, [0.0, 0.0], [1.0, 1.0]
    reliability = fit_relation_reliability(rows)
    learners = sorted({row["learner_id"] for row in rows})
    if len(learners) >= 2:
        # Grouped cross-fitting prevents an observation's label from fitting its own
        # relation feature. Use at most five folds over currently available learners.
        assignment = {learner: i % min(5, len(learners)) for i, learner in enumerate(learners)}
        featured = []
        for fold in sorted(set(assignment.values())):
            train = [row for row in rows if assignment[row["learner_id"]] != fold]
            held = [row for row in rows if assignment[row["learner_id"]] == fold]
            featured.extend(attach_messages(held, fit_relation_reliability(train)))
    else:
        # Before grouped cross-fitting is possible, use features frozen at encounter
        # start under the then-available relation model (never fit-on-self features).
        featured = rows
    _, readout = fit_zero_anchor(featured, featured)
    return reliability, readout["coefficients"], readout["rms_scale"]


class OnlineStateStore:
    """Atomic snapshots of this application's own trusted local state.

    SQLite holds model tensors, fitted sklearn estimators, histories and provenance
    in one transaction. Revision checks reject concurrent stale writers. This is a
    private runtime database, not a portable/untrusted model import format.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS state "
                "(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, payload BLOB NOT NULL)"
            )

    def load(self) -> tuple[int, dict[str, Any] | None]:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT revision, payload FROM state WHERE id=1").fetchone()
        return (0, None) if row is None else (row[0], pickle.loads(row[1]))

    @staticmethod
    def read_only(path: Path) -> tuple[int, dict[str, Any] | None]:
        """Read one committed snapshot without creating or modifying the database."""
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            row = connection.execute("SELECT revision, payload FROM state WHERE id=1").fetchone()
        return (0, None) if row is None else (row[0], pickle.loads(row[1]))

    def save(self, expected_revision: int, payload: dict[str, Any]) -> int:
        encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT revision FROM state WHERE id=1").fetchone()
            actual = row[0] if row else 0
            if actual != expected_revision:
                raise RuntimeError(
                    "online state was updated by another session; reopen this session"
                )
            revision = actual + 1
            connection.execute("INSERT OR REPLACE INTO state VALUES (1, ?, ?)", (revision, encoded))
        return revision
