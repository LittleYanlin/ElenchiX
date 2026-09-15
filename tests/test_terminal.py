import io
from threading import Event

import pytest

from elenchix.terminal import Spinner


class AnimatedTerminal(io.StringIO):
    def __init__(self):
        super().__init__()
        self.animated = Event()

    def isatty(self):
        return True

    def write(self, value):
        if "◓" in value:
            self.animated.set()
        return super().write(value)


def test_spinner_animates_during_work_and_always_stops_after_an_error():
    stream = AnimatedTerminal()
    spinner = Spinner("Tutor is thinking", stream=stream)
    with pytest.raises(RuntimeError, match="request failed"), spinner:
        assert stream.animated.wait(timeout=2), "no animation appeared while waiting"
        raise RuntimeError("request failed")
    assert spinner._thread is not None and not spinner._thread.is_alive()
    assert "◐" in stream.getvalue() and "◓" in stream.getvalue()
    assert stream.getvalue().endswith("\r")


def test_redirected_output_has_one_plain_status_line():
    stream = io.StringIO()
    with Spinner("Tutor is thinking", stream=stream):
        pass
    assert stream.getvalue() == "Tutor is thinking...\n"
