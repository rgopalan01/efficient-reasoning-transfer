from __future__ import annotations

from effreason.utils import format_duration, log_progress


def test_format_duration_is_compact_and_handles_invalid_values():
    assert format_duration(0) == "0s"
    assert format_duration(65) == "1m 05s"
    assert format_duration(3661) == "1h 01m 01s"
    assert format_duration(-1) == "unknown"
    assert format_duration(float("nan")) == "unknown"


def test_log_progress_has_component_prefix(capsys):
    log_progress("evaluate", "completed 3/10")
    assert capsys.readouterr().out == "[evaluate] completed 3/10\n"
