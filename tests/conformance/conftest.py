"""Prints the maximum deviations recorded by the Python conformance tests."""

from __future__ import annotations

from typing import Any

from conformance_support import DEVIATIONS


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if DEVIATIONS:
        terminalreporter.section("conformance: maximum absolute deviations")
        for label, value in sorted(DEVIATIONS.items()):
            terminalreporter.write_line(f"{label}: {value:.3e}")
