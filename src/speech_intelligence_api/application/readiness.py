"""Dependency-agnostic service readiness evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class ReadinessCheck(Protocol):
    """A named asynchronous dependency check."""

    @property
    def name(self) -> str:
        """Return a stable, non-sensitive dependency name."""

    async def check(self) -> None:
        """Raise an exception when the dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Aggregated readiness result."""

    ready: bool
    checks: dict[str, str]


class ReadinessService:
    """Run bounded, adapter-supplied readiness checks."""

    def __init__(self, checks: tuple[ReadinessCheck, ...] = ()) -> None:
        self._checks = checks

    async def evaluate(self) -> ReadinessReport:
        results: dict[str, str] = {"application": "ok"}
        ready = True
        for check in self._checks:
            try:
                await check.check()
            except Exception:
                logger.warning(
                    "Readiness dependency unavailable",
                    extra={"dependency": check.name},
                    exc_info=True,
                )
                results[check.name] = "unavailable"
                ready = False
            else:
                results[check.name] = "ok"
        return ReadinessReport(ready=ready, checks=results)
