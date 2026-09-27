"""Execution services are declared once, independently of schedulable agent labels."""
from dataclasses import dataclass

from codeaction.agents.vendor.clis import VENDOR_CLIS


@dataclass(frozen=True)
class ExecutionDriver:
    mode: str
    image_option: str
    cli_label: str
    compose_profile: str
    compose_service: str


DRIVERS = {
    "reference": ExecutionDriver("reference", "reference_agent_image", "codeaction-reference", "reference", "reference-agent"),
    "fixture": ExecutionDriver("fixture", "fixture_agent_image", "offline-fixture", "fixture", "fixture-agent"),
    **{name: ExecutionDriver(name, seat.image_cli_flag.removeprefix("--").replace("-", "_"),
                             seat.cli_label, seat.compose_profile, seat.compose_service)
       for name, seat in VENDOR_CLIS.items()},
}


def execution_driver(name: str) -> ExecutionDriver:
    try:
        return DRIVERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown execution driver {name!r}") from exc
