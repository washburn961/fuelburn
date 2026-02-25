# Simplified API (recommended for most users)
from .aircraft import Aircraft
from .mission import Mission
from .results import Results
from .presets import list_presets, print_presets, get_aircraft_preset

# Low-level API (for advanced users)
from .simulation import (
    Simulation,
    Aerodynamics,
    Propulsion,
    SpeedProfile,
    ClimbProfile,
    CruiseProfile,
    DescentProfile,
    FlightPhase,
    odefun_climb,
    odefun_cruise,
    odefun_descent
)

__all__ = [
    # Simplified API
    "Aircraft",
    "Mission",
    "Results",
    "list_presets",
    "print_presets",
    "get_aircraft_preset",
    # Low-level API
    "Simulation",
    "Aerodynamics",
    "Propulsion",
    "SpeedProfile",
    "ClimbProfile",
    "CruiseProfile",
    "DescentProfile",
    "FlightPhase",
    "odefun_climb",
    "odefun_cruise",
    "odefun_descent",
]

__version__ = "0.2.0"

