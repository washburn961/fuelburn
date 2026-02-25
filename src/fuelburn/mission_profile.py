from dataclasses import dataclass
from ambiance import Atmosphere

@dataclass
class ClimbProfile:
    initial_speed_kcas: float = 240.0
    crossover_speed_kcas: float = 270.0
    crossover_mach: float = 0.56

@dataclass
class CruiseProfile:
    altitude: float = 35000.0
    mach: float = 0.76

@dataclass
class DescentProfile:
    initial_speed_kcas: float = 240.0
    crossover_speed_kcas: float = 0.0
    crossover_mach: float = 0.0

@dataclass
class SpeedProfile:
    climb: ClimbProfile = None
    cruise: CruiseProfile = None
    descent: DescentProfile = None
    transition_altitude: float = 10000.0