"""
Aircraft class for simplified aircraft definition and simulation.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

from .simulation import (
    Aerodynamics,
    Propulsion,
    SpeedProfile,
    ClimbProfile,
    CruiseProfile,
    DescentProfile,
    Simulation
)


@dataclass
class AircraftSpec:
    """Aircraft specification dataclass."""
    name: str
    wing_area_m2: float
    thrust_per_engine_N: float
    num_engines: int
    bypass_ratio: float
    cd0: float
    oswald_efficiency: float
    max_cruise_mach: float
    max_cruise_altitude_ft: float
    typical_climb_kcas_low: float = 250.0
    typical_climb_kcas_high: float = 280.0
    typical_climb_mach: float = 0.78


class Aircraft:
    """
    Simplified aircraft class for fuel burn simulation.
    
    Usage:
        # From preset
        aircraft = Aircraft.from_preset('B737-800')
        
        # Custom
        aircraft = Aircraft(
            name='My Aircraft',
            mtow_kg=73000,
            wing_area_m2=122.6,
            ...
        )
        
        # Fly mission
        results = aircraft.fly(mission)
    """
    
    def __init__(
        self,
        name: str,
        wing_area_m2: float,
        thrust_per_engine_N: float,
        num_engines: int = 2,
        bypass_ratio: float = 5.5,
        cd0: float = 0.024,
        oswald_efficiency: float = 0.85,
        max_cruise_mach: float = 0.82,
        max_cruise_altitude_ft: float = 41000.0,
        typical_climb_kcas_low: float = 250.0,
        typical_climb_kcas_high: float = 280.0,
        typical_climb_mach: float = 0.78,
        tsfc_sl: float = 9e-6,
        drag_fn: Optional[Callable] = None
    ):
        """
        Initialize aircraft.
        
        Args:
            name: Aircraft name
            wing_area_m2: Wing reference area [m²]
            thrust_per_engine_N: Thrust per engine at sea level [N]
            num_engines: Number of engines
            bypass_ratio: Engine bypass ratio
            cd0: Zero-lift drag coefficient
            oswald_efficiency: Oswald efficiency factor
            max_cruise_mach: Maximum cruise Mach number
            max_cruise_altitude_ft: Maximum cruise altitude [ft]
            typical_climb_kcas_low: Climb speed below 10,000 ft [KCAS]
            typical_climb_kcas_high: Climb speed at crossover [KCAS]
            typical_climb_mach: Climb Mach above crossover
            tsfc_sl: Sea-level TSFC [kg/(N·s)]
            drag_fn: Optional custom drag function
        """
        self.name = name
        self.wing_area_m2 = wing_area_m2
        self.thrust_per_engine_N = thrust_per_engine_N
        self.num_engines = num_engines
        self.bypass_ratio = bypass_ratio
        self.cd0 = cd0
        self.oswald_efficiency = oswald_efficiency
        self.max_cruise_mach = max_cruise_mach
        self.max_cruise_altitude_ft = max_cruise_altitude_ft
        self.typical_climb_kcas_low = typical_climb_kcas_low
        self.typical_climb_kcas_high = typical_climb_kcas_high
        self.typical_climb_mach = typical_climb_mach
        self.tsfc_sl = tsfc_sl
        self._custom_drag_fn = drag_fn
        
        # Computed properties
        self.total_thrust_N = thrust_per_engine_N * num_engines
        
        # Calculate induced drag factor: k = 1/(π * e * AR)
        # Assume typical aspect ratio if not provided
        aspect_ratio = 9.0  # Typical for commercial jets
        self.k = 1.0 / (np.pi * oswald_efficiency * aspect_ratio)
        
        # Create aerodynamics model
        self.aerodynamics = Aerodynamics(
            S=wing_area_m2,
            CD0=cd0,
            k=self.k,
            drag_fn=drag_fn
        )
        
        # Create propulsion model
        self.propulsion = Propulsion(
            BPR=bypass_ratio,
            Tmax_sl_total=self.total_thrust_N,
            TSFC_sl=tsfc_sl
        )
    
    @classmethod
    def from_preset(cls, aircraft_type: str) -> 'Aircraft':
        """
        Create aircraft from preset.
        
        Args:
            aircraft_type: Aircraft type (e.g., 'B737-800', 'A320', 'B777-300ER')
        
        Returns:
            Aircraft instance
        """
        from .presets import get_aircraft_preset
        spec = get_aircraft_preset(aircraft_type)
        
        return cls(
            name=spec.name,
            wing_area_m2=spec.wing_area_m2,
            thrust_per_engine_N=spec.thrust_per_engine_N,
            num_engines=spec.num_engines,
            bypass_ratio=spec.bypass_ratio,
            cd0=spec.cd0,
            oswald_efficiency=spec.oswald_efficiency,
            max_cruise_mach=spec.max_cruise_mach,
            max_cruise_altitude_ft=spec.max_cruise_altitude_ft,
            typical_climb_kcas_low=spec.typical_climb_kcas_low,
            typical_climb_kcas_high=spec.typical_climb_kcas_high,
            typical_climb_mach=spec.typical_climb_mach
        )
    
    def fly(self, mission, max_time_s: float = 14400.0, verbose: bool = True):
        """
        Fly a mission and return results.
        
        Args:
            mission: Mission instance
            max_time_s: Maximum simulation time [s]
            verbose: Print progress messages
        
        Returns:
            Results instance
        """
        from .mission import Mission
        from .results import Results
        
        if not isinstance(mission, Mission):
            raise TypeError("mission must be a Mission instance")
        
        # Build speed profile from mission
        climb_prof = ClimbProfile(
            initial_speed_kcas=self.typical_climb_kcas_low,
            crossover_speed_kcas=self.typical_climb_kcas_high,
            crossover_mach=self.typical_climb_mach
        )
        
        cruise_prof = CruiseProfile(
            altitude_ft=mission.cruise_altitude_ft,
            mach=mission.cruise_mach
        )
        
        descent_prof = DescentProfile(
            initial_speed_kcas=self.typical_climb_kcas_low,
            crossover_speed_kcas=self.typical_climb_kcas_high,
            crossover_mach=self.typical_climb_mach
        )
        
        speed_profile = SpeedProfile(
            climb=climb_prof,
            cruise=cruise_prof,
            descent=descent_prof,
            transition_altitude_ft=10000.0
        )
        
        # Use only takeoff weight from mission
        initial_mass_kg = mission.initial_weight_kg
        
        # Create simulation
        sim = Simulation(self.aerodynamics, self.propulsion, speed_profile)
        
        # Run simulation
        if verbose:
            print(f"Flying {self.name} on {mission.distance_nm:.0f} nm mission...")
        
        raw_results = sim.run(
            initial_mass_kg=initial_mass_kg,
            initial_altitude_m=mission.initial_altitude_m,
            cruise_distance_m=mission.distance_m,
            max_time_s=max_time_s
        )
        
        # Wrap in Results object
        return Results(
            raw_results=raw_results,
            aircraft=self,
            mission=mission,
            initial_mass_kg=initial_mass_kg
        )
    
    def __repr__(self) -> str:
        return (
            f"Aircraft(name='{self.name}', "
            f"MTOW={self.mtow_kg:.0f} kg, "
            f"engines={self.num_engines}x{self.thrust_per_engine_N/1000:.0f} kN)"
        )
