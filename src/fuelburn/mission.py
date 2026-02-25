"""
Mission class for simplified mission definition.
"""


class Mission:
    """
    Simplified mission definition.
    
    Usage:
        mission = Mission(
            distance=810,  # nm
            cruise_altitude=37000,  # ft
            cruise_mach=0.78,
            payload_kg=15000
        )
        
        # Or with explicit units
        mission = Mission(
            distance_nm=810,
            cruise_altitude_ft=37000,
            cruise_mach=0.78,
            payload_kg=15000
        )
    """
    
    def __init__(
        self,
        distance: float = None,
        distance_nm: float = None,
        distance_km: float = None,
        distance_m: float = None,
        cruise_altitude: float = None,
        cruise_altitude_ft: float = None,
        cruise_altitude_m: float = None,
        cruise_mach: float = 0.78,
        initial_weight_kg: float = None,
        initial_altitude_m: float = 0.0
    ):
        """
        Initialize mission.
        
        Args:
            distance: Mission distance (assumes nm if no unit specified)
            distance_nm: Mission distance [nm]
            distance_km: Mission distance [km]
            distance_m: Mission distance [m]
            cruise_altitude: Cruise altitude (assumes ft if no unit specified)
            cruise_altitude_ft: Cruise altitude [ft]
            cruise_altitude_m: Cruise altitude [m]
            cruise_mach: Cruise Mach number
            initial_weight_kg: Takeoff weight [kg]
            initial_altitude_m: Initial altitude [m] (after takeoff)
        """
        # Handle distance with smart unit detection
        if distance is not None:
            self.distance_nm = distance  # Default to nm
        elif distance_nm is not None:
            self.distance_nm = distance_nm
        elif distance_km is not None:
            self.distance_nm = distance_km / 1.852
        elif distance_m is not None:
            self.distance_nm = distance_m / 1852.0
        else:
            raise ValueError("Must specify distance")
        
        # Convert to meters for simulation
        self.distance_m = self.distance_nm * 1852.0
        self.distance_km = self.distance_m / 1000.0
        
        # Handle cruise altitude with smart unit detection
        if cruise_altitude is not None:
            self.cruise_altitude_ft = cruise_altitude  # Default to ft
        elif cruise_altitude_ft is not None:
            self.cruise_altitude_ft = cruise_altitude_ft
        elif cruise_altitude_m is not None:
            self.cruise_altitude_ft = cruise_altitude_m * 3.281
        else:
            raise ValueError("Must specify cruise_altitude")
        
        # Convert to meters for simulation
        self.cruise_altitude_m = self.cruise_altitude_ft / 3.281
        
        # Other parameters
        self.cruise_mach = cruise_mach
        self.initial_weight_kg = initial_weight_kg
        self.initial_altitude_m = initial_altitude_m
    
    def __repr__(self) -> str:
        return (
            f"Mission("\
            f"distance={self.distance_nm:.0f} nm, "\
            f"cruise=FL{int(self.cruise_altitude_ft/100)} @ M{self.cruise_mach:.2f}, "\
            f"takeoff_weight={self.initial_weight_kg:.0f} kg)"
        )
