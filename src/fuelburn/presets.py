"""
Aircraft presets database.
"""

from .aircraft import AircraftSpec


# Aircraft presets database
AIRCRAFT_PRESETS = {
    'B737-800': AircraftSpec(
        name='Boeing 737-800',
        wing_area_m2=124.6,
        thrust_per_engine_N=121400,  # CFM56-7B27
        num_engines=2,
        bypass_ratio=5.5,
        cd0=0.024,
        oswald_efficiency=0.85,
        max_cruise_mach=0.82,
        max_cruise_altitude_ft=41000.0,
        typical_climb_kcas_low=250.0,
        typical_climb_kcas_high=280.0,
        typical_climb_mach=0.78
    ),
    
    'A320': AircraftSpec(
        name='Airbus A320-200',
        wing_area_m2=122.6,
        thrust_per_engine_N=120100,  # CFM56-5B4
        num_engines=2,
        bypass_ratio=5.9,
        cd0=0.024,
        oswald_efficiency=0.85,
        max_cruise_mach=0.82,
        max_cruise_altitude_ft=39000.0,
        typical_climb_kcas_low=250.0,
        typical_climb_kcas_high=280.0,
        typical_climb_mach=0.78
    ),
    
    'B777-300ER': AircraftSpec(
        name='Boeing 777-300ER',
        wing_area_m2=427.8,
        thrust_per_engine_N=512000,  # GE90-115B
        num_engines=2,
        bypass_ratio=8.7,
        cd0=0.019,
        oswald_efficiency=0.87,
        max_cruise_mach=0.89,
        max_cruise_altitude_ft=43100.0,
        typical_climb_kcas_low=250.0,
        typical_climb_kcas_high=310.0,
        typical_climb_mach=0.84
    ),
    
    'A330-300': AircraftSpec(
        name='Airbus A330-300',
        wing_area_m2=361.6,
        thrust_per_engine_N=316000,  # Trent 772B
        num_engines=2,
        bypass_ratio=5.0,
        cd0=0.020,
        oswald_efficiency=0.86,
        max_cruise_mach=0.86,
        max_cruise_altitude_ft=41450.0,
        typical_climb_kcas_low=250.0,
        typical_climb_kcas_high=300.0,
        typical_climb_mach=0.82
    ),
    
    'ERJ-145XR': AircraftSpec(
        name='Embraer ERJ-145XR',
        wing_area_m2=51.18,
        thrust_per_engine_N=39670,  # AE 3007A1E
        num_engines=2,
        bypass_ratio=5.0,
        cd0=0.0186,
        oswald_efficiency=0.84,
        max_cruise_mach=0.78,
        max_cruise_altitude_ft=37000.0,
        typical_climb_kcas_low=240.0,
        typical_climb_kcas_high=270.0,
        typical_climb_mach=0.76
    ),
    
    'CRJ-900': AircraftSpec(
        name='Bombardier CRJ-900',
        wing_area_m2=76.2,
        thrust_per_engine_N=64500,  # CF34-8C5
        num_engines=2,
        bypass_ratio=5.3,
        cd0=0.0190,
        oswald_efficiency=0.83,
        max_cruise_mach=0.85,
        max_cruise_altitude_ft=41000.0,
        typical_climb_kcas_low=250.0,
        typical_climb_kcas_high=280.0,
        typical_climb_mach=0.78
    ),
    
    'B747-400': AircraftSpec(
        name='Boeing 747-400',
        wing_area_m2=541.1,
        thrust_per_engine_N=282000,  # PW4062
        num_engines=4,
        bypass_ratio=5.0,
        cd0=0.021,
        oswald_efficiency=0.86,
        max_cruise_mach=0.855,
        max_cruise_altitude_ft=45100.0,
        typical_climb_kcas_low=250.0,
        typical_climb_kcas_high=310.0,
        typical_climb_mach=0.84
    ),
    
    'A350-900': AircraftSpec(
        name='Airbus A350-900',
        wing_area_m2=442.0,
        thrust_per_engine_N=374000,  # Trent XWB-84
        num_engines=2,
        bypass_ratio=9.3,
        cd0=0.018,
        oswald_efficiency=0.88,
        max_cruise_mach=0.89,
        max_cruise_altitude_ft=43100.0,
        typical_climb_kcas_low=250.0,
        typical_climb_kcas_high=310.0,
        typical_climb_mach=0.85
    ),
}


def get_aircraft_preset(aircraft_type: str) -> AircraftSpec:
    """
    Get aircraft preset by type.
    
    Args:
        aircraft_type: Aircraft type code (e.g., 'B737-800', 'A320')
    
    Returns:
        AircraftSpec instance
    
    Raises:
        ValueError: If aircraft type not found
    """
    # Try exact match first
    if aircraft_type in AIRCRAFT_PRESETS:
        return AIRCRAFT_PRESETS[aircraft_type]
    
    # Try case-insensitive match
    aircraft_type_upper = aircraft_type.upper()
    for key in AIRCRAFT_PRESETS:
        if key.upper() == aircraft_type_upper:
            return AIRCRAFT_PRESETS[key]
    
    # Not found
    available = ', '.join(sorted(AIRCRAFT_PRESETS.keys()))
    raise ValueError(
        f"Aircraft type '{aircraft_type}' not found. "
        f"Available presets: {available}"
    )


def list_presets() -> list:
    """
    List all available aircraft presets.
    
    Returns:
        List of aircraft type codes
    """
    return sorted(AIRCRAFT_PRESETS.keys())


def print_presets():
    """Print formatted list of available aircraft presets."""
    print("Available Aircraft Presets:")
    print("=" * 60)
    for key in sorted(AIRCRAFT_PRESETS.keys()):
        spec = AIRCRAFT_PRESETS[key]
        print(f"{key:15} - {spec.name:30} MTOW: {spec.mtow_kg:>7,.0f} kg")
    print("=" * 60)
