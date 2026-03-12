"""
Test script to verify taxi functionality.
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from fuelburn import Aircraft, Mission

# Use a preset aircraft
try:
    aircraft = Aircraft.from_preset('B737-800')
except:
    # Create a simple aircraft manually if preset fails
    aircraft = Aircraft(
        name='Test Aircraft',
        wing_area_m2=125.0,
        thrust_per_engine_N=121000,
        num_engines=2,
        bypass_ratio=5.1,
        cd0=0.024,
        oswald_efficiency=0.85,
        max_cruise_mach=0.82,
        max_cruise_altitude_ft=41000,
        typical_climb_kcas_low=250,
        typical_climb_kcas_high=280,
        typical_climb_mach=0.78,
        tsfc_sl=2e-5,
        climb_tla=0.95,
        descent_tla=0.10,
        taxi_tla=0.07
    )

# Define mission
mission = Mission(
    distance=500,  # nm  
    cruise_altitude=35000,  # ft
    cruise_mach=0.78,
    initial_weight_kg=70000,  # kg
)

# Fly it!
print("Flying mission with taxi phases...")
results = aircraft.fly(mission, max_time_s=20000, verbose=True)

# Print results
print("\n" + "="*50)
print("TAXI FUEL BURN TEST RESULTS")
print("="*50)
print(f"Total fuel burned: {results.fuel_burned_kg:,.0f} kg")
print(f"Block time: {results.block_time_hr:.2f} hr")

# Check for taxi phases in results
phases = results.phase  # Direct access to phase array
taxi_out_count = np.sum(phases == 'TAXI_OUT')
taxi_in_count = np.sum(phases == 'TAXI_IN')

print(f"\nTaxi-out points: {taxi_out_count}")
print(f"Taxi-in points: {taxi_in_count}")

if taxi_out_count > 0 and taxi_in_count > 0:
    print("\n✓ SUCCESS: Taxi phases are working!")
else:
    print("\n✗ FAILED: Taxi phases not found in results")
