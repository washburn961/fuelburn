"""
Simple example using the new fuelburn API.

This shows how easy it is to run a fuel burn simulation with the new interface.
"""

import fuelburn as fb

# ===== BRAIN-DEAD SIMPLE APPROACH =====

# Use a preset aircraft
aircraft = fb.Aircraft.from_preset('ERJ-145XR')

# Define mission
mission = fb.Mission(
    distance=629,  # nm  
    cruise_altitude=32000,  # ft
    cruise_mach=0.76,
    initial_weight_kg=20000,  # kg
)

# Fly it!
results = aircraft.fly(mission, max_time_s=70e3, verbose=False)

# Print results
print(results)

# Plot everything
results.plot()

# ===== OR ACCESS SPECIFIC DATA =====

print(f"\nFuel burned: {results.fuel_burned_kg:,.0f} kg ({results.fuel_burned_lb:,.0f} lb)")
print(f"Block time: {results.block_time_hr:.2f} hr")
print(f"Average fuel flow: {results.avg_fuel_flow_kg_hr:,.0f} kg/hr")

# Plot specific charts
# results.plot_altitude_profile()
# results.plot_fuel_burn()
# results.plot_thrust()

# Export to CSV
# results.to_csv('mission_results.csv')