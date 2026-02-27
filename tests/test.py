import fuelburn as fb

# Use a preset aircraft
aircraft = fb.Aircraft.from_preset('B777-300ER')

# Define a mission
mission = fb.Mission(
    distance=5425,  # nm
    cruise_altitude=35000,  # ft
    cruise_mach=0.84,
    initial_weight_kg=289662,  # kg
)

# Run the simulation
results = aircraft.fly(mission, max_time_s=360000)  # seconds

# Print summary
print(results)

# Plot all results

# Save all plots to files in the script directory
import os

results.plot(save_path=os.path.join(os.path.dirname(__file__), 'b777_mission.png'))

# Export results to CSV
results.to_csv(os.path.join(os.path.dirname(__file__), 'b777_mission_results.csv'))