# FuelBurn

A Python package for aircraft fuel burn simulation and analysis. Provides a simple, extensible API for defining aircraft, missions, and running detailed fuel burn simulations with plotting and data export capabilities.

---

## Installation

Install directly from the latest GitHub main branch:

```bash
pip install git+https://github.com/yourusername/fuelburn.git
```

To upgrade to the latest version:

```bash
pip install --upgrade git+https://github.com/yourusername/fuelburn.git
```

> **Requirements:** Python 3.7+, [ambiance](https://pypi.org/project/ambiance/), numpy, matplotlib

---

## Usage

### 1. Quickstart: Simulate Fuel Burn for a Preset Aircraft

```python
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
results = aircraft.fly(mission)

# Print summary
print(results)

# Plot all results
results.plot()

# Save all plots to files in the script directory
import os
base_path = os.path.join(os.path.dirname(__file__), 'b777_mission')
results.save_plots(base_path)

# Export results to CSV
results.to_csv('b777_mission_results.csv')
```

---

### 2. List and Use Preset Aircraft

```python
import fuelburn as fb

# List all available presets
fb.print_presets()

# Get a preset by code
aircraft = fb.Aircraft.from_preset('A320')
```

---

### 3. Custom Aircraft and Mission

```python
import fuelburn as fb

# Define a custom aircraft
aircraft = fb.Aircraft(
    name='MyJet',
    wing_area_m2=100.0,
    thrust_per_engine_N=120000,
    num_engines=2,
    bypass_ratio=5.0,
    cd0=0.022,
    oswald_efficiency=0.85,
    max_cruise_mach=0.80,
    max_cruise_altitude_ft=41000.0,
    typical_climb_kcas_low=250.0,
    typical_climb_kcas_high=280.0,
    typical_climb_mach=0.78,
    tsfc_sl=2e-5
)

# Define a mission
mission = fb.Mission(
    distance_nm=1000,
    cruise_altitude_ft=39000,
    cruise_mach=0.78,
    initial_weight_kg=70000
)

results = aircraft.fly(mission)
print(results)
results.plot()
```

---

### 4. Accessing Results Data

```python
# After running a simulation:
print(f"Fuel burned: {results.fuel_burned_kg:.0f} kg")
print(f"Block time: {results.block_time_hr:.2f} hr")
print(f"Max altitude: {results.max_altitude_ft:.0f} ft")

# Access time series:
time = results.time_min  # minutes
altitude = results.altitude_ft  # feet
fuel_burn = results.fuel_burned  # kg
```

---

### 5. Advanced: Custom Drag Function

```python
from ambiance import Atmosphere
import fuelburn as fb

def my_drag(mach, altitude_m, weight):
    atm = Atmosphere(altitude_m)
    a = atm.speed_of_sound[0]
    rho = atm.density[0]
    V = mach * a
    CL = weight / (0.5 * rho * V**2 * 51.18)
    CD = 0.0186 + CL**2 / (np.pi * 0.84 * 9.0)
    return CD * 0.5 * rho * V**2 * 51.18

aircraft = fb.Aircraft(
    name='CustomJet',
    wing_area_m2=51.18,
    thrust_per_engine_N=39670,
    num_engines=2,
    bypass_ratio=5.0,
    cd0=0.0186,
    oswald_efficiency=0.84,
    max_cruise_mach=0.78,
    max_cruise_altitude_ft=37000.0,
    typical_climb_kcas_low=240.0,
    typical_climb_kcas_high=270.0,
    typical_climb_mach=0.76,
    tsfc_sl=2e-5,
    drag_fn=my_drag
)
```

---

## API Overview

- `Aircraft`: Define or load an aircraft (from preset or custom)
- `Mission`: Define a mission profile
- `Results`: Access results, plots, and export
- `print_presets()`, `list_presets()`: List available aircraft presets

---

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade git+https://github.com/yourusername/fuelburn.git
```

---

## License

See LICENSE file.
