import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere
import fuelburn as fb

import designTool_merged as dt

# ---------- Aircraft Configuration ----------
# Based on typical regional jet 145XR

# Get design tool data
ERJ = dt.standard_airplane('ERJ145-XR')
dt.design(ERJ, print_log=False, plot=False)

# Define custom drag function using design tool
def my_drag(mach: float, altitude_m: float, weight: float) -> float:
    atm = Atmosphere(altitude_m)
    a = atm.speed_of_sound[0]  # m/s
    rho = atm.density[0]  # kg/m^3
    V = mach * a  # m/s

    CL = weight/(0.5*rho*(V**2)*ERJ['S_w'])
    
    CD,_,_ = dt.aerodynamics(ERJ, mach, altitude_m, CL, weight)

    return CD * 0.5 * rho * (V**2) * ERJ['S_w']

# Create aircraft using new API with custom drag
aircraft = fb.Aircraft(
    name='ERJ-145XR',
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
    drag_fn=my_drag  # Custom drag function
)

# Define mission using new API
mission = fb.Mission(
    distance_nm=715,  # 1,111 km ≈ 600 nm
    cruise_altitude_ft=34000,
    cruise_mach=0.76,
    initial_weight_kg=20104,
    initial_altitude_m=0.0
)

print("=" * 60)
print("AIRCRAFT FUEL BURN SIMULATION")
print("=" * 60)
print(f"Aircraft: {aircraft.name}")
print(f"Mission:  {mission}")
print("=" * 60)
print()

# ---------- Run Simulation ----------
results = aircraft.fly(mission, max_time_s=14400.0)

# ---------- Display Results ----------
print(results)

# ---------- Plot Results ----------
results.plot()

print("\nPlots displayed.")