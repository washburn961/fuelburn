import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere
from fuelburn import Aerodynamics, Propulsion, ClimbProfile, CruiseProfile, DescentProfile, SpeedProfile, Simulation

import designTool_merged as dt
# import designAnalysis as da

# ---------- Aircraft Configuration ----------
# Based on typical wide body 777-300ER

# Aerodynamics
# ERJ = dt.standard_airplane('ERJ145-XR')
# dt.design(ERJ, print_log=False, plot=False)

# def my_drag(mach: float, altitude_m: float, weight: float) -> float:
#     atm = Atmosphere(altitude_m)
#     a = atm.speed_of_sound[0]  # m/s
#     rho = atm.density[0]  # kg/m^3
#     V = mach * a  # m/s

#     CL = weight/(0.5*rho*(V**2)*ERJ['S_w'])
    
#     CD,_,_ = dt.aerodynamics(ERJ, mach, altitude_m, CL, weight)

#     return CD * 0.5 * rho * (V**2) * ERJ['S_w']

aero = Aerodynamics(
    S=436.8,        # Wing area [m^2]
    CD0=0.0186,      # Zero-lift drag coefficient
    k=0.04112,         # Induced drag factor (1/(pi*e*AR))
    drag_fn=None
)

# Propulsion - twin turbofans
prop = Propulsion(
    BPR=8.7,                    # Bypass ratio
    Tmax_sl_total=1026e3,     # Total static thrust at sea level [N] (~54,000 lbf per engine x 2)
    TSFC_sl=9e-6              # Sea-level TSFC [kg/(N·s)] (~0.33 lb/(lbf·hr))
)

# Speed Profiles
climb_prof = ClimbProfile(
    initial_speed_kcas=250.0,       # 250 KCAS below 10,000 ft
    crossover_speed_kcas=310.0,     # 280 KCAS at crossover
    crossover_mach=0.84             # M0.76 above crossover
)

cruise_prof = CruiseProfile(
    altitude_ft=36000.0,            # FL320
    mach=0.84                       # M0.76 cruise
)

descent_prof = DescentProfile(
    initial_speed_kcas=250.0,       # 250 KCAS below 10,000 ft
    crossover_speed_kcas=310.0,     # 280 KCAS at crossover
    crossover_mach=0.84             # M0.76 above crossover
)

speed_profile = SpeedProfile(
    climb=climb_prof,
    cruise=cruise_prof,
    descent=descent_prof,
    transition_altitude_ft=10000.0  # 10,000 ft transition altitude
)

# ---------- Mission Setup ----------
initial_mass_kg = 289662.0       # Initial mass [kg] (~160,000 lb)
initial_altitude_m = 0.0      # Start at 0 m (~0 ft) after takeoff
cruise_distance_m = 1.0047e+7      # Total mission distance: 1,111 km (~600 nm)

print("=" * 60)
print("AIRCRAFT FUEL BURN SIMULATION")
print("=" * 60)
print(f"Initial mass:     {initial_mass_kg:,.0f} kg")
print(f"Initial altitude: {initial_altitude_m:,.0f} m")
print(f"Mission distance: {cruise_distance_m/1000:.0f} km ({cruise_distance_m/1852:.0f} nm)")
print(f"Cruise altitude:  FL{int(cruise_prof.altitude_ft/100)}")
print(f"Cruise Mach:      M{cruise_prof.mach:.2f}")
print("=" * 60)
print()

# ---------- Run Simulation ----------
sim = Simulation(aero, prop, speed_profile)
results = sim.run(
    initial_mass_kg=initial_mass_kg,
    initial_altitude_m=initial_altitude_m,
    cruise_distance_m=cruise_distance_m,
    max_time_s=64400.0  # 4 hours max
)

# ---------- Post-Process Results ----------
print()
print("=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)

t = np.array(results['t']) / 60.0  # Convert to minutes
m = np.array(results['m'])
h = np.array(results['h'])
d = np.array(results['d']) / 1000.0  # Convert to km
phase = results['phase']
thrust = np.array(results['thrust']) / 1000.0  # Convert to kN
tsfc = np.array(results['tsfc']) * 3600.0  # Convert to kg/(N·hr)

fuel_burned = initial_mass_kg - m[-1]
flight_time = t[-1]
avg_fuel_flow = fuel_burned / (flight_time / 60.0)  # kg/hr

print(f"Total flight time:   {flight_time:.1f} min ({flight_time/60:.2f} hr)")
print(f"Total fuel burned:   {fuel_burned:,.1f} kg ({fuel_burned*2.20462:.1f} lb)")
print(f"Final mass:          {m[-1]:,.1f} kg")
print(f"Avg fuel flow:       {avg_fuel_flow:,.0f} kg/hr")
print(f"Distance traveled:   {d[-1]:.1f} km ({d[-1]/1.852:.1f} nm)")
print("=" * 60)

# ---------- Plot Results ----------
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Aircraft Fuel Burn Simulation Results', fontsize=14, fontweight='bold')

# Color-code by phase
colors = {'CLB': 'blue', 'CRZ': 'green', 'DES': 'orange'}
phase_colors = [colors[p] for p in phase]

# Plot 1: Altitude vs Distance
ax = axes[0, 0]
for p_type, color in colors.items():
    mask = np.array([p == p_type for p in phase])
    if np.any(mask):
        ax.plot(d[mask], h[mask] / 1000.0, color=color, linewidth=2, label=p_type)
ax.set_xlabel('Distance [km]')
ax.set_ylabel('Altitude [km]')
ax.set_title('Altitude Profile')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Mass vs Time
ax = axes[0, 1]
for p_type, color in colors.items():
    mask = np.array([p == p_type for p in phase])
    if np.any(mask):
        ax.plot(t[mask], m[mask] / 1000.0, color=color, linewidth=2, label=p_type)
ax.set_xlabel('Time [min]')
ax.set_ylabel('Mass [tonnes]')
ax.set_title('Aircraft Mass')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Fuel Burned vs Distance
ax = axes[1, 0]
fuel_burned_arr = (initial_mass_kg - m) / 1000.0
for p_type, color in colors.items():
    mask = np.array([p == p_type for p in phase])
    if np.any(mask):
        ax.plot(d[mask], fuel_burned_arr[mask], color=color, linewidth=2, label=p_type)
ax.set_xlabel('Distance [km]')
ax.set_ylabel('Fuel Burned [tonnes]')
ax.set_title('Cumulative Fuel Burn')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Altitude vs Time
ax = axes[1, 1]
for p_type, color in colors.items():
    mask = np.array([p == p_type for p in phase])
    if np.any(mask):
        ax.plot(t[mask], h[mask] / 1000.0, color=color, linewidth=2, label=p_type)
ax.set_xlabel('Time [min]')
ax.set_ylabel('Altitude [km]')
ax.set_title('Altitude vs Time')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 5: Thrust vs Distance
ax = axes[2, 0]
for p_type, color in colors.items():
    mask = np.array([p == p_type for p in phase])
    if np.any(mask):
        ax.plot(d[mask], thrust[mask], color=color, linewidth=2, label=p_type)
ax.set_xlabel('Distance [km]')
ax.set_ylabel('Thrust [kN]')
ax.set_title('Thrust Profile')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 6: TSFC vs Distance
ax = axes[2, 1]
for p_type, color in colors.items():
    mask = np.array([p == p_type for p in phase])
    if np.any(mask):
        ax.plot(d[mask], tsfc[mask], color=color, linewidth=2, label=p_type)
ax.set_xlabel('Distance [km]')
ax.set_ylabel('TSFC [kg/(N·hr)]')
ax.set_title('TSFC Profile')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

print("\nPlots displayed.")