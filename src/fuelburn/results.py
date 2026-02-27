"""
Results class for simulation output with rich methods.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aircraft import Aircraft
    from .mission import Mission


class Results:
    """
    Rich results object with easy access to data and built-in methods.
    
    Usage:
        results = aircraft.fly(mission)
        
        # Print summary
        print(results)
        
        # Access data
        print(f"Fuel burned: {results.fuel_burned_kg:.0f} kg")
        print(f"Block time: {results.block_time_hr:.2f} hr")
        
        # Plot
        results.plot()
        results.plot_altitude_profile()
    """
    
    def __init__(self, raw_results: dict, aircraft: 'Aircraft', mission: 'Mission', initial_mass_kg: float):
        """
        Initialize results.
        
        Args:
            raw_results: Raw dictionary from simulation
            aircraft: Aircraft instance
            mission: Mission instance
            initial_mass_kg: Initial mass [kg]
        """
        self._raw = raw_results
        self.aircraft = aircraft
        self.mission = mission
        self._initial_mass_kg = initial_mass_kg
        
        # Convert to numpy arrays for easy access
        self.t = np.array(raw_results['t'])  # time [s]
        self.m = np.array(raw_results['m'])  # mass [kg]
        self.h = np.array(raw_results['h'])  # altitude [m]
        self.d = np.array(raw_results['d'])  # distance [m]
        self.phase = np.array(raw_results['phase'])  # flight phase
        self.thrust = np.array(raw_results['thrust'])  # thrust [N]
        self.tsfc = np.array(raw_results['tsfc'])  # TSFC [kg/(N·s)]
    
    # ========== Computed Properties ==========
    
    @property
    def fuel_burned_kg(self) -> float:
        """Total fuel burned [kg]."""
        return self._initial_mass_kg - self.m[-1]
    
    @property
    def fuel_burned_lb(self) -> float:
        """Total fuel burned [lb]."""
        return self.fuel_burned_kg * 2.20462
    
    @property
    def block_time_s(self) -> float:
        """Block time [s]."""
        return self.t[-1]
    
    @property
    def block_time_min(self) -> float:
        """Block time [min]."""
        return self.block_time_s / 60.0
    
    @property
    def block_time_hr(self) -> float:
        """Block time [hr]."""
        return self.block_time_s / 3600.0
    
    @property
    def distance_flown_m(self) -> float:
        """Distance flown [m]."""
        return self.d[-1]
    
    @property
    def distance_flown_km(self) -> float:
        """Distance flown [km]."""
        return self.distance_flown_m / 1000.0
    
    @property
    def distance_flown_nm(self) -> float:
        """Distance flown [nm]."""
        return self.distance_flown_m / 1852.0
    
    @property
    def avg_fuel_flow_kg_hr(self) -> float:
        """Average fuel flow [kg/hr]."""
        return self.fuel_burned_kg / self.block_time_hr
    
    @property
    def avg_fuel_flow_lb_hr(self) -> float:
        """Average fuel flow [lb/hr]."""
        return self.avg_fuel_flow_kg_hr * 2.20462
    
    @property
    def final_mass_kg(self) -> float:
        """Final aircraft mass [kg]."""
        return self.m[-1]
    
    @property
    def max_altitude_m(self) -> float:
        """Maximum altitude reached [m]."""
        return np.max(self.h)
    
    @property
    def max_altitude_ft(self) -> float:
        """Maximum altitude reached [ft]."""
        return self.max_altitude_m * 3.281
    
    # ========== Time Series with Units ==========
    
    @property
    def time_min(self) -> np.ndarray:
        """Time [min]."""
        return self.t / 60.0
    
    @property
    def time_hr(self) -> np.ndarray:
        """Time [hr]."""
        return self.t / 3600.0
    
    @property
    def altitude_ft(self) -> np.ndarray:
        """Altitude [ft]."""
        return self.h * 3.281
    
    @property
    def altitude_km(self) -> np.ndarray:
        """Altitude [km]."""
        return self.h / 1000.0
    
    @property
    def distance_km(self) -> np.ndarray:
        """Distance [km]."""
        return self.d / 1000.0
    
    @property
    def distance_nm(self) -> np.ndarray:
        """Distance [nm]."""
        return self.d / 1852.0
    
    @property
    def thrust_kn(self) -> np.ndarray:
        """Thrust [kN]."""
        return self.thrust / 1000.0
    
    @property
    def tsfc_kg_n_hr(self) -> np.ndarray:
        """TSFC [kg/(N·hr)]."""
        return self.tsfc * 3600.0
    
    @property
    def mass_tonnes(self) -> np.ndarray:
        """Mass [tonnes]."""
        return self.m / 1000.0
    
    @property
    def fuel_burned(self) -> np.ndarray:
        """Cumulative fuel burned [kg]."""
        return self._initial_mass_kg - self.m
    
    @property
    def fuel_burned_tonnes(self) -> np.ndarray:
        """Cumulative fuel burned [tonnes]."""
        return self.fuel_burned / 1000.0
    
    # ========== Phase Masks ==========
    
    @property
    def climb_mask(self) -> np.ndarray:
        """Boolean mask for climb phase."""
        return self.phase == 'CLB'
    
    @property
    def cruise_mask(self) -> np.ndarray:
        """Boolean mask for cruise phase."""
        return self.phase == 'CRZ'
    
    @property
    def descent_mask(self) -> np.ndarray:
        """Boolean mask for descent phase."""
        return self.phase == 'DES'
    
    # ========== Display Methods ==========
    
    def __repr__(self) -> str:
        """Formatted summary of results."""
        lines = [
            "=" * 60,
            "SIMULATION RESULTS",
            "=" * 60,
            f"Aircraft: {self.aircraft.name}",
            f"Mission:  {self.mission.distance_nm:.0f} nm @ FL{int(self.mission.cruise_altitude_ft/100)} M{self.mission.cruise_mach:.2f}",
            "-" * 60,
            f"Block time:       {self.block_time_hr:.2f} hr ({self.block_time_min:.1f} min)",
            f"Fuel burned:      {self.fuel_burned_kg:,.0f} kg ({self.fuel_burned_lb:,.0f} lb)",
            f"Avg fuel flow:    {self.avg_fuel_flow_kg_hr:,.0f} kg/hr ({self.avg_fuel_flow_lb_hr:,.0f} lb/hr)",
            f"Distance flown:   {self.distance_flown_nm:.1f} nm ({self.distance_flown_km:.1f} km)",
            f"Max altitude:     {self.max_altitude_ft:.0f} ft (FL{int(self.max_altitude_ft/100)})",
            f"Final mass:       {self.final_mass_kg:,.0f} kg",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """Same as __repr__."""
        return self.__repr__()
    
    # ========== Plotting Methods ==========
    
    def plot(self, figsize=(12, 12), save_path: str = None):
        """
        Plot all results in a 3x2 grid.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
        fig.suptitle(f'{self.aircraft.name} - Fuel Burn Simulation Results', 
                     fontsize=14, fontweight='bold')
        
        colors = {'CLB': 'blue', 'CRZ': 'green', 'DES': 'orange'}
        
        # Plot 1: Altitude vs Distance
        ax = axes[0, 0]
        for phase, color in colors.items():
            mask = self.phase == phase
            if np.any(mask):
                ax.plot(self.distance_nm[mask], self.altitude_ft[mask] / 1000.0, 
                       color=color, linewidth=2, label=phase)
        ax.set_xlabel('Distance [nm]')
        ax.set_ylabel('Altitude [1000 ft]')
        ax.set_title('Altitude Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Mass vs Time
        ax = axes[0, 1]
        for phase, color in colors.items():
            mask = self.phase == phase
            if np.any(mask):
                ax.plot(self.time_min[mask], self.mass_tonnes[mask], 
                       color=color, linewidth=2, label=phase)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Mass [tonnes]')
        ax.set_title('Aircraft Mass')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Fuel Burned vs Distance
        ax = axes[1, 0]
        for phase, color in colors.items():
            mask = self.phase == phase
            if np.any(mask):
                ax.plot(self.distance_nm[mask], self.fuel_burned_tonnes[mask], 
                       color=color, linewidth=2, label=phase)
        ax.set_xlabel('Distance [nm]')
        ax.set_ylabel('Fuel Burned [tonnes]')
        ax.set_title('Cumulative Fuel Burn')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: Altitude vs Time
        ax = axes[1, 1]
        for phase, color in colors.items():
            mask = self.phase == phase
            if np.any(mask):
                ax.plot(self.time_min[mask], self.altitude_ft[mask] / 1000.0, 
                       color=color, linewidth=2, label=phase)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Altitude [1000 ft]')
        ax.set_title('Altitude vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 5: Thrust vs Distance
        ax = axes[2, 0]
        for phase, color in colors.items():
            mask = self.phase == phase
            if np.any(mask):
                ax.plot(self.distance_nm[mask], self.thrust_kn[mask], 
                       color=color, linewidth=2, label=phase)
        ax.set_xlabel('Distance [nm]')
        ax.set_ylabel('Thrust [kN]')
        ax.set_title('Thrust Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 6: TSFC vs Distance
        ax = axes[2, 1]
        for phase, color in colors.items():
            mask = self.phase == phase
            if np.any(mask):
                ax.plot(self.distance_nm[mask], self.tsfc_kg_n_hr[mask], 
                       color=color, linewidth=2, label=phase)
        ax.set_xlabel('Distance [nm]')
        ax.set_ylabel('TSFC [kg/(N·hr)]')
        ax.set_title('TSFC Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig, axes
        def save_plots(self, base_path: str):
            """
            Save all plots to files. base_path is used as prefix for filenames.
            """
            self.plot(save_path=f"{base_path}_grid.png")
            self.plot_altitude_profile(save_path=f"{base_path}_altitude.png")
            self.plot_fuel_burn(save_path=f"{base_path}_fuelburn.png")
            self.plot_thrust(save_path=f"{base_path}_thrust.png")
    
    def plot_altitude_profile(self, save_path: str = None):
        """Plot altitude profile vs distance."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'CLB': 'blue', 'CRZ': 'green', 'DES': 'orange'}
        
        for phase, color in colors.items():
            mask = self.phase == phase
            if np.any(mask):
                ax.plot(self.distance_nm[mask], self.altitude_ft[mask] / 1000.0,
                       color=color, linewidth=2, label=phase)
        
        ax.set_xlabel('Distance [nm]')
        ax.set_ylabel('Altitude [1000 ft]')
        ax.set_title(f'{self.aircraft.name} - Altitude Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig, ax
    
    def plot_fuel_burn(self, save_path: str = None):
        """Plot cumulative fuel burn vs distance."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'CLB': 'blue', 'CRZ': 'green', 'DES': 'orange'}
        
        for phase, color in colors.items():
            mask = self.phase == phase
            if np.any(mask):
                ax.plot(self.distance_nm[mask], self.fuel_burned_tonnes[mask],
                       color=color, linewidth=2, label=phase)
        
        ax.set_xlabel('Distance [nm]')
        ax.set_ylabel('Fuel Burned [tonnes]')
        ax.set_title(f'{self.aircraft.name} - Fuel Burn')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig, ax
    
    def plot_thrust(self, save_path: str = None):
        """Plot thrust profile vs distance."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'CLB': 'blue', 'CRZ': 'green', 'DES': 'orange'}
        
        for phase, color in colors.items():
            mask = self.phase == phase
            if np.any(mask):
                ax.plot(self.distance_nm[mask], self.thrust_kn[mask],
                       color=color, linewidth=2, label=phase)
        
        ax.set_xlabel('Distance [nm]')
        ax.set_ylabel('Thrust [kN]')
        ax.set_title(f'{self.aircraft.name} - Thrust Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig, ax
    
    def to_dict(self) -> dict:
        """Return raw results dictionary."""
        return self._raw.copy()
    
    def to_csv(self, filename: str):
        """
        Save results to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'mass_kg', 'altitude_m', 'distance_m', 
                           'phase', 'thrust_N', 'tsfc_kg_N_s'])
            
            for i in range(len(self.t)):
                writer.writerow([
                    self.t[i],
                    self.m[i],
                    self.h[i],
                    self.d[i],
                    self.phase[i],
                    self.thrust[i],
                    self.tsfc[i]
                ])
