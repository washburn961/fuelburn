from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
from ambiance import Atmosphere
from scipy.integrate import solve_ivp

# ---------- Profiles ----------

@dataclass
class ClimbProfile:
    initial_speed_kcas: float = 240.0   # knots CAS below crossover
    crossover_speed_kcas: float = 270.0 # knots CAS at crossover altitude
    crossover_mach: float = 0.56        # Mach above crossover

@dataclass
class CruiseProfile:
    altitude_ft: float = 35000.0
    mach: float = 0.76

@dataclass
class DescentProfile:
    initial_speed_kcas: float = 240.0
    crossover_speed_kcas: float = 270.0
    crossover_mach: float = 0.56

@dataclass
class SpeedProfile:
    climb: Optional[ClimbProfile] = None
    cruise: Optional[CruiseProfile] = None
    descent: Optional[DescentProfile] = None
    transition_altitude_ft: float = 10000.0

    def _ft_to_m(self, ft: float) -> float:
        return ft / 3.281

    def _m_to_ft(self, m: float) -> float:
        return m * 3.281
    
    def _kts_to_mps(self, kts: float) -> float:
        return kts * 0.5144444444444445

    def _mps_to_kts(self, ms: float) -> float:
        return ms / 0.5144444444444445

    def _cas_to_eas(self, v_kcas: float) -> float:
        """
        Convert CAS (in m/s) to EAS (in m/s) using the correct compressible formula.
        Note: CAS→EAS uses SEA-LEVEL standard conditions only; it does NOT depend on altitude.
            The h_m argument is unused (kept for API consistency).
        Steps:
        1) Convert KCAS → CAS [m/s]
        2) Compute impact pressure at SL: q_c = p0 * [ (1 + (γ-1)/2 * (CAS/a0)^2)^(γ/(γ-1)) - 1 ]
        3) EAS = sqrt( 2 * q_c / ρ0 )
        """
        # SEA-LEVEL standard constants for CAS/EAS
        p0 = 101325.0         # Pa
        rho0 = 1.225          # kg/m^3
        gamma = 1.4

        V_cas = v_kcas  # m/s

        # We can compute q_c without explicitly using a0 if we write it in terms of M,
        # but using a0 is fine. Get a0 from ISA sea level:
        atm0 = Atmosphere([0.0])
        a0 = float(atm0.speed_of_sound[0])  # m/s

        # Impact (pitot) pressure at sea level, compressible isentropic
        # qc = p0 * [ (1 + (γ-1)/2 * (V_cas/a0)^2)^(γ/(γ-1)) - 1 ]
        term = 1.0 + 0.5 * (gamma - 1.0) * (V_cas / a0) ** 2
        qc = p0 * (term ** (gamma / (gamma - 1.0)) - 1.0)

        # EAS from qc at sea level
        V_eas = np.sqrt(2.0 * qc / rho0)
        return V_eas
    
    def _eas_to_tas(self, v_eas: float, h_m: float) -> float:
        """
        Convert EAS [m/s] to TAS [m/s] at altitude h_m using: TAS = EAS * sqrt(rho0 / rho(h)).
        This relation is exact by definition of EAS.
        """

        rho0 = 1.225
        atm = Atmosphere(h_m)
        rho = float(np.atleast_1d(atm.density)[0])

        return v_eas * np.sqrt(rho0 / rho)
    
    def _cas_to_tas(self, v_cas: float, h_m: float) -> float:
        return self._eas_to_tas(v_eas=self._cas_to_eas(v_cas), h_m=h_m)
    
    # def _get_crossover_alt(self, crossover_kcas: float, crossover_mach: float) -> float:

    #     h = self._ft_to_m(self.transition_altitude_ft)
    #     V_x = self._eas_to_tas(self._cas_to_eas(crossover_kcas), h)
    #     atm = Atmosphere(h)
    #     a = atm.speed_of_sound

    #     return 0.0

    def get_speed(self, h_m: float, phase: FlightPhase) -> float:
        """
        Return commanded TAS [m/s] at altitude h_m for phase.
        Uses a simple rule:
        - Below crossover: fly KCAS (converted to TAS using density ratio, incompressible approx).
        - Above crossover: fly constant Mach.
        """

        # Ensure h_m is scalar for comparisons
        h_m_scalar = float(np.atleast_1d(h_m)[0])

        # Atmosphere at current altitude
        atm = Atmosphere(h_m_scalar)
        a = float(np.atleast_1d(atm.speed_of_sound)[0])

        # Select profile by phase
        if phase == FlightPhase.CLB:
            prof = self.climb
        elif phase == FlightPhase.CRZ:
            prof = self.cruise
            return prof.mach * a
        elif phase == FlightPhase.DES:
            prof = self.descent
        else:
            raise ValueError(f"No speed profile defined for {phase}")
        
        if (h_m_scalar < self._ft_to_m(self.transition_altitude_ft)):
            initial_speed_mps = self._kts_to_mps(prof.initial_speed_kcas)
            return self._cas_to_tas(v_cas=initial_speed_mps, h_m=h_m_scalar)
        
        crossover_speed_mps = self._kts_to_mps(prof.crossover_speed_kcas)
        V = self._cas_to_tas(v_cas=crossover_speed_mps, h_m=h_m_scalar)
        mach = V / a
        abs_tol = 0.01  # dimensionless Mach tolerance

        if (abs(mach - prof.crossover_mach) < abs_tol):
            return prof.crossover_mach * a
        else:
            return V

class FlightPhase(str, Enum):
    CLB = "clb"
    CRZ = "crz"
    DES = "des"

# ---------- Models ----------

from typing import Callable, Optional
from ambiance import Atmosphere
import numpy as np

class Aerodynamics:
    """
    Aerodynamics model with pluggable drag function.

    Default drag model:
        D = q S (CD0 + k * CL^2), with CL = L / (q S) and L ≈ W (small-γ)

    You can pass your own drag function via `drag_fn` with the signature:
        drag_fn(mach: float, altitude_m: float, weight: float) -> float

    Notes:
    - 'weight' is a reserved argument (N). For small γ, weight ≈ lift.
    - Atmosphere properties are looked up internally when using the default model.
    """

    def __init__(
        self,
        S: float,
        CD0: float,
        k: float,
        drag_fn: Optional[Callable[[float, float, float], float]] = None
    ):
        """
        Args:
            S    : wing reference area [m^2]
            CD0  : zero-lift drag coefficient
            k    : induced-drag factor (≈ 1/(π e AR))
            drag_fn : optional user-defined drag function with signature
                      drag_fn(mach, altitude_m, weight) -> drag [N]
        """
        self.S = float(S)
        self.CD0 = float(CD0)
        self.k = float(k)
        self._drag_fn = drag_fn  # user-defined, may be None

    def drag(self, mach: float, altitude_m: float, weight: float) -> float:
        """
        Compute drag [N] at (Mach, altitude) for given weight [N].

        Signature is intentionally:
            (mach, altitude_m, weight) -> float

        If a custom drag_fn was provided in the constructor, it is used.
        Otherwise, the default parasitic + induced drag model is used.

        Returns:
            D [N]
        """
        if self._drag_fn is not None:
            return float(self._drag_fn(mach, altitude_m, weight))
        else:
            return float(self._default_drag(mach, altitude_m, weight))

    # -------------------- Defaults & helpers --------------------

    def _default_drag(self, mach: float, altitude_m: float, weight: float) -> float:
        """
        Parasitic + induced drag with small-γ lift ≈ weight:
            D = q S (CD0 + k * CL^2),  CL = L/(qS),  L ≈ W
        """
        # Atmosphere at altitude
        atm = Atmosphere([altitude_m])
        a = float(atm.speed_of_sound[0])   # m/s
        rho = float(atm.density[0])        # kg/m^3

        V = mach * a                       # TAS [m/s]
        q = 0.5 * rho * V * V              # dynamic pressure [Pa == N/m^2]

        # small-γ => Lift ~ Weight
        L = float(weight)

        CL = L / (q * self.S)
        CD = self.CD0 + self.k * CL * CL
        D = q * self.S * CD
        return D


class Propulsion:
    """
    Propulsion using Howe-style turbofan correlations for TSFC and thrust lapse.

    Constructor arguments:
      - BPR: float
          Bypass ratio (unitless).
      - Tmax_sl_total: float
          Total static sea-level thrust across all engines [N].
      - TSFC_sl: float
          Base TSFC at static sea-level, static condition [kg/(N·s)].

    Public API (use in ODEs):
      - thrust(mach: float, altitude_m: float, thrust_lever: float = 1.0) -> float
      - TSFC(mach: float, altitude_m: float) -> float
    """

    def __init__(self, BPR: float, Tmax_sl_total: float, TSFC_sl: float):
        self.BPR = float(BPR)
        self.T_sl_static_total = float(Tmax_sl_total)
        self.TSFC_sl = float(TSFC_sl)
        # Reference sea-level density used in the original correlation (Howe)
        self.rho0_ref = 1.225

    # -------------------- internals --------------------

    def _ambient_density(self, altitude_m: float) -> float:
        atm = Atmosphere(altitude_m)
        return float(np.atleast_1d(atm.density)[0])

    def _tsfc(self, Mach: float, altitude_m: float) -> float:
        """
        TSFC at (Mach, altitude) using Howe-like correlation.
        Returns:
            C : TSFC [kg/(N·s)]
        """
        rho = self._ambient_density(altitude_m)
        sigma = rho / self.rho0_ref  # rho / 1.225 from Howe model

        BPR = self.BPR
        Cbase = self.TSFC_sl  # user-provided sea-level static TSFC

        # TSFC correlation (Howe-like)
        C = (
            1.9 * Cbase
            * (1.0 - 0.15 * (BPR ** 0.65))
            * (1.0 + 0.28 * (1.0 + 0.063 * (BPR ** 2)) * Mach)
            * (sigma ** 0.08)
        )

        return C


    def _kT(self, altitude_m: float) -> float:
        """
        Thrust correction factor vs static sea-level thrust.
        Howe linearized model.
        Returns:
            kT : dimensionless thrust lapse factor
        """
        BPR = self.BPR

        if BPR < 13.0:
            # altitude entered in km
            kT = (0.0013 * BPR - 0.0397) * (altitude_m / 1000.0) - 0.0248 * BPR + 0.7125
        else:
            kT = 0.2

        return kT * 1.25

    # -------------------- public API --------------------

    def thrust(self, mach: float, altitude_m: float, thrust_lever: float = 1.0) -> float:
        """
        Thrust at (Mach, altitude) with thrust lever angle fraction in [0,1].
        T = kT(h, BPR) * T_sl_static_total * TLA
        Result is clamped to >= 0.
        """
        kT = self._kT(altitude_m)
        TLA = float(np.clip(thrust_lever, 0.0, 1.0))
        T = kT * self.T_sl_static_total * TLA
        return max(0.0, T)

    def TSFC(self, mach: float, altitude_m: float) -> float:
        """
        TSFC at (Mach, altitude) in kg/(N·s).
        """
        C = self._tsfc(mach, altitude_m)
        return C


# ---------- ODEs ----------

def odefun_climb(
    t: float,
    state: np.ndarray,
    aerodynamics: Aerodynamics,
    propulsion: Propulsion,
    speed_schedule: SpeedProfile
) -> np.ndarray:
    """
    States:
      state[0] = m [kg]
      state[1] = h [m]
      state[2] = d [m]
    Returns:
      [dm/dt, dh/dt, dd/dt]
    """
    m, h, d = state

    # Atmosphere at h
    atm = Atmosphere(h)
    a   = float(np.atleast_1d(atm.speed_of_sound)[0])
    rho = float(np.atleast_1d(atm.density)[0])
    g   = 9.80665

    W = m * g

    # Commanded TAS from schedule
    V = speed_schedule.get_speed(h, FlightPhase.CLB)  # m/s
    M = V / a

    # Thrust at climb (e.g., 90% TLA)
    T = propulsion.thrust(mach=M, altitude_m=h, thrust_lever=1.0)  # N

    # Lift with small-gamma approx
    L = W

    # Drag
    D = aerodynamics.drag(mach=M, altitude_m=h, weight=W)  # N

    # Kinematics
    h_dot = (T - D) * V / W     # m/s (excess power / weight)
    d_dot = V                   # m/s (no wind, small gamma)

    # Fuel burn
    m_dot = -propulsion.TSFC(mach=M, altitude_m=h) * T  # kg/s

    return np.array([m_dot, h_dot, d_dot], dtype=float)

def odefun_cruise(
    t: float,
    state: np.ndarray,
    aerodynamics: Aerodynamics,
    propulsion: Propulsion,
    speed_schedule: SpeedProfile
) -> np.ndarray:

    m, h, d = state

    atm = Atmosphere(h)
    a   = float(np.atleast_1d(atm.speed_of_sound)[0])
    rho = float(np.atleast_1d(atm.density)[0])
    g   = 9.80665

    V = speed_schedule.get_speed(h, FlightPhase.CRZ)  # TAS [m/s]
    M = V/a
    L = m * g
    D = aerodynamics.drag(mach=M, altitude_m=h, weight=L)  # N
    T = D  # thrust balanced with drag in steady, level cruise

    m_dot = -propulsion.TSFC(mach=M, altitude_m=h) * T
    h_dot = 0.0
    d_dot = V

    return np.array([m_dot, h_dot, d_dot], dtype=float)

def odefun_descent(
    t: float,
    state: np.ndarray,
    aerodynamics: Aerodynamics,
    propulsion: Propulsion,
    speed_schedule: SpeedProfile
) -> np.ndarray:

    m, h, d = state

    atm = Atmosphere(h)
    a   = float(np.atleast_1d(atm.speed_of_sound)[0])
    rho = float(np.atleast_1d(atm.density)[0])
    g   = 9.80665

    W = m * g

    V = speed_schedule.get_speed(h, FlightPhase.DES)
    M = V / a

    # Idle-ish thrust
    T = propulsion.thrust(mach=M, altitude_m=h, thrust_lever=0.10)

    L = W
    D = aerodynamics.drag(mach=M, altitude_m=h, weight=L)  # N

    h_dot = (T - D) * V / W   # will be negative in descent when D > T
    d_dot = V

    m_dot = -propulsion.TSFC(mach=M, altitude_m=h) * T

    return np.array([m_dot, h_dot, d_dot], dtype=float)

# ---------- Simulation shell ----------

class Simulation:
    def __init__(self, aerodynamics: Aerodynamics, propulsion: Propulsion, speed_profile: SpeedProfile):
        self.aero = aerodynamics
        self.prop = propulsion
        self.speed_profile = speed_profile

    def _calculate_tod(self, cruise_altitude_ft: float, final_altitude_ft: float = 0.0) -> float:
        """
        Calculate Top of Descent (TOD) distance using the 3:1 rule of thumb.
        
        Rule: 3 nautical miles per 1000 ft of altitude loss.
        
        Args:
            cruise_altitude_ft: Cruise altitude [ft]
            final_altitude_ft: Target altitude for end of descent [ft]
        
        Returns:
            TOD distance [m]
        """
        altitude_loss_ft = cruise_altitude_ft - final_altitude_ft
        tod_distance_nm = 3.0 * (altitude_loss_ft / 1000.0)  # 3 nm per 1000 ft
        tod_distance_m = tod_distance_nm * 1852.0  # Convert nm to meters
        return tod_distance_m

    def run(
        self,
        initial_mass_kg: float,
        initial_altitude_m: float,
        cruise_distance_m: float,
        max_time_s: float = 10800.0  # 3 hours default
    ):
        """
        Run the simulation using scipy's solve_ivp with RK45 (ODE45 equivalent).
        
        Args:
            initial_mass_kg: Initial aircraft mass [kg]
            initial_altitude_m: Starting altitude [m] (typically low for takeoff)
            cruise_distance_m: Total mission distance [m]
            max_time_s: Maximum simulation time [s]
        
        Returns:
            dict: Simulation results with keys 't', 'm', 'h', 'd', 'phase'
        """
        
        # Convert cruise altitude from speed profile to meters
        cruise_alt_ft = self.speed_profile.cruise.altitude_ft
        cruise_alt_m = cruise_alt_ft / 3.281
        
        # Calculate TOD using 3:1 rule
        tod_distance_m = self._calculate_tod(cruise_alt_ft, final_altitude_ft=0.0)
        
        # Mission waypoints
        d_toc = 0.0  # Will be determined when reaching cruise altitude
        d_tod = cruise_distance_m - tod_distance_m
        
        # Storage for results
        results = {
            't': [],
            'm': [],
            'h': [],
            'd': [],
            'phase': [],
            'thrust': [],
            'tsfc': []
        }
        
        # Initial state: [mass, altitude, distance]
        state0 = np.array([initial_mass_kg, initial_altitude_m, 0.0])
        t0 = 0.0
        
        # ========== PHASE 1: CLIMB ==========
        print(f"Starting CLIMB from {initial_altitude_m:.0f} m to {cruise_alt_m:.0f} m")
        
        def climb_event(t, state):
            """Event: reached cruise altitude"""
            return state[1] - cruise_alt_m
        climb_event.terminal = True
        climb_event.direction = 1  # Trigger when crossing upward
        
        def climb_ode(t, state):
            return odefun_climb(t, state, self.aero, self.prop, self.speed_profile)
        
        sol_climb = solve_ivp(
            climb_ode,
            (t0, max_time_s),
            state0,
            method='RK45',
            events=climb_event,
            dense_output=True,
            max_step=30.0  # 30 second max time step
        )
        
        # Store climb results
        for i in range(len(sol_climb.t)):
            m_i = sol_climb.y[0, i]
            h_i = sol_climb.y[1, i]
            d_i = sol_climb.y[2, i]
            
            # Calculate thrust and TSFC
            V_i = self.speed_profile.get_speed(h_i, FlightPhase.CLB)
            atm = Atmosphere(h_i)
            a_i = float(np.atleast_1d(atm.speed_of_sound)[0])
            M_i = V_i / a_i
            T_i = self.prop.thrust(mach=M_i, altitude_m=h_i, thrust_lever=1)
            tsfc_i = self.prop.TSFC(mach=M_i, altitude_m=h_i)
            
            results['t'].append(sol_climb.t[i])
            results['m'].append(m_i)
            results['h'].append(h_i)
            results['d'].append(d_i)
            results['phase'].append('CLB')
            results['thrust'].append(T_i)
            results['tsfc'].append(tsfc_i)
        
        # Get state at top of climb (TOC)
        if sol_climb.t_events[0].size > 0:
            t_toc = sol_climb.t_events[0][0]
            state_toc = sol_climb.sol(t_toc)
            d_toc = state_toc[2]  # Distance at TOC
            print(f"Reached TOC at t={t_toc:.0f} s, d={d_toc/1000:.1f} km")
        else:
            print("Warning: Did not reach cruise altitude")
            return results
        
        # ========== PHASE 2: CRUISE ==========
        cruise_duration_m = d_tod - d_toc
        print(f"Starting CRUISE for {cruise_duration_m/1000:.1f} km (TOD at {d_tod/1000:.1f} km)")
        
        def cruise_event(t, state):
            """Event: reached TOD"""
            return state[2] - d_tod
        cruise_event.terminal = True
        cruise_event.direction = 1
        
        def cruise_ode(t, state):
            return odefun_cruise(t, state, self.aero, self.prop, self.speed_profile)
        
        sol_cruise = solve_ivp(
            cruise_ode,
            (t_toc, max_time_s),
            state_toc,
            method='RK45',
            events=cruise_event,
            dense_output=True,
            max_step=60.0  # 60 second max time step for cruise
        )
        
        # Store cruise results (skip first point to avoid duplication)
        for i in range(1, len(sol_cruise.t)):
            m_i = sol_cruise.y[0, i]
            h_i = sol_cruise.y[1, i]
            d_i = sol_cruise.y[2, i]
            
            # Calculate thrust and TSFC
            V_i = self.speed_profile.get_speed(h_i, FlightPhase.CRZ)
            atm = Atmosphere(h_i)
            a_i = float(np.atleast_1d(atm.speed_of_sound)[0])
            M_i = V_i / a_i
            # In cruise, thrust equals drag
            rho_i = float(np.atleast_1d(atm.density)[0])
            L_i = m_i * 9.80665
            D_i = self.aero.drag(mach=M_i, altitude_m=h_i, weight=L_i)
            T_i = D_i
            tsfc_i = self.prop.TSFC(mach=M_i, altitude_m=h_i)
            
            results['t'].append(sol_cruise.t[i])
            results['m'].append(m_i)
            results['h'].append(h_i)
            results['d'].append(d_i)
            results['phase'].append('CRZ')
            results['thrust'].append(T_i)
            results['tsfc'].append(tsfc_i)
        
        # Get state at TOD
        if sol_cruise.t_events[0].size > 0:
            t_tod = sol_cruise.t_events[0][0]
            state_tod = sol_cruise.sol(t_tod)
            print(f"Reached TOD at t={t_tod:.0f} s, d={state_tod[2]/1000:.1f} km")
        else:
            print("Warning: Did not reach TOD")
            return results
        
        # ========== PHASE 3: DESCENT ==========
        print(f"Starting DESCENT from {cruise_alt_m:.0f} m to ground")
        
        def descent_event(t, state):
            """Event: reached ground level (or near zero altitude)"""
            return state[1] - 100.0  # Stop at 100 m AGL
        descent_event.terminal = True
        descent_event.direction = -1  # Trigger when crossing downward
        
        def descent_ode(t, state):
            return odefun_descent(t, state, self.aero, self.prop, self.speed_profile)
        
        sol_descent = solve_ivp(
            descent_ode,
            (t_tod, max_time_s),
            state_tod,
            method='RK45',
            events=descent_event,
            dense_output=True,
            max_step=30.0  # 30 second max time step
        )
        
        # Store descent results (skip first point to avoid duplication)
        for i in range(1, len(sol_descent.t)):
            m_i = sol_descent.y[0, i]
            h_i = sol_descent.y[1, i]
            d_i = sol_descent.y[2, i]
            
            # Calculate thrust and TSFC
            V_i = self.speed_profile.get_speed(h_i, FlightPhase.DES)
            atm = Atmosphere(h_i)
            a_i = float(np.atleast_1d(atm.speed_of_sound)[0])
            M_i = V_i / a_i
            T_i = self.prop.thrust(mach=M_i, altitude_m=h_i, thrust_lever=0.15)
            tsfc_i = self.prop.TSFC(mach=M_i, altitude_m=h_i)
            
            results['t'].append(sol_descent.t[i])
            results['m'].append(m_i)
            results['h'].append(h_i)
            results['d'].append(d_i)
            results['phase'].append('DES')
            results['thrust'].append(T_i)
            results['tsfc'].append(tsfc_i)
        
        if sol_descent.t_events[0].size > 0:
            t_final = sol_descent.t_events[0][0]
            state_final = sol_descent.sol(t_final)
            print(f"Landed at t={t_final:.0f} s, d={state_final[2]/1000:.1f} km")
            print(f"Final mass: {state_final[0]:.0f} kg")
            print(f"Fuel burned: {initial_mass_kg - state_final[0]:.0f} kg")
        else:
            print("Warning: Did not complete descent")
        
        return results