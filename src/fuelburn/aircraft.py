"""
Aircraft class for simplified aircraft definition and simulation.
"""

from dataclasses import dataclass
from typing import Optional, Callable, TYPE_CHECKING, Dict, Any
import contextlib
import io
import math
import numpy as np

from .simulation import (
    Aerodynamics,
    Propulsion,
    SpeedProfile,
    ClimbProfile,
    CruiseProfile,
    DescentProfile,
    TaxiProfile,
    Simulation
)

if TYPE_CHECKING:
    from .mission import Mission
    from .results import Results


@dataclass
class RequiredFuelSolution:
    """Solution payload returned by Aircraft.solve_required_fuel()."""

    converged: bool
    method: str
    iterations: int
    takeoff_weight_kg: float
    required_fuel_kg: float
    primary_block_fuel_kg: float
    primary_trip_fuel_kg: float
    primary_taxi_fuel_kg: float
    alternate_fuel_kg: float
    alternate_block_fuel_kg: float
    alternate_trip_fuel_kg: float
    alternate_taxi_fuel_kg: float
    alternate_block_time_min: float
    alternate_air_time_min: float
    alternate_distance_flown_nm: float
    reserve_fuel_kg: float
    reserve_flow_kg_hr: float
    contingency_fuel_kg: float
    contingency_method: str
    contingency_minutes_used: float
    contingency_fraction_used: float
    primary_results: 'Results'
    alternate_results: 'Results'

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable summary without raw Results objects."""
        return {
            'converged': self.converged,
            'method': self.method,
            'iterations': self.iterations,
            'takeoff_weight_kg': self.takeoff_weight_kg,
            'required_fuel_kg': self.required_fuel_kg,
            'primary_block_fuel_kg': self.primary_block_fuel_kg,
            'primary_trip_fuel_kg': self.primary_trip_fuel_kg,
            'primary_taxi_fuel_kg': self.primary_taxi_fuel_kg,
            'alternate_fuel_kg': self.alternate_fuel_kg,
            'alternate_block_fuel_kg': self.alternate_block_fuel_kg,
            'alternate_trip_fuel_kg': self.alternate_trip_fuel_kg,
            'alternate_taxi_fuel_kg': self.alternate_taxi_fuel_kg,
            'alternate_block_time_min': self.alternate_block_time_min,
            'alternate_air_time_min': self.alternate_air_time_min,
            'alternate_distance_flown_nm': self.alternate_distance_flown_nm,
            'reserve_fuel_kg': self.reserve_fuel_kg,
            'reserve_flow_kg_hr': self.reserve_flow_kg_hr,
            'contingency_fuel_kg': self.contingency_fuel_kg,
            'contingency_method': self.contingency_method,
            'contingency_minutes_used': self.contingency_minutes_used,
            'contingency_fraction_used': self.contingency_fraction_used,
        }

class Aircraft:
    """
    Aircraft class for fuel burn simulation.
    
    Usage:
        # From preset
        aircraft = Aircraft.from_preset('B737-800')
        
        # Custom
        aircraft = Aircraft(
            name='My Aircraft',
            mtow_kg=73000,
            wing_area_m2=122.6,
            ...
        )
        
        # Fly mission
        results = aircraft.fly(mission)
    """
    
    def __init__(
        self,
        name: str,
        wing_area_m2: float,
        thrust_per_engine_N: float,
        num_engines: int = 2,
        bypass_ratio: float = 5.5,
        cd0: float = 0.024,
        oswald_efficiency: float = 0.85,
        max_cruise_mach: float = 0.82,
        max_cruise_altitude_ft: float = 41000.0,
        typical_climb_kcas_low: float = 250.0,
        typical_climb_kcas_high: float = 280.0,
        typical_climb_mach: float = 0.78,
        tsfc_sl: float = 2e-5,
        climb_tla: float = 1.0,
        descent_tla: float = 0.10,
        taxi_tla: float = 0.07,
        drag_fn: Optional[Callable] = None
    ):
        """
        Initialize aircraft.
        
        Args:
            name: Aircraft name
            wing_area_m2: Wing reference area [m²]
            thrust_per_engine_N: Thrust per engine at sea level [N]
            num_engines: Number of engines
            bypass_ratio: Engine bypass ratio
            cd0: Zero-lift drag coefficient
            oswald_efficiency: Oswald efficiency factor
            max_cruise_mach: Maximum cruise Mach number
            max_cruise_altitude_ft: Maximum cruise altitude [ft]
            typical_climb_kcas_low: Climb speed below 10,000 ft [KCAS]
            typical_climb_kcas_high: Climb speed at crossover [KCAS]
            typical_climb_mach: Climb Mach above crossover
            tsfc_sl: Sea-level TSFC [kg/(N·s)]
            climb_tla: Climb thrust lever angle [0-1]
            descent_tla: Descent thrust lever angle [0-1]
            taxi_tla: Taxi thrust lever angle [0-1]
            drag_fn: Optional custom drag function
        """
        self.name = name
        self.wing_area_m2 = wing_area_m2
        self.thrust_per_engine_N = thrust_per_engine_N
        self.num_engines = num_engines
        self.bypass_ratio = bypass_ratio
        self.cd0 = cd0
        self.oswald_efficiency = oswald_efficiency
        self.max_cruise_mach = max_cruise_mach
        self.max_cruise_altitude_ft = max_cruise_altitude_ft
        self.typical_climb_kcas_low = typical_climb_kcas_low
        self.typical_climb_kcas_high = typical_climb_kcas_high
        self.typical_climb_mach = typical_climb_mach
        self.tsfc_sl = tsfc_sl
        self.climb_tla = climb_tla
        self.descent_tla = descent_tla
        self.taxi_tla = taxi_tla
        self._custom_drag_fn = drag_fn
        
        # Computed properties
        self.total_thrust_N = thrust_per_engine_N * num_engines
        
        # Calculate induced drag factor: k = 1/(π * e * AR)
        # Assume typical aspect ratio if not provided
        aspect_ratio = 9.0  # Typical for commercial jets
        self.k = 1.0 / (np.pi * oswald_efficiency * aspect_ratio)
        
        # Create aerodynamics model
        self.aerodynamics = Aerodynamics(
            S=wing_area_m2,
            CD0=cd0,
            k=self.k,
            drag_fn=drag_fn
        )
        
        # Create propulsion model
        self.propulsion = Propulsion(
            BPR=bypass_ratio,
            Tmax_sl_total=self.total_thrust_N,
            TSFC_sl=tsfc_sl,
            climb_tla=climb_tla,
            descent_tla=descent_tla,
            taxi_tla=taxi_tla
        )
    
    @classmethod
    def from_preset(cls, aircraft_type: str) -> 'Aircraft':
        """
        Create aircraft from preset.
        
        Args:
            aircraft_type: Aircraft type (e.g., 'B737-800', 'A320', 'B777-300ER')
        
        Returns:
            Aircraft instance
        """
        from .presets import get_aircraft_preset
        spec = get_aircraft_preset(aircraft_type)
        
        return cls(
            name=spec.name,
            wing_area_m2=spec.wing_area_m2,
            thrust_per_engine_N=spec.thrust_per_engine_N,
            num_engines=spec.num_engines,
            bypass_ratio=spec.bypass_ratio,
            cd0=spec.cd0,
            oswald_efficiency=spec.oswald_efficiency,
            max_cruise_mach=spec.max_cruise_mach,
            max_cruise_altitude_ft=spec.max_cruise_altitude_ft,
            typical_climb_kcas_low=spec.typical_climb_kcas_low,
            typical_climb_kcas_high=spec.typical_climb_kcas_high,
            typical_climb_mach=spec.typical_climb_mach,
            tsfc_sl=spec.tsfc_sl,
        )
    
    def fly(self, mission, max_time_s: float = 14400.0, verbose: bool = True):
        """
        Fly a mission and return results.
        
        Args:
            mission: Mission instance
            max_time_s: Maximum simulation time [s]
            verbose: Print progress messages
        
        Returns:
            Results instance
        """
        from .mission import Mission
        from .results import Results
        
        if not isinstance(mission, Mission):
            raise TypeError("mission must be a Mission instance")

        if mission.initial_weight_kg is None:
            raise ValueError(
                "mission.initial_weight_kg must be provided for fly(). "
                "Use solve_required_fuel() to size fuel from empty weight + payload."
            )

        if mission.initial_weight_kg <= 0.0:
            raise ValueError("mission.initial_weight_kg must be positive")
        
        # Build speed profile from mission
        climb_prof = ClimbProfile(
            initial_speed_kcas=self.typical_climb_kcas_low,
            crossover_speed_kcas=self.typical_climb_kcas_high,
            crossover_mach=self.typical_climb_mach
        )
        
        cruise_prof = CruiseProfile(
            altitude_ft=mission.cruise_altitude_ft,
            mach=mission.cruise_mach
        )
        
        descent_prof = DescentProfile(
            initial_speed_kcas=self.typical_climb_kcas_low,
            crossover_speed_kcas=self.typical_climb_kcas_high,
            crossover_mach=self.typical_climb_mach
        )
        
        taxi_prof = TaxiProfile(
            taxi_out_time_s=mission.taxi_out_time_s,
            taxi_in_time_s=mission.taxi_in_time_s,
            taxi_tla=self.taxi_tla
        )
        
        speed_profile = SpeedProfile(
            climb=climb_prof,
            cruise=cruise_prof,
            descent=descent_prof,
            taxi=taxi_prof,
            transition_altitude_ft=10000.0
        )
        
        # Use only takeoff weight from mission
        initial_mass_kg = mission.initial_weight_kg
        
        # Create simulation
        sim = Simulation(self.aerodynamics, self.propulsion, speed_profile)
        
        # Run simulation
        if verbose:
            print(f"Flying {self.name} on {mission.distance_nm:.0f} nm mission...")
        
        raw_results = sim.run(
            initial_mass_kg=initial_mass_kg,
            initial_altitude_m=mission.initial_altitude_m,
            cruise_distance_m=mission.distance_m,
            max_time_s=max_time_s
        )
        
        # Wrap in Results object
        return Results(
            raw_results=raw_results,
            aircraft=self,
            mission=mission,
            initial_mass_kg=initial_mass_kg
        )

    def solve_required_fuel(
        self,
        mission,
        empty_weight_kg: float,
        payload_kg: float,
        alternate_mission,
        reserve_minutes: float = 45.0,
        contingency_fraction: Optional[float] = None,
        contingency_minutes: Optional[float] = None,
        tol_kg: float = 10.0,
        max_iterations: int = 12,
        bisection_max_iterations: int = 30,
        max_time_s: float = 14400.0,
        verbose: bool = True
    ) -> RequiredFuelSolution:
        """
        Solve required departure fuel using FAA Part 121 domestic-style components.

        Required fuel is defined as:
            primary block burn + alternate trip burn + final reserve + contingency

        The method first attempts fixed-point iteration and falls back to bisection
        if fixed-point does not converge.

        Args:
            mission: Primary destination mission profile (no initial weight required)
            empty_weight_kg: Operating empty weight [kg]
            payload_kg: Payload mass [kg]
            alternate_mission: Alternate-route mission profile (required)
            reserve_minutes: Final reserve time at normal cruise fuel flow [min]
            contingency_fraction: Fraction contingency on primary trip fuel (exclusive with contingency_minutes)
            contingency_minutes: Time-based contingency at normal cruise flow [min] (exclusive with contingency_fraction)
            tol_kg: Convergence tolerance on fuel [kg]
            max_iterations: Maximum fixed-point iterations
            bisection_max_iterations: Maximum bisection iterations
            max_time_s: Max integration horizon passed to fly()
            verbose: Print solver progress

        Returns:
            RequiredFuelSolution with component breakdown and diagnostic Results.
        """
        from .mission import Mission

        if not isinstance(mission, Mission):
            raise TypeError("mission must be a Mission instance")

        if not isinstance(alternate_mission, Mission):
            raise TypeError("alternate_mission must be a Mission instance")

        if empty_weight_kg <= 0.0:
            raise ValueError("empty_weight_kg must be positive")

        if payload_kg < 0.0:
            raise ValueError("payload_kg must be non-negative")

        if reserve_minutes < 0.0:
            raise ValueError("reserve_minutes must be non-negative")

        if contingency_fraction is not None and contingency_minutes is not None:
            raise ValueError(
                "Only one of contingency_fraction or contingency_minutes may be specified"
            )

        if contingency_fraction is None and contingency_minutes is None:
            # Preserve previous default behavior.
            contingency_fraction = 0.05

        if contingency_fraction is not None and contingency_fraction < 0.0:
            raise ValueError("contingency_fraction must be non-negative")

        if contingency_minutes is not None and contingency_minutes < 0.0:
            raise ValueError("contingency_minutes must be non-negative")

        if tol_kg <= 0.0:
            raise ValueError("tol_kg must be positive")

        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        if bisection_max_iterations <= 0:
            raise ValueError("bisection_max_iterations must be positive")

        def _clone_mission_with_weight(base_mission: Mission, initial_weight_kg: float) -> Mission:
            return Mission(
                distance_nm=base_mission.distance_nm,
                cruise_altitude_ft=base_mission.cruise_altitude_ft,
                cruise_mach=base_mission.cruise_mach,
                initial_weight_kg=initial_weight_kg,
                initial_altitude_m=base_mission.initial_altitude_m,
                taxi_out_time_s=base_mission.taxi_out_time_s,
                taxi_in_time_s=base_mission.taxi_in_time_s,
            )

        def _run_for_solver(mission_for_run: Mission):
            # Suppress verbose integration prints inside iterative solver.
            with contextlib.redirect_stdout(io.StringIO()):
                return self.fly(mission_for_run, max_time_s=max_time_s, verbose=False)

        def _validate_route_completion(results, planned_mission: Mission, label: str):
            if planned_mission.distance_m > 0.0:
                distance_ratio = results.distance_flown_m / planned_mission.distance_m
                if distance_ratio < 0.97:
                    raise RuntimeError(
                        f"{label} mission did not complete route: "
                        f"{results.distance_flown_nm:.1f} nm flown / "
                        f"{planned_mission.distance_nm:.1f} nm planned."
                    )

            phase_set = set(results.phase.tolist())
            missing_required = [phase for phase in ("CLB", "DES") if phase not in phase_set]
            if missing_required:
                missing_fmt = ", ".join(missing_required)
                raise RuntimeError(
                    f"{label} mission did not complete required phases: {missing_fmt}."
                )

            if results.final_mass_kg <= 0.0:
                raise RuntimeError(f"{label} mission ended with non-physical final mass.")

        def _evaluate_guess(fuel_guess_kg: float) -> Dict[str, Any]:
            if fuel_guess_kg < 0.0:
                raise ValueError("Fuel guess must be non-negative")

            takeoff_weight_kg = empty_weight_kg + payload_kg + fuel_guess_kg
            if takeoff_weight_kg <= 0.0:
                raise RuntimeError("Computed takeoff weight is non-positive")

            primary_input = _clone_mission_with_weight(mission, initial_weight_kg=takeoff_weight_kg)
            primary_results = _run_for_solver(primary_input)
            _validate_route_completion(primary_results, mission, label="Primary")

            # Option B handoff from the plan: alternate starts at landing mass before taxi-in.
            alternate_departure_mass_kg = primary_results.landing_mass_kg
            alternate_input = _clone_mission_with_weight(
                alternate_mission,
                initial_weight_kg=alternate_departure_mass_kg,
            )
            alternate_results = _run_for_solver(alternate_input)
            _validate_route_completion(alternate_results, alternate_mission, label="Alternate")

            reserve_flow_kg_hr = primary_results.normal_cruise_fuel_flow_kg_hr
            if not np.isfinite(reserve_flow_kg_hr) or reserve_flow_kg_hr <= 0.0:
                raise RuntimeError("Unable to derive positive reserve fuel flow from primary mission")

            reserve_fuel_kg = reserve_flow_kg_hr * (reserve_minutes / 60.0)
            alternate_block_fuel_kg = alternate_results.fuel_burned_kg
            alternate_trip_fuel_kg = alternate_results.trip_fuel_kg
            alternate_taxi_fuel_kg = alternate_results.total_taxi_fuel_kg
            alternate_fuel_kg = alternate_trip_fuel_kg

            if contingency_minutes is not None:
                contingency_method = 'minutes'
                contingency_minutes_used = contingency_minutes
                contingency_fraction_used = 0.0
                contingency_fuel_kg = reserve_flow_kg_hr * (contingency_minutes / 60.0)
            else:
                contingency_method = 'fraction'
                contingency_minutes_used = 0.0
                contingency_fraction_used = float(contingency_fraction)
                contingency_fuel_kg = contingency_fraction_used * primary_results.trip_fuel_kg

            required_fuel_kg = (
                primary_results.fuel_burned_kg
                + alternate_fuel_kg
                + reserve_fuel_kg
                + contingency_fuel_kg
            )

            return {
                'required_fuel_kg': required_fuel_kg,
                'takeoff_weight_kg': takeoff_weight_kg,
                'primary_block_fuel_kg': primary_results.fuel_burned_kg,
                'primary_trip_fuel_kg': primary_results.trip_fuel_kg,
                'primary_taxi_fuel_kg': primary_results.total_taxi_fuel_kg,
                'alternate_fuel_kg': alternate_fuel_kg,
                'alternate_block_fuel_kg': alternate_block_fuel_kg,
                'alternate_trip_fuel_kg': alternate_trip_fuel_kg,
                'alternate_taxi_fuel_kg': alternate_taxi_fuel_kg,
                'alternate_block_time_min': alternate_results.block_time_min,
                'alternate_air_time_min': alternate_results.air_time_min,
                'alternate_distance_flown_nm': alternate_results.distance_flown_nm,
                'reserve_fuel_kg': reserve_fuel_kg,
                'reserve_flow_kg_hr': reserve_flow_kg_hr,
                'contingency_fuel_kg': contingency_fuel_kg,
                'contingency_method': contingency_method,
                'contingency_minutes_used': contingency_minutes_used,
                'contingency_fraction_used': contingency_fraction_used,
                'primary_results': primary_results,
                'alternate_results': alternate_results,
            }

        def _safe_residual(fuel_guess_kg: float):
            try:
                evaluation = _evaluate_guess(fuel_guess_kg)
            except (RuntimeError, ValueError):
                return -math.inf, None

            return fuel_guess_kg - evaluation['required_fuel_kg'], evaluation

        def _build_solution(
            evaluation: Dict[str, Any],
            method: str,
            iterations: int,
            converged: bool
        ) -> RequiredFuelSolution:
            required_fuel_kg = evaluation['required_fuel_kg']
            takeoff_weight_kg = empty_weight_kg + payload_kg + required_fuel_kg
            return RequiredFuelSolution(
                converged=converged,
                method=method,
                iterations=iterations,
                takeoff_weight_kg=takeoff_weight_kg,
                required_fuel_kg=required_fuel_kg,
                primary_block_fuel_kg=evaluation['primary_block_fuel_kg'],
                primary_trip_fuel_kg=evaluation['primary_trip_fuel_kg'],
                primary_taxi_fuel_kg=evaluation['primary_taxi_fuel_kg'],
                alternate_fuel_kg=evaluation['alternate_fuel_kg'],
                alternate_block_fuel_kg=evaluation['alternate_block_fuel_kg'],
                alternate_trip_fuel_kg=evaluation['alternate_trip_fuel_kg'],
                alternate_taxi_fuel_kg=evaluation['alternate_taxi_fuel_kg'],
                alternate_block_time_min=evaluation['alternate_block_time_min'],
                alternate_air_time_min=evaluation['alternate_air_time_min'],
                alternate_distance_flown_nm=evaluation['alternate_distance_flown_nm'],
                reserve_fuel_kg=evaluation['reserve_fuel_kg'],
                reserve_flow_kg_hr=evaluation['reserve_flow_kg_hr'],
                contingency_fuel_kg=evaluation['contingency_fuel_kg'],
                contingency_method=evaluation['contingency_method'],
                contingency_minutes_used=evaluation['contingency_minutes_used'],
                contingency_fraction_used=evaluation['contingency_fraction_used'],
                primary_results=evaluation['primary_results'],
                alternate_results=evaluation['alternate_results'],
            )

        iterations_total = 0
        fuel_guess_kg = max(1000.0, 0.15 * (empty_weight_kg + payload_kg))

        if verbose:
            print("Solving required fuel via fixed-point iteration...")

        for fp_iter in range(1, max_iterations + 1):
            iterations_total += 1
            try:
                evaluation = _evaluate_guess(fuel_guess_kg)
            except (RuntimeError, ValueError) as exc:
                if verbose:
                    print(f"Fixed-point halted at iteration {fp_iter}: {exc}")
                break

            required_fuel_kg = evaluation['required_fuel_kg']
            delta_kg = required_fuel_kg - fuel_guess_kg

            if verbose:
                print(
                    f"  iter {fp_iter:02d}: guess={fuel_guess_kg:,.1f} kg, "
                    f"required={required_fuel_kg:,.1f} kg, delta={delta_kg:,.1f} kg"
                )

            if abs(delta_kg) <= tol_kg:
                if verbose:
                    print("Fixed-point converged.")
                return _build_solution(evaluation, method="fixed-point", iterations=iterations_total, converged=True)

            fuel_guess_kg = max(required_fuel_kg, 0.0)
        else:
            if verbose:
                print("Fixed-point did not converge within iteration limit; switching to bisection.")

        # Bisection fallback on g(F) = F - required_fuel(F)
        low_fuel_kg = 0.0
        residual_low, eval_low = _safe_residual(low_fuel_kg)
        if math.isfinite(residual_low) and residual_low >= 0.0 and eval_low is not None:
            if verbose:
                print("Bisection not required; zero-fuel lower bound already satisfies residual >= 0.")
            return _build_solution(eval_low, method="bisection", iterations=iterations_total + 1, converged=True)

        high_fuel_kg = max(fuel_guess_kg, 1000.0)
        residual_high, eval_high = _safe_residual(high_fuel_kg)

        expand_steps = 0
        while (not math.isfinite(residual_high) or residual_high < 0.0) and expand_steps < 25:
            high_fuel_kg = high_fuel_kg * 1.8 + 250.0
            residual_high, eval_high = _safe_residual(high_fuel_kg)
            expand_steps += 1

        if eval_high is None or not math.isfinite(residual_high) or residual_high < 0.0:
            raise RuntimeError(
                "Unable to bracket required fuel for bisection; route may be infeasible under current assumptions."
            )

        best_evaluation = eval_high
        converged = False

        if verbose:
            print("Running bisection fallback...")

        for bisect_iter in range(1, bisection_max_iterations + 1):
            iterations_total += 1
            mid_fuel_kg = 0.5 * (low_fuel_kg + high_fuel_kg)
            residual_mid, eval_mid = _safe_residual(mid_fuel_kg)

            if eval_mid is not None and math.isfinite(residual_mid) and residual_mid >= 0.0:
                high_fuel_kg = mid_fuel_kg
                best_evaluation = eval_mid
            else:
                low_fuel_kg = mid_fuel_kg

            if verbose:
                bracket_width = high_fuel_kg - low_fuel_kg
                print(
                    f"  bisect {bisect_iter:02d}: low={low_fuel_kg:,.1f} kg, "
                    f"high={high_fuel_kg:,.1f} kg, width={bracket_width:,.1f} kg"
                )

            if (high_fuel_kg - low_fuel_kg) <= tol_kg:
                converged = True
                break

        if verbose:
            status = "converged" if converged else "stopped at iteration limit"
            print(f"Bisection {status}.")

        return _build_solution(
            best_evaluation,
            method="bisection",
            iterations=iterations_total,
            converged=converged,
        )
    
    def __repr__(self) -> str:
        return (
            f"Aircraft(name='{self.name}', "
            f"MTOW={self.mtow_kg:.0f} kg, "
            f"engines={self.num_engines}x{self.thrust_per_engine_N/1000:.0f} kN)"
        )
