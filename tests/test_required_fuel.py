"""
Regression checks for required fuel sizing solver.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fuelburn as fb


def _build_missions():
    primary = fb.Mission(
        distance_nm=450.0,
        cruise_altitude_ft=33000.0,
        cruise_mach=0.76,
        taxi_out_time_s=600.0,
        taxi_in_time_s=300.0,
    )

    # Alternate is modeled as airborne diversion only; taxi is set to zero.
    alternate = fb.Mission(
        distance_nm=90.0,
        cruise_altitude_ft=20000.0,
        cruise_mach=0.70,
        taxi_out_time_s=0.0,
        taxi_in_time_s=0.0,
    )

    return primary, alternate


def test_required_fuel_converges_and_breakdown_balances():
    aircraft = fb.Aircraft.from_preset('ERJ-145XR')
    primary, alternate = _build_missions()

    solution = aircraft.solve_required_fuel(
        mission=primary,
        empty_weight_kg=12500.0,
        payload_kg=3000.0,
        alternate_mission=alternate,
        reserve_minutes=45.0,
        contingency_fraction=0.05,
        tol_kg=20.0,
        max_iterations=10,
        bisection_max_iterations=25,
        max_time_s=20000.0,
        verbose=False,
    )

    assert solution.converged, "Required fuel solver did not converge"
    assert solution.required_fuel_kg > 0.0

    expected_required = (
        solution.primary_block_fuel_kg
        + solution.alternate_fuel_kg
        + solution.reserve_fuel_kg
        + solution.contingency_fuel_kg
    )
    assert math.isclose(solution.required_fuel_kg, expected_required, rel_tol=0.0, abs_tol=1e-9)

    assert solution.primary_results.distance_flown_nm >= 0.97 * primary.distance_nm
    assert solution.alternate_results.distance_flown_nm >= 0.97 * alternate.distance_nm


def test_required_fuel_increases_with_payload():
    aircraft = fb.Aircraft.from_preset('ERJ-145XR')
    primary, alternate = _build_missions()

    base = aircraft.solve_required_fuel(
        mission=primary,
        empty_weight_kg=12500.0,
        payload_kg=2500.0,
        alternate_mission=alternate,
        tol_kg=20.0,
        max_iterations=10,
        bisection_max_iterations=25,
        max_time_s=20000.0,
        verbose=False,
    )

    heavier = aircraft.solve_required_fuel(
        mission=primary,
        empty_weight_kg=12500.0,
        payload_kg=3500.0,
        alternate_mission=alternate,
        tol_kg=20.0,
        max_iterations=10,
        bisection_max_iterations=25,
        max_time_s=20000.0,
        verbose=False,
    )

    assert heavier.converged
    assert base.converged
    assert heavier.required_fuel_kg > base.required_fuel_kg
    assert heavier.takeoff_weight_kg > base.takeoff_weight_kg


def test_required_fuel_supports_minute_contingency():
    aircraft = fb.Aircraft.from_preset('ERJ-145XR')
    primary, alternate = _build_missions()

    solution = aircraft.solve_required_fuel(
        mission=primary,
        empty_weight_kg=12500.0,
        payload_kg=3000.0,
        alternate_mission=alternate,
        reserve_minutes=30.0,
        contingency_minutes=15.0,
        tol_kg=20.0,
        max_iterations=10,
        bisection_max_iterations=25,
        max_time_s=20000.0,
        verbose=False,
    )

    assert solution.converged
    assert solution.contingency_method == 'minutes'

    expected_contingency = solution.reserve_flow_kg_hr * (15.0 / 60.0)
    assert math.isclose(solution.contingency_fuel_kg, expected_contingency, rel_tol=0.0, abs_tol=1e-9)


def test_contingency_mode_is_mutually_exclusive():
    aircraft = fb.Aircraft.from_preset('ERJ-145XR')
    primary, alternate = _build_missions()

    try:
        aircraft.solve_required_fuel(
            mission=primary,
            empty_weight_kg=12500.0,
            payload_kg=3000.0,
            alternate_mission=alternate,
            contingency_fraction=0.05,
            contingency_minutes=15.0,
            tol_kg=20.0,
            max_iterations=10,
            bisection_max_iterations=25,
            max_time_s=20000.0,
            verbose=False,
        )
    except ValueError as exc:
        assert 'Only one of contingency_fraction or contingency_minutes' in str(exc)
    else:
        raise AssertionError('Expected ValueError when both contingency modes are specified')


def run_all_tests():
    test_required_fuel_converges_and_breakdown_balances()
    test_required_fuel_increases_with_payload()
    test_required_fuel_supports_minute_contingency()
    test_contingency_mode_is_mutually_exclusive()
    print("test_required_fuel.py: all checks passed")


if __name__ == '__main__':
    run_all_tests()
