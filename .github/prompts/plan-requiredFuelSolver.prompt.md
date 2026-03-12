## Plan: Required Fuel Solver for Route Sizing

Add a new Aircraft-level solver that computes required departure fuel and takeoff weight from empty weight + payload + route using iterative simulation. The solver will target block fuel plus FAA Part 121 domestic-style additions (explicit alternate fuel + 45-minute final reserve + 5% contingency on trip fuel), while keeping core ODE physics unchanged and avoiding hard weight-capacity constraints.

**Steps**
1. Phase 1 - API + validation scaffolding.
2. Add a new public method on Aircraft: solve_required_fuel(...), with inputs for primary mission, empty_weight_kg, payload_kg, and required alternate mission; include policy knobs with defaults: reserve_minutes=45, contingency_fraction=0.05, and convergence controls (tol_kg, max_iterations). This is the user-facing entry point.
3. Add explicit guardrails in Aircraft.fly() for missing mission.initial_weight_kg and make Mission.__repr__ robust when takeoff weight is None so users can build missions for solver-driven sizing without formatting errors. Depends on step 1.
4. Phase 2 - fuel policy quantities from existing outputs.
5. Extend Results with helper properties for phase-level fuel accounting needed by policy math (at minimum: taxi fuel total, trip fuel excluding taxi, and cruise fuel-flow estimate for reserve computation). Keep this additive so existing APIs remain unchanged. Depends on step 1.
6. Define FAA domestic policy math inside solver flow:
7. Primary mission block burn comes from simulation output.
8. Alternate fuel is computed by running the user-supplied alternate mission from the primary landing mass (or equivalent handoff mass convention defined in docs).
9. Final reserve fuel uses 45 minutes at normal cruise fuel flow (derived from cruise segment fuel-flow, with fallback to average airborne flow if cruise samples are unavailable).
10. Contingency fuel defaults to 5% of primary trip fuel excluding taxi.
11. Phase 3 - iterative numerical solve.
12. Implement fixed-point iteration on departure fuel guess: set W0 = OEW + payload + fuel_guess, run primary + alternate simulations, recompute required fuel from policy components, and iterate until absolute fuel delta <= tol_kg. Depends on steps 2 and 5.
13. Add robust fallback to bisection if fixed-point fails to converge or oscillates; objective is g(F) = F - required_fuel(F). Use adaptive bracketing because no hard MTOW/fuel-cap constraints are enforced by design. Depends on step 12.
14. Add route-completion checks per iteration (e.g., required distance/phase completion) and fail fast with actionable errors when simulations terminate early.
15. Phase 4 - output contract + docs + tests. Parallel with step 14 once solver core stabilizes.
16. Return a structured result object/dict from solve_required_fuel including: converged flag, iterations, takeoff_weight_kg, required_fuel_kg, and component breakdown (block, trip, taxi, alternate, reserve, contingency), plus references to primary/alternate Results for diagnostics.
17. Update README with one end-to-end sizing example and clearly documented FAA policy assumptions and limitations.
18. Add targeted automated tests for convergence and arithmetic consistency; include at least one regression-style scenario proving that required_fuel ~= block+alternate+reserve+contingency at convergence.

**Relevant files**
- src/fuelburn/aircraft.py - add solve_required_fuel(), convergence loop, fallback logic, and fly() input validation.
- src/fuelburn/results.py - add helper properties for trip/taxi/cruise fuel accounting used by reserve policy.
- src/fuelburn/mission.py - make __repr__ safe when initial_weight_kg is None to support solver-first workflows.
- README.md - document new sizing workflow, FAA defaults, and explicit alternate-mission requirement.
- tests/test_required_fuel.py - add convergence and fuel-breakdown correctness tests.
- src/fuelburn/__init__.py - export any new public result type if one is introduced.

**Verification**
1. Run the new solver test file and confirm convergence plus component accounting assertions pass.
2. Run existing smoke scripts (test_simple/test_taxi equivalents) to ensure fly() behavior remains unchanged for users who still provide initial_weight_kg directly.
3. Manually validate one realistic scenario:
4. Build primary + alternate missions, specify OEW and payload, run solve_required_fuel().
5. Confirm returned required fuel equals policy sum within tolerance and that both primary and alternate simulations complete expected route phases.
6. Confirm sensitivity sanity checks: increasing payload or distance increases required fuel and takeoff weight.

**Decisions**
- Public API shape: new method on Aircraft.
- Fuel target: trip + taxi + reserve + contingency (total required departure fuel).
- Reserve policy default: FAA domestic, Part 121 IFR airline-style simplification.
- Alternate handling: explicit alternate mission input is required for the FAA default mode.
- Contingency default: 5% of trip fuel (excluding taxi).
- Constraints: no hard MTOW or max-fuel-capacity enforcement in this scope.

**Further Considerations**
1. Handoff mass convention for alternate mission start:
Option A: start alternate at primary gate-arrival mass (includes taxi-in burn in primary).
Option B: start alternate at landing mass before taxi-in (closer to dispatch logic).
Recommendation: Option B for regulatory realism; Option A for minimum implementation complexity.
2. Reserve fuel flow source:
Option A: cruise-segment mean fuel flow (preferred).
Option B: airborne mean flow fallback only.
Recommendation: A with automatic fallback to B when cruise phase is sparse/missing.
3. Solver output type:
Option A: dict for minimal change.
Option B: dedicated dataclass for stronger typing and discoverability.
Recommendation: Option B if adding one public type is acceptable; otherwise use dict and document keys strictly.
