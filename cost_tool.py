"""
File: cost_tool.py
Aircraft Direct Operating Cost Tool

Author: Gabriel Bortoletto Molz
Based on: AEA 1989a/b method (Association of European Airlines)
Date: February 5, 2026

Usage:
    # Use default AEA 1989a parameters
    params = MethodParameters()
    result = calculate_costs(aircraft_params, params)
    
    # Use fitted parameters (calibrated to regional jets)
    params = MethodParameters(maintenance=FITTED_MAINTENANCE_PARAMS)
    result = calculate_costs(aircraft_params, params)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class AircraftParameters:
    """Aircraft design and operational parameters.
    
    Attributes for utilization, pricing, and engine specifications.
    """
    # Utilization
    block_time_hours: float | None = None  # block time per flight, t_b [h]
    flight_time_hours: float | None = None  # mission flight time, t_f [h]
    flights_per_year: int | None = None # annual number of flights, n_flights [1/year]
    
    # Airplane pricing
    aircraft_delivery_price_usd: float | None = None # price of the aircraft, without spares, from manufacturer [USD]
    engine_price_usd: float | None = None # price per engine unit [USD]

    # Weights
    maximum_takeoff_weight_kg: float | None = None # aircraft MTOW [kg]
    operational_empty_weight_kg: float | None = None # aircraft OEW [kg]
    engine_weight_kg: float | None = None # engine weight per unit [kg]
    fuel_weight_kg: float | None = None  # m_f, fuel weight consumed during flight [kg]
    payload_weight_kg: float | None = None  # payload weight [kg]
    
    # Mission
    range_nm: float | None = None  # range [nautical miles]
    
    # Engine
    engine_count: int = 2  # n_E
    bypass_ratio: float | None = None  # BPR
    overall_pressure_ratio: float | None = None  # OAPR
    compressor_stages: int | None = None  # n_c (incl. fan)
    engine_shafts: int | None = None  # n_s in {1,2,3}
    takeoff_thrust_per_engine_N: float | None = None  # T_TO,E [N]

    # Crew
    cockpit_crew_count: int = 2 # Amount of dudes in the cockpit, typically 2 for transport airplanes (pilot + co-pilot)
    cabin_crew_count: int = 1 # Amount of cabin crew members, typically 1 per 50 passengers


@dataclass
class MaintenanceParameters:
    """Maintenance cost model parameters (AEA 1989a method).
    
    All coefficients and constants used in airframe and engine maintenance
    cost calculations. These parameters can be tuned to fit empirical data.
    """
    # Airframe maintenance - labor hours calculation
    airframe_labor_weight_coefficient: float = 9e-5  # Weight-based labor coefficient [h/kg]
    airframe_labor_base_hours: float = 6.7  # Base labor hours constant [h]
    airframe_labor_weight_numerator_kg: float = 350000  # Weight formula numerator [kg]
    airframe_labor_weight_denominator_offset_kg: float = 75000  # Weight formula offset [kg]
    airframe_labor_time_base_factor: float = 0.8  # Base time scaling factor
    airframe_labor_time_coefficient: float = 0.68  # Flight time multiplier
    
    # Airframe maintenance - material cost calculation
    airframe_material_base_coefficient: float = 4.2e-6  # Base material cost fraction
    airframe_material_time_coefficient: float = 2.2e-6  # Time-dependent material cost fraction
    
    # Engine k-factors - k1 (bypass ratio dependency)
    engine_k1_base: float = 1.27  # k1 base value
    engine_k1_bpr_coefficient: float = 0.2  # Bypass ratio coefficient for k1
    engine_k1_bpr_exponent: float = 0.2  # Bypass ratio exponent for k1
    
    # Engine k-factors - k2 (pressure ratio dependency)
    engine_k2_base: float = 0.4  # k2 base value
    engine_k2_opr_coefficient: float = 0.4  # Overall pressure ratio coefficient
    engine_k2_opr_exponent: float = 1.3  # Overall pressure ratio exponent
    engine_k2_opr_divisor: float = 20  # Overall pressure ratio divisor
    
    # Engine k-factors - k4 (shaft configuration dependency)
    engine_k4_single_shaft: float = 0.5  # k4 for single-shaft engines
    engine_k4_twin_shaft: float = 0.57  # k4 for twin-shaft engines
    engine_k4_triple_shaft: float = 0.64  # k4 for triple-shaft engines
    
    # Engine k-factors - k3 (compressor stages dependency)
    engine_k3_compressor_coefficient: float = 0.032  # Compressor stage coefficient
    
    # Engine maintenance - labor hours calculation
    engine_labor_base_coefficient: float = 0.17  # Base labor coefficient (AEA 1989a uses 0.17, some sources cite 0.21)
    engine_labor_thrust_coefficient: float = 1.02e-4  # Thrust scaling coefficient [1/N]
    engine_labor_thrust_exponent: float = 0.4  # Thrust exponent for labor hours
    engine_labor_flight_time_constant: float = 1.3  # Flight time constant for labor
    
    # Engine maintenance - material cost calculation
    engine_material_base_coefficient: float = 2.0  # Base material coefficient (AEA 1989a uses 2.0, some sources cite 2.56)
    engine_material_thrust_exponent: float = 0.8  # Thrust exponent for material cost
    engine_material_flight_time_constant: float = 1.3  # Flight time constant for material


# Fitted maintenance parameters (calibrated to 3 regional jets: ERJ-145 XR, CRJ-700, CRJ-200)
# Optimization achieved RMSE of $21.75 across all aircraft with 10-parameter fitting
# Use this instead of defaults for improved accuracy on regional jet maintenance costs
FITTED_MAINTENANCE_PARAMS = MaintenanceParameters(
    airframe_labor_base_hours=8.122902723888275,
    airframe_labor_time_base_factor=0.16870063740188052,
    airframe_labor_time_coefficient=0.9003414675473087,
    airframe_labor_weight_coefficient=3.636374535885983e-05,
    airframe_labor_weight_denominator_offset_kg=74999.77124528734,
    airframe_labor_weight_numerator_kg=349999.9647928149,
    airframe_material_base_coefficient=4.2e-06,
    airframe_material_time_coefficient=2.2e-06,
    engine_k1_base=0.5542373120826335,
    engine_k1_bpr_coefficient=0.2,
    engine_k1_bpr_exponent=0.2,
    engine_k2_base=0.4,
    engine_k2_opr_coefficient=0.4,
    engine_k2_opr_divisor=20,
    engine_k2_opr_exponent=0.554237323109609,
    engine_k3_compressor_coefficient=0.032,
    engine_k4_single_shaft=0.5,
    engine_k4_triple_shaft=0.64,
    engine_k4_twin_shaft=0.57,
    engine_labor_base_coefficient=0.0662712266888201,
    engine_labor_flight_time_constant=1.3,
    engine_labor_thrust_coefficient=0.000102,
    engine_labor_thrust_exponent=0.4,
    engine_material_base_coefficient=2.0,
    engine_material_flight_time_constant=1.3,
    engine_material_thrust_exponent=0.42892656740853,
)


@dataclass
class MethodParameters:
    """Cost calculation method parameters.
    
    Factors and rates used in AEA 1989a/b cost methods.
    """
    # Spares
    airframe_spares_factor: float = 0.1
    engine_spares_factor: float = 0.3
    
    # Depreciation
    depreciation_period_years: int = 16  # useful service life, n_dep [years]
    depreciation_relative_residual: float = 0.1
    
    # Interest
    interest_rate: float = 0.08
    repayment_period_years: int = 16  # repayment period, n_pay [years]
    balloon_fraction: float = 0.1
    
    # Insurance
    insurance_factor: float = 0.005  # k_ins, fraction of delivery price
    
    # Inflation
    inflation_rate: float = 0.027  # p_inf 2.7% average annual yields good results
    
    # Fuel
    fuel_price_usd: float = 0.785 # fuel price [USD/kg] 1 usd per kg = 3 usd per gal

    # Engine installation
    installed_engine_factor: float = 1.15 # 1.15 for jet engines in transport airplanes
    installed_engine_reverse_factor: float = 1.18 # 1.0 for non-reverse thrust engines, 1.18 for reverse thrust engines
    
    # Maintenance
    labor_rate_usd_per_hour: float = 65 # [USD/hour] 65 is the value for 1989 labor rate
    maintenance: MaintenanceParameters = None  # Maintenance model parameters
    
    def __post_init__(self):
        """Initialize maintenance parameters with defaults if not provided."""
        if self.maintenance is None:
            self.maintenance = MaintenanceParameters()

    # Crew
    cabin_crew_rate_usd_per_hour : float = 44.76 # [USD/hour] estimated cabin crew cost per hour
    cockpit_crew_rate_usd_per_hour : float = 226.22 / 2 # [USD/hour] estimated cockpit crew cost per hour
    
    # Fees and charges (AEA 1989a/b values)
    landing_fee_factor: float = 0.0078 / 15  # k_LD [USD/kg] - 0.0078 for AEA 1989a, 0.0059 for AEA 1989b
    navigation_fee_factor: float = 0 #0.00414 / 15  # k_NAV [USD/(nm·√kg)] - 0.00414 for AEA 1989a, 0.00166 for AEA 1989b
    ground_handling_factor: float = 0.10 / 15  # k_GND [USD/kg] - adjusted for realistic ground handling costs


@dataclass
class PricingBreakdown:
    """Aircraft pricing component breakdown."""
    engine_price_usd: float
    delivery_price_usd: float
    airframe_price_usd: float
    spares_price_usd: float
    purchase_price_usd: float


@dataclass
class CostBreakdown:
    """Direct operating cost breakdown by category."""
    depreciation: float
    interest: float
    insurance: float
    fuel: float
    maintenance: float
    crew: float
    fees_and_charges: float
    total: float


@dataclass
class DOCResult:
    """Complete Direct Operating Cost calculation results.
    
    Attributes:
        prices: Pricing breakdown (engine, delivery, airframe, spares, purchase)
        annual: Annual costs breakdown [USD/year]
        per_flight: Per-flight costs breakdown [USD/flight]
        per_hour: Per-hour costs breakdown [USD/flight hour]
    """
    prices: PricingBreakdown
    annual: CostBreakdown
    per_flight: CostBreakdown
    per_hour: CostBreakdown


def estimate_engine_price(
    takeoff_thrust_per_engine_N: float
) -> float:
    """Estimate engine price from takeoff thrust.
    
    Functionality:
        Estimates individual engine price based on takeoff thrust rating.
        Uses empirical power-law correlation.
        
    Args:
        takeoff_thrust_per_engine_N: Takeoff thrust per engine [N]
        
    Returns:
        float: Estimated engine price per unit [USD]
        
    Formula:
        P_E = 293 USD · (T_TO,E / N)^0.81
        where T_TO,E is thrust in Newtons
    """
    P_E = 293 * (takeoff_thrust_per_engine_N) ** 0.81
    return P_E

def estimate_purchase_price_from_oew(
    operational_empty_weight_kg: float
) -> float:
    """Estimate total aircraft purchase price from operational empty weight.
    
    Functionality:
        Estimates the aircraft acquisition cost (purchase price including spares)
        based on the operational empty weight. Uses empirical correlation formulas
        that differ based on aircraft size category.
        
    Args:
        operational_empty_weight_kg: Aircraft operational empty weight [kg]
        
    Returns:
        float: Estimated purchase price (delivery + spares) [USD]
        
    Formulas:
        For m_oe >= 10000 kg:
            C_AC = 10^6 × (1.18 × m_oe^0.48 - 116)
        For m_oe < 10000 kg:
            C_AC = -0.002695 × m_oe^2 + 1967 × m_oe - 2158000
    """
    if operational_empty_weight_kg >= 10000:
        # Large aircraft formula
        C_AC = 1e6 * (1.18 * operational_empty_weight_kg**0.48 - 116)
    else:
        # Small aircraft formula
        C_AC = (-0.002695 * operational_empty_weight_kg**2 + 
                1967 * operational_empty_weight_kg - 
                2158000)
    
    return C_AC

def calculate_delivery_price_from_oew(
    operational_empty_weight_kg: float,
    engine_price_usd: float,
    engine_count: int,
    airframe_spares_factor: float,
    engine_spares_factor: float
) -> float:
    """Calculate aircraft delivery price from OEW and component parameters.
    
    Functionality:
        Estimates the aircraft delivery price (excluding spares) by first
        estimating the total purchase price from OEW, then working backwards
        to separate out the spares investment. Uses algebraic solution to
        avoid circular dependency between delivery and spares prices.
        
    Args:
        operational_empty_weight_kg: Aircraft operational empty weight [kg]
        engine_price_usd: Price per engine unit [USD]
        engine_count: Number of engines on the aircraft
        airframe_spares_factor: Fraction of airframe price for spares (typically 0.1)
        engine_spares_factor: Fraction of total engine price for spares (typically 0.3)
        
    Returns:
        float: Aircraft delivery price (excluding spares) [USD]
        
    Mathematical derivation:
        Given: P_total = P_delivery + P_spares
               P_spares = P_AF × k_s_AF + n_E × P_E × k_s_E
               P_AF = P_delivery - n_E × P_E
        
        Solving for P_delivery:
        P_delivery = (P_total - n_E × P_E × (k_s_E - k_s_AF)) / (1 + k_s_AF)
    """
    # Estimate total purchase price from OEW
    purchase_price_usd = estimate_purchase_price_from_oew(operational_empty_weight_kg)
    
    # Calculate delivery price using algebraic solution
    # This avoids circular dependency between delivery price and spares price
    numerator = (purchase_price_usd - 
                 engine_count * engine_price_usd * (engine_spares_factor - airframe_spares_factor))
    denominator = 1 + airframe_spares_factor
    
    delivery_price_usd = numerator / denominator
    
    return delivery_price_usd

def calculate_airframe_price(
    delivery_price_usd: float,
    engine_price_usd: float,
    engine_count: int
) -> float:
    """Calculate airframe price by subtracting engine costs from delivery price.
    
    Functionality:
        Separates the airframe cost from total delivery price by removing
        the cost of all engines. Used for maintenance cost calculations.
        
    Args:
        delivery_price_usd: Aircraft delivery price from manufacturer [USD]
        engine_price_usd: Price per engine unit [USD]
        engine_count: Number of engines on the aircraft
        
    Returns:
        float: Airframe price (delivery price minus all engines) [USD]
    """
    return delivery_price_usd - engine_count * engine_price_usd

def calculate_spares_price(
    engine_price_usd: float,
    engine_count: int,
    airframe_price_usd: float,
    airframe_spares_factor: float,
    engine_spares_factor: float
) -> float:
    """Calculate total spares investment price.
    
    Functionality:
        Calculates initial spares investment as a fraction of airframe and
        engine costs. Spares are kept in inventory for maintenance.
        Based on AEA 1989a method.
        
    Args:
        engine_price_usd: Price per engine unit [USD]
        engine_count: Number of engines on the aircraft
        airframe_price_usd: Airframe price (delivery minus engines) [USD]
        airframe_spares_factor: Fraction of airframe price for spares (typically 0.1)
        engine_spares_factor: Fraction of total engine price for spares (typically 0.3)
        
    Returns:
        float: Total spares investment price [USD]
    """
    P_s_af = airframe_price_usd * airframe_spares_factor
    P_s_e = engine_price_usd * engine_count * engine_spares_factor
    P_s = P_s_af + P_s_e
    return P_s

def calculate_purchase_price(
    delivery_price_usd: float,
    spares_price_usd: float
) -> float:
    """Calculate total aircraft purchase price including spares.
    
    Functionality:
        Combines delivery price and initial spares investment to get
        total capital investment for the aircraft.
        
    Args:
        delivery_price_usd: Aircraft delivery price [USD]
        spares_price_usd: Initial spares investment [USD]
        
    Returns:
        float: Total purchase price (delivery + spares) [USD]
    """
    return delivery_price_usd + spares_price_usd


def calculate_depreciation(
    purchase_price_usd: float,
    residual_value_fraction: float,
    depreciation_period_years: int
) -> float:
    """Calculate annual depreciation cost.
    
    Functionality:
        Calculates straight-line depreciation over the useful service life.
        Assumes linear value loss from purchase price to residual value.
        Based on AEA 1989a method.
        
    Args:
        purchase_price_usd: Total aircraft purchase price (delivery + spares) [USD]
        residual_value_fraction: Residual value as fraction of purchase price (typically 0.1)
        depreciation_period_years: Useful service life for depreciation (typically 16 years)
        
    Returns:
        float: Annual depreciation cost [USD/year]
        
    Formula:
        C_dep = P_total * (1 - residual_fraction) / n_dep
    """
    C_dep = purchase_price_usd * (1 - residual_value_fraction) / depreciation_period_years
    return C_dep


def calculate_interest(
    purchase_price_usd: float,
    interest_rate: float,
    repayment_period_years: int,
    depreciation_period_years: int,
    balloon_fraction: float
) -> float:
    """Calculate annual interest cost on aircraft financing.
    
    Functionality:
        Calculates average annual interest payment considering balloon payment
        structure. Accounts for declining principal balance over time.
        Based on AEA 1989a method.
        
    Args:
        purchase_price_usd: Total aircraft purchase price [USD]
        interest_rate: Annual interest rate (e.g., 0.08 for 8%)
        repayment_period_years: Loan repayment period (typically 16 years)
        depreciation_period_years: Depreciation period (typically 16 years)
        balloon_fraction: Final balloon payment as fraction of initial price (typically 0.1)
        
    Returns:
        float: Annual interest cost [USD/year]
        
    Formula:
        p_av = [((q^n_pay - kn_k0)*(q-1))/(q^n_pay - 1)] * (n_pay/n_dep) - (1-kn_k0)/n_dep
        C_int = P_total * p_av
        where q = 1 + interest_rate
    """
    q = 1 + interest_rate
    
    p_av = (
        (((q**repayment_period_years - balloon_fraction) * (q - 1)) / 
         (q**repayment_period_years - 1)) * 
        (repayment_period_years / depreciation_period_years) - 
        (1 - balloon_fraction) / depreciation_period_years
    )
    
    C_int = purchase_price_usd * p_av
    return C_int


def calculate_insurance(
    delivery_price_usd: float,
    insurance_factor: float
) -> float:
    """Calculate annual insurance cost.
    
    Functionality:
        Calculates insurance premium as a fraction of aircraft delivery price.
        Based on AEA 1989a method.
        
    Args:
        delivery_price_usd: Aircraft delivery price [USD]
        insurance_factor: Insurance cost factor (typically 0.005, i.e., 0.5%)
        
    Returns:
        float: Annual insurance cost [USD/year]
        
    Formula:
        C_ins = P_delivery * k_ins
    """
    return delivery_price_usd * insurance_factor


def calculate_fuel(
    fuel_weight_kg: float,
    fuel_price_usd_per_kg: float,
    flights_per_year: int
) -> float:
    """Calculate annual fuel cost.
    
    Functionality:
        Calculates total fuel cost based on per-flight consumption,
        fuel price, and annual utilization.
        
    Args:
        fuel_weight_kg: Fuel weight consumed per flight [kg]
        fuel_price_usd_per_kg: Fuel price [USD/kg]
        flights_per_year: Number of flights per year
        
    Returns:
        float: Annual fuel cost [USD/year]
        
    Formula:
        C_fuel = m_f * P_f * n_flights
    """
    return fuel_weight_kg * fuel_price_usd_per_kg * flights_per_year

def calculate_installed_engine_weight(
        installed_engine_factor: float,
        installed_engine_reverse_factor: float,
        engine_weight_kg: float,
        engine_count: int,
) -> float:
    """Calculate total installed engine weight including installation equipment.
    
    Functionality:
        Calculates the total weight of all engines including installation
        equipment (mounts, cowlings, etc.) and reverse thrust system if present.
        Uses multiplication factors to account for additional hardware.
        
    Args:
        installed_engine_factor: Installation equipment factor (typically 1.15 for jet engines)
        installed_engine_reverse_factor: Reverse thrust factor (1.0 without, 1.18 with reverse thrust)
        engine_weight_kg: Bare engine weight per unit [kg]
        engine_count: Number of engines on the aircraft
        
    Returns:
        float: Total installed engine weight for all engines [kg]
        
    Formula:
        m_installed = k_install * k_reverse * m_engine * n_engines
    """
    installed_engine_weight_kg = installed_engine_factor * installed_engine_reverse_factor * engine_weight_kg * engine_count
    return installed_engine_weight_kg

def calculate_airframe_weight(
        operational_empty_weight_kg: float,
        installed_engine_weight_kg: float
) -> float:
    return operational_empty_weight_kg - installed_engine_weight_kg

def calculate_inflation_factor(
        inflation_rate: float,
        target_year: int,
        reference_year: int
) -> float:
    return (1 + inflation_rate) ** (target_year - reference_year)

def calculate_maintenance(
        # Airframe parameters
        flight_time_hours: float,
        airframe_weight_kg: float,
        airframe_price_usd: float,
        # Engine parameters
        bypass_ratio: float,
        overall_pressure_ratio: float,
        compressor_stages: int,
        engine_shafts: int,
        takeoff_thrust_per_engine_N: float,
        engine_count: int,
        # Common parameters
        flights_per_year: int,
        labor_rate_usd_per_hour: float,
        inflation_factor: float,
        # Model parameters
        params: MaintenanceParameters
) -> float:
    """Calculate total maintenance cost (airframe + engine).
    
    Functionality:
        Calculates combined airframe and engine maintenance costs based on
        aircraft characteristics and utilization. Based on AEA 1989a method.
        All model coefficients are now parameterized for easy fitting to empirical data.
        
    Args:
        flight_time_hours: Flight time per mission [h]
        airframe_weight_kg: Airframe weight [kg]
        airframe_price_usd: Airframe price [USD]
        bypass_ratio: Engine bypass ratio (BPR)
        overall_pressure_ratio: Overall air pressure ratio (OAPR)
        compressor_stages: Number of compressor stages including fan
        engine_shafts: Number of engine shafts (1, 2, or 3)
        takeoff_thrust_per_engine_N: Takeoff thrust per engine [N]
        engine_count: Number of engines
        flights_per_year: Number of flights per year
        labor_rate_usd_per_hour: Maintenance labor rate [USD/h]
        inflation_factor: Inflation adjustment factor k_INF
        params: Maintenance model parameters (all coefficients and constants)
        
    Returns:
        float: Total annual maintenance cost (airframe + engine) [USD/year]
        
    Formulas (AEA 1989a):
        Airframe:
            t_M,AF,f = (1/t_f) * (c1*m_AF + c2 - c3/(m_AF + c4)) * (c5 + c6*t_f)
            C_M,M,AF,f = (1/t_f) * (c7 + c8*t_f) * P_AF
        Engine:
            k1 = c9 - c10 * BPR^c11
            k2 = c12 * OAPR^c13 / c14 + c12
            k3 = c15 * n_c + k4  (where k4 depends on shaft count)
            t_M,E,f = n_E * c16 * k1 * k3 * (1 + c17*T_TO,E)^c18 * (1 + c19/t_f)
            C_M,M,E,f = n_E * c20 * k1 * (k2 + k3) * (1 + c17*T_TO,E)^c21 * (1 + c22/t_f) * k_INF
    """
    # AIRFRAME MAINTENANCE
    # Maintenance labor hours per flight
    t_M_AF_f = (1 / flight_time_hours * 
                (params.airframe_labor_weight_coefficient * airframe_weight_kg + 
                 params.airframe_labor_base_hours - 
                 params.airframe_labor_weight_numerator_kg / (airframe_weight_kg + params.airframe_labor_weight_denominator_offset_kg)) * 
                (params.airframe_labor_time_base_factor + params.airframe_labor_time_coefficient * flight_time_hours))
    
    # Maintenance material cost per flight
    C_M_M_AF_f = (1 / flight_time_hours * 
                  (params.airframe_material_base_coefficient + params.airframe_material_time_coefficient * flight_time_hours) * 
                  airframe_price_usd)
    
    # ENGINE MAINTENANCE
    # Calculate k-factors with parameterized coefficients
    k1 = params.engine_k1_base - params.engine_k1_bpr_coefficient * bypass_ratio ** params.engine_k1_bpr_exponent
    k2 = params.engine_k2_opr_coefficient * overall_pressure_ratio ** params.engine_k2_opr_exponent / params.engine_k2_opr_divisor + params.engine_k2_base
    
    # k4 depends on number of engine shafts
    match engine_shafts:
        case 1:
            k4 = params.engine_k4_single_shaft
        case 2:
            k4 = params.engine_k4_twin_shaft
        case 3:
            k4 = params.engine_k4_triple_shaft
        case _:
            raise ValueError(f"engine_shafts must be 1, 2, or 3, got {engine_shafts}")
    
    k3 = params.engine_k3_compressor_coefficient * compressor_stages + k4
    
    # Maintenance labor hours per flight
    t_M_E_f = (engine_count * params.engine_labor_base_coefficient * k1 * k3 * 
               (1 + params.engine_labor_thrust_coefficient * takeoff_thrust_per_engine_N) ** params.engine_labor_thrust_exponent * 
               (1 + params.engine_labor_flight_time_constant / flight_time_hours))
    
    # Maintenance material cost per flight
    C_M_M_E_f = (engine_count * params.engine_material_base_coefficient * k1 * (k2 + k3) * 
                 (1 + params.engine_labor_thrust_coefficient * takeoff_thrust_per_engine_N) ** params.engine_material_thrust_exponent * 
                 (1 + params.engine_material_flight_time_constant / flight_time_hours) * inflation_factor)
    
    # Total maintenance cost
    return ((t_M_AF_f + t_M_E_f) * labor_rate_usd_per_hour + C_M_M_AF_f + C_M_M_E_f) * flight_time_hours * flights_per_year


def calculate_crew(
        block_time_hours: float,
        flights_per_year: int,
        cockpit_crew_count: int,
        cabin_crew_count: int,
        cockpit_crew_rate_usd_per_hour: float,
        cabin_crew_rate_usd_per_hour: float
) -> float:
    """Calculate crew cost.
    
    Functionality:
        Calculates crew costs including salaries,
        training, and benefits. Based on AEA 1989b method.
        
    Returns:
        float: Annual crew cost [USD/year]
    """
    return (cockpit_crew_count * cockpit_crew_rate_usd_per_hour + 
            cabin_crew_count * cabin_crew_rate_usd_per_hour) * block_time_hours * flights_per_year


def calculate_fees_and_charges(
        maximum_takeoff_weight_kg: float,
        payload_weight_kg: float,
        range_nm: float,
        flights_per_year: int,
        landing_fee_factor: float,
        navigation_fee_factor: float,
        ground_handling_factor: float,
        inflation_factor: float
) -> float:
    """Calculate annual fees and charges.
    
    Functionality:
        Calculates total operating fees including landing fees, ATC/navigation
        charges, and ground handling charges. Based on AEA 1989a/b method.
        
    Args:
        maximum_takeoff_weight_kg: Maximum takeoff weight [kg]
        payload_weight_kg: Payload weight [kg]
        range_nm: Range [nautical miles]
        flights_per_year: Number of flights per year
        landing_fee_factor: Landing fee factor k_LD [USD/kg] 
                           (e.g., 0.0078 for AEA 1989a, 0.0059 for AEA 1989b)
        navigation_fee_factor: Navigation fee factor k_NAV [USD/(nm·√kg)]
                              (e.g., 0.00414 for AEA 1989a, 0.00166 for AEA 1989b)
        ground_handling_factor: Ground handling factor k_GND [USD/kg]
                               (e.g., 0.10 for AEA 1989a, 0.11 for AEA 1989b)
        inflation_factor: Inflation adjustment factor k_INF
        
    Returns:
        float: Total annual fees and charges [USD/year]
        
    Formulas (AEA 1989a/b):
        C_FEE,LD = k_LD · m_MTO · n_t,a · k_INF
        C_FEE,NAV = k_NAV · R · √(m_MTO) · n_t,a · k_INF
        C_FEE,GND = k_GND · m_PL · n_t,a · k_INF
        C_FEE = C_FEE,LD + C_FEE,NAV + C_FEE,GND
    """
    # Landing fees - based on maximum takeoff weight
    C_FEE_LD = (landing_fee_factor * 
                maximum_takeoff_weight_kg * 
                flights_per_year * 
                inflation_factor)
    
    # Navigation/ATC charges - based on distance and √(MTOW)
    C_FEE_NAV = (navigation_fee_factor * 
                 range_nm * 
                 (maximum_takeoff_weight_kg ** 0.5) * 
                 flights_per_year * 
                 inflation_factor)
    
    # Ground handling charges - based on payload weight
    C_FEE_GND = (ground_handling_factor * 
                 payload_weight_kg * 
                 flights_per_year * 
                 inflation_factor)
    
    # Total fees and charges
    C_FEE = C_FEE_LD + C_FEE_NAV + C_FEE_GND
    
    return C_FEE


def calculate_costs(
    aircraft: AircraftParameters,
    params: MethodParameters,
    target_year: int = 2026,
    verbose: bool = False
) -> DOCResult:
    """Calculate complete Direct Operating Cost breakdown.
    
    Functionality:
        Orchestrates all DOC calculations, handling both provided and estimated
        prices with proper inflation adjustments. Returns comprehensive cost 
        breakdown at annual, per-flight, and per-hour bases.
        
    Args:
        aircraft: Aircraft parameters (design, operational, pricing)
        params: Method parameters (factors, rates, spares fractions)
        target_year: Year for cost calculation (default: 2026)
        
    Returns:
        DOCResult: Comprehensive cost breakdown containing:
            - prices: PricingBreakdown with all pricing components
            - annual: CostBreakdown with annual costs [USD/year]
            - per_flight: CostBreakdown with per-flight costs [USD/flight]
            - per_hour: CostBreakdown with per-hour costs [USD/flight hour]
    
    Notes:
        - Engine price estimates are from 1999, inflation-adjusted to target_year
        - Purchase price estimates are from 2010, inflation-adjusted to target_year
        - If aircraft.engine_price_usd is provided, uses it directly
        - If aircraft.aircraft_delivery_price_usd is provided, uses it directly
    """
    
    # ========== PHASE 0: Inflation Factors ==========
    if verbose:
        print("\n" + "="*80)
        print("PHASE 0: INFLATION FACTORS & UTILIZATION")
        print("="*80)
    
    # General inflation factor (1989 -> target_year)
    inflation_factor = calculate_inflation_factor(
        params.inflation_rate,
        target_year,
        1989
    )
    
    if verbose:
        print(f"\nInflation calculation (1989 → {target_year}):")
        print(f"  Inflation rate: {params.inflation_rate:.4f} ({params.inflation_rate*100:.2f}%)")
        print(f"  Years elapsed: {target_year - 1989}")
        print(f"  Inflation factor: {inflation_factor:.4f}")

    labor_rate_target_year = params.labor_rate_usd_per_hour * inflation_factor
    
    if verbose:
        print(f"\nLabor rate adjustment:")
        print(f"  Base labor rate (1989): ${params.labor_rate_usd_per_hour:.2f}/hour")
        print(f"  Target year labor rate: ${labor_rate_target_year:.2f}/hour")
    
    # Estimate annual utilization if not provided
    if aircraft.flights_per_year is None:
        if verbose:
            print(f"\nEstimating annual utilization (flights_per_year not provided):")
            print(f"  Block time: {aircraft.block_time_hours:.2f} hours")
        
        Uannbl = 1000 * (3.4546 * aircraft.block_time_hours + 2.994 - 
                         np.sqrt(12.289 * aircraft.block_time_hours**2 - 
                                5.6626 * aircraft.block_time_hours + 8.964))
        flights_per_year = Uannbl / aircraft.block_time_hours
        
        if verbose:
            print(f"  Annual block hours (Uannbl): {Uannbl:.2f} hours/year")
            print(f"  Estimated flights per year: {flights_per_year:.2f}")
    else:
        flights_per_year = aircraft.flights_per_year
        if verbose:
            print(f"\nUsing provided annual utilization:")
            print(f"  Flights per year: {flights_per_year:.2f}")
    
    # ========== PHASE 1: Pricing Estimation ==========
    if verbose:
        print("\n" + "="*80)
        print("PHASE 1: PRICING ESTIMATION")
        print("="*80)
    
    # Estimate or use provided engine price
    if aircraft.engine_price_usd is not None:
        engine_price_usd = aircraft.engine_price_usd
        if verbose:
            print(f"\nUsing provided engine price:")
            print(f"  Engine price: ${engine_price_usd:,.2f} per engine")
    else:
        # Estimate from thrust (1999 baseline) and inflate to target year
        engine_price_1999 = estimate_engine_price(aircraft.takeoff_thrust_per_engine_N)
        engine_inflation = calculate_inflation_factor(params.inflation_rate, target_year, 1999)
        engine_price_usd = engine_price_1999 * engine_inflation
        
        if verbose:
            print(f"\nEstimating engine price from thrust:")
            print(f"  Takeoff thrust per engine: {aircraft.takeoff_thrust_per_engine_N:,.0f} N")
            print(f"  Estimated price (1999): ${engine_price_1999:,.2f}")
            print(f"  Inflation factor (1999 → {target_year}): {engine_inflation:.4f}")
            print(f"  Adjusted engine price: ${engine_price_usd:,.2f} per engine")
    
    # Estimate or use provided delivery price
    if aircraft.aircraft_delivery_price_usd is not None:
        delivery_price_usd = aircraft.aircraft_delivery_price_usd
        if verbose:
            print(f"\nUsing provided delivery price:")
            print(f"  Delivery price: ${delivery_price_usd:,.2f}")
    else:
        # Estimate from OEW (2010 baseline) and inflate to target year
        # delivery_price_1999 = calculate_delivery_price_from_oew(
        #     aircraft.operational_empty_weight_kg,
        #     engine_price_usd,
        #     aircraft.engine_count,
        #     params.airframe_spares_factor,
        #     params.engine_spares_factor
        # )

        delivery_price_1999 = 860 * aircraft.operational_empty_weight_kg
        delivery_inflation = calculate_inflation_factor(params.inflation_rate, target_year, 1999)
        delivery_price_usd = delivery_price_1999 * delivery_inflation
        
        if verbose:
            print(f"\nEstimating delivery price from OEW:")
            print(f"  Operational empty weight: {aircraft.operational_empty_weight_kg:,.0f} kg")
            print(f"  Estimated price (1999): ${delivery_price_1999:,.2f} (using $860/kg factor)")
            print(f"  Inflation factor (1999 → {target_year}): {delivery_inflation:.4f}")
            print(f"  Adjusted delivery price: ${delivery_price_usd:,.2f}")
    
    # Calculate airframe price (delivery minus engines)
    airframe_price_usd = calculate_airframe_price(
        delivery_price_usd,
        engine_price_usd,
        aircraft.engine_count
    )
    
    if verbose:
        print(f"\nAirframe price calculation:")
        print(f"  Delivery price: ${delivery_price_usd:,.2f}")
        print(f"  Engine price × {aircraft.engine_count} engines: ${engine_price_usd * aircraft.engine_count:,.2f}")
        print(f"  Airframe price: ${airframe_price_usd:,.2f}")
    
    # Calculate spares price
    spares_price_usd = calculate_spares_price(
        engine_price_usd,
        aircraft.engine_count,
        airframe_price_usd,
        params.airframe_spares_factor,
        params.engine_spares_factor
    )
    
    if verbose:
        print(f"\nSpares investment calculation:")
        print(f"  Airframe spares ({params.airframe_spares_factor*100:.1f}%): ${airframe_price_usd * params.airframe_spares_factor:,.2f}")
        print(f"  Engine spares ({params.engine_spares_factor*100:.1f}%): ${engine_price_usd * aircraft.engine_count * params.engine_spares_factor:,.2f}")
        print(f"  Total spares: ${spares_price_usd:,.2f}")
    
    # Calculate total purchase price
    purchase_price_usd = calculate_purchase_price(delivery_price_usd, spares_price_usd)
    
    if verbose:
        print(f"\nTotal purchase price:")
        print(f"  Delivery + Spares: ${purchase_price_usd:,.2f}")
    
    # ========== PHASE 2: Weight Calculations ==========
    if verbose:
        print("\n" + "="*80)
        print("PHASE 2: WEIGHT CALCULATIONS")
        print("="*80)
    
    installed_engine_weight_kg = calculate_installed_engine_weight(
        params.installed_engine_factor,
        params.installed_engine_reverse_factor,
        aircraft.engine_weight_kg,
        aircraft.engine_count
    )
    
    if verbose:
        print(f"\nInstalled engine weight:")
        print(f"  Bare engine weight: {aircraft.engine_weight_kg:,.0f} kg per engine")
        print(f"  Installation factor: {params.installed_engine_factor:.2f}")
        print(f"  Reverse thrust factor: {params.installed_engine_reverse_factor:.2f}")
        print(f"  Number of engines: {aircraft.engine_count}")
        print(f"  Total installed engine weight: {installed_engine_weight_kg:,.0f} kg")
    
    airframe_weight_kg = calculate_airframe_weight(
        aircraft.operational_empty_weight_kg,
        installed_engine_weight_kg
    )
    
    if verbose:
        print(f"\nAirframe weight:")
        print(f"  Operational empty weight: {aircraft.operational_empty_weight_kg:,.0f} kg")
        print(f"  Installed engine weight: {installed_engine_weight_kg:,.0f} kg")
        print(f"  Airframe weight: {airframe_weight_kg:,.0f} kg")
    
    # ========== PHASE 3: Fixed Costs (Annual) ==========
    if verbose:
        print("\n" + "="*80)
        print("PHASE 3: FIXED COSTS (ANNUAL)")
        print("="*80)
    
    depreciation = calculate_depreciation(
        purchase_price_usd,
        params.depreciation_relative_residual,
        params.depreciation_period_years
    )
    
    if verbose:
        print(f"\nDepreciation:")
        print(f"  Purchase price: ${purchase_price_usd:,.2f}")
        print(f"  Residual value ({params.depreciation_relative_residual*100:.0f}%): ${purchase_price_usd * params.depreciation_relative_residual:,.2f}")
        print(f"  Depreciation period: {params.depreciation_period_years} years")
        print(f"  Annual depreciation: ${depreciation:,.2f}/year")
    
    interest = calculate_interest(
        purchase_price_usd,
        params.interest_rate,
        params.repayment_period_years,
        params.depreciation_period_years,
        params.balloon_fraction
    )
    
    if verbose:
        print(f"\nInterest:")
        print(f"  Purchase price: ${purchase_price_usd:,.2f}")
        print(f"  Interest rate: {params.interest_rate*100:.2f}%")
        print(f"  Repayment period: {params.repayment_period_years} years")
        print(f"  Balloon payment ({params.balloon_fraction*100:.0f}%): ${purchase_price_usd * params.balloon_fraction:,.2f}")
        print(f"  Annual interest: ${interest:,.2f}/year")
    
    insurance = calculate_insurance(
        delivery_price_usd,
        params.insurance_factor
    )
    
    if verbose:
        print(f"\nInsurance:")
        print(f"  Delivery price: ${delivery_price_usd:,.2f}")
        print(f"  Insurance factor: {params.insurance_factor*100:.2f}%")
        print(f"  Annual insurance: ${insurance:,.2f}/year")
    
    # ========== PHASE 4: Variable Costs (Annual) ==========
    if verbose:
        print("\n" + "="*80)
        print("PHASE 4: VARIABLE COSTS (ANNUAL)")
        print("="*80)
    
    fuel = calculate_fuel(
        aircraft.fuel_weight_kg,
        params.fuel_price_usd,
        flights_per_year
    )
    
    if verbose:
        print(f"\nFuel cost:")
        print(f"  Fuel per flight: {aircraft.fuel_weight_kg:,.0f} kg")
        print(f"  Fuel price: ${params.fuel_price_usd:.4f}/kg")
        print(f"  Flights per year: {flights_per_year:.2f}")
        print(f"  Annual fuel: ${fuel:,.2f}/year")
        print(f"  Per flight: ${fuel/flights_per_year:,.2f}/flight")
    
    maintenance = calculate_maintenance(
        aircraft.flight_time_hours,
        airframe_weight_kg,
        airframe_price_usd,
        aircraft.bypass_ratio,
        aircraft.overall_pressure_ratio,
        aircraft.compressor_stages,
        aircraft.engine_shafts,
        aircraft.takeoff_thrust_per_engine_N,
        aircraft.engine_count,
        flights_per_year,
        labor_rate_target_year,
        inflation_factor,
        params.maintenance
    )
    
    if verbose:
        print(f"\nMaintenance cost:")
        print(f"  Flight time: {aircraft.flight_time_hours:.2f} hours")
        print(f"  Airframe weight: {airframe_weight_kg:,.0f} kg")
        print(f"  Airframe price: ${airframe_price_usd:,.2f}")
        print(f"  Engine specs: BPR={aircraft.bypass_ratio:.2f}, OPR={aircraft.overall_pressure_ratio:.1f}")
        print(f"  Compressor stages: {aircraft.compressor_stages}, Shafts: {aircraft.engine_shafts}")
        print(f"  Labor rate: ${labor_rate_target_year:.2f}/hour")
        print(f"  Annual maintenance: ${maintenance:,.2f}/year")
        print(f"  Per flight: ${maintenance/flights_per_year:,.2f}/flight")
    
    crew = calculate_crew(
        aircraft.block_time_hours,
        flights_per_year,
        aircraft.cockpit_crew_count,
        aircraft.cabin_crew_count,
        params.cockpit_crew_rate_usd_per_hour,
        params.cabin_crew_rate_usd_per_hour
    )
    
    if verbose:
        print(f"\nCrew cost:")
        print(f"  Block time: {aircraft.block_time_hours:.2f} hours")
        print(f"  Cockpit crew: {aircraft.cockpit_crew_count} @ ${params.cockpit_crew_rate_usd_per_hour:.2f}/hour")
        print(f"  Cabin crew: {aircraft.cabin_crew_count} @ ${params.cabin_crew_rate_usd_per_hour:.2f}/hour")
        print(f"  Flights per year: {flights_per_year:.2f}")
        print(f"  Annual crew cost: ${crew:,.2f}/year")
        print(f"  Per flight: ${crew/flights_per_year:,.2f}/flight")
    
    fees_and_charges = calculate_fees_and_charges(
        aircraft.maximum_takeoff_weight_kg,
        aircraft.payload_weight_kg,
        aircraft.range_nm,
        flights_per_year,
        params.landing_fee_factor,
        params.navigation_fee_factor,
        params.ground_handling_factor,
        inflation_factor
    )
    
    if verbose:
        print(f"\nFees and charges:")
        print(f"  MTOW: {aircraft.maximum_takeoff_weight_kg:,.0f} kg")
        print(f"  Payload: {aircraft.payload_weight_kg:,.0f} kg")
        print(f"  Range: {aircraft.range_nm:.0f} nm")
        print(f"  Landing fee factor: ${params.landing_fee_factor:.6f}/kg")
        print(f"  Navigation fee factor: ${params.navigation_fee_factor:.6f}/(nm·√kg)")
        print(f"  Ground handling factor: ${params.ground_handling_factor:.6f}/kg")
        print(f"  Annual fees & charges: ${fees_and_charges:,.2f}/year")
        print(f"  Per flight: ${fees_and_charges/flights_per_year:,.2f}/flight")
    
    # ========== PHASE 5: Aggregate and Normalize ==========
    # Annual totals
    total_annual = depreciation + interest + insurance + fuel + maintenance + crew + fees_and_charges
    
    if verbose:
        print("\n" + "="*80)
        print("PHASE 5: COST AGGREGATION & NORMALIZATION")
        print("="*80)
        print(f"\nAnnual cost breakdown:")
        print(f"  Depreciation:     ${depreciation:>12,.2f}  ({depreciation/total_annual*100:>5.1f}%)")
        print(f"  Interest:         ${interest:>12,.2f}  ({interest/total_annual*100:>5.1f}%)")
        print(f"  Insurance:        ${insurance:>12,.2f}  ({insurance/total_annual*100:>5.1f}%)")
        print(f"  Fuel:             ${fuel:>12,.2f}  ({fuel/total_annual*100:>5.1f}%)")
        print(f"  Maintenance:      ${maintenance:>12,.2f}  ({maintenance/total_annual*100:>5.1f}%)")
        print(f"  Crew:             ${crew:>12,.2f}  ({crew/total_annual*100:>5.1f}%)")
        print(f"  Fees & Charges:   ${fees_and_charges:>12,.2f}  ({fees_and_charges/total_annual*100:>5.1f}%)")
        print(f"  {'-'*50}")
        print(f"  TOTAL:            ${total_annual:>12,.2f}  (100.0%)")
    
    annual_costs = CostBreakdown(
        depreciation=depreciation,
        interest=interest,
        insurance=insurance,
        fuel=fuel,
        maintenance=maintenance,
        crew=crew,
        fees_and_charges=fees_and_charges,
        total=total_annual
    )
    
    # Per-flight costs
    per_flight_costs = CostBreakdown(
        depreciation=depreciation / flights_per_year,
        interest=interest / flights_per_year,
        insurance=insurance / flights_per_year,
        fuel=fuel / flights_per_year,
        maintenance=maintenance / flights_per_year,
        crew=crew / flights_per_year,
        fees_and_charges=fees_and_charges / flights_per_year,
        total=total_annual / flights_per_year
    )
    
    if verbose:
        print(f"\nPer-flight cost breakdown (÷ {flights_per_year:.1f} flights/year):")
        print(f"  Depreciation:     ${per_flight_costs.depreciation:>10,.2f}")
        print(f"  Interest:         ${per_flight_costs.interest:>10,.2f}")
        print(f"  Insurance:        ${per_flight_costs.insurance:>10,.2f}")
        print(f"  Fuel:             ${per_flight_costs.fuel:>10,.2f}")
        print(f"  Maintenance:      ${per_flight_costs.maintenance:>10,.2f}")
        print(f"  Crew:             ${per_flight_costs.crew:>10,.2f}")
        print(f"  Fees & Charges:   ${per_flight_costs.fees_and_charges:>10,.2f}")
        print(f"  {'-'*40}")
        print(f"  TOTAL:            ${per_flight_costs.total:>10,.2f}")
    
    # Per-hour costs
    total_flight_hours = flights_per_year * aircraft.flight_time_hours
    per_hour_costs = CostBreakdown(
        depreciation=depreciation / total_flight_hours,
        interest=interest / total_flight_hours,
        insurance=insurance / total_flight_hours,
        fuel=fuel / total_flight_hours,
        maintenance=maintenance / total_flight_hours,
        crew=crew / total_flight_hours,
        fees_and_charges=fees_and_charges / total_flight_hours,
        total=total_annual / total_flight_hours
    )
    
    if verbose:
        print(f"\nPer-hour cost breakdown (÷ {total_flight_hours:.1f} flight hours/year):")
        print(f"  Depreciation:     ${per_hour_costs.depreciation:>10,.2f}")
        print(f"  Interest:         ${per_hour_costs.interest:>10,.2f}")
        print(f"  Insurance:        ${per_hour_costs.insurance:>10,.2f}")
        print(f"  Fuel:             ${per_hour_costs.fuel:>10,.2f}")
        print(f"  Maintenance:      ${per_hour_costs.maintenance:>10,.2f}")
        print(f"  Crew:             ${per_hour_costs.crew:>10,.2f}")
        print(f"  Fees & Charges:   ${per_hour_costs.fees_and_charges:>10,.2f}")
        print(f"  {'-'*40}")
        print(f"  TOTAL:            ${per_hour_costs.total:>10,.2f}")
    
    # Pricing breakdown
    pricing = PricingBreakdown(
        engine_price_usd=engine_price_usd,
        delivery_price_usd=delivery_price_usd,
        airframe_price_usd=airframe_price_usd,
        spares_price_usd=spares_price_usd,
        purchase_price_usd=purchase_price_usd
    )
    
    # ========== PHASE 6: Return Comprehensive Results ==========
    if verbose:
        print("\n" + "="*80)
        print("CALCULATION COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"  Total annual DOC: ${total_annual:,.2f}/year")
        print(f"  Cost per flight: ${per_flight_costs.total:,.2f}/flight")
        print(f"  Cost per hour: ${per_hour_costs.total:,.2f}/flight hour")
        print("\n" + "="*80 + "\n")
    
    return DOCResult(
        prices=pricing,
        annual=annual_costs,
        per_flight=per_flight_costs,
        per_hour=per_hour_costs
    )

