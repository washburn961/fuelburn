# -*- coding: utf-8 -*-
"""
Conceptual Aircraft Design Tool
(for PRJ-22 and AP-701 courses)

Maj. Eng. Ney Rafael Secco (ney@ita.br)
Aircraft Design Department
Aeronautics Institute of Technology

05-2025
"""

"""

This version of the designTool is being modified to 
support the Learis Team on the development of our aircraft.

01-2026                         
                                                     
"""

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import cost_tool as ct
import copy

# CONSTANTS
ft2m = 0.3048     # [m/ft] - Conversion factor of feet to meters
kt2ms = 0.514444  # [m/s / knots] - Conversion factor of knots to meters per second
lb2N = 4.44822    # [N/lb] - Conversion factor of pounds to Newtons
nm2m = 1852.0     # [m/nm] - Conversion factor of nautical miles to meter
gravity = 9.81    # [m/s^2] - Gravity acceleration
gamma_air = 1.4   # [-] - Adibatic coefficient
R_air = 287       # [J/(kg/K)] - Specific gas constant for dry air 

# ========================================

# MAIN FUNCTION

def design(
    airplane=None,
    print_log=False,  # Plot results on the terminal screen
    plot=False,       # Generate 3D plot of the aircraft
):
    """
        
    
        Parameters
        ----------
        airplane : TYPE, optional
            DESCRIPTION. The default is None.
        print_log : TYPE, optional
            Plot results on the terminal screen. The default is False.
        plot : Generate 3D plot of the aircraft
    
        Returns
        -------
        None.
        
        Description
        ----------
        This is the main function that should be used for aircraft design.
        
        Version Control
        ----------
        (Version/Date/Author/Modification)
        
        > Version 01 - 24/02/2026 - Rian - Creation
    
    """

    # Load standard airplane if none is provided
    if airplane is None:
        airplane = standard_airplane()

    # Use an average wing loading for transports
    # to estime W0_guess and T0_guess if none are provided
    if "W0_guess" in airplane.keys():
        W0_guess = airplane["W0_guess"]
    else:
        W0_guess = 5e3 * airplane["S_w"]

    if "T0_guess" in airplane.keys():
        T0_guess = airplane["T0_guess"]
    else:
        T0_guess = 0.3 * W0_guess

    # Generate geometry
    geometry(airplane)

    # Converge MTOW and Takeoff Thrust
    thrust_matching(W0_guess, T0_guess, airplane)
    

    if print_log:
        print("W_empty [kgf]: %d" % (airplane["W_empty"] / gravity))
        print("W_fuel [kgf]: %d" % (airplane["W_fuel"] / gravity))
        print("W0 [kgf]: %d" % (airplane["W0"] / gravity))
        print("T0 [kgf]: %d" % (airplane["T0"] / gravity))
        print("T0/W0: %.3f" %
              (2*airplane["engine"]['T_eng_spec'] / airplane["W0"]))
        print("W0/S [kgf/m2]: %d" %
              (airplane["W0"] / gravity / airplane["S_w"]))
        print("deltaS_wlan [m2]: %.1f" % (airplane["deltaS_wlan"]))
 
    # Plot again now that we have CG and NP
    if plot:
        plot3d(airplane)

    return airplane

# ========================================
# DISCIPLINE MODULES

def geometry(airplane):
    """    
    Parameters
    ----------
    airplane : airplane to be analyzed.

    Returns
    -------
    None.

    Version Control
    ----------
    (Version/Date/Author/Modification)
    
    > Version 01 - 19/02/2026 - Rian - Headder adition, Outputs removal

    """
    # Unpack dictionary
    S_w = airplane["S_w"] # [m^2] - Wing area
    AR_w = airplane["AR_w"] # [-] - Wing aspect ratio
    taper_w = airplane["taper_w"] # [-] Wing taper ratio
    sweep_w = airplane["sweep_w"] # [rad] - Wing sweep (at c/4)
    dihedral_w = airplane["dihedral_w"] # [rad] - Wing dihedral
    xr_w = airplane["xr_w"] # [m] - Longidudinal position (x-axis) of the wing root leading edge w.r.t the airplane nose
    zr_w = airplane["zr_w"] # [m] - Vertical position (z-axis) of the wing root in half-thickness (t/c_max = 0.5) w.r.t the the airplane nose
    Cht = airplane["Cht"] # [-] - Horizontal tail volume coefficient
    AR_h = airplane["AR_h"] # [-] - Horizontal tail aspect ratio 
    taper_h = airplane["taper_h"] # [-] - Horizontal tail aspect ratio
    sweep_h = airplane["sweep_h"] # [rad] - Horizontal tail sweep
    dihedral_h = airplane["dihedral_h"] # [rad] - Horizontal tail dihedral 
    Lc_h = airplane["Lc_h"] # [-] - Non-dimensional lever of the horizontal tail (lever/wing MAC)
    zr_h = airplane["zr_h"] # [m] - Vertical position (z-axis) of the horizontal tail root in half-thickness (t/c_max = 0.5) w.r.t the the fuselage centerline
    Cvt = airplane["Cvt"] # [-] - Vertical tail volume coefficient
    AR_v = airplane["AR_v"] # [-] - Vertical tail aspect ratio 
    taper_v = airplane["taper_v"] # [-] Vertical tail taper ratio
    sweep_v = airplane["sweep_v"] # [rad] - Vertical tail sweep
    Lb_v = airplane["Lb_v"] # [-] - Non-dimensional lever of the vertical tail (lever/wing MAC)
    zr_v = airplane["zr_v"] # [m] - Vertical position (z-axis) of the vertical tail root w.r.t the the fuselage centerline
        
    # -------------------------------------------------------   
    # Wing Sizing
    b_w = np.sqrt(AR_w * S_w) # [m] - Wing span
    cr_w = 2 * S_w / (b_w * (1 + taper_w)) # [m] - Wing root chord
    ct_w = taper_w * cr_w # [m] - Wing tip chord
    yt_w = b_w * 0.5 # [m] - Wing semi-span
    xt_w = (xr_w + yt_w * np.tan(sweep_w) + (cr_w - ct_w) * 0.25) # [m] - Wing tip leading x-position w.r.t the fuselage nose
    zt_w = zr_w + yt_w * np.tan(dihedral_w) # [m] - Wing tip z-position w.r.t the fuselage nose
    cm_w = ((2 / 3) * cr_w * ((1 + taper_w + taper_w**2) / (1 + taper_w))) # [m] - Wing mean aerodynamic chord (MAC)
    ym_w = (b_w / 6) * ((1 + 2 * taper_w) / (1 + taper_w)) #[m] - Transversal position (y-axis) (spanwise station) of wing MAC w.r.t the wing root 
    xm_w = (xr_w + ym_w * np.tan(sweep_w) + (cr_w - cm_w) * 0.25) # [m] - Wing MAC leading edge x-position w.r.t the fuselage nose
    zm_w = zr_w + ym_w * np.tan(dihedral_w) # [m] - Wing MAC z-position w.r.t the fuselage nose

    # -------------------------------------------------------
    # Vertical Tail (VT) sizing
    S_v = S_w * Cvt / Lb_v # [m^2] - Vertical tail area
    b_v = np.sqrt(AR_v * S_v) # [m] - Vertical tail span (geometric height)
    cr_v = 2 * S_v / (b_v * (1 + taper_v)) # [m] - Vertical tail root chord
    ct_v = taper_v * cr_v # [m] - Vertical tail tip chord
    cm_v = ((2 / 3) * cr_v * ((1 + taper_v + taper_v**2) / (1 + taper_v))) # [m] - Vertical tail mean aerodynamic chord (VT MAC)
    xm_v = xm_w + Lb_v * b_w + (cm_w - cm_v) * 0.25 # [m] - VT MAC leading-edge x-position w.r.t the fuselage nose
    zm_v = zr_v + (b_v / 3) * ((1 + 2 * taper_v) / (1 + taper_v)) # [m] - VT MAC z-position w.r.t the fuselage centerline
    xr_v = (xm_v - (zm_v - zr_v) * np.tan(sweep_v) + (cm_v - cr_v) * 0.25) # [m] - VT root leading edge x-position w.r.t the fuselage nose
    zt_v = zr_v + b_v # [m] - VT tip z-position w.r.t the fuselage centerline
    xt_v = (xr_v + (zt_v - zr_v) * np.tan(sweep_v) + (cr_v - ct_v) * 0.25) # [m] - VT tip leading-edge x-position w.r.t the fuselage nose
    
    # -------------------------------------------------------
    # Horizontal Tail (HT) Sizing (Dimensionamento do Estabilizador Horizontal)
    if airplane.get("has_HT", True):

        S_h = S_w * Cht / Lc_h  # Área do estabilizador horizontal [m²]
        b_h = np.sqrt(AR_h * S_h)  # Envergadura do EH [m]

        cr_h = 2 * S_h / (b_h * (1 + taper_h))  # Corda na raiz do EH [m]
        ct_h = taper_h * cr_h  # Corda na ponta do EH [m]
        cm_h = ((2 / 3) * cr_h * ((1 + taper_h + taper_h**2) / (1 + taper_h)))# Corda aerodinâmica média do EH [m]
        
        ym_h = (b_h / 6) * ((1 + 2 * taper_h) / (1 + taper_h))  # Posição Y da MAC do EH [m]
        xm_h = xm_w + Lc_h * cm_w + (cm_w - cm_h) / 4 # Posição X da MAC do EH [m]
        xr_h = (xm_h - ym_h * np.tan(sweep_h) + (cm_h - cr_h)* 0.25)  # Coordenada X da raiz do EH [m]
        zm_h = zr_h + ym_h * np.tan(dihedral_h)  # Posição Z da MAC do EH [m]

        yt_h = b_h * 0.5  # Semi-envergadura do EH [m]
        xt_h = (xr_h + yt_h * np.tan(sweep_h) + (cr_h - ct_h)* 0.25)  # Coordenada X da ponta do EH [m]
        zt_h = zr_h + yt_h*np.tan(dihedral_h)# Coordenada Z da ponta do EH [m]

        airplane["S_h"] = S_h
        airplane["b_h"] = b_h
        airplane["cr_h"] = cr_h
        airplane["xt_h"] = xt_h
        airplane["xr_h"] = xr_h
        airplane["yt_h"] = yt_h
        airplane["zt_h"] = zt_h
        airplane["zr_h"] = zr_h
        airplane["ct_h"] = ct_h
        airplane["xm_h"] = xm_h
        airplane["ym_h"] = ym_h
        airplane["zm_h"] = zm_h
        airplane["cm_h"] = cm_h
    else:
        airplane["S_h"] = 0.0	
	# -------------------------------------------------------

	# Canard Sizing (Dimensionamento do Canard) [Julia]

    if airplane.get("has_canard", True):

        Ccan = airplane["Ccan"]
        AR_c = airplane["AR_c"]
        taper_c = airplane["taper_c"]
        sweep_c = airplane["sweep_c"]
        dihedral_c = airplane["dihedral_c"]
        Lc_c = airplane["Lc_c"]
        zr_c = airplane["zr_c"]

        S_c = S_w * Ccan / Lc_c

        # Planform trapezoidal
        b_c = np.sqrt(AR_c * S_c)
        cr_c = 2 * S_c / (b_c * (1 + taper_c))
        ct_c = taper_c * cr_c

        yt_c = b_c * 0.5

        # MAC e posição spanwise da MAC
        cm_c = (2/3) * cr_c * ((1 + taper_c + taper_c**2) / (1 + taper_c))
        ym_c = (b_c / 6) * ((1 + 2*taper_c) / (1 + taper_c))

        # Posição X da MAC do canard: À FRENTE da asa
        # (análoga à EH, com sinal invertido no braço)
        xm_c = xm_w - Lc_c * cm_w + (cm_w - cm_c) * 0.25

        # Coordenada X da raiz do canard
        xr_c = xm_c - ym_c * np.tan(sweep_c) + (cm_c - cr_c) * 0.25

        # Coordenadas da ponta
        xt_c = xr_c + yt_c * np.tan(sweep_c) + (cr_c - ct_c) * 0.25
        zt_c = zr_c + yt_c * np.tan(dihedral_c)

        # MAC Z
        zm_c = zr_c + ym_c * np.tan(dihedral_c)

        # Salvar no dicionário
        airplane["S_c"] = S_c
        airplane["b_c"] = b_c
        airplane["cr_c"] = cr_c
        airplane["ct_c"] = ct_c
        airplane["cm_c"] = cm_c
        airplane["xm_c"] = xm_c
        airplane["ym_c"] = ym_c
        airplane["zm_c"] = zm_c
        airplane["xr_c"] = xr_c
        airplane["xt_c"] = xt_c
        airplane["yt_c"] = yt_c
        airplane["zr_c"] = zr_c
        airplane["zt_c"] = zt_c
    else:
        airplane["S_c"] = 0.0

    # -------------------------------------------------------
    # Box Wing Sizing (Front + Rear wings) - SEM prefix
    if airplane.get("box_wing", False):

        # ==========================
        # Asa dianteira (front) - wf
        # ==========================
        S_wf = airplane["S_wf"]
        AR_wf = airplane["AR_wf"]
        taper_wf = airplane["taper_wf"]
        sweep_wf = airplane["sweep_wf"]
        dihedral_wf = airplane["dihedral_wf"]
        xr_wf = airplane["xr_wf"]
        zr_wf = airplane["zr_wf"]

        b_wf = np.sqrt(AR_wf * S_wf)
        cr_wf = 2 * S_wf / (b_wf * (1 + taper_wf))
        ct_wf = taper_wf * cr_wf
        yt_wf = b_wf * 0.5
        xt_wf = xr_wf + yt_wf * np.tan(sweep_wf) + (cr_wf - ct_wf) * 0.25
        zt_wf = zr_wf + yt_wf * np.tan(dihedral_wf)

        cm_wf = (2/3) * cr_wf * ((1 + taper_wf + taper_wf**2) / (1 + taper_wf))
        ym_wf = (b_wf / 6) * ((1 + 2*taper_wf) / (1 + taper_wf))
        xm_wf = xr_wf + ym_wf * np.tan(sweep_wf) + (cr_wf - cm_wf) * 0.25
        zm_wf = zr_wf + ym_wf * np.tan(dihedral_wf)

        airplane["b_wf"] = b_wf
        airplane["cr_wf"] = cr_wf
        airplane["ct_wf"] = ct_wf
        airplane["yt_wf"] = yt_wf
        airplane["xt_wf"] = xt_wf
        airplane["zt_wf"] = zt_wf
        airplane["cm_wf"] = cm_wf
        airplane["xm_wf"] = xm_wf
        airplane["ym_wf"] = ym_wf
        airplane["zm_wf"] = zm_wf

        # ==========================
        # Asa traseira (rear) - wr
        # ==========================
        S_wr = airplane["S_wr"]
        AR_wr = airplane["AR_wr"]
        taper_wr = airplane["taper_wr"]
        sweep_wr = airplane["sweep_wr"]
        dihedral_wr = airplane["dihedral_wr"]
        xr_wr = airplane["xr_wr"]
        zr_wr = airplane["zr_wr"]

        b_wr = np.sqrt(AR_wr * S_wr)
        cr_wr = 2 * S_wr / (b_wr * (1 + taper_wr))
        ct_wr = taper_wr * cr_wr
        yt_wr = b_wr * 0.5
        xt_wr = xr_wr + yt_wr * np.tan(sweep_wr) + (cr_wr - ct_wr) * 0.25
        zt_wr = zr_wr + yt_wr * np.tan(dihedral_wr)

        cm_wr = (2/3) * cr_wr * ((1 + taper_wr + taper_wr**2) / (1 + taper_wr))
        ym_wr = (b_wr / 6) * ((1 + 2*taper_wr) / (1 + taper_wr))
        xm_wr = xr_wr + ym_wr * np.tan(sweep_wr) + (cr_wr - cm_wr) * 0.25
        zm_wr = zr_wr + ym_wr * np.tan(dihedral_wr)

        airplane["b_wr"] = b_wr
        airplane["cr_wr"] = cr_wr
        airplane["ct_wr"] = ct_wr
        airplane["yt_wr"] = yt_wr
        airplane["xt_wr"] = xt_wr
        airplane["zt_wr"] = zt_wr
        airplane["cm_wr"] = cm_wr
        airplane["xm_wr"] = xm_wr
        airplane["ym_wr"] = ym_wr
        airplane["zm_wr"] = zm_wr


    # Update dictionary with new results       

    airplane["b_w"] = b_w   # [m] - Wing span
    airplane["cr_w"] = cr_w # [m] - Wing root chord
    airplane["xt_w"] = xt_w # [m] - Wing tip leading-edge x-position w.r.t the fuselage nose
    airplane["yt_w"] = yt_w # [m] - Wing semi-span
    airplane["zt_w"] = zt_w # [m] - Wing tip z-position w.r.t the fuselage nose
    airplane["ct_w"] = ct_w # [m] - Wing tip chord
    airplane["xm_w"] = xm_w # [m] - Wing MAC leading-edge x-position w.r.t the fuselage nose
    airplane["ym_w"] = ym_w # [m] - Wing MAC spanwise (y-axis) position w.r.t the wing root
    airplane["zm_w"] = zm_w # [m] - Wing MAC z-position w.r.t the fuselage nose
    airplane["cm_w"] = cm_w # [m] - Wing mean aerodynamic chord (MAC)
    
    airplane["xr_v"] = xr_v
    airplane["S_v"] = S_v   # [m^2] - Vertical tail area
    airplane["b_v"] = b_v   # [m] - Vertical tail span (geometric height)
    airplane["cr_v"] = cr_v # [m] - Vertical tail root chord
    airplane["xt_v"] = xt_v # [m] - VT tip leading-edge x-position w.r.t the fuselage nose
    airplane["zt_v"] = zt_v # [m] - VT tip z-position w.r.t the fuselage centerline
    airplane["ct_v"] = ct_v # [m] - Vertical tail tip chord
    airplane["xm_v"] = xm_v # [m] - VT MAC leading-edge x-position w.r.t the fuselage nose
    airplane["zm_v"] = zm_v # [m] - VT MAC z-position w.r.t the fuselage centerline
    airplane["cm_v"] = cm_v # [m] - Vertical tail mean aerodynamic chord (MAC)
    
	# All output variables are stored in the dictionary.
    # There is no need to return anything
    return None

# ----------------------------------------


def aerodynamics(
    airplane,
    Mach,
    altitude,
    CL,
    W0,
    n_engines_failed=0,
    highlift_config="clean",
    lg_down=0,
    h_ground=0,
    method=2,
    ind_drag_method="Nita",
    ind_drag_flap_method="Roskam",
):
    """
    
    Parameters
    ----------
    
    Mach: float -> Freestream Mach number.

    altitude: float -> Flight altitude [meters].

    CL: float -> Lift coefficient

    W0_guess: float -> Latest MTOW estimate [N]

    n_engines_failed: integer -> number of engines failed. Windmilling drag is
                                 added here. This number should be less than the
                                 total number of engines.

    highlift_config: 'clean', 'takeoff', or 'landing' -> Configuration of high-lift devices

    lg_down: 0 or 1 -> 0 for retraced landing gear or 1 for extended landing gear

    h_ground: float -> Distance between wing and the ground for ground effect [m].
                       Use 0 for no ground effect.

    method: 1 or 2 -> Method 1 applies a single friction coefficient
                      to the entire wetted area of the aircraft (based on Howe).
                      Method 2 is more refined since it computes friction and
                      form factors for each component.

    ind_drag_flap_method: 'Roskam' or 'Raymer' -> Roskam is simpler for conceptual design.
    
    Returns
    -------
    CD: total drag coefficient
        
    CL_max: maximum lift coefficient
            
    dragDict: drag brakdown dictionary
        
    Version Control
    ----------
    (Version/Date/Author/Modification)
    
    > Version 01 - 19/02/2026 - Rian - Headder adition, Outputs removal
    """

    # Unpacking dictionary
    S_w = airplane["S_w"] # [m^2] - Wing area
    AR_w = airplane["AR_w"] # [-] - Wing aspect ratio
    cr_w = airplane["cr_w"] # [m] - Wing root chord
    ct_w = airplane["ct_w"] # [m] - Wing tip chord
    taper_w = airplane["taper_w"] # [-] - Wing taper ratio
    sweep_w = airplane["sweep_w"] # [rad] - Wing sweep (at c/4)
    tcr_w = airplane["tcr_w"] # [-] - Wing relative thickness (t/c) at the root
    tct_w = airplane["tct_w"] # [-] - Wing relative thickness (t/c) at the tip
    b_w = airplane["b_w"] # [m] - Wing span
    cm_w = airplane["cm_w"] # [m] - Wing mean aerodynamic chord (MAC)

    clmax_w = airplane["clmax_w"]
    k_korn = airplane["k_korn"]

    S_h = airplane["S_h"]
    cr_h = airplane["cr_h"]
    ct_h = airplane["ct_h"]
    taper_h = airplane["taper_h"]
    sweep_h = airplane["sweep_h"]
    tcr_h = airplane["tcr_h"]
    tct_h = airplane["tct_h"]
    b_h = airplane["b_h"]
    cm_h = airplane["cm_h"]

    if airplane.get("has_canard", True):
        S_c = airplane["S_c"]
        cr_c = airplane["cr_c"]
        ct_c = airplane["ct_c"]
        taper_c = airplane["taper_c"]
        sweep_c = airplane["sweep_c"]
        tcr_c = airplane["tcr_c"]
        tct_c = airplane["tct_c"]
        b_c = airplane["b_c"]
        cm_c = airplane["cm_c"]
    S_v = airplane["S_v"]
    cr_v = airplane["cr_v"]
    ct_v = airplane["ct_v"]
    taper_v = airplane["taper_v"]
    sweep_v = airplane["sweep_v"]
    tcr_v = airplane["tcr_v"]
    tct_v = airplane["tct_v"]
    b_v = airplane["b_v"]
    cm_v = airplane["cm_v"]

    L_f = airplane["L_f"]
    D_f = airplane["D_f"]

    L_n = airplane["L_n"]
    D_n = airplane["D_n"]

    x_nlg = airplane["x_nlg"]  # This is only used to check if we have LG

    n_engines = airplane["n_engines"]
    n_engines_under_wing = airplane["n_engines_under_wing"]

    flap_type = airplane["flap_type"]
    c_flap_c_wing = airplane["c_flap_c_wing"]
    b_flap_b_wing = airplane["b_flap_b_wing"]

    slat_type = airplane["slat_type"]
    c_slat_c_wing = airplane["c_slat_c_wing"]
    b_slat_b_wing = airplane["b_slat_b_wing"]

    k_exc_drag = airplane["k_exc_drag"]

    has_winglet = airplane["winglet"]

    if "W0" in airplane.keys():
        W0 = airplane["W0"]
    else:
        W0 = airplane["W0_guess"]

    # Default rugosity value (smooth paint from Raymer Tab 12.5)
    rugosity = 0.634e-5

    # ------------------------------------------------------------------------
    # 3.2.3 Parasite drag - clean configuration
    
    # ====================
    # Parasite Drag - Wing
    # ====================
    # Área escondida pela fuselagem
    Shid_Sw = (D_f / (b_w * (1 + taper_w))) * (2 - (D_f / b_w) * (1 - taper_w))

    # Área exposta da asa
    Sexp_w = (1 - Shid_Sw) * S_w

    # Área molhada da asa
    Swet_w = (
        2
        * Sexp_w
        * (1 + (tcr_w / (4 * (1 + taper_w))) * (1 + taper_w * (tcr_w / tct_w)))
    )

    # Adição dos winglets (caso existam)
    if has_winglet:
        lambda_winglet = 0.21
        tc_winglet = 0.08
        Swet_w += (
            2
            * (ct_w**2)
            * (1 + lambda_winglet)
            * (1 + (tc_winglet / (4 * (1 + lambda_winglet))) * (1 + lambda_winglet))
        )

    # Coeficiente de fricção da asa
    Cf_w = Cf_calc(Mach, altitude, length=cm_w, rugosity=rugosity, k_lam=0.05)

    # Fator de forma da asa
    FF_w = FF_surface(Mach, tcr_w, tct_w, sweep_w, b_w, cr_w, ct_w)

    # Fator de interferência (assumido como 1 para asa com carenagens)
    Q_w = 1.0

    # Coeficiente de arrasto parasita da asa
    CD0_w = Cf_w * FF_w * Q_w * (Swet_w / S_w)

    # =============================
    # Parasite Drag - Horizontal Tail
    # =============================
    Sexp_h = S_h  # Toda a área considerada exposta
    Swet_h = (
        2
        * Sexp_h
        * (1 + (tcr_h / (4 * (1 + taper_h))) * (1 + taper_h * (tcr_h / tct_h)))
    )
    Cf_h = Cf_calc(Mach, altitude, length=cm_h, rugosity=rugosity, k_lam=0.05)
    FF_h = FF_surface(Mach, tcr_h, tct_h, sweep_h, b_h, cr_h, ct_h)
    Q_h = 1.04
    CD0_h = (Cf_h * FF_h * Q_h * (Swet_h / S_w))  # Atenção: divide por área de referência da asa

	# =============================
    # Parasite Drag - Canard - [Julia]
    # =============================
    CD0_c = 0.0
    has_canard = airplane.get("has_canard", True)

    if has_canard:
        Sexp_c = S_c
        Swet_c = (
            2
            * Sexp_c
            * (1 + (tcr_c / (4 * (1 + taper_c))) * (1 + taper_c * (tcr_c / tct_c)))
        )
        Cf_c = Cf_calc(Mach, altitude, length=cm_c,
                       rugosity=rugosity, k_lam=0.05)
        FF_c = FF_surface(Mach, tcr_c, tct_c, sweep_c, b_c, cr_c, ct_c)
        # Verificar fator
        Q_c = 1.04
        CD0_c = (Cf_c * FF_c * Q_c * (Swet_c / S_w))

    # =============================
    # Parasite Drag - Vertical Tail
    # =============================
    Sexp_v = S_v  # Toda a área considerada exposta
    Swet_v = (
        2
        * Sexp_v
        * (1 + (tcr_v / (4 * (1 + taper_v))) * (1 + taper_v * (tcr_v / tct_v)))
    )
    Cf_v = Cf_calc(Mach, altitude, length=cm_v, rugosity=rugosity, k_lam=0.05)
    FF_v = FF_surface(Mach, tcr_v, tct_v, sweep_v, 2 * b_v, cr_v, ct_v)
    Q_v = 1.04
    CD0_v = Cf_v * FF_v * Q_v * (Swet_v / S_w)


    # ===================
    # Parasite Drag - Fuselage
    # ===================
    taper_f = L_f / D_f
    Swet_f = (np.pi * D_f * L_f * ((1 - 2 / taper_f) ** (2 / 3)) * (1 + 1 / (taper_f**2)))  # Área molhada de um cilindro
    Cf_f = Cf_calc(Mach, altitude, length=L_f, rugosity=rugosity, k_lam=0.05)
    FF_f = 1 + 60 / (taper_f**3) + (taper_f / 400)
    Q_f = 1.0
    CD0_f = Cf_f * FF_f * Q_f * (Swet_f / S_w)

    # ===================
    # Parasite Drag - Nacelles
    # ===================
    Swet_n = np.pi * D_n * L_n * n_engines
    Cf_n = Cf_calc(Mach, altitude, length=L_n, rugosity=rugosity, k_lam=0.05)
    FF_n = 1 + 0.35 * (D_n / L_n)
    Q_n = 1.2
    CD0_n = Cf_n * FF_n * Q_n * (Swet_n / S_w)

    # ===================
    # Somatório do CD0 total - Configuração limpa
    # ===================
    # Arrasto parasita total (parcial)
    CD0_clean = CD0_w + CD0_h + CD0_v + CD0_f + CD0_n + CD0_c
    # ------------------------------------------------------------------------
    
    
    # ------------------------------------------------------------------------
    # 3.2.4 Oswald factor (clean configuration)
    AR_eff = AR_w

    if has_winglet:
        AR_eff = 1.2 * AR_w

    delta_lambda = -0.357 + 0.45 * np.exp(-0.0375 * (180 * sweep_w / np.pi))
    lambda_opt = taper_w - delta_lambda

    f_lambda = (
        0.0524 * lambda_opt**4
        - 0.15 * lambda_opt**3
        + 0.1659 * lambda_opt**2
        - 0.0706 * lambda_opt
        + 0.0119
    )

    e_theo = 1 / (1 + f_lambda * AR_eff)

    k_em = 1 / (1 + 0.12 * (Mach**6))

    k_ef = 1 - 2 * ((D_f / b_w) ** 2)

    e_clean = e_theo * k_ef * k_em * 0.873
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # 3.2.5 Wave Drag
    # Check if M > 0.4

    CD_wave = 0

    if Mach > 0.4:
        tc_w = 0.25 * tcr_w + 0.75 * tct_w
        sweep_50 = geo_change_sweep(0.25, 0.5, sweep_w, b_w / 2, cr_w, ct_w)
        Mach_dd = (
            k_korn / np.cos(sweep_50)
            - tc_w / np.square(np.cos(sweep_50))
            - CL/(10 * np.power(np.cos(sweep_50), 3))
        )
        Mach_c = Mach_dd - np.power(0.1/80, 1 / 3)
        CD_wave = 20 * ((max(0, Mach - Mach_c)) ** 4)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # 3.2.6 Maximum lift coefficient
    # CLmax_clean
    CLmax_clean = 0.9 * clmax_w * np.cos(sweep_w) 

    # Flaps & Slats
    dlift_dict = {"clean": 0, "takeoff": 0.75, "landing": 1}
    dlift = dlift_dict[highlift_config]

    # Flaps
    CD0_flap = 0
    Delta_eflap = 0
    deltaCLmax_flap = 0

    if flap_type is not None:
        flaptype_dict = {
            "Plain": [0.9, 0.9],
            "single slotted": [1.3 * (1 + c_flap_c_wing), 1.0],
            "double slotted": [1.6 * (1 + c_flap_c_wing), 1.2],
            "triple slotted": [1.9 * (1 + c_flap_c_wing), 1.5],
        }
        deltaclmax_flap = flaptype_dict[flap_type][0]
        Fflap = flaptype_dict[flap_type][1]
        flap_highlift_config_dict = {
            "clean": [0, 0],
            "takeoff": [(0.03 * Fflap - 0.004) / AR_eff**0.33, -0.05],
            "landing": [0.12 * Fflap / AR_eff**0.33, -0.1],
        }
        CD0_flap = flap_highlift_config_dict[highlift_config][0]
        Delta_eflap = flap_highlift_config_dict[highlift_config][1]

        Sflap_Sw = (
            b_flap_b_wing * (2 - b_flap_b_wing * (1 - taper_w)) / (1 + taper_w)
            - Shid_Sw
        )
        sweep_flap = geo_change_sweep(
            0.25, 1 - c_flap_c_wing, sweep_w, b_w / 2, cr_w, ct_w
        )

        deltaCLmax_flap = (
            0.9
            * deltaclmax_flap
            * Sflap_Sw
            * np.cos(sweep_flap)
            * dlift
            * c_flap_c_wing
            / 0.3
        )

    # Slats
    CD0_slat = 0
    deltaCLmax_slat = 0
    # slattype_dict
    if slat_type is not None:
        slattype_dict = {
            "slot": 0.2,
            "leading edge flap": 0.3,
            "Kruger flap": 0.3,
            "moving slat": 0.4 * (1 + c_slat_c_wing),
        }
        deltaclmax_slat = slattype_dict[slat_type]
        sweep_slat = geo_change_sweep(
            0.25, c_slat_c_wing, sweep_w, b_w / 2, cr_w, ct_w
        )
        Sslat_Sw = (
            b_slat_b_wing * (2 - b_slat_b_wing * (1 - taper_w)) / (1 + taper_w)
            - Shid_Sw
        )

        deltaCLmax_slat = (
            0.9
            * deltaclmax_slat
            * Sslat_Sw
            * np.cos(sweep_slat)
            * dlift
            * c_slat_c_wing
            / 0.15
        )
        CD0_slat = CD0_w * c_slat_c_wing * Sslat_Sw * np.cos(sweep_w) * dlift

    CLmax = CLmax_clean + deltaCLmax_flap + deltaCLmax_slat

    [T, p, rho, mi] = atmosphere(altitude)
    a = np.sqrt(1.4 * R_air * T)



    if highlift_config == "clean":
        V_stall = np.sqrt(2*0.95*W0/(rho*S_w*CLmax))
        Mach_stall = Mach
    elif highlift_config == "takeoff":
        V_stall = np.sqrt(2*W0/(rho*S_w*CLmax))
        Mach_stall = (V_stall * 1.2) / a
    elif highlift_config == "landing":
        V_stall = np.sqrt(2*0.85*W0/(rho*S_w*CLmax))
        Mach_stall = (V_stall * 1.3) / a


    #tol = ((Mach - Mach_stall) ** 2) ** 0.5
    Mach = Mach_stall

    # ------------------------------------------------------------------------
    # 3.2.7 Induced Drag
    # Effective Osvald Factor
    # From 3.2.4 Osvald Factor of the clean configuration & 3.2.6 Maximum lift coefficient and high-lift devices
    e = (e_clean + Delta_eflap)
    # From 3.2.4 Osvald Factor of the clean configuration
    K = 1 / (np.pi * AR_eff * e)
    #Ground Effect
    if h_ground > 0:
        GE = 33 * np.power(h_ground / b_w, 1.5)
        K_GE = GE / (1 + GE)
        K = K * K_GE

    CD_ind_clean = K * (CL ** 2)

    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # 3.2.8 Additional Components
    # Landing Gear parasite drag
    CD0_lg = 0
    if lg_down == 1:
        CD0_lg = 0.02

    # Inoperative Engines parasite drag
    CD0_windmill = n_engines_failed * 0.3 * np.pi / 4 * np.square(D_n) / S_w
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # 3.2.9 Excrescence Drag and total parasite drag
    CD0 = (CD0_clean + CD0_flap + CD0_slat + CD0_lg + CD0_windmill)/(1 - k_exc_drag)  # From 3.2.3 Clean Configuration
    CD0_exc = CD0 * k_exc_drag
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # 3.2.10 Total Drag
    CD = CD0 + CD_ind_clean + CD_wave
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # 3.2.13 - Implementado
    # Atualização do dicionário
    Swet = Swet_w + Swet_h + Swet_v + Swet_f + Swet_n

    # Create a drag breakdown dictionary
    dragDict = {
        "CD0_lg": CD0_lg,
        "CD0_wdm": CD0_windmill,
        "CD0_exc": CD0_exc,
        "CD0_flap": CD0_flap,
        "CD0_slat": CD0_slat,
        "CD0": CD0,
        "CDind_clean": CD_ind_clean,
        # 'CDind_flap' : CDind_flap,
        'CD' : CD,
        "CDwave": CD_wave,
        "CLmax_clean": CLmax_clean,
        "deltaCLmax_flap": deltaCLmax_flap,
        "deltaCLmax_slat": deltaCLmax_slat,
        "CLmax": CLmax,
        "K": K,
        "e": e,
        "Swet": Swet,
        "V_stall": V_stall,
        "Mach_stall": Mach_stall
    }

    if method == 2:
        dragDict["CD0_w"] = CD0_w
        dragDict["CD0_h"] = CD0_h
        dragDict["CD0_v"] = CD0_v
        dragDict["CD0_f"] = CD0_f
        dragDict["CD0_n"] = CD0_n

    # Update dictionary
    airplane["Swet_f"] = Swet_f
    airplane["AR_eff"] = AR_eff

    return CD, CLmax, dragDict


# ----------------------------------------


def engineTSFC(Mach, altitude, airplane):
    """
    Parameters
    ----------
    Mach : float
        Freestream Mach number.
    altitude : float
        Flight altitude [m].
    airplane : dict
        Reference aircraft.

    Returns
    -------
    C : float
        thrust-specific fuel consumption @altitude.
    kT : float
        thrust correction factor compared to
        static sea-level conditions.
    
    Description
    --------
    
    This function computes the engine thrust-specific fuel
    consumption and thrust correction factor compared to
    static sea-level conditions. The user has to define the
    engine parameters in a 'engine' dictionary within
    the airplane dictionary. The engine model must be
    identified by the 'model' field of the engine dictionary.
    The following engine models are available:

    Howe TSFC turbofan model:
    requires the bypass ratio. An optional sea-level TSFC
    could also be provided. Otherwise, standard parameters
    are used.
    airplane['engine'] = {'model': 'howe turbofan',
                          'BPR': 3.04,
                          'Cbase': 0.7/3600} # Could also be None

    Thermodynamic cycle turbojet:
    This model uses a simplified thermodynamic model of
    turbofans to estimate maximum thrust and TSFC

    airplane['engine'] = {'model': 'thermo turbojet'
                          'data': dictionary (check turbojet_model function)}

    The user can also leave a 'weight' field in the dictionary
    to replace the weight estimation.
    
            
    Version Control
    --------
    (Version/Date/Author/Modification)
    
    > Version 01 - 20/02/2026 - Rian - Headder adition
    
    
    """

    # Get a reference to the engine dictionary
    engine = airplane["engine"]

    # Check if C is defined, if not, define it by BPR
    if "Cbase" in engine.keys():
        Cbase = engine["Cbase"]
    else:
        if "BPR" in engine.keys():
            if engine["BPR"] < 4:
                Cbase = 0.85 / 3600
            else:
                Cbase = 0.7 / 3600
        else:
            raise ValueError("Engine model not defined or BPR not provided.")

    # Check if BPR is defined
    if "BPR" in engine.keys():
        BPR = engine["BPR"]
    else:
        raise ValueError("BPR not defined in engine dictionary.")

    T, p, rho, mi = atmosphere(altitude)
    sigma = rho / 1.225  # Ratio of air density to sea-level density

    # Specific fuel consumption at current flight condition [Howe model]
    C = (
        Cbase
        * (1 - 0.15 * (BPR**0.65))
        * (1 + 0.28 * (1 + 0.063 * (BPR**2)) * Mach)
        * (sigma**0.08)
    )

    # Thrust correction factor
    if BPR < 13:
        kT = (0.0013 * BPR - 0.0397) * altitude / 1000 - 0.0248 * BPR + 0.7125
    else:
        kT = 0.2
        


    K1_tau = 0.88
    K2_tau = -0.016
    K3_tau = -0.3
    K4_tau = 0
    S = 0.7
    M_N = 0.78
    R = 5
    F_tau = 1

    tau = F_tau*(K1_tau + K2_tau*R +(K3_tau + K4_tau*R)*M_N)*sigma**S

    return C, kT


# ----------------------------------------


def empty_weight(W0_guess, T0_guess, airplane):
    
    """
       Parameters
       ----------
       W0_guess : float
           MTOW estimation [N].
       T0_guess : float
           Thrust estimation [N].
       airplane : dict
           Reference aircraft.
    
       Returns
       -------
       W_empty : disct
           Empty weight dictionary
    
       x_cg : disct
           center of gravity dictionary
       
       Description
       --------
       This function calculates de empty weight and center of gravity of main components (eg. wing, eh, ev, etc)
       
               
       Version Control
       --------
       (Version/Date/Author/Modification)
       
       > Version 01 - 20/02/2026 - Rian - Headder adition  

    """
    # Unpack dictionary
    S_w = airplane["S_w"]
    AR_eff = airplane["AR_eff"]
    taper_w = airplane["taper_w"]
    sweep_w = airplane["sweep_w"]
    xm_w = airplane["xm_w"]
    cm_w = airplane["cm_w"]
    tcr_w = airplane["tcr_w"]
    b_w = airplane["b_w"]

    flap_type = airplane["flap_type"]
    c_flap_c_wing = airplane["c_flap_c_wing"]
    b_flap_b_wing = airplane["b_flap_b_wing"]
    slat_type = airplane["slat_type"]
    c_slat_c_wing = airplane["c_slat_c_wing"]
    b_slat_b_wing = airplane["b_slat_b_wing"]
    c_ail_c_wing = airplane["c_ail_c_wing"]
    b_ail_b_wing = airplane["b_ail_b_wing"]

    S_h = airplane["S_h"]
    xm_h = airplane["xm_h"]
    cm_h = airplane["cm_h"]

    if airplane.get("has_canard", True):
        S_c = airplane["S_c"]
        xm_c = airplane["xm_c"]
        cm_c = airplane["cm_c"]
    S_v = airplane["S_v"]
    xm_v = airplane["xm_v"]
    cm_v = airplane["cm_v"]

    L_f = airplane["L_f"]
    D_f = airplane["D_f"]
    Swet_f = airplane["Swet_f"]

    n_engines = airplane["n_engines"]
    x_n = airplane["x_n"]
    L_n = airplane["L_n"]

    x_nlg = airplane["x_nlg"]
    x_mlg = airplane["x_mlg"]

    altitude_cruise = airplane["altitude_cruise"]
    Mach_cruise = airplane["Mach_cruise"]
    BPR = airplane["engine"]["BPR"]

    airplane_type = airplane["type"]

    # ----------------------------

    # 3.4.3 Wing weight - Transport Aircraft

    Nz = 1.5 * 2.5

    def aux_func(x1, y1, y2, m, taper):
        ss = (x1/(1+taper))*(y2*(2 - y2*(1 - taper)) - y1*(2 - y1*(1-taper)))*m
        return ss

    # FLAP
    x1 = c_flap_c_wing
    y1 = D_f / b_w
    y2 = b_flap_b_wing
    if flap_type == 'Plain':
        m = 1
    elif flap_type == 'single slotted':
        m = 1.15*1.25
    elif flap_type == 'double slotted':
        m = 1.30*1.25
    elif flap_type == 'triple slotted':
        m = 1.45*1.25
    else:
        m = 0

    S_flap_Sw = aux_func(x1, y1, y2, m, taper_w)

    # SLAT
    x1 = c_slat_c_wing
    y1 = D_f / b_w
    y2 = b_slat_b_wing
    if slat_type == 'slot':
        m = 1
    elif slat_type == 'leading edge flap':
        m = 1
    elif slat_type == 'Kruger flap':
        m = 1
    elif slat_type == 'moving slat':
        m = 1.25
    else:
        m = 0

    S_slat_Sw = aux_func(x1, y1, y2, m, taper_w)

    # AILERON
    x1 = c_ail_c_wing
    y1 = 1 - b_ail_b_wing
    y2 = 1
    m = 1
    S_ail_Sw = aux_func(x1, y1, y2, m, taper_w)

    S_csw = (S_flap_Sw + S_slat_Sw + S_ail_Sw) * S_w

    W_w = lb2N * 0.0051*((W0_guess/lb2N*Nz)**0.557)*((S_w/(ft2m**2))**0.649)*(AR_eff**0.55)*(
        tcr_w**(-0.4))*((1+taper_w)**0.1)*(np.cos(sweep_w))**(-1)*(S_csw/ft2m**2)**(0.1)

    xcg_w = xm_w + 0.4 * cm_w

    # ------------------------------------------------------------------------------------------------------------------------------------
    # 3.4.4 Horizontal tail weight
    W_h = 27 * gravity * S_h
    xcg_h = xm_h + 0.4 * cm_h # Center of gravity of horizontal tail
    # ------------------------------------------------------------------------------------------------------------------------------------
    # Canard weight
    W_c = 0
    xcg_c = 0
    if airplane.get("has_canard", True):
        W_c = 27 * 9.81 * S_c
        xcg_c = xm_c + 0.4 * cm_c
    # ---------------------------------------------------------------------------------------------------------------------
    # 3.4.5 Vertical tail weight
    W_v = 27 * gravity * S_v
    xcg_v = xm_v + 0.4 * cm_v # Center of gravity of vertical tail
    # ------------------------------------------------------------------------------------------------------------------------------------
    # 3.4.6 Fuselage weight
    W_f = 24 * gravity * Swet_f
    xcg_f = 0.45 * L_f          # Center of gravity of fuselage
    # ------------------------------------------------------------------------------------------------------------------------------------
    # 3.4.7 Nose landing gear weight
    W_nlg = 0.15 * 0.043 * W0_guess
    xcg_nlg = x_nlg
    # ------------------------------------------------------------------------------------------------------------------------------------
    # 3.4.8 Main landing gear weight
    W_mlg = 0.85 * 0.043 * W0_guess
    xcg_mlg = x_mlg
    # ------------------------------------------------------------------------------------------------------------------------------------
    # 3.4.9 Installed engine weight
    Teng_s = T0_guess / n_engines

    Weng_s = 14.7 * gravity * (Teng_s / 1000) ** 1.1 * math.exp(-0.045 * BPR)

    # Get a reference to the engine dictionary
    engine = airplane["engine"]

    # Check if C is defined, if not, define it by BPR
    if "W_eng_spec" in engine.keys():
        W_eng = n_engines * airplane['engine']['W_eng_spec']
    else:
        W_eng = 1.3 * n_engines * Weng_s

    xcg_eng = x_n + 0.5 * L_n

    # ------------------------------------------------------------------------------------------------------------------------------------
    # 3.4.10 All-else weight

    if airplane['name'] == "CRJ200":
        W_ae = 0.2 * W0_guess
        W_ae = 0.2 * W0_guess
    else: 
        W_ae = 0.17 * W0_guess
        W_ae = 0.15 * W0_guess

    xcg_ae = airplane['perc_cg_ae'] * L_f

    # ------------------------------------------------------------------------------------------------------------------------------------

    # 3.4.11 Empty weight
    W_empty = W_w + W_h + W_c + W_v + W_f + W_nlg + W_mlg + W_eng + W_ae
    xcg_empty = (
        W_w * xcg_w
        + W_h * xcg_h
		+ W_c * xcg_c
        + W_v * xcg_v
        + W_f * xcg_f
        + W_nlg * xcg_nlg
        + W_mlg * xcg_mlg
        + W_eng * xcg_eng
        + W_ae * xcg_ae
    ) / W_empty

    # ------------------------------------------------------------------------------------------------------------------------------------

    # 3.4.12 Auxiliary function to compute control surface area

    def control_surface_fraction(control_surface_type):
        S_flap_S_wing = flap_area_fraction(
            c_flap_c_wing, D_f / b_w, b_flap_b_wing, taper_w, flap_type
        )

        S_slat_S_wing = slat_area_fraction(
            c_slat_c_wing, D_f / b_w, b_slat_b_wing, taper_w, slat_type
        )

        S_ail_S_wing = aileron_area_fraction(
            c_ail_c_wing, 1 - b_ail_b_wing, 1, taper_w)

        if control_surface_type == "flap":
            return S_flap_S_wing
        elif control_surface_type == "slat":
            return S_slat_S_wing
        elif control_surface_type == "aileron":
            return S_ail_S_wing

    # ------------------------------------------------------------------------------------------------------------------------------------

    empty_weight_Dict = {
        'W_w' : W_w,
        'W_h' : W_h,
        'W_c' : W_c,
        'W_v' : W_v,
        'W_f' : W_f,
        'W_nlg' : W_nlg,
        'W_mlg' : W_mlg,
        'W_eng' : W_eng,
        'W_allelse' : W_ae,
    }
    
    CG_Dict = {
        'xcg_empty' : xcg_empty,
        'xcg_w'  : xcg_w,
        'xcg_h' : xcg_h,
        'xcg_c' : xcg_c,
        'xcg_v' : xcg_v,
        'xcg_f' : xcg_f,
        'xcg_nlg' : xcg_nlg,
        'xcg_mlg' : xcg_mlg,
        'xcg_eng' : xcg_eng,
        'xcg_ae' : xcg_ae,
    }
 

    return W_empty, empty_weight_Dict, CG_Dict

# ----------------------------------------


def fuel_weight(W0_guess, airplane, range_cruise=0, altcruise=True, loiter=True, update_Mf_hist=False):
    """
    
    Parameters
    ----------
    W0_guess : float
        Weight estimation [N].
    airplane : dict
        reference aircraft.
    range_cruise : float, optional
        Cruise range [m] . The default is 0.
    altcruise : bool, optional
        If yes, count alternative cruise section. The default is True.
    loiter : bool, optional
        If yes, count loiter section. The default is True.
    update_Mf_hist : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    W_fuel: float.
         Fuel weight [N].      
    W_cruise:float.
         Aircraft weight in the beginning of cruise phase [N].
    Mf_vec:float.
        fuel fractions at each flight phase.
    Mf:float
        total fuel fraction.
    W_fuel_mission: float.
        total fuel consumed during flight.
    
    Description
    --------
    This function calculates the fuel consumption during the mission.
    
    Version Control
    --------
    (Version/Date/Author/Modification)
    
    > Version 01 - 20/02/2026 - Rian - Headder adition  

    """
    
    # Unpacking dictionary
    S_w = airplane["S_w"]

    altitude_cruise = airplane["altitude_cruise"]
    Mach_cruise = airplane["Mach_cruise"]
    
    if range_cruise != 0:
        R_cruise = range_cruise
    else:
        R_cruise = airplane["range_cruise"]

    loiter_time = airplane["loiter_time"]

    altitude_altcruise = airplane["altitude_altcruise"]
    Mach_altcruise = airplane["Mach_altcruise"]
    range_altcruise = airplane["range_altcruise"]

    airplane_type = airplane["type"]

    if loiter == False:
        loiter_time = 0
    if altcruise == False:
        range_altcruise = 0
    # 3.5.3 - Initialization [Rogério]
    C_cruise, kt = engineTSFC(Mach_cruise, altitude_cruise, airplane)
    C_altcruise, kt = engineTSFC(Mach_altcruise, altitude_altcruise, airplane)

    # 3.5.4 - Engine Start and Warm-up 
    Mf_start = 0.990

    # 3.5.5 - Taxi
    Mf_taxi = 0.990

    # 3.5.6 - Takeoff 
    Mf_takeoff = 0.995

    # 3.5.7 - Climb 
    Mf_climb = 0.980

    # 3.5.8 - Cruise 
    W_cruise = W0_guess * Mf_start * Mf_taxi * Mf_takeoff * Mf_climb

    T, p, rho, mi = atmosphere(altitude_cruise, 288.15)
    R = 287
    gamma = 1.4
    a_cruise = np.sqrt(gamma * R * T)
    V_cruise = Mach_cruise * a_cruise

    CL_cruise = 2 * W_cruise / (rho * S_w * (V_cruise**2))
    CD_cruise, CLmax_cruise, dragDict_cruise = aerodynamics(
        airplane,
        Mach_cruise,
        altitude_cruise,
        CL_cruise,
        W_cruise,
        highlift_config="clean",
    )

    Mf_cruise = np.exp(-R_cruise * C_cruise *
                       CD_cruise / (V_cruise * CL_cruise))
    

    # 3.5.9 - Loiter 
    L_D_max = 1 / \
        (2*np.sqrt((dragDict_cruise['CD0'] +
         dragDict_cruise['CDwave'])*dragDict_cruise['K']))
    C_loiter = C_cruise - 0.1 / 3600
    E_loiter = loiter_time
    Mf_loiter = np.exp(-E_loiter * C_loiter / L_D_max)

    # 3.5.10 - Descent 
    Mf_descent = 0.99

    # 3.5.11 - Alternate cruise 
    T_altcruise, _, rho, _ = atmosphere(altitude_altcruise, 288.15)
    a_altcruise = np.sqrt(gamma_air * R_air * T_altcruise)

    V_altcruise = Mach_altcruise*a_altcruise

    W_altcruise = W_cruise * Mf_cruise * Mf_loiter * Mf_descent

    CL_altcruise = 2 * W_altcruise / (rho * S_w * V_altcruise**2)


    CD_altcruise, CLmax_altcruise, dragDict_altcruise = aerodynamics(
        airplane,
        Mach_altcruise,
        altitude_altcruise,
        CL_altcruise,
        W_altcruise,
        highlift_config="clean",
    )


    Mf_altcruise = np.exp(-range_altcruise*C_altcruise*CD_altcruise /(V_altcruise*CL_altcruise))

    # 3.5.12 - Landing taxi and shutdown 
    Mf_landing = 0.992

    # 3.5.13 - Fuel weight 

    Mf = (
        Mf_start
        * Mf_taxi
        * Mf_takeoff
        * Mf_climb
        * Mf_cruise
        * Mf_loiter
        * Mf_descent
        * Mf_altcruise
        * Mf_landing
    )

    Mf_vec = [Mf_start, Mf_taxi, Mf_takeoff, Mf_climb, Mf_cruise,
              Mf_loiter, Mf_descent, Mf_altcruise, Mf_landing]
    
    W_fuel = 1.06 * (1 - Mf) * W0_guess
    
    W_fuel_loiter = 1.06 * Mf/(Mf_loiter * Mf_descent * Mf_altcruise * Mf_landing) * (1 - Mf_loiter) * W0_guess
    
    W_fuel_alt_cruise = 1.06 * Mf/(Mf_altcruise * Mf_landing) * (1 - Mf_altcruise) * W0_guess
    
    W_fuel_extra = W_fuel_loiter + W_fuel_alt_cruise
    
    W_fuel_mission = W_fuel - W_fuel_extra


    return W_fuel, W_cruise, Mf_vec, Mf, W_fuel_mission


# ----------------------------------------


def weight(W0_guess, T0_guess, airplane, R_cruise=0):
    """
    Parameters
    ----------
    W0_guess : float
        MTOW estimation [N].
    T0_guess : float
        thrust estimation [N].
    airplane : dict
        reference aircraft.
    R_cruise : float, optional
        cruise range estimation. The default is 0.

    Returns
    -------
    W0: float.
        MTOW [N]
    W_empty: float.
        Empty weight [N]
    W_fuel: float.
         Fuel weight [N].      
    W_cruise:float.
         Aircraft weight in the beginning of cruise phase [N].
    W_fuel_mission: float.
        total fuel consumed during flight.
   
    Description
    -------
    This function calculates all fractions of aircraft weights.
    
    
    Version Control
    -------
   (Version/Date/Author/Modification)
   
   > Version 01 - 20/02/2026 - Rian - Headder adition  
    """    
    # Unpacking dictionary
    W_payload = airplane["W_payload"]
    W_crew = airplane["W_crew"]
    
    if R_cruise!=0:
        range_cruise = R_cruise
    else:
        range_cruise = airplane["range_cruise"]


    # Set iterator
    delta = 1000
    # w = 1.2
    while abs(delta) > 10:
        w = min(1 + abs(delta)/1000, 2)

        # We need to call fuel_weight first since it
        # calls the aerodynamics module to get Swet_f used by
        # the empty weight function
        W_fuel, W_cruise, _, _, W_fuel_mission = fuel_weight(
            W0_guess, airplane, range_cruise=range_cruise, update_Mf_hist=True)
        W_empty,_,_ = empty_weight(W0_guess, T0_guess, airplane)
            
        W0 = W_empty + W_fuel + W_payload + W_crew
        delta = W0 - W0_guess
        W0_guess = (1-w)*W0_guess + w*W0


    return W0, W_empty, W_fuel, W_cruise, W_fuel_mission
# ----------------------------------------

def performance(W0, W_cruise, airplane):
    """
    Parameters
    ----------
    W0 : float
        MTOW [N].
    W_cruise : float 
        Weight at the beguining of the cruise.
    airplane : dict
        aircraft to be analyzed.

    Returns
    -------
    T0: Total thrust required to meet all mission phases [N]
        
    T0vec: Required Thrust by flight phase
        
    deltaS_w_lan: Wing area margin for landing. This value should be positive
                  for a feasible landing.
        
    CLmaxTO: Maximum CL at takeoff
    
    Description
    ----------
    This function computes the required thrust and wing areas
    required to meet takeoff, landing, climb, and cruise requirements.
    
    Version Control
    -------
   (Version/Date/Author/Modification)
   
   > Version 01 - 23/02/2026 - Rian - Headder adition 
   > Version 02 - 24/02/2026 - Julia - Climb methods added 
    """

    # Unpacking dictionary
    S_w = airplane["S_w"]

    n_engines = airplane["n_engines"]

    h_ground = airplane["h_ground"]

    altitude_takeoff = airplane["altitude_takeoff"]
    distance_takeoff = airplane["distance_takeoff"]
    deltaISA_takeoff = airplane["deltaISA_takeoff"]

    altitude_landing = airplane["altitude_landing"]
    distance_landing = airplane["distance_landing"]
    deltaISA_landing = airplane["deltaISA_landing"]
    MLW_frac = airplane["MLW_frac"]

    altitude_cruise = airplane["altitude_cruise"]
    Mach_cruise = airplane["Mach_cruise"]

    altitude_ceiling = airplane["altitude_ceiling"]
    Mach_ceiling = airplane["Mach_ceiling"]

    # 3.7.3 - takeoff analysis 
    T, p, rho, mi = atmosphere(altitude_takeoff, 288.15 + deltaISA_takeoff)

    sig = rho/1.225

    Mach_takeoff = 0.2
    CL_takeoff = 0.5  # This parameter is not used to estimate CLmax, so any value is okay

    _, CLmaxTO, _= aerodynamics(
        airplane,
        Mach_takeoff,
        altitude_takeoff,
        CL_takeoff,
        W0,
        n_engines_failed=0,
        highlift_config='takeoff',
        lg_down=1,
        h_ground=h_ground
    )

    T0_W0 = 0.2387/(sig*CLmaxTO*distance_takeoff)*W0/S_w

    T0_to = T0_W0*W0

    # 3.7.4 - Landing analysis

    T, p, rho, mi = atmosphere(altitude_landing, 288.15 + deltaISA_landing)

    Mach_landing = 0.2  # This parameter is not used to estimate CLmax, so any value is okay
    CL_landing = 0.5  # This parameter is not used to estimate CLmax, so any value is okay

    _, CLmaxLA, _ = aerodynamics(
        airplane,
        Mach_landing,
        altitude_landing,
        CL_landing,
        W0,
        n_engines_failed=0,
        highlift_config='landing',
        lg_down=1,
        h_ground=h_ground
    )

    h_lan = 15.3
    f_lan = 5/3
    a_g = 0.5

    x_lan = 1.52/a_g + 1.69 

    A_lan = gravity/(f_lan*x_lan)

    B_lan = -10*gravity*(h_lan/x_lan)

    S_w_lan = (W0*(MLW_frac))/(rho*CLmaxLA*(A_lan*distance_landing + B_lan))

    deltaS_w_lan = S_w - S_w_lan

    # 3.7.5 - cruise analysis 

    # Weight at the beginning of cruise
    # W_cruise = W0 * Mf_cruise

    # Atmospheric properties
    T, p, rho, mi = atmosphere(altitude_cruise)
    a_cruise = np.sqrt(gamma_air * R_air * T)
    V_cruise = Mach_cruise * a_cruise # Cruise speed

    # Cruise lift coefficient
    CL_cruise = (2 * W_cruise) / (rho * S_w * V_cruise ** 2)

    # Drag coefficient at cruise
    CD_cruise, _, _= aerodynamics(
        airplane,
        Mach_cruise,
        altitude_cruise,
        CL_cruise,
        W_cruise,
        n_engines_failed=0,
        highlift_config='clean',
        lg_down=0,
        h_ground=0
    )

    # Required thrust at cruise
    T_cruise = 0.5 * rho * V_cruise**2 * S_w * CD_cruise

    P_cruise = T_cruise * V_cruise

    # Correction factor to takeoff conditions
    _, kT = engineTSFC(Mach_cruise, altitude_cruise, airplane)
    T0_cruise = T_cruise / kT

    # ===================================
    # Ceiling Analysis 

    # Atmospheric properties
    T, p, rho, mi = atmosphere(altitude_ceiling)

    a_ceiling = np.sqrt(gamma_air * R_air * T)
    # a_cruise = np.sqrt(1.4 * 287 * T)

    # Cruise speed
    V_ceiling = Mach_ceiling * a_ceiling # [Julia] Dúvida: Mach Ceiling

    # Cruise lift coefficient
    CL_ceiling = (2 * W_cruise) / (rho * S_w * V_ceiling ** 2)  # [Julia] Dúvida: mesmo peso de cruzeiro?

    # Drag coefficient at cruise
    CD_ceiling, _, dragDict_ceiling = aerodynamics(
        airplane,
        Mach_ceiling,
        altitude_ceiling,
        CL_ceiling,
        W_cruise,
        n_engines_failed=0,
        highlift_config='clean',
        lg_down=0,
        h_ground=0
    )
    
    CD0_ceiling = dragDict_ceiling['CD0']
    K_ceiling = dragDict_ceiling['K']

    _, kT = engineTSFC(Mach_ceiling, altitude_ceiling, airplane)

    G = 0.001

    Tce_Wce = (G + 2 * (CD0_ceiling * K_ceiling) ** (0.5))/kT 

    T_ceiling = Tce_Wce*W0

    # ===================================
    # 3.7.6 Climb Analysis 

    def climb_analysis(gamma_climb, ks, altitude_climb, lg_down, h_ground_climb, highlift_config, n_engines_failed, Mf, W0, S_w, kT, deltaISA_takeoff):

        T, p, rho, mi = atmosphere(altitude_climb, 288.15 + deltaISA_takeoff)

        a_climb = np.sqrt(gamma_air*R_air*T)

        Mach = 0.2  # This parameter is not used to estimate CLmax, so any value is okay
        CL = 0.5  # This parameter is not used to estimate CLmax, so any value is okay

        _, CLmax_climb, _ = aerodynamics(
            airplane,
            Mach,
            altitude_climb,
            CL,
            W0,
            n_engines_failed,
            highlift_config,
            lg_down,
            h_ground_climb
        )

        # Climb lift coeff
        CL_climb = CLmax_climb/ks**2

        # Climb speed
        V_climb = np.sqrt((2*W0*Mf)/(rho*S_w*CL_climb))

        # Climb mach
        Mach_climb = V_climb/a_climb

        # corresponding cd
        CD_climb, _, _= aerodynamics(
            airplane,
            Mach_climb,
            altitude_climb,
            CL_climb,
            W0,
            n_engines_failed,
            highlift_config,
            lg_down,
            h_ground_climb
        )

        # thrust-to-weight ratio of this climb condition
        T0_W_climb = (n_engines/(n_engines-n_engines_failed)) * \
            (gamma_climb+(CD_climb/CL_climb))
    

        T0_climb = T0_W_climb*W0*Mf/kT

        # Use kT = 0.94 when requirement asks for maximum continuous thrust instead of maximum takeoff thrust.

        return T0_climb

    gamma_FAR25111 = {
        2: 0.012,
        3: 0.015,
        4: 0.017,
    }
    gamma_FAR25121a = {
        2: 0,
        3: 0.003,
        4: 0.005,
    }

    gamma_FAR25121b = {
        2: 0.024,
        3: 0.027,
        4: 0.030,
    }

    gamma_FAR25121c = {
        2: 0.012,
        3: 0.015,
        4: 0.017,
    }

    gamma_FAR25121d = {
        2: 0.021,
        3: 0.024,
        4: 0.027,
    }

    T0_climb_FAR25111 = climb_analysis(
        gamma_climb=gamma_FAR25111.get(n_engines),
        ks=1.2,
        altitude_climb=altitude_takeoff,
        lg_down=0,  # extended
        h_ground_climb=h_ground,
        highlift_config='takeoff',
        n_engines_failed=1,  # OEI Condition
        Mf=1,
        W0=W0,
        S_w=S_w,  # isso aqui pode deixar sem o "=" né?
        kT=1,
        deltaISA_takeoff=deltaISA_takeoff,
    )

    T0_climb_FAR25121a = climb_analysis(
        gamma_climb=gamma_FAR25121a.get(n_engines),
        ks=1.1,
        altitude_climb=altitude_takeoff,
        lg_down=1,  # retracted
        h_ground_climb=h_ground,
        highlift_config='takeoff',
        n_engines_failed=1,  # OEI Condition
        Mf=1,
        W0=W0,
        S_w=S_w,  # isso aqui pode deixar sem o "=" né?
        kT=1,
        deltaISA_takeoff=deltaISA_takeoff,
    )

    T0_climb_FAR25121b = climb_analysis(
        gamma_climb=gamma_FAR25121b.get(n_engines),
        ks=1.2,
        altitude_climb=altitude_takeoff,
        lg_down=0,  # retracted
        h_ground_climb=0,
        highlift_config='takeoff',
        n_engines_failed=1,  # OEI Condition
        Mf=1,
        W0=W0,
        S_w=S_w,  # isso aqui pode deixar sem o "=" né?
        kT=1,
        deltaISA_takeoff=deltaISA_takeoff,
    )

    T0_climb_FAR25121c = climb_analysis(
        gamma_climb=gamma_FAR25121c.get(n_engines),
        ks=1.25,
        altitude_climb=altitude_takeoff,
        lg_down=0,  # retracted
        h_ground_climb=0,
        highlift_config='clean',
        n_engines_failed=1,  # OEI Condition
        Mf=1,
        W0=W0,
        S_w=S_w,  # isso aqui pode deixar sem o "=" né?
        kT=0.94,
        deltaISA_takeoff=deltaISA_takeoff,
    )

    T0_climb_FAR25119 = climb_analysis(
        gamma_climb=0.032,
        ks=1.3,
        altitude_climb=altitude_landing,
        lg_down=1,  # extended
        h_ground_climb=0,
        highlift_config='landing',
        n_engines_failed=0,
        Mf=MLW_frac,
        W0=W0,
        S_w=S_w,  # isso aqui pode deixar sem o "=" né?
        kT=1,
        deltaISA_takeoff=deltaISA_takeoff,
    )

    T0_climb_FAR25121d = climb_analysis(
        gamma_climb=gamma_FAR25121d.get(n_engines),
        ks=1.40,
        altitude_climb=altitude_landing,
        lg_down=1,  # extended
        h_ground_climb=0,
        highlift_config='takeoff',
        n_engines_failed=1,  # OEI Condition
        Mf=MLW_frac,
        W0=W0,
        S_w=S_w,  # isso aqui pode deixar sem o "=" né?
        kT=1,
        deltaISA_takeoff=deltaISA_takeoff,
    )
    
    def time_to_climb_mean_ROC_constraint_old(
    t_req,                  # [s] tempo total requerido 
    h_start,                # [m]
    h_end,                  # [m]
    dh,                     # [m] passo em altitude
    ks,
    lg_down,
    h_ground_climb,
    highlift_config,
    n_engines_failed,
    Mf,
    W0,                     # [N]
    S_w,                    # [m²]
):

        # ----------------- razão de subida média requerida -----------------
        ROC_req = (h_end - h_start) / t_req   # [m/s]

        W_eff = W0 * Mf

        T_W_req_max = -1.0
        h_req = None
        Mach_req = None
        kT_req = None

        # ----------------- varredura em altitude -----------------
        for h in np.arange(h_start, h_end + dh, dh):

            # Atmosfera
            T, p, rho, mi = atmosphere(h, 288.15)
            a = np.sqrt(gamma_air * R_air * T)

            # CLmax
            Mach_dummy = 0.2
            CL_dummy = 0.5

            _, CLmax, _ = aerodynamics(
                airplane,
                Mach_dummy,
                h,
                CL_dummy,
                W0,
                n_engines_failed,
                highlift_config,
                lg_down,
                h_ground_climb
            )

            # CL e velocidade representativa de climb
            CL = CLmax / ks**2
            V = np.sqrt((2 * W_eff) / (rho * S_w * CL))
            Mach = V / a

            # Arrasto
            CD, _, _ = aerodynamics(
                airplane,
                Mach,
                h,
                CL,
                W0,
                n_engines_failed,
                highlift_config,
                lg_down,
                h_ground_climb
            )

            D_W = CD / CL

            T_W_req = D_W + ROC_req / V

            if T_W_req > T_W_req_max:
                T_W_req_max = T_W_req
                h_req = h
                Mach_req = Mach
                _, kT_req = engineTSFC(Mach, h, airplane)

        T_req = T_W_req_max * W_eff      # [N]
        T_corr = T_req / kT_req          # [N]  tração corrigida

        crit_data = {
            "T_req": T_req,
            "T_corr": T_corr,
            "h_req": h_req,
            "Mach_req": Mach_req,
            "kT_req": kT_req,
            "ROC_req": ROC_req
        }

        return T_corr, crit_data
    
    T_corr_old, crit = time_to_climb_mean_ROC_constraint_old(
    t_req=20*60,
    h_start=0.0,
    h_end=35000 * ft2m,
    dh=50.0,
    ks=1.2,
    lg_down=1,
    h_ground_climb=0.0,
    highlift_config="clean",
    n_engines_failed=0,
    Mf=1.0,
    W0=W0,
    S_w=S_w
)
    
    def time_marching_climb_constraint(
        t_req,                  # [s] 
        h_start,                # [m]
        h_end,                  # [m]
        dt,                     # [s]
        ks,
        lg_down,
        h_ground_climb,
        highlift_config,
        n_engines_failed,
        Mf,
        W0,                     # [N]
        S_w,                    # [m²]
        T_inst_min,             # [N]
        T_inst_max,             # [N]
        gamma_min=0.024,
        tol=1e-3
    ):

        W_eff = W0 * Mf

        def simulate_climb(T_inst):

            h = h_start
            t = 0.0

            # ----------------- ponto que governa o diagrama -----------------
            T_req_max = -1.0
            h_req = None
            Mach_req = None
            kT_req = None

            # Velocidade inicial
            T0, p0, rho0, _ = atmosphere(h, 288.15)

            Mach_dummy = 0.2
            CL_dummy = 0.5

            _, CLmax0, _ = aerodynamics(
                airplane,
                Mach_dummy,
                h,
                CL_dummy,
                W0,
                n_engines_failed,
                highlift_config,
                lg_down,
                h_ground_climb
            )

            CL = CLmax0 / ks**2
            V = np.sqrt((2 * W_eff) / (rho0 * S_w * CL))


            # ----------------- integração no tempo -----------------
            while h < h_end:

                # Atmosfera
                T, p, rho, mi = atmosphere(h, 288.15)
                a = np.sqrt(gamma_air * R_air * T)
                Mach = V / a

                # Aerodinâmica
                CD, _, _ = aerodynamics(
                    airplane,
                    Mach,
                    h,
                    CL,
                    W0,
                    n_engines_failed,
                    highlift_config,
                    lg_down,
                    h_ground_climb
                )

                D = 0.5 * rho * V**2 * S_w * CD

                # -------- tração requerida local --------
                T_req_local = D + W_eff * gamma_min

                if T_req_local > T_req_max:
                    T_req_max = T_req_local
                    h_req = h
                    Mach_req = Mach
                    _, kT_req = engineTSFC(Mach, h, airplane)

                # -------- verificação de capacidade --------
                gamma_real = (T_inst - D) / W_eff

                if gamma_real < gamma_min:
                    return np.inf, None

                ROC = V * gamma_real

                # Atualizações
                h += ROC * dt
                t += dt

                if t > t_req * 1.5:
                    return np.inf, None

            crit_data = {
                "T_req_max": T_req_max,
                "h_req": h_req,
                "Mach_req": Mach_req,
                "kT_req": kT_req
            }

            return t, crit_data

        # ----------------- busca por bisseção -----------------
        T_low = T_inst_min
        T_high = T_inst_max

        t_test, _ = simulate_climb(T_high)
        if t_test > t_req:
            raise ValueError("Tração máxima insuficiente para cumprir o climb.")

        crit_data_final = None

        while (T_high - T_low) / T_high > tol:
            T_mid = 0.5 * (T_low + T_high)
            t_mid, crit_data = simulate_climb(T_mid)

            if t_mid <= t_req:
                T_high = T_mid
                crit_data_final = crit_data
            else:
                T_low = T_mid

        # ----------------- correção com kT do ponto governante -----------------
        kT_req = crit_data_final["kT_req"]

        # print(kT_req)

        T_corrected = T_high / kT_req

        return T_high, T_corrected, crit_data_final
    
    
    T_inst_req, T0_time_to_climb, crit = time_marching_climb_constraint(
    t_req=20*60,
    h_start=0.0,
    h_end=35000 * ft2m,
    dt=5.0,
    ks=1.2,
    lg_down=1,
    h_ground_climb=0.0,
    highlift_config="clean",
    n_engines_failed=0,
    Mf=1.0,
    W0=W0,
    S_w=S_w,
    T_inst_min=20000.0,
    T_inst_max=200000.0
)

    # ==================================
    # 3.7.7 Gathering outputs [Sawada]

    T0vec = [T0_to, T0_cruise, T_ceiling, T0_climb_FAR25111, T0_climb_FAR25121a,
             T0_climb_FAR25121b, T0_climb_FAR25121c, T0_climb_FAR25119, T0_climb_FAR25121d, T0_time_to_climb]
    # Get the maximum required thrust with a 5% margin
    T0 = 1.05 * max(T0vec)

    return T0, T0vec, deltaS_w_lan, CLmaxTO
# ----------------------------------------



def thrust_matching(W0_guess, T0_guess, airplane, R_cruise=0):
    """

    Parameters
    ----------
    W0_guess : float
        MTOW estimation [N].
    T0_guess : float
        Thrust estimation [N].
    airplane : dict
        Aircraft to be analyzed.
    R_cruise : float, optional
        Cruise range [m]. The default is 0.

    Returns
    -------
    None.
    
    Version Control
    -------
   (Version/Date/Author/Modification)
   
   > Version 01 - 23/02/2026 - Rian - Headder adition 
   > Version 02 - 24/02/2026 - Rian - variables output correction
    """    

    # Set iterator
    delta = 100
    
    while abs(delta) > 10:
        w = min(1 + abs(delta)/100, 1.2)
        W0, W_empty, W_fuel, W_cruise, W_fuel_mission = weight(W0_guess, T0_guess, airplane, R_cruise)
        T0, T0vec, deltaS_wlan, CLmaxTO = performance(W0, W_cruise, airplane)

        # Compute change with respect to previous iteration
        delta = T0 - T0_guess

        # Update guesses for the next iteration
        w = 1
        T0_guess = (1-w)*T0_guess + w*T0
        W0_guess = (1-w)*W0_guess + w*W0


    #Empty Weight Dictionary
    _, empty_weight_Dict, CG_dict = empty_weight(W0, T0, airplane)
     
    # Update dictionary
    airplane["W0"] = W0
    airplane["W_empty"] = W_empty
    airplane["W_cruise"] = W_cruise
    airplane["W_fuel"] = W_fuel
    airplane["W_fuel_mission"] = W_fuel_mission
    airplane["empty_weight_Dict"] = empty_weight_Dict
    airplane["CG_Dict"] = CG_dict
    
    airplane["T0"] = T0
    airplane["T0vec"] = T0vec
    
    airplane["deltaS_wlan"] = deltaS_wlan
    airplane["CLmaxTO"] = CLmaxTO
   
    
    # Return
    return None


# ----------------------------------------


def balance(airplane):
    """
    Parameters
    ----------
    airplane : dict
        Aircrfat to be analyzed.

    Returns
    -------
    None.
    
    Description
    ---------
    This function calculates the static margin.
    
    Version Control
    -----------
    (Version/Date/Author/Modification)
    
    > Version 01 - 24/02/2026 - Rian - Headder adition 
    

    """
    # Unpack dictionary
    W0 = airplane["W0"]
    W_payload = airplane["W_payload"]
    xcg_payload = airplane["xcg_payload"]
    W_crew = airplane["W_crew"]
    xcg_crew = airplane["xcg_crew"]
    W_empty = airplane["W_empty"]
    xcg_empty = airplane['empty_weight']['xcg_empty']
    W_fuel = airplane["W_fuel"]

    Mach_cruise = airplane["Mach_cruise"]

    S_w = airplane["S_w"]
    AR_eff = airplane["AR_eff"]
    taper_w = airplane["taper_w"]
    sweep_w = airplane["sweep_w"]
    b_w = airplane["b_w"]
    xr_w = airplane["xr_w"]
    zr_w = airplane["zr_w"]
    cr_w = airplane["cr_w"]
    ct_w = airplane["ct_w"]
    xm_w = airplane["xm_w"]
    cm_w = airplane["cm_w"]
    tcr_w = airplane["tcr_w"]
    tct_w = airplane["tct_w"]

    S_h = airplane["S_h"]
    AR_h = airplane["AR_h"]
    sweep_h = airplane["sweep_h"]
    b_h = airplane["b_h"]
    cr_h = airplane["cr_h"]
    ct_h = airplane["ct_h"]
    xm_h = airplane["xm_h"]
    zm_h = airplane["zm_h"]
    cm_h = airplane["cm_h"]
    eta_h = airplane["eta_h"]
    Lc_h = airplane["Lc_h"]

    Cvt = airplane["Cvt"]

    L_f = airplane["L_f"]
    D_f = airplane["D_f"]

    y_n = airplane["y_n"]

    T0 = airplane["T0"]
    n_engines = airplane["n_engines"]

    c_tank_c_w = airplane["c_tank_c_w"]
    x_tank_c_w = airplane["x_tank_c_w"]
    b_tank_b_w_start = airplane["b_tank_b_w_start"]
    b_tank_b_w_end = airplane["b_tank_b_w_end"]

    rho_fuel = airplane["rho_fuel"]

    CLmaxTO = airplane["CLmaxTO"]

    # 3.9.3 Fuel tank center of gravity
    V_maxfuel, W_maxfuel, xcg_fuel, ycg_fuel = tank_properties(
        cr_w,
        ct_w,
        tcr_w,
        tct_w,
        b_w,
        sweep_w,
        xr_w,
        x_tank_c_w,
        c_tank_c_w,
        b_tank_b_w_start,
        b_tank_b_w_end,
        rho_fuel,
        gravity)
    tank_excess = W_maxfuel/W_fuel-1

    # 3.9.4 Center of gravity variation - CG foward and after
    xcg1 = xcg_empty
    xcg2 = (W_empty*xcg_empty + W_crew*xcg_crew)/(W_empty+W_crew)
    xcg3 = (W_empty*xcg_empty + W_payload*xcg_payload +
            W_crew*xcg_crew)/(W_empty+W_payload+W_crew)
    xcg4 = (W_empty*xcg_empty + W_fuel*xcg_fuel +
            W_crew*xcg_crew)/(W_empty+W_fuel+W_crew)
    xcg5 = (W_empty*xcg_empty + W_fuel*xcg_fuel +
            W_payload*xcg_payload+W_crew*xcg_crew)/W0

    xcg_fwd = min(xcg1, xcg2, xcg3, xcg4, xcg5) - 0.02*cm_w
    xcg_aft = max(xcg1, xcg2, xcg3, xcg4, xcg5) + 0.02*cm_w
    # 3.9.5 Neutral Point
    sweep_maxt_w = geo_change_sweep(0.25, 0.40, sweep_w, b_w/2, cr_w, ct_w)
    beta2 = 1-Mach_cruise**2

    CLa_w = (2*np.pi*AR_eff)/(2+np.sqrt(4+AR_eff**2*beta2 /
                                        0.95**2*(1+np.tan(sweep_maxt_w)**2/beta2)))
    CLa_w_M0 = (2*np.pi*AR_eff)/(2+np.sqrt(4+AR_eff**2 *
                                           1/0.95**2*(1+np.tan(sweep_maxt_w)**2)))

    xac_w = xm_w + cm_w/4

    K_f = 0.1462*np.exp(4.8753*(xr_w+0.25*cr_w)/L_f)
    CMa_f = K_f*D_f**2*L_f/(cm_w*S_w)

    CLa_wf = 0.98*CLa_w
    xac_wf = xac_w - CMa_f/CLa_wf*cm_w

    sweep_maxt_h = geo_change_sweep(0.25, 0.40, sweep_h, b_h/2, cr_h, ct_h)
    CLa_h = 0.98*(2*np.pi*AR_h)/(2+np.sqrt(4+AR_h**2*beta2 /
                                           0.95**2*(1+np.tan(sweep_maxt_h)**2/beta2)))
    xac_h = xm_h + cm_h/4

    L_h = Lc_h*cm_w

    ka = 1/AR_eff - 1/(1+AR_eff**1.7)
    kl = (10-3*taper_w)/7
    kh = (1-np.abs((zm_h-zr_w)/b_w))/(2*L_h/b_w)**(1/3)
    deda = 4.44*(ka*kl*kh*np.sqrt(np.cos(sweep_w)))**1.19*CLa_w/CLa_w_M0

    xnp = (CLa_wf*xac_wf+eta_h*S_h/S_w*CLa_h*(1-deda)*xac_h) / \
        (CLa_wf+eta_h*S_h/S_w*CLa_h*(1-deda))

    # 3.9.6 Static Margin
    SM_fwd = (xnp-xcg_fwd)/cm_w
    SM_aft = (xnp-xcg_aft)/cm_w

    # 3.9.7 Vertical tail lift for OEI takeoff condition

    ks = 1.2/1.1
    CLv = y_n/b_w*CLmaxTO/ks**2*T0/(W0*n_engines*Cvt)

    # Update dictionary
    airplane["CG_Dict"]["xcg_fuel"] = xcg_fuel
    airplane["CG_Dict"]["xcg_fwd"] = xcg_fwd
    airplane["CG_Dict"]["xcg_aft"] = xcg_aft
    airplane["Stability"]["xnp"] = xnp
    airplane["Stability"]["SM_fwd"] = SM_fwd
    airplane["Stability"]["SM_aft"] = SM_aft
    airplane["Stability"]["CLv"] = CLv
    airplane["tank_excess"] = tank_excess
    airplane["V_maxfuel"] = V_maxfuel
    
    return None


# ----------------------------------------


def landing_gear(airplane):
    """
    Parameters
    ----------
    airplane : dict
        Aircraft to be analyzed.

    Returns
    -------
    None.
    
    Description
    -----------
    This function calculates the weight fraction
    distribution in each landing gear and the main
    angles of tipback and tail strike.
    
    Version Control
    -------
    (Version/Date/Author/Modification)
    
    > Version 01 - 24/02/2026 - Rian - Headder adition 

    """
    
    
    # Unpack dictionary
    x_nlg = airplane["x_nlg"]
    x_mlg = airplane["x_mlg"]
    y_mlg = airplane["y_mlg"]
    z_lg = airplane["z_lg"]
    xcg_fwd = airplane["xcg_fwd"]
    xcg_aft = airplane["xcg_aft"]
    x_tailstrike = airplane["x_tailstrike"]
    z_tailstrike = airplane["z_tailstrike"]

    # 3.10.3 Nose landing gear weight fraction
    frac_nlg_fwd = (x_mlg - xcg_fwd)/(x_mlg-x_nlg)

    frac_nlg_aft = (x_mlg - xcg_aft)/(x_mlg-x_nlg)

    # 3.10.4 Tipback angle
    alpha_tipback = np.arctan((xcg_aft-x_mlg)/z_lg)

    # 3.10.5 Tailstrike angle
    alpha_tailstrike = np.arctan((z_tailstrike-z_lg)/(x_tailstrike-x_mlg))

    # 3.10.6 Overturn angle
    SGL = (xcg_fwd - x_nlg)*y_mlg/np.sqrt((x_mlg-x_nlg)**2+y_mlg**2)
    phi_overturn = np.arctan(-z_lg/SGL)

    # Update dictionary
    airplane["frac_nlg_fwd"] = frac_nlg_fwd
    airplane["frac_nlg_aft"] = frac_nlg_aft
    airplane["alpha_tipback"] = alpha_tipback
    airplane["alpha_tailstrike"] = alpha_tailstrike
    airplane["phi_overturn"] = phi_overturn

    return None


# ----------------------------------------

# ----------------------------------------
def doc(airplane,plot=False, print_log=False):
    
    R_cruise = airplane['block_range']-(200*nm2m)
    _, _, _ , _ , W_fuel_mission = fuel_weight(airplane["W0"], airplane, range_cruise=R_cruise, altcruise= True, loiter= True)
    
    aircraft_params = ct.AircraftParameters(
    # Utilization
    block_time_hours=airplane['block_time']/3600,                    # Average block time per flight
    flight_time_hours=airplane['block_time']/3600 - 20/60 - 8/60,          # Flight time, 20 min taxi out 8 min taxi in
    flights_per_year=None, # THIS DOESNT MATTER FOR COC
    
    # Weights    
    maximum_takeoff_weight_kg=airplane['W0'] / gravity,        # MTOW
    operational_empty_weight_kg=airplane['W_empty'] / gravity,      # OEW
    engine_weight_kg=airplane['engine']['W_eng_spec'] / gravity,
    fuel_weight_kg=W_fuel_mission / gravity,                    # Trip fuel
    payload_weight_kg=airplane['W_payload'] / gravity,                # Typical payload for 80% load factor
    
    # Mission
    range_nm=airplane['block_range'] / nm2m,                          # Stage length
    
    # Engine specifications
    engine_count=airplane['n_engines'],
    bypass_ratio=airplane['engine']['BPR'],
    overall_pressure_ratio=airplane['engine']['OAPR'],
    compressor_stages=airplane['engine']['Compressor_stages'],
    engine_shafts=airplane['engine']['Engine_shafts'],
    takeoff_thrust_per_engine_N=airplane['engine']['T_eng_spec'],
    
    # Crew
    cockpit_crew_count=2,
    cabin_crew_count=1,
    
    # Pricing (optional - will be estimated if not provided)
    aircraft_delivery_price_usd=None,        # Will be estimated from OEW
    engine_price_usd=None                    # Will be estimated from thrust
    )

    fitted_maintenance_params = ct.FITTED_MAINTENANCE_PARAMS

    params = ct.MethodParameters(maintenance=fitted_maintenance_params)

    coc_results = ct.calculate_costs(aircraft_params, params, target_year=2025, verbose=print_log)

    COCcrew = coc_results.per_flight.crew / aircraft_params.range_nm
    COCfuel = coc_results.per_flight.fuel / aircraft_params.range_nm
    COCmnt = coc_results.per_flight.maintenance / aircraft_params.range_nm
    COCfee = coc_results.per_flight.fees_and_charges / aircraft_params.range_nm
    COC = COCcrew + COCfuel + COCmnt + COCfee
    
    COC_breakdown = {
                     'COC': COC,
                     'crew': COCcrew,
                     'fuel': COCfuel,
                     'maintenance': COCmnt,
                     'fees': COCfee,
                     }
    
    DOC = COC + (
        coc_results.per_flight.depreciation / aircraft_params.range_nm
        + coc_results.per_flight.interest / aircraft_params.range_nm
        + coc_results.per_flight.insurance / aircraft_params.range_nm
    )
 
    # Generate dictionary with results
    DOC_breakdown = {'DOC': DOC,
                     'crew': COCcrew,
                     'fuel': COCfuel,
                     'maintenance': COCmnt,
                     'fees': COCfee,
                     'depreciation': coc_results.per_flight.depreciation / aircraft_params.range_nm,
                     'interest': coc_results.per_flight.interest / aircraft_params.range_nm,
                     'insurance': coc_results.per_flight.insurance / aircraft_params.range_nm
                     }
 
    # Update dictionary
    # airplane['DOC'] = DOC
    # airplane['DOC_breakdown'] = DOC_breakdown
    # airplane['COC_breakdown'] = COC_breakdown
    # airplane['COC'] = COC


    # -----------------------------------------------------
    # PLOT - COC
    # -----------------------------------------------------
    if plot:

        # Paleta
        cores = np.array([
            [239/255, 163/255,  11/255],   # laranja
            [0/255,    70/255, 171/255],   # azul médio
            [0/255,    25/255, 114/255],   # azul escuro
            [0.5,      0.5,     0.5],      # cinza
            [255/255, 200/255,   0/255],   # amarelo
            [135/255, 221/255, 241/255],   # azul claro
        ])

        # Componentes do COC 
        coc_labels = ["Fuel", "Crew", "Maintenance", "Fees and Charges"]
        coc_values = np.array([COCfuel, COCcrew, COCmnt, COCfee], dtype=float)

        coc_values = np.nan_to_num(coc_values, nan=0.0, posinf=0.0, neginf=0.0)
        coc_values[coc_values < 0] = 0.0

        total = coc_values.sum()
        unidade = "US$/NM"

        def autopct_format(values):
            vals = np.array(values, dtype=float)
            tot = vals.sum()

            def _fmt(pct):
                val = pct * tot / 100.0
                if pct < 3:
                    return ""
                return f"{pct:.1f}%\n{val:.2f}"
            return _fmt

        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

        wedgeprops = dict(width=0.45, edgecolor="white", linewidth=1.2)

        wedges, texts, autotexts = ax.pie(
            coc_values,
            labels=None, 
            autopct=autopct_format(coc_values),
            startangle=90,
            counterclock=False,
            colors=cores[:len(coc_values)],
            pctdistance=0.78,
            wedgeprops=wedgeprops,
            textprops=dict(color="black", fontsize=9, fontweight="semibold"),
        )

        centre_circle = plt.Circle((0, 0), 0.55, fc="white")
        ax.add_artist(centre_circle)

        ax.text(
            0, 0,
            f"COC Total\n{total:.2f} {unidade}",
            ha="center", va="center",
            fontsize=12, fontweight="bold"
        )

        ax.legend(
            wedges, coc_labels,
            title=" ",
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=False,
            fontsize=10,
            title_fontsize=11
        )

        ax.axis("equal")

        plt.tight_layout()
        plt.show()

    return DOC_breakdown, COC_breakdown

def Cbase_matching(airplane):

    W_max_usable_fuel = airplane["W_max_usable_fuel"]
 
    geometry(airplane)
 
    Cbase_vec = np.arange(0.5/3600, 1/3600, 0.001/3600)
 
    W_fuel_vec = np.zeros(len(Cbase_vec))

    i = 0

    #W0 = airplane["MTOW"]

    W0 = airplane["W_empty"] + airplane["W_crew"] + airplane["W_payload"] + W_max_usable_fuel
 
 
    for idx, Cbase in enumerate(Cbase_vec):

        airplane["engine"]["Cbase"] = Cbase

        W_fuel, W_cruise, Mf_vec, Mf, W_fuel_mission = fuel_weight(W0, airplane)

        W_fuel_vec[idx] = W_fuel

        error = abs(W_fuel_vec[idx] - W_max_usable_fuel)/W_max_usable_fuel

        if error < 0.01:

            i = idx

            Cbase_final = Cbase_vec[i]

    airplane["engine"]["Cbase"] = Cbase_final


# ----------------------------------------
# ========================================
# AUXILIARY FUNCTIONS


def atmosphere(z, Tba=288.15):
    """
    Parameters
    ----------
    z : float
        Altitude [m].
    Tba : float, optional
        Mean temperature. The default is 288.15.

    Returns
    -------
    None.
    
    Description
    --------
    Funçao que retorna a Temperatura, Pressao e Densidade para uma determinada
    altitude z [m]. Essa funçao usa o modelo padrao de atmosfera para a
    temperatura no solo de Tba.
    
    Version Control
    ----------
    (Version/Date/Author/Modification)
    
    > Version 01 - 24/02/2026 - Rian - Headder adition 
    
    """

    # Zbase (so para referencia)
    # 0 11019.1 20063.1 32161.9 47350.1 50396.4

    # DEFINING CONSTANTS
    # Earth radius
    r = 6356766
    # gravity
    g0 = 9.80665
    # air gas constant
    R = 287.05287
    # layer boundaries
    Ht = [0, 11000, 20000, 32000, 47000, 50000]
    # temperature slope in each layer
    A = [-6.5e-3, 0, 1e-3, 2.8e-3, 0]
    # pressure at the base of each layer
    pb = [101325, 22632, 5474.87, 868.014, 110.906]
    # temperature at the base of each layer
    Tstdb = [288.15, 216.65, 216.65, 228.65, 270.65]
    # temperature correction
    Tb = Tba - Tstdb[0]
    # air viscosity
    mi0 = 18.27e-6  # [Pa s]
    T0 = 291.15  # [K]
    C = 120  # [K]

    # geopotential altitude
    H = r * z / (r + z)

    # selecting layer
    if H < Ht[0]:
        raise ValueError("Under sealevel")
    elif H <= Ht[1]:
        i = 0
    elif H <= Ht[2]:
        i = 1
    elif H <= Ht[3]:
        i = 2
    elif H <= Ht[4]:
        i = 3
    elif H <= Ht[5]:
        i = 4
    else:
        raise ValueError("Altitude beyond model boundaries")

    # Calculating temperature
    T = Tstdb[i] + A[i] * (H - Ht[i]) + Tb

    # Calculating pressure
    if A[i] == 0:
        p = pb[i] * np.exp(-g0 * (H - Ht[i]) / R / (Tstdb[i] + Tb))
    else:
        p = pb[i] * (T / (Tstdb[i] + Tb)) ** (-g0 / A[i] / R)

    # Calculating density
    rho = p / R / T

    # Calculating viscosity with Sutherland's Formula
    mi = mi0 * (T0 + C) / (T + C) * (T / T0) ** (1.5)

    return T, p, rho, mi


# ----------------------------------------


def geo_change_sweep(x, y, sweep_x, panel_length, chord_root, chord_tip):
    """
    This function converts sweep computed at chord fraction x into
    sweep measured at chord fraction y
    (x and y should be between 0 (leading edge) and 1 (trailing edge).
    """

    sweep_y = np.arctan(
        np.tan(sweep_x) + (x - y) * (chord_root - chord_tip) / panel_length
    )

    return sweep_y


# ----------------------------------------


def Cf_calc(Mach, altitude, length, rugosity, k_lam, Tba=288.15):
    """
    This function computes the flat plate friction coefficient
    for a given Reynolds number while taking transition into account

    k_lam: float -> Fraction of the length (from 0 to 1) where
                    transition occurs
    """

    # Dados atmosféricos
    T, p, rho, mi = atmosphere(altitude, Tba)

    # Velocidade
    v = np.sqrt(gamma_air * R_air * T) * Mach

    # Reynolds na transição
    Re_conv = rho * v * k_lam * length / mi
    Re_rug = 38.21 * (k_lam * length / rugosity) ** 1.053
    Re_trans = min(Re_conv, Re_rug)

    # Reynolds no fim
    Re_conv = rho * v * length / mi
    if Mach < 0.7:
        Re_rug = 38.21 * (length / rugosity) ** 1.053
    else:
        Re_rug = 44.62 * (length / rugosity) ** 1.053 * Mach**1.16
    Re_fim = min(Re_conv, Re_rug)

    # Coeficientes de fricção
    # Laminar na transição
    Cf1 = 1.328 / np.sqrt(Re_trans)

    # Turbulento na transição
    Cf2 = 0.455 / (np.log10(Re_trans) ** 2.58 * (1 + 0.144 * Mach**2) ** 0.65)

    # Turbulento no fim
    Cf3 = 0.455 / (np.log10(Re_fim) ** 2.58 * (1 + 0.144 * Mach**2) ** 0.65)

    # Média
    Cf = (Cf1 - Cf2) * k_lam + Cf3

    return Cf


# ----------------------------------------


def FF_surface(Mach, tcr, tct, sweep, b, cr, ct, x_c_max_tc=0.4):
    """
    This function computes the form factor for lifting surfaces

    INPUTS

    tcr: float -> Thickness/chord ratio at the root
    tct: float -> Thickness/chord ratio at the tip
    sweep: float -> Quarter-chord sweep angle [rad]
    b: float -> Wing span (considering both sides. Double this value for vertical tails if necessary)
    cr: float -> Root chord
    ct: float -> Tip chord
    x_c_max_tc: float -> Chord fraction with maximum thickness
    """

    # Average chord fraction
    t_c = 0.25 * tcr + 0.75 * tct

    # Sweep at maximum thickness position
    sweep_maxtc = geo_change_sweep(0.25, x_c_max_tc, sweep, b / 2, cr, ct)

    # Form factor
    FF = (
        1.34
        * Mach**0.18
        * np.cos(sweep_maxtc) ** 0.28
        * (1 + 0.6 * t_c / x_c_max_tc + 100 * (t_c) ** 4)
    )

    return FF


# ----------------------------------------


def tank_properties(
    cr_w,
    ct_w,
    tcr_w,
    tct_w,
    b_w,
    sweep_w,
    xr_w,
    x_tank_c_w,
    c_tank_c_w,
    b_tank_b_w_start,
    b_tank_b_w_end,
    rho_fuel,
    gravity,
):
    """
    This function computes the maximum fuel tank volume and center of gravity.
    We assume that the tank has a prism shape.

    c_tank_c_w: float -> fraction of the chord where tank begins (0-leading edge, 1-trailing edge)
    c_tank_c_w: float -> fraction of the chord occupied by the tank (between 0 and 1)
    bf_w_start: float -> semi-span fraction where tank begins (0-root, 1-tip)
    bf_w_end: float -> semi-span fraction where tank ends (0-root, 1-tip)
    """

    # Compute the local chords where the tank begins and ends
    c_tank_start = cr_w + b_tank_b_w_start * (ct_w - cr_w)
    c_tank_end = cr_w + b_tank_b_w_end * (ct_w - cr_w)

    # Compute the local thickness where the tank begins and ends
    tc_tank_start = tcr_w + b_tank_b_w_start * (tct_w - tcr_w)
    tc_tank_end = tcr_w + b_tank_b_w_end * (tct_w - tcr_w)

    # Compute the prism area where the tank begins.
    # We assume that this face is rectangular, and that its height
    # is 85% of the maximum airfoil thickness (Gudmundsson, page 87).
    ll = c_tank_start * c_tank_c_w
    hh = c_tank_start * tc_tank_start * 0.85
    S1 = ll * hh

    # Compute the prism area where the tank ends.
    ll = c_tank_end * c_tank_c_w
    hh = c_tank_end * tc_tank_end * 0.85
    S2 = ll * hh

    # Compute distance between prism faces along the wing span
    Lprism = 0.5 * b_w * (b_tank_b_w_end - b_tank_b_w_start)

    # Compute fuel volume with the prism expression (Torenbeek Fig B-4, pg 448).
    # We multiply by 2 to take into account both semi-wings.
    # The 0.91 factor is to take into account internal structures and fuel expansion,
    # as suggested by Torenbeek.
    V_maxfuel = 0.91 * 2 * Lprism / 3 * (S1 + S2 + np.sqrt(S1 * S2))

    # Compute corresponding fuel weight
    W_maxfuel = V_maxfuel * rho_fuel * gravity

    # Compute the span-wise distance between the first prism face and its center of gravity
    # using the expression from Jenkinson, Fig 7.13, pg 148.
    Lprism_cg = (
        Lprism / 4 * (S1 + 3 * S2 + 2 * np.sqrt(S1 * S2)) /
        (S1 + S2 + np.sqrt(S1 * S2))
    )

    # Now find the span-wise distance between the tank CG and the aircraft centerline
    ycg_fuel = Lprism_cg + 0.5 * b_w * b_tank_b_w_start

    # Find the sweep angle at the chord position located on the middle of the chord
    # fraction occupied by the fuel tank
    c_pos = x_tank_c_w + 0.5 * c_tank_c_w

    # Sweep at the tank center line
    sweep_tank = geo_change_sweep(0.25, c_pos, sweep_w, b_w / 2, cr_w, ct_w)

    # Longitudinal position of the tank CG
    xcg_fuel = xr_w + cr_w * c_pos + ycg_fuel * np.tan(sweep_tank)

    return V_maxfuel, W_maxfuel, xcg_fuel, ycg_fuel


# ----------------------------------------


def lin_interp(x0, x1, y0, y1, x):
    """
    Linear interpolation function
    """

    y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return y


# ----------------------------------------

# ----------------------------------------


def flap_area_fraction(alpha, beta1, beta2, taper, flap_type):
    """
    alpha: flap_chord/wing_chord
    beta1: spanwise fraction where flap starts
    beta2: spanwise fraction where flap ends
    taper: taper ratio of the wing
    m: multiplier to take into account more complex devices (such as slotted flaps)
    """

    if flap_type == "plain":
        m = 1
    elif flap_type == "single_slotted":
        m = 1.15 * 1.25
    elif flap_type == "double_slotted":
        m = 1.30 * 1.25
    elif flap_type == "triple_slotted":
        m = 1.45 * 1.25

    S_flap_S_wing = (
        alpha
        / (1 + taper)
        * (beta2 * (2 - beta2 * (1 - taper)) - beta1 * (2 - beta1 * (1 - taper)))
        * m
    )

    return S_flap_S_wing


def slat_area_fraction(alpha, beta1, beta2, taper, slat_type):
    """
    alpha: slat_chord/wing_chord
    beta1: spanwise fraction where slat starts
    beta2: spanwise fraction where slat ends
    taper: taper ratio of the wing
    m: multiplier to take into account more complex devices (such as slotted flaps)
    """
    if slat_type == "fixed_slot" or slat_type == "none":
        m = 0
    elif slat_type == "leadinf_edge_flaps":
        m = 1
    elif slat_type == "Kruger_flaps":
        m = 1
    elif slat_type == "slats":
        m = 1.25

    S_slat_S_wing = (
        alpha
        / (1 + taper)
        * (beta2 * (2 - beta2 * (1 - taper)) - beta1 * (2 - beta1 * (1 - taper)))
        * m
    )

    return S_slat_S_wing


def aileron_area_fraction(alpha, beta1, beta2, taper):
    """
    alpha: ail_chord/wing_chord
    beta1: spanwise fraction where aileron starts
    beta2: spanwise fraction where aileron ends
    taper: taper ratio of the wing
    m: multiplier to take into account more complex devices (such as slotted flaps)
    """

    m = 1
    S_ail_S_wing = (
        alpha
        / (1 + taper)
        * (beta2 * (2 - beta2 * (1 - taper)) - beta1 * (2 - beta1 * (1 - taper)))
        * m
    )

    return S_ail_S_wing


# ----------------------------------------


def standard_airplane(name="ERJ145-ER"):
        
    if name == "ERJ145-ER":
        airplane = {
            'name': name,
            "type": "transport",            # Can be 'transport', 'fighter', or 'general'

            # ------------------------------------------------------------------
            # Wing
            "S_w": 51.182,                  # Wing area [m2]
            "AR_w": 7.847,                  # Wing aspect ratio
            "taper_w": 0.255,               # Wing taper ratio
            "sweep_w": 0.393898,            # Wing sweep (at c/4) [rad]
            "dihedral_w": 0.050456141,      # Wing dihedral [rad]
            # Longitudinal position of the wing (with respect to the fuselage nose) [m]
            "xr_w": 12.649,
            # Vertical position of the wing (with respect to the fuselage nose) [m]
            "zr_w": -1.141,
            "tcr_w": 0.14,           # t/c of the root section of the wing
            "tct_w": 0.10,           # t/c of the tip section of the wing
            # ------------------------------------------------------------------
            # Horizontal tail
			"has_HT": True,
            "Cht": 0.979,                   # Horizontal tail volume coefficient
            # Non-dimensional lever of the horizontal tail (lever/wing_mac)
            "Lc_h": 4.476,
            "AR_h": 5.089,                  # HT aspect ratio
            "taper_h": 0.605,               # HT taper ratio
            "sweep_h": 0.31384251,          # HT sweep [rad]
            "dihedral_h": 0,                # HT dihedral [rad]
            # Vertical position of the HT em relação ao centro da fuselagem [m]
            "zr_h": 3.772,
            "tcr_h": 0.097890752,           # t/c of the root section of the HT
            "tct_h": 0.093023256,           # t/c of the tip section of the HT
            "eta_h": 1.0,                   # Dynamic pressure factor of the HT
            # ------------------------------------------------------------------
            # Vertical tail
            "Cvt": 0.091,                   # Vertical tail volume coefficient
            # Non-dimensional lever of the vertical tail (lever/wing_span)
            "Lb_v": 0.538,
            "AR_v": 1.254,                  # VT aspect ratio
            "taper_v": 0.578,               # VT taper ratio
            "sweep_v": 0.473175,            # VT sweep [rad]
            "zr_v": 0.608,                  # Vertical position of the VT [m]
            "tcr_v": 0.122601918,           # t/c of the root section of the VT
            "tct_v": 0.136099955,           # t/c of the tip section of the VT
            # ----------------------------------------------------------------------
            # Canard
            "has_canard": False,
            # Canard volume coefficient
            "Ccan": 0.48,
            # Lever arm non-dimensional (lever / wing_mac)
            "Lc_c": 2.95,
            "AR_c": 9.43,                   # Canard aspect ratio
            "taper_c": 0.6,                 # Canard taper ratio
            "sweep_c": 0.488692,            # Canard sweep [rad]
            "dihedral_c": 0.0,              # Canard dihedral [rad]

            # Vertical position of canard root relative to fuselage reference [m]
            "zr_c": 1.15,
            "tcr_c": 0.1,                   # t/c at canard root
            "tct_c": 0.09,               # t/c at canard tip
            # ------------------------------------------------------------------
            # Fuselage
            "L_f": 27.93,                   # Fuselage length [m]
            "D_f": 2.28,  # Fuselage diameter [m]
            # ------------------------------------------------------------------
            # Nacelle
            # Longitudinal position of the nacelle frontal face [m]
            "x_n": 20.478,
            # Lateral position of the nacelle centerline [m]
            "y_n": 2.107,
            # Vertical position of the nacelle centerline [m]
            "z_n": 0.555,

            # ------------------------------------------------------------------
            # Engine
            # Model: AE 3007A1

            "n_engines": 2,                 # Number of engines
            "n_engines_under_wing": 0,      # Number of engines installed under the wing
            "engine": {
                "model": "Howe turbofan",   # Check engineTSFC function for options
                "BPR": 5,                   # Engine bypass ratio
                "OAPR": 20,
                "Compressor_stages": 9,
                "Engine_shafts": 2,
                #  I adjusted this value by hand to match the fuel weight
                "Cbase": 0.785/3600,
                "T_eng_spec": 33032.5,     # Thurst of 1 engine [N]
                "W_eng_spec": 746*9.81,  # Engine Weight [N]
            },

            "L_n": 4.313,                   # Nacelle length [m]
            "D_n": 1.309,                   # Nacelle diameter [m]
            # ------------------------------------------------------------------
            # Landing gear
            # Longitudinal position of the nose landing gear [m]
            "x_nlg": 2.185,
            # Longitudinal position of the main landing gear [m]
            "x_mlg": 16.625,
            # Lateral position of the main landing gear [m]
            "y_mlg": 2.059,
            # Vertical position of the landing gear [m]
            "z_lg": -1.964,
            # Longitudinal position of critical tailstrike point [m]
            "x_tailstrike": 24.348,
            # Vertical position of critical tailstrike point [m]
            "z_tailstrike": -0.941,
            # ------------------------------------------------------------------
            # Tank [Ta faltando tudo isso]
            "c_tank_c_w": 0.6,             # Fraction of the wing chord occupied by the fuel tank
            "x_tank_c_w": 0.2,              # Fraction of the wing chord where fuel tank starts
            "b_tank_b_w_start": 0.0,        # Fraction of the wing semi-span where fuel tank starts
            "b_tank_b_w_end": 0.425,         # Fraction of the wing semi-span where fuel tank ends
            # ------------------------------------------------------------------
            # Airfoil
            #  Maximum lift coefficient of wing airfoil
            "clmax_w": 1.62, #1.62
            #  Airfoil technology factor for Korn equation (wave drag)
            "k_korn": 0.91,
            # ------------------------------------------------------------------
            # High-Lift devices
            "flap_type": "double slotted",  # Flap type
            "c_flap_c_wing": 0.22,#0.1579,        # Fraction of the wing chord occupied by flaps
            # Fraction of the wing span occupied by flaps (including fuselage portion)
            "b_flap_b_wing": 0.7182,
            "slat_type": None,              # Slat type
            "c_slat_c_wing": 0.00,          # Fraction of the wing chord occupied by slats
            "b_slat_b_wing": 0.00,          # Fraction of the wing span occupied by slats
            "c_ail_c_wing": 0.2428,         # Fraction of the wing chord occupied by aileron
            "b_ail_b_wing": 0.2418,         # Fraction of the wing span occupied by aileron
            # Distance to the ground for ground effect computation [m]
            "h_ground": 35.0 * ft2m,
            "k_exc_drag": 0.218,             #  Excrescence drag factor
            "winglet": False,               # Add winglet
            # Até aqui
            # ------------------------------------------------------------------
            # Flight conditions
            # Altitude for takeoff computation [m]
            "altitude_takeoff": 0.0,
            "distance_takeoff": 1970.0,     # 1970 Required takeoff distance [m]
            # Variation from ISA standard temperature [ C] - From Obert's paper
            'deltaISA_takeoff': 0.0,
            'deltaISA_landing': 0.0,
            'MLW_frac': 18700/20600,

            # Altitude for landing computation [m]
            "altitude_landing": 0.0,
            "distance_landing": 1350.0,     # Required landing distance [m]
            # Cruise altitude [m] (From 01_aircraft_survey.xlsx)
            "altitude_cruise": 35000 * ft2m,
            # Cruise Mach number (From 01_aircraft_survey.xlsx)
            "Mach_cruise": 0.75,
            "range_cruise": (1150 - 200) * nm2m,    # Cruise range [m]
            "loiter_time": 45 * 60,         # Loiter time [s]

            #  Ceiling altitude [m]
            "altitude_ceiling": 37000 * ft2m, #4572
            #  Ceiling Mach number
            "Mach_ceiling": 0.7,

            #  Alternative cruise altitude [m]
            "altitude_altcruise": 4600, #4572
            #  Alternative cruise Mach number
            "Mach_altcruise": 0.4,
            # Alternative cruise range [m]
            "range_altcruise": 100 * nm2m,
            # ------------------------------------------------------------------
            # Payload
            "W_payload": 4600 * gravity,    # Payload weight [N]
            #  Longitudinal position of the Payload center of gravity [m]
            "xcg_payload": 13.5,
            # [Julia - duas fileiras so tem um assento] Number of rows
            "N_rows": 18,
            # ------------------------------------------------------------------
            # Crew
            "W_crew": 1 * 90 * gravity,     # Crew weight [N]
            #  Longitudinal position of the Crew center of gravity [m]
            "xcg_crew": 9.677,

            
			"block_range": 600 * nm2m,      #  Block range [m]
            "block_time": 108*60,         #(1.0 + 2 * 40 / 60) * 3600,  #  Block time [s]
            "n_captains": 1,  # Number of captains in flight
            "n_copilots": 1,  # Number of copilots in flight
            "rho_fuel": 811,  # Fuel density kg/m3 (This is Jet A-1)
            # ------------------------------------------------------------------
            # All else
            #  Percentage of fuselage legnth for the Xcg all else
            "perc_cg_ae": 0.45,
            # ------------------------------------------------------------------
            # Mtow
            # Guess for MTOW (From 01_aircraft_survey.xlsx)
            "W0_guess": 20600 * gravity,
        }

    if name == "ERJ145-ER-bw":
        airplane = {
            'name': name,
            "type": "transport",            # Can be 'transport', 'fighter', or 'general'

            # ------------------------------------------------------------------
            # Wing
            "S_w": 51.182,                  # Wing area [m2]
            "AR_w": 7.847,                  # Wing aspect ratio
            "taper_w": 0.255,               # Wing taper ratio
            "sweep_w": 0.393898,            # Wing sweep (at c/4) [rad]
            "dihedral_w": 0.050456141,      # Wing dihedral [rad]
            # Longitudinal position of the wing (with respect to the fuselage nose) [m]
            "xr_w": 12.649,
            # Vertical position of the wing (with respect to the fuselage nose) [m]
            "zr_w": -1.141,
            "tcr_w": 0.14,           # t/c of the root section of the wing
            "tct_w": 0.10,           # t/c of the tip section of the wing
            # ------------------------------------------------------------------
            # Horizontal tail
			"has_HT": False,
            "Cht": 0.979,                   # Horizontal tail volume coefficient
            # Non-dimensional lever of the horizontal tail (lever/wing_mac)
            "Lc_h": 4.476,
            "AR_h": 5.089,                  # HT aspect ratio
            "taper_h": 0.605,               # HT taper ratio
            "sweep_h": 0.31384251,          # HT sweep [rad]
            "dihedral_h": 0,                # HT dihedral [rad]
            # Vertical position of the HT em relação ao centro da fuselagem [m]
            "zr_h": 3.772,
            "tcr_h": 0.097890752,           # t/c of the root section of the HT
            "tct_h": 0.093023256,           # t/c of the tip section of the HT
            "eta_h": 1.0,                   # Dynamic pressure factor of the HT
            # ------------------------------------------------------------------
            # Vertical tail
            "Cvt": 0.091,                   # Vertical tail volume coefficient
            # Non-dimensional lever of the vertical tail (lever/wing_span)
            "Lb_v": 0.538,
            "AR_v": 1.254,                  # VT aspect ratio
            "taper_v": 0.578,               # VT taper ratio
            "sweep_v": 0.473175,            # VT sweep [rad]
            "zr_v": 0.608,                  # Vertical position of the VT [m]
            "tcr_v": 0.122601918,           # t/c of the root section of the VT
            "tct_v": 0.136099955,           # t/c of the tip section of the VT
            # ----------------------------------------------------------------------
            # Canard
            "has_canard": False,
            # Canard volume coefficient
            "Ccan": 0.48,
            # Lever arm non-dimensional (lever / wing_mac)
            "Lc_c": 2.95,
            "AR_c": 9.43,                   # Canard aspect ratio
            "taper_c": 0.6,                 # Canard taper ratio
            "sweep_c": 0.488692,            # Canard sweep [rad]
            "dihedral_c": 0.0,              # Canard dihedral [rad]

            # Vertical position of canard root relative to fuselage reference [m]
            "zr_c": 1.15,
            "tcr_c": 0.1,                   # t/c at canard root
            "tct_c": 0.09,               # t/c at canard tip

            # ----------------------------------------------------------------------
            # Box wing
            "box_wing": True,

            # Asa dianteira (front wing) - mesmos inputs de uma asa normal
            "S_wf": 51.182,
            "AR_wf": 7.847,
            "taper_wf": 0.255,
            "sweep_wf": 0.393898,      # [rad]
            "dihedral_wf": 0.0504561,   # [rad]
            "xr_wf": 12.649,         # x raiz
            "zr_wf": -1.141,         # z raiz
            "tcr_wf": 0.14,       # t/c na raiz
            "tct_wf": 0.10,       # t/c na ponta

            # Asa traseira (rear wing) - mesmos inputs de uma asa normal
            "S_wr": 51.182,
            "AR_wr": 7.847,
            "taper_wr": 0.255,
            "sweep_wr": -0.393898,      # [rad]
            "dihedral_wr": 0.0504561,   # [rad]
            "xr_wr": 25.649,         # x raiz
            "zr_wr": 3.772,         # z raiz
            "tcr_wr": 0.14,       # t/c na raiz
            "tct_wr": 0.10,       # t/c na ponta

            # (Opcional) “espessura extra”/fator visual do painel de junção
            # Se quiser, pode deixar 1.0 e pronto.
            "box_join_thickness_factor": 1.0,
            # ------------------------------------------------------------------
            # Fuselage
            "L_f": 27.93,                   # Fuselage length [m]
            "D_f": 2.28,  # Fuselage diameter [m]
            # ------------------------------------------------------------------
            # Nacelle
            # Longitudinal position of the nacelle frontal face [m]
            "x_n": 20.478,
            # Lateral position of the nacelle centerline [m]
            "y_n": 2.107,
            # Vertical position of the nacelle centerline [m]
            "z_n": 0.555,

            # ------------------------------------------------------------------
            # Engine
            # Model: AE 3007A1

            "n_engines": 2,                 # Number of engines
            "n_engines_under_wing": 0,      # Number of engines installed under the wing
            "engine": {
                "model": "Howe turbofan",   # Check engineTSFC function for options
                "BPR": 5,                   # Engine bypass ratio
                #  I adjusted this value by hand to match the fuel weight
                "Cbase": 0.785/3600,
                "T_eng_spec": 33032.5,     # Thurst of 1 engine [N]
                "W_eng_spec": 746*9.81,  # Engine Weight [N]
            },

            "L_n": 4.313,                   # Nacelle length [m]
            "D_n": 1.309,                   # Nacelle diameter [m]
            # ------------------------------------------------------------------
            # Landing gear
            # Longitudinal position of the nose landing gear [m]
            "x_nlg": 2.185,
            # Longitudinal position of the main landing gear [m]
            "x_mlg": 16.625,
            # Lateral position of the main landing gear [m]
            "y_mlg": 2.059,
            # Vertical position of the landing gear [m]
            "z_lg": -1.964,
            # Longitudinal position of critical tailstrike point [m]
            "x_tailstrike": 24.348,
            # Vertical position of critical tailstrike point [m]
            "z_tailstrike": -0.941,
            # ------------------------------------------------------------------
            # Tank [Ta faltando tudo isso]
            "c_tank_c_w": 0.6,             # Fraction of the wing chord occupied by the fuel tank
            "x_tank_c_w": 0.2,              # Fraction of the wing chord where fuel tank starts
            "b_tank_b_w_start": 0.0,        # Fraction of the wing semi-span where fuel tank starts
            "b_tank_b_w_end": 0.425,         # Fraction of the wing semi-span where fuel tank ends
            # ------------------------------------------------------------------
            # Airfoil
            #  Maximum lift coefficient of wing airfoil
            "clmax_w": 1.62, #1.62
            #  Airfoil technology factor for Korn equation (wave drag)
            "k_korn": 0.91,
            # ------------------------------------------------------------------
            # High-Lift devices
            "flap_type": "double slotted",  # Flap type
            "c_flap_c_wing": 0.22,#0.1579,        # Fraction of the wing chord occupied by flaps
            # Fraction of the wing span occupied by flaps (including fuselage portion)
            "b_flap_b_wing": 0.7182,
            "slat_type": None,              # Slat type
            "c_slat_c_wing": 0.00,          # Fraction of the wing chord occupied by slats
            "b_slat_b_wing": 0.00,          # Fraction of the wing span occupied by slats
            "c_ail_c_wing": 0.2428,         # Fraction of the wing chord occupied by aileron
            "b_ail_b_wing": 0.2418,         # Fraction of the wing span occupied by aileron
            # Distance to the ground for ground effect computation [m]
            "h_ground": 35.0 * ft2m,
            "k_exc_drag": 0.218,             #  Excrescence drag factor
            "winglet": False,               # Add winglet
            # Até aqui
            # ------------------------------------------------------------------
            # Flight conditions
            # Altitude for takeoff computation [m]
            "altitude_takeoff": 0.0,
            "distance_takeoff": 1970.0,     # 1970 Required takeoff distance [m]
            # Variation from ISA standard temperature [ C] - From Obert's paper
            'deltaISA_takeoff': 0.0,
            'deltaISA_landing': 0.0,
            'MLW_frac': 18700/20600,

            # Altitude for landing computation [m]
            "altitude_landing": 0.0,
            "distance_landing": 1350.0,     # Required landing distance [m]
            # Cruise altitude [m] (From 01_aircraft_survey.xlsx)
            "altitude_cruise": 35000 * ft2m,
            # Cruise Mach number (From 01_aircraft_survey.xlsx)
            "Mach_cruise": 0.75,
            "range_cruise": (1150 - 200) * nm2m,    # Cruise range [m]
            "loiter_time": 45 * 60,         # Loiter time [s]

            #  Ceiling altitude [m]
            "altitude_ceiling": 37000 * ft2m, #4572
            #  Ceiling Mach number
            "Mach_ceiling": 0.7,

            #  Alternative cruise altitude [m]
            "altitude_altcruise": 4600, #4572
            #  Alternative cruise Mach number
            "Mach_altcruise": 0.4,
            # Alternative cruise range [m]
            "range_altcruise": 100 * nm2m,
            # ------------------------------------------------------------------
            # Payload
            "W_payload": 4600 * gravity,    # Payload weight [N]
            #  Longitudinal position of the Payload center of gravity [m]
            "xcg_payload": 13.5,
            # [Julia - duas fileiras so tem um assento] Number of rows
            "N_rows": 18,
            # ------------------------------------------------------------------
            # Crew
            "W_crew": 1 * 90 * gravity,     # Crew weight [N]
            #  Longitudinal position of the Crew center of gravity [m]
            "xcg_crew": 9.677,

            
			"block_range": 300 * nm2m,      #  Block range [m]
            "block_time": 1*3600,         #(1.0 + 2 * 40 / 60) * 3600,  #  Block time [s]
            "n_captains": 1,  # Number of captains in flight
            "n_copilots": 1,  # Number of copilots in flight
            "rho_fuel": 811,  # Fuel density kg/m3 (This is Jet A-1)
            # ------------------------------------------------------------------
            # All else
            #  Percentage of fuselage legnth for the Xcg all else
            "perc_cg_ae": 0.45,
            # ------------------------------------------------------------------
            # Mtow
            # Guess for MTOW (From 01_aircraft_survey.xlsx)
            "W0_guess": 20600 * gravity,
        }

    if name == "ERJ145-XR":
        airplane = {
            'name': name,
            "type": "transport",            # Can be 'transport', 'fighter', or 'general'

            # ------------------------------------------------------------------
            # Wing
            "S_w": 51.18,                  # Wing area [m2]
            "AR_w": 7.9,                  # Wing aspect ratio
            "taper_w": 0.224,               # Wing taper ratio
            "sweep_w": 0.39697,            # Wing sweep [rad]
            "dihedral_w": 0.05487,      # Wing dihedral [rad]
            # Longitudinal position of the wing (with respect to the fuselage nose) [m]
            "xr_w": 12.646,
            # Vertical position of the wing (with respect to the fuselage nose) [m]
            "zr_w": -1.056,
            "tcr_w": 0.14,           # t/c of the root section of the wing
            "tct_w": 0.10,           # t/c of the tip section of the wing
            # ------------------------------------------------------------------
            # Horizontal tail
            "has_HT": True,
            "Cht": 0.979,                   # Horizontal tail volume coefficient
            # Non-dimensional lever of the horizontal tail (lever/wing_mac)
            "Lc_h": 4.476,
            "AR_h": 5.089,                  # HT aspect ratio
            "taper_h": 0.605,               # HT taper ratio
            "sweep_h": 0.31384251,          # HT sweep [rad]
            "dihedral_h": 0,                # HT dihedral [rad]
            # Vertical position of the HT em relação ao centro da fuselagem [m]
            "zr_h": 3.772,
            "tcr_h": 0.097890752,           # t/c of the root section of the HT
            "tct_h": 0.093023256,           # t/c of the tip section of the HT
            "eta_h": 1.0,                   # Dynamic pressure factor of the HT
            # ------------------------------------------------------------------
            # Vertical tail
            "Cvt": 0.076,                   # Vertical tail volume coefficient
            # Non-dimensional lever of the vertical tail (lever/wing_span)
            "Lb_v": 0.541,
            "AR_v": 0.926,                  # VT aspect ratio
            "taper_v": 0.672,               # VT taper ratio
            "sweep_v": 0.601,         # VT sweep [rad]
            "zr_v": 0.608,                  # Vertical position of the VT [m]
            "tcr_v": 0.122601918,           # t/c of the root section of the VT
            "tct_v": 0.136099955,           # t/c of the tip section of the VT
            # ------------------------------------------------------------------
            # Canard
            "has_canard": False,
            # ------------------------------------------------------------------
            # Fuselage
            "L_f": 27.93,                   # Fuselage length [m]
            "D_f": 2.28,  # Fuselage diameter [m]
            # ------------------------------------------------------------------
            # Nacelle
            # Longitudinal position of the nacelle frontal face [m]
            "x_n": 20.478,
            # Lateral position of the nacelle centerline [m]
            "y_n": 2.107,
            # Vertical position of the nacelle centerline [m]
            "z_n": 0.555,

            # ------------------------------------------------------------------
            # Engine
            # Model: AE 3007A1E
            # Fan diameter: 0,98 m
            # Trhust: 7950 lbf
            "n_engines": 2,                 # Number of engines
            "n_engines_under_wing": 0,      # Number of engines installed under the wing
            "engine": {
                "model": "Howe turbofan",   # Check engineTSFC function for options
                "BPR": 5,                   # Engine bypass ratio
                "OAPR": 20,
                "Compressor_stages": 9,
                "Engine_shafts": 2,
                "Cbase": 0.696/3600,
                "T_eng_spec": 35363.349,     # Thurst of 1 engine [N]
                "W_eng_spec": 746*9.81,  # Engine Weight [N]
            },

            "L_n": 4.313,                   # Nacelle length [m]
            "D_n": 1.309,                   # Nacelle diameter [m]

            # ------------------------------------------------------------------
            # Landing gear
            # Longitudinal position of the nose landing gear [m]
            "x_nlg": 2.185,
            # Longitudinal position of the main landing gear [m]
            "x_mlg": 16.625,
            # Lateral position of the main landing gear [m]
            "y_mlg": 2.059,
            # Vertical position of the landing gear [m]
            "z_lg": -1.964,
            # Longitudinal position of critical tailstrike point [m]
            "x_tailstrike": 24.348,
            # Vertical position of critical tailstrike point [m]
            "z_tailstrike": -0.941,
            # ------------------------------------------------------------------
            # Tank [Ta faltando tudo isso]
            "c_tank_c_w": 0.6,             # Fraction of the wing chord occupied by the fuel tank
            "x_tank_c_w": 0.2,              # Fraction of the wing chord where fuel tank starts
            "b_tank_b_w_start": 0.0,        # Fraction of the wing semi-span where fuel tank starts
            "b_tank_b_w_end": 0.425,         # Fraction of the wing semi-span where fuel tank ends
            # ------------------------------------------------------------------
            # Airfoil
            #  Maximum lift coefficient of wing airfoil
            "clmax_w": 1.55,
            #  Airfoil technology factor for Korn equation (wave drag)
            "k_korn": 0.91,
            # ------------------------------------------------------------------
            # High-Lift devices
            "flap_type": "double slotted",  # Flap type
            "c_flap_c_wing": 0.1550,        # Fraction of the wing chord occupied by flaps
            # Fraction of the wing span occupied by flaps (including fuselage portion)
            "b_flap_b_wing": 0.6917,
            "slat_type": None,              # Slat type
            "c_slat_c_wing": 0.00,          # Fraction of the wing chord occupied by slats
            "b_slat_b_wing": 0.00,          # Fraction of the wing span occupied by slats
            "c_ail_c_wing": 0.2354,         # Fraction of the wing chord occupied by aileron
            "b_ail_b_wing": 0.2298,         # Fraction of the wing span occupied by aileron
            # Distance to the ground for ground effect computation [m]
            "h_ground": 35.0 * ft2m,
            "k_exc_drag": 0.03,             #  Excrescence drag factor
            "winglet": True,               # Add winglet
            # Até aqui
            # ------------------------------------------------------------------
            # Flight conditions
            # Altitude for takeoff computation [m]
            "altitude_takeoff": 0.0,
            "distance_takeoff": 2070.0,     # Required takeoff distance [m]
            # Variation from ISA standard temperature [ C] - From Obert's paper
            'deltaISA_takeoff': 0.0,
            'deltaISA_landing': 0.0,
            'MLW_frac': 20000/24100,

            # Altitude for landing computation [m]
            "altitude_landing": 0.0,
            "distance_landing": 1430.0,     # Required landing distance [m]
            # Cruise altitude [m] (From 01_aircraft_survey.xlsx)
            "altitude_cruise": 35000 * ft2m,
            # Cruise Mach number (From 01_aircraft_survey.xlsx)
            "Mach_cruise": 0.8,
            "range_cruise": 2000 * nm2m,    # Cruise range [m]
            "loiter_time": 45 * 60,         # Loiter time [s]

            #  Ceiling altitude [m]
            "altitude_ceiling": 37000 * ft2m, #4572
            #  Ceiling Mach number
            "Mach_ceiling": 0.7,

            #  Alternative cruise altitude [m]
            "altitude_altcruise": 4600,  # 4572
            #  Alternative cruise Mach number
            "Mach_altcruise": 0.4,
            # Alternative cruise range [m]
            "range_altcruise": 100 * nm2m,
            # ------------------------------------------------------------------
            # Payload
            "W_payload": 50*91 * gravity,    # Payload weight [N]
            #  Longitudinal position of the Payload center of gravity [m]
            "xcg_payload": 20,  # 13.5
            # [Julia - duas fileiras so tem um assento] Number of rows
            "N_rows": 18,
            # ------------------------------------------------------------------
            # Crew
            "W_crew": 3 * 90 * gravity,     # Crew weight [N]
            #  Longitudinal position of the Crew center of gravity [m]
            "xcg_crew": 9.677,

            "block_range": 600 * nm2m,      #  Block range [m]
            "block_time": 108*60,           #  Block time [s]
            "n_captains": 1,  # Number of captains in flight
            "n_copilots": 1,  # Number of copilots in flight
            "rho_fuel": 804,  # Fuel density kg/m3 (This is Jet A-1)
            # ------------------------------------------------------------------
            # All else
            #  Percentage of fuselage legnth for the Xcg all else
            "perc_cg_ae": 0.45,
            # ------------------------------------------------------------------
            # Mtow
            # Guess for MTOW (From 01_aircraft_survey.xlsx)
            "W0_guess": 24100 * gravity,
            "W_empty": (13100 - 3 * 90) * gravity,
        }

    elif name == "CRJ200":
        airplane = {
            'name': name,
            "type": "transport",            # Can be 'transport', 'fighter', or 'general'

            # ------------------------------------------------------------------
            # Wing
            "S_w": 54.5,                            # Wing area [m2]
            "AR_w": 8.9,                           # Wing aspect ratio
            "taper_w": 0.364,                       # Wing taper ratio
            "sweep_w": 0.386408659,                 # Wing sweep [rad]
            "dihedral_w": 0.034766955,              # Wing dihedral [rad]
            "xr_w": 9.733,                          # Longitudinal position of the wing (with respect to the fuselage nose) [m]
            "zr_w": -0.5127,                         # Vertical position of the wing (with respect to the fuselage nose) [m]
            "tcr_w": 0.132,                   # t/c of the root section of the wing
            "tct_w": 0.10,                   # t/c of the tip section of the wing
            # ------------------------------------------------------------------
            # Horizontal tail
            "has_HT": True,
            "Cht": 0.669,                           # Horizontal tail volume coefficient
            "Lc_h": 4.7, #adjusted to view          # Non-dimensional lever of the horizontal tail (lever/wing_mac)
            "AR_h": 4.611,                          # HT aspect ratio
            "taper_h": 0.431,                       # HT taper ratio
            "sweep_h": 0.517112,                  # HT sweep [rad]
            "dihedral_h": 0.007839,              # HT dihedral [rad]
            "zr_h": 3.8,                         # Vertical position of the HT em relação ao centro da fuselagem [m]
            "tcr_h": 0.114520891,                   # t/c of the root section of the HT
            "tct_h": 0.120972719,                   # t/c of the tip section of the HT
            "eta_h": 1.0,                           # Dynamic pressure fator of the HT                           # Dynamic pressure factor of the HT
            # ------------------------------------------------------------------
            # Vertical tail
            "Cvt": 0.081,                           # Vertical tail volume coefficient
            "Lb_v": 0.48,                           # Non-dimensional lever of the vertical tail (lever/wing_span)
            "AR_v": 1.014,                          # VT aspect ratio
            "taper_v": 0.667,                       # VT taper ratio
            "sweep_v": 0.725725404,                 # VT sweep [rad]
            
            "xr_v": 20.5169,
            
            "zr_v": 0.71,                         # Vertical position of the VT [m]
            "tcr_v": 0.147049866,                   # t/c of the root section of the VT
            "tct_v": 0.133003698,                   # t/c of the tip section of the VT
            # ------------------------------------------------------------------
            # Canard
            "has_canard": False,
            # ------------------------------------------------------------------
            # Fuselage
            "L_f": 24.38,                            # Fuselage length [m]
            "D_f": 2.69,                            # Fuselage diameter [m]
            # ------------------------------------------------------------------
            # Nacelle
            "x_n": 16.6533,                         # Longitudinal position of the nacelle frontal face [m]
            "y_n": 2.2292,                          # Lateral position of the nacelle centerline [m]
            "z_n": 0.6639,                          # Vertical position of the nacelle centerline [m]

            # ------------------------------------------------------------------
            # Engine
            # Model: GE CF34-3B1 turbofans
            # Fan diameter: 
            # Trhust: 8729 lbf
            "n_engines": 2,                         # Number of engines
            "n_engines_under_wing": 0,              # Number of engines installed under the wing
            "engine": {
                "model": "Howe turbofan",           # Check engineTSFC function for options
                "BPR": 6.2,                         # Engine bypass ratio
                "Cbase": 0.69/3600,                 # 0.69 I adjusted this value by hand to match the fuel weight
                "T_eng_spec":41012.6, #38828.5/41012.6              # Thurst of 1 engine [kN]
                "W_eng_spec": 757.5*9.81,           # Engine Weight [N]
            },

            "L_n": 3.7306,                          # Nacelle length [m]
            "D_n": 1.5054,                           # Nacelle diameter [m]

            # ------------------------------------------------------------------
            # Landing gear
            "x_nlg": 2.1689,                        # Longitudinal position of the nose landing gear [m]
            "x_mlg": 13.511,                        # Longitudinal position of the main landing gear [m]
            "y_mlg": 1.58997,                       # Lateral position of the main landing gear [m]
            "z_lg": -1.1429,                        # Vertical position of the landing gear [m]
            "x_tailstrike":17.69,                   # Longitudinal position of critical tailstrike point [m]
            "z_tailstrike": -1.42,                  # Vertical position of critical tailstrike point [m]
            # ------------------------------------------------------------------
            # Tank
            "c_tank_c_w": 0.6,                     # Fraction of the wing chord occupied by the fuel tank
            "x_tank_c_w": 0.2,                      # Fraction of the wing chord where fuel tank starts
            "b_tank_b_w_start": 0.0,                # Fraction of the wing semi-span where fuel tank starts
            "b_tank_b_w_end": 0.502,                 # Fraction of the wing semi-span where fuel tank ends
            # ------------------------------------------------------------------
            # Airfoil
            "clmax_w": 1.54,                         #  Maximum lift coefficient of wing airfoil
            "k_korn": 0.91,                         #  Airfoil technology factor for Korn equation (wave drag)
            # ------------------------------------------------------------------
            # High-Lift devices
            "flap_type": "double slotted",          # Flap type
            "c_flap_c_wing": 0.213825,                # Fraction of the wing chord occupied by flaps
            "b_flap_b_wing": (2.2321+3.3428)*2/(20.04 - 2.69),         # Fraction of the wing span occupied by flaps (including fuselage portion)
            "slat_type": None,                      # Slat type
            "c_slat_c_wing": 0.00,                  # Fraction of the wing chord occupied by slats
            "b_slat_b_wing": 0.00,                  # Fraction of the wing span occupied by slats
            "c_ail_c_wing": 0.34045,                 # Fraction of the wing chord occupied by aileron
            "b_ail_b_wing": 0.16537,                # Fraction of the wing span occupied by aileron
            "h_ground": 35.0 * ft2m,                # Distance to the ground for ground effect computation [m]
            "k_exc_drag": 0.03,                     #  Excrescence drag factor
            "winglet": True,                       #  Add winglet
            # ------------------------------------------------------------------
            # Flight conditions
            "altitude_takeoff": 0.0,                # Altitude for takeoff computation [m]
            "distance_takeoff": 1547.62,             # Required takeoff distance [m]
            # Variation from ISA standard temperature [ C] - From Obert's paper
            'deltaISA_takeoff': 0.0,
            'deltaISA_landing': 0.0,
            'MLW_frac': 20276/21523,                # Max Landing Weight / Max Takeoff Weight - From Obert's paper

            # Altitude for landing computation [m]
            "altitude_landing": 0.0,
            "distance_landing": 1440.0,             # Required landing distance [m]

            "altitude_cruise": 35000 * ft2m,        # Cruise altitude [m] (From 01_aircraft_survey.xlsx)
            "Mach_cruise": 0.74,                    # Cruise Mach number (From 01_aircraft_survey.xlsx)
            "range_cruise": 1500 * nm2m,             # Cruise range [m]
            "loiter_time": 45 * 60,                 # Loiter time [s]

            #  Ceiling altitude [m]
            "altitude_ceiling": 41000 * ft2m, #4572
            #  Ceiling Mach number
            "Mach_ceiling": 0.74,

            "altitude_altcruise": 4600,             #  Alternative cruise altitude [m]
            "Mach_altcruise": 0.4,                  #  Alternative cruise Mach number
            "range_altcruise": 100 * nm2m,          #  Alternative cruise range [m]
            # ------------------------------------------------------------------
            # Payload
            "W_payload": 4000 * gravity,           # Payload weight [N]
            "xcg_payload": 13.5,                    #  Longitudinal position of the Payload center of gravity [m]
            "N_rows": 13,                           #  Number of rows
            # ------------------------------------------------------------------
            # Crew
            "W_crew": 1 * 90 * gravity,             # Crew weight [N]
            "xcg_crew": 9.677,                      #  Longitudinal position of the Crew center of gravity [m]

            "block_range": 300 * nm2m,              #  Block range [m]
            "block_time": 1.5*3600,                 #  Block time [s]
            "n_captains": 1,                        # Number of captains in flight
            "n_copilots": 1,                        # Number of copilots in flight
            "rho_fuel": 811,                        # Fuel density kg/m3 (This is Jet A-1)
            # ------------------------------------------------------------------
            # All else
            #  Percentage of fuselage legnth for the Xcg all else
            "perc_cg_ae": 0.45,
            # ------------------------------------------------------------------
            # Mtow
            # Guess for MTOW (From 01_aircraft_survey.xlsx)
            "W0_guess": 21523 * gravity,
        }

    if name == "LRJ_C1":
        airplane = {
            'name': name,
            "type": "transport",            # Can be 'transport', 'fighter', or 'general'
    
            # ------------------------------------------------------------------
            # Wing
            
            # Data from Joseph regressions
            "S_w": 53.5,                           # Wing area [m2]
            
            # Data from bibliography 
            "AR_w": 9,                            # Wing aspect ratio
            "taper_w": 0.25,                      # Wing taper ratio
            "sweep_w": (23)*np.pi/180,             # Wing sweep [rad]
            "dihedral_w": (2)*np.pi/180,           # Wing dihedral [rad]
            
            # Data from benchmarking (ERJ145 and CRJ200)
            "xr_w": 10,                         # Longitudinal position of the wing (with respect to the fuselage nose) [m]
            "zr_w": -0.5127,                       # Vertical position of the wing (with respect to the fuselage nose) [m]
            "tcr_w": 0.12,                         # t/c of the root section of the wing
            "tct_w": 0.10,                         # t/c of the tip section of the wing
            # ------------------------------------------------------------------
            # Horizontal tail
            "has_HT": True,
            # Data from bibliography           
            "Cht": 0.9,                            # Horizontal tail volume coefficient
            "AR_h": 5,                          # HT aspect ratio
            "taper_h": 0.35,                       # HT taper ratio
            
            # Data from benchmarking
            "sweep_h": (30)*np.pi/180,                  # HT sweep [rad]
            "dihedral_h": 0.007839,              # HT dihedral [rad]            
            "tcr_h": 0.114520891,                   # t/c of the root section of the HT
            "tct_h": 0.120972719,                   # t/c of the tip section of the HT
            "eta_h": 1.0,                           # Dynamic pressure factor of the HT
            
            # Parameters will be changed manually to optimize the aircraft
            "Lc_h": 4.58*1.1,                          # Non-dimensional lever of the horizontal tail (lever/wing_mac)
            "zr_h": 4.1+0.5,                         # Vertical position of the HT em relação ao centro da fuselagem [m]
            # ------------------------------------------------------------------
            # Vertical tail
            
            # Data from bibliography
            "Cvt": 0.081,                           # Vertical tail volume coefficient
            "AR_v": 1.2,                          # VT aspect ratio
            "taper_v": 0.7,                       # VT taper ratio
            "sweep_v": 0.610865238,                 # VT sweep [rad]
            
            # Data from benchmarking
            "tcr_v": 0.147049866,                   # t/c of the root section of the VT
            "tct_v": 0.133003698,                   # t/c of the tip section of the VT

            # Parameters will be changed manually to optimize the aircraft
            "Lb_v": 0.48*1.1,                           # Non-dimensional lever of the vertical tail (lever/wing_span)
            "zr_v": 0.7875+0.5,                         # Vertical position of the VT [m]
            # ------------------------------------------------------------------
            # Canard
            "has_canard": False,
            # ------------------------------------------------------------------
            # Fuselage
            
            # Data from preliminar LOPA (for 50pax with 36pax regular class and 14pax first class)
            "L_f": 25.94,                            # Fuselage length [m]
            "D_f": 3.12,                            # Fuselage diameter [m]
            # ------------------------------------------------------------------
            # Nacelle
        
            "x_n": 17.5,                         # Longitudinal position of the nacelle frontal face [m]
            "y_n": 2.2292,                          # Lateral position of the nacelle centerline [m]
            "z_n": 0.6639,                          # Vertical position of the nacelle centerline [m]
    
            # ------------------------------------------------------------------
            # Engine
            
            "n_engines": 2,                 # Number of engines
            "n_engines_under_wing": 0,      # Number of engines installed under the wing
            "engine": {
                "model": "Howe turbofan",   # Check engineTSFC function for options
                "BPR": 5.3,                   # Engine bypass ratio
                #  I adjusted this value by hand to match the fuel weight
                "Cbase": 0.785/3600,
                "T_eng_spec": 37080.36,     # Thurst of 1 engine [N]
                "W_eng_spec": 751.6*9.81,  # Engine Weight [N]
            },

            "L_n": 4.313,                   # Nacelle length [m]
            "D_n": 1.309,                   # Nacelle diameter [m]            
    
            # ------------------------------------------------------------------
            # Landing gear
            
            # Parameters will be changed manually to optimize the aircraft
            "x_nlg": 2.1689,                        # Longitudinal position of the nose landing gear [m]
            "x_mlg": 13.511,                        # Longitudinal position of the main landing gear [m]
            "y_mlg": 1.58997,                       # Lateral position of the main landing gear [m]
            "z_lg": -1.8499,#-1.1429,                        # Vertical position of the landing gear [m]
            "x_tailstrike":17.69,                   # Longitudinal position of critical tailstrike point [m]
            "z_tailstrike": -1.42,                  # Vertical position of critical tailstrike point [m]
            # ------------------------------------------------------------------
            # Tank
            
            # Parameters will be changed manually to optimize the aircraft
            "c_tank_c_w": 0.6,                     # Fraction of the wing chord occupied by the fuel tank
            "x_tank_c_w": 0.2,                      # Fraction of the wing chord where fuel tank starts
            "b_tank_b_w_start": 0.0,                # Fraction of the wing semi-span where fuel tank starts
            "b_tank_b_w_end": 0.502,                 # Fraction of the wing semi-span where fuel tank ends
            # ------------------------------------------------------------------
            # Airfoil
            
            #
            "clmax_w": 1.54,                         #  Maximum lift coefficient of wing airfoil
            "k_korn": 0.91,                         #  Airfoil technology factor for Korn equation (wave drag)
            # ------------------------------------------------------------------
            # High-Lift devices
            
            #
            "flap_type": "double slotted",  # Flap type
            "c_flap_c_wing": 0.25,        # Fraction of the wing chord occupied by flaps
            # Fraction of the wing span occupied by flaps (including fuselage portion)
            "b_flap_b_wing": 0.7182,
            "slat_type": None,              # Slat type
            "c_slat_c_wing": 0.00,          # Fraction of the wing chord occupied by slats
            "b_slat_b_wing": 0.00,          # Fraction of the wing span occupied by slats
            "c_ail_c_wing": 0.2428,         # Fraction of the wing chord occupied by aileron
            "b_ail_b_wing": 0.2418,         # Fraction of the wing span occupied by aileron
            # Distance to the ground for ground effect computation [m]
            "h_ground": 35.0 * ft2m,
            "k_exc_drag": 0.218,             #  Excrescence drag factor
            "winglet": False,               # Add winglet
            # ------------------------------------------------------------------
            # Flight conditions
            "altitude_takeoff": 0.0,                # Altitude for takeoff computation [m]
            "distance_takeoff": 1500,             # Required takeoff distance [m]
            # Variation from ISA standard temperature [ C] - From Obert's paper
            'deltaISA_takeoff': 0.0,
            'deltaISA_landing': 0.0,
            'MLW_frac': 20276/21523,                # Max Landing Weight / Max Takeoff Weight - From Obert's paper
    
            # Altitude for landing computation [m]
            "altitude_landing": 0.0,
            "distance_landing": 1400,             # Required landing distance [m]
    
            "altitude_cruise": 35000 * ft2m,        # Cruise altitude [m] (From 01_aircraft_survey.xlsx)
            "Mach_cruise": 0.75,                    # Cruise Mach number (From 01_aircraft_survey.xlsx)
            "range_cruise": (800-200) * nm2m,             # Cruise range [m]
            "loiter_time": 45 * 60,                 # Loiter time [s]

            #  Ceiling altitude [m]
            "altitude_ceiling": 37000 * ft2m, #4572
            #  Ceiling Mach number
            "Mach_ceiling": 0.7,
    
            "altitude_altcruise": 4600,             #  Alternative cruise altitude [m]
            "Mach_altcruise": 0.4,                  #  Alternative cruise Mach number
            "range_altcruise": 100 * nm2m,          #  Alternative cruise range [m]
            # ------------------------------------------------------------------
            # Payload
            
            "W_payload": 50*100 * gravity,           # Payload weight [N]
            "xcg_payload": 10.5,                    #  Longitudinal position of the Payload center of gravity [m]
            "N_rows": 13,                           #  Number of rows
            # ------------------------------------------------------------------
            # Crew
            
            "W_crew": 1 * 90 * gravity,             # Crew weight [N]
            "xcg_crew": 3,                      #  Longitudinal position of the Crew center of gravity [m]
            "block_range": 300 * nm2m,              #  Block range [m]
            "block_time": 1.5*3600,                 #  Block time [s]
            "n_captains": 1,                        # Number of captains in flight
            "n_copilots": 1,                        # Number of copilots in flight
            "rho_fuel": 811,                        # Fuel density kg/m3 (This is Jet A-1)
            # ------------------------------------------------------------------
            # All else
            #  Percentage of fuselage legnth for the Xcg all else
            "perc_cg_ae": 0.40,
            # ------------------------------------------------------------------
            # Mtow
            # Guess for MTOW (From 01_aircraft_survey.xlsx)
            "W0_guess": 21523 * gravity,
        } 

    if name == "LRJ_01_Canard":
        airplane = {
            'name': name,
            "type": "transport",            # Can be 'transport', 'fighter', or 'general'

            # ------------------------------------------------------------------
            # Wing
            "S_w": 50,                  # Wing area [m2]
            "AR_w": 9.43,                  # Wing aspect ratio
            "taper_w": 0.255,               # Wing taper ratio
            "sweep_w": 0.4014,            # Wing sweep [rad]
            "dihedral_w": 0.050456141,      # Wing dihedral [rad]
            # Longitudinal position of the wing (with respect to the fuselage nose) [m]
            "xr_w": 12.649,
            # Vertical position of the wing (with respect to the fuselage nose) [m]
            "zr_w": -1.141,
            "tcr_w": 0.14,           # t/c of the root section of the wing
            "tct_w": 0.10,           # t/c of the tip section of the wing
            # ------------------------------------------------------------------
            # Horizontal tail
            "has_HT": True,
            "Cht": 0.9,                   # Horizontal tail volume coefficient
            # Non-dimensional lever of the horizontal tail (lever/wing_mac)
            "Lc_h": 4.93,
            "AR_h": 5,                  # HT aspect ratio
            "taper_h": 0.605,               # HT taper ratio
            "sweep_h": 0.49,          # HT sweep [rad]
            "dihedral_h": 0,                # HT dihedral [rad]
            # Vertical position of the HT em relação ao centro da fuselagem [m]
            "zr_h": 3.772,
            "tcr_h": 0.097890752,           # t/c of the root section of the HT
            "tct_h": 0.093023256,           # t/c of the tip section of the HT
            "eta_h": 1.0,                   # Dynamic pressure factor of the HT
            # ------------------------------------------------------------------
            # Vertical tail
            "Cvt": 0.085,                   # Vertical tail volume coefficient
            # Non-dimensional lever of the vertical tail (lever/wing_span)
            "Lb_v": 0.538,
            "AR_v": 1.254,                  # VT aspect ratio
            "taper_v": 0.578,               # VT taper ratio
            "sweep_v": 0.473175,         # VT sweep [rad]
            "zr_v": 0.608,                  # Vertical position of the VT [m]
            "tcr_v": 0.122601918,           # t/c of the root section of the VT
            "tct_v": 0.136099955,           # t/c of the tip section of the VT
            # ----------------------------------------------------------------------
            # Canard
            "has_canard": True,
            # Canard volume coefficient (análogo ao Cht)
            "Ccan": 0.77,
            # Lever arm non-dimensional (lever / wing_mac)
            "Lc_c": 3.08,
            "AR_c": 7.43,                 # Canard aspect ratio
            "taper_c": 0.8,              # Canard taper ratio
            "sweep_c": 0.488692,              # Canard sweep [rad]
            "dihedral_c": 0.0,           # Canard dihedral [rad]

            # Vertical position of canard root relative to fuselage reference [m]
            "zr_c": 1.15,
            "tcr_c": 0.1,               # t/c at canard root
            "tct_c": 0.09,               # t/c at canard tip
            # ------------------------------------------------------------------
            # Fuselage
            "L_f": 25.94,                            # Fuselage length [m]
            "D_f": 3.12,                            # Fuselage diameter [m]
            # ------------------------------------------------------------------
            # Nacelle
            # Longitudinal position of the nacelle frontal face [m]
            "x_n": 20.478,
            # Lateral position of the nacelle centerline [m]
            "y_n": 2.107,
            # Vertical position of the nacelle centerline [m]
            "z_n": 0.555,

            # ------------------------------------------------------------------
            # Engine
            # Model: AE 3007A1
            # Fan diameter: 0,98 m
            # Trhust: 7580 lbf
            "n_engines": 2,                 # Number of engines
            "n_engines_under_wing": 0,      # Number of engines installed under the wing
            "engine": {
                "model": "Howe turbofan",   # Check engineTSFC function for options
                "BPR": 5,                   # Engine bypass ratio
                #  I adjusted this value by hand to match the fuel weight
                "Cbase": 0.785/3600,
                "T_eng_spec": 33717.52,     # Thurst of 1 engine [N]
                "W_eng_spec": 746*9.81,  # Engine Weight [N]
            },

            "L_n": 4.313,                   # Nacelle length [m]
            "D_n": 1.309,                   # Nacelle diameter [m]

            # ------------------------------------------------------------------
            # Landing gear
            # Longitudinal position of the nose landing gear [m]
            "x_nlg": 2.185,
            # Longitudinal position of the main landing gear [m]
            "x_mlg": 16.625,
            # Lateral position of the main landing gear [m]
            "y_mlg": 2.059,
            # Vertical position of the landing gear [m]
            "z_lg": -1.964,
            # Longitudinal position of critical tailstrike point [m]
            "x_tailstrike": 24.348,
            # Vertical position of critical tailstrike point [m]
            "z_tailstrike": -0.941,
            # ------------------------------------------------------------------
            # Tank [Ta faltando tudo isso]
            "c_tank_c_w": 0.6,             # Fraction of the wing chord occupied by the fuel tank
            "x_tank_c_w": 0.2,              # Fraction of the wing chord where fuel tank starts
            "b_tank_b_w_start": 0.0,        # Fraction of the wing semi-span where fuel tank starts
            "b_tank_b_w_end": 0.425,         # Fraction of the wing semi-span where fuel tank ends
            # ------------------------------------------------------------------
            # Airfoil
            #  Maximum lift coefficient of wing airfoil
            "clmax_w": 1.62,  # 1.62
            #  Airfoil technology factor for Korn equation (wave drag)
            "k_korn": 0.91,
            # ------------------------------------------------------------------
            # High-Lift devices
            "flap_type": "double slotted",  # Flap type
            "c_flap_c_wing": 0.22,  # 0.1579,        # Fraction of the wing chord occupied by flaps
            # Fraction of the wing span occupied by flaps (including fuselage portion)
            "b_flap_b_wing": 0.7182,
            "slat_type": None,              # Slat type
            "c_slat_c_wing": 0.00,          # Fraction of the wing chord occupied by slats
            "b_slat_b_wing": 0.00,          # Fraction of the wing span occupied by slats
            "c_ail_c_wing": 0.2428,         # Fraction of the wing chord occupied by aileron
            "b_ail_b_wing": 0.2418,         # Fraction of the wing span occupied by aileron
            # Distance to the ground for ground effect computation [m]
            "h_ground": 35.0 * ft2m,
            "k_exc_drag": 0.218,             #  Excrescence drag factor
            "winglet": False,               # Add winglet
            # Até aqui
            # ------------------------------------------------------------------
            # Flight conditions
            # Altitude for takeoff computation [m]
            "altitude_takeoff": 0.0,
            # 1970 Required takeoff distance [m]
            "distance_takeoff": 1500.0,
            # Variation from ISA standard temperature [ C] - From Obert's paper
            'deltaISA_takeoff': 0.0,
            'deltaISA_landing': 0.0,
            'MLW_frac': 18700/20600,

            # Altitude for landing computation [m]
            "altitude_landing": 0.0,
            "distance_landing": 1400.0,     # Required landing distance [m]
            # Cruise altitude [m] (From 01_aircraft_survey.xlsx)
            "altitude_cruise": 35000 * ft2m,
            # Cruise Mach number (From 01_aircraft_survey.xlsx)
            "Mach_cruise": 0.75,
            "range_cruise": (800 - 200) * nm2m,    # Cruise range [m]
            "loiter_time": 45 * 60,         # Loiter time [s]

            #  Ceiling altitude [m]
            "altitude_ceiling": 37000 * ft2m, #4572
            #  Ceiling Mach number
            "Mach_ceiling": 0.7,

            #  Alternative cruise altitude [m]
            "altitude_altcruise": 4600,  # 4572
            #  Alternative cruise Mach number
            "Mach_altcruise": 0.4,
            # Alternative cruise range [m]
            "range_altcruise": 100 * nm2m,
            # ------------------------------------------------------------------
            # Payload
            "W_payload": 5000 * gravity,    # Payload weight [N]
            #  Longitudinal position of the Payload center of gravity [m]
            "xcg_payload": 13.5,
            # [Julia - duas fileiras so tem um assento] Number of rows
            "N_rows": 18,
            # ------------------------------------------------------------------
            # Crew
            "W_crew": 1 * 90 * gravity,     # Crew weight [N]
            #  Longitudinal position of the Crew center of gravity [m]
            "xcg_crew": 9.677,

            "block_range": 300 * nm2m,      #  Block range [m]
            # (1.0 + 2 * 40 / 60) * 3600,  #  Block time [s]
            "block_time": 1*3600,
            "n_captains": 1,  # Number of captains in flight
            "n_copilots": 1,  # Number of copilots in flight
            "rho_fuel": 811,  # Fuel density kg/m3 (This is Jet A-1)
            # ------------------------------------------------------------------
            # All else
            #  Percentage of fuselage legnth for the Xcg all else
            "perc_cg_ae": 0.45,
            # ------------------------------------------------------------------
            # Mtow
            # Guess for MTOW (From 01_aircraft_survey.xlsx)
            "W0_guess": 20600 * gravity,
        }

    if name == "LRJ_01_BW":
        airplane = {
            'name': name,
            "type": "transport",            # Can be 'transport', 'fighter', or 'general'

            # ------------------------------------------------------------------
            # Wing
            "S_w": 50,                  # Wing area [m2]
            "AR_w": 9.43,                  # Wing aspect ratio
            "taper_w": 0.255,               # Wing taper ratio
            "sweep_w": 0.4014,            # Wing sweep [rad]
            "dihedral_w": 0.050456141,      # Wing dihedral [rad]
            # Longitudinal position of the wing (with respect to the fuselage nose) [m]
            "xr_w": 12.649,
            # Vertical position of the wing (with respect to the fuselage nose) [m]
            "zr_w": -1.141,
            "tcr_w": 0.14,           # t/c of the root section of the wing
            "tct_w": 0.10,           # t/c of the tip section of the wing
            # ------------------------------------------------------------------
            # Horizontal tail
            "has_HT": False,
            "Cht": 0.9,                   # Horizontal tail volume coefficient
            # Non-dimensional lever of the horizontal tail (lever/wing_mac)
            "Lc_h": 4.93,
            "AR_h": 5,                  # HT aspect ratio
            "taper_h": 0.605,               # HT taper ratio
            "sweep_h": 0.49,          # HT sweep [rad]
            "dihedral_h": 0,                # HT dihedral [rad]
            # Vertical position of the HT em relação ao centro da fuselagem [m]
            "zr_h": 3.772,
            "tcr_h": 0.097890752,           # t/c of the root section of the HT
            "tct_h": 0.093023256,           # t/c of the tip section of the HT
            "eta_h": 1.0,                   # Dynamic pressure factor of the HT
            # ------------------------------------------------------------------
            # Vertical tail
            "Cvt": 0.085,                   # Vertical tail volume coefficient
            # Non-dimensional lever of the vertical tail (lever/wing_span)
            "Lb_v": 0.538,
            "AR_v": 1.254,                  # VT aspect ratio
            "taper_v": 0.578,               # VT taper ratio
            "sweep_v": 0.473175,         # VT sweep [rad]
            "zr_v": 0.608,                  # Vertical position of the VT [m]
            "tcr_v": 0.122601918,           # t/c of the root section of the VT
            "tct_v": 0.136099955,           # t/c of the tip section of the VT
            # ----------------------------------------------------------------------
            # Canard
            "has_canard": False,
            # Canard volume coefficient (análogo ao Cht)
            "Ccan": 0.77,
            # Lever arm non-dimensional (lever / wing_mac)
            "Lc_c": 3.08,
            "AR_c": 7.43,                 # Canard aspect ratio
            "taper_c": 0.8,              # Canard taper ratio
            "sweep_c": 0.488692,              # Canard sweep [rad]
            "dihedral_c": 0.0,           # Canard dihedral [rad]

            # Vertical position of canard root relative to fuselage reference [m]
            "zr_c": 1.15,
            "tcr_c": 0.1,               # t/c at canard root
            "tct_c": 0.09,               # t/c at canard tip
            # ----------------------------------------------------------------------
            # Box wing
            "box_wing": True,

            # Asa dianteira (front wing) - mesmos inputs de uma asa normal
            "S_wf": 50,
            "AR_wf": 9.43,
            "taper_wf": 0.255,
            "sweep_wf": 0.4014,      # [rad]
            "dihedral_wf": 0.050456141,   # [rad]
            "xr_wf": 12.649,         # x raiz
            "zr_wf": -1.141,         # z raiz
            "tcr_wf": 0.14,       # t/c na raiz
            "tct_wf": 0.10,       # t/c na ponta

            # Asa traseira (rear wing) - mesmos inputs de uma asa normal
            "S_wr": 50,
            "AR_wr": 9.43,
            "taper_wr": 0.255,
            "sweep_wr": -0.4014,      # [rad]
            "dihedral_wr": 0.050456141,   # [rad]
            "xr_wr": 25.649,         # x raiz
            "zr_wr": 3.772,         # z raiz
            "tcr_wr": 0.14,       # t/c na raiz
            "tct_wr": 0.10,       # t/c na ponta

            # (Opcional) “espessura extra”/fator visual do painel de junção
            # Se quiser, pode deixar 1.0 e pronto.
            "box_join_thickness_factor": 1.0,
            # ------------------------------------------------------------------
            # Fuselage
            "L_f": 25.94,                            # Fuselage length [m]
            "D_f": 3.12,                            # Fuselage diameter [m]
            # ------------------------------------------------------------------
            # Nacelle
            # Longitudinal position of the nacelle frontal face [m]
            "x_n": 20.478,
            # Lateral position of the nacelle centerline [m]
            "y_n": 2.107,
            # Vertical position of the nacelle centerline [m]
            "z_n": 0.555,

            # ------------------------------------------------------------------
            # Engine
            # Model: AE 3007A1
            # Fan diameter: 0,98 m
            # Trhust: 7580 lbf
            "n_engines": 2,                 # Number of engines
            "n_engines_under_wing": 0,      # Number of engines installed under the wing
            "engine": {
                "model": "Howe turbofan",   # Check engineTSFC function for options
                "BPR": 5,                   # Engine bypass ratio
                #  I adjusted this value by hand to match the fuel weight
                "Cbase": 0.785/3600,
                "T_eng_spec": 33717.52,     # Thurst of 1 engine [N]
                "W_eng_spec": 746*9.81,  # Engine Weight [N]
            },

            "L_n": 4.313,                   # Nacelle length [m]
            "D_n": 1.309,                   # Nacelle diameter [m]

            # ------------------------------------------------------------------
            # Landing gear
            # Longitudinal position of the nose landing gear [m]
            "x_nlg": 2.185,
            # Longitudinal position of the main landing gear [m]
            "x_mlg": 16.625,
            # Lateral position of the main landing gear [m]
            "y_mlg": 2.059,
            # Vertical position of the landing gear [m]
            "z_lg": -1.964,
            # Longitudinal position of critical tailstrike point [m]
            "x_tailstrike": 24.348,
            # Vertical position of critical tailstrike point [m]
            "z_tailstrike": -0.941,
            # ------------------------------------------------------------------
            # Tank [Ta faltando tudo isso]
            "c_tank_c_w": 0.6,             # Fraction of the wing chord occupied by the fuel tank
            "x_tank_c_w": 0.2,              # Fraction of the wing chord where fuel tank starts
            "b_tank_b_w_start": 0.0,        # Fraction of the wing semi-span where fuel tank starts
            "b_tank_b_w_end": 0.425,         # Fraction of the wing semi-span where fuel tank ends
            # ------------------------------------------------------------------
            # Airfoil
            #  Maximum lift coefficient of wing airfoil
            "clmax_w": 1.62,  # 1.62
            #  Airfoil technology factor for Korn equation (wave drag)
            "k_korn": 0.91,
            # ------------------------------------------------------------------
            # High-Lift devices
            "flap_type": "double slotted",  # Flap type
            "c_flap_c_wing": 0.22,  # 0.1579,        # Fraction of the wing chord occupied by flaps
            # Fraction of the wing span occupied by flaps (including fuselage portion)
            "b_flap_b_wing": 0.7182,
            "slat_type": None,              # Slat type
            "c_slat_c_wing": 0.00,          # Fraction of the wing chord occupied by slats
            "b_slat_b_wing": 0.00,          # Fraction of the wing span occupied by slats
            "c_ail_c_wing": 0.2428,         # Fraction of the wing chord occupied by aileron
            "b_ail_b_wing": 0.2418,         # Fraction of the wing span occupied by aileron
            # Distance to the ground for ground effect computation [m]
            "h_ground": 35.0 * ft2m,
            "k_exc_drag": 0.218,             #  Excrescence drag factor
            "winglet": False,               # Add winglet
            # Até aqui
            # ------------------------------------------------------------------
            # Flight conditions
            # Altitude for takeoff computation [m]
            "altitude_takeoff": 0.0,
            # 1970 Required takeoff distance [m]
            "distance_takeoff": 1500.0,
            # Variation from ISA standard temperature [ C] - From Obert's paper
            'deltaISA_takeoff': 0.0,
            'deltaISA_landing': 0.0,
            'MLW_frac': 18700/20600,

            # Altitude for landing computation [m]
            "altitude_landing": 0.0,
            "distance_landing": 1400.0,     # Required landing distance [m]
            # Cruise altitude [m] (From 01_aircraft_survey.xlsx)
            "altitude_cruise": 35000 * ft2m,
            # Cruise Mach number (From 01_aircraft_survey.xlsx)
            "Mach_cruise": 0.75,
            "range_cruise": (800 - 200) * nm2m,    # Cruise range [m]
            "loiter_time": 45 * 60,         # Loiter time [s]

            #  Ceiling altitude [m]
            "altitude_ceiling": 37000 * ft2m, #4572
            #  Ceiling Mach number
            "Mach_ceiling": 0.7,

            #  Alternative cruise altitude [m]
            "altitude_altcruise": 4600,  # 4572
            #  Alternative cruise Mach number
            "Mach_altcruise": 0.4,
            # Alternative cruise range [m]
            "range_altcruise": 100 * nm2m,
            # ------------------------------------------------------------------
            # Payload
            "W_payload": 5000 * gravity,    # Payload weight [N]
            #  Longitudinal position of the Payload center of gravity [m]
            "xcg_payload": 13.5,
            # [Julia - duas fileiras so tem um assento] Number of rows
            "N_rows": 18,
            # ------------------------------------------------------------------
            # Crew
            "W_crew": 1 * 90 * gravity,     # Crew weight [N]
            #  Longitudinal position of the Crew center of gravity [m]
            "xcg_crew": 9.677,

            "block_range": 300 * nm2m,      #  Block range [m]
            # (1.0 + 2 * 40 / 60) * 3600,  #  Block time [s]
            "block_time": 1*3600,
            "n_captains": 1,  # Number of captains in flight
            "n_copilots": 1,  # Number of copilots in flight
            "rho_fuel": 811,  # Fuel density kg/m3 (This is Jet A-1)
            # ------------------------------------------------------------------
            # All else
            #  Percentage of fuselage legnth for the Xcg all else
            "perc_cg_ae": 0.45,
            # ------------------------------------------------------------------
            # Mtow
            # Guess for MTOW (From 01_aircraft_survey.xlsx)
            "W0_guess": 20600 * gravity,
        }

    return airplane


# ----------------------------------------


def plot3d(airplane, figname="3dview.png", dirname="imagens_opt",  az1=0, az2=-90, ax=None, auxlines=True):
    """
    az1 and az2: degrees of azimuth and elevation for the 3d plot view
    """
    if ax == None:
        fig = plt.figure(facecolor='#ffffff')
        ax = fig.add_subplot(projection="3d")
        save = True
    else:
        save = False

    from matplotlib.patches import Ellipse
    import mpl_toolkits.mplot3d.art3d as art3d

    def plot_trap_surface(ax, xr, zr, xt, yt, zt, cr, ct, tcr, tct, color="blue", lw=1.0):
        # Upper
        ax.plot(
            [xr, xt, xt + ct, xr + cr, xt + ct, xt, xr],
            [0.0, yt, yt, 0.0, -yt, -yt, 0.0],
            [
                zr + cr * tcr / 2,
                zt + ct * tct / 2,
                zt + ct * tct / 2,
                zr + cr * tcr / 2,
                zt + ct * tct / 2,
                zt + ct * tct / 2,
                zr + cr * tcr / 2,
            ],
            color=color,
            lw=lw,
        )

        # Lower
        ax.plot(
            [xr, xt, xt + ct, xr + cr, xt + ct, xt, xr],
            [0.0, yt, yt, 0.0, -yt, -yt, 0.0],
            [
                zr - cr * tcr / 2,
                zt - ct * tct / 2,
                zt - ct * tct / 2,
                zr - cr * tcr / 2,
                zt - ct * tct / 2,
                zt - ct * tct / 2,
                zr - cr * tcr / 2,
            ],
            color=color,
            lw=lw,
        )

    xr_w = airplane["xr_w"]
    zr_w = airplane["zr_w"]
    b_w = airplane["b_w"]

    tct_w = airplane["tct_w"]
    tcr_w = airplane["tcr_w"]

    cr_w = airplane["cr_w"]
    xt_w = airplane["xt_w"]
    yt_w = airplane["yt_w"]
    zt_w = airplane["zt_w"]
    ct_w = airplane["ct_w"]

    has_HT = airplane.get("has_HT", False) and ("xr_h" in airplane)

    if has_HT:
        xr_h = airplane["xr_h"]
        zr_h = airplane["zr_h"]

        tcr_h = airplane["tcr_h"]
        tct_h = airplane["tct_h"]

        cr_h = airplane["cr_h"]
        xt_h = airplane["xt_h"]
        yt_h = airplane["yt_h"]
        zt_h = airplane["zt_h"]
        ct_h = airplane["ct_h"]
        b_h = airplane["b_h"]

    xr_v = airplane["xr_v"]
    zr_v = airplane["zr_v"]

    tcr_v = airplane["tcr_v"]
    tct_v = airplane["tct_v"]

    cr_v = airplane["cr_v"]
    xt_v = airplane["xt_v"]
    zt_v = airplane["zt_v"]
    ct_v = airplane["ct_v"]
    b_v = airplane["b_v"]

    # --- CANARD ---
    has_canard = airplane.get("has_canard", False) and ("xr_c" in airplane)

    if has_canard:
        xr_c = airplane["xr_c"]
        zr_c = airplane["zr_c"]

        tcr_c = airplane["tcr_c"]
        tct_c = airplane["tct_c"]

        cr_c = airplane["cr_c"]
        xt_c = airplane["xt_c"]
        yt_c = airplane["yt_c"]
        zt_c = airplane["zt_c"]
        ct_c = airplane["ct_c"]

    L_f = airplane["L_f"]
    D_f = airplane["D_f"]
    x_n = airplane["x_n"]
    y_n = airplane["y_n"]
    z_n = airplane["z_n"]
    L_n = airplane["L_n"]
    D_n = airplane["D_n"]

    has_winglet = airplane["winglet"]

    if "xcg_fwd" in airplane:
        xcg_fwd = airplane["xcg_fwd"]
        xcg_aft = airplane["xcg_aft"]
    else:
        xcg_fwd = None
        xcg_aft = None

    if "xnp" in airplane:
        xnp = airplane["xnp"]
    else:
        xnp = None

    x_nlg = airplane["x_nlg"]
    y_nlg = 0
    z_nlg = airplane["z_lg"]
    x_mlg = airplane["x_mlg"]
    y_mlg = airplane["y_mlg"]
    z_mlg = airplane["z_lg"]
    x_tailstrike = airplane["x_tailstrike"]
    z_tailstrike = airplane["z_tailstrike"]

    flap_type = airplane["flap_type"]
    b_flap_b_wing = airplane["b_flap_b_wing"]
    c_flap_c_wing = airplane["c_flap_c_wing"]

    slat_type = airplane["slat_type"]
    b_slat_b_wing = airplane["b_slat_b_wing"]
    c_slat_c_wing = airplane["c_slat_c_wing"]

    b_ail_b_wing = airplane["b_ail_b_wing"]
    c_ail_c_wing = airplane["c_ail_c_wing"]

    # PLOT

    # fig = plt.figure(fignum,figsize=(20, 10))

    # ax.set_aspect('equal')

    ax.plot(
        [xr_w, xt_w, xt_w + ct_w, xr_w + cr_w, xt_w + ct_w, xt_w, xr_w],
        [0.0, yt_w, yt_w, 0.0, -yt_w, -yt_w, 0.0],
        [
            zr_w + cr_w * tcr_w / 2,
            zt_w + ct_w * tct_w / 2,
            zt_w + ct_w * tct_w / 2,
            zr_w + cr_w * tcr_w / 2,
            zt_w + ct_w * tct_w / 2,
            zt_w + ct_w * tct_w / 2,
            zr_w + cr_w * tcr_w / 2,
        ],
        color='#0046AB',
    )

    if has_winglet:
        ttw = 0.21  # Winglet taper ratio
        ax.plot(
            [xt_w, xt_w + (1 - ttw) * ct_w, xt_w + ct_w, xt_w + ct_w, xt_w],
            [yt_w, yt_w, yt_w, yt_w, yt_w],
            [zt_w, zt_w + ct_w, zt_w + ct_w, zt_w, zt_w],
            color="#0046AB",
        )
        ax.plot(
            [xt_w, xt_w + (1 - ttw) * ct_w, xt_w + ct_w, xt_w + ct_w, xt_w],
            [-yt_w, -yt_w, -yt_w, -yt_w, -yt_w],
            [zt_w, zt_w + ct_w, zt_w + ct_w, zt_w, zt_w],
            color="#0046AB",
        )

    if has_HT:
        ax.plot(
            [xr_h, xt_h, xt_h + ct_h, xr_h + cr_h, xt_h + ct_h, xt_h, xr_h],
            [0.0, yt_h, yt_h, 0.0, -yt_h, -yt_h, 0.0],
            [
                zr_h + cr_h * tcr_h / 2,
                zt_h + ct_h * tct_h / 2,
                zt_h + ct_h * tct_h / 2,
                zr_h + cr_h * tcr_h / 2,
                zt_h + ct_h * tct_h / 2,
                zt_h + ct_h * tct_h / 2,
                zr_h + cr_h * tcr_h / 2,
            ],
            color="#0046AB",
        )

    ax.plot(
        [xr_v, xt_v, xt_v + ct_v, xr_v + cr_v, xr_v],
        [
            tcr_v * cr_v / 2,
            tct_v * ct_v / 2,
            tct_v * ct_v / 2,
            tcr_v * cr_v / 2,
            tcr_v * cr_v / 2,
        ],
        [zr_v, zt_v, zt_v, zr_v, zr_v],
        color="#0046AB",
    )

    ax.plot(
        [xr_v, xt_v, xt_v + ct_v, xr_v + cr_v, xr_v],
        [
            -tcr_v * cr_v / 2,
            -tct_v * ct_v / 2,
            -tct_v * ct_v / 2,
            -tcr_v * cr_v / 2,
            -tcr_v * cr_v / 2,
        ],
        [zr_v, zt_v, zt_v, zr_v, zr_v],
        color="#0046AB",
    )

    ax.plot(
    [xr_v, xr_v,
     xt_v, xt_v],
    [-tcr_v * cr_v / 2,  tcr_v * cr_v / 2,
     tct_v * ct_v / 2,  -tct_v * ct_v / 2],
    [zr_v, zr_v,
     zt_v, zt_v],
    color="#0046AB",
)
    ax.plot(
    [xr_v+cr_v, xr_v+cr_v,
     xt_v+ct_v, xt_v+ct_v],
    [-tcr_v * cr_v / 2,  tcr_v * cr_v / 2,
     -tct_v * ct_v / 2,  tct_v * ct_v / 2],
    [zr_v, zr_v,
     zt_v, zt_v],
    color="#0046AB",
)

    # Centerlines for fuselage and nacelles
    ax.plot([0.0, L_f], [0.0, 0.0], [0.0, 0.0])
    #ax.plot([x_n, x_n + L_n], [y_n, y_n], [z_n, z_n])
    #ax.plot([x_n, x_n + L_n], [-y_n, -y_n], [z_n, z_n])

    # Forward CG point
    if xcg_fwd is not None:
        ax.plot([xcg_fwd], [0.0], [0.0], "ko")

    # Rear CG point
    if xcg_aft is not None:
        ax.plot([xcg_aft], [0.0], [0.0], "ko")

    # Neutral point
    if xnp is not None:
        ax.plot([xnp], [0.0], [0.0], "x")

    # Define a parametrized fuselage by setting height and width
    # values along its axis
    # xx is non-dimensionalized by fuselage length
    # hh and ww are non-dimensionalized by fuselage diameter
    # There are 6 stations where we define the arrays:
    # nose1; nose2; nose3; cabin start; tailstrike; tail
    xx = [0.0, 1.24 / 41.72, 3.54 / 41.72,
          7.55 / 41.72, x_tailstrike / L_f, 1.0]
    hh = [0.0, 2.27 / 4.0, 3.56 / 4.0, 1.0, 1.0, 1.07 / 4.0]
    ww = [0.0, 1.83 / 4.0, 3.49 / 4.0, 1.0, 1.0, 0.284 / 4]
    num_tot_ell = 40  # Total number of ellipses

    # Loop over every section
    for ii in range(len(xx) - 1):
        # Define number of ellipses based on the section length
        num_ell = int((xx[ii + 1] - xx[ii]) * num_tot_ell) + 1

        # Define arrays of dimensional positions, heights and widths
        # for the current section
        xdim = np.linspace(xx[ii], xx[ii + 1], num_ell) * L_f
        hdim = np.linspace(hh[ii], hh[ii + 1], num_ell) * D_f
        wdim = np.linspace(ww[ii], ww[ii + 1], num_ell) * D_f

        # Loop over every ellipse
        for xc, hc, wc in zip(xdim, hdim, wdim):
            # Define ellipse center to make flat top at the fuselage tail
            if xc > x_tailstrike:
                yye = (D_f - hc) / 2
            else:
                yye = 0

            p = Ellipse(
                (0, yye), wc, hc, angle=0, facecolor="none", edgecolor="#5C6C74", lw=0.6
            )
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=xc, zdir="x")

    # ____________________________________________________________
    #                                                            \
    # MLG / NLG

    # Check if LG is activated
    d_lg = 0
    if x_nlg is not None:
        # Make landing gear dimensions based on the fuselage
        w_lg = 0.05 * D_f
        d_lg = 4 * w_lg

        mlg_len = np.linspace(y_mlg - w_lg / 2, y_mlg + w_lg / 2, 2)
        nlg_len = np.linspace(y_nlg - w_lg / 2, y_nlg + w_lg / 2, 2)

        for i in range(len(mlg_len)):
            p = Ellipse(
                (x_mlg, z_mlg),
                d_lg,
                d_lg,
                angle=0,
                facecolor="gray",
                edgecolor="k",
                lw=2,
            )
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=mlg_len[i], zdir="y")

            p = Ellipse(
                (x_mlg, z_mlg),
                d_lg,
                d_lg,
                angle=0,
                facecolor="gray",
                edgecolor="k",
                lw=2,
            )
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-mlg_len[i], zdir="y")

            # NLG
            p = Ellipse(
                (x_nlg, z_nlg),
                d_lg,
                d_lg,
                angle=0,
                facecolor="gray",
                edgecolor="k",
                lw=1.5,
            )
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=nlg_len[i], zdir="y")

    # Nacelle
    nc_len = np.linspace(x_n, x_n + L_n, 20)
    for i in range(len(nc_len)):
        p = Ellipse(
            (y_n, z_n), D_n, D_n, angle=0, facecolor="none", edgecolor="#FFC800", lw=1.0
        )
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=nc_len[i], zdir="x")

        # Inner wall
        p = Ellipse((y_n, z_n), D_n*0.8, D_n*0.8, angle=0,\
        facecolor = 'none', edgecolor = 'k', lw=.1)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=nc_len[i], zdir="x")

        p = Ellipse(
            (-y_n, z_n), D_n, D_n, angle=0, facecolor="none", edgecolor="#FFC800", lw=1.0
        )
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=nc_len[i], zdir="x")

        # Inner wall
        p = Ellipse((-y_n, z_n), D_n*0.8, D_n*0.8, angle=0, \
        facecolor = 'none', edgecolor = 'k', lw=.1)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=nc_len[i], zdir="x")

    # Aileron
    ail_tip_margin = 0.02  # Margem entre flap e aileron em % de b_w

    # Spanwise positions (root and tip)
    yr_a = (1.0 - (ail_tip_margin + b_ail_b_wing)) * b_w / 2
    yt_a = (1.0 - (ail_tip_margin)) * b_w / 2

    cr_a = lin_interp(0, b_w / 2, cr_w, ct_w, yr_a) * c_ail_c_wing
    ct_a = lin_interp(0, b_w / 2, cr_w, ct_w, yt_a) * c_ail_c_wing

    # To find the longitudinal position of the aileron LE, we find the TE position first
    # then we subtract the chord
    xr_a = lin_interp(0, b_w / 2, xr_w + cr_w, xt_w + ct_w, yr_a) - cr_a
    xt_a = lin_interp(0, b_w / 2, xr_w + cr_w, xt_w + ct_w, yt_a) - ct_a

    zr_a = lin_interp(0, b_w / 2, zr_w, zt_w, yr_a)
    zt_a = lin_interp(0, b_w / 2, zr_w, zt_w, yt_a)

    # Airfoil thickness at aileron location
    tcr_a = lin_interp(0, b_w / 2, tcr_w, tct_w, yr_a)
    tct_a = lin_interp(0, b_w / 2, tcr_w, tct_w, yt_a)

    ax.plot(
        [xr_a, xt_a, xt_a + ct_a, xr_a + cr_a, xr_a],
        [yr_a, yt_a, yt_a, yr_a, yr_a],
        [
            zr_a + cr_a * tcr_a / 2 / c_ail_c_wing,
            zt_a + ct_a * tct_a / 2 / c_ail_c_wing,
            zt_a + ct_a * tct_a / 2 / c_ail_c_wing,
            zr_a + cr_a * tcr_a / 2 / c_ail_c_wing,
            zr_a + cr_a * tcr_a / 2 / c_ail_c_wing,
        ],
        lw=1,
        color="#1CB794",
    )

    ax.plot(
        [xr_a, xt_a, xt_a + ct_a, xr_a + cr_a, xr_a],
        [-yr_a, -yt_a, -yt_a, -yr_a, -yr_a],
        [
            zr_a + cr_a * tcr_a / 2 / c_ail_c_wing,
            zt_a + ct_a * tct_a / 2 / c_ail_c_wing,
            zt_a + ct_a * tct_a / 2 / c_ail_c_wing,
            zr_a + cr_a * tcr_a / 2 / c_ail_c_wing,
            zr_a + cr_a * tcr_a / 2 / c_ail_c_wing,
        ],
        lw=1,
        color="#1CB794",
    )

    # Fuel tank
    c_tank_c_w = airplane["c_tank_c_w"]
    x_tank_c_w = airplane["x_tank_c_w"]
    b_tank_b_w_start = airplane["b_tank_b_w_start"]
    b_tank_b_w_end = airplane["b_tank_b_w_end"]

    # Spanwise positions (root and tip)
    yr_tk = b_tank_b_w_start * b_w / 2
    yt_tk = b_tank_b_w_end * b_w / 2

    cr_tk = lin_interp(0, b_w / 2, cr_w, ct_w, yr_tk) * c_tank_c_w
    ct_tk = lin_interp(0, b_w / 2, cr_w, ct_w, yt_tk) * c_tank_c_w

    # To find the longitudinal position of the tank LE
    xr_tk = lin_interp(0, b_w / 2, xr_w, xt_w, yr_tk) + \
        cr_tk * x_tank_c_w / c_tank_c_w
    xt_tk = lin_interp(0, b_w / 2, xr_w, xt_w, yt_tk) + \
        ct_tk * x_tank_c_w / c_tank_c_w

    zr_tk = lin_interp(0, b_w / 2, zr_w, zt_w, yr_tk)
    zt_tk = lin_interp(0, b_w / 2, zr_w, zt_w, yt_tk)

    # Airfoil thickness at tank location
    tcr_tk = lin_interp(0, b_w / 2, tcr_w, tct_w, yr_tk)
    tct_tk = lin_interp(0, b_w / 2, tcr_w, tct_w, yt_tk)

    ax.plot(
        [xr_tk, xt_tk, xt_tk + ct_tk, xr_tk + cr_tk, xr_tk],
        [yr_tk, yt_tk, yt_tk, yr_tk, yr_tk],
        [
            zr_tk + cr_tk * tcr_tk / 2,
            zt_tk + ct_tk * tct_tk / 2,
            zt_tk + ct_tk * tct_tk / 2,
            zr_tk + cr_tk * tcr_tk / 2,
            zr_tk + cr_tk * tcr_tk / 2,
        ],
        lw=1,
        color="#1E3137",
    )

    ax.plot(
        [xr_tk, xt_tk, xt_tk + ct_tk, xr_tk + cr_tk, xr_tk],
        [-yr_tk, -yt_tk, -yt_tk, -yr_tk, -yr_tk],
        [
            zr_tk + cr_tk * tcr_tk / 2,
            zt_tk + ct_tk * tct_tk / 2,
            zt_tk + ct_tk * tct_tk / 2,
            zr_tk + cr_tk * tcr_tk / 2,
            zr_tk + cr_tk * tcr_tk / 2,
        ],
        lw=1,
        color="#1E3137",
    )

    # Slat
    if slat_type is not None:
        # slat_tip_margin = 0.02  # Margem da ponta como % da b_w
        # slat_root_margin = 0.12 # Margem da raiz como % da b_w
        # hist_c_s = 0.25        # Corda do Flap
        # hist_b_s = 1 - slat_root_margin - slat_tip_margin

        # Spanwise positions (root and tip)
        yr_s = D_f / 2
        yt_s = b_slat_b_wing * b_w / 2

        cr_s = lin_interp(0, b_w / 2, cr_w, ct_w, yr_s) * c_slat_c_wing
        ct_s = lin_interp(0, b_w / 2, cr_w, ct_w, yt_s) * c_slat_c_wing

        # Find the longitudinal position of the slat LE
        xr_s = lin_interp(0, b_w / 2, xr_w, xt_w, yr_s)
        xt_s = lin_interp(0, b_w / 2, xr_w, xt_w, yt_s)

        zr_s = lin_interp(0, b_w / 2, zr_w, zt_w, yr_s)
        zt_s = lin_interp(0, b_w / 2, zr_w, zt_w, yt_s)

        # Airfoil thickness at slat location
        tcr_s = lin_interp(0, b_w / 2, tcr_w, tct_w, yr_s)
        tct_s = lin_interp(0, b_w / 2, tcr_w, tct_w, yt_s)

        ax.plot(
            [xr_s, xt_s, xt_s + ct_s, xr_s + cr_s, xr_s],
            [yr_s, yt_s, yt_s, yr_s, yr_s],
            [
                zr_s + cr_s * tcr_s / 2 / c_slat_c_wing,
                zt_s + ct_s * tct_s / 2 / c_slat_c_wing,
                zt_s + ct_s * tct_s / 2 / c_slat_c_wing,
                zr_s + cr_s * tcr_s / 2 / c_slat_c_wing,
                zr_s + cr_s * tcr_s / 2 / c_slat_c_wing,
            ],
            lw=1,
            color="#1CB794",
        )

        ax.plot(
            [xr_s, xt_s, xt_s + ct_s, xr_s + cr_s, xr_s],
            [-yr_s, -yt_s, -yt_s, -yr_s, -yr_s],
            [
                zr_s + cr_s * tcr_s / 2 / c_slat_c_wing,
                zt_s + ct_s * tct_s / 2 / c_slat_c_wing,
                zt_s + ct_s * tct_s / 2 / c_slat_c_wing,
                zr_s + cr_s * tcr_s / 2 / c_slat_c_wing,
                zr_s + cr_s * tcr_s / 2 / c_slat_c_wing,
            ],
            lw=1,
            color="#1CB794",
        )

    # Flap outboard
    if flap_type is not None:
        # Spanwise positions (root and tip)
        yr_f = D_f / 2
        yt_f = b_flap_b_wing * b_w / 2

        cr_f = lin_interp(0, b_w / 2, cr_w, ct_w, yr_f) * c_flap_c_wing
        ct_f = lin_interp(0, b_w / 2, cr_w, ct_w, yt_f) * c_flap_c_wing

        # To find the longitudinal position of the flap LE, we find the TE position first
        # then we subtract the chord
        xr_f = lin_interp(0, b_w / 2, xr_w + cr_w, xt_w + ct_w, yr_f) - cr_f
        xt_f = lin_interp(0, b_w / 2, xr_w + cr_w, xt_w + ct_w, yt_f) - ct_f

        zr_f = lin_interp(0, b_w / 2, zr_w, zt_w, yr_f)
        zt_f = lin_interp(0, b_w / 2, zr_w, zt_w, yt_f)

        # Airfoil thickness at flap location
        tcr_f = lin_interp(0, b_w / 2, tcr_w, tct_w, yr_f)
        tct_f = lin_interp(0, b_w / 2, tcr_w, tct_w, yt_f)

        ax.plot(
            [xr_f, xt_f, xt_f + ct_f, xr_f + cr_f, xr_f],
            [yr_f, yt_f, yt_f, yr_f, yr_f],
            [
                zr_f + cr_f * tcr_f / 2 / c_flap_c_wing,
                zt_f + ct_f * tct_f / 2 / c_flap_c_wing,
                zt_f + ct_f * tct_f / 2 / c_flap_c_wing,
                zr_f + cr_f * tcr_f / 2 / c_flap_c_wing,
                zr_f + cr_f * tcr_f / 2 / c_flap_c_wing,
            ],
            lw=1,
            color="#1CB794",
        )

        ax.plot(
            [xr_f, xt_f, xt_f + ct_f, xr_f + cr_f, xr_f],
            [-yr_f, -yt_f, -yt_f, -yr_f, -yr_f],
            [
                zr_f + cr_f * tcr_f / 2 / c_flap_c_wing,
                zt_f + ct_f * tct_f / 2 / c_flap_c_wing,
                zt_f + ct_f * tct_f / 2 / c_flap_c_wing,
                zr_f + cr_f * tcr_f / 2 / c_flap_c_wing,
                zr_f + cr_f * tcr_f / 2 / c_flap_c_wing,
            ],
            lw=1,
            color="#1CB794",
        )

    # Elevator

    if has_HT:

        ele_tip_margin = 0.1  # Margem do profundor para a ponta
        ele_root_margin = 0.1  # Margem do profundor para a raiz
        hist_b_e = 1 - ele_root_margin - ele_tip_margin
        hist_c_e = 0.25

        ct_e_loc = (1 - ele_tip_margin) * (ct_h - cr_h) + cr_h
        cr_e_loc = (1 - hist_b_e - ele_tip_margin) * (ct_h - cr_h) + cr_h

        ct_e = ct_e_loc * hist_c_e
        cr_e = cr_e_loc * hist_c_e

        xr_e = (
            (1 - hist_b_e - ele_tip_margin) * (xt_h - xr_h)
            + xr_h
            + cr_e_loc * (1 - hist_c_e)
        )
        xt_e = (1 - ele_tip_margin) * (xt_h - xr_h) + \
            xr_h + ct_e_loc * (1 - hist_c_e)

        yr_e = (1 - hist_b_e - ele_tip_margin) * b_h / 2
        yt_e = (1 - ele_tip_margin) * b_h / 2

        zr_e = (1 - hist_b_e - ele_tip_margin) * (zt_h - zr_h) + zr_h
        zt_e = (1 - ele_tip_margin) * (zt_h - zr_h) + zr_h

        ax.plot(
            [xr_e, xt_e, xt_e + ct_e, xr_e + cr_e, xr_e],
            [yr_e, yt_e, yt_e, yr_e, yr_e],
            [zr_e, zt_e, zt_e, zr_e, zr_e],
            lw=1,
            color="#1CB794",
        )

        ax.plot(
            [xr_e, xt_e, xt_e + ct_e, xr_e + cr_e, xr_e],
            [-yr_e, -yt_e, -yt_e, -yr_e, -yr_e],
            [zr_e, zt_e, zt_e, zr_e, zr_e],
            lw=1,
            color="#1CB794",
        )

    # Rudder
    ver_base_margin = 0.1  # Local da base % de b_v
    ver_tip_margin1 = 0.1  # Local da base % de b_v
    ver_tip_margin = 1 - ver_tip_margin1  # Local do topo % de b_v
    hist_c_v = 0.32

    cr_v_loc = ver_base_margin * (ct_v - cr_v) + cr_v
    ct_v_loc = ver_tip_margin * (ct_v - cr_v) + cr_v

    cr_v2 = cr_v_loc * hist_c_v
    ct_v2 = ct_v_loc * hist_c_v

    xr_v2 = ver_base_margin * (xt_v - xr_v) + xr_v + cr_v_loc * (1 - hist_c_v)
    xt_v2 = ver_tip_margin * (xt_v - xr_v) + xr_v + ct_v_loc * (1 - hist_c_v)

    zr_v2 = ver_base_margin * (zt_v - zr_v) + zr_v
    zt_v2 = ver_tip_margin * (zt_v - zr_v) + zr_v

    ax.plot(
        [xr_v2, xt_v2, xt_v2 + ct_v2, xr_v2 + cr_v2, xr_v2],
        [
            tcr_v * cr_v_loc / 2,
            tct_v * ct_v_loc / 2,
            tct_v * ct_v_loc / 2,
            tcr_v * cr_v_loc / 2,
            tcr_v * cr_v_loc / 2,
        ],
        [zr_v2, zt_v2, zt_v2, zr_v2, zr_v2],
        color="#1CB794",
    )

    ax.plot(
        [xr_v2, xt_v2, xt_v2 + ct_v2, xr_v2 + cr_v2, xr_v2],
        [
            -tcr_v * cr_v_loc / 2,
            -tct_v * ct_v_loc / 2,
            -tct_v * ct_v_loc / 2,
            -tcr_v * cr_v_loc / 2,
            -tcr_v * cr_v_loc / 2,
        ],
        [zr_v2, zt_v2, zt_v2, zr_v2, zr_v2],
        color="#1CB794",
    )

    # _______ONLY FRONT VIEW_______

    # Wing Lower
    # ------------------------------
    ax.plot(
        [xr_w, xt_w, xt_w + ct_w, xr_w + cr_w, xt_w + ct_w, xt_w, xr_w],
        [0.0, yt_w, yt_w, 0.0, -yt_w, -yt_w, 0.0],
        [
            zr_w - tcr_w * cr_w / 2,
            zt_w - tct_w * ct_w / 2,
            zt_w - tct_w * ct_w / 2,
            zr_w - tcr_w * cr_w / 2,
            zt_w - tct_w * ct_w / 2,
            zt_w - tct_w * ct_w / 2,
            zr_w - tcr_w * cr_w / 2,
        ],
        color="#0046AB",
    )

    ax.plot(
        [xr_w, xr_w],
        [0.0, 0.0],
        [zr_w - tcr_w * cr_w / 2, zr_w + tcr_w * cr_w / 2],
        color="#0046AB",
    )
    ax.plot(
        [xr_w + cr_w, xr_w + cr_w],
        [0.0, 0.0],
        [zr_w - tcr_w * cr_w / 2, zr_w + tcr_w * cr_w / 2],
        color="#0046AB",
    )

    ax.plot(
        [xt_w, xt_w],
        [yt_w, yt_w],
        [zt_w - tct_w * ct_w / 2, zt_w + tct_w * ct_w / 2],
        color="#0046AB",
    )
    ax.plot(
        [xt_w + ct_w, xt_w + ct_w],
        [yt_w, yt_w],
        [zt_w - tct_w * ct_w / 2, zt_w + tct_w * ct_w / 2],
        color="#0046AB",
    )

    ax.plot(
        [xt_w, xt_w],
        [-yt_w, -yt_w],
        [zt_w - tct_w * ct_w / 2, zt_w + tct_w * ct_w / 2],
        color="#0046AB",
    )
    ax.plot(
        [xt_w + ct_w, xt_w + ct_w],
        [-yt_w, -yt_w],
        [zt_w - tct_w * ct_w / 2, zt_w + tct_w * ct_w / 2],
        color="#0046AB",
    )

    # -------------------------------------------------------
    # BOX WING PLOT: front wing + rear wing + tip-join panels
    if airplane.get("box_wing", False):

        # ---------------------
        # Read front wing (wf)
        xr_wf = airplane["xr_wf"]
        zr_wf = airplane["zr_wf"]
        cr_wf = airplane["cr_wf"]
        ct_wf = airplane["ct_wf"]
        xt_wf = airplane["xt_wf"]
        yt_wf = airplane["yt_wf"]
        zt_wf = airplane["zt_wf"]
        tcr_wf = airplane["tcr_wf"]
        tct_wf = airplane["tct_wf"]

        # Read rear wing (wr)
        xr_wr = airplane["xr_wr"]
        zr_wr = airplane["zr_wr"]
        cr_wr = airplane["cr_wr"]
        ct_wr = airplane["ct_wr"]
        xt_wr = airplane["xt_wr"]
        yt_wr = airplane["yt_wr"]
        zt_wr = airplane["zt_wr"]
        tcr_wr = airplane["tcr_wr"]
        tct_wr = airplane["tct_wr"]

        # Draw wings
        plot_trap_surface(ax, xr_wf, zr_wf, xt_wf, yt_wf, zt_wf, cr_wf, ct_wf, tcr_wf, tct_wf, color="blue", lw=1.5)
        plot_trap_surface(ax, xr_wr, zr_wr, xt_wr, yt_wr, zt_wr, cr_wr, ct_wr, tcr_wr, tct_wr, color="navy", lw=1.5)

        # ---------------------
        # Tip-join panels (right and left)
        # We'll connect the tip LE/TE of front wing to tip LE/TE of rear wing
        # Panel thickness: use local tip thickness (ct*tct/2), scaled by optional factor
        tf = airplane.get("box_join_thickness_factor", 1.0)

        th_f = tf * (ct_wf * tct_wf / 2.0)
        th_r = tf * (ct_wr * tct_wr / 2.0)

        # Helper to plot a quadrilateral "panel" as outline, upper and lower
        def plot_join_panel(x1_le, x1_te, y1, z1, th1,
                            x2_le, x2_te, y2, z2, th2,
                            color="blue", lw=1.5):

            # Upper perimeter
            ax.plot(
                [x1_le, x1_te, x2_te, x2_le, x1_le],
                [y1,    y1,    y2,    y2,    y1],
                [z1+th1, z1+th1, z2+th2, z2+th2, z1+th1],
                color=color, lw=lw
            )
            # Lower perimeter
            ax.plot(
                [x1_le, x1_te, x2_te, x2_le, x1_le],
                [y1,    y1,    y2,    y2,    y1],
                [z1-th1, z1-th1, z2-th2, z2-th2, z1-th1],
                color=color, lw=lw
            )

            # “Thickness edges” (connect upper to lower at corners)
            for (x, y, zup, zlo) in [
                (x1_le, y1, z1+th1, z1-th1),
                (x1_te, y1, z1+th1, z1-th1),
                (x2_le, y2, z2+th2, z2-th2),
                (x2_te, y2, z2+th2, z2-th2),
            ]:
                ax.plot([x, x], [y, y], [zlo, zup], color=color, lw=lw)

        # Coordinates at tips (mid-surface)
        # Front tip LE/TE
        wf_le = xt_wf
        wf_te = xt_wf + ct_wf
        # Rear tip LE/TE
        wr_le = xt_wr
        wr_te = xt_wr + ct_wr

        # Right side join (positive y)
        plot_join_panel(
            wf_le, wf_te, +yt_wf, zt_wf, th_f,
            wr_le, wr_te, +yt_wr, zt_wr, th_r,
            color="blue", lw=1.5
        )

        # Left side join (negative y)
        plot_join_panel(
            wf_le, wf_te, -yt_wf, zt_wf, th_f,
            wr_le, wr_te, -yt_wr, zt_wr, th_r,
            color="blue", lw=1.5
        )

    # ------------------------------
    # Canard Lower
    # ------------------------------

    if has_canard:
        ax.plot(
            [xr_c, xt_c, xt_c + ct_c, xr_c + cr_c, xt_c + ct_c, xt_c, xr_c],
            [0.0, yt_c, yt_c, 0.0, -yt_c, -yt_c, 0.0],
            [
                zr_c - cr_c * tcr_c / 2,
                zt_c - ct_c * tct_c / 2,
                zt_c - ct_c * tct_c / 2,
                zr_c - cr_c * tcr_c / 2,
                zt_c - ct_c * tct_c / 2,
                zt_c - ct_c * tct_c / 2,
                zr_c - cr_c * tcr_c / 2,
            ],
            color="#0046AB",
        )

    # CANARD - Upper surface
    if has_canard:
        ax.plot(
            [xr_c, xt_c, xt_c + ct_c, xr_c + cr_c, xt_c + ct_c, xt_c, xr_c],
            [0.0, yt_c, yt_c, 0.0, -yt_c, -yt_c, 0.0],
            [
                zr_c + cr_c * tcr_c / 2,
                zt_c + ct_c * tct_c / 2,
                zt_c + ct_c * tct_c / 2,
                zr_c + cr_c * tcr_c / 2,
                zt_c + ct_c * tct_c / 2,
                zt_c + ct_c * tct_c / 2,
                zr_c + cr_c * tcr_c / 2,
            ],
            color="#0046AB",
        )

    # CANARD thickness edges
    if has_canard:
        # Root LE and TE verticals
        ax.plot([xr_c, xr_c], [0.0, 0.0],
                [zr_c - tcr_c * cr_c / 2, zr_c + tcr_c * cr_c / 2],
                color="#0046AB")
        ax.plot([xr_c + cr_c, xr_c + cr_c], [0.0, 0.0],
                [zr_c - tcr_c * cr_c / 2, zr_c + tcr_c * cr_c / 2],
                color="#0046AB")

        # Right tip LE and TE verticals
        ax.plot([xt_c, xt_c], [yt_c, yt_c],
                [zt_c - tct_c * ct_c / 2, zt_c + tct_c * ct_c / 2],
                color="#0046AB")
        ax.plot([xt_c + ct_c, xt_c + ct_c], [yt_c, yt_c],
                [zt_c - tct_c * ct_c / 2, zt_c + tct_c * ct_c / 2],
                color="#0046AB")

        # Left tip LE and TE verticals
        ax.plot([xt_c, xt_c], [-yt_c, -yt_c],
                [zt_c - tct_c * ct_c / 2, zt_c + tct_c * ct_c / 2],
                color="#0046AB")
        ax.plot([xt_c + ct_c, xt_c + ct_c], [-yt_c, -yt_c],
                [zt_c - tct_c * ct_c / 2, zt_c + tct_c * ct_c / 2],
                color="#0046AB")

    # ------------------------------
    # HT Lower
    # ------------------------------

    if has_HT:
        ax.plot(
            [xr_h, xt_h, xt_h + ct_h, xr_h + cr_h, xt_h + ct_h, xt_h, xr_h],
            [0.0, yt_h, yt_h, 0.0, -yt_h, -yt_h, 0.0],
            [
                zr_h - tcr_h * cr_h / 2,
                zt_h - tct_h * ct_h / 2,
                zt_h - tct_h * ct_h / 2,
                zr_h - tcr_h * cr_h / 2,
                zt_h - tct_h * ct_h / 2,
                zt_h - tct_h * ct_h / 2,
                zr_h - tcr_h * cr_h / 2,
            ],
            color="#0046AB",
        )

        ax.plot(
            [xr_h, xr_h],
            [0.0, 0.0],
            [zr_h - tcr_h * cr_h / 2, zr_h + tcr_h * cr_h / 2],
            color="#0046AB",
        )
        ax.plot(
            [xr_h + cr_h, xr_h + cr_h],
            [0.0, 0.0],
            [zr_h - tcr_h * cr_h / 2, zr_h + tcr_h * cr_h / 2],
            color="#0046AB",
        )

        ax.plot(
            [xt_h, xt_h],
            [yt_h, yt_h],
            [zt_h - tct_h * ct_h / 2, zt_h + tct_h * ct_h / 2],
            color="#0046AB",
        )
        ax.plot(
            [xt_h + ct_h, xt_h + ct_h],
            [yt_h, yt_h],
            [zt_h - tct_h * ct_h / 2, zt_h + tct_h * ct_h / 2],
            color="#0046AB",
        )

        ax.plot(
            [xt_h, xt_h],
            [-yt_h, -yt_h],
            [zt_h - tct_h * ct_h / 2, zt_h + tct_h * ct_h / 2],
            color="#0046AB",
        )
        ax.plot(
            [xt_h + ct_h, xt_h + ct_h],
            [-yt_h, -yt_h],
            [zt_h - tct_h * ct_h / 2, zt_h + tct_h * ct_h / 2],
            color="#0046AB",
        )

    # Slat Lower
    # ------------------------------
    if slat_type is not None:
        ax.plot(
            [xr_s, xt_s, xt_s + ct_s, xr_s + cr_s, xr_s],
            [yr_s, yt_s, yt_s, yr_s, yr_s],
            [
                zr_s - tcr_s * cr_s / 2 / c_slat_c_wing,
                zt_s - tct_s * ct_s / 2 / c_slat_c_wing,
                zt_s - tct_s * ct_s / 2 / c_slat_c_wing,
                zr_s - tcr_s * cr_s / 2 / c_slat_c_wing,
                zr_s - tcr_s * cr_s / 2 / c_slat_c_wing,
            ],
            lw=1,
            color="#1CB794",
        )

        ax.plot(
            [xr_s, xt_s, xt_s + ct_s, xr_s + cr_s, xr_s],
            [-yr_s, -yt_s, -yt_s, -yr_s, -yr_s],
            [
                zr_s - tcr_s * cr_s / 2 / c_slat_c_wing,
                zt_s - tct_s * ct_s / 2 / c_slat_c_wing,
                zt_s - tct_s * ct_s / 2 / c_slat_c_wing,
                zr_s - tcr_s * cr_s / 2 / c_slat_c_wing,
                zr_s - tcr_s * cr_s / 2 / c_slat_c_wing,
            ],
            lw=1,
            color="#1CB794",
        )
    # ------------------------------

    # Flap Lower
    # ------------------------------
    if flap_type is not None:
        ax.plot(
            [xr_f, xt_f, xt_f + ct_f, xr_f + cr_f, xr_f],
            [yr_f, yt_f, yt_f, yr_f, yr_f],
            [
                zr_f - tcr_f * cr_f / 2 / c_flap_c_wing,
                zt_f - tct_f * ct_f / 2 / c_flap_c_wing,
                zt_f - tct_f * ct_f / 2 / c_flap_c_wing,
                zr_f - tcr_f * cr_f / 2 / c_flap_c_wing,
                zr_f - tcr_f * cr_f / 2 / c_flap_c_wing,
            ],
            lw=1,
            color="#1CB794",
        )

        ax.plot(
            [xr_f, xt_f, xt_f + ct_f, xr_f + cr_f, xr_f],
            [-yr_f, -yt_f, -yt_f, -yr_f, -yr_f],
            [
                zr_f - tcr_f * cr_f / 2 / c_flap_c_wing,
                zt_f - tct_f * ct_f / 2 / c_flap_c_wing,
                zt_f - tct_f * ct_f / 2 / c_flap_c_wing,
                zr_f - tcr_f * cr_f / 2 / c_flap_c_wing,
                zr_f - tcr_f * cr_f / 2 / c_flap_c_wing,
            ],
            lw=1,
            color="#1CB794",
        )
    # ------------------------------

    # Aleron Lower
    # ------------------------------
    ax.plot(
        [xr_a, xt_a, xt_a + ct_a, xr_a + cr_a, xr_a],
        [yr_a, yt_a, yt_a, yr_a, yr_a],
        [
            zr_a - tcr_a * cr_a / 2 / c_ail_c_wing,
            zt_a - tct_a * ct_a / 2 / c_ail_c_wing,
            zt_a - tct_a * ct_a / 2 / c_ail_c_wing,
            zr_a - tcr_a * cr_a / 2 / c_ail_c_wing,
            zr_a - tcr_a * cr_a / 2 / c_ail_c_wing,
        ],
        lw=1,
        color="#1CB794",
    )

    ax.plot(
        [xr_a, xt_a, xt_a + ct_a, xr_a + cr_a, xr_a],
        [-yr_a, -yt_a, -yt_a, -yr_a, -yr_a],
        [
            zr_a - tcr_a * cr_a / 2 / c_ail_c_wing,
            zt_a - tct_a * ct_a / 2 / c_ail_c_wing,
            zt_a - tct_a * ct_a / 2 / c_ail_c_wing,
            zr_a - tcr_a * cr_a / 2 / c_ail_c_wing,
            zr_a - tcr_a * cr_a / 2 / c_ail_c_wing,
        ],
        lw=1,
        color="#1CB794",
    )
    # ------------------------------
    if auxlines==True:
        # Avoiding blanketing the rudder
        if has_HT:
            ax.plot(
                [xr_h, xr_h + b_v / np.tan(60 * np.pi / 180)],
                [0.0, 0.0],
                [zr_h, zr_h + b_v],
                "k--",
            )

            ax.plot(
                [xr_h + cr_h, xr_h + 0.6 * b_v / np.tan(30 * np.pi / 180) + cr_h],
                [0.0, 0.0],
                [zr_h, zr_h + 0.6 * b_v],
                "k--",
            )

        # Auxiliary landing gear lines
        if x_nlg is not None:
            # Water Spray
            ax.plot(
                [x_nlg, x_nlg + 0.25 * b_w / np.tan(22 * np.pi / 180)],
                [0.0, 0.25 * b_w],
                [z_nlg, z_nlg],
                "k--",
            )

            ax.plot(
                [x_nlg, x_nlg + 0.25 * b_w / np.tan(22 * np.pi / 180)],
                [0.0, -0.25 * b_w],
                [z_nlg, z_nlg],
                "k--",
            )

            # Tailstrike
            tailstrike_angle = np.arctan(
                (-D_f / 2 - z_mlg) / (x_tailstrike - x_mlg))
            ax.plot(
                [x_mlg, L_f],
                [0.0, 0.0],
                [z_mlg, z_mlg + (L_f - x_mlg) * np.tan(tailstrike_angle)],
                "k--",
            )

            ax.plot([x_mlg, L_f], [0.0, 0.0], [z_mlg, z_mlg], "k--")

    # Create cubic bounding box to simulate equal aspect ratio
    # First create o list of possible critical points along each coordinate
    if has_HT:
        X = np.array(
            [
                0,
                xr_w,
                xt_h + ct_h,
                xt_v + ct_v,
                L_f,
                xr_h + b_v / np.tan(60 * np.pi / 180),
                xr_h + 0.6 * b_v / np.tan(30 * np.pi / 180) + cr_h,
            ]
        )
        Y = np.array([-yt_w, yt_w])
        Z = np.array([-D_f / 2, zt_w, zt_h, zt_v,
                     z_mlg - d_lg / 2, zr_h + b_v])
        max_range = np.array(
            [X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]
        ).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
            X.max() + X.min()
        )
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
            Y.max() + Y.min()
        )
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
            Z.max() + Z.min()
        )

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], "w")

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(az1, az2)

    axis_dim = 18

    ax.set_xlim(0, 2*axis_dim)
    ax.set_ylim(-axis_dim, axis_dim)
    ax.set_zlim(-axis_dim, axis_dim)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    if save:
        # Caminho da pasta onde está este script
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Caminho completo da subpasta "imagens"
        output_folder = os.path.join(base_dir, dirname)

        # Cria a pasta se não existir
        os.makedirs(output_folder, exist_ok=True)

        filename = os.path.join(output_folder, figname)

        fig.savefig(filename, dpi=300)

    plt.show()

