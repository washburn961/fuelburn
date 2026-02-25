# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 13:26:06 2026

@author: JRVASCON
"""


# IMPORTS
import designTool_merged as dt
import numpy as np
import pprint
import matplotlib.pylab as plt
from tabulate import tabulate
import math
import plotly.express as px
import plotly.io as pio
import io
from PIL import Image
import os
from collections import Counter


colors = {
    "blue":  "#0F6D84",
    "yellow": "#ffc800",
    "green": "#5c6c74"
}

# CONSTANTS
ft2m = 0.3048
kt2ms = 0.514444
lb2N = 4.44822
nm2m = 1852.0
gravity = 9.81
gamma_air = 1.4
R_air = 287

# ======================================================================
# DRAG PLOT
# Plot of drag componentes
def dragPlot(airplane,mach_range, plot=True, save=False):
    
    """
    Inputs:
        airplane: aircraft to be analyzed
        mach_range: Mach number range to calculate drag
        plot: if True -> plot graphs
        save: if True -> save graphs
        
    Output: 
        Plot of drag as a funtion of mach number
    """
    
    airplane_copy = dt.standard_airplane(airplane['name'])
    dt.analyze(airplane_copy)

    altitude = airplane_copy['altitude_cruise']
    M_c = airplane_copy['Mach_cruise']

    # -------------------------------
    # Parameters
    Mach = mach_range
    [T, p, rho, mi] = dt.atmosphere(altitude)
    a = np.sqrt(1.4*R_air*T)

    V = a*Mach

    #Cruise Condition
    V_cruise = a*airplane_copy['Mach_cruise']

    if "W_cruise" in airplane_copy.keys():
        W_cruise = airplane_copy['W_cruise']
    else:
        W_cruise = airplane_copy['W0_guess']*0.99*0.99*0.995*0.98

    CL_cruise = W_cruise/(0.5*rho*(V_cruise**2)*airplane_copy['S_w'])

    CD_cruise, _, dragDict_cruise = dt.aerodynamics(
        airplane_copy, airplane_copy['Mach_cruise'], altitude, CL_cruise, W_cruise)

    CD_cruise_list = np.zeros(np.size(Mach))  # CD Total
    CDind_clean_cruise_list = np.zeros(np.size(Mach))  # CD Indusido
    CDwave_cruise_list = np.zeros(np.size(Mach))  # CD Wave
    CD0_cruise_list = np.zeros(np.size(Mach))  # CD Parasita

    D0_cruise = np.zeros(np.size(Mach))  # Arrasto parasia
    Dind_cruise = np.zeros(np.size(Mach))  # Arrasto induzido
    Dwave_cruise = np.zeros(np.size(Mach))  # Arrasto de onda

    D = np.zeros(np.size(Mach))  # Arrasto total

    CL_cruise_vec = W_cruise/(0.5*rho*(V**2)*airplane_copy['S_w'])

    for ii in range(len(Mach)):
        Mach_airp = Mach[ii]
        # Recalcula o CL para manter o cruzeiro
        CL = CL_cruise_vec[ii]#*(M_c**2)/(Mach_airp**2)
        CD, _, dragDict = dt.aerodynamics(
            airplane_copy, Mach_airp, altitude, CL, W_cruise)

        CD_cruise_list[ii] = CD
        CDwave_cruise_list[ii] = dragDict['CDwave']
        CDind_clean_cruise_list[ii] = dragDict['CDind_clean']  # .item()
        CD0_cruise_list[ii] = dragDict['CD0']

        Q = rho*(V[ii]**2)*airplane_copy['S_w']/2
        D0_cruise[ii] = CD0_cruise_list[ii]*Q
        Dind_cruise[ii] = CDind_clean_cruise_list[ii]*Q
        Dwave_cruise[ii] = CDwave_cruise_list[ii]*Q

        D[ii] = D0_cruise[ii]+Dind_cruise[ii]+Dwave_cruise[ii]

    if plot:
        # ------------ PLOTS -------------------
        # Drag coeff.
        plt.figure()
        plt.plot(Mach, CD0_cruise_list/(10**(-4)),
                 label='CD - Parasite', color='lime')
        plt.plot(Mach, CDind_clean_cruise_list/(10**(-4)),
                 label='CD - Induced', color='blue')
        plt.plot(Mach, CDwave_cruise_list/(10**(-4)),
                 label='CD - Wave', color='fuchsia')
        plt.plot(Mach, CD_cruise_list/(10**(-4)),
                 label='CD - Total', color='red')
        plt.vlines(M_c, 0, max(CD_cruise_list/(10**(-4))),
                   linestyles='--', colors='black', label='Cruise Mach')
        plt.xlabel("Mach [-]", {'fontname': 'Times New Roman'}, fontsize=12)
        plt.ylabel("CD [counts]", {'fontname': 'Times New Roman'}, fontsize=12)
        plt.title(f"Drag Breakdown x Mach Number - {airplane_copy['name']}", {
                  'fontname': 'Times New Roman'}, fontsize=14)
        plt.xlim(Mach[0], Mach[-1])
        #plt.ylim(0, 700)

        plt.grid(True)
        plt.legend(fontsize=8)

    if save:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Caminho completo da subpasta "imagens"
        output_folder = os.path.join(base_dir, "imagens/aerodinamica")

        # Cria a pasta se não existir
        os.makedirs(output_folder, exist_ok=True)

        filename = os.path.join(output_folder, "dragplot.pdf")
        plt.savefig(filename)

    plt.show()
# ======================================================================


# ======================================================================
# DRAG POLAR
# Plot of aricraft drag polar

def dragPolar(airplane, condition, plot = False):
    
    """
    Inputs:
        airplane: aircraft to be analyzed
        condition: flight condition to be analyzed
                    - cruise_subsonic: Clean condition with Mach number of 0.4
                    - cruise_transsonic: Clean condition with Mach number greater 0.4
                    - cruise_hightranssonic: Clean condition with Mach of 0.9 and FL400
                    - takeoff: takeoff condition with Mach number of 0.3
                    - landing: landing condition with Mach number of 0.3
    """
    
    airplane_copy = dt.standard_airplane(airplane['name'])
    dt.analyze(airplane_copy)
    
    if condition == "cruise_subsonic":  
        altitude = airplane_copy['altitude_cruise']
        Mach = 0.4
        if "W_cruise" in airplane_copy.keys():
            W = airplane_copy['W_cruise']
        else:
            W = airplane_copy['W0_guess']*0.99*0.99*0.995*0.98
        highlift = "clean"
    elif condition == "cruise_transsonic":  
        altitude = airplane_copy['altitude_cruise']
        Mach = airplane_copy['Mach_cruise']
        if "W_cruise" in airplane_copy.keys():
            W = airplane_copy['W_cruise']
        else:
            W = airplane_copy['W0_guess']*0.99*0.99*0.995*0.98
        highlift = "clean"    
    elif condition == "cruise_hightranssonic":  
        altitude = 40000*ft2m
        Mach = 0.9
        if "W_cruise" in airplane_copy.keys():
            W = airplane_copy['W_cruise']
        else:
            W = airplane_copy['W0_guess']*0.99*0.99*0.995*0.98
        highlift = "clean"
    elif condition == "takeoff":
        altitude = 0
        Mach = 0.3
        if "W0" in airplane_copy.keys():
            W = airplane_copy['W0']
        else:
            W = airplane_copy['W0_guess']
        highlift = "takeoff"
    elif condition == "landing":
        altitude = 0
        Mach = 0.3
        if "W0" in airplane_copy.keys():
            W = airplane_copy['W0']*airplane_copy['MLW_frac']
        else:
            W = airplane_copy['W0_guess']*airplane_copy['MLW_frac']
        highlift = "landing"
    
    # ---------------------------------------
    # 1- Polar de Arrasto
    [T, p, rho, mi] = dt.atmosphere(altitude)
    a = np.sqrt(1.4*R_air*T)
    V = Mach*a
    
    # ---------------------------------------
    # 5- Polar de Arrasto
    CL = W/(0.5*rho*(V**2)*airplane_copy['S_w'])

    CD,CLmax,_ = dt.aerodynamics(airplane_copy, Mach, altitude, CL, W, highlift_config=highlift)
    L_D_opt = CL/CD
    # ---------------------------------------
    CL_vec = np.linspace(-0.5, CLmax, 50)
    CD_vec = np.zeros(np.size(CL_vec))

    
    
    for i in range(0, np.size(CL_vec)):
        CD_vec[i],_,_ = dt.aerodynamics(airplane_copy, Mach, altitude, CL_vec[i], W, highlift_config=highlift)
        
    L_D_max = max(CL_vec/CD_vec)
    i_LDmax = np.argmax((CL_vec/CD_vec))
    
    if plot:
        plt.figure()
        plt.plot(CD_vec, CL_vec, 'b-')
        plt.text(max(CD_vec)*0.9, max(CL_vec)
                 * 1.05, f'CL_max = {max(CL_vec):.2f}', color='blue')
        plt.plot(CD_vec[i_LDmax],
                 CL_vec[i_LDmax],"o",color = colors['blue'])
        plt.plot(CD_vec[-1], CL_vec[-1], 'yx')
        plt.plot(CD,CL, 'ys')
        
        #plt.legend(["Cruise", "Takeoff", "Landing"])
        plt.xlabel("CD [-]", {'fontname': 'Times New Roman'})
        plt.ylabel("CL [-]", {'fontname': 'Times New Roman'})
        plt.title(
            f"Drag Polar - {airplane_copy['name']}", {'fontname': 'Times New Roman'})
        plt.text(CD_vec[i_LDmax]*1.1, CL_vec[i_LDmax]*1, f'L/D_max = {L_D_max:.2f}', color='blue')
        plt.text(CD*1.1, CL*1, f'L/D_oper = {L_D_opt:.2f}', color='blue')
        plt.grid()
        plt.ylim([-0.5, max(CL_vec)*1.2])
        plt.xlim([0, max(CD_vec)*1.2])


        plt.show()
    
    
    
    return CL_vec, CD_vec, L_D_opt, L_D_max

# ======================================================================

def dragPizza(airplane, condition, colors = None):
    
    airplane_copy = dt.standard_airplane(airplane['name'])
    dt.analyze(airplane_copy)
    
    if colors is None:
        colors = ['#1f4fff', '#308dff', '#4fbeff']
    
    if condition == "cruise_subsonic":  
        altitude = airplane_copy['altitude_cruise']
        Mach = 0.4
        if "W_cruise" in airplane_copy.keys():
            W = airplane_copy['W_cruise']
        else:
            W = airplane_copy['W0_guess']*0.99*0.99*0.995*0.98
        highlift = "clean"
    elif condition == "cruise_transsonic":  
        altitude = airplane_copy['altitude_cruise']
        Mach = airplane_copy['Mach_cruise']
        if "W_cruise" in airplane_copy.keys():
            W = airplane_copy['W_cruise']
        else:
            W = airplane_copy['W0_guess']*0.99*0.99*0.995*0.98
        highlift = "clean"    
    elif condition == "cruise_hightranssonic":  
        altitude = 40000*ft2m
        Mach = 0.9
        if "W_cruise" in airplane_copy.keys():
            W = airplane_copy['W_cruise']
        else:
            W = airplane_copy['W0_guess']*0.99*0.99*0.995*0.98
        highlift = "clean"
    elif condition == "takeoff":
        altitude = 0
        Mach = 0.3
        if "W0" in airplane_copy.keys():
            W = airplane_copy['W0']
        else:
            W = airplane_copy['W0_guess']
        highlift = "takeoff"
    elif condition == "landing":
        altitude = 0
        Mach = 0.3
        if "W0" in airplane_copy.keys():
            W = airplane_copy['W0']*airplane_copy['MLW_frac']
        else:
            W = airplane_copy['W0_guess']*airplane_copy['MLW_frac']
        highlift = "landing"
   

    # ---------------------------------------
    # 1- Polar de Arrasto
    [T, p, rho, mi] = dt.atmosphere(altitude)
    a = np.sqrt(1.4*286*T)
    V = Mach*a
    
    # ---------------------------------------
    # 5- Polar de Arrasto
    CL = W/(0.5*rho*(V**2)*airplane_copy['S_w'])
    CD,CLmax,dragDict = dt.aerodynamics(airplane_copy, Mach, altitude, CL, W, highlift_config=highlift)
    
    values = np.array([dragDict['CD0'],dragDict['CDind_clean'], dragDict['CDwave']])
    sizes = values/dragDict['CD']  # The values for each slice
    labels = ['Parasite', 'Induced', 'Wave']  # The labels for each slice
    #explode = (0, 0.1, 0, 0)  # Offset the second slice (Hogs) slightly
    
    def autopct_format(valores):
        total = sum(valores)
        def _inner(pct):
            # valor absoluto arredondado ao inteiro mais próximo
            valor_abs = int(round(pct * total / 100.0))
            # Formatação: 1) percentual com 1 casa, 2) valor com separador de milhar
            return f'{pct:.1f}%\n({valor_abs:,} counts)'.replace(',', '.')
        return _inner
    
    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors,
           autopct=autopct_format(values*1e4), shadow=False, startangle=90, labeldistance=1.05, textprops={'fontfamily': 'Times New Roman','fontsize': 12})
    ax.axis('equal')  # Ensures the pie chart is drawn as a perfect circle
    
    plt.title(f"Drag breakdown - {airplane_copy['name']} ", {'fontname': 'Times New Roman'})
    plt.legend(labels, loc="upper left", prop={'family': 'Times New Roman','size': 12}) # Add a legend
    plt.show()

# ======================================================================
# WHEIGHTS FRACTION
def weightsFraction(airplane, colors = None ,plot=True):

    # airplane = dt.standard_airplane(airplane['name'])
    
    if colors is None:
        colors = ['#001972', '#efa30b', '#5c6c74']

    dt.analyze(airplane, print_log=False, plot=False)
 
    gravity = 9.81 # [m/s^2] - Gravity acceleration    
 
    values = [

        airplane['W_payload']/gravity,

        airplane['W_fuel']/gravity,

        (airplane['W_empty'] - airplane['W_crew']) /gravity,

    ]

 
    labels = [

        "Payload", "Fuel", "OEW"

    ]
 
 
    sizes = values/airplane['W0']  # The values for each slice

    #explode = (0, 0.1, 0, 0)  # Offset the second slice (Hogs) slightly

    def autopct_format(valores):

        total = sum(valores)

        def _inner(pct):

            # valor absoluto arredondado ao inteiro mais próximo

            valor_abs = int(round(pct * total / 100.0))

            # Formatação: 1) percentual com 1 casa, 2) valor com separador de milhar

            return f'{pct:.1f}%\n({valor_abs:,} [kg])'.replace(',', '.')

        return _inner

    # Create the pie chart

    fig, ax = plt.subplots()

    ax.pie(sizes, labels=labels, colors=colors,

           autopct=autopct_format(values), shadow=False, startangle=90, labeldistance=1.05, textprops={'fontfamily': 'Times New Roman','fontsize': 12,'color': "w"})

    ax.axis('equal')  # Ensures the pie chart is drawn as a perfect circle

    plt.title(f"Weight Breakdown - {airplane['name']} ", {'fontname': 'Times New Roman'})

    plt.legend(labels, loc="upper left", prop={'family': 'Times New Roman','size': 12}) # Add a legend

    plt.show()
 
# ======================================================================


# ====================================================================
def restrictionDiagram(airplane, plot=True):
    # airplane = dt.standard_airplane(airplane['name'])
    # dt.analyze(airplane, print_log=False, plot=False)
    # Variables inialization
    Sw_vec = np.linspace(40, 100, 100)
    W0_Sw_airplane = np.zeros(np.size(Sw_vec))
    T0_airplane = np.zeros(np.size(Sw_vec))
    T0vec_airplane = np.zeros((np.size(Sw_vec), 10))
    deltaS_wlan_airplane = np.zeros(np.size(Sw_vec))
    CLmaxTO_airplane = np.zeros(np.size(Sw_vec))

    W_S_airplane = np.zeros(np.size(Sw_vec))
    T_W_airplane = np.zeros((np.size(Sw_vec), 10))

    CD_airplane = np.zeros(np.size(Sw_vec))

    deltaS_wlan_airplane_min = 100

    # T_W_req = airplane["T_W_req"] 

    # Tce_Wce = airplane["Tce_Wce"] 

    # Initial guess for Weight and Thrust
    W0_guess_airplane = airplane["W0_guess"]
    T0_guess_airplane = 0.3*W0_guess_airplane

    airplane2 = airplane.copy()

    for i in range(0, np.size(Sw_vec)):
        airplane2['S_w'] = Sw_vec[i]
        dt.geometry(airplane2)

        dt.design(airplane2)

        W0_Sw_airplane[i] = airplane2["W0"]

        deltaS_wlan_airplane[i] = airplane2["deltaS_wlan"]

        T0vec_airplane[i, :] = airplane2["T0vec"]

        W_S_airplane[i] = W0_Sw_airplane[i]/Sw_vec[i]* 0.10197162129779

        T_W_airplane[i, :] = T0vec_airplane[i, :]/W0_Sw_airplane[i]

        if abs(deltaS_wlan_airplane[i]) < abs(deltaS_wlan_airplane_min):
            deltaS_wlan_airplane_min = deltaS_wlan_airplane[i]
            deltaS_wlan_airplane_min_index = i


    # ===========================================================

    S1_interp = np.linspace(40, 100, 10000)
    T1_interp = np.transpose([np.interp(
        S1_interp, Sw_vec, T0vec_airplane[:, i]) for i in range(len(T0vec_airplane[0]))])
    W1_interp = np.interp(S1_interp, Sw_vec, W0_Sw_airplane)
    T1_border = np.nanmax(T1_interp, axis=1)
    T1_border[S1_interp < Sw_vec[deltaS_wlan_airplane_min_index]] = math.inf
    opt_ind = np.argmin(T1_border)
    opt_S1 = S1_interp[opt_ind]
    opt_T1 = T1_border[opt_ind]
    opt_W1 = W1_interp[opt_ind]
    opt_W_S1 = opt_W1/opt_S1* 0.10197162129779
    opt_T_W1 = opt_T1/opt_W1

    aircraft = ["ERJ145-ER", "ERJ145-XR", "CRJ200"]
    aircraftTP = ["Saab 2000", "ATP", "DHC8-300",
                  "DHC8-400", "ATR 42-600", "Fokker 50"]
    # =========================
    # 1) CORES DEFINIDAS POR NOME
    # =========================
    aircraft_color_map = {
        "ERJ145-ER": "#1CB794",   # azul
        "ERJ145-XR": "#3FE1BE",   # laranja
        "CRJ200":       "#FFC800",   # verde
        "CRJ550":       "#420951",   # vermelho
        "E175 Plus":    "#9467bd",   # roxo
    }
    default_color = "#333333"  # se aparecer alguma aeronave fora do mapa

    Sw_compar = np.array([51.18, 51.18, 54.5])  # , 70.6, 72.72])  # [m²]
    T_compar = np.array([14852, 15900, 17458]) * \
        0.0044482216*1000  # , 25340, 28400]
    W0_compar = np.array([20600, 24100, 21523])*9.81  # , 29484, 38600])*9.815
    
    name1 = airplane['name']

    W_S_compar = None
    T_W_compar = None

    if name1 in aircraft:
        i = aircraft.index(name1)
        Sw_point = Sw_compar[i]
        T_point  = T_compar[i]
        W0_point = W0_compar[i]
        # W/S em [kPa] se 0.10197... for conversão para kPa (N/m² -> kPa)
        W_S_compar = (W0_point / Sw_point) * 0.10197162129779
        T_W_compar = T_point / W0_point  # adimensional
    else:
        # não faz nada: sem ponto comparativo para essa aeronave
        pass

  
    # ===========================================================
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(Sw_vec, T0vec_airplane, label='airplane', linewidth=2)
        plt.axvline(x=Sw_vec[deltaS_wlan_airplane_min_index],
                    color='r', linestyle='--', linewidth=2)
        # plt.fill_between(Sw_vec[deltaS_wlan_airplane_min_index:], 275000, np.maximum(T0vec_airplane[deltaS_wlan_airplane_min_index:,0], T0vec_airplane[deltaS_wlan_airplane_min_index:,1]), color='lightgreen',alpha=0.3)
        plt.fill_between(S1_interp, 275000, T1_border,
                         color='lightgreen', alpha=0.3)
        # plt.axhline(y=airplane['engine']['T_eng_spec']*2,color='k', linestyle='--', linewidth=1)

        plt.xlabel('Wing Area (m²)', {
            'fontname': 'Times New Roman'}, fontsize=12)
        plt.ylabel('Required Thrust (N)', {
            'fontname': 'Times New Roman'}, fontsize=12)
        plt.title(f"Required Thrust vs Wing Area - {airplane['name']}", {
                  'fontname': 'Times New Roman'}, fontsize=16)

        for i, name in enumerate(aircraft):
            plt.scatter(
                Sw_compar[i], T_compar[i],
                color=aircraft_color_map.get(name, default_color),
                s=80, label=name
            )
        plt.plot(airplane['S_w'], (airplane['engine']
                                   ['T_eng_spec'])*2, 'rx', linewidth=4)
        # plt.text(airplane['S_w'], (airplane['engine']['T_eng_spec'])*2*1.1,
        #          f"{airplane['name']}", horizontalalignment='center', verticalalignment='top', color='red')

        plt.legend(["Takeoff", "Cruise", "Climb-FAR25111", "Climb-FAR25121a", "Climb-FAR25121b",
                    # , bbox_to_anchor=(1, 1))
                    "Climb-FAR25121c", "Climb-FAR25119", "Climb-FAR25121d", "Time-to-climb", "Minimum Sw - Landing", "Feasible Area", "ERJ145 ER", "ERJ145 LR", "CRJ200", f"{airplane['name']} (calculated)"], loc='upper right')

        plt.xlim(45, 65)
        plt.ylim(30000, 100000)
        plt.grid()

        plt.scatter(opt_S1, opt_T1, s=80, color="green", zorder=2)
        plt.text(opt_S1, opt_T1*0.95, "Optimal Point\n(S_w = "+str(round(opt_S1, ndigits=2))+", T = " + str(round(opt_T1)) + ")",
                 fontsize=10, horizontalalignment='center', verticalalignment='top', color='green')

        plt.show()

    # ====================================================================
    # Questão 02

    fg1_S, fg1_T, fg1_W = [84.46, 148000, 44893*9.81]  # First guess
    fg2_S, fg2_T, fg2_W = [84.46, 152000, 46000*9.81]

    Sw_compar_TP = np.array([55.7, 78.32, 50.96, 62.5, 54.5, 70])  # [m²]
    T_compar_TP = np.array([6192.2, 4102, 3208, 7562, 4102, 3728])*1000  # [W]
    W0_compar_TP = np.array([22999, 22930, 18640, 27987, 18600, 19990])*9.815
    W_S_compar_TP = (W0_compar_TP/Sw_compar_TP)* 0.10197162129779
    T_W_compar_TP = (T_compar_TP/W0_compar_TP)*0.00597


    # Cores diferentes para cada aeronave
    colors = plt.cm.tab10(range(len(aircraft)))  # paleta com 10 cores
    coordenadas = [(124, 244*1e3), (73, 115*1e3), (73, 134*1e3), (94, 166*1e3),
                   (104, 210*1e3), (63, 127*1e3), (79, 127*1e3), (96, 130*1e3)]


    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))

        # =========================
        # 2) NOMES DAS RESTRIÇÕES
        # =========================
        restr_labels = [
            "Takeoff", "Cruise","Ceiling", "Climb-FAR25111", "Climb-FAR25121a",
            "Climb-FAR25121b", "Climb-FAR25121c", "Climb-FAR25119", "Climb-FAR25121d", "Time-to-climb (1 min, ISA0)"
        ]


        climb_indices = [
            j for j, lab in enumerate(restr_labels)
            if lab.startswith("Climb")
        ]

        non_climb_indices = [
            j for j, lab in enumerate(restr_labels)
            if j not in climb_indices
        ]

        T_W_climb = T_W_airplane[:, climb_indices]  

        T_W_climb_max = np.max(T_W_climb, axis=1)

        idx_critical = np.argmax(T_W_climb, axis=1)

        labels_active = [
            restr_labels[climb_indices[i]] for i in idx_critical
        ]

        most_common = Counter(labels_active).most_common(1)[0][0]


        for j in non_climb_indices:

            if restr_labels[j] == "Takeoff":
                color = "#0F6D84"
            elif restr_labels[j] == "Cruise":
                color = "#FFC800"
            elif restr_labels[j] == "Ceiling":
                color = "#3FE1BE"
            elif restr_labels[j] == "Time-to-climb (1 min, ISA0)":
                color ="#0046AB"
            else:
                color = "#999999" 

            ax.plot(
                W_S_airplane,
                T_W_airplane[:, j],
                linewidth=2,
                label=restr_labels[j],
                color=color
            )

        ax.plot(
            W_S_airplane,
            T_W_climb_max,
            linewidth=2,
            color="#98D284",
            label=f"{most_common}"
        )

        # linha vertical
        ax.axvline(
            x=(W0_Sw_airplane[deltaS_wlan_airplane_min_index] /
               Sw_vec[deltaS_wlan_airplane_min_index] *  0.10197162129779),
            color="#1CB794", linestyle='--', linewidth=2,
            label="Minimum Sw - Landing"
        )

        # ax.axhline(
        #     y=Tce_Wce,
        #     color="#3FE1BE", linestyle='-', linewidth=2,
        #     label="Ceiling"
        # )

        # área viável
        ax.fill_between(
            W1_interp / S1_interp *  0.10197162129779, 0.8,
            T1_border / W1_interp,
            color='lightblue', alpha=0.3,
            label="Feasible Area"
        )

        # ponto da aeronave principal (calculado)
        main_ws =  0.10197162129779 * airplane['W0'] / airplane['S_w']
        main_tw = airplane['engine']['T_eng_spec'] * 2 / airplane['W0']

        c = aircraft_color_map.get(airplane['name'], default_color)

        # =========================
        # 4) AERONAVES DE COMPARAÇÃO (com cor escolhida por nome)
        # =========================
        
        def _is_number(x):
            import numpy as np
            return (x is not None) and (not (isinstance(x, float) and np.isnan(x))) and (not (hasattr(x, "__len__") and len(np.atleast_1d(x)) == 0))

        # Só plota se o índice existir e os valores forem válidos
        if (i is not None) and _is_number(W_S_compar) and _is_number(T_W_compar):
            ax.scatter(
                W_S_compar,
                T_W_compar,
                color=aircraft_color_map.get(name1, default_color),
                s=80,
                label=f"{name1} (reference)",
                zorder=5
            )


        # ponto ótimo
        ax.scatter(opt_W_S1, opt_T_W1, s=80, color="#A3B5BF", zorder=3,label=f"Opt. Point (W/S={opt_W_S1:.0f}, T/W={opt_T_W1:.2f})")

        # estética
        ax.set_xlabel('W/S (kg/m²)',
                      {'fontname': 'Times New Roman'}, fontsize=12)
        ax.set_ylabel('T/W (-)', {'fontname': 'Times New Roman'}, fontsize=12)
        ax.set_title(f"Restriction Diagram - {airplane['name']}",
                     {'fontname': 'Times New Roman'}, fontsize=16)

        ax.scatter(main_ws, main_tw, color="#001972", marker='x', s=100, zorder=4,
                   label=f"{airplane['name']} (calculated)")

        ax.set_xlim(320, 440)
        ax.set_ylim(top=0.5)
        ax.grid(True)

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.75)
        plt.show()



# Plot de Teste - Não é o certo [Julia]
def RestrictionDiagramPlot(airplane, plot=True, save=False):

    W0_Sw = np.linspace(0, 140, 100)

    # Takeoff

    distance_takeoff = airplane['distance_takeoff']

    CLmaxTO = 2  # Incerto
    sig = 1             # Nível do mar

    T0_W0 = (0.2387/(sig*CLmaxTO*distance_takeoff) *
             (W0_Sw*47.88))  # Documentação DesignTool

    # Landing
    distance_landing = airplane["distance_landing"]
    MLW_frac = airplane["MLW_frac"]

    g = 9.81
    Clmax = 2.15  # Incerto
    rho = 1.225
    hland = 15.3
    fland = 5/3
    ag = 0.5

    xlan = 1.52/ag + 1.59
    Alan = g/(fland*xlan)
    Blan = - 10 * g * hland/xlan

    W_S_land = 0.020885434273039*rho*Clmax * \
        (Alan*distance_landing+Blan)/(MLW_frac)  # Documentação do DesignTool

    # Cruise
    Cd0 = 0.01878776399953577  # Incerto
    K = 0.049767995214952176  # Incerto

    V = 230*0.514444                # Velocidade de cruzeiro
    q = 0.5*rho*V*V

    Tcr_Wcr = ((q*Cd0/(W0_Sw*47.88)) + ((W0_Sw*47.88)*K/q))  # Slide do Ney

    # Climb

    W0 = airplane["W0"]

    T0_W_climb_25111 = 44357.7/W0
    T0_W_climb_25121a = 47703.5/W0
    T0_W_climb_25121b = 51330.6/W0
    T0_W_climb_25121c = 37388.1/W0
    T0_W_climb_25119 = 35264/W0
    T0_W_climb_25121d = 49947/W0

    # TClimb_W = ks*ks*Cd0/CLmaxCL + CLmaxCL*K/(ks*ks) + G

    # Manouver
    Tmn_W = (q*Cd0/(W0_Sw*47.88)) + 2.5*2.5*((W0_Sw*47.88)*K/q)

    if plot:
        plt.figure()

        colors = {

            "takeoff":  "#1f77b4",
            "cruise":   "#ff7f0e",
            "manouver": "#f4f746",

            "25111":    "#2ca02c",
            "25121a":   "#d62728",
            "25121b":   "#9467bd",
            "25121c":   "#8c564b",
            "25119":    "#e377c2",
            "25121d":   "#7f7f7f",

            "landing":  "#ff0000",
            "feasible": "#98df8a",
        }

        plt.scatter([(20600*9.81/51.18)/47.88],
                    [14852/(0.2248*9.81*20600)],
                    label='ERJ-145', color='black', s=60, zorder=10)

        plt.plot(W0_Sw, T0_W0,   label='Takeoff',  color=colors["takeoff"])
        plt.plot(W0_Sw, Tcr_Wcr, label='Cruise',   color=colors["cruise"])
        plt.plot(W0_Sw, Tmn_W,   label='Manouver', color=colors["manouver"])

        xmin, xmax = plt.gca().get_xlim()

        plt.plot([xmin, xmax], [T0_W_climb_25111, T0_W_climb_25111],
                 label='Climb_25111', color=colors["25111"], linestyle='-', linewidth=2)

        plt.plot([xmin, xmax], [T0_W_climb_25121a, T0_W_climb_25121a],
                 label='Climb_25121a', color=colors["25121a"], linestyle='-', linewidth=2)

        plt.plot([xmin, xmax], [T0_W_climb_25121b, T0_W_climb_25121b],
                 label='Climb_25121b', color=colors["25121b"], linestyle='-', linewidth=2)

        plt.plot([xmin, xmax], [T0_W_climb_25121c, T0_W_climb_25121c],
                 label='Climb_25121c', color=colors["25121c"], linestyle='-', linewidth=2)

        plt.plot([xmin, xmax], [T0_W_climb_25119, T0_W_climb_25119],
                 label='Climb_25119', color=colors["25119"], linestyle='-', linewidth=2)

        plt.plot([xmin, xmax], [T0_W_climb_25121d, T0_W_climb_25121d],
                 label='Climb_25121d', color=colors["25121d"], linestyle='-', linewidth=2)

        ymin, ymax = plt.gca().get_ylim()
        plt.plot([W_S_land, W_S_land], [ymin, ymax],
                 label='Landing', color=colors["landing"], linestyle='--', linewidth=2)

        plt.ylim(0, 0.5)
        y_top = plt.gca().get_ylim()[1]

        curvas_tw = [
            T0_W0,
            Tcr_Wcr,
            Tmn_W,
            np.full_like(W0_Sw, T0_W_climb_25111),
            np.full_like(W0_Sw, T0_W_climb_25121a),
            np.full_like(W0_Sw, T0_W_climb_25121b),
            np.full_like(W0_Sw, T0_W_climb_25121c),
            np.full_like(W0_Sw, T0_W_climb_25119),
            np.full_like(W0_Sw, T0_W_climb_25121d),
        ]

        TW_req = np.maximum.reduce(curvas_tw)

        mask = (W0_Sw <= W_S_land)

        plt.fill_between(
            W0_Sw[mask],
            TW_req[mask],
            y_top,
            where=(y_top >= TW_req[mask]),
            facecolor=colors["feasible"],
            edgecolor='none',
            alpha=0.25,
            hatch=None,
            zorder=0,
            label='Feasible Area'
        )

        # =========================================================

        plt.xlabel("W/S [lb/ft^2]",
                   {'fontname': 'Times New Roman'}, fontsize=12)
        plt.ylabel("T/W [-]", {'fontname': 'Times New Roman'}, fontsize=12)
        plt.title(f"Diagrama de Restrição - {airplane['name']}",
                  {'fontname': 'Times New Roman'}, fontsize=14)

        plt.xlim([0, 140])

        plt.subplots_adjust(right=0.75)

        plt.grid(True)

        plt.legend(
            fontsize=8,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5)
        )

        if save:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_folder = os.path.join(base_dir, "imagens/aerodinamica")
            os.makedirs(output_folder, exist_ok=True)
            filename = os.path.join(output_folder, "teste.pdf")

            plt.savefig(filename, bbox_inches="tight")

        plt.show()

 

# ===========================================================================================================
# CG


def ballance(airplane, plot=True):
    # Execute the geometry function
    dt.geometry(airplane)

    # Guess values for initial iteration
    W0_guess = airplane['W0_guess']
    T0_guess = 0.3*W0_guess

    # Execute the weight and thrust estimation
    dt.thrust_matching(W0_guess, T0_guess, airplane)

    # Execute the balance analysis
    dt.balance(airplane)

    W1 = airplane['W_empty']/9.81
    W2 = airplane['W0']/9.81

    perc_np = (airplane['xnp'] - airplane['xm_w'])/airplane['cm_w']
    perc_mlg = (airplane['xcg_mlg'] - airplane['xm_w'])/airplane['cm_w']
    perc_aft_cg = (airplane['xcg_aft'] - airplane['xm_w'])/airplane['cm_w']
    perc_fwd_cg = (airplane['xcg_fwd'] - airplane['xm_w'])/airplane['cm_w']

    # Empty
    W_ept = airplane['W_empty']/9.81
    xcg_ept = (airplane['empty_weight']['xcg_empty'] -
               airplane['xm_w'])/airplane['cm_w']

    # BOW
    W_bow = (airplane['W_empty'] + airplane['W_crew'])/9.81
    xcg_bow = (airplane['W_empty']*airplane['empty_weight']['xcg_empty'] +
               airplane['W_crew']*airplane['xcg_crew'])/(airplane['W_empty'] + airplane['W_crew'])
    xcg_bow = (xcg_bow - airplane['xm_w'])/airplane['cm_w']

    # OW
    W_ow = (airplane['W_empty'] + airplane['W_crew'] + airplane['W_fuel'])/9.81
    xcg_ow = (airplane['W_empty']*airplane['empty_weight']['xcg_empty'] + airplane['W_crew']*airplane['xcg_crew'] +
              airplane['W_fuel']*airplane['xcg_fuel'])/(airplane['W_empty'] + airplane['W_crew'] + airplane['W_fuel'])
    xcg_ow = (xcg_ow - airplane['xm_w'])/airplane['cm_w']

    # MTOW
    W0 = (airplane['W_empty'] + airplane['W_crew'] +
          airplane['W_fuel'] + airplane['W_payload'])/9.81
    xcg_w0 = (airplane['W_empty']*airplane['empty_weight']['xcg_empty'] + airplane['W_crew']*airplane['xcg_crew'] + airplane['W_fuel']*airplane['xcg_fuel'] +
              airplane['W_payload']*airplane['xcg_payload'])/(airplane['W_empty'] + airplane['W_crew'] + airplane['W_fuel'] + airplane['W_payload'])
    xcg_w0 = (xcg_w0 - airplane['xm_w'])/airplane['cm_w']

    # MTOW - Fuel
    W0_mf = (airplane['W_empty'] + airplane['W_crew'] +
             airplane['W_payload'])/9.81
    xcg_w0mf = (airplane['W_empty']*airplane['empty_weight']['xcg_empty'] + airplane['W_crew']*airplane['xcg_crew'] +
                airplane['W_payload']*airplane['xcg_payload'])/(airplane['W_empty'] + airplane['W_crew'] + airplane['W_payload'])
    xcg_w0mf = (xcg_w0mf - airplane['xm_w'])/airplane['cm_w']

    if plot:
        plt.figure
        plt.vlines(x=perc_np, ymin=W1, ymax=W2, color="orange")
        plt.vlines(x=perc_mlg, ymin=W1, ymax=W2, color="blue")
        plt.vlines(x=perc_aft_cg, ymin=W1, ymax=W2,
                   color='green', linestyle='--', linewidth=2)
        plt.plot(xcg_ept, W_ept, 'go')
        plt.plot(xcg_bow, W_bow, 'bo')
        plt.plot(xcg_ow, W_ow, 'co')
        plt.plot(xcg_w0, W0, 'ro')
        plt.plot(xcg_w0mf, W0_mf, 'yo')
        plt.plot([xcg_ept, xcg_bow, xcg_ow, xcg_w0, xcg_w0mf, xcg_bow], [
                 W_ept, W_bow, W_ow, W0, W0_mf, W_bow], 'k-')
        plt.vlines(x=perc_fwd_cg, ymin=W1, ymax=W2,
                   color='green', linestyle='--', linewidth=2)
        plt.hlines(y=W1, xmin=perc_fwd_cg, xmax=perc_aft_cg,
                   color='green', linestyle='--', linewidth=2)
        plt.hlines(y=W2, xmin=perc_fwd_cg, xmax=perc_aft_cg,
                   color='green', linestyle='--', linewidth=2)
        plt.title(f"{airplane['name']}")
        plt.xlabel("Posição do CG [%cma]")
        plt.ylabel("Peso [kgf]")
        plt.legend(["Neutral Point", "Landing Gear", "Envelope",
                   "Empty Weight", "BOW", "OW", "MTOW", "MTOW - Fuel Weight"])
        plt.ylim([0.99*W1, W2*1.01])
        plt.xlim([0, 0.8])
        plt.grid()
        plt.show()
        # Print results

        # print(airplane['name'])
        # print(" airplane [ ' xcg_fwd '] = " , airplane['xcg_fwd'])
        # print(" airplane [ ' xcg_aft '] = " , airplane['xcg_aft'])
        # print(" airplane [ ' xnp '] = " , airplane['xnp'])
        # print(" airplane [ ' SM_fwd '] = " , airplane['SM_fwd'])
        # print(" airplane [ ' SM_aft '] = " , airplane['SM_aft'])
        # print(" airplane [ ' tank_excess '] = " , airplane['tank_excess'])
        # print(" airplane [ ' V_maxfuel '] = " , airplane['V_maxfuel'])
        # print(" airplane [ ' CLv '] = " , airplane['CLv'])
        # print("=======================================")


def CG_lopa(airplane, passageiros, crew, plot=False, img_path="path", comp=0, larg=0):
    # Função para calcular o Xcg, Ycg e W_paylaod dos passageiros

    # Separar coordenadas e massas
    x = np.array([p[0] for p in passageiros])
    y = np.array([p[1] for p in passageiros])
    m = np.array([p[2] for p in passageiros])

    x_c = crew[0]  # np.array([p[0] for p in crew])
    y_c = crew[1]  # np.array([p[1] for p in crew])

    x_cg_c = np.mean(x_c)
    y_cg_c = np.mean(y_c)

    # Calcular centro de gravidade
    x_cg = np.sum(x * m) / np.sum(m)
    y_cg = np.sum(y * m) / np.sum(m)
    W_payload = np.sum(m)*9.815

    if plot:
        # Carregar imagem
        img = plt.imread(img_path)

        # Dimensões da imagem em pixels
        altura_px, largura_px, _ = img.shape

        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 6))

        # Mostrar imagem como fundo
        ax.imshow(img, extent=[0, comp, -larg/2, larg/2])

        # Plotar passageiros
        # plt.figure()
        ax.scatter(x, y, c="red", s=15, label="Passageiros")
        ax.scatter(x_c, y_c, c="green", s=15, label="Tripulantes")

        # Plotar CG
        ax.scatter(x_cg, y_cg, c="red", marker="x", s=100,
                   label=f"CG - Pax ({x_cg:.2f}, {y_cg:.2f})")
        ax.scatter(x_cg_c, y_cg_c, c="green", marker="x", s=100,
                   label=f"CG - Tripulantes ({x_cg_c:.2f}, {y_cg_c:.2f})")
        # ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -1))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.7), ncol=4)

        # Configurações do gráfico
        ax.set_xlabel("Comprimento (m)", {
                      'fontname': 'Times New Roman'}, fontsize=12)
        ax.set_ylabel("Largura (m)", {
                      'fontname': 'Times New Roman'}, fontsize=12)
        ax.set_title(f"Distribuição dos passageiros e tripulantes - {airplane['name']}", {
                     'fontname': 'Times New Roman'}, fontsize=12)
        # ax.legend()
        ax.grid(True)
        # plt.savefig(f"{airplane['name']}_lopa.pdf", format="pdf")

        plt.show()

    return x_cg, y_cg, W_payload


def passageiros_cg(pax, n_rows, xr, yr1, dx, dy, yr2=0, mtd=1):
    # Função para calcular a distribuição de passageiros: x, y e peso
    passageiros = np.zeros([pax, 3])
    n_fileiras = int(pax/n_rows)

    if mtd == 1:
        if n_rows % 2 == 0:
            for i in range(0, int(n_fileiras)):
                x = xr + dx*(i)
                for j in range(0, n_rows):
                    k = j + n_rows*i
                    passageiros[k, 0] = x
                    passageiros[k, 2] = 100*9.81
                    y = yr1 + dy*j
                    if j >= n_rows*0.5:
                        y = -yr1 - dy*(j-int(n_rows*0.5))

                    passageiros[k, 1] = y

        if n_rows % 2:
            for i in range(0, int(n_fileiras)):
                x = xr + dx*(i)
                for j in range(0, n_rows):
                    k = j + n_rows*i
                    passageiros[k, 0] = x
                    passageiros[k, 2] = 100*9.81
                    y = yr1 + dy*j

                    if j > n_rows*0.5:
                        y = -yr2 - dy*(j-np.ceil(n_rows*0.5))

                    passageiros[k, 1] = y
    if mtd == 2:
        if n_rows % 2 == 0:
            for j in range(0, n_rows):
                for i in range(0, int(n_fileiras)):
                    x = xr + dx*(i)
                    k = i + n_fileiras*j
                    passageiros[k, 0] = x
                    passageiros[k, 2] = 100*9.81
                    y = yr1 + dy*j
                    if j >= n_rows*0.5:
                        y = -yr1 - dy*(j-int(n_rows*0.5))

                    passageiros[k, 1] = y
        if n_rows % 2:
            for j in range(0, n_rows):
                for i in range(0, int(n_fileiras)):
                    x = xr + dx*(i)
                    k = i + n_fileiras*j
                    passageiros[k, 0] = x
                    passageiros[k, 2] = 100*9.81
                    y = yr1 + dy*j

                    if j > n_rows*0.5:
                        y = -yr2 - dy*(j-np.ceil(n_rows*0.5))

                    passageiros[k, 1] = y

    return passageiros


def plot_balance(airplane, passageiros, plot=False):
    # # Execute the geometry function
    # dt.geometry( airplane )

    # # Guess values for initial iteration
    # W0_guess = airplane['W0_guess']
    # T0_guess = 0.3*W0_guess

    # # Execute the weight and thrust estimation
    # dt.thrust_matching( W0_guess , T0_guess , airplane )

    # # Execute the balance analysis
    # dt.balance( airplane )

    # dt.analyze(airplane)

    W1 = airplane['W_empty']/9.81
    W2 = airplane['W0']/9.81

    perc_np = (airplane['xnp'] - airplane['xm_w'])/airplane['cm_w']
    perc_mlg = (airplane['xcg_mlg'] - airplane['xm_w'])/airplane['cm_w']
    perc_aft_cg = (airplane['xcg_aft'] - airplane['xm_w'])/airplane['cm_w']
    perc_fwd_cg = (airplane['xcg_fwd'] - airplane['xm_w'])/airplane['cm_w']

    # Empty
    W_ept = airplane['W_empty']/9.81
    xcg_ept = airplane['empty_weight']['xcg_empty']
    xcg_ept_cma = (airplane['empty_weight']
                   ['xcg_empty'] - airplane['xm_w'])/airplane['cm_w']

    # BOW
    W_bow = (airplane['W_empty'] + airplane['W_crew'])/9.81
    xcg_bow = (airplane['W_empty']*airplane['empty_weight']['xcg_empty'] +
               airplane['W_crew']*airplane['xcg_crew'])/(airplane['W_empty'] + airplane['W_crew'])
    xcg_bow_cma = (xcg_bow - airplane['xm_w'])/airplane['cm_w']

    # OW
    W_ow = (airplane['W_empty'] + airplane['W_crew'] + airplane['W_fuel'])/9.81
    xcg_ow = (airplane['W_empty']*airplane['empty_weight']['xcg_empty'] + airplane['W_crew']*airplane['xcg_crew'] +
              airplane['W_fuel']*airplane['xcg_fuel'])/(airplane['W_empty'] + airplane['W_crew'] + airplane['W_fuel'])
    xcg_ow_cma = (xcg_ow - airplane['xm_w'])/airplane['cm_w']

    # MTOW
    W0 = (airplane['W_empty'] + airplane['W_crew'] +
          airplane['W_fuel'] + airplane['W_payload'])/9.81
    xcg_w0 = (airplane['W_empty']*airplane['empty_weight']['xcg_empty'] + airplane['W_crew']*airplane['xcg_crew'] + airplane['W_fuel']*airplane['xcg_fuel'] +
              airplane['W_payload']*airplane['xcg_payload'])/(airplane['W_empty'] + airplane['W_crew'] + airplane['W_fuel'] + airplane['W_payload'])
    xcg_w0_cma = (xcg_w0 - airplane['xm_w'])/airplane['cm_w']

    # MTOW - Fuel
    W0_mf = (airplane['W_empty'] + airplane['W_crew'] +
             airplane['W_payload'])/9.81
    xcg_w0mf = (airplane['W_empty']*airplane['empty_weight']['xcg_empty'] + airplane['W_crew']*airplane['xcg_crew'] +
                airplane['W_payload']*airplane['xcg_payload'])/(airplane['W_empty'] + airplane['W_crew'] + airplane['W_payload'])
    xcg_w0mf_cma = (xcg_w0mf - airplane['xm_w'])/airplane['cm_w']

    pax = np.size(passageiros[:, 0])

    MTOW = np.zeros(2*pax+4)
    CG = np.zeros(2*pax+4)
    CG_cma = np.zeros(2*pax+4)

    MTOW[0] = W_ept
    CG[0] = xcg_ept
    CG_cma[0] = xcg_ept_cma

    MTOW[1] = W_bow
    CG[1] = xcg_bow
    CG_cma[1] = xcg_bow_cma

    MTOW[2] = W_ow
    CG[2] = xcg_ow
    CG_cma[2] = xcg_ow_cma

    for i in range(0, pax):
        m_pax = passageiros[i, 2]/9.81
        xcg_pax = passageiros[i, 0]
        S_mx = m_pax*xcg_pax + MTOW[2+i]*CG[2+i]
        S_m = m_pax + MTOW[2+i]
        MTOW[3+i] = S_m
        CG[3+i] = S_mx/S_m
        CG_cma[3+i] = (CG[3+i] - airplane['xm_w'])/airplane['cm_w']

    MTOW[pax+3] = W0_mf
    CG[pax+3] = xcg_w0mf
    CG_cma[pax+3] = (CG[pax+3] - airplane['xm_w'])/airplane['cm_w']

    for i in range(0, pax):
        m_pax = -passageiros[i, 2]/9.81
        xcg_pax = passageiros[i, 0]
        S_mx = m_pax*xcg_pax + MTOW[pax+3+i]*CG[pax+3+i]
        S_m = m_pax + MTOW[pax+3+i]
        MTOW[pax+4+i] = S_m
        CG[pax+4+i] = S_mx/S_m
        CG_cma[pax+4+i] = (CG[pax+4+i] - airplane['xm_w'])/airplane['cm_w']

    if plot:
        plt.figure()
        plt.vlines(x=perc_np, ymin=W1, ymax=W2, color="orange")
        plt.vlines(x=perc_mlg, ymin=W1, ymax=W2, color="blue")
        plt.vlines(x=perc_aft_cg, ymin=W1, ymax=W2,
                   color='green', linestyle='--', linewidth=2)
        plt.plot(xcg_ept_cma, W_ept, 'go')
        plt.plot(xcg_bow_cma, W_bow, 'bo')
        plt.plot(xcg_ow_cma, W_ow, 'co')
        plt.plot(xcg_w0_cma, W0, 'ro')
        plt.plot(xcg_w0mf_cma, W0_mf, 'yo')
        plt.plot(CG_cma, MTOW, 'k-', linewidth=2)
        plt.plot(xcg_ept_cma, W_ept, 'go')
        plt.plot(xcg_bow_cma, W_bow, 'bo')
        plt.plot(xcg_ow_cma, W_ow, 'co')
        plt.plot(xcg_w0_cma, W0, 'ro')
        plt.plot(xcg_w0mf_cma, W0_mf, 'yo')
        plt.vlines(x=perc_fwd_cg, ymin=W1, ymax=W2,
                   color='green', linestyle='--', linewidth=2)
        plt.hlines(y=W1, xmin=perc_fwd_cg, xmax=perc_aft_cg,
                   color='green', linestyle='--', linewidth=2)
        plt.hlines(y=W2, xmin=perc_fwd_cg, xmax=perc_aft_cg,
                   color='green', linestyle='--', linewidth=2)
        plt.title(f"Passeio do CG - {airplane['name']}",
                  {'fontname': 'Times New Roman'}, fontsize=12)
        plt.xlabel("Posição do CG [%cma]", {
                   'fontname': 'Times New Roman'}, fontsize=12)
        plt.ylabel("Peso [kgf]", {'fontname': 'Times New Roman'}, fontsize=12)
        plt.legend(["Neutral Point", "Landing Gear", "Envelope", "Empty Weight",
                   "BOW", "OW", "MTOW", "MTOW - Fuel Weight"], fontsize=8)
        plt.ylim([0.99*W1, W2*1.01])
        plt.xlim([0, 1])
        plt.grid()
        plt.savefig(f"{airplane['name']}_passeio_cg.pdf", format="pdf")
        plt.show()
        # Print results

        # print(airplane['name'])
        # print(" airplane [ ' xcg_fwd '] = " , airplane['xcg_fwd'])
        # print(" airplane [ ' xcg_aft '] = " , airplane['xcg_aft'])
        # print(" airplane [ ' xnp '] = " , airplane['xnp'])
        # print(" airplane [ ' SM_fwd '] = " , airplane['SM_fwd'])
        # print(" airplane [ ' SM_aft '] = " , airplane['SM_aft'])
        # print(" airplane [ ' tank_excess '] = " , airplane['tank_excess'])
        # print(" airplane [ ' V_maxfuel '] = " , airplane['V_maxfuel'])
        # print(" airplane [ ' CLv '] = " , airplane['CLv'])
        # print("=======================================")


def airleron(airplane, save=False):
    x = np.linspace(0.12, 0.34, 200)

    y_upper = 0.2720 + 0.006571 * x**(-2.1726)  # Curva superior
    y_lower = 0.2230 + 0.008951 * x**(-1.8471)  # Cruva inferior

    # Plot
    plt.figure(figsize=(7, 5))
    plt.fill_between(x, y_lower, y_upper, color="lightgray",
                     alpha=0.8, label="Historical guidelines")

    plt.plot(x, y_upper, color="gray", linewidth=1)
    plt.plot(x, y_lower, color="gray", linewidth=1)
    plt.scatter(airplane["c_ail_c_wing"], airplane["b_ail_b_wing"], c="red", marker="*", s=150,
                label=f"{airplane['name']} ({airplane['c_ail_c_wing']}, {airplane['b_ail_b_wing']})")
    plt.xlabel("Aileron chord/wing chord",
               {'fontname': 'Times New Roman'}, fontsize=12)
    plt.ylabel("Aileron span/wing span",
               {'fontname': 'Times New Roman'}, fontsize=12)
    plt.title("Aileron Dimensions", {
              'fontname': 'Times New Roman'}, fontsize=14)

    plt.xlim(0.1, 0.35)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig("aileron.pdf", format="pdf")

    plt.show()


def full_analysis(airplane, passageiros, plot=False):
    dt.plot3d(airplane)
    dragPlot(airplane, plot)
    dragPolar(airplane, plot)
    weightsFraction(airplane, plot)
    restrictionDiagram(airplane, plot)
    plot_balance(airplane, passageiros, plot)
    # ballance(airplane, plot)


def payload_range(airplane, plot=True):

    # payload_max = airplane['W_payload']/9.81  # kg
    payload_max = 6124
    Empty_weight = airplane['W_empty']/9.81 + 270  # kg
    # Empty_weight = 13835 + 270
    # Range = airplane['range_cruise']/dt.nm2m  # nm
    # fuel_max = airplane['W_fuel']/9.81  # kg
    fuel_max = 6554
    MTOW = airplane['W0']/9.81  # kg
    # MTOW = 20600
    Mach_cruise = airplane["Mach_cruise"]

    [CL_cruise, CD_cruise] = dragPolar(
        airplane, plot=False, print_log=False, save=False)

    #        max payload and MTOW                      # MTOW max fuel              # max fuel empty airplane
    fuel = [(MTOW - payload_max - Empty_weight),
            fuel_max,
            fuel_max]

    payload = [payload_max,
               (MTOW - Empty_weight - fuel_max),
               0]
    weight = [MTOW,
              MTOW,
              (fuel_max + Empty_weight)]

    [T, p, rho, mi] = dt.atmosphere(airplane['altitude_cruise'])

    V = Mach_cruise * np.sqrt(dt.gamma_air * dt.R_air * T)  # m/s

    CL = np.zeros(len(fuel))
    CD = np.zeros(len(fuel))
    L_D = np.zeros(len(fuel))
    Mission_Range = np.zeros(len(fuel))

    for i in range(len(fuel)):

        CL[i] = weight[i]*9.81 / (0.5*rho*V**2*airplane['S_w'])
        CD[i] = np.interp(CL[i], CL_cruise, CD_cruise)

        L_D[i] = CL[i]/CD[i]

        Mission_Range[i] = V/(0.69/3600) * L_D[i] * \
            np.log(weight[i]/(weight[i]-fuel[i]))/dt.nm2m

    # print('Total Fuel mass', fuel_max)
    # print('Empty Weight =', Empty_weight)
    print('Mission Fuel mass [kg] =', fuel)
    print('Mission Payload [kg] =', payload)
    print('Mission Range [nm] =', Mission_Range)
    print('Mission Total Weight [kg] = ', weight)

    print('L/D = ', L_D)
    print('MTOW [kg] = ', MTOW)
    print('Empty Weight [kg] = ', Empty_weight)

    plt.scatter(0, payload_max, color='blue')
    plt.scatter(Mission_Range, payload, color='blue')
    plt.plot([0, Mission_Range[0]], [payload[0], payload[0]],
             color='black', linewidth=1.5)
    plt.plot(Mission_Range, payload, color='black', linewidth=1.5)
    plt.grid(True)
    plt.xlabel('Range [nm]', {'fontname': 'Times New Roman'}, fontsize=14)
    plt.ylabel('Payload [kg]', {'fontname': 'Times New Roman'}, fontsize=14)
    plt.show()

    return fuel, payload, Mission_Range
