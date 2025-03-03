import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import scipy.optimize as opt
from scipy.integrate import odeint, quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from pathlib import Path
import os
import sys
from byutpl.properties import water as water


# ----- Constants ----- #
# heat exchanger physical parameters
di = .206 * 2.54 / 100                  # m
do = .25 * 2.54 / 100                   # m
L = 14 * 2.54 / 100                     # m
k = 16.3                                # W/m.K 316 SS
g = 9.81                                # m/s^2
N = 56                                  # number of tubes    

# calculate the heat transfer area
# Ai = .25 * np.pi * di**2 * N
# Ao = .25 * np.pi * do**2 * N
Ai = np.pi * di * N * L
Ao = np.pi * do * N * L

# ---------------------------------------------- Functions ---------------------------------------------- #
# ----- Functions ----- #
def hi(Qi,Ti):
    # calculate the velocity
    Ac = np.pi * di**2 * .25 * N
    v = Qi / Ac

    # calculate the Reynolds number
    Re = water.ldn(Ti) * v * di / water.lvs(Ti)

    # calculate the Nusselt number
    if Re < 2300:
        Nu = 3.66
    else:
        Nu = .023 * (Re**.8) * water.lpr(Ti)**.4

    # calculate heat transfer coefficient
    h = Nu * water.ltc(Ti) / di
    return h

def s1(Ts,Ps,heat_water):
    if Ts < (20 + 273.15) or Ts > (110 + 273.15):
        return 1000000
    # pull in the values
    rhol = water.ldn(Ts)
    rhov = water.vdn(Ts,Ps)
    kl = water.ltc(Ts)
    mul = water.lvs(Ts)
    Tsat = water.tsat(Ps)
    cpl = water.vcp(Ts,Ps) / water.mw
    # cpl = water.lcp(Ts) / water.mw
    hfg = water.hvp(Ts) / water.mw
    nul = water.lnu(Ts)
    Prl = water.lpr(Ts)
    Lbaf = 6 * 2.54 / 100

    corr_val = kl / (nul**2 / g)**(1/3)
    Pval = Lbaf * (Tsat - Ts) * corr_val / (mul * hfg)

    if Pval < 15.8:
        h = .943 * Pval**-.25 * corr_val
    elif Pval <= 2530:
        h = ((.68 * Pval) + .89)**.82 * corr_val / Pval
    else: 
        h = ((((.042 * Pval) - 53) * Prl**.5) + 89)**(4 / 3) * corr_val / Pval 

    
    Q_steam = h * (Tsat - Ts) * Ao

    return Q_steam - heat_water

def ho(Ps,Ts,heat_water):
    # pull in the values
    rhol = water.ldn(Ts)
    rhov = water.vdn(Ts,Ps)
    kl = water.ltc(Ts)
    mul = water.lvs(Ts)
    Tsat = water.tsat(Ps)
    cpl = water.vcp(Ts,Ps) / water.mw
    # cpl = water.lcp(Ts) / water.mw
    hfg = water.hvp(Ts) / water.mw
    nul = water.lnu(Ts)
    Prl = water.lpr(Ts)
    Lbaf = 6 * 2.54 / 100

    # # find Ja
    # Ja = cpl * (Tsat - Ts) / hfg

    # # calculate the condensation energy
    # hfp = hfg * (1 + (.68 * Ja))

    # # calculate the heat transfer coefficient
    # h = .729 * (rhol * g * (rhol - rhov) * hfp * kl**3 / (N * mul * (Tsat - Ts) * do))**.25

    # solve for the surface temperature
    Tsurf = opt.fsolve(s1,Ts,args=(Ps,heat_water))[0]

    # calculate the P value
    corr_val = kl / (nul**2 / g)**(1/3)
    Pval = Lbaf * (Tsat - Tsurf) * corr_val / (mul * hfg)

    if Pval < 15.8:
        h = .943 * Pval**-.25 * corr_val
    elif Pval <= 2530:
        h = ((.68 * Pval) + .89)**.82 * corr_val / Pval
    else: 
        h = ((((.042 * Pval) - 53) * Prl**.5) + 89)**(4 / 3) * corr_val / Pval 

    return h

hi_vec = np.vectorize(hi)
ho_vec = np.vectorize(ho)


def model(inputs,Rf):
    Qwd,Psd,Tweffd,heat_water_array = inputs
    heat_water = np.average(heat_water_array)

    #                                  |               |                                        |
    #      convection_inner            | fouling_inner |               conduction               | convection_outer
    #                                  |               |                                        |
    h1 = (hi_vec(Qwd, Tweffd) * Ai)**-1
    h2 = (ho_vec(Psd, Tweffd,heat_water) * Ao)**-1
    sumR = h1 + (Rf / Ai) + (np.log(do / di) / (2 * np.pi * k * L * N)) + h2
    # sumR = .1 + (Rf / Ai) + (np.log(do / di) / (2 * np.pi * k * L * N)) + (ho_vec(Psd, Tweffd) * Ao)**-1
    # print(((hi_vec(Qwd, Tweffd) * Ai)**-1)[0], (Rf / Ai), (np.log(do / di) / (2 * np.pi * k * L * N)), ((ho_vec(Psd, Tweffd) * Ao)**-1)[0])

    print(h1,h2)
    UA = 1 / sumR
    return UA

# def model(inputs, Rfi,Rfo):
#     Qwd,Psd,Tweffd = inputs
#     #                       |                 |                                  |                 |
#     #      convection_inner | fouling_inner   |             conduction           |  fouling_outer  | convection_outer
#     #                       |                 |                                  |                 |
#     sumR = (hi_vec(Qwd,Tweffd) * Ai)**-1 + (Rfi / Ai) + (np.log(do / di) / (2 * np.pi * k * L * N)) + (Rfo / Ao) + (ho_vec(Psd, Tweffd) / Ao)**-1
#     UA = 1 / sumR
#     return UA



# ---------------------------------------------- Import Data ---------------------------------------------- #
data1 = pd.read_csv('data/Trial1.csv')
data2 = pd.read_csv('data/Trial2.csv')
data3 = pd.read_csv('data/Trial3.csv')
data4 = pd.read_csv('data/Trial4.csv')
data5 = pd.read_csv('data/Trial5.csv')
data6 = pd.read_csv('data/Trial6.csv')
data7 = pd.read_csv('data/Trial7.csv')
data8 = pd.read_csv('data/Trial8.csv')
data9 = pd.read_csv('data/Trial9.csv')
dataA = pd.read_csv('data/TrialA.csv')
dataB = pd.read_csv('data/TrialB.csv')
dataC = pd.read_csv('data/TrialC.csv')
dataD = pd.read_csv('data/TrialD.csv')
dataE = pd.read_csv('data/TrialE.csv')
dataF = pd.read_csv('data/TrialF.csv')


# data_collection =                 np.array([data5,data6,data7,data8,data9,dataA,dataB,dataC,dataD,dataE,dataF])
data_collection = np.array([data1,data2,data3,data4,data5,data6,data7,data8,data9,dataA,dataB,dataC,dataD,dataE,dataF])

print(data5.keys())


qs = np.array([])
Twout = np.array([])
Twin = np.array([])
Ps = np.array([])

# create the data arrays
for i, df in enumerate(data_collection):
    qs = np.append(qs,df[:,2])
    Twout = np.append(Twout,df[:,6])
    Twin = np.append(Twin,df[:,5])
    Ps = np.append(Ps,df[:,4])
    
Tavg = (Twout + Twin) / 2

# conver the data to SI units
qs_good = qs * .003785 / 60                 # gal/min to m^3/s
Ps_good = (Ps + 12.57) * 101325 / 14.7       # psig to Pa
Cpw = water.lcp(Tavg + 273.15) / water.mw   # J/kg.K

tsat = np.vectorize(water.tsat)

Tsat = tsat(Ps_good)

# ---------------------------------------------- Calculations ---------------------------------------------- #
# calculate the delta T values
dT1 = Tsat - Twout
dT2 = Tsat - Twin

# calcualate the delta T log mean
dTlm = (dT1 - dT2) / np.log(dT1 / dT2)

# calculate the mass flow rate of the water 
rho = water.ldn(Tavg + 273.15)
m = qs_good * rho

# find the heat transfer
Q_water = -m * Cpw * (Twin - Twout)

# calculate the heat transfer coefficient
UA_array = Q_water / dTlm


# # fit the data with the model
# Rf = curve_fit(model, (qs_good,Ps_good,Tavg + 273.15), UA)

xdata = np.array([qs_good, Ps_good, Tavg + 273.15,Q_water])  # Stack inputs correctly

Rf, _ = curve_fit(model, xdata, UA_array)
Rf = Rf[0]

print(Rf)
print(_[0,0])

# Rf, _ = curve_fit(model, xdata, UA_array, bounds = (0,np.inf))
# Rfi = Rf[0]
# Rfo = Rf[1]

# print(Rfi,Rfo)


# ---------------------------------------------- Plotting ---------------------------------------------- #
# Create a 3D mesh grid for flow rate and pressure
flow_rate_range = np.linspace(min(qs_good), max(qs_good),100)
pressure_range = np.linspace(min(Ps_good), max(Ps_good), 100)
flow_rate_mesh, pressure_mesh = np.meshgrid(flow_rate_range, pressure_range)

# Calculate the UA coefficient for each point in the mesh grid using the fit function
temperature_avg = np.mean(Tavg + 273.15)
UA_mesh = model((flow_rate_mesh, pressure_mesh, temperature_avg, Q_water),Rf)
# UA_mesh = model((flow_rate_mesh, pressure_mesh, temperature_avg),Rfi,Rfo)

# Plot the 3D surface of the UA coefficient
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(flow_rate_mesh * 1000, pressure_mesh / 101325, UA_mesh, cmap='viridis')

ax.scatter(qs_good * 1000, Ps_good / 101325, UA_array, color = '#ff9933', label='Data Points', s = 1)

ax.set_xlabel('Flow Rate (L/s)')
ax.set_ylabel('Pressure (atm)')
ax.set_zlabel('UA Coefficient (W/K)')
ax.set_title('UA Coefficient as a Function of Flow Rate and Pressure')

plt.show()

# Plot the fit function with the minimized Rf value
UA_fit = model((qs_good, Ps_good, Tavg + 273.15), Rf)

plt.figure()
plt.plot(UA_array, label='Actual UA')
plt.plot(UA_fit, label='Fitted UA')
plt.xlabel('Data Points')
plt.ylabel('UA Coefficient (W/K)')
plt.legend()
plt.title('Actual vs Fitted UA Coefficient')
# plt.show()


# ---------------------------------------------- Array of Rf Values ---------------------------------------------- #

# Calculate Rf value for each trial
Rf_values = []
cv_values = []

for df in data_collection:
    qs = df[:,2]
    Twout = df[:,6]
    Twin = df[:,5]
    Ps = df[:,4]
    
    Tavg = (Twout + Twin) / 2

    # Convert the data to SI units
    qs_good = qs * .003785 / 60                 # gal/min to m^3/s
    Ps_good = (Ps + 14.7) * 101325 / 14.7       # psig to Pa
    Cpw = water.lcp(Tavg + 273.15) / water.mw   # J/kg.K

    tsat = np.vectorize(water.tsat)
    Tsat = tsat(Ps_good)

    # Calculate the delta T values
    dT1 = Tsat - Twout
    dT2 = Tsat - Twin

    # Calculate the delta T log mean
    dTlm = (dT1 - dT2) / np.log(dT1 / dT2)

    # Calculate the mass flow rate of the water 
    rho = water.ldn(Tavg + 273.15)
    m = qs_good * rho

    # Find the heat transfer
    Q_water = -m * Cpw * (Twin - Twout)

    # Calculate the heat transfer coefficient
    UA_array = Q_water / dTlm

    # Fit the data with the model
    xdata = np.array([qs_good, Ps_good, Tavg + 273.15])  # Stack inputs correctly
    Rf,cv = curve_fit(model, xdata, UA_array,Q_water)
    Rf_values.append(Rf[0])
    cv_values.append(cv[0,0])

# Perform statistical analysis on the Rf values
Rf_values = np.array(Rf_values)
cv_values = np.array(cv_values)
mean_Rf = np.mean(Rf_values)
std_Rf = np.std(Rf_values)

print("Rf values for each trial:", Rf_values)
print("Coefficient of variation for each trial:", cv_values)
print("Mean Rf value:", mean_Rf)
print("Standard deviation of Rf values:", std_Rf)
