# Copyright (C) 2018 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, ethanol.py, is a python module of the                         #
# thermophysical properties of ethanol.  The properties, both              #
# constant and temperature-dependent, are taken from the DIPPR(R) Sample   #
# database which can be accessed at <https://dippr.aiche.org>.             #
# The vapor phase density is obtained from the Soave-Redlich-Kwong         #
# equation of state.                                                       #
#                                                                          #
# ethanol.py is distributed in the hope that it will be useful,            #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# You should have received a copy of the GNU General Public License        #
# along with ethanol.py.  If not, see                                      #
# <http://www.gnu.org/licenses/>.                                          #
#                                                                          #
# All published work which utilizes this module, or other property data    #
# from the DIPPR(R) database, should include the citation below.           #
# R. L. Rowley, W. V. Wilding, J. L. Oscarson, T. A. Knotts, N. F. Giles,  #
# DIPPR® Data Compilation of Pure Chemical Properties, Design Institute    #
# for Physical Properties, AIChE, New York, NY (2017).                     #
#                                                                          #
# ======================================================================== #
# ethanol.py                                                               #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - December 2024                                              #
# ======================================================================== #
"""
This library contains functions for the properties of ethanol.
The values come from the DIPPR(R) Sample database [1], 
and the DIPPR(R) abbreviations are used. Vapor properties that are 
dependent on pressure are obtained using the Soave-Redlich-Kwong equation
of state.

This module is part of the byutpl package. Import the module using

  import byutpl.properties.ethanol as etl

When imported in this way, constant properties can be called as   

  etl.acen
  
which returns the acentric factor. Temperature dependent properties
can be called as
  
  etl.vtc(t)

which returns the vapor thermal conductivity at `t` where `t` is 
temperature in units of K. Temperature and pressure dependent properties
can be called as
  
  etl.vcp(t,p)

which returns the vapor heat capacity at `t` and `p` where 
`t` is temperature in units of K and `p` is pressure in units of Pa.
    
A complete list of properties, and the associated units, are found       
below.                                                                   

Function    Return Value                             Input Value(s)         
---------   --------------------------------------   ----------------- 
tc          critical temperature in K                none              
pc          critical pressure in Pa                  none              
vc          critical volume in m**3/mol              none              
zc          critical compress. factor (unitless)     none              
mw          molecular weight in kg/mol               none              
acen        acentric factor (unitless)               none              
ldn(t)      liquid density in kg/m**3                temperature in K  
lcp(t)      liquid heat capacity in J/(mol*K)        temperature in K  
ltc(t)      liquid thermal conductivity in W/(m*K)   temperature in K  
vp(t)       liquid vapor pressure in Pa              temperature in K  
hvp(t)      heat of vaporization in J/mol            temperature in K  
lpr(t)      liquid Prandtl number (unitless)         temperature in K  
lvs(t)      liquid viscosity in Pa*s                 temperature in K  
lnu(t)      liquid kinematic viscosity in m**2/s     temperature in K  
tsat(p)     temperature at saturation in K           pressure in Pa    
vvs(t)      vapor (steam) viscosity in Pa*s          temperature in K  
vtc(t)      vapor (steam) therm. conduct. in W/(m*K) temperature in K  
vdn(t,p)    vapor (steam) density in kg/m**3         temperature in K
                                                     pressure in Pa
vcp(t,p)    vapor (steam) isobaric heat capacity     temperature in K
            in J/(mol*K)                             pressure in Pa
vnu(t,p)    vapor (steam) kinematic viscosity        temperature in K
            in m**2/s                                pressure in Pa                                                     
vpr(t,p)    vapor (steam) Prandtl number (unitless)  temperature in K
                                                     pressure in Pa                                                     

References
----------
.. [1] W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
   DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
   for Physical Properties, AIChE, New York, NY (2017).
"""

import numpy as np
from scipy.optimize import fsolve
import byutpl.eos.srk as srk
import byutpl.equations.dippreqns as dippr

# critical temperature
tc = 514 # units of K

# critical pressure
pc = 6.137e6 # units of Pa

# critical volume
vc = 0.000168 # units of m**3/mol

# critical compressibility factor
zc = 0.241 # unitless

# acentric factor
acen = 0.643558 # unitless

# molecular weight
mw = 46.06844e-3 # units of kg/mol
  
def ldn(t):
    """liquid density of ethanol 
	
    Liquid density of ethanol from the DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 105; valid from 159.05 - 514 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid density
        of ethanol.

    Returns
    -------
    float
        The value of the liquid density (kg/m**3) of ethanol at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([1.6288,0.27469,514,0.23178])
    y = dippr.eq105(t,c)
    y = y * 1000 # convert from kmol/m**3 to mol/m**3
    y = y * mw # convert from mol/m**3 to kg/m**3
    return(y)
  
def lcp(t):
    """liquid heat capacity of ethanol 
	
    Liquid heat capacity of ethanol from the DIPPR(R) correlation
    (Correlation B: DIPPR Equation 100; valid from 159.05 - 390.00 K;
    uncertainty: < 3%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid heat capacity
        of ethanol.

    Returns
    -------
    float
        The value of the liquid heat capacity (J mol**-1 K**-1) of ethanol at
        `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([102640,-139.63,-0.030341,0.0020386,0])
    y = dippr.eq100(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def ltc(t):
    """liquid thermal conductivity of ethanol 
	
    Liquid thermal conductivity of ethanol from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 100; valid from 159.05 - 353.15 K;
    uncertainty: < 5%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid thermal
        conductivity of ethanol.

    Returns
    -------
    float
        The liquid thermal conductivity (W m**-1 K**-1) of ethanol at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([0.2468,-0.000264,0,0])
    y = dippr.eq100(t,c)
    return(y)

def vp(t):
    """liquid vapor pressure of ethanol 
	
    Liquid vapor pressure of ethanol from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 159.05 — 514.00 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid vapor pressure of
        ethanol.

    Returns
    -------
    float
        The liquid vapor pressure (Pa) of ethanol at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([73.304,-7122.3,-7.1424,2.8853E-06,2])
    y = dippr.eq101(t,c)
    return(y)
    
def hvp(t):
    """heat of vaporization of ethanol 
	
    Heat of vaporization of ethanol from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 106; valid from 159.05 — 514.00 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat of vaporization of
        ethanol.

    Returns
    -------
    float
        The heat of vaporization (J/mol) of ethanol at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([65831000,1.1905,-1.7666,1.0012,0])
    tr = t/tc
    y = dippr.eq106(tr,c)
    y = y / 1000 # convert from J/kmol to J/mol
    return(y)
    
def lvs(t):
    """liquid viscosity of ethanol 
	
    Liquid viscosity of ethanol from the DIPPR(R) correlation
    (Correlation A: DIPPR Equation 101; valid from 200 - 440 K;
    uncertainty: < 10%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid viscosity of
        ethanol.

    Returns
    -------
    float
        The liquid viscosity (Pa*s) of ethanol at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([7.875,781.98,-3.0418,0,0])
    y = dippr.eq101(t,c)
    return(y)

def lnu(t):
    """liquid kinematic viscosity of ethanol 
	
    Liquid kinematic viscosity of ethanol calculated from the lvs and ldn
    functions in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the liquid kinematic 
        viscosity of ethanol.

    Returns
    -------
    float
        The liquid kinematic viscosity (m**2/s) of ethanol at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(lvs(t)/ldn(t))

def lpr(t):
    """Prandtl number of liquid ethanol 
	
    Prandtl number of liquid ethanol calculated from the lcp, lvs, and ltc
    functions in this module.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of
        liquid ethanol.

    Returns
    -------
    float
        The Prandtl number (dimensionless) of liquid ethanol at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return lcp(t)*lvs(t)/ltc(t)/mw

def ftsat(t,p):
    """function supplied to fsolve in tsat function 
	
    Function supplied to fsolve (in the f(x)=0 form) to solve for the 
    temperature at saturation for a given pressure.  This 
    function is of little use to users.  Users should use 
    the function tsat.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the function.  It is
    	the "x" value for which fsolve solves.
	
    p : float
    	The pressure (Pa) at which the saturated temperature is desired.

    Returns
    -------
    float
        The value of the function supplied to fsolve.  This value will be 
    	zero if 't' is the saturated temperature for `p`. 

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(vp(t) - p)

def tsat(p):
    """saturated temperature for ethanol
	
    Saturation temperature of ethanol for a given pressure 'p'.  It is
    the temperature for which the following equation is true:
    vp(t)= `p`
    where vp is the function in this module and t is the value
    this function (tsat) returns.
	
    Parameters
    ----------
    p : float
        The pressure (Pa) at which to find the saturated temperature.

    Returns
    -------
    float
        The temperature (K) of ethanol at saturation at pressure `p`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    x = 700 # guess in K
    y = fsolve(ftsat,x,p)
    return(y[0])
    
def vvs(t):
    """vapor viscosity of ethanol
	
    Vapor viscosity of ethanol at temperature `t` from the DIPPR(R)
    correlation.
    (Correlation A: DIPPR Equation 102; valid from 200 — 1000 K;
    uncertainty: < 10%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor viscosity of
        ethanol.

    Returns
    -------
    float
        The vapor viscosity of ethanol at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([1.0613E-07,0.8066,52.7,0.0])
    y = dippr.eq102(t,c)
    return(y)

def vtc(t):
    """vapor thermal conductivity of ethanol
	
    The vapor thermal conductivity of ethanol at temperature `t`
    from the DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 102; valid from 293.15 — 1000 K;
    uncertainty: < 10%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor thermal
        conductivity of ethanol.

    Returns
    -------
    float
        The vapor thermal conductivity of ethanol (W m**-1 K**-1) 
        at `t`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    c = np.array([-0.010109,0.6475,-7332,-268000])
    y = dippr.eq102(t,c)
    return(y)

def vdn(t,p):
    """vapor density of ethanol
	
    The vapor density of ethanol at temperature `t` and
    pressure `p` from the Soave-Redlich-Kwong equation of state. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the vapor density of
        ethanol.

    p : float
        The pressure (Pa) at which to evaluate the vapor density of
        ethanol.

    Returns
    -------
    float
        The vapor density of ethanol (kg/m**3) at `t` and `p`.
	"""   
    v = srk.vv(t,p,tc,pc,acen)
    v = v / mw # convert from m**3/mol to m**3/kg
    return(1/v)
    
def icp(t): 
    """ideal gas heat capacity of ethanol
    
    The ideal gas heat capacity of ethanol at temperature `t` from the 
    DIPPR(R) correlation.
    (Correlation A: DIPPR Equation 107; valid from 273.15 — 1500 K;
    uncertainty: < 1%)
    
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the ideal gas heat
        capacity of ethanol.

    Returns
    -------
    float
        The ideal gas heat capacity of ethanol (J/(mol*K)) at `t`.
    """     
    c=np.array([49200,145770,1662.8,93900,744.7])    
    y=dippr.eq107(t,c)
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return(y)

def vcp(t,p):
    """vapor heat capacity ethanol
	
    Heat capacity of vapor ethanol calculated from the DIPPR(R) 
    correlation for ideal gas heat capacity and the residual property
    from the Soave-Redlich-Kwong equation of state.
    (ICP Correlation A: DIPPR Equation 127; valid from 20 — 1500 K;
    uncertainty: < 1%)
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the heat 
        capacity of vapor ethanol.

    p : float
        The pressure (Pa) at which to evaluate the heat capacity
        of vapor ethanol.

    Returns
    -------
    float
        The heat capacity (J/(mol*K)) of vapor ethanol
        at `t` and `p`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    x = icp(t) + srk.cprv(t,p,tc,pc,acen)
    return(x)

def vnu(t,p):
    """vapor kinematic viscosity of ethanol
	
    Kinematic viscosity of vapor ethanol calculated from the vvs and
    vdn functions in this module. The calculation uses the Soave-Redlich-
    Kwong equation of state for the vapor density.
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the kinematic 
        viscosity of vapor ethanol.

    p : float
        The pressure (Pa) at which to evaluate the kinematic viscosity
        of vapor ethanol.

    Returns
    -------
    float
        The kinematic viscosity (m**2/s) of vapor ethanol
        at `t` and `p`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(vvs(t)/vdn(t,p))

def vpr(t, p):
    """Prandtl number of vapor ethanol
	
    Prandtl number of vapor ethanol calculated from the vcp, vvs, 
    and vtc functions in this module. The calculation uses the Soave-
    Redlich-Kwong equation of state to correct the ideal gas heat capacity
    to the real gas at `t` and `p`. 
	
    Parameters
    ----------
    t : float
        The temperature (K) at which to evaluate the Prandtl number of vapor
        ethanol.

    p : float
        The pressure (Pa) at which to evaluate the Prandtl number of vapor
        ethanol.

    Returns
    -------
    float
        The Prandtl number (dimensionless) of vapor ethanol at `t`
        and `p`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    return(vcp(t,p)*vvs(t)/vtc(t)/mw)


def unit(key):
    """Returns the units of `key` 
	
    Returns the units of the constant or function in this module identified
    by 'key'
	
    Parameters
    ----------
    key : string
        The name of the constant or function in this module for which the
        units are needed.

    Returns
    -------
    string
        The units for the constant or function identified by `key`.

    References
    ----------
    .. W. V. Wilding, T. A. Knotts, N. F. Giles, R. L. Rowley, J. L. Oscarson, 
       DIPPR® Data Compilation of Pure Chemical Properties, Design Institute
       for Physical Properties, AIChE, New York, NY (2017).
	"""
    if type(key) !=str:
        return('The parameter must be a string.')
    if key == 'tc':
        return('K')
    if key == 'pc':
        return('Pa')
    if key == 'vc':
        return('m**3/mol')
    if key == 'zc':
        return('unitless')
    if key == 'acen':
        return('unitless')
    if key == 'mw':
        return('kg/mol')
    if key == 'ldn':
        return('kg/m**3')
    if key == 'lcp':
        return('J mol**-1 K**-1')
    if key == 'icp':
        return('J mol**-1 K**-1')
    if key == 'vcp':
        return('J mol**-1 K**-1')
    if key == 'ltc':
        return('W m**-1 K**-1')
    if key == 'vp':
        return('Pa')
    if key == 'hvp':
        return('J/mol')
    if key == 'lpr':
        return('unitless')
    if key == 'lvs':
        return('Pa*s')
    if key == 'lnu':
        return('m**2/s')
    if key == 'tsat':
        return('K')
    if key == 'vvs':
        return('Pa*s')
    if key == 'vtc':
        return('W m**-1 K**-1')
    if key == 'vnu':
        return('m**2/s')
    if key == 'vdn':
        return('kg/m**3')
    if key == 'vpr':
        return('unitless')
    else:
        return('"'+key+'" is not a constant or function in this module.')



  
