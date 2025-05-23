#!/usr/bin/env python

import numpy as np
import torch
from torch import nn
import os
import sys
from typing import Union



if 'DIIM_PATH' in os.environ:
    HOME_PATH = MODEL_HOME = os.environ["DIIM_PATH"]
else:
    
    print("Missing local variable DIIM_PATH. \nPlease add it with '$:export DIIM_PATH=path/to/diimpy'.")
    sys.exit()
        
@torch.jit.script
class AbsorptionParams:
    def __init__(self, dCDOM: float,
                 sCDOM: float,
                 dNAP: float,
                 sNAP: float,
                 absortion_w: torch.Tensor,
                 absortion_PH: torch.Tensor,
                 eNAP: float,
                 fNAP: float,
                 linear_regression_slope_b: float,
                 linear_regression_intercept_b: float,
                 Theta_o: float,
                 Theta_min: float,
                 beta: float,
                 sigma: float,
                 backscattering_w: torch.Tensor,
                 linear_regression_slope_s: float,
                 linear_regression_intercept_s: float,
                 scattering_w: torch.Tensor,
                 vs: float,
                 rs: float,
                 vu: float,
                 ru: float,
                 rd: float,
                 T: float,
                 gammaQ: float):
        self.dCDOM=dCDOM
        self.sCDOM = sCDOM
        self.dNAP = dNAP
        self.sNAP = sNAP
        self.absortion_w = absortion_w
        self.absortion_PH = absortion_PH
        self.eNAP = eNAP
        self.fNAP = fNAP
        self.linear_regression_slope_b = linear_regression_slope_b
        self.linear_regression_slope_s = linear_regression_slope_s
        self.linear_regression_intercept_b = linear_regression_intercept_b
        self.linear_regression_intercept_s = linear_regression_intercept_s
        self.Theta_o = Theta_o
        self.Theta_min = Theta_min
        self.beta = beta
        self.sigma = sigma
        self.backscattering_w = backscattering_w
        self.scattering_w = scattering_w
        self.vs = vs
        self.rs = rs
        self.vu = vu
        self.ru = ru
        self.rd = rd
        self.T = T
        self.gammaQ = gammaQ


###################################################################################################################################################################################################
############################################################################FUNCTIONS NEEDED TO DEFINE THE FORWARD MODEL###########################################################################
###################################################################################################################################################################################################

################Functions for the absortion coefitient####################
@torch.jit.script
def absortion_CDOM(lambda_,perturbation_factors,params: AbsorptionParams):
    """
    Function that returns the mass-specific absorption coefficient of CDOM, function dependent of the wavelength lambda. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.dCDOM*torch.exp(-(params.sCDOM * perturbation_factors[6])*(lambda_ - 450.))

@torch.jit.script
def absortion_NAP(lambda_,params: AbsorptionParams):
    """
    Mass specific absorption coefficient of NAP.
    See Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.dNAP*torch.exp(-params.sNAP*(lambda_ - 440.))

@torch.jit.script
def absortion(lambda_,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    Total absortion coeffitient.
    aW,λ (values used from Pope and Fry, 1997), aP H,λ (values averaged and interpolated from
    Alvarez et al., 2022).
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.absortion_w + (params.absortion_PH* perturbation_factors[0])*chla + \
                (absortion_CDOM(lambda_, perturbation_factors,params)* perturbation_factors[5])*CDOM + absortion_NAP(lambda_,params)*NAP
   
    
##############Functions for the scattering coefitient########################

@torch.jit.script
def Carbon(chla,PAR, perturbation_factors,params: AbsorptionParams):
    """
    defined from the carbon to Chl-a ratio. 
    theta_o, sigma, beta, and theta_min constants (equation and values computed from Cloern et al., 1995), and PAR
    the Photosynthetically available radiation, obtained from the OASIM model, see Lazzari et al., 2021.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    nominator = chla
    beta =  params.beta * perturbation_factors[11]
    sigma = params.sigma * perturbation_factors[12]
    exponent = -(PAR - beta)/sigma
    denominator = (params.Theta_o* perturbation_factors[10]) * ( torch.exp(exponent)/(1+torch.exp(exponent)) ) + \
        (params.Theta_min * perturbation_factors[9])
    return nominator/denominator

@torch.jit.script
def scattering_ph(lambda_,perturbation_factors,params: AbsorptionParams):
    """
    The scattering_ph is defined initially as a linear regression between the diferent scattering_ph for each lambda, and then, I
    change the slope and the intercept gradually. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    
    return (params.linear_regression_slope_s * perturbation_factors[1]) *\
        lambda_ + params.linear_regression_intercept_s * perturbation_factors[2]

@torch.jit.script
def backscattering_ph(lambda_,perturbation_factors,params: AbsorptionParams):
    """
    The scattering_ph is defined initially as a linear regression between the diferent scattering_ph for each lambda, and then, I
    change the slope and the intercept gradually. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    
    return (params.linear_regression_slope_b * perturbation_factors[3]) *\
        lambda_ + params.linear_regression_intercept_b * perturbation_factors[4]

@torch.jit.script
def scattering_NAP(lambda_,params: AbsorptionParams):
    """
    NAP mass-specific scattering coefficient.
    eNAP and fNAP constants (equation and values used from Gallegos et al., 2011)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.eNAP*(550./lambda_)**params.fNAP
    
@torch.jit.script
def scattering(lambda_,PAR,chla,NAP,perturbation_factors,params: AbsorptionParams):
    """
    Total scattering coefficient.
    bW,λ (values interpolated from Smith and Baker, 1981,), bP H,λ (values used
    from Dutkiewicz et al., 2015)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.scattering_w + scattering_ph(lambda_,perturbation_factors,params) * \
            Carbon(chla,PAR,perturbation_factors,params) + scattering_NAP(lambda_,params) * NAP

    
#################Functions for the backscattering coefitient#############

@torch.jit.script
def backscattering(lambda_,PAR,chla,NAP,perturbation_factors,params: AbsorptionParams):
    """
    Total backscattering coefficient.
     Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.backscattering_w + backscattering_ph(lambda_,perturbation_factors,params) * \
                Carbon(chla,PAR,perturbation_factors,params) + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,params) * NAP

        


###############Functions for the end solution of the equations###########
#The final result is written in terms of these functions, see ...

@torch.jit.script
def c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams): 
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return (absortion(lambda_,chla,NAP,CDOM,perturbation_factors,params)\
            + scattering(lambda_,PAR,chla,NAP,perturbation_factors,params))/torch.cos(zenith*3.1416/180)

@torch.jit.script
def F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return (scattering(lambda_,PAR,chla,NAP,perturbation_factors,params)\
            - params.rd * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,params))/\
        torch.cos(zenith*3.1416/180.)

@torch.jit.script
def B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model.
    """
    return  params.rd*backscattering(lambda_,PAR,chla,NAP,perturbation_factors,params)/torch.cos(zenith*3.1416/180) 

@torch.jit.script
def C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model.
    """
    return (absortion(lambda_,chla,NAP,CDOM,perturbation_factors,params)\
            + params.rs * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,params) )/params.vs

@torch.jit.script
def B_u(lambda_,PAR,chla,NAP,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return (params.ru * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,params))/params.vu

@torch.jit.script
def B_s(lambda_,PAR,chla,NAP,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return (params.rs * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,params))/params.vs

@torch.jit.script
def C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return (absortion(lambda_,chla,NAP,CDOM,perturbation_factors,params)\
            + params.ru * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,params))/ params.vu

@torch.jit.script
def D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return (0.5) * (C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params) + \
                     C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params) + \
                    ((C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params) + \
                      C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params))**2 - \
                     4. * B_s(lambda_,PAR,chla,NAP,perturbation_factors,params) *  \
                     B_u(lambda_,PAR,chla,NAP,perturbation_factors,params) )**(0.5))

@torch.jit.script
def x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model.
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) - \
                   C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params)) * \
                   (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) + \
                    C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params)) + \
                    B_s(lambda_,PAR,chla,NAP,perturbation_factors,params) * \
                    B_u(lambda_,PAR,chla,NAP,perturbation_factors,params)
    
    nominator = -(C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params) + \
                  c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params)) * \
                  F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,params) - \
                  B_u(lambda_,PAR,chla,NAP,perturbation_factors,params) * \
                  B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,params)

    return nominator/denominator

@torch.jit.script
def y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) - \
                   C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params)) * \
                   (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) + \
                    C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params)) + \
                    B_s(lambda_,PAR,chla,NAP,perturbation_factors,params) * \
                    B_u(lambda_,PAR,chla,NAP,perturbation_factors,params)
    
    nominator = (-B_s(lambda_,PAR,chla,NAP,perturbation_factors,params) * \
                 F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,params) ) + \
                 (-C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params) + \
                  c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params)) * \
                  B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,params)
    
    return nominator/denominator

@torch.jit.script
def C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model.
    """
    return E_dif_o - x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) * E_dir_o

@torch.jit.script
def r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return B_s(lambda_,PAR,chla,NAP,perturbation_factors,params) / \
            D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params)

@torch.jit.script
def k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params) - \
            C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params)

@torch.jit.script
def E_dir(z:float,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    This is the analytical solution of the bio-chemical model. (https://doi.org/10.5194/gmd-2024-174)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return E_dir_o*torch.exp(-z*c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params))

@torch.jit.script
def E_u(z:float,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    This is the analytical solution of the bio-chemical model. (https://doi.org/10.5194/gmd-2024-174)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) *\
            r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params)*\
            torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params)*z)+\
            y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) * \
            E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params)

@torch.jit.script
def E_dif(z:float,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    This is the analytical solution of the bio-chemical model. (https://doi.org/10.5194/gmd-2024-174)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) * \
        torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,params)*z) + \
        x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) * \
        E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params)
        
@torch.jit.script
def bbp(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    Particulate backscattering at depht z
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return backscattering_ph(lambda_,perturbation_factors,params) * \
                Carbon(chla,PAR,perturbation_factors,params) + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,params) * NAP
    
@torch.jit.script
def kd(z:float,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    Atenuation Factor
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model.
    """
    return (z**-1)*torch.log((E_dir_o + E_dif_o)/ \
                             (E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params) +\
                              E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params)))
                                                  

##########################from the bio-optical model to RRS(Remote Sensing Reflectance)##############################
#defining Rrs
#Q=5.33*np.exp(-0.45*np.sin(np.pi/180.*(90.0-Zenith)))

@torch.jit.script
def Q_rs(zenith,perturbation_factors):
    """
    Empirical result for the Radiance distribution function, 
    equation from Aas and Højerslev, 1999, 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return (5.33 * perturbation_factors[7])*torch.exp(-(0.45 * perturbation_factors[8])*torch.sin((3.1416/180.0)*(90.0-zenith)))

@torch.jit.script
def Rrs_minus(Rrs,params: AbsorptionParams):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return Rrs/(params.T+params.gammaQ*Rrs)

@torch.jit.script
def Rrs_plus(Rrs,params: AbsorptionParams):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return Rrs*params.T/(1-params.gammaQ*Rrs)

@torch.jit.script
def Rrs_MODEL(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params: AbsorptionParams):
    """
    Remote Sensing Reflectance.
    Aas and Højerslev, 1999.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    Rrs = E_u(0.,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,params)  /  \
          (Q_rs(zenith,perturbation_factors)*(E_dir_o + E_dif_o)   )
    return Rrs_plus( Rrs ,params)


class Forward_Model(nn.Module):
    """
    Bio-Optical model plus corrections, in order to have the Remote Sensing Reflectance, in terms of the inversion problem. 
    Forward_Model(x) returns a tensor, with each component being the Remote Sensing Reflectance for each given wavelength. 
    if the data has 5 rows, each with a different wavelength, RRS will return a vector with 5 components.  RRS has tree parameters, 
    self.chla is the chlorophil-a, self.NAP the Non Algal Particles, and self.CDOM the Colored Dissolved Organic Mather. 
    According to the invention problem, we want to estimate them by making these three parameters have two constraints,
    follow the equations of the bio-optical model, plus, making the RRS as close as possible to the value
    measured by the satellite.
    
    """
    def __init__(self,precision = torch.float32,num_days=1,learning_chla = True,learning_perturbation_factors = False):
        super().__init__()
        if learning_chla == True:
            self.chparam = nn.Parameter(torch.rand((num_days,1,3), dtype=precision), requires_grad=True)
        self.learning_chla = learning_chla

        if learning_perturbation_factors == False:
            self.perturbation_factors = torch.ones(14, dtype=precision)
        else:
            self.perturbation_factors =  nn.Parameter(torch.ones(14, dtype=precision), requires_grad=True)

        self.perturbation_factors_names = [
            '$\epsilon_{a,ph}$',
            '$\epsilon_{tangent,s,ph}$',
            '$\epsilon_{intercept,s,ph}$',
            '$\epsilon_{tangent,b,ph}$',
            '$\epsilon_{intercept,b,ph}$',
            '$\epsilon_{a,cdom}$',
            '$\epsilon_{exp,cdom}$',
            '$\epsilon_{q,1}$',
            '$\epsilon_{q,2}$',
            '$\epsilon_{theta,min}$',
            '$\epsilon_{theta,o}$',
            '$\epsilon_\\beta$',
            '$\epsilon_\sigma$',
            '$\epsilon_{b,nap}$',
        ]
        self.precision = precision

    def forward(self,x_data,parameters = None, axis = None,perturbation_factors_ = None,constant = None):
        """
        x_data: torch tensor. 
        """
        if type(perturbation_factors_) == type(None):
            perturbations = self.perturbation_factors
        else:
            perturbations = perturbation_factors_
        if type(parameters) == type(None):
            if self.learning_chla == False:
                print('Please provide a tensor with the value of chla,nap and cdom')
            
            Rrs = Rrs_MODEL(x_data[:,:,0],x_data[:,:,1],x_data[:,:,2],\
                            x_data[:,:,3],x_data[:,:,4],torch.exp(self.chparam[:,:,0]),torch.exp(self.chparam[:,:,1]),torch.exp(self.chparam[:,:,2]),perturbations,constant)

            if type(axis) != type(None):
                Rrs = Rrs[:,axis]
               
            return Rrs.to(self.precision)
            
        else:

            Rrs = Rrs_MODEL(x_data[:,:,0],x_data[:,:,1],x_data[:,:,2],\
                                x_data[:,:,3],x_data[:,:,4],torch.exp(parameters[:,:,0]),torch.exp(parameters[:,:,1]),torch.exp(parameters[:,:,2]),perturbations,constant)
            

            if type(axis) != type(None):
                Rrs = Rrs[:,axis]
                
                
            return Rrs.to(self.precision)
            
######Functions for error propagation########
def error_propagation(df,sigma):

    error_ = df @  sigma @ torch.transpose(df,1,2)
    error = torch.diagonal(error_,dim1=1,dim2=2)
    return error

class evaluate_model_class():
    """
    class to evaluate functions needed to compute the uncerteinty. 
    """
    def __init__(self,model,X,axis=None,constant = None,which_parameters='chla',chla=None):
        self.axis = axis
        self.model = model
        self.X = X
        self.constant = constant
        self.chla = chla
        self.which_parameters = which_parameters
    def model_der(self,parameters_eval,perturbation_factors_ = None):
        
        if type(perturbation_factors_) == type(None):
            perturbations = self.model.perturbation_factors
        else:
            perturbations = perturbation_factors_
            
        if self.which_parameters == 'chla':    
            return self.model(self.X,parameters = parameters_eval,axis = self.axis,perturbation_factors_ = perturbations,constant = self.constant)
        elif self.which_parameters == 'perturbations':
            return self.model(self.X,parameters = self.chla,axis = self.axis,perturbation_factors_ = parameters_eval,constant = self.constant)
        
    def kd_der(self,parameters_eval,perturbation_factors_ = None):
        
        if type(perturbation_factors_) == type(None):
            perturbations = self.model.perturbation_factors
        else:
            perturbations = perturbation_factors_
            

        if self.which_parameters == 'chla': 
            kd_values = kd(9,self.X[:,:,0],self.X[:,:,1],self.X[:,:,2],\
                           self.X[:,:,3],self.X[:,:,4],torch.exp(parameters_eval[:,:,0]),torch.exp(parameters_eval[:,:,1]),torch.exp(parameters_eval[:,:,2]),perturbations,self.constant)
                
        elif self.which_parameters == 'perturbations':
            kd_values = kd(9,self.X[:,:,0],self.X[:,:,1],self.X[:,:,2],\
                           self.X[:,:,3],self.X[:,:,4],torch.exp(self.chla[:,:,0]),torch.exp(self.chla[:,:,1]),torch.exp(self.chla[:,:,2]),parameters_eval,self.constant)
        if self.axis != None:
            kd_values = kd_values[:,axis]
            
        return kd_values

    def bbp_der(self,parameters_eval,perturbation_factors_ = None):

        if type(perturbation_factors_) == type(None):
            perturbations = self.model.perturbation_factors
        else:
            perturbations = perturbation_factors_
            
        if self.which_parameters == 'chla': 
            bbp_values = bbp(self.X[:,:,0],self.X[:,:,1],self.X[:,:,2],\
                             self.X[:,:,3],self.X[:,:,4],torch.exp(parameters_eval[:,:,0]),torch.exp(parameters_eval[:,:,1]),torch.exp(parameters_eval[:,:,2]),perturbations,self.constant)
                
        elif self.which_parameters == 'perturbations':
            bbp_values = bbp(self.X[:,:,0],self.X[:,:,1],self.X[:,:,2],\
                             self.X[:,:,3],self.X[:,:,4],torch.exp(self.chla[:,:,0]),torch.exp(self.chla[:,:,1]),torch.exp(self.chla[:,:,2]),parameters_eval,self.constant)
            
        if self.axis != None:
            return bbp_values[:,axis]
                        
        return bbp_values[:,[1,2,4]]
        


class RRS_loss(nn.Module):

    def __init__(self,x_a,s_a,s_e,precision = torch.float32,num_days:int =1,my_device:str = 'cpu'):
        super(RRS_loss, self).__init__()
        """
        Class to evaluate the loss function RRS_loss = -2log(p(z|x,y)), minus the log posterior distribution of the latent variable z=(chla,nap,cdom). 
        p(z|x,y) uses a Gaussian likelihood, and a Gaussian prior. 
        parameters:
          x_a: mean value of the prior values for chla, nap and cdom, with dimension (3).
          s_a: covariance matrix of the prior for chla, nap and cdom, dimension (3,3).
          s_e: covariance matrix of RRS. Dimension (5,5).
        """
        self.x_a = torch.stack([x_a for _ in range(num_days)]).to(my_device).to(precision)
        self.s_a = s_a.to(my_device).to(precision)
        self.s_e = s_e.to(my_device).to(precision)
        self.s_e_inverse = torch.inverse(self.s_e)
        self.s_a_inverse = torch.inverse(self.s_a)
        self.precision = precision


    def forward(self,y,f_x,x):
        diff_y = y - f_x                  # (num_days, 5)
        diff_x = x[:, 0, :] - self.x_a    # (num_days, 5)

        # Compute batch-wise quadratic forms
        term1 = torch.einsum('bi,ij,bj->b', diff_y, self.s_e_inverse, diff_y)  # (num_days,)
        term2 = torch.einsum('bi,ij,bj->b', diff_x, self.s_a_inverse, diff_x)  # (num_days,)

        total = term1 + term2
        loss = total.mean()
        return loss

class OBS_loss(nn.Module):

    def __init__(self,precision = torch.float32,my_device:str = 'cpu',normalization_values = torch.tensor([1,1,1,1,1,1,1,1,1])):
        super(OBS_loss, self).__init__()
        self.precision = precision
        self.normalization_values = torch.tensor(normalization_values).to(my_device).to(precision)


    def forward(self,Y_l,pred_l,nan_array):
        
        custom_array = ((Y_l-pred_l)/self.normalization_values)**2
        lens = torch.tensor([len(element[~element.isnan()]) for element in nan_array])

        means_output = torch.sum(custom_array,dim=1)/lens
        return means_output.mean().to(self.precision)

if __name__ == "__main__":
    
    pass
