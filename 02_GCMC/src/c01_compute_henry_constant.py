### --- import block --- ###
import numpy as np


### --- method block --- ###
def compute_henry_constant(energies, T, P0=1e5):
    """
    Responsibilities:
    - generate theorical Henry Const.


    """

    # R_GAS = 8.314462618
    # beta = 1.0 / (R_GAS * T)
    beta = 1.0 / T

    KH = np.mean(np.exp(-beta * energies)) / P0

    return KH
