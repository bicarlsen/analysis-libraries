# PLQY Analysis

import numpy as np
import pandas as pd
import scipy.constants as phys

from .. import standard_functions as stdfn


def plqy_defect_temperature( t, e, a = 1, b = 1 ):
    """
    Returns the temperature dependent PLQY curve
    assuming a single defect level.

    PLQY = A/( 1 + B* T^(1/2)* exp(C/ T)
    C = e/ kB
    e: defect energy offset from band edge

    :param t: Temperature.
    :param e: Defect energy offset from band edge.
    :param a: Numerator coefficient. [Default: 1]
    :param b: Denominator coefficient. [Default: 1]
    :returns: PLQY value.
    """
    k = phys.Boltzmann/ phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ]
    c = e/ k
    return a/( 1 + b* np.sqrt( t )* np.exp( -c/ t ) )


def fit_plqy_defect_temperature( df, **kwargs ):
    """
    Fits a Pandas Series or DataFrame of PLQY data
    indexed by temperature to find an effective defect energy level.
    (See #plqy_defect_temperature for more info.)

    :param df: Pandas Series or DataFrame of PLQY values
        indexed by temperature.
    :param **kwargs: Additional parameters passed to
        scypi.optimize.curve_fit().
    :returns: A Pandas DataFrame with the fit parameters according
        to #plqy_defect_temperature.
    """
    def guess( df ):
        return (
            0,
            df.max() - df.min(),
            1
        )

    fitter = stdfn.df_fit_function(
        plqy_defect_temperature,
        guess = guess,
        **kwargs
    )

    res = fitter( df )
    return res

    return res


