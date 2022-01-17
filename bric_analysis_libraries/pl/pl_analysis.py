#!/usr/bin/env python
# coding: utf-8

# Photoluminescence Analysis

import logging
from numbers import Number
from collections.abc import Iterable

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp
import scipy.constants as phys
from   scipy           import integrate
from   scipy.optimize  import curve_fit
from   scipy.signal    import deconvolve
from   scipy.stats     import linregress

from .. import standard_functions as std


# Helper functions

def convert_beta_temperature( p ):
    """
    Converts beta (coldness) to temperature and vice versa.

    :param p: Temperature in Kelvin or beta (coldness).
    :returns: Beta (coldness) if temperature was given,

        Temperature in Kelvin if beta was given.
    """
    return 1/( phys.physical_constants[ 'Boltzmann constant in eV/K' ][ 0 ]* p )


# ## Distributions


def gaussian_distribution( mu = 0, sigma = 1, x = None ):
    """
    A Normal or Gaussian distribution

    :param mu: The mean, or None [Default: 0]
    :param sigma: The standard deviation, or None [Default: None]
    """
    return np.exp( -np.power( ( x - mu ), 2 )/( 2* np.power( sigma, 2 ) ) )


def boltzmann_distribution( t = 300, e = None ):
    """
    The Boltzmann distribution at a given temperature

    :param t: The temperature in Kelvin, or None [Default: 300]
    :param e: The energy in eV, or None [Default: None]
    :returns: Returns a function describing the Boltmann distribution
        as a function of energy and or temperature, if either are None;
        or the value if both are not None
    """
    a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ]  # J
    k = phys.Boltzmann/ a

    if ( T is None ) and ( E is None ):
        # function of energy and temperature
        boltzmann = lambda E, T: np.exp( -E/( k* T ) )

    elif ( T is None ) and ( E is not None ):
        # function of temperature only
        boltzmann = lambda T: np.exp( -e/( k* T ) )

    elif ( T is not None ) and ( E is None ):
        # function of energy only
        boltzmann = lambda E: np.exp( -E/( k* t ) )

    else:
        # both values passed, return value
        boltzmann = np.exp( -e/( k* t ) )

    return boltzmann


def fermi_distribution( ef = 0, t = 300, e = None ):
    """
    The Fermi distribution at a given temperature and Fermi energy

    :param ef: The Fermi energy of the system in eV, or None [Default: 0]
    :param t: The temperature at which to calculate the distribution in K, or None [Default: 300]
    :param e: The energy at whcih to calculate the distribution in eV, or None [Default: None]
    :returns: A function representing the Fermi distribution taking temperature and or energy
        as parameters, or returning a value if both are specified
    """
    a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ] # J
    k = phys.Boltzmann/ a

    if ( ef is None ) and ( t is None ) and ( e is None ):
        # function of Ef, T, and E
        fermi = lambda Ef, T, E: np.power( 1 + np.exp( ( E - Ef )/( k* T ) ), -1 )

    elif ( ef is None ) and ( t is None ) and ( e is not None ):
        # function of Ef and T
        fermi = lambda Ef, T: np.power( 1 + np.exp( ( e - Ef )/( k* T ) ), -1 )

    elif ( ef is None ) and ( t is not None ) and ( e is None ):
        # function of Ef and E
        fermi = lambda Ef, E: np.power( 1 + np.exp( ( E - Ef )/( k* t ) ), -1 )

    elif ( ef is None ) and ( t is not None ) and ( e is not None ):
        # function of Ef
        fermi = lambda Ef: np.power( 1 + np.exp( ( e - Ef )/( k* t ) ), -1 )

    elif ( ef is not None ) and ( t is None ) and ( e is None ):
        # function of E and T
        fermi = lambda T, E: np.power( 1 + np.exp( ( E - Ef )/( k* T ) ), -1 )

    elif ( ef is not None ) and ( t is None ) and ( e is not None ):
        # function of T
        fermi = lambda T: np.power( 1 + np.exp( ( e - Ef )/( k* T ) ), -1 )

    elif ( ef is not None ) and ( t is not None ) and ( e is None ):
        # function of E
        fermi = lambda E: np.power( 1 + np.exp( ( E - ef )/( k* t ) ), -1 )

    else:
        # value
        fermi = n.power( 1 + np.exp( ( e - Ef )/( k* t ) ), -1 )

    return fermi


def dos( e0 = 0, e = None ):
    """
    The density of states

    :param e0: An energy in eV, or None [Default: 0]
    :param e: An energy in eV, or None [Default: None]
    :returns: A function that takes in an energy and reference energy and

        returns the density of states function or a value
    """

    if ( e0 is None ) and ( e is None ):
        # function of e and e0
        density = lambda E, E0: np.sqrt( E - E0 )

    elif ( e0 is None ) and ( e is not None ):
        # function of e0
        density = lambda E0: np.sqrt( e - E0 )

    elif ( e0 is not None ) and ( e is None ):
        # function of e
        density = lambda E: np.sqrt( E - e0 )

    else:
        # value
        density = np.sqrt( e - e0 )

    return density


# TODO: Multiply functions together
def population( ef = 0, t = 300, e0 = 0, e = None ):
    """
    Returns a function or value representing the distribution of the population at a given energy

    :param ef: The Fermi energy in eV, or None [Default: 0]
    :param t: The temperature in K, or None [Default: 300]
    :param e0: The density of states base energy in eV, or None [ Default: 0 ]
    :param e: The energy to evaluate the system at, or None [Default: None]
    :returns: A function representing the population that takes as parameters

        the values passed as None.
        If all values are specified, returns the value directly.
    """

    return fermi_distribution( ef, t, e )* boltzmann_distribution( t, e )* dos( e0, e )

# ## Spectral analysis


def fwhm( df, start = None, end = None ):
    """
    Calculates the full width at half max (fwhm)

    :param df: The Pandas DataFrame with spectral data to analyze
    :param start: Lower bound of the search range, or None. [Default: None]
    :param end: Upper bound of the search range, or None. [Default: None]
    :returns: A Pandas Series containing the full width at half max data
    """
    fwhm = []
    cols = []
    peaks = peak_position( df, start, end )

    for name, data in df.items():
        peak = peaks.xs( name )
        if isinstance( peak, pd.DataFrame ) or isinstance( peak, pd.Series ):

            peak = peak.values[ 0 ]

        data = data.loc[ start : end ]
        hm = data.max()/ 2
        hml = ( data.loc[ :peak ] - hm ).abs().idxmin()
        hmr = ( data.loc[ peak: ] - hm ).abs().idxmin()

        fwhm.append( abs( hmr - hml ) )
        cols.append( name )

    if type( df.columns ) is pd.MultiIndex:
        cols = pd.MultiIndex.from_tuples( cols, names = df.columns.names )

    else:
        # basic index
        cols = pd.Index( cols, name = df.columns.name )

    fwhm = pd.Series( fwhm, index = cols )
    return fwhm.rename( 'fwhm' )


def peak_position( df, start = None, end = None ):
    """
    Finds the peak position of a spectrum.

    :param df: A Pandas DataFrame containing the spectrum.
    :param start: Lower bound of the search range, or None. [Default: None]
    :param end: Upper bound of the search range, or None. [Default: None]
    :returns: A Pandas DataSeries container positions of the max.
    """
    peak = df.loc[ start : end ].idxmax()
    if not isinstance( peak, pd.Series ):
        # single columns passed in, transform back to Series
        peak = pd.Series( peak, index = df.index, name = 'peak' ).rename( 'peak' )

    return peak


def center_of_mass( df ):
    """
    Retuns a Pandas Series of the center of mass for the spectra.

    :param df: DataFrame to calaculate the center of mass on.
    :returns: Pandas Series of center of masses.
    """
    idx = df.index
    weights = df/ df.sum()
    com = weights.multiply( idx, axis = 0 ).sum()
    return com


def integrated_intensity( df, start = None, end = None ):
    """
    Calculates the integrated intensity (area under the curve) of a spectrum.

    :param df: A Pandas DataFrame containing the spectrum.
    :param start: Lower bound of the integration, or None. [Default: None]
    :param end: Upper bound of the integration, or None. [Default: None]
    :returns: A Pandas DataFrame of spectral areas.
    """
    df = df.loc[ start : end ]
    return df.apply( lambda datum: integrate.simps( datum ) ).rename( 'area' )


def peak_analysis( df, groups = None, start = None, end = None ):
    """
    Performs analysis on the peak positions of the spectra.

    :param df: A Pandas DataFrame containing spectrum.
    :param groups: How to group the data based on the DataFrame's index. [Default: None]
    :param start: Lower bound of the search range, or None. [Default: None]
    :param end: Upper bound of the search range, or None. [Default: None]
    :returns: A Pandas DataFrame containing analysis of the peak positions.
        If groups is None, return peak and fwhm of each sample [peak, fhwm]
        If groups is not None, return mean and standard deviation of

        peak position and full width at half max for each group
        [ [peak, fwhm], [mean, std] ]
    """
    peaks = peak_position( df, start, end )
    fw    = fwhm( df, start, end )
    area  = integrated_intensity( df, start, end )

    if groups is None:
        # return data sample by sample, no statistics
        return pd.concat(

            [ peaks, fw, area ],

            axis = 1,

            keys = [ 'peak', 'fwhm', 'area' ]

        )

    # group analysis, include statistics
    peaks = peaks.groupby( groups )
    fw    = fwhm.groupby(  groups )
    area  = area.groupby(  groups )

    return pd.concat( [

        peaks.mean().rename( ( 'peak', 'mean' ) ),
        peaks.std().rename(  ( 'peak', 'std' ) ),
        fw.mean().rename( ( 'fwhm', 'mean' ) ),
        fw.std().rename(  ( 'fwhm', 'std' ) ),
        area.mean().rename( ( 'area', 'mean' ) ),
        area.std().rename(  ( 'area', 'std' ) )
    ], axis = 1 )


def extract_temperature(
    df,
    value_threshold = 0,
    grad_threshold = 10,
    curve_threshold = 1e5,
    side = 'high',
    mask_window = 75,
):
    """
    Finds the temperature coefficient from a PL curve.
    Performs a linear fit on the log of PL spectra on the low or high energy side.
    The fit is performed on an area with gradient higher than the given threshold,
    and curvature less that the given threshold.

    :param df: DataFrame of PL specra indexed by energy.
    :param value_threshold: Minimum value relative to max to consider.
        [Default: 0]
    :param grad_threshold: Minimum gradient threshold. [Default: 40]
    :param curve_threshold: Maximum curvature threshold. [Default: 1000]
    :param side: 'low' for low energy, 'high' for high energy. [Default: 'high']
    :param mask_window: Smoothing window for data mask. [Default: 75]
    :returns: Dictionary of tuples of ( temperature, linear fit ).
        If no valid data for a particular dataset vlaue is None.

    """
    logger = logging.getLogger( __name__ )
    df = df.copy()

    # calculate needed data
    ldf = df.apply( np.log ).replace( [ -np.inf, np.inf ], np.nan ).dropna( how = 'all' )
    gdf = ldf.apply( std.df_grad )
    cdf = gdf.apply( std.df_grad )

    fits = {}
    for name, data in ldf.items():
        mask = (
            data.index < data.idxmax()
            if side == 'low' else
            data.index > data.idxmax()
        )

        if not np.any( mask ):
            # no valid data
            fits[ name ] = None
            logger.info( f'No data for { name } due to side mask.' )
            continue
        
        v_mask = ( 
            ldf[ name ][ mask ].apply( np.exp ) > value_threshold
            if value_threshold > 0 else
            ldf[ name ][ mask ] > -np.inf
        )

        g_mask = (
            gdf[ name ][ mask ] > grad_threshold
            if side == 'low' else
            gdf[ name ][ mask ] < -grad_threshold
        )

        c_mask = (
            ( 0 > cdf[ name ][ mask ] ) &
            ( cdf[ name ][ mask ] > -curve_threshold )
        )

        v_mask = std.smooth_mask( v_mask, window = mask_window )
        g_mask = std.smooth_mask( g_mask, window = mask_window )
        c_mask = std.smooth_mask( c_mask, window = mask_window )

        tdf = data[ mask ]
        tdf = tdf[ v_mask & g_mask & c_mask ]
        tdf = tdf.dropna()
        
        if tdf.shape[ 0 ] == 0:
            # no data
            fits[ name ] = None
            logger.info( f'No data for { name } due to masking.' )
            continue

        # valid data, fit
        fit = linregress( x = tdf.index.values, y = tdf.values )
        if np.isnan( fit.slope ):
            # could not fit
            fits[ name ] = None
            logger.info( f'Could not fit { name }.' )
            continue

        beta = fit.slope
        if side == 'high':
            beta *= -1

        temp = convert_beta_temperature( beta )
        fits[ name ] = ( temp, fit )

    return fits


def differential_temperature( df, window = 11, normalize = True ):
    """
    Extracts the differential temperature from a DataFrame.

    :param df: Pandas DataFrame of PL spectra.
    :param window: Window size for linear fitting.
        [Default: 11]
    :returns: Pandas DataFrame of differential temperatures.
    :raises ValueError: If window is smaller than 1.
    :raises ValueError: If window is not odd valued.
    """
    def _temp_fit( row, data ):
        ind_loc = data.index.get_loc( row.name )
        tdf = data.iloc[ ind_loc - half_window : ind_loc + half_window + 1 ]
        fit = linregress( tdf.index, tdf.values )
        temp = -convert_beta_temperature( fit.slope )
        return ( temp, fit )
    
    if window < 2:
        raise ValueError( 'Window must be larger than 1.' )

    if window% 2 == 0:
        raise ValueError( 'Window must be odd.' )

    half_window = int( ( window - 1 )/ 2 )
    ldf = df.apply( np.log )
    fdf = []
    for name, data in ldf.items():
        tdf = data.dropna().to_frame()
        tdf = tdf.iloc[ half_window : -half_window ]
        tdf = tdf.apply(
            _temp_fit,
            axis = 1,
            args = ( data, ),
            result_type = 'expand'
        )
        
        tdf = tdf.rename( { 0: 'temperature', 1: 'fit' }, axis = 1 )
        headers = [
            ( *name, val ) if isinstance( name, Iterable ) else ( name, val )
            for val in  tdf.columns.values
        ]
        
        tdf.columns = pd.MultiIndex.from_tuples(
            headers,
            names = ( *df.columns.names, 'fits' )
        )
        
        fdf.append( tdf )
    
    if len( fdf ) > 1:
        fdf = pd.concat( fdf, axis = 1 )
    
    else:
        fdf = fdf[ 0 ]

    return fdf


def differential_temperature_stats(
    df,
    grad_threshold = 10,
    curve_threshold = 1e5,
    side = 'high',
    mask_window = 75,
):
    """
    Returns statistics on differential temepratures.
    Used in conjunction with #differential_temperature

    :param df: DataFrame of differential temperatures indexed by energy.
    :param grad_threshold: Maximum gradient threshold. [Default: 40]
    :param curve_threshold: Maximum curvature threshold. [Default: 1000]
    :param side: 'low' for low energy, 'high' for high energy. [Default: 'high']
    :param mask_window: Smoothing window for data mask. [Default: 75]
    :returns: Dictionary of tuples of ( temperature, linear fit ).
        If no valid data for a particular dataset vlaue is None.

    """
    logger = logging.getLogger( __name__ )
    df = df.copy()

    # calculate needed data
    gdf = df.apply( std.df_grad )
    cdf = gdf.apply( std.df_grad )

    stats = {}
    for name, data in df.items():
        g_data = gdf[ name ]
        
        g_mask = g_data.abs() < grad_threshold
        g_mask = std.smooth_mask( g_mask, window = mask_window )

        c_mask = cdf[ name ].abs() < curve_threshold 
        c_mask = std.smooth_mask( c_mask, window = mask_window )

        mask  = (
            g_data.index < g_data.idxmin()
            if side == 'low' else
            g_data.index > g_data.idxmax()
        )
        
        tdf = data[ mask & g_mask & c_mask ]
        tdf = tdf.dropna()

        if tdf.shape[ 0 ] == 0:
            # no data
            mean = None
            stddev = None
            floor = None

        else:
            mean = tdf.mean()
            stddev = tdf.std()
            floor = (
                tdf.max()
                if side == 'low' else
                tdf.min()
            )        

        if not isinstance( name, Iterable ):
            # normalize name to tuple if required
            name = ( name, )

        stats[ ( *name, 'mean' ) ] = mean
        stats[ ( *name, 'std' ) ] = stddev
        stats[ ( *name, 'min' ) ] = floor

    stats = pd.Series( stats )
    stats.index = stats.index.rename( ( *df.columns.names, 'metrics' ) )
    return stats



# ## Spectral functions


def intensity_ideal_population( Eg, t = 300):
    """
    The PL intensity predicted for an ideal direct bandgap material.
    ( e - Eg )^2 Exp( -beta ( e - Eg ) )

    :param Eg: The bandgap energy.
    :param t: Temperature in Kelvin. [Default: 300]
    :returns: A function of wavelength energy for the given bandgap and temperature.
    """
    def intensity( e ):
        """
        The PL intensity predicted for an ideal direct bandgap material.
        ( e - Eg )^2 Exp( -beta ( e - Eg ) )

        :param e: The wavelength energies at which to evaluate the intensity.
        :returns: The predicted intensities.
        """
        a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ] # J
        k = phys.Boltzmann/ a

        beta = 1/( k* t )
        delta = e - Eg

        return np.piecewise( delta,
            [ delta > 0 ],
            [
                lambda x: np.square( x )* np.exp( -beta* x ),
                lambda x: 0
            ]
        )

    return intensity


def intensity_gaussian_population( Eg0, sigma, t = 300 ):
    """
    The PL intensity predicted for a direct bandgap material 
    with Gaussian noise applied to its bandgap.
    Uses an asymptotic approximation of the true function.

    True:
    ( sigma / 2 ) Exp( -phi ) {
        - 2 sigma e
        + 2 sigma ( zeta - e )( 1 - Exp( ( e/sigma^2 )( zeta - e/2 ) ) )
        + Sqrt( 2 pi ) Exp( zeta/( 2 simga^2 ) )( sigma^2 + ( zeta - e )^2 )
            ( Erfc( ( zeta - e )/( Sqrt( 2 ) sigma ) ) - Erfc( zeta/( Sqrt( 2 ) sigma ) ) )
    }

    phi = beta e + Eg0/( 2 sigma^2 )
    zeta = Eg0 + beta sigma^2

    Approximation:
    sigma^2 ( delta - shift ) Exp( -delta^2/( 2 sigma^2 ) ) +
    Sqrt( pi/2 ) sigma ( ( delta - shift )^2 + sigma^2 ) *
        Exp( -beta( delta - shift/ 2 ) ) *
        Erfc( -( delta - shift )/ Sqrt( 2 sigma^2 ) )

    e: Wavelength energy
    Eg0: Center bandgap energy.
    sigma: Standard deviation of bandgap energy.
    beta: Coldness
    shift: beta sigma^2

    :param Eg0: Center of the bandgap energy distribution.
    :param sigma: Standard deviation of the bandgap energies.
    :param t: Temperature in Kelvin. [Default: 300]
    :returns: A function of wavelength energy for the given

        bandgap center and deviation, and temperature.
    """
    def intensity( e ):
        """
        The PL intensity predicted for a direct bandgap material with Gaussian noise

        applied to its bandgap.
        Uses an asymptotic approximation of the true function.

        Approximation:
        sigma^2 ( delta - shift ) Exp( -delta^2/( 2 sigma^2 ) ) +
        Sqrt( pi/2 ) sigma ( ( delta - shift )^2 + sigma^2 )

            Exp( -beta( delta - shift/ 2 ) ) Erfc( -( delta - shift )/ Sqrt( 2 sigma^2 ) )

        :param e: The wavelength energies at which to evaluate the intensity.
        :returns: The predicted intensities.
        """
        # helper variables
        a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ] # J
        k = phys.Boltzmann/ a

        var = np.square( sigma )
        beta = 1/( k* t )
        delta = e - Eg0
        shift = beta* var

        p1 = var*( delta - shift )* np.exp( -np.square( delta )/( 2* var ) )
        p2 = (
            np.sqrt( np.pi/ 2 )* sigma*
            ( np.square( delta - shift ) + var )*
            np.exp( -beta*( delta - shift/2 ) )*
            sp.special.erfc( -( delta - shift )/ np.sqrt( 2* var ) )
        )

        return ( p1 + p2 )

    return intensity


def fit_gaussian( df ):
    """
    Fit a Gaussian to the data

    :param df: A Pandas DataFrame inedxed by wavelength with spectral data
    :returns: A Pandas DataFrame of the fit parameters
    """

    gaussian = lambda x, A, mu, sigma: (

        A* np.exp( -np.power( ( x - mu ), 2 )/( 2* np.power( sigma, 2 ) ) )
    )

    fw = fwhm( df )
    guess = lambda data: ( data.max(), data.idxmax(), fw.loc[ data.name ] )

    fit = std.df_fit_function( gaussian, guess = guess )
    return fit( df )


def fit_lorentzian( df ):
    """
    Fit a Lorentzian to the data

    :param df: A Pandas DataFrame indexed by wavelength with spectral data
    :returns: A Pandas DataFrame containing the fit parameters
    """
    lorentzian = lambda x, A, x0, gamma: (
        A* gamma / ( np.power( ( x - x0 ), 2 ) + np.power( gamma, 2 ) )
    )

    fw = fwhm( df )
    guess = lambda data: ( data.max(), data.idxmax(), fw.loc[ data.name ] )

    fit = std.df_fit_function( lorentzian, guess = guess )
    return fit( df )


def fit_gaussian_tails( df ):
    """
    Fit a Gaussian with exponential tails

    :param df: A Pandas DataFrame inedxed by wavelength with spectral data
    :returns: A Pandas DataFrame of the fit parameters
    """

    # Create peicewise function with variable transition points
    # Passed data is log of original, so exponentials are linear and
    # Gaussian is quadratic
    gaussian_tails = lambda x, lim_low, lim_high, ml, mh, x0l, x0h, A, mu, sigma: (
        np.piecewise( x,
            [ x < lim_low, x > lim_high, ( x >= lim_low )*( x <= lim_high ) ],
            [
                lambda x: ml*( x - x0l ), # left tail
                lambda x: mh*( x - x0h ), # right tail
                lambda x: A - np.power( x - mu, 2 )/( 2* np.power( sigma, 2 ) ) # gaussian
            ]
        ) )

    fw = fwhm( df )
    guess = lambda data: (
        data.idxmax() - fw.loc[ data.name ]/ 2,  # low limit at half max
        data.idxmax() + fw.loc[ data.name ]/ 2,  # high limit at half max
        np.power( fw.loc[ data.name ], -1 ),  # At half max the slopes of the logs are 1
        -np.power( fw.loc[ data.name ], -1 ),
        data.idxmax() - fw.loc[ data.name ],  # low intercept, guess full width shift
        data.idxmax() + fw.loc[ data.name ],  # high intercept, guess full width shift
        data.max(), data.idxmax(), fw.loc[ data.name ]  # gaussian parameters

    )

    modify = lambda df: df.apply( np.log10 ).replace( -np.inf, np.nan )

    fit = std.df_fit_function( gaussian_tails, guess = guess, modify = modify )
    return fit( df )


def fit_intensity_ideal_population( df, temp = 300 ):
    """
    Fits the ideal population intensity function to the data
        I( E, E0, beta ) = ( E - E0 )^2 exp( - beta* ( E - E0 ) )

    :param df: A Pandas DataFrame inedxed by energy with spectral data
    :param temp: The temperature of the experiment in K [Default: 300]
    :returns: A Pandas DataFrame of the fit parameters
    """
    a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ]  # J
    k = phys.Boltzmann/ a

    intensity = lambda e, A, e0, t: np.piecewise( e,
        [ e > e0 ],
        [
            lambda e: A* np.power( e - e0, 2 )* np.exp( -( e - e0 )/( k* t ) ),
            lambda e: 0
        ]
    )

    # max of ideal fit is pi/( 2 beta^2) e^( -2 ) = 1.6e-9 T^2
    guess = lambda data: (
        data.max()/( 1.6e-9* np.power( temp, 2) ), # A
        data.idxmax(), # e0
        temp # t
    )

    fit = std.df_fit_function( intensity, guess = guess )
    return fit( df )


def fit_intensity_gaussian_population( df, temp = 300 ):
    """
    Fits the population intensity with Gaussian noise function to the data
        I0( E, E0, beta ) = ( E - E0 )^2 exp( - beta* ( E - E0 ) )
        I( E, E0, sigma, beta) = int( 0, E ) N( mu = E0, sigma ) I0( E, E0, beta ) dE0

    :param df: A Pandas DataFrame inedxed by energy with spectral data
    :param temp: The temperature of the experiment in Kelvin. [Default: 300]
    :returns: A Pandas DataFrame of the fit parameters
    """

    # intensity
    def intensity( e, A, Eg0, sigma, t ):
        # helper variables
        a = phys.physical_constants[ 'electron volt-joule relationship' ][ 0 ]  # J
        k = phys.Boltzmann/ a

        var = np.square( sigma )
        beta = 1/( k* t )
        delta = e - Eg0
        shift = beta* var

        p1 = var*( delta - shift )* np.exp( -np.square( delta )/( 2* var ) )
        p2 = (
            np.sqrt( np.pi/ 2 )* sigma*
            ( np.square( delta - shift ) + var )*
            np.exp( -beta*( delta - shift/2 ) )*
            sp.special.erfc( -( delta - shift )/ np.sqrt( 2* var ) )
        )

        return A*( p1 + p2 )

    fw = fwhm( df )
    guess = lambda data: (
        10e5* data.max(),     # A
        data.idxmax(),        # Eg0
        fw.loc[ data.name ],  # sigma
        temp                  # t
    )

    fit = std.df_fit_function( intensity, guess = guess, bounds = ( 0, np.inf ), maxfev = 1000 )
    return fit( df )


def fit_voigt( df ):
    def voigt( x, alpha, gamma ):
        """
        Return the Voigt line shape at x with Lorentzian component HWHM gamma
        and Gaussian component HWHM alpha.
        """
        sigma = alpha / np.sqrt( 2 * np.log( 2 ) )

        return np.real(
            sp.specialwofz(
                ( x + 1j* gamma )/ sigma/ np.sqrt( 2 )
            )

        )/ sigma/ np.sqrt( 2* np.pi )


def bandgap_distribution( df, temperature = 300, freq_kernel = None ):
    """
    Finds the distribution of bandgaps from the PL spectrum.
    Deconvolves the ideal crystal PL spectrum from the signal.

    :param df: Pandas DataFrame of spectrum indexed by energy.
    :param temperature: Temperature in Kelvin of the sample.
    :param freq_kernel: Frequency kernel of Fourier Transorm.

        If a number cuts frequencies above the given value.
        If callable should accept a NumPy.array of frequencies and

            return an array of the same length indicating the

            relative value of each frequency.
        If None, performs no filtering.
        [Default: None]
    :returns: Pandas DataFrame representing band gap distribution.
    """

    # sort and normalize
    df = df.sort_index()/ df.max()

    # resample signal for equal spacing
    index = df.index
    step = np.min( np.diff( index ) )
    new_index = np.arange( index.min(), index.max() + step, step )
    combined_index = np.unique( np.concatenate( ( new_index, index.values ) ) )

    df = df.reindex( combined_index ).interpolate().reindex( new_index )

    # fourier transform data
    vals = df.values.ravel()
    ftdf = np.fft.fft( vals )

    # --- get sample frequency
    freq = np.fft.fftfreq( vals.shape[ -1 ], step )

    # sample ideal crystal spectrum at same frequency
    intensity = intensity_ideal_population(

        df.index.min(),
        t = temperature
    )
    kernel = intensity( df.index.values )
    kernel = np.fft.fft( kernel )

    # deconvolve
    ftdf = np.divide( ftdf, kernel )

    # frequency threshold filter
    if freq_kernel is not None:
        if isinstance( freq_kernel, Number ):
            # threhsold is a number
            sig_freq_kern = ( np.abs( freq ) > freq_kernel )
            ftdf[ sig_freq_kern ] = 0

        elif callable( freq_kernel ):
            sig_freq_kern = freq_kernel( freq )
            ftdf *= sig_freq_kern

        else:
            raise TypeError( 'Invalid frequency kernel.' )

    # inverse fourier transform
    iftdf = np.fft.ifft( ftdf )
    df = pd.Series( iftdf.real, index = new_index )

    return df


def absorption_from_bandgap_distribution( df, absorption = None ):
    """
    Calculates the absorption curves from a band gap distribution.
    Convolves the absorption with the band gap distribution.

    :param df: DataFrame indexed by energy container band gap distribution.
    :param absorption: Absorption function.

        Accepts energy and NumPy array of index energies as parameters.
        Returns NumPy array of absorption values corresponding to energies of index
        [Default: ( e - index )^(1/2) / index]
    :returns: DataFrame of absorption.
    """
    if absorption is None:
        # default absorption
        def abs_default( energy, index ):
            energies = np.maximum( eps - index, 0 )  # only take positiv values
            abs_dos = np.sqrt( energies )

            # account for additional eps term
            abs_dos = abs_dos/ df.index
            return abs_dos

        absorption = abs_default

    df_values = np.nan_to_num( df.values )
    adf = []
    for eps in df.index:
        abs_dos = absorption( eps, df.index )
        integrand = df_values * abs_dos
        value = np.trapz( integrand, df.index )

        adf.append( value )

    adf = pd.Series( adf )
    adf.index = df.index.copy()
    adf = adf.rename( df.name )

    return adf


def df_bandgap_distributions(
    df,
    temperature = 300,
    temperature_level = None,
    freq_kernel = None
):
    """
    Finds the distribution of bandgaps from the PL spectrum.
    Deconvolves the ideal crystal PL spectrum from the signal.

    :param df: Pandas DataFrame of spectrum indexed by energy.
    :param temperature: Temperature in Kelvin to evaluate the bandgap at.
        If temperature_level is not None, this is ignored.
        [Default: 300]
    :param temperature_level: Index level of temperature information,
        or None to use a static temperature. [Default: None]
    :param freq_kernel: Frequency kernel of Fourier Transorm.

        If a number cuts frequencies above the given value.
        If callable should accept a NumPy.array of frequencies and

            return an array of the same length indicating the

            relative value of each frequency.
        If None, performs no filtering.
        [Default: None]
    :returns: Pandas DataFrame representing band gap distribution.
    """
    dists = []
    for index in df:
        sample_temp = (
            temperature
            if temperature is not None else
            index[ temperature_level ]
        )

        data = df[ index ]
        data = bandgap_distribution(

            data,

            temperature = sample_temp,

            freq_kernel = freq_kernel

        )

        data = data.rename( index )
        dists.append( data )

    dists = pd.concat( dists, axis = 1 )
    dists /= dists.max()

    return dists


def df_absorption_from_bandgap_distributions( df, absorption = None ):
    """
    Calculates the absorption curves from a bandgap distribution DataFrame.

    :param df: DataFrame indexed by energy with distribution values.
    :param absorption: Absorption function or None to use default

        (see #absorption_from_bandgap_distribution).
        [Default: None]
    :returns: DataFrame indexed by energy of absorption values.
    """
    kwargs = {}

    adf = []
    for index in df:
        data = df[ index ]
        data = absorption_from_bandgap_distribution( data, absorption = absorption )
        adf.append( data )

    adf = pd.concat( adf, axis = 1 )
    adf /= adf.max()
    return adf
