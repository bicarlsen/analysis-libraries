#!/usr/bin/env python
# coding: utf-8

# # TRPL Analysis
# For analyzing TRPL data


import pandas as pd
import numpy as np

from .. import standard_functions as std


# ## Basic


def remove_background( df, grad_threshold = 1000 ):
    """
    Subtracts background from TRPL signal based on the gradient jump in the initial signal.

    :param df: DataFrame of TRPL signals.
    :param grad_threshold: Gradient threshold to cut background signal at. [Default: 1000]
    :returns: DataFrame with background subtracted from signals.
    """
    df = df.copy()
    grad = df.apply( std.df_grad )
    idx = grad[ grad > grad_threshold ]

    for name, data in df.items():
        tidx = idx[ name ].dropna()
        tdf = data[ :tidx.index.min() ]
        df[ name ] = df[ name ] - tdf.mean()

    return df


def remove_init( df ):
    """
    Removes the initial signal from a TRPL signal.

    :param df: DataFrame of TRPL signals.
    :returns: DataFrame with initial backgroudn signal removed from the data.
    """
    df = df.copy()
    idx = df.idxmax()

    for name, data in df.items():
        df[ name ] = data[ idx[ name ]: ]

    return df


# ## Advanced


def decay_from_rates( df ):
    """
    Creates a TRPL decay spectrum from rates.
    Assumes that the time step between bins is constant.

    :param df: Pandas DataFrame indexed by time with emission rates as values.
    :returns: Pandas DataFrame with relative counts of a TRPL decay.
    """
    intensities = pd.DataFrame( columns = df.columns )
    for time, rate in df.iterrows():
        reach = -df.loc[ :time ].sum()
        reach = reach.apply( np.exp )

        prob = -rate
        prob = 1 - prob.apply( np.exp )
        intensities.loc[ time ] = reach* prob

    return intensities


def rates( df, alpha = 1 ):
    """
    Finds emission rates of a TRPL experiment.
    Assumes the time step between counts is constant.

    :param df: Pandas DataFrame indexed by time, with counts as values.
    :param alpha: Alpha rate of the experiment. [Default: 1]
    :returns: Pandas DataFrame with emission rates for each time step.
    """
    if ( 0 > alpha ) or ( alpha > 1 ):
        # ensure alpha between 0 and 1
        raise ValueError( 'Invalid alpha rate, must be between 0 and 1.' )

    df = df.copy()

    # normalize counts
    df *= alpha / df.sum()

    rates = pd.DataFrame( columns = df.columns )

    for time, row in df.iterrows():
        # rate = -log( 1 - I* exp[ sum[ previous rates ] ] )
        rate = rates.sum()
        rate = rate.apply( np.exp )
        rate *= row
        rate = 1 - rate
        rate = -rate.apply( np.log )

        rates.loc[ time ] = rate

    return rates


# ## Probabilistic Modeling


def signal_from_rates_monte_carlo( rates, trials = 1000, alpha = True ):
    """
    Creates a TRPL signal from given rates.
    Assumes each channel is of same time delta.

    :param rates: Array of rates.
    :param trials: How many experiements to run.
    :param alpha: Whether to return the alpha value. [Default: True]
    :returns: NumPy array of counts by channel.
    """
    if not isinstance( rates, np.ndarray ):
        rates = np.array( rates )

    channel_counts = np.array( [ np.random.poisson( rate, trials ) for rate in rates ] )
    channel_hits = ( channel_counts != 0 ) # whether a hit was registered

    # remove trials with no hits
    valid = np.any( channel_hits, axis = 0 )
    remove = np.argwhere( valid == False ).flatten()
    channel_hits = np.delete( channel_hits, remove, axis = 1 )

    trial_stop = np.argmax( channel_hits, axis = 0 ) # first index at which a count was registered
    channels, counts = np.unique( trial_stop, return_counts = True )
    signal = np.zeros_like( rates )
    for ch, count in zip( channels, counts ):
        signal[ ch ] = count

    if not alpha:
        # don't return alpha value
        return signal

    else:
        alpha = 1 - remove.shape[ 0 ]/ trials
        return ( signal, alpha )


