#!/usr/bin/env python
# coding: utf-8

# # JV Curve Analysis

# In[6]:


import math

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp
from scipy.optimize import curve_fit

from importlib import reload
import standard_functions as std


# In[ ]:


reload( std )


# In[7]:


def jv_model_basic( v, a, b, c ):
    """
    A basic diode equation model giving J(V).
    
    :param v: The input voltage.
    :param a: The amplitude.
    :param b: The exponential factor.
    :param c: The offset.
    :returns: a*e^( b* v ) + c
    """
    return a* np.exp( b* v ) + c 


def extrapolate_jv_curve( df, func = jv_model_basic ):
    """
    Extrapolates a JV curve using a given model.
    
    :param df: The DataFrame to extrapolate. 
        Should have np.nan values where extrapolation is needed.
    :param func: the model to use for extrapolation. [Default: Basic Diode Equation]
    :returns: An extrapolated DAtaFrame and the fitted parameters of the model.
    """
    df_fit = df.dropna()
    x_fit = df_fit.index.astype( float ).values
    y_fit = df_fit.current.values

    guess = ( 1, 1, y_fit.min() )
    ( coeffs, covar ) = curve_fit( func, x_fit, y_fit, guess )
    
    # Get the index values for NaNs in the column
    x = df[ pd.isnull( df.current ) ].index.astype( float ).values
    
    # Extrapolate those points with the fitted function
    filled = df.copy()
    filled.loc[ x, 'current' ] = jv_model( x, *coeffs )
    
    return ( filled, coeffs ) 


def get_hysteresis_curves( df ):
    """
    Computes the difference between forward and reverse scan currents at each voltage.
    
    :param df: A JV scan DataFrame.
    :returns: A DataFrame of differences.
    """
    num_rows = 500
    channels = df.columns.get_level_values( 'channel' ).unique()
    scans = df.columns.get_level_values( 'index' ).unique()
    header_metrics =  [ 'voltage', 'reverse', 'forward', 'diff' ]
    header_names = [ 'channel', 'index', 'metric' ]
    
    header = pd.MultiIndex.from_product( [ channels, scans, header_metrics ], names = header_names )
    hysteresis = pd.DataFrame( index = np.arange( num_rows ), columns = header, dtype = np.float32 )
    errors = { 'fit': [], 'other': [] }
    
    for channel in channels:
        channel_scans = df.loc[ :, channel ].columns.get_level_values( 'index' ).unique()
        for scan in channel_scans:
            forw = curves.loc[ :, ( channel, scan, 'forward' ) ].dropna().set_index( 'voltage' )
            revr = curves.loc[ :, ( channel, scan, 'reverse' ) ].dropna().set_index( 'voltage' )
            combined = forw.index.union( revr.index.values )

            forw = forw.reindex( combined ).interpolate( how = 'linear' )
            revr = revr.reindex( combined ).interpolate( how = 'linear' )
            
            try:
                forw = extrapolate_jv_curve( forw )[ 0 ]
                revr = extrapolate_jv_curve( revr )[ 0 ]
                
            except RuntimeError as err:
                print( 'Fit error at Channel {} Scan {}: {}'.format( channel, scan, err ) )
                errors[ 'fit' ].append( ( channel, scan ) )
                
            except Exception as err:
                print( 'Error at Channel {} Scan {}: {}'.format( channel, scan, err ) )
                errors[ 'other' ].append( ( channel, scan ) )
            
            forw.columns = pd.Index( [ 'forward' ], name = 'metric' )
            revr.columns = pd.Index( [ 'reverse' ], name = 'metric' )
            
            diff = revr.join( forw )
            diff = diff.assign( diff = lambda row: row.forward - row.reverse )
            diff.reset_index( inplace = True )
            diff = diff.astype( np.float32 )
            diff.columns = pd.MultiIndex.from_product( 
                [ [ channel ], [ scan ] , header_metrics ], 
                names = header_names 
            )

            hysteresis.loc[ :, ( channel, scan ) ] = diff
    
    return ( hysteresis.dropna( how = 'all' ), errors )


# In[8]:


def hysteresis_area( df ):
    """
    Computes the hysteresis area for a scan.
    
    :param df: A JV scan DataFrame.
    :returns: A Series of areas.
    """    
    areas = []
    for name, data in df.groupby( level = [ 'sample', 'index' ], axis =1  ):
        forward = data.xs( 'forward', level = 'direction', axis = 1, drop_level = False ).dropna()
        reverse = data.xs( 'reverse', level = 'direction', axis = 1, drop_level = False ).dropna().iloc[ ::1 ] # reverse index order
        
        # save headers
        index_for = forward.columns
        index_rev = reverse.columns
        
        # strip headers
        forward = std.keep_levels( forward, 'metric' )
        reverse = std.keep_levels( reverse, 'metric' )

        # create common index for scan directions
        ( forward, reverse ) = std.common_reindex( [ forward, reverse ], index = 'voltage' )
        
        curr_for = forward.values[ :, 0 ]
        volt_for = forward.index.values
        
        curr_rev = reverse.values[ :, 0 ]
        volt_rev = reverse.index.values

        area_for = abs( np.trapz( y = curr_for, x = volt_for ) )
        area_rev = abs( np.trapz( y = curr_rev, x = volt_rev ) )
    
        area_hist = np.sqrt( np.trapz( 
            y = np.square( curr_rev - curr_for ),
            x = volt_for
        ) )

        area = pd.Series( 
            { 
                'forward': area_for, 
                'reverse': area_rev,
                'hyst_abs': area_hist,
                'hyst_rel': area_hist/ max( area_for, area_rev )
                
            },
            name = name
        )
        areas.append( area )
    
    areas = pd.concat( areas, axis = 1 )
    areas.columns.set_names( [ 'sample', 'index' ], inplace = True )
    areas.index.set_names( 'metric', inplace = True )
    areas = areas.stack( level = 'index' ).unstack( 0 ) # move index as index, metrics as columns
    
    return areas

    
def hysteresis_metrics( hysteresis ):
    """
    
    """
    reverse = hysteresis.loc[ :, ( slice( None ), slice( None ), ( 'voltage', 'reverse' ) ) ]   
    forward = hysteresis.loc[ :, ( slice( None ), slice( None ), ( 'voltage', 'forward' ) ) ]
    
    reverse = pd.DataFrame( hysteresis_area( reverse ), columns = [ 'reverse' ] )
    forward = pd.DataFrame( hysteresis_area( forward ), columns = [ 'forward' ] )
    metrics = pd.concat( 
        [ reverse, forward ], 
        axis = 1
    )
    return metrics
    metrics = metrics.reorder_levels( [ 'sample', 'index', 'direction' ], axis = 1 )
    metrics = metrics.sort_index( axis = 1 )
    
    return metrics


# In[9]:


def trim_jv_curves( df, fill = np.nan ) :
    """
    Trims the data values such that all values are in the fourth quadrant
    
    :param df: A Pandas DataFrame to trim. Must have voltage as index
    :returns: The trimmed Pandas DataFrame
    """
    # trim current above 0
    currents = df.xs( 'current', level = 'metric', axis = 1, drop_level = False )
    currents = currents.where( currents <= 0 )
    
    # trim voltages less than 0
    voltages = df.xs( 'voltage', level = 'metric', axis = 1, drop_level = False )
    voltages = voltages.where( voltages >= 0 )
    
    # recombine and remove invalid rows
    df = pd.concat( [ voltages, currents ], axis = 1 ).sort_index( axis = 1 )
    scans = []
    for name, scan in df.groupby( level = [ 'sample', 'index', 'direction' ], axis = 1 ):
        scans.append( scan.dropna().reset_index( drop = True ) )
        
    df = pd.concat( scans, axis = 1 )
    return df
    

def get_power( df ):
    """
    Creates a Pandas DataFrame containing the power of the JV curves
    
    :param df: The DataFrame containing the JV curves
    :returns: A Pandas DataFrame containg the power at each voltage index value
    """
    return - df.mul( df.index, axis = 0 )


def get_mpp( df ):
    """
    Gets the maximum power point
    
    :param df: A Pandas DataFrame containing JV scans to evaluate
    :returns: A Pandas DataFrame with Vmpp with Jmpp and Pmpp
    """
    pdf  = get_power( df )
    pmpp = pdf.max()    # Pmpp
    vmpp = pdf.idxmax() # Vmpp
    jmpp = pmpp/ vmpp   # Jmpp
    
    return pd.concat( [ vmpp, pmpp, jmpp ], keys = [ 'vmpp', 'pmpp', 'jmpp' ], axis = 1 )


def get_jsc( df ):
    """
    Get the short circuit current
    
    :param df: A Pandas DataFrame containing JV sweeps
    :returns: A Pandas Series of short circuit currents
    """
    # TODO: If no 0 index
    jsc = - df.loc[ 0 ]
    return jsc.rename( 'jsc' )


def get_voc( df ):
    """
    Get the open circuit voltage
    
    :param df: A Pandas DataFrame containing JV sweeps
    :returns: A Pandas Series of open circuit voltages
    """
    voc = pd.Series( index = df.columns )
    for col in df:
        tdf =  df.xs( col, axis = 1 )
        pos = tdf[ tdf > 0 ].dropna()
        neg = tdf[ tdf < 0 ].dropna()

        pos_v = pos.idxmin()
        pos_j = pos.loc[ pos_v ]

        neg_v = neg.idxmax()
        neg_j = neg.loc[ neg_v ]

        slope = ( pos_j - neg_j )/ ( pos_v - neg_v )
        voc[ col ] = neg_v - neg_j/ slope

    return voc.rename( 'voc' )



def get_metrics( df ):
    """
    Creates a Pandas DataFrame containing metric about JV curves
    Metrics include maximum power point (vmpp, jmpp, pmpp), open circuit voltage,
    short circuit current, fill factor, and efficiency
    
    :params df: The DataFrame containing the JV curves
    :returns: A Pandas DataFrame containing information about the curves
    """
    metrics = [ get_mpp( df ), get_voc( df ), get_jsc( df ) ]
    metrics = pd.concat( metrics, axis = 1 )
    metrics = metrics.assign( ff = lambda x: x.pmpp/ ( x.voc* x.jsc )  )

    return metrics
    
    
    
def get_pces( df, suns = 1 ):
    """
    Calculate the power efficiency conversion
    
    :param df: A Pandas DataFrame containing JV metrics, with pmpp column
    :param suns: The intensity of the illumination [Default: 1]
    :returns: A DataFrame with pce calculated
    """
    
    return df.assign( pce = lambda x: x.pmpp* 10* suns )


# In[10]:


def plot_metrics( metrics, errors = None ):
    """
    Plots the metrics data
    
    :param metrics: A Pandas DataFrame containing metrics data
        As output by get_metrics()
    :param errors: A pandas DataFrame containing error data
        Indices should match those of means
    """
    
    num_plots = len( metrics.columns )
    cols = math.ceil( math.sqrt( num_plots ) )
    rows = math.ceil( num_plots/ cols )

    fig, axs = plt.subplots( rows, cols, figsize = ( 10, 15 ) )
    x_data = list( map( str, metrics.index.values ) )

    for index in range( num_plots ):
        row = int( index/ cols )
        col = int( index - row* cols )
        ax = axs[ row, col ]

        key = metrics.columns[ index ]
        y_data = metrics[ key ].values
        y_error = errors[ key ].values if errors is not None else None

        ax.bar( x_data, y_data , yerr = y_error )
        ax.set_title( key )
        ax.tick_params( labelrotation = 75 )

    fig.tight_layout()
    plt.show()


# # Work

# In[ ]:




