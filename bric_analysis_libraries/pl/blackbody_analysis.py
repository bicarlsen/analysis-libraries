# pl / blackbody analysis

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.constants as phys
from scipy.stats import linregress
from scipy.signal import savgol_filter as savgol

from .. import standard_functions as std


def slope_to_beta( slope, side = 'high' ):
    """
    Converts a linear fit slope to beta.
    Beta and slope are same value,
    just determines if it needs to be multiplied by -1
    based on the side.

    :param slope: Slope to convert.
    :param side: High or low energy.
        Valid values: [ 'high', 'low' ]
        [Default: 'high']
    """
    if side == 'high':
        slope *= -1

    return slope


def convert_beta_temperature( p ):
    """
    Converts beta (coldness) to temperature and vice versa.

    :param p: Temperature in Kelvin or beta (coldness).
    :returns: Beta (coldness) if temperature was given,

        Temperature in Kelvin if beta was given.
    """
    return 1/( phys.physical_constants[ 'Boltzmann constant in eV/K' ][ 0 ]* p )


def linear_fit_threshold(
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

        beta = slope_to_beta( fit.slope, side = side )
        temp = convert_beta_temperature( beta )
        fits[ name ] = ( temp, fit )

    return fits


# differential temperature

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


# use ML to find best fit

def default_linear_fit_error(
    df,
    inflection_pts = None,
    full_width = None,
    grid = ( 10, 10 ),
    weights = np.ones( 4 ),
    side = 'high'
):
    """
    Performs a linear fit for a variety of ( center, width )
    values on the data.

    Uses the relative intergrated fit error, relative fit width,
    distance form an inflection point, and relative slope
    to calculate the fitness of the fit.

    :param df: pandas.Series to fit.
    :param inflection_pts: List of inflection points.
        If None, inflection points are ignored.
        [Default: None]
    :param full_width: Full width of the data.
        If None the width of the passed in data is used.
        [Default: None]
    :param grid: Grid size to break the ( center, width ) space into.
        [Default: ( 10, 10 )]
    :param weights: numpy array to weight the errors.
        ( relative error area, relative width, inflection point distance, relative slope )
        [Default: ( 1, 1, 1, 1 )]
    :param side: Fitting high side or low energy side.
        Valid values: [ 'high', 'low' ]
        [Default: 'high']
    :returns: Dictionary keyed by tuples ( center, width ) with values of the error.
    """
    n_segments, n_widths = grid
    
    x_min = df.index.min()
    x_max = df.index.max()
    x_rng = x_max - x_min
    
    # centers
    segment_width = x_rng/ n_segments
    centers = [
        x_min + ( n + 0.5 )* segment_width
        for n in range( n_segments )
    ]
        
    # widths
    width_size = x_rng/ n_widths
    widths = [
        ( n + 0.5 )* width_size
        for n in range( n_widths )
    ]
    
    max_width = (
        max( widths )
        if full_width is None else
        full_width
    )
    
    # combine 
    grid_pts = [
        ( center, width )
        for center in centers
        for width in widths
        if (  # ensure bounds
            center - width/ 2 >= x_min and
            center + width/ 2 <= x_max
        )
    ]
    
    # fits
    fits = {}
    for x0, w0 in grid_pts:
        tdf = df[
            ( df.index >= x0 - w0/ 2 ) &
            ( df.index <= x0 + w0/ 2 )
        ].dropna()
        
        if tdf.shape[ 0 ] < 2:
            continue
        
        x = tdf.index
        y = tdf.values
        fit = linregress( x, y )
        if np.isnan( fit.slope ):
            continue

        fits[ ( x0, w0 ) ] = ( fit, tdf )
        
    if len( fits ) == 0:
        return None
    
    # errors
    slopes = [
        fit.slope
        for _, ( fit, _ ) in fits.items()
    ]

    ref_slope = (  # most extreme slope
        max( slopes )
        if side == 'low' else
        min( slopes )
    )
    
    errors = {}
    for ( x0, w0 ), ( fit, tdf ) in fits.items():
        x = tdf.index
        y = tdf.values
        y_fit = fit.slope* x + fit.intercept
        
        rms = np.trapz( np.absolute( y_fit - y ), x )
        ref_area = np.trapz( tdf - tdf.min(), tdf.index )
        if not ref_area:
            rms_err = 1

        else:
            rms_err = rms/ ref_area

        width_err = 1 - w0/ max_width
        
        slope_err = abs( ( fit.slope - ref_slope )/ ref_slope )

        if inflection_pts is not None:
            inflection_err = np.nanmin( [
                x0 - xi
                for xi in inflection_pts
            ] )
            
            infelction_err = abs( inflection_err/ x_rng )
            
        else:
            # no inflection points given
            inflection_err = 0
        
        errs = (
            rms_err,
            width_err,
            inflection_err,
            slope_err
        )
        energy = np.dot( weights, errs )/ np.sum( weights )
        errors[ ( x0, w0 ) ] = energy
    
    return errors


def linear_fit_search(
    df,
    grid = ( 10, 10 ),
    weights = None,
    error_fn = None,
    tolerance = 1e-2,
    maxiter = 100,
    side = 'high',
    history = False
):
    """
    Use a simple search in ( center x width ) space
    to find the best linear fit for the data quantified by an error function.

    :param df: DataFrame to fit.
    :param grid: Grid size to break space into on each iteration.
    :param weights: Error weights.
        [Default: None]
    :param error_fn: Error function to use.
        Should accept arguments:
            + df - DataFrame to fit.
            + inflection_pts - Inflection points.
            + full_width - Widht of the original data.
            + grid - Grid size to break space into.
            + weights - Error weights.
            + side - Fit high or low energy side.
        Should return a dictionary keyed by ( center, width ) and 
        values of the error.

        If None uses #default_linear_fit_error with default weights.
        [See #default_linear_fit_error for an example]
        [Default: None]
    :param tolerance: Maximum relative change in error to terminate fitting.
        [Default: 1e-2]
    :param maxiter: Maximum number of iterations.
        [Default: 100]
    :param side: Fit high or low energy side.
        Valid values: [ 'high', 'low' ]
        [Default: 'high']
    :param history: Track history of fits.
    :returns: Tuple of ( pdf, cdf [, hdf] ) where
        + `pdf` - pandas.Series final position and width of the best fit.
            These values should be used with to extract the data from the original DataFrame,
            and fit with scipy.stats#linregress or another linear fitting method.
        + `cdf` - pandas.Series of convergence information.
            If the value is a number, it is the iteration the fit converged on.
            If it is False, the data did not converge.
        + hdf - Only returned if `history` if True.
            List of return values from the the error function for each iteration.
    """
    if error_fn is None:
        error_fn = default_linear_fit_error

        if weights is None:
            weights = np.ones( 4 )


    # find inflection points
    cdf = df.apply( std.df_grad ).apply( std.df_grad )
    cdf = cdf.dropna( how = 'all' )
    
    idf = []
    for name, tdf in cdf.items():
        tdf = tdf.dropna().apply( lambda x: x < 0 ).diff().dropna().astype( np.int0 ).abs()
        sdf = pd.Series( tdf[ tdf == 1 ].index.values, name = name )
        sdf = sdf.sort_values( ascending = ( side == 'high' )  )
        idf.append( sdf )
    
    idf = pd.concat( idf, axis = 1 )
    hdf = {}
    converged = {}
    rdf = {}
    for name, tdf in df.items():
        converged[ name ] = False
        data_peak = tdf.idxmax()

        # mask data by side
        mask = (
            tdf.index < data_peak
            if side == 'low' else
            tdf.index > data_peak
        )

        if not np.any( mask ):
            # no valid data
            rdf[ name ] = None
            logger.info( f'No data for { name } due to side mask.' )
            continue

        tdf = tdf[ mask ]
        tdf = tdf.dropna()
        
        # get relevant inflection points
        inflection_pts = idf.loc[ :, name ].dropna()
        mask = (
            inflection_pts < data_peak
            if side == 'low' else
            inflection_pts > data_peak
        )

        inflection_pts = inflection_pts[ mask ]

        # init fitting
        thdf = []
        wtdf = tdf.index
        full_width = wtdf.max() - wtdf.min()
        
        itdf = tdf
        iteration = 0
        err_change = 1
        while iteration < maxiter:
            # calculate errors
            errors = error_fn(
                itdf,
                inflection_pts = inflection_pts,
                full_width = full_width,
                grid = grid,
                weights = weights,
                side = side
            )
            
            if errors is None:
                # could not fit
                break
            
            edf = pd.Series( errors ) 
            if history:
                thdf.append( errors )
            
            # caluclate relative error change
            min_err = edf.min()
            if iteration == 0:
                # first run, no comparison
                err = min_err
                err_change = tolerance
                
            else:
                err_change = abs( ( min_err - err )/ err )
                if min_err != 0:
                    err = min_err
            
            if err_change < tolerance:
                # fit converged better than tolerance
                # break from fitting
                converged[ name ] = iteration + 1
                break
            
            # calculate new domain
            centers = list( edf.index.get_level_values( 0 ).unique().sort_values() )
            widths = list( edf.index.get_level_values( 1 ).unique().sort_values() )
            min_c, min_w = edf.idxmin()
            min_c_index = centers.index( min_c )
            min_w_index = widths.index( min_w )
            
            # move up one width level if possible
            w_index = min( min_w_index + 1, len( widths ) - 1 )
            width_bound = widths[ w_index ]/ 2
            
            # get min and max x positions
            c_min_index = max( min_c_index - 1, 0 )
            c_max_index = min( min_c_index + 1, len( centers ) - 1 )
            x_min = centers[ c_min_index ] - width_bound
            x_max = centers[ c_max_index ] + width_bound
            
            itdf = tdf[
                ( tdf.index >= x_min ) &
                ( tdf.index <= x_max)
            ]
            
            if itdf.shape[ 0 ] < 2:
                break
            
            iteration += 1
            
        if history:
            hdf[ name ] = thdf
            
        rdf[ name ] = ( min_c, min_w )
        
    rdf = pd.Series( rdf )
    converged = pd.Series( converged )
    ret = [ rdf, converged ]
    
    if history:
        ret.append( hdf )
        
    return tuple( ret )


def center_width_to_fit_df(
    df,
    cw_df
):
    """
    Takes a DataFrame and Series of ( center, width ) values
    calculating the fit data and fit of that data for each data series.
    
    For use with #linear_fit_search results.
    e.g.
    cw, _ = blackbody_analysis.linear_fit_search( df )
    fit_results = blackbody_analysis.center_width_to_df( df, cw )

    :param df: Original DataFrame that was fit.
    :param cw_df:  pandas.Series of ( center, width ) values indexed by data name.
        [Value of index 0 of result returned from #best_linear_fit]
    :returns: Dictionary keyed by data series name with values ( fit_df, fit ),
        where fit_df is the fit data, and fit is the linear fit from scypi.stats#linregress.
    """
    fdf = {}
    for name, tdf in df.items():
        x0, w0 = cw_df.loc[ name  ]
        min_x = x0 - w0/2
        max_x = x0 + w0/2

        tdf = tdf.dropna()
        fit_df = tdf[ ( min_x <= tdf.index ) & ( tdf.index <= max_x ) ]
        fit = linregress( fit_df.index, fit_df.values )

        fdf[ name ] = ( fit_df, fit )

    return fdf


def temperature_fit_using_search(
    df,
    smooth_kwargs = None,
    search_kwargs = None
):
    """
    Extract the temperature using a linear fit search.
    Uses #linear_fit_search.

    :param df: DataFrame to fit.
    :param smooth_kwargs: Dictionary of keyword arguments passed to scipy.signal#savgol_filter, or False.
        + If None uses default value of 
            { window_length: 0.01* df.shape[ 0 ], polyorder = 2 }
        + If False, does not smooth.
        [Default: None]
    :param search_kwargs: Dictionary of keyword arguments passed to #linear_fit_search.
        [Default: None]
    :returns: Tuple of ( temperatures, fits, results )
        where temperatures is a Pandas.Series of the extracted temperatures,
        fits is the return value from #center_width_to_fit_df, and
        results is the return value of #linear_fit_search.
    """
    df = df.clip( lower = 1 )
    ldf = df.apply( np.log )

    # smooth data
    if smooth_kwargs is not False:
        if smooth_kwargs is None:
            n_pts = ldf.shape[ 0 ]
            window = int( 0.01* n_pts )
            if not ( window % 2 ):
                # ensure odd window
                window += 1

            smooth_kwargs = {
                'window_length': window,
                'polyorder': 2
            }
        
        smooth = lambda x: savgol( x, **smooth_kwargs )
        
        ldf = std.resample( ldf, method = 'samples', value = n_pts )
        ldf = ldf.apply( smooth )

    # fit data
    res = linear_fit_search( ldf, **search_kwargs )
    fits = center_width_to_fit_df( ldf, res[ 0 ] )

    # extract temperatures
    slopes = { name: fit.slope for name, ( _, fit ) in fits.items() }

    side = (
        search_kwargs[ 'side' ]
        if 'side' in search_kwargs else
        'high'
    )

    betas = {
        name: slope_to_beta( slope, side = side )
        for name, slope in slopes.items()
    }

    temps = {
        name: convert_beta_temperature( beta )
        for name, beta in betas.items()
    }

    temps = pd.Series( temps )

    return ( temps, fits, res )