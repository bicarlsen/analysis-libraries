#!/usr/bin/env python
# coding: utf-8

# Plotting functions

# Imports
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# @todo: use dataclass
"""
Base font
"""
font_base = {
    'figure': { 'figsize': ( 10, 8 ) }
}

"""
Default font.
Use for data exploration.
"""
font_def = {
    **font_base,
    'font': { 'size': 16 },
    'xtick': { 'labelsize': 14 },
    'ytick': { 'labelsize': 14 },
}


"""
Publication font.
Use for publication figures.
Enlarged font for better readability on paper.
"""
font_pub = {
    **font_base,
    'font': { 'size': 24 },
    'xtick': { 'labelsize': 18 },
    'ytick': { 'labelsize': 18 },
    'axes': { 'labelpad': 15, 'titlepad': 15 },
}


def set_plot_defaults( font = font_def ):
    """
    Set matplotlib plotting defaults.

    :param font: Dictionary representing the font to set.
        Keys should be the matplotlib.pyplot.rcParams.group.
        Values are a dictionary of the arguments to set for the group,
        keyed by argument name with values the value to set.
        [Default: bric_analysis_libraries.plot.font_def]
    """
    for group, args in font.items():
        mpl.rc( group, **args )


def get_scaled_colormap( name, min_val = 0, max_val = 1, raise_err = False ):
    """
    :param name: Name of the color map.
    :param min_val: Minimum value of range. [Default: 0]
    :param max_val: Maximum value of range. [Default: 1]
    :param raise_err: Raise an error if value is outside bounds.
        If False, clips to range.
        [Default: False]
    :returns: Function that accepts an argument of the value,
        and returns the corresponding color value.
    """
    if min_val >= max_val:
        raise ValueError( '`min_val` is less or equal to `max_val`.' )

    cmap = cm.get_cmap( name )
    val_range = max_val - min_val
    def _scaled_map( val ):
        """
        Converts a value in a range to a color value.

        :param val: Value or iterable of values to convert to a color.
        :returns: ( r, g, b, a ) or list of ( r, g, b, a ) color values.
        """
        if not( isinstance( val, int ) or isinstance( val, float ) ):
            val = np.array( val )

        norm_val = ( val - min_val )/ val_range
        if raise_err:
            if ( norm_val < 0 ) or ( norm_val > 1 ):
                raise ValueError( 'Value outside bounds.' )

        np.clip( norm_val, a_min = 0, a_max = 1 )
        return cmap( norm_val )


    return _scaled_map


def fig_for_publication( fig, font = None, label_map = None ):
    """
    Set a figures fonts for publication.

    :param fig: Figure to prepare.
    :param font: Dictionary of font values.
        Only keys that should be changes from the default should be included.
        + major_label_size: Font size of the major labels. [Default: 24]
        + major_labelpad: Padding of the major labels. [Default: 15]
        + major_tick_size: Font size of the major tick labels. [Default: 18]
    :param label_map: Dictionary of labels and their mapped values. [Default: {}]
    """
    if font is None:
        font = {}

    if label_map is None:
        label_map = {}

    defaults = {
        'major_label_size': 24,
        'major_labelpad': 15,
        'major_tick_size': 18
    }

    font = { **defaults, **font }

    for ax in fig.axes:
        saxs = [ ax.xaxis, ax.yaxis ]
        for sax in saxs:
            lbl_text = sax.label.get_text()
            if lbl_text in label_map:
                sax.label.set_text( label_map[ lbl_text ] )

            sax.label.set_fontsize( font[ 'major_label_size' ] )
            sax.labelpad = font[ 'major_labelpad' ]
            sax.set_tick_params( labelsize = font[ 'major_tick_size' ] )


def save_figure( path, kind = 'png', fig = None ):
    """
    Save a figure.

    :param path: Path to save file.
    :param kind: Format to save file. [Default: 'png']
    :param fig: Figure to save. If None, saves current figure.
        [Default: None]
    """
    if fig is None:
        fig = plt.gcf()

    fig.savefig( path, format = kind, bbox_inches = 'tight' )


def index_from_counter( counter, rows, cols ):
    """
    Get the row and column of a matrix from a counter.

    :param counter: Counter.
    :param rows: Number of rows in matrix.
    :param cols: Number of columns in matrix.
    :returns: ( row, column ) of counter.
    :raises: ValueError if `counter` is invalid.
    """
    if counter < 0:
        raise ValueError( 'Counter must be a non-negative integer.' )

    if counter > rows* cols:
        raise ValueError( 'Counter exceeds total grid size.' )

    row = int( np.floor( counter / cols ) )
    col = int( counter % cols )

    return ( row, col )


def ax_from_counter( counter, axs, reverse = False ):
    """
    Gets an axis from an array of axes based on a counter.

    :param counter: Counter.
    :param axs: Matrix of axes.
    :param reverse: Reverse order of axes. [Defualt: False]
    :returns: Axis.
    """
    row, col = index_from_counter( counter, *axs.shape )
    if reverse:
        row = axs.shape[ 0 ] - 1 - row

    if len( axs.shape ) == 1:
        # only rows
        ax = ax[ row ]

    else:
        # rows and cols
        ax = axs[ row, col ]

    return ax


def rows_needed( df, cols = 1, level = None, axis = 1 ):
    """
    Returns the number of rows needed for a plot, givent the number of columns.

    :param df: DataFrame, DataFrameGroupBy, Series, or SeriesGroupBy.
    :param cols: Number of columns. [Default: 1]
    :param level: DataFrame level to use for grouping,
        or None to count each column.
        If `df` is a DataFrameGroupBy or SeriesGroupBy this does not have an effect.
        [Default: None]
    :param axis: Axis to determine rows from.
        Not applicable if `df` is not DataFrame.
        [Default: 1]
    :returns: Number of rows needed for iterating over the data given the number of columns.
    """
    if (
        isinstance( df, pd.core.groupby.DataFrameGroupBy ) or
        isinstance( df, pd.core.groupby.SeriesGroupBy )
    ):
        num_plots = len( df )

    elif (
        isinstance( df, pd.DataFrame ) or
        isinstance( df, pd.Series )
    ):
        if isinstance( df, pd.Series ):
            axis = 0

        if level is not None:
            num_plots = df.axes[ axis ].get_level_values( level ).unique().shape[ 0 ]

        else:
            num_plots = df.axes[ axis ].shape[ 0 ]

    else:
        # invalid type
        raise TypeError( 'Invalid `df` object. Must be a pandas DataFrame, DataFrameGroupBy, Series, or SeriesGroupBy.' )

    return int( np.ceil( num_plots/ cols ) )


def rows_cols_square( df, level = None, axis = 1 ):
    """
    Returns the number of rows and columns needed to
    create a square set of axes for the data.

    :param df: DataFrame, DataFrameGroupBy, Series, or SeriesGroupBy.
    :param level: DataFrame level to use for grouping,
        or None to count each column.
        If `df` is a DataFrameGroupBy or SeriesGroupBy this does not have an effect.
        [Default: None]
    :param axis: Axis to determine rows from.
        Not applicable if `df` is not DataFrame.
        [Default: 1]
    :returns: Tuple of ( rows, cols ) needed for iterating over the data.
    """
    if (
        isinstance( df, pd.core.groupby.DataFrameGroupBy ) or
        isinstance( df, pd.core.groupby.SeriesGroupBy )
    ):
        num_plots = len( df )

    elif (
        isinstance( df, pd.DataFrame ) or
        isinstance( df, pd.Series )
    ):
        if isinstance( df, pd.Series ):
            axis = 0

        if level is not None:
            num_plots = df.axes[ axis ].get_level_values( level ).unique().shape[ 0 ]

        else:
            num_plots = df.axes[ axis ].shape[ 0 ]

    else:
        # invalid type
        raise TypeError( 'Invalid `df` object. Must be a pandas DataFrame, DataFrameGroupBy, Series, or SeriesGroupBy.' )

    sqrt = num_plots**( 1/2 )
    rows = int( np.ceil( sqrt ) )
    cols = int( np.round( sqrt ) )

    return ( rows, cols )


def plot_levels(
    plot,
    df,
    show = True,
    level = 'metrics',
    axis = 1,
    flip_axes = False,
    **fig_args
    ):
    """
    Plots each element of a Pandas DataFrame in a separate subplot.

    :param plot: A function that receives a Pandas DataSeries and axis to plot it on ( ax, data, name ).
    :param df: The DataFrame to plot.
    :param show: Show the plot. [Defualt: True]
    :param level: Which level to iterate over. [Default: 'metrics']
    :param axis: The axis to iterate over. [Default: 'columns']
    :param flip_axes: Flip the order of the axes, plotting from the bottom first.
        [Default: False]
    :param fig_args: Keyword arguments passed to plt.subplot().
    :returns: The Figure and Axes of the plot as a tuple ( fig, axs ).
    """
    if axis == 'rows':
        axis = 0

    elif axis == 'columns':
        axis = 1

    ax = df.axes[ axis ]

    levels = list( range( ax.names.index( level ) + 1 ) )
    groups = df.groupby( level = levels, axis = axis )

    num_plots = len( groups )
    cols = int( np.floor( np.sqrt( num_plots ) ) )
    rows = int( np.ceil( num_plots/ cols ) )
    fig, axs = plt.subplots( rows, cols, **fig_args )
    index = 0

    for name, data in groups:
        ax = ax_from_counter( index, axs, reverse = flip_axes )
        plot( ax, data, name )
        index += 1

    fig.tight_layout()

    if show:
        plt.show()

    return ( fig, axs )


def plot_df( plot, df, show = True, **fig_args ):
    """
    Plots each element of a Pandas DataFrame in a separate subplot.

    :param plot: A function that receives a Pandas DataSeries and axis to plot it on ( ax, data, name ).
    :param df: The DataFrame to plot.
    :param show: Show the plot. [Defualt: True]
    :param fig_args: Keyword arguments passed to plt.subplot().
    :returns: The Figure and Axes of the plot as a tuple ( fig, axs ).
    """
    num_plots = int( df.columns.shape[ 0 ] )
    cols = int( np.floor( np.sqrt( num_plots ) ) )
    rows = int( np.ceil( num_plots/ cols ) )
    fig, axs = plt.subplots( rows, cols, **fig_args )
    index = 0

    for name, data in df.items():
        row = int( np.floor( index/ cols ) )
        col = int( index% cols )

        if len( axs.shape ) == 1:
            # only rows
            ax = ax[ row ]

        else:
            # rows and cols
            ax = axs[ row, col ]

        plot( ax, data, name )
        index += 1

    fig.tight_layout()

    if show:
        plt.show()

    else:
        return ( fig, axs )



def boxplot_groups( df, groups, total = True, show = True ):
    """
    Creates a box plot of a grouped Pandas Series

    :param df: A Pandas Series containing the data to be plotted
    :param groups: A single or list of index levels to group by
    :param total: Whether to include a plot for all data [Default: True]
    :param show: Whether to show the plot or return the axis [Default: True]
    :returns: None if show is True, else the matplotlib Axis it is plotted on
    """
    fig, axs = plt.subplots()
    data = [ df.values ] if total else []
    labels = [ 'All' ] if total else []
    for name, group in df.groupby( groups ):
        labels.append( name )
        data.append( group.values )

    axs.boxplot( data, labels = labels )
    plt.xticks( rotation = 70 )

    if show:
        plt.show()

    else:
        return axs


def temperature_plot_rainbow(
    df,
    colorbar = True,
    level = 0,
    ax = None,
    color_by_order = False,
    cmap = 'jet',
    **kwargs
):
    """
    Plots a DataFrame by temperature.

    :param df: A DataFrame or Series with temperature as the first colum index level.
    :param colorbar: Whether to include the color bar legend. [Default: True]
    :param level: Index or name of the level used for coloring. [Default: 0]
    :param ax: Axis to plot on. If None, creates one. [Default: None]
    :param color_by_order: Color traces by order of values rather than actual value.
        [Default: False]
    :param cmap: Colormap to use. [Defualt: 'jet']
    :param **kwargs: Arguments passed to pandas.DataFrame#plot
    :returns: Tuple of ( fig, ax, None ) if colorbar is False,
        or ( fig, axs, cbar ) if colorbar is True.
    """
    is_df = isinstance( df, pd.DataFrame )
    if ax is None:
        fig, ax = plt.subplots()
    
    else:
        fig = ax.get_figure()

    temp_vals = (
        df.columns.get_level_values( level ).values
        if is_df else
        df.index.get_level_values( level ).values
    )

    val_min = temp_vals.min()
    val_max = temp_vals.max()
    
    cm = (
        plt.get_cmap( cmap )
        if isinstance( cmap, str ) else
        cmap
    )

    if color_by_order:
        # color by vlaue index
        sort_vals = np.unique( temp_vals )
        sort_vals.sort()

        indices = [
            np.argwhere( sort_vals == temp_val )[ 0 ][ 0 ]
            for temp_val in temp_vals
        ]

        num_vals = temp_vals.shape[ 0 ]
        colors = [ cm( float( ind/ num_vals ) ) for ind in indices ]

    else:
        # color by value
        colors = [
            cm( ( temp_val - val_min )/( val_max - val_min ) )  # normalize values between 0 and 1
            for temp_val in temp_vals
        ]

    df.plot( color = colors, ax = ax, **kwargs )

    if colorbar:
        cax = fig.add_axes( [ 0.92, 0.15, 0.05, 0.7 ] )
        cbar = mpl.colorbar.ColorbarBase(
            ax = cax,
            cmap = cm,
            orientation = 'vertical',
            norm = mpl.colors.Normalize(
                vmin = val_min,
                vmax = val_max
            )
        )

        if is_df:
            cbar_label = (
                df.columns.names[ level ]
                if isinstance( level, int ) else
                level
            )

        else:
            cbar_label = (
                df.index.names[ level ]
                if isinstance( level, int ) else
                level
            )

        cbar.set_label( cbar_label, labelpad = 15 )

    if colorbar:
        return ( fig, ax, cbar )

    else:
        return ( fig, ax, None )


def color_plot_2d(
    df,
    shading = 'nearest',
    colorbar = True,
    ax = None,
    **kwargs
):
    """
    Creates a 2D color plot from a DataFrame.
    The DataFrame must have a pandas.index for both
    the index and columns, i.e. non pandas.MultiIndex's.
    The columns map to the x-axis, and the index to the y-axis.

    :param df: DataFrame.
    :param shading: Shading to use.
        Values are [ 'nearest', 'gouraud' ].
        [Default: 'nearest']
    :param colorbar: Include colorbar. [Defualt: True]
    :param ax: Axis to plot on.
        If None creates a new axis.
        [Default: None]
    :param **kwargs: Args passed to the plotting function.
    :returns: ( fig, ax, cbar, img )
        If `colorbar` is False, cbar is None.
    :raises TypeError: If DataFrame's index or columns are not a pandas.Index.
    :raises ValueError: If `shading` is not valid. 
    """
    if not(
        isinstance( df.index, pd.Index ) and
        isinstance( df.columns, pd.Index )
    ):
        raise TypeError( 'Both index and columns must be pandas.Index, no MultiIndexes are allowed.' )

    if shading not in [ 'nearest', 'gouraud' ]:
        raise ValueError( 'Invalid shading value. Must be in [ "nearest", "gouraud" ].' )

    cbar = None
    x, y = np.meshgrid( df.columns, df.index )

    if ax is not None:
        fig = ax.get_figure()
    
    else:    
        fig, ax = plt.subplots()
    
    img = ax.pcolormesh( x, y, df.values, shading = shading, **kwargs )
    
    if colorbar:
        cbar = fig.colorbar( img, ax = ax )
    
    ax.set_xlabel( df.columns.name )
    ax.set_ylabel( df.index.name )

    return ( fig, ax, cbar, img )


def outer_plot(
    inner_plot,
    df,
    outer_axes,
    outer_logx = False,
    outer_logy = False,
    normalize_inner_axes = True,
    axes_scale = 1,
    ax = None,
    **kwargs
):
    """
    2D plot of Axes.

    :param inner_plot: Function to plot data on inner axes.
        Function should have signature ( ax, data, name, outer_axes ).
    :param df: DataFrame.
    :param outer_axes: ( x, y ) tuple of names of outer axes labels.
    :param outer_logx: Log scale for outer x-axis. [Default: False]
    :param outer_logy: Log scale for outer x-axis. [Default: False]
    :param normalize_inner_axes: Normalize inner axes to be the same size,
        otherwise each is made as large as possible. [Default: True]
    :param axes_scale: Scaling factor of inner axes. [Default: 1]
    :param ax: Axes to use as outer axes. [Default: None]
    :params **kwargs: Arguments passed to inner_plot function.
    :returns: If ax is not None returns a dictionary of inner axes as { ( x, y ): axes },
        otherwise returns  tuple of ( fig, axes, inner_axes ).
    """

    def rescale( val, minimum, maximum ):
        """
        Rescale value from 0 to 1.
        """
        return ( val - minimum )/( maximum - minimum )


    # get xy points for inner axes
    outer_x = df.index.get_level_values( outer_axes[ 0 ] )
    outer_y = df.index.get_level_values( outer_axes[ 1 ] )
    o_xy = np.unique( list( zip( outer_x, outer_y ) ), axis = 0 )

    if outer_logx:
        outer_x = np.log10( outer_x )

    if outer_logy:
        outer_y = np.log10( outer_y )

    outer_xy = np.unique( list( zip( outer_x, outer_y ) ), axis = 0 )

    outer_minmax = tuple(
        ( np.min( vals ), np.max( vals ) )
        for vals in zip( *outer_xy )
    )

    outer_range = tuple( val[ 1 ] - val[ 0 ] for val in outer_minmax )

    # get distances between inner center points
    # only compute upper triangle to save computing
    distances_xy = [
        [
            None
            if i1 <= i0 else
            (
                abs( p1[ 0 ] - p0[ 0 ] )/ outer_range[ 0 ],
                abs( p1[ 1 ] - p0[ 1 ] )/ outer_range[ 1 ]
            )
            for i1, p1 in enumerate( outer_xy )
        ]
        for i0, p0 in enumerate( outer_xy )
    ]

    distances = [
        [
            None
            if p is None else
            math.hypot( *p )
            for p in dist
        ]
        for dist in distances_xy
    ]

    if normalize_inner_axes:
        # normalize inner axes
        dmin = min( [ d for dist in distances for d in dist if d is not None ] )
        distances = [
            [
                None if d is None else dmin
                for d in dist
            ]
            for dist in distances
        ]

    # symmeterize distances
    distances = np.array( [
        [
            0 if d is None else d
            for d in dist
        ]
        for dist in distances
    ] )
    distances += distances.transpose()

    # find min distance for each axes
    distances = [
        np.delete( d, i ).min()  # ignore self distances
        for i, d in enumerate( distances )
    ]

    # find axes positions
    inner_posdim = tuple(
        (
            rescale( xy[ 0 ], *outer_minmax[ 0 ] ) - distances[ i ]/ 2,
            rescale( xy[ 1 ], *outer_minmax[ 1 ] ) - distances[ i ]/ 2,
            distances[ i ]* axes_scale,
            distances[ i ]* axes_scale
        )
        for i, xy in enumerate( outer_xy )
    )

    bounds = tuple(
        ( p[ 0 ], p[ 1 ], p[ 0 ] + p[ 2 ], p[ 1 ] + p[ 3 ] )
        for p in inner_posdim
    )
    bounds = tuple( zip( *bounds ) )
    bounds = (
        ( np.min( bounds[ 0 ] ), np.max( bounds[ 2 ] ) ),  # x
        ( np.min( bounds[ 1 ] ), np.max( bounds[ 3 ] ) ),  # y
    )

    dim_rescale = tuple( 1/( b[ 1 ] - b[ 0 ] ) for b in bounds )

    inner_posdim = tuple(
        (
            rescale( p[ 0 ], *bounds[ 0 ] ),
            rescale( p[ 1 ], *bounds[ 1 ] ),
            p[ 2 ]* dim_rescale[ 0 ],
            p[ 3 ]* dim_rescale[ 1 ]
        )
        for p in inner_posdim
    )

    centers = list( zip( *[
        ( p[ 0 ] + p[ 2 ]/ 2, p[ 1 ] + p[ 3 ]/ 2 )
        for p in inner_posdim
    ] ) )

    inner_posdim = { tuple( vals ): inner_posdim[ i ] for i, vals in enumerate( outer_xy )  }

    # outer axes
    return_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xticks( np.unique( centers[ 0 ] ) )
    ax.set_yticks( np.unique( centers[ 1 ] ) )

    x_labels, y_labels = list( zip( *outer_xy ) )
    x_labels = np.sort( np.unique( x_labels ) )
    y_labels = np.sort( np.unique( y_labels ) )
    if outer_logx:
        x_labels = [ int( x ) if x.is_integer() else x for x in x_labels ]
        x_labels = [ f'$10^{{{ x }}}$' for x in x_labels ]

    if outer_logy:
        y_labels = [ int( y ) if y.is_integer() else y for y in y_labels ]
        y_labels = [ f'$10^{{{ y }}}$' for y in y_labels ]

    ax.set_xticklabels( x_labels )
    ax.set_yticklabels( y_labels )

    ax.set_xlabel( outer_axes[ 0 ] )
    ax.set_ylabel( outer_axes[ 1 ] )

    # inner plots
    inner_axes = {}
    for name, data in df.groupby( level = outer_axes ):
        oi = [ *name ]
        if outer_logx:
            oi[ 0 ] = np.log10( oi[ 0 ] )

        if outer_logy:
            oi[ 1 ] = np.log10( oi[ 1 ] )

        ax_dims = inner_posdim[ tuple( oi ) ]
        in_ax = ax.inset_axes( ax_dims )
        inner_axes[ ax_dims[ :2 ] ] = in_ax

        inner_plot( in_ax, data, name, outer_axes, **kwargs )

    if return_ax:
        return fig, ax, inner_axes

    return inner_axes