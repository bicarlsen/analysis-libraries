#!/usr/bin/env python
# coding: utf-8

# EC Lab Analysis


def plot_measurement( ax, data, name ):
    # remove extra levels
    levels = len( data.columns.levels )
    if levels > 1:
        data.columns = data.columns.droplevel( list( range( levels - 1 ) ) )

    ax.scatter( x = data.index.values, y = data.voltage )
    ax.set_title( name, fontsize = 10 )
    ax.set_xlabel( '' )


def plot_measurement_density( ax, data, name ):
    """
    Plot measurement density.
    """
    if 'metric' in data.columns.names:
        data = data.xs( 'voltage', level = 'metrics', axis = 1 )

    data.plot.hist( ax = ax, legend = False, logy = True )
    ax.set_ylim( bottom = 1, top = 1e6 )
    ax.set_title( name, fontsize = 10 )
    ax.set_xlabel( '' )





