# ARS Cryostat Data Prep

import pandas as pd

from .. import standard_functions as std


def calibrate_temperatures( df, substrate, axis = 1, level = 'temperature' ):
	"""
	Calibrate temperatures based on substrate.
	Calibration factors found using high quality FAPbI3 on different substrates
	referencing room temperature and phase tranisition aas seen in PL measurements.

	:param df: pandas DataFrame or Series to calibrate.
	:param substrate: Substrate used.
		Valid substrate values:
		+ 'glass': Microscope glass. [scale: 5/7, lower: 90, upper: 300]
		+ 'fto': FTO glass. [scale: 3/5, lower: 50, upper: 300]
	:param axis: Axis that contains the temperatures to calibrate. [Default: 1]
	:param level: Index level to calibrate. [Default: 'temperature']
	:returns: DataFrame or Series with the specified index values calibrated according to the scaling.
	"""
	df = df.copy()

	true_ref = 150  # actual reference, using turning point of PL energy peak of FAPbI3
	sub_params = {
		'glass': {
			'scale': 5/ 7,
			'ref': 90
		},

		'fto': {
			'scale': 3/ 5,
			'ref': 50
		}
	}

	if substrate not in sub_params:
		raise KeyError( 'Invalid substrate key.' )

	# create conversion function
	params = sub_params[ substrate ]
	convert = lambda T: ( T - params[ 'ref' ] )* params[ 'scale' ] + true_ref

	# get index to convert
	index = df.axes[ axis ]

	# convert and recreate index
	if isinstance( index, pd.MultiIndex ): 
		# multi index
		level_ind = std.get_level_index( df, level, axis = axis )
		index_vals = index.get_level_values( level )

		c_vals = convert( index_vals )
		c_index = pd.MultiIndex.from_tuples( [
			( *head[ : level_ind ], c_val, *head[ level_ind + 1 : ] )
			for head, c_val in zip( index, c_vals )
		], names = index.names )

	else:
		# basic index
		c_index = convert( index )
	
	if ( axis == 0 ) or ( axis == 'index' ):
		df.index = c_index

	elif ( axis == 1 ) or ( axis == 'columns' ):
		df.columns = c_index

	else:
		raise ValueError( f'Unknown axis value. Should be 0, `index`, 1, or `columns`, bur received {axis}.' )

	return df
