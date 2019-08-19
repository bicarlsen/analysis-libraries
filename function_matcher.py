#!/usr/bin/env python
# coding: utf-8

# # Function Matcher

# In[1]:


import math

import numpy as np
import pandas as pd


# In[2]:


def initCoeffs( data ):
    coeffs = pd.DataFrame( columns = [ 'coeff' ], index = data.columns )
    return coeffs.fillna( 1 )


def fcnDotProduct( y1, y2, x ):
    return np.trapz( y2* y1, x )


def pairwiseDotProducts( df ):
    cols = df.columns.values
    x = df.index.values
    
    vals = [ [ fcnDotProduct( df[ k1 ].values, df[ k2 ].values, x ) for k2 in cols ] for k1 in cols ]
    return pd.DataFrame(
        data = vals,
        index = cols,
        columns = cols
    )

def getCoefficient( key, dps, coeffs, gkey = 'goal' ):
    okeys = list( dps.index.values )
    okeys.remove( key )
    okeys.remove( gkey )
    
    others = 0
    for okey in okeys:
        others += coeffs.loc[ okey, 'coeff' ]* dps.loc[ key, okey ]
    
    val = ( dps.loc[ key, gkey ] - others )/ dps.loc[ key, key ]
        
    return max( val, 0 )


def findCoefficients( data, error, iterations = 100, update = 1, gkey = 'goal' ):
    # initialize
    dps = pairwiseDotProducts( data )
    coeffs = initCoeffs( data )
    
    # update coeffecients
    e = math.inf
    its = 0
    while ( e > error ) & ( its < iterations ):
        nc = coeffs.copy()
        keys = list( data.columns.values )
        keys.remove( gkey )
        for key in keys:
            nc.loc[ key, 'coeff' ] = getCoefficient( key, dps, coeffs, gkey )

        # moderate coefficient change
        nc[ 'old' ] = coeffs.coeff
        nc[ 'dif' ] = nc.coeff - nc.old
        nc[ 'coeff' ] = nc.old + update* nc.dif
        
        coeffs = nc[[ 'coeff' ]].copy()
        
        # update error
        total = computeTotalFcn( data.drop( [ gkey ], axis = 1 ), coeffs )
        total[ gkey ] = -data[ gkey ]
        e = np.trapz( ( total.sum( axis = 1) )**2, total.index.values )
        its += 1
        
    return ( coeffs, e )


def computeTotalFcn( data, coeffs ):
    # intialize dataframe
    total = data.copy()
    
    for key in data.columns.values:
        total.loc[ :, key ] *= coeffs.loc[ key, 'coeff' ]
        
    total[ 'total' ] = total.sum( axis = 1 )
  
    return total[[ 'total' ]].copy()


# In[ ]:




