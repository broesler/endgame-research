#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: a_Model.py
#  Created: 06/19/2018, 09:34
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

def ModelIt(fromUser  = 'Default', births = []):
    in_month = len(births)
    print('The number born is %i' % in_month)
    result = in_month
    if fromUser != 'Default':
        return result
    else:
        return 'check your input'

#==============================================================================
#==============================================================================
