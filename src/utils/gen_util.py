'''
File: gen_util.py
Description: Basic utility functions.
Author: Katherine Korobov
Created: 20 June 2026
Last Modified: 20 June 2026
'''

import numpy as np

def readCSV(filename):
    '''
    @brief Reads medium-sized CSV file
    @param File to read data from
    @return Nunmpy array labeled data

    Notes: names = True -> first row of file is column names, dtype = None -> infers data types automatically
    '''
    data = np.genfromtxt(filename, delimiter = ',', names = True, dtype = None) 
    return data

def toFloat(val):
        '''
        @brief Converts a quoted string (or any value) to a float
        @args val: A value expected to represent a float, wrapped in quotes
        @return float: The floating-pointing number after removing quotes and converting

        Note: Could error if the cleaned value cannot be converted to a flaot
        '''
        return float(str(val).strip().replace('"', '')) # Get rid of quotes and convert to float