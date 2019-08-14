# Preston Tai

#from __future__ import division
import os
#import sys
#import glob
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import time
import random
from numba import jit, njit, typeof, int32, int64, float32, float64, prange, vectorize, guvectorize
import io
from contextlib import redirect_stdout
from functools import wraps
import llvmlite.binding as llvm
llvm.set_option('', '--debug-only=loop-vectorize')


#https://reviews.llvm.org/D36220
#os.environ['NUMBA_DEBUG_ARRAY_OPT_STATS'] = str(0)  #1
#os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = str(0)   #2
#os.environ['NUMBA_DEBUG'] = str(0)

'''
def make_v(old_func):
    #guvectorize( "(n,m)->a,b)",nopython = True)
    #@jit(int64(int64))
    #@vectorize( ['int64(int64)'] )
    #@vectorize( ret_type = int32, arg_types = int32[:])
    def new_func( *args ):
        return old_func( *args )
    return new_func
'''
def init_diagnostics():
    llvm.set_option('', '--debug-only=loop-vectorize')
    return

def vector_wrapper(old_func):
    @wraps(old_func)
    def new_func(*args, **kwargs):
        #vector timer
        print( "\n", old_func.__name__)
        start = time.time()
        old_func(*args, **kwargs)
        end = time.time()
        no_diff = end - start
        print("\t", " without optimizations took", "%.3f" % no_diff, "seconds to run ")

        start = time.time()
        #temp_f = make_v(old_func)
        temp_f = jit(old_func)
        temp_f(*args, **kwargs)
        #temp_f = np.vectorize(old_func)
        #temp_f(*args)
        end = time.time()
        v_diff = end - start
        print("\t", " with optimizations took", "%.3f" % v_diff, "seconds to run ")

        print(" It is", "%.3f" % (no_diff/v_diff), "times faster with optimizations" )

        return old_func(*args, **kwargs)
    return new_func


def vector_print(old_func):
    @wraps(old_func)
    def new_func(*args, **kwargs):
        #old_func(*args, **kwargs)
        
        oldstdchannel = os.dup(2)
        f = open('err.txt', 'w')
        try:
            os.dup2(f.fileno(), 2)
            old_func(*args, **kwargs)
        except:
            os.dup2(oldstdchannel, 2)
            old_func(*args, **kwargs)
        finally:
            os.dup2(oldstdchannel, 2)
        file = open('err.txt', 'r')
        for line in file:
            if ( len(line) < 100 and ('pass' not in line ) and 
                ('legality' in line or 'Found a loop' in line or 'We can vectorize' in line )): #vectorizable in line
                print( line[:-1] )
        print()
        file.close() 
        
        return 
    return new_func

