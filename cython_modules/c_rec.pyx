import cython

import numpy as np

cimport numpy as cnp

cimport cython

cnp.import_array()

DTYPEfloat = np.float64

DTYPEint = np.int8

ctypedef cnp.float64_t DTYPEfloat_t

ctypedef cnp.int8_t DTYPEint_t

# Euclidean distance
@cython.boundscheck(False)
@cython.wraparound(False)
def dist(cnp.ndarray[DTYPEfloat_t, ndim = 1] x, cnp.ndarray[DTYPEfloat_t, ndim = 1] y, bint m_norm, int m):

    cdef DTYPEfloat_t d

    cdef int N = x.shape[0]
    cdef int i

    for i in range(0,N):

        d += (x[i] - y[i])**2

    d = np.sqrt(d)

    if m_norm == True and m != None:

        d = d/m

    return d

# Recurrence Plot for a single embeddend time series
@cython.boundscheck(False)
@cython.wraparound(False)
def rec_plt(cnp.ndarray[DTYPEfloat_t, ndim = 2] emb_ts, float r, int T, bint m_norm, int m):

    cdef int N = emb_ts.shape[1]

    cdef cnp.ndarray[DTYPEint_t, ndim = 2] rplt = np.full((T,T), -1, dtype = DTYPEint)

    cdef int i, j

    cdef DTYPEfloat_t dij

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i + 1):

                dij = dist(x = emb_ts[:,i], y = emb_ts[:,j], m_norm = m_norm, m = m)

                # Get value of theta
                if dij < r:

                    rplt[i,j] = 1
                    rplt[j,i] = 1

                else:

                    rplt[i,j] = 0
                    rplt[j,i] = 0

    return rplt