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
cpdef DTYPEfloat_t dist(cnp.ndarray[DTYPEfloat_t, ndim = 1] x, cnp.ndarray[DTYPEfloat_t, ndim = 1] y, bint m_norm, int m):

    cdef DTYPEfloat_t d

    cdef int N = x.shape[0]
    cdef int i

    for i in range(0,N):

        d += (x[i] - y[i])**2

    d = np.sqrt(d)

    if m_norm == True and m != None:

        d = d/m

    return d

# Distance Matrix for a single embeddend time series
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPEfloat_t, ndim = 2] distance_matrix(cnp.ndarray[DTYPEfloat_t, ndim = 2] emb_ts, bint m_norm, int m):

    cdef int N = emb_ts.shape[1]

    cdef cnp.ndarray[DTYPEfloat_t, ndim = 2] dist_matrix = np.full((N,N), 0, dtype = DTYPEfloat)

    cdef int i, j

    cdef DTYPEfloat_t dij

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i + 1):

            dist_matrix[i][j] = dist(x = emb_ts[:,i], y = emb_ts[:,j], m_norm = m_norm, m = m)

    return dist_matrix

# Recurrence Plot for a single embeddend time series
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPEint_t, ndim = 2] rec_plt(cnp.ndarray[DTYPEfloat_t, ndim = 2] dist_matrix, DTYPEfloat_t r, int T):

    cdef int N = dist_matrix.shape[0]

    cdef cnp.ndarray[DTYPEint_t, ndim = 2] rplt = np.full((T,T), 0, dtype = DTYPEint)

    cdef int i, j

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i + 1):

            # Get value of theta
            if dist_matrix[i][j] < r:

                rplt[i][j] = 1
                rplt[j][i] = 1

    return rplt

# Recurrence Plot for a single embeddend time series
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPEfloat_t corr_sum(cnp.ndarray[DTYPEfloat_t, ndim = 2] dist_matrix, DTYPEfloat_t r, w = None):

    cdef int N = dist_matrix.shape[0]

    cdef DTYPEfloat_t csum

    cdef int i, j, n

    cdef int c = 0

    if w == None:

        # Cycle through all different couples of points
        for i in range(0,N):
            for j in range(0, i):

                # Get value of theta
                if dist_matrix[i][j] < r:

                    c += 1

        csum = (2/(N*(N-1)))*c

    else:

        n = 0
        # Cycle through all different couples of points
        for i in range(0,N):
            for j in range(0, i):

                # Get value of theta
                if dist_matrix[i][j] < r and (i - j) > w:

                    c += 1

                elif (i - j) <= w:

                    n += 1

        csum = (2/((N-n)*(N-n-1)))*c

    return csum