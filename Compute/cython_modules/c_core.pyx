import cython

import numpy as np

cimport numpy as cnp

cimport cython

from libc.math cimport sqrt

cnp.import_array()

DTYPEfloat = np.float64

DTYPEint = np.int8

ctypedef cnp.float64_t DTYPEfloat_t

ctypedef cnp.int8_t DTYPEint_t

# Euclidean distance
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPEfloat_t dist(cnp.ndarray[DTYPEfloat_t, ndim = 1] x, cnp.ndarray[DTYPEfloat_t, ndim = 1] y, bint m_norm, int m):

    cdef:

        Py_ssize_t N = x.shape[0]
        Py_ssize_t i

        DTYPEfloat_t d

    for i in range(0,N):

        d += (x[i] - y[i])*(x[i] - y [i])

    d = sqrt(d)

    if m_norm == True and m != None:

        d /= m

    return d

# Distance Matrix for a single embedded time series
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPEfloat_t, ndim = 2] distance_matrix(cnp.ndarray[DTYPEfloat_t, ndim = 2] emb_ts, bint m_norm, int m):

    cdef:

        int T = emb_ts.shape[1]

        cnp.ndarray[DTYPEfloat_t, ndim = 2] dist_matrix = np.full((T,T), 0, dtype = DTYPEfloat)

        Py_ssize_t N = emb_ts.shape[1]
        Py_ssize_t i, j

        DTYPEfloat_t dij

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i + 1):

            dij = dist(x = emb_ts[:,i], y = emb_ts[:,j], m_norm = m_norm, m = m)

            dist_matrix[i][j] = dij
            dist_matrix[j][i] = dij

    return dist_matrix

# Recurrence Plot for a single embedded time series
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPEint_t, ndim = 2] rec_plt(cnp.ndarray[DTYPEfloat_t, ndim = 2] dist_matrix, DTYPEfloat_t r, int T):

    cdef:

        cnp.ndarray[DTYPEint_t, ndim = 2] rplt = np.full((T,T), 0, dtype = DTYPEint)

        Py_ssize_t N = dist_matrix.shape[0]
        Py_ssize_t i, j

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i + 1):

            # Get value of theta
            if dist_matrix[i][j] < r:

                rplt[i][j] = 1
                rplt[j][i] = 1

    return rplt

# Spacetime Separation Plot for a single embedded time series
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPEfloat_t, ndim = 2] sep_plt(cnp.ndarray[DTYPEfloat_t, ndim = 2] dist_matrix, list percentiles, int T):

    cdef:

        int m = len(percentiles)

        cnp.ndarray[DTYPEfloat_t, ndim = 1] perc = np.full((m), 0, dtype = DTYPEfloat)
        cnp.ndarray[DTYPEfloat_t, ndim = 2] splt = np.full((m,T), 0, dtype = DTYPEfloat)

        Py_ssize_t N = dist_matrix.shape[0]
        Py_ssize_t n = percentiles.shape[0]
        Py_ssize_t i, j

        list dist

    # Compose distribution of distances for each relative time distance
    for i in range(0,N):

        dist = []
        for j in range(0, N - i):

            dist.append(dist_matrix[j][i + j])

        if (N - i) > 2*m:

            perc = np.percentile(dist, percentiles)

            for j in range(0,n):

                splt[j][i] = perc[j]

    return splt


# Recurrence Plot for a single embedded time series
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPEfloat_t corr_sum(cnp.ndarray[DTYPEfloat_t, ndim = 2] dist_matrix, DTYPEfloat_t r, int w):

    cdef:

        DTYPEfloat_t csum

        int c = 0

        Py_ssize_t N = dist_matrix.shape[0]
        Py_ssize_t i, j

    if w == 0:

        # Cycle through all different couples of points
        for i in range(0,N):
            for j in range(0, i):

                # Get value of theta
                if dist_matrix[i][j] < r:

                    c += 1

        csum = (2/(N*(N-1)))*c

    else:

        # Cycle through all different couples of points
        for i in range(0,N):
            for j in range(0, i):

                # Get value of theta
                if dist_matrix[i][j] < r and (i - j) > w:

                    c += 1

        csum = (2/((N-w)*(N-w-1)))*c

    return csum
