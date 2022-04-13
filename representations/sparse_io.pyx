from libc.stdio cimport FILE, fopen, fwrite, fclose, feof, fread
import os
from scipy.sparse import coo_matrix

#os.environ["CFLAGS"] = os.getenv("CFLAGS", "") + "-I"'/home/vivi/.local/lib/python3.8/site-packages/numpy/core/include'
#os.environ["CFLAGS"] = os.getenv("CPPFLAGS", "") + "-I" + np.get_include()

os.environ["CFLAGS"] = os.getenv("CFLAGS", "") + "-I"'/home/vivi/anaconda3/envs/histwords/lib/python3.8/site-packages/numpy/core/include'


import numpy as np
cimport numpy as np

def export_mat_from_dict(counts, filename):
    cdef FILE* fout
    cdef int word1
    cdef int word2
    cdef double val
    cdef char* fn
    fn = filename
    fout = fopen(fn, 'w')
    for (i, c), v in counts.items():
        word1 = i
        word2  = c
        val = v
        fwrite(&word1, sizeof(int), 1, fout) 
        fwrite(&word2, sizeof(int), 1, fout) 
        fwrite(&val, sizeof(double), 1, fout) 
    fclose(fout)

def export_mat_eff(row_d, col_d, data_d, out_file):
    cdef FILE* fout
    cdef int word1
    cdef int word2
    cdef double val
    cdef char* fn
    cdef int i
    filename = out_file
    fn = filename
    fout = fopen(fn, 'w')
    for i in xrange(len(row_d)):
        word1 = row_d[i]
        word2  = col_d[i]
        val = data_d[i]
        fwrite(&word1, sizeof(int), 1, fout) 
        fwrite(&word2, sizeof(int), 1, fout) 
        fwrite(&val, sizeof(double), 1, fout) 
    fclose(fout)


def retrieve_mat_as_coo(matfn, min_size=None):
    """
    matfn = file name of matrix
    min_size = pad with zeros to this size
    """
    cdef FILE* fin
    cdef int word1, word2, ret
    cdef double val
    cdef char* fn
    fn = matfn
    fin = fopen(fn, 'r')
    cdef int size = (os.path.getsize(matfn) / 16)
    if min_size != None:
        size += 1
    cdef np.ndarray[np.int32_t, ndim=1] row = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] col = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] data = np.empty(size, dtype=np.float64)
    cdef int i = 0
    while not feof(fin):
        fread(&word1, sizeof(int), 1, fin) 
        fread(&word2, sizeof(int), 1, fin) 
        ret = fread(&val, sizeof(double), 1, fin) 
        if ret != 1:
            break
        row[i] = word1
        col[i] = word2
        data[i] = val
        i += 1
    fclose(fin)
    if min_size != None:
        row[-1] = min_size
        col[-1] = min_size
        data[-1] = 0
    return coo_matrix((data, (row, col)), dtype=np.float64)


def retrieve_mat_as_coo_thresh(matfn, thresh):
    """
    matfn = file name of matrix
    thresh = min cooccurrence value
    """
    cdef FILE* fin
    cdef int word1, word2, ret
    cdef double val
    cdef char* fn
    fn = matfn
    fin = fopen(fn, 'r')
    cdef int size = (os.path.getsize(matfn) / 16)
    size = 2*size + 1
    print("mat_as_coo -- size = {}".format(size))
    cdef np.ndarray[np.int32_t, ndim=1] row = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] col = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] data = np.empty(size, dtype=np.float64)
    cdef int i = 0
    while not feof(fin):
        fread(&word1, sizeof(int), 1, fin) 
        fread(&word2, sizeof(int), 1, fin) 
        ret = fread(&val, sizeof(double), 1, fin) 
        if ret != 1:
            break
        if val < thresh:
            val = 0
        row[i] = word1
        col[i] = word2
        data[i] = val
        i += 1
        ## the matrix in the bin file may not be symmetrical (because one of the words is low frequency maybe), and the second argument is actually the context which may be longer than a word
        ## this leads to problem for the netstats computation, which assumes a symmetrical matrix, where all words that may appear on rows also appear on columns (to extract collocation information)
        ## so I will deliberately make this symmetrical
        row[i] = word2
        col[i] = word1
        data[i] = val        
        i += 1
    fclose(fin)
    return coo_matrix((data, (row, col)), dtype=np.float64)
