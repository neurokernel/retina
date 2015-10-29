#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
from pycuda.compiler import SourceModule

import skcuda.cublas as cublas
import parray


""" assuming row major storage as in PitchArray """
class cublashandle(object):
    """ Create a cublas handle """
    def __init__(self):
        self.handle = None
        self.create()

    def create(self):
        if self.handle is None:
            self.handle = cublas.cublasCreate()

    def destroy(self):
        if self.handle is not None:
            cublas.cublasDestroy(self.handle)

    def __del__(self):
        self.destroy()


def dot(A, B, opa = 'n', opb = 'n',
        C = None, Cstart = None,
        scale = 1.0, Cscale = 0.0, handle = None):
    """
    Multiplication of two matrices A and B in PitchArray format
    if C is specified, use the memory in C.
    Specified C must have the same leading dimension as that of the result and
    the other dimension must be bigger or equal to that of the result.
    
    Parameters:
    -----------
    A: parray.PitchArray
    B: parray.PitchArray
    opa: str
         operation on A
         'n' or 'N': use A itself
         't' or 'T': use transpose of A
         'c' or 'C': use conjugate transpose of A
    opb: str
         operation on B
         'n' or 'N': use B itself
         't' or 'T': use transpose of B
         'c' or 'C': use conjugate transpose of B
    C: parray.PitchArray
       if specified, the result will be stored in C
    Cstart: int
            the offset start of C array
    scale: float
            scaling factor for A*B
            see Cscale
    Cscale: float
            scaling factor for C
            result will be C = C*Cscale + scale*A*B
    
    Note:
    -----
    works only for CUDA VERSION > 4.0 where handle is introduced.
    
    Will NOT work for complex case when A and B shares overlapping
    memory, but should work if A==B.
    """
    
    if A.dtype != B.dtype:
        raise TypeError("matrix multiplication must have same dtype")

    if (len(A.shape) != 2) | (len(B.shape) != 2):
        raise TypeError("A, B must both be matrices")

    if opa in ['n', 'N']:
        m,n = A.shape
    elif opa in ['t','T', 'c','C']:
        n,m = A.shape
    else:
        raise ValueError("unknown value assigned to opa")

    if opb in ['n', 'N']:
        k,l = B.shape
    elif opb in ['t','T', 'c','C']:
        l,k = B.shape
    else:
        raise ValueError("unknown value assigned to opa")

    if (k != n) | (0 in [m,n,l]):
        raise ValueError("matrix dimension mismatch, "
                         "(%d,%d) with (%d,%d)" % (m,n,k,l))

    dtype = A.dtype
    if dtype in [np.float32, np.float64]:
        if opb in ['c', 'C']:
            opb = 't'

        if opa in ['c', 'C']:
            opa = 't'
        
    scale = dtype.type(scale)
    Cscale = dtype.type(Cscale)
    
    if dtype == np.float64:
        tp = 'cublas.cublasD'
        complex_type = False
    elif dtype == np.complex128:
        tp = 'cublas.cublasZ'
        complex_type = True
    elif dtype == np.float32:
        tp = 'cublas.cublasS'
        complex_type = False
    elif dtype == np.complex64:
        tp = 'cublas.cublasC'
        complex_type = True

    if C is None:
        C = parray.empty((m,l), dtype)
        Cstart = 0
        Cempty = True
        Cscale = dtype.type(0)
    else:
        Cempty = False
        if Cstart is None:
            Cstart = 0
        if C.shape[1] != l:
            raise AttributeError("shape of the provided result array "
                                 + C.shape.__str__()
                                 + " does not match intended result " 
                                 + (m,l).__str__())
        if C.shape[0] < m + Cstart:
            raise AttributeError("shape of the provided result array "
                                 + C.shape.__str__()
                                 + " does not match intended result "
                                + (m,l).__str__())
        if C.dtype != dtype:
            raise TypeError("Result array C provided must have "
                            "the same dtype as inputs")
    
    conjA = False
    conjB = False
    conjC = False
    
    sameflag = (A==B)
    
    itemsize = C.dtype.itemsize
    handlestr = "handle.handle"
    if m == 1:
        if n == 1:
            alpha = A.get()[0,0]
            if opa in ['c','C']:
                alpha = np.conj(alpha)
            C*=Cscale
            if opb in ['c','C']:
                func = (tp+"axpy(handle.handle, l, alpha*scale, "
                        + "parray.conj(B).gpudata, 1,"
                        + "int(C.gpudata)+Cstart*itemsize, 1)")
            else:
                func = (tp+"axpy(handle.handle, l, alpha*scale, "
                        + "B.gpudata, 1, "
                        + "int(C.gpudata)+Cstart*itemsize, 1)")
        else:
            if l > 1:
                alpha = scale
                beta = Cscale
                if opa in ['c','C']:
                    A.conj()
                    conjA = True
                func = (tp+"gemv(handle.handle, '"+opb+"',B.shape[1], "
                        + "B.shape[0], alpha, B.gpudata, B.ld, A.gpudata, "
                        + "1, beta, int(C.gpudata)+Cstart*itemsize*C.ld, 1)")
            else:
                if opa in ['c','C']:
                    if opb in ['c', 'C']:
                        func = ("C.set(np.array(scale*" + tp
                                + "dotu(handle.handle, n, A.gpudata, "
                                + "1, B.gpudata, 1)"
                                +").conj()+C.get()*Cscale)")
                    else:
                        func = ("C.set(np.array(scale*" + tp
                                + "dotc(handle.handle, n, A.gpudata, "
                                + "1, B.gpudata, 1)) + C.get()*Cscale)")
                elif opb in ['c', 'C']:
                    func = ("C.set(np.array(scale*" + tp
                            + "dotc(handle.handle, n, B.gpudata, 1, "
                            + "A.gpudata, 1)) + C.get()*Cscale)")
                else:
                    if complex_type:
                        func = ("C.set(np.array(scale*" + tp
                                + "dotu(handle.handle, n, A.gpudata, 1, "
                                + "B.gpudata, 1)) + C.get()*Cscale)")
                    else:
                        func = ("C.set(np.array(scale*" + tp
                                + "dot(handle.handle, n, A.gpudata, 1, "
                                + "B.gpudata, 1)) + C.get()*Cscale)")
    else:#m!=1
        if n == 1:
            if l == 1:
                alpha = B.get()[0,0]
                if opb in ['c','C']:
                    alpha = np.conj(alpha)
                C*=Cscale
                if opa in ['c','C']:
                    func = (tp+"axpy(handle.handle, m, alpha*scale, "
                            + "parray.conj(A).gpudata, 1, "
                            + "int(C.gpudata)+Cstart*itemsize, 1)")
                else:
                    func = (tp+"axpy(handle.handle, m, alpha*scale, "
                            + "A.gpudata, 1, "
                            + "int(C.gpudata)+Cstart*itemsize, 1)")
            else:
                if Cempty:
                    C.fill(0)
                else:
                    C*=Cscale
                if opa in ['c','C']:
                    if opb in ['c', 'C']:
                        B.conj()
                        conjB = True
                        func = (tp + "gerc(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, "
                                + "C.ld)")
                    else:
                        func = (tp + "gerc(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, "
                                + "C.ld)")
                elif opb in ['c', 'C']:
                    if sameflag:
                        B.conj()
                        conjB = True
                        func = (tp + "gerc(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, C.ld)")
                    
                    else:
                        B.conj()
                        conjB = True
                        func = (tp + "geru(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, C.ld)")
                else:
                    if complex_type:
                        func = (tp + "geru(handle.handle, l, m, scale, "
                                + "B.gpudata, 1,  A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, C.ld)")
                    else:
                        func = (tp + "ger(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, C.ld)")
        else:
            if l == 1:
                if opb in ['c', 'C']:
                    if opa in ['c', 'C']:
                        conjC = True
                        if not Cempty:
                            C.conj()
                            Cscale = Cscale.conj()
                        func = (tp + "gemv(handle.handle, 'n', A.shape[1], "
                                + "A.shape[0], scale, A.gpudata, A.ld, "
                                + "B.gpudata, 1, Cscale, int(C.gpudata) + "
                                + "Cstart * itemsize * C.ld, 1)")
                    else:
                        B.conj()
                        conjB = True
                        if opa in ['t', 'T']:
                            opa = 'n'
                        else:
                            opa = 't'
                        
                        func = (tp + "gemv(handle.handle, '" + opa + "', "
                                + "A.shape[1], A.shape[0], scale, A.gpudata, "
                                + "A.ld, B.gpudata, 1, Cscale, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, 1)")
                else:
                    if opa in ['c', 'C']:
                        B.conj()
                        conjB = True
                        conjC = True
                        if not Cempty:
                            C.conj()
                            Cscale = Cscale.conj()
                        func = (tp + "gemv(handle.handle, 'n', A.shape[1], "
                                + "A.shape[0], scale, A.gpudata, A.ld, "
                                + "B.gpudata, 1, Cscale, int(C.gpudata) + "
                                + "Cstart * itemsize * C.ld, 1)")
                    else:
                        if opa in ['t', 'T']:
                            opa = 'n'
                        else:
                            opa = 't' 
                        func = (tp + "gemv(handle.handle, '" + opa + "', "
                                + "A.shape[1],  A.shape[0], scale, A.gpudata, "
                                + "A.ld, B.gpudata, 1, Cscale, int(C.gpudata) "
                                + "+ Cstart * itemsize * C.ld, 1)")
            else:
                func = (tp+"gemm(handle.handle, '" + opb + "','" + opa + "', "
                        + "l, m, k, scale, B.gpudata, B.ld, A.gpudata, A.ld, "
                        + "Cscale, int(C.gpudata) + "
                        + "Cstart * itemsize * C.ld, C.ld)")

    if handle is None:
        handle = cublashandle()
    eval(func)
    
    if conjC:
        C.conj()

    if conjA:
        A.conj()

    if conjB:
        B.conj()
    return C


def norm(A, handle = None):
    """
    computes the l2 norm of a vector A

    Parameters
    ----------
    A : parray.PitchArray
        a one dimensional vector
    handle : cublashandle, optional
        handle to cublas
    """
    if handle is None:
        handle = cublashandle()
    dtype = A.dtype
    if dtype == np.float64:
        nrmfunc = cublas.cublasDnrm2
    elif dtype == np.complex128:
        nrmfunc = cublas.cublasDznrm2
    elif dtype == np.float32:
        nrmfunc = cublas.cublasSnrm2
    elif dtype == np.complex64:
        nrmfunc = cublas.cublasScnrm2
    result = nrmfunc(handle.handle, A.size, A.gpudata, 1)
    return result
