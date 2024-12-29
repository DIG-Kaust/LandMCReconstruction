import numpy as np
import cupy as cp
from pylops.basicoperators import Identity, Pad, BlockDiag, VStack, Transpose, FirstDerivative, Diagonal,Gradient
from pylops.basicoperators import *
from pylops.signalprocessing import Shift, FFT2D
from pylops.optimization.sparsity import fista

from pylops.signalprocessing.patch2d import patch2d_design
from pylops.signalprocessing import *

from pyproximal.proximal import AffineSet,L1,Orthogonal,L2,L0,TV,L21
from pyproximal.optimization.primal import *
from pyproximal.optimization.primaldual import *
from pylops.utils.tapers import taper

from landmc.slopes import multicomponent_slopes_inverse
from landmc.preprocessing import subsample
import pylops

import pyproximal
import time
import numpy as np
from pylops.utils.backend import get_array_module, to_numpy

cp_asarray = cp.asarray
cp_asnumpy = cp.asnumpy

import time
import numpy as np

from pylops.utils.backend import get_array_module, to_numpy

class MCdata():
    def __init__(self, data, data_grad, x,t,sc1, Rop,D1op, Fop, Mf,nsub,fulldata_grid=None):
        """
        Initialize the MCdata object.

        Parameters:
        -----------
        data : ndarray
            Sparse data array of shape (nx, nt).
        data_grad : ndarray
            Sparse gradient data array of shape (nx, nt).
        x : ndarray
            Dense offset axis.
        t : ndarray
            Dense time axis.
        sc1 : float
            Scaling factor for gradient data.
        Rop : Pylops.LinearOperator
            Decimation operator.
        D1op : Pylops.LinearOperator
            F-K domain derivative operator.
        Fop : Pylops.LinearOperator
            Fourier transform operator.
        Mf : Pylops.LinearOperator
            F-K domain masking operator.
        nsub : int
            Subsampling factor. Defined as the ratio of nx_full over nx (nx_full / nx)
        fulldata : ndarray, optional
            Dense data array of shape (nx_full, nt). If unknown, should be initialized with zeros with desired grid size.
        """
        
        self.data = data
        self.data_grad = data_grad
        self.fulldata = fulldata_grid
        
        self.Rop = Rop
        self.Fop = Fop
        self.Mf = Mf
        self.D1op = D1op
        
        self.sc1 = sc1
        self.nsub = nsub
        self.x = x
        self.t = t
        self.dx = x[1]-x[0]
        self.dt = t[1]-t[0]
        
        #Fulldatasize
        self.nx = self.fulldata.shape[0]
        self.nt = self.fulldata.shape[1]
    
    def update_rop(self):
        _,_,self.Rop = subsample(self.fulldata,self.nsub)        
        
    def estimate_slope(self,reg=100,niter=200,use_weighted=False):
        """
        Estimates local slopes and create the slope regularization operator (SRegop).

        Parameters:
        -----------
        reg : float, optional
            Regularization parameter for slope estimation. Default is 100.
        niter : int, optional
            Number of iterations for slope optimization. Default is 200.
        use_weighted : bool, optional
            Flag to indicate whether to use weighted estimation. Default is False.
        """
        dict_slope_opt = dict(niter=niter)
        self.slope,Wop = multicomponent_slopes_inverse(self.Rop.H @ cp_asarray(self.data), self.dx, self.dt,
                                                              graddata = self.Rop.H @ cp_asarray(self.data_grad), 
                                                              reg=reg, Rop = self.Rop,use_weighted=use_weighted,**dict_slope_opt)
        self.slope = cp_asnumpy(self.slope)
        self.Wop = Wop

        D1op0 = FirstDerivative(dims=(self.nx, self.nt), axis=0, sampling=self.dx, order=5, edge=True, dtype="complex128")
        D1op1 = FirstDerivative(dims=(self.nx, self.nt), axis=1, sampling=self.dt, order=5, edge=True, dtype="complex128")

        slope_D1op1 = Diagonal(cp_asarray(self.slope).T.ravel()) * D1op1
        self.SRegop = D1op0 + slope_D1op1
        
        self.update_rop()
        
        
    def MC_inversion_FISTA(self,niter,eps_FK=1e-4,eps_slope=1, 
                           firstgrad=True, slope=True):
        """
        Performs Soft Data Constraint multi-channel inversion using the FISTA algorithm .

        Parameters:
        -----------
        niter : int
            Number of iterations.
        eps_FK : float
            Sigma for sparsity norm. Default is 1e-4
        eps_slope : float, optional
            Regularization parameter for slope data. Default is 1.
        firstgrad : bool, optional
            Flag for using first gradient data. Default is True.
        slope : bool, optional
            Flag for using slope data. Default is True.
        
        Returns:
        --------
        xinv : ndarray
            Inverted model after primal-dual optimization.
        """
        
        if firstgrad==False and slope==False:
            F1op = VStack([self.Rop*self.Fop.H*self.Mf])
            data_ = cp_asarray(self.data.ravel())
            xinv, _, _ = fista(F1op, data_, niter=niter, eps=eps_FK,
                                eigsdict=dict(niter=5, tol=1e-2), show=True)
            data_ = cp_asnumpy(data_)
            
        elif firstgrad==True and slope==False:
            F1op = VStack([self.Rop*self.Fop.H, 
                           self.sc1*self.Rop * self.Fop.H * self.D1op ]) * self.Mf

            data_ = np.concatenate((self.data.ravel(), self.sc1*self.data_grad.ravel()), axis=0)
            data_ = cp_asarray(data_)
            xinv, _, _ = fista(F1op, data_, niter=niter, eps=eps_FK,
                               eigsdict=dict(niter=5, tol=1e-2), show=True)
            data_ = cp_asnumpy(data_)
            
        elif firstgrad==False and slope==True:
            F1op = VStack([self.Rop*self.Fop.H, 
                           eps_slope * self.SRegop * self.Fop.H]) * self.Mf

            data_ = np.concatenate((self.data.ravel(), np.zeros(self.nt*self.nx)), axis=0)
            data_ = cp_asarray(data_)
            xinv, _, _ = fista(F1op, data_, niter=niter, eps=eps_FK,
                               eigsdict=dict(niter=5, tol=1e-2), show=True)
            data_ = cp_asnumpy(data_)

        elif firstgrad==True and slope==True:
            F1op = VStack([self.Rop*self.Fop.H, 
                           self.sc1*self.Rop * self.Fop.H * self.D1op, 
                           eps_slope*self.SRegop * self.Fop.H]) * self.Mf

            data_ = np.concatenate((self.data.ravel(), self.sc1*self.data_grad.ravel(), np.zeros(self.nt*self.nx)), axis=0)
            data_ = cp_asarray(data_)
            xinv, _, _ = fista(F1op, data_, niter=niter, eps=eps_FK,
                               eigsdict=dict(niter=5, tol=1e-2), show=True)
            data_ = cp_asnumpy(data_)
        
        xinv = np.real(self.Fop.H * self.Mf * xinv).reshape(self.nx,self.nt)
        xinv = cp_asnumpy(xinv)
        self.update_rop()
        
        return xinv
    
    def MC_inversion_PD(self,niter,tau,mu,
                        eps_FK=1e-4,sigma_slope=1,
                        x0=None):
        
        """
        Perform hard data constraint multi-channel inversion using the Primal-Dual algorithm.

        Parameters:
        -----------
        niter : int
            Number of iterations for the primal-dual algorithm.
        eps_FK : float
            Regularization parameter for L1 sparsity. Default is 1e-4
        sigma_slope : float
            Regularization parameter for slope (L2 norm). Default is 1
        tau : float
            Primal step size.
        mu : float
            Dual step size.
        x0 : array-like, optional
            Initial model (default is None, which initializes with data).

        Returns:
        --------
        xinv : ndarray
            Inverted model after primal-dual optimization.
        """
        self.update_rop()
        #L1 Sparsity
        L1sparse = Orthogonal(L1(eps_FK),self.Mf*self.Fop)

        #Indicator Function of data
        F1data = self.Rop
        laffdata = AffineSet(Op=F1data, b=cp_asarray(self.data.ravel()),niter=1)
        
        #L2 of first derivative data
        F1deriv = self.sc1 * self.Rop 
        laffderiv = L2(Op=F1deriv, b=cp_asarray(self.sc1*self.data_grad.ravel()),niter=100)

        #L2 Slope
        b_slope = cp_asarray(np.zeros(self.nt*self.nx).ravel())
        L2slope = L2(Op=self.SRegop,b=b_slope,sigma=sigma_slope,niter=100)
        
        if x0 is None:
            data0 = self.Rop.H*cp_asarray(self.data)
        else:
            data0=x0
            
        datasize = self.nx*self.nt
        self.datasize = datasize
        
        #Stacking Proximal Operator
        Iop = Identity(datasize)
        Kop = VStack([self.Fop.H*self.Mf*self.Fop,
                      self.Fop.H*self.D1op*self.Mf*self.Fop,
                      self.Fop.H*self.Mf*self.Fop],dtype=np.complex128)

        lg = pyproximal.VStack([laffdata,laffderiv,L2slope],nn=[datasize,datasize,datasize])
        lf = L1sparse

        #Inversion
        xinv = PrimalDual(lf,lg,A=Kop,x0=cp_asarray(data0.ravel()), 
                        tau=tau,mu=mu,theta=1.,show=True,niter=niter,gfirst=True)
        
        xinv = np.real(xinv).reshape(self.nx,self.nt)
        xinv = cp_asnumpy(xinv)
        
        self.update_rop()
        return xinv

    