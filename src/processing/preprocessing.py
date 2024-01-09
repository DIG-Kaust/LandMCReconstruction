import numpy as np

from scipy.signal import butter, filtfilt
from pylops.signalprocessing import FFT2D, FFTND
from scipy.signal import butter, filtfilt
from pylops.basicoperators import Diagonal, Restriction, FirstDerivative

def fk_transform(data, nfft_x, nfft_t, dx, dt):
    # Data shape == nt,nx
    nt, nx = data.shape
    f = np.fft.fftfreq(nfft_t, dt/1000)
    ks = np.fft.fftfreq(nfft_x, dx)
    Fop = FFT2D(dims=(nx, nt), nffts=(nfft_x, nfft_t), dtype=complex)

    data_fk = Fop*data.T
    
    return data_fk, f, ks

def subsample(data, nsub, dtype="float64"):
    r"""Subsample data

    Create restriction operator and apply to data

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    nsub : :obj:`int`
        Subsampling factor
    dtype : :obj:`str`, optional
        Dtype of operator

    Returns
    -------
    data_obs : :obj:`np.ndarray`
        Restricted data of size :math:`(n_x // n_{sub}) \times n_t`
    data_mask : :obj:`np.ndarray`
        Masked data of size :math:`n_x \times n_t`
    Rop : :obj:`pylops.LinearOperator`
        Restriction operator

    """
    # identify available traces
    nx = data.shape[0]
    traces_index = np.arange(nx)
    traces_index_sub = traces_index[::nsub]
    

    # Define restriction operator
    Rop = Restriction(dims=data.shape, iava=traces_index_sub, axis=0, dtype=dtype)

    # Apply restriction operator
    data_obs = Rop * data
    data_mask = Rop.mask(data.ravel())

    return data_obs, data_mask, Rop