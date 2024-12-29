import numpy as np

from scipy.signal import butter, filtfilt
from pylops.signalprocessing import FFT2D, FFTND
from scipy.signal import butter, filtfilt
from pylops.basicoperators import Diagonal, Restriction, FirstDerivative
from pylops.utils.metrics import snr

import matplotlib.pyplot as plt
import random

def butter_lowpass(cutoff, fs, order=5):
    r"""Butterworth low-pass filter

    Design coefficients of butterworth low-pass filter

    Parameters
    ----------
    cutoff : :obj:`float`
        Cut-off frequency
    fs : :obj:`float`
        Sampling frequency
    order : :obj:`int`
        Order of filter

    Returns
    -------
    b : :obj:`np.ndarray`
        Numerator coefficients
    a : :obj:`np.ndarray`
        Denominator coefficients

    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    r"""Butterworth low-pass filter

    Design coefficients of butterworth low-pass filter

    Parameters
    ----------
    cutoff : :obj:`float`
        Cut-off frequency
    fs : :obj:`float`
        Sampling frequency
    order : :obj:`int`
        Order of filter

    Returns
    -------
    b : :obj:`np.ndarray`
        Numerator coefficients
    a : :obj:`np.ndarray`
        Denominator coefficients

    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_filter(data, cutoff, fs, order=5,mode='lowpass'):
    r"""Apply Butterworth Low-pass filter

    Apply Butterworth low-pass filter over time axis of input data

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    cutoff : :obj:`float`
        Cut-off frequency
    fs : :obj:`float`
        Sampling frequency
    order : :obj:`int`
        Order of filter

    Returns
    -------
    y : :obj:`np.ndarray`
        Filtered data

    """
    if mode == 'lowpass':
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
    elif mode == 'highpass':
        b, a = butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
    return y

def fk_transform(data, nfft_x, nfft_t, dx, dt):
    """
    data = data of size (nx,nt)
    nfft_x,nfft_t = number of fourier coeff in ks, number of fourier coeff in f
    dx = x sampling (m)
    dt = t sampling (s)
    
    return:
    D = data in F-K domain
    f = frequency axis
    ks = wavenumber axis 
    Fop = Fourier operator
    """
    nx, nt = data.shape
    f = np.fft.rfftfreq(nfft_t, dt)
    ks = np.fft.fftfreq(nfft_x, dx)
    
    Fop = FFT2D(dims=(nx, nt), nffts=(nfft_x, nfft_t),real=True, dtype=complex)

    # apply FK transform to data
    D = Fop * data
    
    return D, f, ks, Fop

def subsample(data, nsub, dtype="float64", idx=None):
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
    if idx is not None:
        traces_index_sub = idx
    # Define restriction operator
    Rop = Restriction(dims=data.shape, iava=traces_index_sub, axis=0, dtype=dtype)

    # Apply restriction operator
    data_obs = Rop * data
    data_mask = Rop.mask(data.ravel())

    return data_obs, data_mask, Rop

def irregular_subsample(data,nsub,dtype="float64"):
    r"""Irregular subsample data

    Create a randomly irregular restriction operator for compressive sensing manner and apply to data

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
    randomidx = random.sample(range(0,nx),nx//nsub+1)
    randomidx = np.array(randomidx)
    randomidx.sort()

    # for i in range(len(randomidx)-1):
    #     dist = randomidx[i+1]-randomidx[i]
    #     while dist>9:
    #         if dist>9:
    #             randomidx[i+1] = randomidx[i+1] - np.random.randint(nsub,dist-1)
    #         dist = randomidx[i+1]-randomidx[i]

    randomidx[-1] = nx-2
    print(randomidx)
    print(randomidx[1:]-randomidx[:-1])

    # randomidx = np.random.randint(0,nx,nx//nsub+1)
    # randomidx.sort()
    traces_index_sub = randomidx

    # Define restriction operator
    Rop = Restriction(dims=data.shape, iava=traces_index_sub, axis=0, dtype=dtype)

    # Apply restriction operator
    data_obs = Rop * data
    data_mask = Rop.mask(data.ravel())

    return data_obs, data_mask, Rop, randomidx

def jittered_subsample(data,nsub,dtype='float64',e=None):
    if e is None:
        e=nsub
    nx = data.shape[0]
    idx = np.arange(0,nx//nsub+1)
    traces_index_sub = np.zeros_like(idx)

    for index in idx:
        j = (1-nsub)/2 + nsub*index + np.random.uniform(np.floor(-(e-1)/2),np.floor((e-1)/2))
        traces_index_sub[index] = j
    traces_index_sub[0] = 0
    print(traces_index_sub)
    print(traces_index_sub[1:]-traces_index_sub[:-1])
   # Define restriction operator
    Rop = Restriction(dims=data.shape, iava=traces_index_sub, axis=0, dtype=dtype)

    # Apply restriction operator
    data_obs = Rop * data
    data_mask = Rop.mask(data.ravel())

    return data_obs, data_mask, Rop, traces_index_sub

def fk_filter_design(f, ks, vel, fmax, critical=1.00, koffset=0.002):
    r"""FK filter mask

    Design mask to be applied in FK domain to filter energy outside of the chosen signal cone
    based on the following relation ``|k_x| < f / v``.

    Parameters
    ----------
    f : :obj:`np.ndarray`
        Frequency axis
    ks : :obj:`np.ndarray`
        Spatial wavenumber axis
    vel : :obj:`float`
        Maximum velocity to retain
    fmax : :obj:`float`, optional
        Maximum frequency to retain
    critical : :obj:`float`, optional
        Critical angle (used to proportionally adjust velocity and therefore the wavenumber cut-off )
    koffset : :obj:`float`, optional
        Offset to apply over the wavenumber axis to the mask

    Returns
    -------
    mask : :obj:`np.ndarray`
        Mask of size :math:`n_{ks} \times n_f`

    """
    nfft_t = f.size
    fmask = np.zeros(nfft_t)
    fmask[np.abs(f) < fmax] = 1

    [kx, ff] = np.meshgrid(ks, f, indexing='ij')
    mask = np.abs(kx) < (critical * np.abs(ff) / vel + koffset)
    mask = mask.T
    mask *= fmask[:, np.newaxis].astype(bool)
    mask = mask.astype(np.float)

    return mask

def mask(data, thresh, itoff=20):
    r"""Apply mask

    Create mask trace-wise using threshold to indentify the start of the mask.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    thresh : :obj:`float`
        Threshold (the mask excludes everything before the first value larger than ``thresh``)
    itoff : :obj:`int`, optional
        Number of samples used to shift the mask upward

    Returns
    -------
    masktx : :obj:`np.ndarray`
        Mask of size :math:`n_x \times n_t`

    """
    nx, nt = data.shape
    masktx = np.ones((nx, nt))
    for ix in range(nx):
        itmask = np.where(np.abs(data[ix]) > thresh)[0]
        if len(itmask) > 0:
            masktx[ix, :max(0, itmask[0] - itoff)] = 0.
        else:
            masktx[ix] = masktx[ix - 1]
    return masktx

def mask_cut(data,vcut,t,dt,offset,it0=0,cut='up'):
    #Mute direct and refracted wave
    pcut = 1/vcut
    data_cut = data.copy()
    nx = len(offset)

    ix = np.arange(nx)
    tevent = t[0] + offset * pcut
    tevent = (tevent - t[0]) / dt
    itevent = tevent.astype(int)+it0
    mask = (itevent < nt - 1) & (itevent >= it0)
    for i in range(nx):
        #For every trace
        if i<len(itevent[mask]):
            if cut=='up':
                data_cut[ix[mask][i],:itevent[mask][i]] = 0
            elif cut=='down':
                data_cut[ix[mask][i],itevent[mask][i]:] = 0
    #     else:
    #         data_cut[i,:nt]=0
    
    return data_cut,mask,itevent
    
def plot_reconstruction_2d_real(data, data_obs, datarec, Fop, x, t, dx, f, ks, nfft_t, vel=None,seis_scale=1e-1,fk_scale=1e-1,fk_uplim=150 ):
    """2D reconstruction visualization

    Display original and reconstructed datasets and their error.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Full data of size :math:`n_x \times n_t`
    datarec : :obj:`np.ndarray`
        Observed data of size :math:`n_x \times n_t`
    datarec : :obj:`np.ndarray`
        Reconstructed data of size :math:`n_x \times n_t`
    Fop : :obj:`pylops.LinearOperator`, optional
        2D Fourier operator
    x : :obj:`np.ndarray`
       Spatial axis
    t : :obj:`np.ndarray`
       Time axis
    dx : :obj:`float`
       Spatial sampling
    f : :obj:`np.ndarray`
       Frequency axis
    ks : :obj:`np.ndarray`
       Spatial wavenumber axis
    vel : :obj:`float`
       Velocity at receivers

    """
    D = Fop * data
    D_obs = Fop * data_obs
    Drec = Fop * datarec
    nt = data.shape[1]

    fig, axs = plt.subplots(2, 4, figsize=(24, 12), gridspec_kw={'height_ratios': [2, 1]})
    axs[0, 0].imshow(data.T, cmap='gray', aspect='auto', vmin=-seis_scale, vmax=seis_scale,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 0].set_title('Original')
    axs[0, 0].set_xlabel('Offset (m)')
    axs[0, 0].set_ylabel('TWT (s)')
    
    axs[0, 1].imshow(data_obs.T, cmap='gray', aspect='auto', vmin=-seis_scale, vmax=seis_scale,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 1].set_title('Decimated (SNR=%.2f)'%(snr(data_obs,data)))
    axs[0, 1].set_xlabel('Offset (m)')
    
    axs[0, 2].imshow(datarec.T, cmap='gray', aspect='auto', vmin=-seis_scale, vmax=seis_scale,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 2].set_title('Reconstructed (SNR=%.2f)'%(snr(datarec,data)))
    axs[0, 2].set_xlabel('Offset (m)')
    
    axs[0, 3].imshow(data.T - datarec.T, cmap='gray', aspect='auto', vmin=-seis_scale, vmax=seis_scale,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 3].set_title('Error')
    axs[0, 3].set_xlabel('Offset (m)')

    axs[1, 0].imshow(np.fft.fftshift(np.abs(D).T, axes=1), cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=fk_scale,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nfft_t//2-1], f[0]))
    axs[1, 0].plot(f / vel, f, 'w'), axs[1, 0].plot(-f / vel, f, 'w')
    axs[1, 0].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 0].set_ylim(fk_uplim, 0)
    axs[1, 0].set_xlabel('Wavenumber (1/m)')
    axs[1, 0].set_ylabel('Frequency (Hz)')
    
    axs[1, 1].imshow(np.fft.fftshift(np.abs(D_obs).T, axes=1), cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=fk_scale,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nfft_t//2-1], f[0]))
    axs[1, 1].plot(f / vel, f, 'w'), axs[1, 0].plot(-f / vel, f, 'w')
    axs[1, 1].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 1].set_ylim(fk_uplim, 0)
    axs[1, 1].set_xlabel('Wavenumber (1/m)')
    axs[1, 1].set_ylabel('Frequency (Hz)')
    
    axs[1, 2].imshow(np.fft.fftshift(np.abs(Drec).T, axes=1), cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=fk_scale,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nfft_t//2-1], f[0]))
    axs[1, 2].plot(f / vel, f, 'w'), axs[1, 1].plot(-f / vel, f, 'w')
    axs[1, 2].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 2].set_ylim(fk_uplim, 0)
    axs[1, 2].set_xlabel('Wavenumber (1/m)')
    
    axs[1, 3].imshow(np.fft.fftshift(np.abs(D-Drec).T, axes=1), cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=fk_scale,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nfft_t//2-1], f[0]))
    axs[1, 3].plot(f / vel, f, 'w'), axs[1, 2].plot(-f / vel, f, 'w')
    axs[1, 3].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 3].set_ylim(fk_uplim, 0)
    axs[1, 3].set_xlabel('Wavenumber (1/m)')
