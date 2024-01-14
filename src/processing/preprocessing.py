import numpy as np

from scipy.signal import butter, filtfilt
from pylops.signalprocessing import FFT2D, FFTND
from scipy.signal import butter, filtfilt
from pylops.basicoperators import Diagonal, Restriction, FirstDerivative

import matplotlib.pyplot as plt

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

def irregular_subsample(data, nsub, dtype="float64"):
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
    traces_index_sub_0 = traces_index[::nsub*2]
    traces_index_sub_1 = traces_index[1::nsub*2]
    traces_index_sub = np.concatenate((traces_index_sub_0,traces_index_sub_1))
    traces_index_sub = np.sort(traces_index_sub)
    
    # Define restriction operator
    Rop = Restriction(dims=data.shape, iava=traces_index_sub, axis=0, dtype=dtype)

    # Apply restriction operator
    data_obs = Rop * data
    data_mask = Rop.mask(data.ravel())

    return data_obs, data_mask, Rop

def gradient_data(data, nfft_x, nfft_t, dx, dt):
    r"""Gradient data of 2D data

    Compute gradient data of 2D data in frequency-wavenumber domain - i.e.
    apply j*k_x for first derivative and -k_x^2 for second derivative.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    nfft_x : :obj:`int`
        Number of samples in wavenumber axis
    nfft_t : :obj:`int`
        Number of samples in frequency axis
    dx : :obj:`float`
        Spatial sampling
    dt : :obj:`float`
        Time sampling

    Returns
    -------
    d1 : :obj:`np.ndarray`
        First gradient data of size :math:`n_x \times n_t`
    d2 : :obj:`np.ndarray`
        Second gradient data of size :math:`n_x \times n_t`
    sc1 : :obj:`float`
        Scaling for first gradient data
    sc2 : :obj:`np.ndarray`
        Scaling for second gradient data
    Fop : :obj:`pylops.LinearOperator`
        2D Fourier operator
    D1op : :obj:`pylops.LinearOperator`
        First gradient scaling operator in FK domain
    D2op : :obj:`pylops.LinearOperator`
        Second gradient scaling operator in FK domain
    D : :obj:`np.ndarray`
        FK spectrum of data
    D1 : :obj:`np.ndarray`
        FK spectrum of first gradient data
    D2 : :obj:`np.ndarray`
        FK spectrum of second gradient data
    ks : :obj:`np.ndarray`
        Spatial wavenumber axis
    f : :obj:`np.ndarray`
        Frequency axis

    """
    nx, nt = data.shape
    f = np.fft.fftfreq(nfft_t, dt)
    ks = np.fft.fftfreq(nfft_x, dx)
    Fop = FFT2D(dims=(nx, nt), nffts=(nfft_x, nfft_t), dtype=np.complex)

    # apply FK transform to data
    D = Fop * data

    # Compute derivatives in FK domain
    coeff1 = 1j * 2 * np.pi * ks
    coeff2 = -(2 * np.pi * ks) ** 2
    coeff1 = np.repeat(coeff1[:, np.newaxis], nfft_t, axis=1).ravel()
    coeff2 = np.repeat(coeff2[:, np.newaxis], nfft_t, axis=1).ravel()
    D1op = Diagonal(coeff1)
    D2op = Diagonal(coeff2)

    D1 = (D1op * D.ravel()).reshape(nfft_x, nfft_t)
    D2 = (D2op * D.ravel()).reshape(nfft_x, nfft_t)

    d1 = np.real(Fop.H * D1)
    d2 = np.real(Fop.H * D2)

    # Compute scalars that normalize gradients to data
    sc1 = np.max(np.abs(data)) / np.max(np.abs(d1))
    sc2 = np.max(np.abs(data)) / np.max(np.abs(d2))

    return d1, d2, sc1, sc2, Fop, D1op, D2op, D, D1, D2, ks, f

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

def plot_reconstruction_2d(data, datarec, Fop, x, t, dx, f, ks, vel, nfft_t):
    """2D reconstruction visualization

    Display original and reconstructed datasets and their error.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Full data of size :math:`n_x \times n_t`
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
    Drec = Fop * datarec
    nt = data.shape[1]

    fig, axs = plt.subplots(2, 3, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1]})
    axs[0, 0].imshow(data.T, cmap='gray', aspect='auto', vmin=-1e-2, vmax=1e-2,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 0].set_title('Original')
    axs[0, 0].set_xlabel('Offset (m)')
    axs[0, 0].set_ylabel('TWT (s)')
    axs[0, 1].imshow(datarec.T, cmap='gray', aspect='auto', vmin=-1e-2, vmax=1e-2,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 1].set_title('Reconstructed')
    axs[0, 1].set_xlabel('Offset (m)')
    axs[0, 2].imshow(data.T - datarec.T, cmap='gray', aspect='auto', vmin=-1e-2, vmax=1e-2,
                     extent=(x[0], x[-1], t[-1], t[0]))
    axs[0, 2].set_title('Error')
    axs[0, 2].set_xlabel('Offset (m)')

    axs[1, 0].imshow(np.fft.fftshift(np.abs(D).T)[nfft_t//2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=0.1,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nfft_t//2-1], f[0]))
    axs[1, 0].plot(f / vel, f, 'w'), axs[1, 0].plot(f / vel, -f, 'w')
    axs[1, 0].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 0].set_ylim(150, 0)
    axs[1, 0].set_xlabel('Wavenumber (1/m)')
    axs[1, 0].set_ylabel('Frequency (Hz)')
    axs[1, 1].imshow(np.fft.fftshift(np.abs(Drec).T)[nfft_t//2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=0.1,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nfft_t//2-1], f[0]))
    axs[1, 1].plot(f / vel, f, 'w'), axs[1, 1].plot(f / vel, -f, 'w')
    axs[1, 1].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 1].set_ylim(150, 0)
    axs[1, 1].set_xlabel('Wavenumber (1/m)')
    axs[1, 2].imshow(np.fft.fftshift(np.abs(D - Drec).T)[nfft_t//2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=0.1,
                     extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nfft_t//2-1], f[0]))
    axs[1, 2].plot(f / vel, f, 'w'), axs[1, 2].plot(f / vel, -f, 'w')
    axs[1, 2].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
    axs[1, 2].set_ylim(150, 0)
    axs[1, 2].set_xlabel('Wavenumber (1/m)')
    plt.tight_layout()