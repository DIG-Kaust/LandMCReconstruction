import numpy as np
import os
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from examples.seismic.elastic.wavesolver import ElasticWaveSolver
from examples.seismic.utils import AcquisitionGeometry
from examples.seismic import Model, RickerSource
from examples.seismic.source import TimeAxis
from examples.seismic.source import RickerSource, Receiver, TimeAxis
from devito import *
from sympy import init_printing, latex
init_printing(use_latex='mathjax')

class Elastic2DDevito():
    def __init__(self):
        pass

    def create_model(self, vp, vs, rho, shape, origin, spacing, space_order: int=8, nbl: int=20, fs: bool=False, seismic_model=None):
        if seismic_model:
            self.model = seismic_model
        else :
            self.model = Model(vp=vp, vs=vs, b=1/rho, origin=origin, shape=shape, spacing=spacing, 
                               dtype=np.float32, space_order=space_order, nbl=nbl, bcs="mask", fs=fs)


    def create_geometry(self, src_x: npt.DTypeLike, src_z: npt.DTypeLike, 
                        rec_x: npt.DTypeLike, rec_z: npt.DTypeLike, t0: float, tn: int, 
                        src_type: str=None, f0: float=60):
        """Create geometry and time axis

        Parameters
        ----------
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in m
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in m
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in m
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in m
        t0 : :obj:`float`
            Initial time
        tn : :obj:`int`
            Number of time samples
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nrec,2))
        src_coordinates[:,0] = src_x
        src_coordinates[:,1] = src_z
        
        rec_coordinates = np.empty((nrec,2))
        rec_coordinates[:,0] = rec_x
        rec_coordinates[:,1] = rec_z

        self.geometry = AcquisitionGeometry(self.model, rec_coordinates, src_coordinates, 
                                            t0=t0, tn=tn, src_type=src_type, 
                                            f0=None if f0 is None else f0 * 1e-3,fs=self.model.fs)

    def solve_one_shot(self, isrc, wav: npt.DTypeLike=None, dt: float=None, saveu: bool=False):
        """Solve wave equation for one shot

        Parameters
        ----------
        isrc : :obj:`float`
            Index of source to model
        wav : :obj:`float`, optional
            Wavelet (if not provided, use wavelet in geometry)
        dt : :obj:`float`, optional
            Time sampling of data (will be resampled)
        saveu : :obj:`bool`, optional
            Save snapshots

        Returns
        -------
        d : :obj:`np.ndarray`
            Data
        v : :obj:`np.ndarray`
            Wavefield snapshots

        """
        #Create geometry for single source
        geometry = AcquisitionGeometry(self.model,self.geometry.rec_positions,self.geometry.src_positions[isrc,:],
                                       self.geometry.t0, self.geometry.tn, f0 = self.geometry.f0,
                                       src_type = self.geometry.src_type, fs=self.model.fs)

        src = None
        if wav is not None:
            # assign wavelet
            dt = self.model.critical_dt
            time_range = TimeAxis(start=self.geometry.t0, stop=self.geometry.tn, step=dt)
            src = RickerSource(name='src', grid=self.model.grid, f0=self.geometry.f0, time_range=time_range)
            # src = RickerSource(name='src', grid=self.model.grid, f0=20,
            #                    npoint=1, time_range=geometry.time_axis)
            src.coordinates.data[:, 0] = geometry.src.coordinates.data[isrc, 0]
            src.coordinates.data[:, 1] = geometry.src.coordinates.data[isrc, 1]
            # src.data[:] = wav

        #Solve
        solver = ElasticWaveSolver(self.model, geometry, space_order=self.model.space_order)
        d1,d2,v,tau,_ = solver.forward(src=src,save=saveu)

        #Resample
        taxis = None
        if dt is not None:
            d1 = d1.resample(dt)
            d2 = d1.resample(dt)
            taxis = d1.time_values

        return d1,d2,v,tau

    def plot_velocity(self, source=True, receiver=True, colorbar=True, cmap="jet", figsize=(7, 7), figpath=None):
        """Display velocity model

        Plot a two-dimensional velocity field. Optionally also includes point markers for
        sources and receivers.

        Parameters
        ----------
        source : :obj:`bool`, optional
            Display sources
        receiver : :obj:`bool`, optional
            Display receivers
        colorbar : :obj:`bool`, optional
            Option to plot the colorbar
        cmap : :obj:`str`, optional
            Colormap
        figsize : :obj:`tuple`, optional
            Size of figure
        figpath : :obj:`str`, optional
            Full path (including filename) where to save figure

        """
        domain_size = 1.e-3 * np.array(self.model.domain_size)
        extent = [self.model.origin[0], self.model.origin[0] + domain_size[0],
                  self.model.origin[1] + domain_size[1], self.model.origin[1]]

        slices = list(slice(self.model.nbl, -self.model.nbl) for _ in range(2))
        if self.model.fs:
            slices[1] = slice(0, -self.model.nbl)
        if getattr(self.model, 'vp', None) is not None:
            field = self.model.vp.data[slices]
        else:
            field = self.model.lam.data[slices]

        plt.figure(figsize=figsize)
        plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                          vmin=np.min(field), vmax=np.max(field),
                          extent=extent)
        plt.xlabel('X position (km)')
        plt.ylabel('Depth (km)')

        # Plot source points, if provided
        if receiver:
            plt.scatter(1e-3 * self.geometry.rec_positions[::5, 0], 1e-3 * self.geometry.rec_positions[::5, 1],
                        s=25, c='black', marker='D')

        # Plot receiver points, if provided
        if source:
            plt.scatter(1e-3 * self.geometry.src_positions[::5, 0], 1e-3 * self.geometry.src_positions[::5, 1],
                        s=25, c='red', marker='o')

        # Ensure axis limits
        plt.xlim(self.model.origin[0], self.model.origin[0] + domain_size[0])
        plt.ylim(self.model.origin[1] + domain_size[1], self.model.origin[1])

        # Create aligned colorbar on the right
        if colorbar==True:
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(plot, cax=cax)
            cbar.set_label('Velocity (km/s)')

        # Save figure
        if figpath:
            plt.savefig(figpath)


    def plot_shotrecord(self, rec, colorbar=True, clip=1, extent=None, figsize=(8, 8), figpath=None, cmap=cm.gray):
        """Plot a shot record (receiver values over time).


        Plot a two-dimensional velocity field. Optionally also includes point markers for
        sources and receivers.

        Parameters
        ----------
        rec : :obj:`np.ndarray`, optional
            Receiver data of shape (time, points).
        colorbar : :obj:`bool`, optional
            Option to plot the colorbar
        clip : :obj:`str`, optional
            Clipping
        figsize : :obj:`tuple`, optional
            Size of figure
        figpath : :obj:`str`, optional
            Full path (including filename) where to save figure

        """

        scale = np.max(rec) * clip
        # extent = [self.model.origin[0], self.model.origin[0] + 1e-3 * self.model.domain_size[0],
        #           1e-3 * self.geometry.tn, self.geometry.t0]
        extent = [self.model.origin[0], self.model.origin[0] + 1e-3 * self.model.domain_size[0],
                  1e-3 * self.geometry.time_axis.stop, 0]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot = ax.imshow(rec, vmin=-scale, vmax=scale, cmap=cmap, extent=extent)
        ax.axis('tight')
        ax.set_xlabel('X position (km)')
        ax.set_ylabel('Time (s)')

        # Create aligned colorbar on the right
        # if colorbar:
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     plt.colorbar(plot, cax=cax)

        # Save figure
        if figpath:
            plt.savefig(figpath)

        return ax
        
        
    def solve_one_manual(self, space_order: int=8):
        x, z = self.model.grid.dimensions
        t = self.model.grid.stepping_dim
        time = self.model.grid.time_dim
        s = time.spacing

        v = VectorTimeFunction(name='v', grid=self.model.grid, space_order=space_order, time_order=1)
        tau = TensorTimeFunction(name='t', grid=self.model.grid, space_order=space_order, time_order=1)

        src = self.source
        src_xx = src.inject(field=tau.forward[0, 0], expr=s*src)
        src_zz = src.inject(field=tau.forward[1, 1], expr=s*src)

        rec = self.receiver
        rec_term = rec.interpolate(expr=tau[0, 0] + tau[1, 1])
        
        l, mu, ro = self.model.lam, self.model.mu, self.model.b
        pde_v = v.dt - ro * div(tau)
        pde_tau = tau.dt - l * diag(div(v.forward)) - mu * (grad(v.forward) + grad(v.forward).transpose(inner=False))

        u_v = Eq(v.forward, self.model.damp * solve(pde_v, v.forward))
        u_t = Eq(tau.forward,  self.model.damp * solve(pde_tau, tau.forward))
        
        op = Operator([u_v] + [u_t] + src_xx + src_zz + rec_term)

        op(dt=self.model.critical_dt)

        return rec
        

    def add_geometry(self, src_x, src_z, rec_x, rec_z, t0, tn, f0):
        dt = self.model.critical_dt
        time_range = TimeAxis(start=t0, stop=tn, step=dt)
        nrec = len(rec_x)
        rec = Receiver(name="rec", grid=self.model.grid, npoint=nrec, time_range=time_range)
        rec.coordinates.data[:, 0] = rec_x
        rec.coordinates.data[:, -1] = rec_z
        self.receiver = rec
        
        src = RickerSource(name='src', grid=self.model.grid, f0=f0*1e-3, time_range=time_range)
        src.coordinates.data[:,0] = src_x
        src.coordinates.data[:,-1] = src_z
        self.source = src



    