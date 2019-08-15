"""
Ensemble Kalman Filter class

Created on 2019-04-16-12-31
Author: Stephan Rasp, raspstephan@gmail.com
"""
import sys
def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules
import numpy as np
from copy import deepcopy
import xarray as xr
if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
import multiprocessing as mp
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pdb


def multi_helper(l, nt):
    l.iterate(nt)
    return l


class EnKF():
    def __init__(self, l96, nens, obs_noise, cyc_len, H=lambda l: l.state,
                 get_state=lambda l: l.state, set_state=lambda l, x: l.set_state(x),
                 mp=None, par_noise=0, y=None, climate=False):
        """
        Ensemble Kalman Filter class.
        Runs an ensemble, a deterministic run and (if not in climate mode) a truth run. Performs DA updates each cycle.
        Can run in NWP setup and climate setup. Climate setup basically means that we are fitting to long term statistics and only learning parameters, not the state.
        # Arguments
            l96: L96 model that will be copied for each of the ensemble members
            nens: Number of ensemble members
            obs_noise: Vector (shape m) with observation variance
            cyc_len: Integration time (in days) for each DA cycle
            H: oObservation operator n --> m, function that takes a L96 object and returns the observations
            get_state: Function that takes the L96 object and returns the state vector x
            set_state: Reverse function. Takes the L96 object and the analysis state and accordingly modifies the L96 object.
            mp: Parallel processes for ensemble integration. Not parallel if None.
            par_noise: Currently not implemented
            y: Fixed observations (shape m), to be used in climate setup. If given, no truth run is run.
            climate: Boolian flag whether climate mode is on. Basically means that after each cycle the initial conditions are set back to the original.
        """
        # Copy arguments
        self.l96, self.nens, self.obs_noise, self.cyc_len, self.mp, self.H, self.get_state, self.set_state, self.par_noise, self.y, self.climate = \
            l96, nens, obs_noise, cyc_len, mp, H, get_state, set_state, par_noise, y, climate
        # Create diagonal observation error matrix
        self.R = np.diag(np.ones(H(l96).shape) * obs_noise)
        # Allocate history lists for parameter estimation
        self.parameter_history_det = []; self.parameter_history_ens = []
        if self.climate: self.climate_error = []
        else: self.mse_det, self.mse_ens = [], []
        # Get dimensions
        self.n = self.get_state(self.l96).shape[0]
        self.m = H(l96).shape[0]
        print(f'Dimensions: n = {self.n}, m = {self.m}')

    def initialize(self, ic, ic_noise):
        """Initialize models with initial conditions. Deterministic and ensemble ICs are perturbed with ic_noise
        # Arguments
            ic: Initial conditions (shape n)
            ic_noise: Noise to perturb IC (shape n or 1)
        """
        self.ic, self.ic_noise = ic, ic_noise
        self.l96_det = deepcopy(self.l96)
        self.l96_det.set_state(ic.copy() + np.random.normal(0, ic_noise, ic.shape))
        self.l96_ens = [deepcopy(self.l96) for n in range(self.nens)]
        for i, l in enumerate(self.l96_ens):
            l.set_state(ic.copy() + np.random.normal(0, ic_noise, ic.shape))
        if self.y is None:
            self.l96_tru = deepcopy(self.l96)
            self.l96_tru.set_state(ic.copy())

    def initialize_parameters(self, fn, priors, sigmas):
        """For parameter estimation we also need to perturb the parameters of the ensemble in the beginning.
        The deterministic parameter will not be perturbed from its prior.
        # Arguments
            fn: Function that takes the L96 object, a vector (shape N_param) of priors and sigmas and randomly perturbs the parameter.
            priors: mean of distribution for each parameter
            sigmas: variances of distribution for each parameter
        """
        # Deterministic run without noise
        fn(self.l96_det, priors, np.zeros(len(np.atleast_1d(sigmas))))
        self.parameter_history_det.append(self.l96_det.parameters)
        ens_hist = []
        for l in self.l96_ens:
            fn(l, priors, sigmas)
            ens_hist.append(l.parameters)
        self.parameter_history_ens.append(np.array(ens_hist))

    def step(self):
        # 1. Forecast
        #
        if self.mp is None:
            self.l96_det.iterate(self.cyc_len)
            for l in self.l96_ens:
                l.iterate(self.cyc_len)
            if self.y is None: self.l96_tru.iterate(self.cyc_len)
        else:
            self.mp_forecast()

        # Get arrays
        x_f = self.get_state(self.l96_det)
        x_f_ens = np.array([self.get_state(l) for l in self.l96_ens])
        hx_f = self.H(self.l96_det)
        hx_f_ens = np.array([self.H(l) for l in self.l96_ens])
        if self.y is None:
            y = self.H(self.l96_tru) + np.random.normal(0, np.sqrt(self.obs_noise), self.H(self.l96_tru).shape)
        else:
            y = self.y
        y_ens = np.array([y + np.random.normal(0, np.sqrt(self.obs_noise), y.shape) for i in range(self.nens)])

        # New analysis
        K = self.kalman_gain(x_f_ens, hx_f_ens)
        x_a = x_f + K @ (y - hx_f)
        x_a_ens = np.array([
            x_f_ens[i] + K @ (y_ens[i] - hx_f_ens[i]) for i in range(self.nens)
        ])

        # Update state in models
        self.set_state(self.l96_det, x_a)
        for i, l in enumerate(self.l96_ens):
            self.set_state(l, x_a_ens[i])

        # Write analysis in history
        self.parameter_history_det.append(self.l96_det.parameters)
        self.parameter_history_ens.append(np.array([l.parameters for l in self.l96_ens]))

        if self.climate:
            self.l96_det.erase_history()
            self.l96_det.set_state(self.ic.copy() + np.random.normal(0, self.ic_noise, self.ic.shape))
            for i, l in enumerate(self.l96_ens):
                l.erase_history()
                l.set_state(self.ic.copy() + np.random.normal(0, self.ic_noise, self.ic.shape))
            self.climate_error.append(np.sqrt((((y - hx_f_ens)/np.sqrt(self.obs_noise))**2)).mean())
        else:
            self.mse_det.append(((self.l96_tru.X - self.l96_det.X)**2).mean())
            self.mse_ens.append([((self.l96_tru.X - l.X)**2).mean() for l in self.l96_ens])

    def kalman_gain(self, x_f_ens, hx_f_ens):
        X = (x_f_ens - x_f_ens.mean(0)).T
        Y = (hx_f_ens - hx_f_ens.mean(0)).T
        K = X @ Y.T @ inv(Y @ Y.T + (self.nens - 1) * self.R)
        return K

    def iterate(self, ncycles, noprog=False):
        for n in tqdm(range(ncycles), disable=noprog):
            self.step()

    def mp_forecast(self):
        if self.y is None:
            all_ls = [self.l96_tru, self.l96_det] + self.l96_ens
        else:
            all_ls = [self.l96_det] + self.l96_ens
        pool = mp.Pool(processes=self.mp)
        results = [pool.apply_async(multi_helper, args=(l, self.cyc_len)) for l in all_ls]
        all_ls = [p.get() for p in results]
        pool.close()
        if self.y is None:
            self.l96_tru, self.l96_det, self.l96_ens = all_ls[0], all_ls[1], all_ls[2:]
        else:
            self.l96_det, self.l96_ens = all_ls[0], all_ls[1:]


def plot_mse(enkf, var='X'):
    for l in enkf.l96_ens:
        mse = (enkf.l96_tru.history - l.history)**2
        mse = mse.mean('x')
        mse[var].plot(c='gray')
    mse = (enkf.l96_tru.history - enkf.l96_det.history)**2
    mse = mse.mean('x')
    mse[var].plot(c='r')
    plt.ylabel(f'MSE({var})')

def plot_params(enkf=None, names=['F', 'h', 'c', 'b'], truths=[10,1,10,10], show_title=True,
                det=None, ens=None):
    det = np.array(enkf.parameter_history_det if enkf is not None else det)
    ens = np.array(enkf.parameter_history_ens if enkf is not None else ens)
    fig, axs = plt.subplots(1, len(names), figsize=(15, 5))
    def panel(ax, det, ens, title, truth):
        for i in range(ens.shape[1]):
            ax.plot(ens[:, i], c='gray')
        ax.plot(det, c='r')
        if show_title: ax.set_title('%s mean = %1.2f / std = %1.2f' % (title, ens[-1].mean(), ens[-1].std()))
        ax.axhline(truth)
    for i, n in enumerate(names):
        panel(axs[i], det[..., i], ens[..., i], n, truths[i])
    plt.tight_layout()


class EnKFParamClimate2():
    def __init__(self, l96, nens, obs_noise, cyc_len, y,
                 H=lambda l: l.state, state=lambda l: l.state, mp=None,
                 par_noise=0, ):
        self.y = y
        self.l96 = l96
        self.nens = nens
        self.obs_noise = obs_noise
        self.R = np.diag(np.ones(H(l96).shape) * obs_noise)
        self.cyc_len = cyc_len
        self.mp = mp
        self.H, self.state, self.par_noise = H, state, par_noise

    def initialize(self, ic, ic_noise, priors=(10, 0, 2, 5), sigmas=(10, 1, 0.1, 10)):
        # Initial conditions
        self.ic = ic
        self.ic_noise = ic_noise
        self.x_t = ic.copy()
        self.x_a = ic.copy() + np.random.normal(0, ic_noise, self.x_t.shape)
        self.x_a_ens = np.array([ic.copy()] * self.nens)
        self.x_a_ens += np.random.normal(0, ic_noise, self.x_a_ens.shape)

        # Models
        self.l96_det = deepcopy(self.l96);
        self.l96_det.set_state(self.x_a)
        self.l96_det.F = priors[0]
        self.l96_det.h = priors[1]
        self.l96_det.c = np.exp(priors[2])
        self.l96_det.b = priors[3]
        self.l96_ens = [deepcopy(self.l96) for n in range(self.nens)]
        for i, l in enumerate(self.l96_ens):
            l.set_state(self.x_a_ens[i])
            l.F = priors[0] + np.random.normal(0, np.sqrt(sigmas[0]))
            l.h = priors[1] + np.random.normal(0, np.sqrt(sigmas[1]))
            l.c = np.exp(priors[2] + np.random.normal(0, np.sqrt(sigmas[2])))
            l.b = priors[3] + np.random.normal(0, np.sqrt(sigmas[3]))
        self.F_det = [self.l96_det.F];
        self.F_ens = [[l.F for l in self.l96_ens]]
        self.h_det = [self.l96_det.h];
        self.h_ens = [[l.h for l in self.l96_ens]]
        self.c_det = [self.l96_det.c];
        self.c_ens = [[l.c for l in self.l96_ens]]
        self.b_det = [self.l96_det.b];
        self.b_ens = [[l.b for l in self.l96_ens]]

    def step(self):
        # Forecast
        if self.mp is None:
            self.l96_det.iterate(self.cyc_len)
            for l in self.l96_ens:
                l.iterate(self.cyc_len)
        else:
            self.mp_forecast()

        # Get arrays
        # pdb.set_trace()
        self.x_f = self.state(self.l96_det)
        self.x_f_ens = np.array([self.state(l) for l in self.l96_ens])
        self.hx_f = self.H(self.l96_det)
        self.hx_f_ens = np.array([self.H(l) for l in self.l96_ens])

        # New analysis
        K = self.kalman_gain(self.x_f_ens, self.hx_f_ens)
        self.x_a = self.x_f + K @ (self.y - self.hx_f)
        self.x_a_ens = np.array([
            self.x_f_ens[i] + K @ (
                    (self.y + np.random.normal(0, np.sqrt(self.obs_noise), self.obs_noise.shape))
                    - self.hx_f_ens[i]) for i in range(self.nens)
        ])

        # Update state in models
        self.l96_det.F = self.x_a[-4]
        self.l96_det.h = self.x_a[-3]
        self.l96_det.c = self.x_a[-2]
        self.l96_det.b = self.x_a[-1]
        for i, l in enumerate(self.l96_ens):
            l.F = self.x_a_ens[i][-4] + np.random.normal(0, self.par_noise)
            l.h = self.x_a_ens[i][-3] + np.random.normal(0, self.par_noise)
            l.c = self.x_a_ens[i][-2] + np.random.normal(0, self.par_noise)
            l.b = self.x_a_ens[i][-1] + np.random.normal(0, self.par_noise)
        # Update parameter history
        self.F_det.append(self.l96_det.F);
        self.F_ens.append([l.F for l in self.l96_ens])
        self.h_det.append(self.l96_det.h);
        self.h_ens.append([l.h for l in self.l96_ens])
        self.c_det.append(self.l96_det.c);
        self.c_ens.append([l.c for l in self.l96_ens])
        self.b_det.append(self.l96_det.b);
        self.b_ens.append([l.b for l in self.l96_ens])

        print('Ensemble mean F = ', self.x_a_ens[:, -4].mean())
        print('Ensemble mean h = ', self.x_a_ens[:, -3].mean())
        print('Ensemble mean c = ', self.x_a_ens[:, -2].mean())
        print('Ensemble mean b = ', self.x_a_ens[:, -1].mean())

        # Erase history and reset ic
        self.l96_det._history_X = []
        self.l96_det._history_Y_mean = []
        self.l96_det._history_Y2_mean = []
        self.l96_det.set_state(self.ic.copy() + np.random.normal(0, self.ic_noise, self.ic.shape))
        for l in self.l96_ens:
            l._history_X = []
            l._history_Y_mean = []
            l._history_Y2_mean = []
            l.set_state(self.ic.copy() + np.random.normal(0, self.ic_noise, self.ic.shape))

    def kalman_gain(self, x_f_ens, hx_f_ens):
        X = (x_f_ens - x_f_ens.mean(0)).T
        Y = (hx_f_ens - hx_f_ens.mean(0)).T
        K = X @ Y.T @ inv(Y @ Y.T + (self.nens - 1) * self.R)
        return K

    def iterate(self, ncycles, noprog=False):
        for n in tqdm(range(ncycles), disable=noprog):
            self.step()

    def mp_forecast(self):
        all_ls = [self.l96_det] + self.l96_ens
        pool = mp.Pool(processes=self.mp)
        results = [pool.apply_async(multi_helper, args=(l, self.cyc_len)) for l in all_ls]
        all_ls = [p.get() for p in results]
        pool.close()
        self.l96_det, self.l96_ens = all_ls[0], all_ls[1:]