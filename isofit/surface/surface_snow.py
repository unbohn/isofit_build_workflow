#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: Niklas Bohn, urs.n.bohn@jpl.nasa.gov
#

import numpy as np
import os
import pickle
from scipy.optimize import least_squares

from .surface_multicomp import MultiComponentSurface
from isofit.configs import Config


class SnowSurface(MultiComponentSurface):
    """A model of the surface based on a collection of multivariate
       Gaussians, extended with snow fractional cover and grain size terms."""

    def __init__(self, full_config: Config):

        super().__init__(full_config)

        # TODO: Enforce this attribute in the config, not here (this is hidden)
        self.statevec_names.extend(['Snow_fc', 'Snow_gs'])
        self.scale.extend([1.0, 1.0])
        self.init.extend([0.3, 500])
        self.bounds.extend([[0, 1.0], [0, 1100]])

        self.n_state = self.n_state + 2
        self.snow_ind = len(self.statevec_names) - 2

        isofit_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        rho_model_file = os.path.join(isofit_path, "data", "iop", "pca_models_snow_EMIT_incl_grain_size.pkl")

        f = open(rho_model_file, 'rb')
        models = pickle.load(f)
        f.close()

        self.pca_rfl_snow = models['snow']['model']
        self.pca_rfl_veg = models['veg']['model']
        self.pca_rfl_rock = models['rock']['model']

        self.ind_snow = self.pca_rfl_snow.n_components
        self.ind_veg = self.pca_rfl_snow.n_components + self.pca_rfl_veg.n_components
        self.ind_rock = self.pca_rfl_snow.n_components + self.pca_rfl_veg.n_components + self.pca_rfl_rock.n_components

        state_bounds = []
        state_x0 = []

        for endmember in models.keys():
            min2use, max2use = [0, 1]
            sc_min = np.quantile(models[endmember]['scores'], [min2use], axis=0).T
            sc_max = np.quantile(models[endmember]['scores'], [max2use], axis=0).T
            x0 = np.ones(models[endmember]['scores'].shape[1]) * 0.0
            state_bounds.extend(np.hstack((sc_min, sc_max)))
            state_x0.extend(x0)

        minbound = np.array([0.0 for a in range(len(models.keys()))])[:, np.newaxis]
        maxbound = np.array([1.0 for a in range(len(models.keys()))])[:, np.newaxis]
        mixing_bounds = np.hstack((minbound, maxbound))
        x0 = np.ones(len(models.keys())) * 0.3
        state_bounds.extend(mixing_bounds)
        state_x0.extend(x0)

        x_bounds = np.array(state_bounds)
        self.x0 = np.array(state_x0)

        self.LS_Params = {
            'method': 'trf',
            'bounds': (x_bounds[:, 0], x_bounds[:, 1]),
            'max_nfev': 15
        }

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        mu = MultiComponentSurface.xa(self, x_surface, geom)
        mu[self.snow_ind:] = self.init[self.snow_ind:]
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        Cov = MultiComponentSurface.Sa(self, x_surface, geom)
        f = (1000 * np.array(self.scale[self.snow_ind:])) ** 2
        Cov[self.snow_ind:, self.snow_ind:] = np.diag(f)
        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a surface state vector
        represented by snow fractional cover and grain size."""

        xopt = least_squares(self.err_obj, self.x0, jac='2-point', **self.LS_Params, args=(rfl_meas, ))

        x = MultiComponentSurface.fit_params(self, rfl_meas, geom)
        rfl_hat_snow = self.pca_rfl_snow.inverse_transform(xopt.x[:self.ind_snow])
        x[self.snow_ind:] = np.array([xopt.x[self.ind_rock], rfl_hat_snow.T[-1]])
        return x

    def err_obj(self, x, y):

        x_snow = x[:self.ind_snow]
        x_veg = x[self.ind_snow:self.ind_veg]
        x_rock = x[self.ind_veg:self.ind_rock]
        x_mix = x[self.ind_rock:]

        snow_hat = self.pca_rfl_snow.inverse_transform(x_snow)
        veg_hat = self.pca_rfl_veg.inverse_transform(x_veg)
        rock_hat = self.pca_rfl_rock.inverse_transform(x_rock)

        rho_hat = snow_hat[:-1] * x_mix[0] + veg_hat * x_mix[1] + rock_hat * x_mix[2]

        resid = rho_hat - y

        constraint = np.abs(1 - sum(x_mix))

        return np.hstack((resid / len(resid), constraint))

    def calc_rfl(self, x_surface, geom):
        """Returns lambertian reflectance for a given state."""

        return self.calc_lamb(x_surface, geom)

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        drfl = self.dlamb_dsurface(x_surface, geom)
        drfl[:, self.snow_ind:] = 1
        return drfl

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return MultiComponentSurface.summarize(self, x_surface, geom) + \
            ' Snow Fractional Cover: %5.3f, Snow Grain Size: %5.3f' % (x_surface[self.snow_ind],
                                                                       x_surface[self.snow_ind + 1])
