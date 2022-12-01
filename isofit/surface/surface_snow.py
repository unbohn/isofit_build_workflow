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
from scipy.optimize import least_squares

from .surface_multicomp import MultiComponentSurface
from isofit.core.common import eps, VectorInterpolator
from isofit.configs import Config


class SnowSurface(MultiComponentSurface):
    """A snow radiative transfer model that combines optical properties based on Mie theory and multiscattering
    calculations using the multistream DIScrete Ordinate Radiative Transfer (DISORT) code. The surface state vector
    holds snow grain radius, liquid water fraction as well as mass mixing ratios of various light-absorbing particles
    (LAP), including snow algae, black carbon, and mineral dust."""

    def __init__(self, full_config: Config):

        super().__init__(full_config)

        self.statevec_names = (['Grain_size', 'Liquid_water', 'Algae', 'Black_carbon', 'Mineral_dust'])
        self.scale = ([1.0, 1.0, 1.0, 1.0, 1.0])
        self.init = ([500.0, 5.0, 50000.0, 50.0, 50000.0])
        self.bounds = np.array([[0.0, 1500.0], [0.0, 25.0], [0.0, 400000.0], [0.0, 1000.0], [0.0, 400000.0]])

        self.n_state = len(self.statevec_names)

        # Load DISORT LUT
        # __file__ should live at isofit/isofit/surface/
        isofit_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path_lut = os.path.join(isofit_path, "data", "iop", "DISORT_LUT_regular_grid.npz")

        with np.load(path_lut) as data:
            state = data['x']
            hdrf = data['y']

        grid = []
        for ii in range(state.shape[1]):
            grid.append(list(np.unique(state[:, ii])))

        n = int(np.exp(np.log(state.shape[0]) / state.shape[1]))
        data = hdrf.reshape(n, n, n, n, n, n, n, n, n, hdrf.shape[1])

        lut = ["r", "r", "r", "r", "r", "r", "r", "r", "r"]
        self.VecInt = VectorInterpolator(grid_input=grid, data_input=data, lut_interp_types=lut, version="mlg")

        self.LS_Params = {
            'method': 'trf',
            'bounds': (self.bounds[:, 0], self.bounds[:, 1]),
            'max_nfev': 15
        }

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        mu = self.init
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        f = (1000 * np.array(self.scale)) ** 2
        Cov = np.diag(f)
        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a surface state vector
        represented by snow fractional cover and grain size."""

        xopt = least_squares(self.err_obj, self.init, jac='2-point', **self.LS_Params,
                             args=(rfl_meas, geom))

        return xopt.x

    def err_obj(self, x, y, geom):

        x_hat = np.array([geom.solar_zenith, geom.solar_azimuth, geom.observer_zenith, geom.observer_azimuth,
                          x[0], x[1], x[2], x[3], x[4]])

        rho_hat = self.VecInt(x_hat)
        resid = rho_hat - y

        return resid

    def calc_rfl(self, x_surface, geom):
        """Returns lambertian reflectance for a given state."""

        x_hat = np.array([geom.solar_zenith, geom.solar_azimuth, geom.observer_zenith, geom.observer_azimuth,
                          x_surface[0], x_surface[1], x_surface[2], x_surface[3], x_surface[4]])

        rho_hat = self.VecInt(x_hat)

        return rho_hat

    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""

        return self.calc_rfl(x_surface, geom)

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        # first the reflectance at the current state vector
        rho_hat = self.calc_rfl(x_surface, geom)

        # perturb each element of the RT state vector (finite difference)
        drfl_dsurface = []
        x_surfaces_perturb = x_surface + np.eye(len(x_surface)) * eps

        for x_surface_perturb in list(x_surfaces_perturb):
            rho_hat_perturb = self.calc_rfl(x_surface_perturb, geom)
            drfl_dsurface.append((rho_hat_perturb - rho_hat) / eps)

        drfl_dsurface = np.array(drfl_dsurface).T

        return drfl_dsurface

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface."""

        dLs = np.zeros((self.n_wl, self.n_state), dtype=float)

        return dLs

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return 'Grain Size: %5.3f, Liquid Water: %5.3f, Algae: %5.3f, Black Carbon: %5.3f,' \
               'Mineral Dust: %5.3f' % (x_surface[0], x_surface[1], x_surface[2], x_surface[3], x_surface[4])
