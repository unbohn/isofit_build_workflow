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
import pandas as pd

from .surface_multicomp import MultiComponentSurface
from isofit.configs import Config
from isofit.core.common import table_to_array


class LiquidWaterSurface(MultiComponentSurface):
    """A model of the surface based on a collection of multivariate
       Gaussians, extended with a surface liquid water term."""

    def __init__(self, full_config: Config):

        super().__init__(full_config)

        # TODO: Enforce this attribute in the config, not here (this is hidden)
        self.statevec_names.extend(['Liquid_Water'])
        self.scale.extend([1.0])
        self.init.extend([0.02])
        self.bounds.extend([[0, 0.5]])

        self.n_state = self.n_state + 1
        self.lw_ind = len(self.statevec_names) - 1

        # load imaginary part of liquid water refractive index and calculate wavelength dependent absorption coefficient
        # __file__ should live at isofit/isofit/inversion/
        isofit_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path_k = os.path.join(isofit_path, "data", "iop", "k_liquid_water_ice.xlsx")

        k_wi = pd.read_excel(io=path_k, sheet_name='Sheet1', engine='openpyxl')
        wl_water, k_water = table_to_array(k_wi=k_wi, a=0, b=982, col_wvl="wvl_6", col_k="T = 20°C")
        kw = np.interp(x=self.wl, xp=wl_water, fp=k_water)

        self.abs_co_w = 4 * np.pi * kw / self.wl

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        mu = MultiComponentSurface.xa(self, x_surface, geom)
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        Cov = MultiComponentSurface.Sa(self, x_surface, geom)
        f = (1000 * np.array(self.scale[self.lw_ind])) ** 2
        Cov[self.lw_ind, self.lw_ind] = f

        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a surface state vector."""

        x = MultiComponentSurface.fit_params(self, rfl_meas, geom)
        return x

    def calc_rfl(self, x_surface, geom):
        """Returns lambertian reflectance for a given state."""

        rfl = self.calc_lamb(x_surface[:self.lw_ind], geom) * np.exp(-x_surface[self.lw_ind] * self.abs_co_w)

        return rfl

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        drfl = self.dlamb_dsurface(x_surface, geom)
        drfl[:, self.lw_ind:] = 1
        return drfl

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return MultiComponentSurface.summarize(self, x_surface, geom) + \
            ' Liquid Water: %5.3f' % x_surface[self.lw_ind]
