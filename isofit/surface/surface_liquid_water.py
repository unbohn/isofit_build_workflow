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
import pandas as pd

from ..core.common import table_to_array
from .surface_multicomp import MultiComponentSurface
from isofit.configs import Config


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
        self.liquid_water_ind = len(self.statevec_names) - 1

        self.liquid_water_feature_left = np.argmin(abs(1050 - self.wl))
        self.liquid_water_feature_right = np.argmin(abs(1250 - self.wl))

        self.wl_sel = self.wl[self.liquid_water_feature_left:self.liquid_water_feature_right + 1]

        self.path_k = "../../data/iop/k_liquid_water_ice.xlsx"
        self.k_wi = pd.read_excel(io=self.path_k, engine='openpyxl')
        self.wvl_water, self.k_water = table_to_array(k_wi=self.k_wi, a=0, b=982, col_wvl="wvl_6", col_k="T = 20°C")
        self.kw = np.interp(x=self.wl_sel, xp=self.wvl_water, fp=self.k_water)
        self.abs_co_w = 4 * np.pi * self.kw / self.wl_sel

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        mu = MultiComponentSurface.xa(self, x_surface, geom)
        mu[self.liquid_water_ind] = self.init[self.liquid_water_ind]
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        Cov = MultiComponentSurface.Sa(self, x_surface, geom)
        f = np.array([[(1e3 * self.scale[self.liquid_water_ind])**2]])
        Cov[self.liquid_water_ind, self.liquid_water_ind] = f
        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector including surface liquid water."""

        rfl_meas_sel = rfl_meas[self.liquid_water_feature_left:self.liquid_water_feature_right+1]

        intercept = rfl_meas_sel[-1] - ((rfl_meas_sel[0] - rfl_meas_sel[-1]) /
                                        (self.wl_sel[0] - self.wl_sel[-1])) * self.wl_sel[-1]
        slope = (rfl_meas_sel[0] - rfl_meas_sel[-1]) / (self.wl_sel[0] - self.wl_sel[-1])

        attenuation = rfl_meas_sel / (intercept + slope * self.wl_sel)
        dw = -(np.log(attenuation) / 1e7 / self.abs_co_w)

        x = MultiComponentSurface.fit_params(self, rfl_meas, geom)
        x[self.liquid_water_ind] = dw
        return x

    def calc_rfl(self, x_surface, geom):
        """Reflectance (includes surface liquid water path length)."""

        return self.calc_lamb(x_surface, geom) + x_surface[self.liquid_water_ind]

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        drfl = self.dlamb_dsurface(x_surface, geom)
        drfl[:, self.liquid_water_ind] = 1
        return drfl

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return MultiComponentSurface.summarize(self, x_surface, geom) + \
            ' Liquid Water: %5.3f' % x_surface[self.liquid_water_ind]
