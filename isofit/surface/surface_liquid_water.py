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

from .surface_multicomp import MultiComponentSurface
from isofit.configs import Config


class LiquidWaterSurface(MultiComponentSurface):
    """A model of the surface based on a collection of multivariate
       Gaussians, extended with a surface liquid water term."""

    def __init__(self, full_config: Config):

        super().__init__(full_config)

        # TODO: Enforce this attribute in the config, not here (this is hidden)
        self.statevec_names.extend(['Liquid_Water', 'Intercept', 'Slope'])
        self.scale.extend([1.0, 1.0, 1.0])
        self.init.extend([0.02, 0.3, 0.0002])
        self.bounds.extend([[0, 0.5], [0, 1.0], [-0.0004, 0.0004]])

        self.n_state = self.n_state + 3
        self.lw_ind = len(self.statevec_names) - 3

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        mu = MultiComponentSurface.xa(self, x_surface, geom)
        mu[self.lw_ind:] = self.init[self.lw_ind:]
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        Cov = MultiComponentSurface.Sa(self, x_surface, geom)
        f = (1000 * np.array(self.scale[self.lw_ind:])) ** 2
        Cov[self.lw_ind:, self.lw_ind:] = np.diag(f)
        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a surface state vector."""

        x = MultiComponentSurface.fit_params(self, rfl_meas, geom)
        return x[:self.lw_ind]

    def calc_rfl(self, x_surface, geom):
        """Returns lambertian reflectance for a given state."""

        return self.calc_lamb(x_surface, geom)

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
