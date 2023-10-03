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
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
#          Niklas Bohn, urs.n.bohn@jpl.nasa.gov
#          James Montgomery, j.montgomery@jpl.nasa.gov
#

import pytest

import isofit
from isofit.configs import configs
from isofit.radiative_transfer.modtran import ModtranRT


@pytest.fixture(scope="session")
def pasadena_files():
    """Get files from the Pasadena example."""
    config_file = (
        f"{isofit.root}/../examples/20171108_Pasadena/configs/ang20171108t184227_beckmanlawn-multimodtran"
        f"-topoflux.json"
    )
    wavelength_file = f"{isofit.root}/../examples/20171108_Pasadena/remote/20170320_ang20170228_wavelength_fit.txt"

    files = (config_file, wavelength_file)

    return files


@pytest.mark.parametrize("files", ["pasadena_files"])
def test_combined(files, request):
    """Tests multiple class inits and class functions in a sequence to ensure recursive
    compatibility.

    Parameters
    ----------
    request: pytest.fixture
        Built-in fixture to resolve fixtures by name
    """
    print("Load config file from the Pasadena example.")
    config_file = request.getfixturevalue(files)
    config = configs.create_new_config(config_file[0])
    config.get_config_errors()
    engine_config = config.forward_model.radiative_transfer.radiative_transfer_engines[
        0
    ]

    print("Load wavelength file from the Pasadena example.")
    wavelength_file = config_file[1]

    print("Initialize radiative transfer engine without prebuilt LUT file.")
    lut_grid = config.forward_model.radiative_transfer.lut_grid
    rt_engine = ModtranRT(
        engine_config=engine_config,
        interpolator_style="mlg",
        instrument_wavelength_file=wavelength_file,
        lut_grid=lut_grid,
    )

    # First, we don't provide a prebuilt LUT and run simulations based on a given LUT grid
    print("Run radiative transfer simulations.")
    rt_engine.run_simulations()
    del rt_engine

    # Second, we use the just built LUT file and initialize the engine class again
    print("Initialize radiative transfer engine with prebuilt LUT file.")
    ModtranRT(
        engine_config=engine_config,
        interpolator_style="mlg",
        instrument_wavelength_file=wavelength_file,
    )

    print("Done!")
