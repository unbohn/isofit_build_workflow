Tutorials
=========

We have developed a few application tutorials that include both different instruments and surface types, and are likely
to produce good results over a wide range of conditions. All needed input data are stored on google drive and can be
downloaded and prepared by using the ``utils/data_handling.py`` module.

EMIT Snow
---------

This image was acquired by the spaceborne EMIT imaging spectrometer over the Patagonian Ice Sheet in South America on
September 12, 2022. It provides a multitude of different snow and ice surface types, ranging from dry, clean snow over
blue ice to melting snow and ice contaminated by high loadings of light-absorbing particles (e.g., mineral dust and
algae).

First, we apply the ``utils/data_handling.py`` module to download the image cube and prepare the input files to ISOFIT.

.. code-block:: python



.. code-block:: python

    input_radiance = "../../Snow_Spectroscopy/Himalayas_Andes/data/EMIT/emit_L1B/emit20220912t154138_o25510_s001_l1b_rdn_b0106_v01.img"
    input_obs = "../../Snow_Spectroscopy/Himalayas_Andes/data/EMIT/emit_L1B/emit20220912t154138_o25510_s001_l1b_obs_b0106_v01.img"
    input_loc = "../../Snow_Spectroscopy/Himalayas_Andes/data/EMIT/emit_L1B/emit20220912t154138_o25510_s001_l1b_loc_b0106_v01.img"
    out_dir = "../../Snow_Spectroscopy/Himalayas_Andes/Andes/emit20220912t154138_o25510_s001_l1b_loc_b0106_v01_results/L2A/"
    config = "../../JPL_ISOFIT_Project/isofit_build_workflow/multisurface_oe_template.json"

.. code-block:: python

    multisurface_oe.main(
        [
            input_radiance,
            input_loc,
            input_obs,
            out_dir,
            config,
            "--wavelength_path", "../../Snow_Spectroscopy/Himalayas_Andes/data/EMIT/emit_wavelength_fit.txt"
        ]
    )
