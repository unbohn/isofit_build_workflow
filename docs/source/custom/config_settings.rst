Configuration Settings
==============

ISOFIT is highly configurable. This provides a great deal of design flexibility for analysts to build their own custom
retrieval algorithms for specific investigations. Scientists can construct custom instrument models to handle new
sensors, or define new observation uncertainties to account for model discrepancy errors. They can refine or constrain
the prior distribution based on background knowledge of the state vector. This flexibility can be powerful, but it can
also be daunting for the beginner.

Starting with version 3.0.0, ISOFIT comes with a new macro configuration dictionary that allows the user to set all
required and optional processing options within one single file. Whether relying on default settings or specifying
advanced configurations, the dictionary provides a great deal of both flexibility and simplicity. All other needed
model and configuration settings, such as the surface model and radiative transfer options, are then created on
runtime based on the parameters provided in the macro dictionary. In the following, we will take a detailed look at
this macro config.

The macro config file
---------------------

The macro config file consists of a specific structure of high-level options and more specific processing
configurations. Besides general options and inversion parameters, it also includes surface type specific inversion
settings as well as the design of the surface model. A template for the macro config file, called
"multisurface_oe_template.json", can be found at the main directory of the ISOFIT package. This template comes with all
the default settings that are needed to successfully run ISOFIT. In principle, the user just needs to adjust the sensor
name and a few optional filepaths. Now, we will walk you through the different configuration blocks of the macro config:

.. code-block:: JSON

    "general_options": {
            "sensor": "emit",
            "n_cores": null,
            "segmentation_size": 40,
            "num_neighbors": 15,
            "chunksize": 256,
            "n_pca": 5,
            "copy_input_files": false,
            "presolve_wv": false,
            "empirical_line": false,
            "analytical_line": false,
            "debug_mode": false,
            "ray_temp_dir": "/tmp/ray"
        }

.. list-table:: General options
   :widths: 5 25
   :header-rows: 1

   * - Key
     - Value
   * - sensor
     - Sensor name, will be used to determine noise and datetime settings. Choices are:
       [ang, avcl, emit, prism, neon, hyp].
   * - n_cores
     - Number of cores to run ISOFIT with. Substantial parallelism is available, and full runs will be very slow in
       serial. Suggested to max this out on the available system, which can be done by setting n_cores = null.
       Default: 1.
   * - segmentation_size
     - Size of segments to construct for empirical or analytical line (if used). Default: 40.
   * - num_neighbors
     - Number of neighbors for empirical or analytical line extrapolation. If not given, this number is calculated
       based on the segmentation size. Default: 15.
   * - chunksize
     - Size of image chunks to be processed separately. Default: 256.
   * - n_pca
     - Number of principle components to be used for image segmentation. Default: 5.
   * - copy_input_files
     - If set to true, copy input_radiance, input_loc, and input_obs locally into the working_directory. Default: false.
   * - presolve_wv
     - If set to true, use a presolve mode to estimate the available atmospheric water vapor range. Runs a preliminary
       inversion over the image with a 1-D LUT of water vapor, and uses the resulting range (slightly expanded) to
       bound the full LUT. Advisable to only use with small images/image subsets or in concert with the empirical or
       analytical line setting, otherwise, a significant speed penalty will be incurred. Default: false.
   * - empirical_line
     - If set to true, use an empirical line interpolation to run full inversions over only a subset of pixels,
       determined using a SLIC superpixel segmentation, and use a KDTREE of local solutions to interpolate
       radiance->reflectance. Generally a good option if not trying to analyze the atmospheric state at fine scale
       spatial resolution. Default: false.
   * - analytical_line
     - If set to true, perform an analytical inversion of the conditional MAP estimate for a fixed atmosphere. The
       latter is calculated by retrieving the atmospheric state for each superpixel and then interpolating to per-pixel
       level. Based on the "inner loop" from Susiluoto ate al. (2023). Default: false.
   * - debug_mode
     - If set to true, run ISOFIT with single core processing, which facilitates debugging. Default: false.
   * - ray_temp_dir
     - Location of temporary directory for ray parallelization engine. Default: '/tmp/ray'.

.. code-block:: JSON

    "general_inversion_parameters": {
                "filepaths": {
                    "model_discrepancy_path": null,
                    "aerosol_climatology_path": null,
                    "channelized_uncertainty_path": null,
                    "surface_path": null,
                    "rdn_factors_path": null,
                    "modtran_path": null,
                    "lut_config_path": null,
                    "emulator_base": "/Users/bohn/Desktop/sRTMnet_v100/sRTMnet_v100"
                },
                "options": {
                    "multiple_restarts": false,
                    "multipart_transmittance": false,
                    "topography_model": false,
                    "eps": 0.02,
                    "uncorrelated_radiometric_uncertainty": 0.01,
                    "inversion_windows": [[380.0, 1325.0], [1435, 1770.0], [1965.0, 2500.0]],
                    "statevector_elements": ["H2OSTR", "AOT550", "GNDALT"],
                    "surface_category": "multicomponent_surface"
                },
                "radiative_transfer_parameters": {
                    "spectral_DV": 5,
                    "spectral_FWHM": 5,
                    "spectral_BMNAME": "05_2013",
                    "atmosphere_type": "ATM_MIDLAT_SUMMER",
                    "H2OSTR": {
                        "lut_spacing": 0.25,
                        "lut_spacing_min": 0.03,
                        "default_range": [0.05, 5.0],
                        "min": 0.05
                    },
                    "AOT550": {
                        "lut_spacing": 0,
                        "lut_spacing_min": 0,
                        "default_range": [0.001, 1]
                    },
                    "GNDALT": {
                        "lut_spacing": 0.25,
                        "lut_spacing_min": 0.2,
                        "expand_range": 2
                    }
                }
            }

.. list-table:: General inversion parameters
   :widths: 5 25
   :header-rows: 1

   * - Key
     - Value
   * - model_discrepancy_path
     - Specify wavelength-dependent forward model discrepancy, if desired. Default: null.
   * - aerosol_climatology_path
     - Specific aerosol climatology information to use in MODTRAN, if desired. Default: null.
   * - channelized_uncertainty_path
     - Specify channelized radiometric instrument uncertainty, if desired. Default: null.
   * - surface_path
     - Specify costume, pre-built surface model, if desired. If not given, ISOFIT builds the surface model on runtime
       using the settings of the 'surface' block in the macro config. Default: null.
   * - rdn_factors_path
     - Specify wavelength-dependent radiometric correction factors, if desired. Default: null.
   * - modtran_path
     - Specify location of MODTRAN software. If not given, ISOFIT uses the MODTRAN_DIR environment variable to locate
       the executable file. Default: null.
   * - lut_config_path
     - Specify a look up table configuration file, which will override defaults chocies that are set up on runtime.
       Default: null.
   * - emulator_base
     - Specify location of emulator base path. Point this at the model folder (or h5 file) of sRTMnet to use the
       emulator instead of MODTRAN (i.e., your_path/sRTMnet_v100/sRTMnet_v100, see macro config template). If not given,
       ISOFIT tries to use MODTRAN for radiative transfer simulations.
   * - multiple_restarts
     - If set to true, use multiple initializations for calculation of atmospheric state first guess. Default: false.
   * - multipart_transmittance
     - If set to true, ISOFIT runs MODTRAN with 3 different surface reflectance levels in order to separate down- and
       upward transmittance into direct and diffuse parts. Default: false.
   * - topography_model
     - If set to true, apply the topoflux model that accounts for surface slope and aspect by separately scaling direct
       and diffuse downwelling transmittance (Carmon et al. 2022). Only applicable when multipart_transmittance is set
       to true. Default: false.
   * - eps
     - Delta value for perturbing state vector elements for calculating Jacobian. Default: 0.02.
   * - uncorrelated_radiometric_uncertainty
     - Uncorrelated radiometric uncertainty to be added to Rodgers' model error formalism. Default: 0.01.
   * - inversion_windows
     - Spectral ranges to be included in the inversion. Less weight, i.e., higher uncertainties will be put on
       wavelengths outside inversion windows.
   * - statevector_elements
     - Elements of the atmospheric state vector. It is recommended to have at least water vapor and aod as free
       parameters. Default: ["H2OSTR", "AOT550", "GNDALT"].
   * - surface_category
     - Define the surface model to be used. Possible choices are ["multicomponent_surface", "glint_surface",
       "thermal_surface"]. Default: "multicomponent_surface".
   * - spectral_DV
     - Increment or step size at which output is generated from MODTRAN.
   * - spectral_FWHM
     - Resolution of the slit function for MODTRAN simulations.
   * - spectral_BMNAME
     - MODTRAN band model file name.
   * - atmosphere_type
     - MODTRAN atmosphere model holding altitude profiles of pressure, temperature, and molecular species.
   * - lut_spacing
     - Spacing of atmospheric LUT grid points.
   * - lut_spacing_min
     - Minimum spacing of atmospheric LUT grid points.
   * - default_range
     - Lower and upper bounds of atmospheric LUT grid points.
   * - min
     - Minimum value of atmospheric LUT grid points.
   * - expand_range
     - Value in units of km, by which range of DEM-derived pressure altitude grid points in atmospheric LUT should be
       extended in both directions.

We recommend instrument models based on a three-channel parametric noise description. These models predict
noise-equivalent change in radiance as a function of :math:`L`, the radiance at sensor, with the relation
:math:`L_{noisy} = a\sqrt{b+L}+c`. They are stored as five-column ASCII text files with columns representing:
wavelength; the a, b, and c coefficients; and the Root Mean Squared approximation error for the coefficient fitting,
respectively. An example is provided in the data/avirisng_noise.txt file. We also recommend channelized uncertainty
files representing the standard deviation of residuals due to forward model or wavelength calibration and response
errors. Finally, we recommend a 0-1% uncorrelated radiometric uncertainty term, depending on the confidence in the
radiometric calibration of the instrument. Certain extreme cases may require higher values.

We highly recommend the MODTRAN 6.0 radiative transfer model over LibRadTran and 6SV options for full-spectrum
(380-2500) imaging spectroscopy. We recommend retrieving water vapor and aerosol optical depth in the VSWIR range,
water vapor and ozone in the thermal IR. For aerosol optical properties, we recommend the third aerosol type found the
aerosol file data/aerosol_model.txt. This can be selected by including the "AERFRAC_2" element in the state vector and
lookup tables.

Note that all atmospheric parameters have extremely wide and uninformed prior distributions. More advanced users, or
those with very heterogeneous flightlines, may wish to track the unique viewing geometry of every pixel in the image.
It is important to pass in an OBS-format metadata file in the input block, so that the program knows the geometry
associated with each pixel.

We recommend excluding deep water features at 1440 nm and 1880 nm from the inversion windows. We recommend a
multiple-start inversion with four gridpoints at low and high values of atmospheric aerosol and water vapor.

.. code-block:: JSON

    "type_specific_inversion_parameters": {
                "cloud": {
                    "toa_threshold_wavelengths": [450,1250,1650],
                    "toa_threshold_values": [0.31, 0.51, 0.22],
                    "toa_threshold_comparisons": ["gt","gt","gt"],
                    "statevector_elements": ["GNDALT"],
                    "GNDALT": {
                        "lut_spacing": 0.25,
                        "lut_spacing_min": 0.2,
                        "expand_range": 2
                    }
                },
                "water": {
                    "toa_threshold_wavelengths": [1000, 1380],
                    "toa_threshold_values": [0.05, 0.1],
                    "toa_threshold_comparisons": ["lt"],
                    "surface_category":  "glint_surface"
                }
            }

.. list-table:: Type specific inversion parameters
   :widths: 5 25
   :header-rows: 1

   * - Key
     - Value
   * - toa_threshold_wavelengths
     - Threshold wavelengths to be used for surface type classification.
   * - toa_threshold_values
     - Threshold values in units of top-of-atmosphere reflectance to be used for surface type classification.
   * - toa_threshold_comparisons
     - Comparisons of threshold values to be used for surface type classification. "gt": greater than, "lt": lower than.

.. code-block:: JSON

    "surface": {
            "output_model_file": null,
            "wavelength_file": null,
            "normalize": "Euclidean",
            "reference_windows": [[400, 1300], [1450, 1700], [2100, 2450]],
            "sources":
                [
                    {
                        "input_spectrum_files":
                            [
                                "surface_model_ucsb"
                            ],
                        "n_components": 8,
                        "windows": [
                            {
                                "interval": [300, 400],
                                "regularizer": 1e-4,
                                "correlation": "EM"
                            },
                            {
                                "interval": [400, 1300],
                                "regularizer": 1e-6,
                                "correlation": "EM"
                            },
                            {
                                "interval": [1300, 1450],
                                "regularizer": 1e-4,
                                "correlation": "EM"
                            },
                            {
                                "interval": [1450, 1700],
                                "regularizer": 1e-6,
                                "correlation": "EM"
                            },
                            {
                                "interval": [1700, 2100],
                                "regularizer": 1e-4,
                                "correlation": "EM"
                            },
                            {
                                "interval": [2100, 2450],
                                "regularizer": 1e-6,
                                "correlation": "EM"
                            },
                            {
                                "interval": [2450, 2550],
                                "regularizer": 1e-4,
                                "correlation": "EM"
                            }
                                    ]
                    },
                    {
                        "input_spectrum_files":
                            [
                                "ocean_spectra_rev2"
                            ],
                        "n_components": 8,
                        "windows": [
                            {
                                "interval": [300, 400],
                                "regularizer": 1e-4,
                                "correlation": "decorrelated"
                            },
                            {
                                "interval": [400, 1300],
                                "regularizer": 1e-6,
                                "correlation": "EM"
                            },
                            {
                                "interval": [1300, 1450],
                                "regularizer": 1e-4,
                                "correlation": "decorrelated"
                            },
                            {
                                "interval": [1450, 1700],
                                "regularizer": 1e-6,
                                "correlation": "decorrelated"
                            },
                            {
                                "interval": [1700, 2100],
                                "regularizer": 1e-4,
                                "correlation": "decorrelated"
                            },
                            {
                                "interval": [2100, 2450],
                                "regularizer": 1e-6,
                                "correlation": "decorrelated"
                            },
                            {
                                "interval": [2450, 2550],
                                "regularizer": 1e-4,
                                "correlation": "decorrelated"
                            }
                                    ]
                    }
                ]
        }

.. list-table:: Surface options
   :widths: 5 25
   :header-rows: 1

   * - Key
     - Value
   * - output_model_file
     - Output directory of surface .mat file holding multivariate Gaussian distributions of reflectance spectra to be
       used as priors. It is recommended to leave it blank as it is auto-generated during runtime. Default: null.
   * - wavelength_file
     - Instrument wavelengths file. It is recommended to leave it blank as wavelengths are auto-obtained from input
       image data cube. Default: null.
   * - normalize
     - Normalization metric for multivariate Gaussian distribution of surface reflectance. Default: "Euclidean".
   * - reference_windows
     - Spectral ranges to be included in the inversion. Less weight, i.e., higher uncertainties will be put on
       wavelengths outside reference windows.
   * - input_spectrum_files
     - Input file containing spectral library of either lab- or field-measured surface reflectance.
   * - n_components
     - Number of Gaussian distributions to be built for surface priors.
   * - interval
     - Wavelength interval for a specific regularizer, i.e., weight in the inversion.
   * - regularizer
     - Regularizer, i.e., weight in the inversion for given wavelength interval.
   * - correlation
     - Correlation between wavelength channels within given interval. "EM" and "EM-gauss" give a full covariance matrix.
       "decorrelated" gives a diagonal matrix.

The multicomponent surface model is most universal and forgiving. We recommend constructing Gaussian PDFs from diverse
libraries of terrestrial and aquatic spectra, with correlations only in the key water absorption features at 940 and
1140 nm. Use reference wavelengths for normalization and distance calculations that exclude the deep water absorption
features at 1440 and 1880 nm. Note that the surface model is normalized with the Euclidean norm.
