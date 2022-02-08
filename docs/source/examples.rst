Examples
========

This page provides some example configuration for common calibration tasks. To
use one of the following configurations, simply copy the text into an empty
file and save as a .yaml.

.. note::

    The configuration provided in this section will work, but may not include
    every available option. For a full list, please see :ref:`Options`.

Phase-only selfcal
------------------

The following performs diagonal, phase-only selfcal using a measurement set
column and produces corrected residuals and corrected weights.

.. note::
    QuartiCal expects users to ask for all required outputs explicitly. This
    is to ensure that users understand precisely what is ultimately being done
    to their data.

.. code-block:: yaml

    input_ms:
        path: path/to.ms
        data_column: DATA
        sigma_column: SIGMA_SPECTRUM
        time_chunk: '300s'
        freq_chunk: '0'
    input_model:
        recipe: MODEL_DATA
    solver:
        terms:
            - G
        iter_recipe:
            - 25
    output:
        directory: outputs.qc
        overwrite: false
        products:
            - corrected_residual
            - corrected_weight
        columns:
            - CORRECTED_DATA
            - WEIGHT_SPECTRUM
    dask:
        threads: 6
    G:
        type: phase
        time_interval: '60s'
        freq_interval: '128'

Complex (amplitude and phase) selfcal
-------------------------------------

The following performs diagonal, phase and amplitude selfcal using a Tigger
.lsm sky model and produces corrected data.

.. code-block:: yaml

    input_ms:
        path: path/to.ms
        data_column: DATA
        sigma_column: SIGMA_SPECTRUM
        time_chunk: '300s'
        freq_chunk: '0'
    input_model:
        recipe: skymodel.lsm.html
    solver:
        terms:
            - G
        iter_recipe:
            - 25
    output:
        directory: outputs.qc
        overwrite: false
        products:
            - corrected_data
            - corrected_weight
        columns:
            - CORRECTED_DATA
            - WEIGHT_SPECTRUM
    dask:
        threads: 6
    G:
        type: complex
        time_interval: '60s'
        freq_interval: '128'

Gain and bandpass selfcal
-------------------------

The following performs gain and bandpass calibration simultaneously,
using a measurement set column as input and produces uncorrected residuals.

.. code-block:: yaml

    input_ms:
        path: path/to.ms
        data_column: DATA
        sigma_column: SIGMA_SPECTRUM
        time_chunk: '300s'
        freq_chunk: '0'
    input_model:
        recipe: MODEL_DATA
    solver:
        terms:
            - G
            - B
        iter_recipe:
            - 25
            - 25
            - 10
            - 10
    output:
        directory: outputs.qc
        overwrite: false
        products:
            - residual
        columns:
            - CORRECTED_DATA
    dask:
        threads: 6
    G:
        type: complex
        time_interval: '1'
        freq_interval: '0'
    B:
        type: complex
        time_interval: '0'
        freq_interval: '1'

Direction-independent and direction-dependent complex selfcal
-------------------------------------------------------------

The following performs direction-independent and direction-dependent gain
calibration simultaneously, using a tagged sky model as input and produces
(direction-independent) corrected residuals.

.. note::
    Direction-dependent model specification in QuartiCal (via
    ``input_model.recipe``) is flexible, allowing the use of both sky models
    and measurement set columns in fairly complex configurations. Here are
    some examples:

    * :code:`COL_NAME1:COL_NAME2`
      This will create a model with two directions, one for each of the
      supplied measurement set columns.
    * :code:`skymodel.lsm.html~COL_NAME:COL_NAME`
      This will create a model with two directions, one containing the
      visibilities associated with the sky model minus the contribution of
      the MS column and the other containing just the MS column.
    * :code:`skymodel.lsm.html:COL_NAME1:COL_NAME2`
      This will create a model with three directions, one containing the
      visibilities associated with the sky model, the second containing the
      visibilities from the first MS column and the third containing the
      visibilities of the second MS column.
    * :code:`COL_NAME1+COL_NAME2:skymodel.lsm.html@dE`
      This will create a model with at least two directions. This first will
      contain the sum of the specified MS columns and the remaining will be
      generated from the dE tagged sources in the sky model.

    The following example makes use of a tagged Tigger .lsm file to predict
    visibilities in several directions.

.. code-block:: yaml

    input_ms:
        path: path/to.ms
        data_column: DATA
        sigma_column: SIGMA_SPECTRUM
        time_chunk: '300s'
        freq_chunk: '0'
    input_model:
        recipe: skymodel.lsm.html@dE
    solver:
        terms:
            - G
            - dE
        iter_recipe:
            - 25
            - 25
            - 10
            - 10
    output:
        directory: outputs.qc
        overwrite: false
        products:
            - corrected_residual
        columns:
            - CORRECTED_DATA
    dask:
        threads: 6
    G:
        type: complex
        time_interval: '10'
        freq_interval: '10'
    dE:
        type: complex
        time_interval: '100'
        freq_interval: '100'