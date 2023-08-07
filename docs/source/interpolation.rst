Interpolation
=============

QuartiCal is capable of interpolating any gain term which it can solve for.
This functionality is readily available and is triggered automatically when
``gain.load_from`` is set, e.g. ``G.load_from=existing/gains.qc/G``. QuartiCal
will interpolate the on-disk gains to be compatible with the parameters set
on the term which is being loaded/interpolated. 

.. note::
    Gain types cannot change during interpolation i.e. if the on-disk gain is
    of type ``diag_complex``, the term must be loaded as type ``diag_complex``.
    This functionality may be simplified in future.

Example - Interpolate
~~~~~~~~~~~~~~~~~~~~~

This example will attempt to make the interpolation behaviour a little more
intuitive. Let us assume that we have solved for the bandpass on a 
calibrator (``FIELD_ID==0``) using the following basic config:

.. code-block:: yaml

    input_ms:
        path: path/to.ms
        data_column: DATA
        time_chunk: '0'
        freq_chunk: '0'
        select_fields:
            - 0
    input_model:
        recipe: MODEL_DATA
    solver:
        terms:
            - B
        iter_recipe:
            - 25
    output:
        directory: gains.qc
    B:
        type: diag_complex
        time_interval: '0'
        freq_interval: '1'

This will write the bandpass solutions (``B``) to ``gains.qc/B``. We want to
transfer these solutions to the target (``FIELD_ID==1``). That can be
accomplished using the following config:

.. code-block:: yaml

    input_ms:
        path: path/to.ms
        data_column: DATA
        time_chunk: '0'
        freq_chunk: '0'
        select_fields:
            - 1
    solver:
        terms:
            - B
        iter_recipe:
            - 0
    output:
        products:
            - corrected_data
        columns:
            - CORRECTED_DATA
        gain_directory: interp_gains.qc
    B:
        type: diag_complex
        time_interval: '0'
        freq_interval: '1'
        load_from: path/to/gains.qc/B

The above config will read the solutions from ``path/to/gains.qc/B`` and apply
them to the target. Some important points to note:

    * It is not possible to read and write gains to the same place, hence
      specifying a different ``output.gain_directory``.
    * We specified zero iterations for ``B`` in ``solver.iter_recipe`` - this
      means so additional solver iterations will be performed.
    * We utilised the same solution intervals and chunking as in the original
      bandpass solution - this is not required as gains will be interpolated
      to higher or lower resolution as required.  
    * It is possible to load more than one term at the same time.

Example - Interpolate and Solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us assume the same inital solve for the bandpass as above, but instead of
simply interpolating and applying the solutions, we also want to refine the
solutions on the target field (selfcal). That can be accomplished with the 
following config:

.. code-block:: yaml

    input_ms:
        path: path/to.ms
        data_column: DATA
        time_chunk: '0'
        freq_chunk: '0'
        select_fields:
            - 1
    input_model:
        recipe: MODEL_DATA
    solver:
        terms:
            - B
        iter_recipe:
            - 25
    output:
        products:
            - corrected_data
        columns:
            - CORRECTED_DATA
        gain_directory: interp_gains.qc
    B:
        type: diag_complex
        time_interval: '0'
        freq_interval: '1'
        load_from: path/to/gains.qc/B

Some more things to note:

    * It is perfectly possible to specify both load-only and solvable terms
      in the same config by setting ``solver.iter_recipe`` appropriately.
    * Loaded terms can be combined with new solvable terms. This enables
      on-the-fly transfer and selfcal.