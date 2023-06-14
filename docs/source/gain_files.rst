Gain Files
==========

This page details how users can access QuartiCal's gain solutions. By default,
QuartiCal places all its gain outputs in a folder called ``gains.qc``. This
folder (or the folder chosen using ``output.gain_directory``) will contain all
the gain solutions. These gain solutions are stored as ``xarray.Datasets``
backed by zarr arrays and include gain flags and coordinate information. 

.. note::
    The remainder of this section assumes you have a working installation of
    `dask-ms <https://dask-ms.readthedocs.io/en/latest/>`_.

Reading Gains
-------------

Under the assumption that QuartiCal has been run using the defaults (solving
for ``G`` with ``output.gain_directory=gains.qc``) the gains can be read using
the following:

.. code:: python

    from daskms.experimental.zarr import xds_from_zarr

    gains = xds_from_zarr("gains.qc::G")

This will produce a list of ``xarray.Dataset`` objects (the length of the
list will depend on how the data was partitioned during the solve). Users are 
then free to manipulate these ``xarray.Datasets`` as normal.

.. warning::

    QuartiCal outputs include some experimental features such as the inclusion
    of ``jhj`` - the product of the conjugate transpose of the Jacobian and the
    Jacobian. Users are discouraged from relying on this information as it is 
    experimental and may change in a future release.