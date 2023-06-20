Parallelism (Chunking)
======================

Getting the most out of an application like QuartiCal can be tough - there are
many configurable parameters and it isn't always obvious how they relate to
performance in terms of either speed or memory consumption. This section of
the documentation aims to aid users in understanding the relevant settings.

``input_ms.group_by``
~~~~~~~~~~~~~~~~~~~~~

First and foremost, QuartiCal partitions large calibration problems into
several smaller ones in accordance with ``input_ms.group_by``. By default,
this includes:

- ``FIELD_ID``
- ``DATA_DESC_ID``
- ``SCAN_NUMBER``

The reason for this is that gain solutions may not be continuous over changes
in these quantities. This also establishes a coarse grid over the data which
is useful when distributing the problem. 

From a user's perspective, these parameters will not typically have a large
effect on either run-time or memory footprint. They are only worth mentioning
because they are fundamental to the way QuartiCal handles data.

.. warning::

    ``SCAN_NUMBER`` can be removed from the above, but removing ``FIELD_ID``
    or ``DATA_DESC_ID`` will result in incorrect behaviour.

``input_ms.time_chunk`` and ``input_ms.freq_chunk``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These two parameters control how much data QuartiCal will calibrate per
process/thread and essentially describe the size of QuartiCal's fundamental
unit of work. Consequently, these parameters determine how QuartiCal's 
memory footprint will scale as parallelism increases (i.e. more
threads/processes are in use). There are no restrictions on the chunk size
although the chunk size will set the maximum solution interval as gains cannot
be solved over chunk boundaries. This can be problematic for gains which
require a large amount of data in memory simultaneously (e.g. the bandpass)
but QuartiCal does have mechanisms to allow for large chunk sizes when 
needed.

``input_model.source_chunks``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This parameter controls how many sources are predicted simultaneously when
the DFT is in use (by specifying a Tigger sky model). This is a less used 
feature but it does impact memory footprint.

``dask.threads`` and ``dask.workers``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These parameters work in concert to control the horizontal scaling in
QuartiCal. Horizontal scaling here refers to the high level parallelism
implemented using Dask and is synonymous with the number of chunks being
processed simultaneously. The total memory footprint should be proportional
to the product of ``dask.threads``, ``dask.workers`` and the size of each
chunk.

In general ``dask.threads`` should be preferred over ``dask.workers`` as the
latter involves processes which are less lightweight and have additional
overheads. On a single node, set ``dask.threads`` to some fraction of your
total number of available cores, based on your available memory.

``solver.threads``
~~~~~~~~~~~~~~~~~~

As previously mentioned, QuartiCal has mechanisms that enable it to deal with
very large chunks whilst still utilising as many CPU cores as possible. This
vertical scaling is controlled via ``solver.threads``, where vertical scaling
refers to the low-level parallelism inside QuartiCal's solver code. This form 
of parallelism has negligible impact on memory footprint and can be used to
exercise hardware even when chunks are very large. Note that each Dask thread
will be associated with a number of solver threads set by this parameter i.e.
the total number of threads in use will be the product of ``dask.workers``,
``dask.threads`` and ``solver.threads``.
