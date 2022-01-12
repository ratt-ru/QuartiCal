.. _`Options`:

Options
=======

This page details available options. However, invoking ``goquartical`` or
``goquartical help`` should be preferred as it will always be up-to-date. The
following is broken up by configuration section.

.. note::

    These options can be specified either via command line or .yaml file e.g.:

    .. code-block:: bash

        input_ms.path=path/to.ms

    or

    .. code-block:: yaml

        input_ms:
            path: path/to.ms



input_ms
--------

Options pertaining to the input measurement set.

.. jinja:: input_ms

    {% for k, v in input_ms.items() %}

    * {{"{}".format("input_ms." + k)}}:
        {{v}}
    {% endfor %}


input_model
-----------

Options pertaining to the input model.

.. jinja:: input_model

    {% for k, v in input_model.items() %}

    * {{"{}".format("input_model." + k)}}:
        {{v}}
    {% endfor %}


output
------

Options pertaining to output.

.. jinja:: output

    {% for k, v in output.items() %}

    * {{"{}".format("output." + k)}}:
        {{v}}
    {% endfor %}


mad_flags
---------

Options pertaining to MAD (Median Absolute Deviation) flagging.

.. jinja:: mad_flags

    {% for k, v in mad_flags.items() %}

    * {{"{}".format("mad_flags." + k)}}:
        {{v}}
    {% endfor %}


dask
----

Options pertaining to Dask (and therefore parallelism).

.. jinja:: dask

    {% for k, v in dask.items() %}

    * {{"{}".format("dask." + k)}}:
        {{v}}
    {% endfor %}


solver
------

Options pertaining to all solvers (as opposed to specific terms).

.. jinja:: solver

    {% for k, v in solver.items() %}

    * {{"{}".format("solver." + k)}}:
        {{v}}
    {% endfor %}


gain
----

Options pertaining to a specific gain/Jones term.

.. warning::
    This help is generic - users will not typically write ``gain.option`` but
    will instead use the labels specified by ``solver.gain_terms``. Thus, for
    ``solver.gain_terms="[G,B]"``, options would be specified using
    ``G.option`` or ``B.option``.

.. jinja:: gain

    {% for k, v in gain.items() %}

    * {{"{}".format("gain." + k)}}:
        {{v}}
    {% endfor %}