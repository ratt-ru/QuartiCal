*****
Usage
*****

This page details running QuartiCal for the first time, including navigating
the help and specifying input arguments.

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

Once QuartiCal has been successfully installed, it can be run from the command
line using:

.. code:: bash

	goquartical

This will produce detailed help on all the input parameters. The help can also
be displayed using:

.. code:: bash

	goquartical help

We appreciate that this can be overwhelming and difficult to navigate. Thus,
users can ask for help on specific sections using the following:

.. code:: bash

	goquartical help=[section1,section2]

The options for this are listed in green at the bottom of the full help.

QuartiCal can be run directly from the command line by specifying the
relevant inputs, e.g.:

.. code:: bash

	goquartical input_ms.path="/path/to.ms" input_model.recipe="MODEL_DATA" input_ms.sigma_column="SIGMA_SPECTRUM"

Whilst this is convenient for sinple examples, it can become cumbersome when
doing particularly complicated calibration which uses a large number of the
available inputs.

Config Files
~~~~~~~~~~~~

In order to make specifying all of the relevant paramters more convenient,
QuartiCal allows users to provide one or more .yaml files containing the
neccessary arguments.

The .yaml file corresponding to the command above will look like this:

.. code-block:: yaml

    input_ms:
        path: path/to.ms
        sigma_column: SIGMA_SPECTRUM
    
    input_model:
        recipe: MODEL_DATA

If the above parset was named basic.yaml, it could be run by invoking:

.. code-block:: bash

	goquartical basic.yaml

This simple example only uses a fraction of the available options - 
unspecified options are populated from the defaults.

Note that multiple .yaml files can be specified in addition to command line
arguments. The .yaml files will be consumed in order, while command line
arguments will always be consumed last. Consider the following example:

.. code-block:: bash

	goquartical basic.yaml less_basic.yaml input_ms.path=path/to_different.ms

In this example, the contents of basic.yaml would be augmented (and
overwritten in the case of conflicting options) with the
contents of less_basic.yaml. Finally, the remaining command line options would
be taken into account, overwriting any conflicting values specified in the
provided .yaml files. This aims to make configuring QuartiCal as painless and 
flexible as possible.

.. note::

    Quartical provides a command line utility to generate unpopulated .yaml
    files. It can be invoked using:

    .. code-block:: bash

        goquartical-config configname.yaml

    This will produce a .yaml file with the given name and all available
    fields. 