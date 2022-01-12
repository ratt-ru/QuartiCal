Usage
=====

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


Dynamic Fields
~~~~~~~~~~~~~~

QuartiCal is exceptionally flexible when it comes to gain calibration. To
avoid having too many options, it makes use of dynamically created config
fields. This applies to specifying gain terms. 

As an example, consider solving for a gain and bandpass (following their
usual definitions). To do so, a user would need to tell QuartiCal that there
are two gain terms and then provide the relevant arguments for those gains.

The contents of the .yaml would look as follows for this case:

.. code-block:: yaml

    solver:
        terms:
            - G
            - B
        iter_recipe:
            - 25
            - 25
    
    G:
        freq_interval: 0
    B:
        time_interval: 0

QuartiCal will automatically know that each term has its own dynamically
generated section in the config, labelled by the term name (G or B in this
example).

The above can also be specified on the command line using:

.. code-block:: bash

    solver.terms="[G,B]" solver.iter_recipe="[25,25]" G.freq_interval=0 B.time_interval=0


.. note::

    Lists in .yaml files can be specified in two ways:

    .. code-block:: yaml

        solver:
            terms:
                - G
                - B

    or 

    .. code-block:: yaml

        solver:
            terms: [G,B]
