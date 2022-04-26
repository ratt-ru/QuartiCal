Installation
============

This page details QuartiCal's recommended installation procedure.

Ubuntu 18.04+
~~~~~~~~~~~~~

If you wish to install QuartiCal in a virtual environment (recommended), see
`Using a virtual environment`_.

QuartiCal can be installed by running the following:

.. code:: bash

	pip3 install quartical

.. note::

	To install in development mode, assuming that you have already
	cloned the repository, run:

	.. code:: bash

		pip3 install -e path/to/repo/


Using a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing QuartiCal in a virtual environment is highly recommended. To
install virtualenv using apt, run:

.. code:: bash

	sudo apt install python3-virtualenv

To create a virtualenv, run:

.. code:: bash

	virtualenv -p python3 path/to/env/name

Activate the environment using:

.. code:: bash

	source path/to/env/name/bin/activate

This should change the command line prompt to be consistent with the
virtualenv name.

It is often necessary to update pip, setuptools and wheel inside the
environment:

.. code:: bash

    pip3 install -U pip setuptools wheel
