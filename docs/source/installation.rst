Installation
============

This page details QuartiCal's recommended installation procedure.

Ubuntu 18.04+ via pip
~~~~~~~~~~~~~~~~~~~~~

This is the preferred method of installation. It is simple but may be
vulnerable to upstream changes.

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


Ubuntu 18.04+ via poetry
~~~~~~~~~~~~~~~~~~~~~~~~

Installing via poetry is less simple but should always work.

Firstly, install `poetry <https://python-poetry.org/docs/>`_

Assuming you have cloned the repository from git and checked out the relevant
tag, run the following from inside the QuartiCal folder:

.. code:: bash

	poetry install

.. note::

	This will automatically install QuartiCal into a new virtual environment
	matching your system Python. The Python version can be changed prior to
	installation using:

	.. code:: bash

		poetry env use python3.10

Users can enter the QuartiCal virtual environment using:

	.. code:: bash

		poetry -C path/to/repo shell


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
