# -*- coding: utf-8 -*-
#
# QuartiCal: a radio interferometric calibration suite
# (c) 2019 Rhodes University & Jonathan S. Kenyon
# https://github.com/JSKenyon/QuartiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for
# details.
#
# Copyright (c) 2019 SKA South Africa
#
# This file is part of QuartiCal.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import quartical
from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

requirements = [
    "ruamel.yaml",
    "numpy",
    "dask-ms[xarray, zarr]",
    "codex-africanus[dask, scipy, astropy, python-casacore]",
    "dask[array]",
    "astro-tigger-lsm",
    "loguru",
    "numba>=0.55.0",
    "distributed",
    "requests",
    "pytest",
    "omegaconf",
    "colorama",
    "bokeh",
    "xarray>=0.20.0"
]

setup(
    name='quartical',
    version=quartical.__version__,
    description="Fast calibration implementation exploiting complex "
                "optimisation.",
    url='https://github.com/JSKenyon/QuartiCal',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
        "Programming Language :: Python :: 3.8"
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    author='Jonathan Kenyon',
    author_email='jonosken@gmail.com',
    license='GNU GPL v3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'goquartical = quartical.executor:execute',
            'goquartical-config = quartical.config.parser:create_user_config',
            'goquartical-backup = quartical.apps.backup:backup',
            'goquartical-restore = quartical.apps.backup:restore',
            'goquartical-summary = quartical.apps.stats:summary'
        ]
    },
)
