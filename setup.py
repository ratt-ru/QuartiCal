# -*- coding: utf-8 -*-
#
# CubiCalV2: a radio interferometric calibration suite
# (c) 2019 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for
# details.
#
# Copyright (c) 2019 SKA South Africa
#
# This file is part of CubiCalV2.
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

import os
import cubicalv2
from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

# Check for readthedocs environment variable.

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    requirements = ["ruamel.yaml",
                    "numpy",
                    "dask",
                    "xarray-ms"]
else:
    requirements = ["ruamel.yaml",
                    "numpy",
                    "dask",
                    "xarray-ms"]

setup(name='cubicalv2',
      version=cubicalv2.__version__,
      description="""Fast calibration implementation exploiting complex
                        optimisation.""",
      url='https://github.com/JSKenyon/CubiCalV2',
      classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy"],
      author='Jonathan Kenyon',
      author_email='jonosken@gmail.com',
      license='GNU GPL v3',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      python_requires=">=3.6",
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False,
      entry_points={'console_scripts':
                    ['gocubical = cubicalv2.executor:execute']},)
