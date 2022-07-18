# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

requirements = [
    "tbump",
    "columnar",
    "ruamel.yaml",
    "numpy",
    "dask-ms[xarray, zarr]>=0.2.9",
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

# If scabha is not already installed, add it to the requirements. This is just
# a transitionary hack as the next release of stimela will include scabha.
# Once stimela is released, we can add a stimela>=2 dependency and omit this.
try:
    import scabha  # noqa
except ImportError:
    requirements.append('scabha')

setup(
    name='quartical',
    version='0.1.6',
    description="Fast calibration implementation exploiting complex "
                "optimisation.",
    url='https://github.com/JSKenyon/QuartiCal',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    author='Jonathan Kenyon',
    author_email='jonosken@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'goquartical = quartical.executor:execute',
            'goquartical-config = quartical.config.parser:create_user_config',
            'goquartical-backup = quartical.apps.backup:backup',
            'goquartical-restore = quartical.apps.backup:restore',
            'goquartical-summary = quartical.apps.summary:summary'
        ]
    },
)
