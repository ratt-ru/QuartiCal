# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

# TODO: Dependence on stimela branch is temporary. Will be removed post after
# stimela2 is released and this branch is merged.

requirements = [
    "tbump",
    "columnar",
    "ruamel.yaml",
    "numpy",
    "dask-ms[xarray, zarr, s3]>=0.2.9",
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
    "xarray>=0.20.0",
    "stimela@git+https://github.com/caracal-pipeline/stimela2.git@kube"
]

setup(
    name='quartical',
    version='0.1.7',
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
