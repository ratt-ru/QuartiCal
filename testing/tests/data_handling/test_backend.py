# -*- coding: utf-8 -*-
"""The casacure table backend and its measures parity with casacore.

QuartiCal reads Measurement Sets through dask-ms, which imports its backend as
``casacore.tables``. Importing :mod:`quartical` aliases that namespace onto
casacure; the tests here pin the alias and the accuracy of the parallactic
angle built on top of it.
"""

import sys

import numpy as np

from quartical.data_handling import backend


# A MeerKAT antenna (ITRF metres), the 3C147 field centre in J2000 radians and
# an epoch after J2000. casacore 3.8.1's azimuth/altitude position angle is
# 133.550707 deg for this combination; its AZEL position angle (131.192182 deg)
# uses the geocentric latitude, which casacore applies to AZEL alone.
ANT_POSITION = (5109360.133, 2006852.586, -3238948.127)
FIELD_CENTRE = (0.5, -0.5)
EPOCH = 57844.5

CASACORE_PARALLACTIC_ANGLE = 133.550707

# casacure's measures implement the IAU SOFA stack and are held to under an
# arcsecond of astropy's AltAz, so the position angle tracks casacore's
# geodetic AZELGEO path closely. The bound is deliberately looser than the
# measured agreement (a few arcsec over a MeerKAT field/epoch grid): the angle
# is steeply sensitive to the converted declination near the zenith.
PARALLACTIC_ANGLE_TOLERANCE = 5.0 / 3600.0  # degrees


def test_backend_is_casacure():
    """Importing quartical selects casacure for tables, measures and quanta."""

    assert backend.BACKEND_NAME == "casacure"

    for submodule in ("tables", "measures", "quanta"):
        casacore_module = sys.modules["casacore.%s" % submodule]
        casacure_module = sys.modules["casacure.%s" % submodule]
        assert casacore_module is casacure_module


def test_casacore_tables_alias_resolves():
    """``import casacore.tables`` yields casacure.tables, not python-casacore."""

    import casacore.tables

    assert casacore.tables is sys.modules["casacure.tables"]


def test_parallactic_angle_matches_casacore():
    """The measures call chain in angles.py reproduces casacore's angle."""

    import casacure.measures
    import casacure.quanta as pq

    cms = casacure.measures.measures()

    field_centre = cms.direction(
        "J2000", *(pq.quantity(fi, "rad") for fi in FIELD_CENTRE)
    )
    zenith = cms.direction(
        "AZELGEO", *(pq.quantity(fi, "deg") for fi in (0, 90))
    )
    position = cms.position(
        "itrf", *(pq.quantity(p, "m") for p in ANT_POSITION)
    )

    cms.do_frame(cms.epoch("UTC", pq.quantity(EPOCH * 86400.0, "s")))
    cms.do_frame(position)

    angle = np.degrees(cms.posangle(field_centre, zenith).get_value("rad"))

    assert np.isclose(
        angle,
        CASACORE_PARALLACTIC_ANGLE,
        atol=PARALLACTIC_ANGLE_TOLERANCE,
    ), angle
