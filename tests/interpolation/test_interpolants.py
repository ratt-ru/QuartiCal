import pytest
import dask.array as da
import numpy as np
from quartical.interpolation.interpolants import (_interpolate_missing,
                                                  spline2d,
                                                  csaps2d)


# flake8: noqa
MISSING_ARCHETYPES = (
    np.array([[np.nan, np.nan, np.nan],
              [np.nan,      1, np.nan],
              [np.nan, np.nan, np.nan]]),
    np.array([[1,      1, 1],
              [1, np.nan, 1],
              [1,      1, 1]]),
    np.array([[     1, np.nan,      1],
              [np.nan,      1, np.nan],
              [     1, np.nan,      1]]),
    np.array([[np.nan, 1, 1],
              [np.nan, 1, 1],
              [np.nan, 1, 1]]),
    np.array([[np.nan, np.nan, np.nan],
              [     1,      1,      1],
              [     1,      1,      1]])
)


def data_from_archetypes(archetypes):

    test_data = []

    for archetype in archetypes:
        x1, x2 = archetype.shape
        for i in range(x1):
            for j in range(x2):
                test_array = np.roll(archetype, i, axis=0)
                test_array = np.roll(test_array, j, axis=1)
                test_data.append(test_array)

    return test_data


MISSING_TEST_DATA = data_from_archetypes(MISSING_ARCHETYPES)

# -----------------------------_interpolate_missing----------------------------

@pytest.mark.parametrize("data", MISSING_TEST_DATA)
def test_interpolate_missing_archetypes(data):

    n_x1, n_x2 = data.shape
    x1, x2 = np.arange(n_x1), np.arange(n_x2)
    y = data.reshape((n_x1, n_x2, 1, 1, 1))

    assert np.allclose(_interpolate_missing(x1, x2, y), np.ones(data.shape))


@pytest.mark.parametrize("n_x1", (1, 3))
@pytest.mark.parametrize("n_x2", (1, 3))
def test_interpolate_missing_allnan(n_x1, n_x2):

    data = np.zeros((n_x1, n_x2)) * np.nan

    x1, x2 = np.arange(n_x1), np.arange(n_x2)
    y = data.reshape((n_x1, n_x2, 1, 1, 1))

    assert np.allclose(_interpolate_missing(x1, x2, y), np.zeros(data.shape))

# -----------------------------------spline2d----------------------------------


@pytest.mark.parametrize("input_lbound, input_rbound", ((2, 8),))
@pytest.mark.parametrize("output_lbound, output_rbound",
                         ((2, 8), (1, 9), (3, 7)))
@pytest.mark.parametrize("n_t_sample", (5, 10))
@pytest.mark.parametrize("n_f_sample", (5, 10))
def test_spline2d(input_lbound,
                  input_rbound,
                  output_lbound,
                  output_rbound,
                  n_t_sample,
                  n_f_sample):

    data = np.ones((n_t_sample, n_f_sample))

    x1 = np.linspace(input_lbound, input_rbound, n_t_sample)
    x2 = np.linspace(input_lbound, input_rbound, n_f_sample)
    xx1 = np.linspace(output_lbound, output_rbound, n_t_sample)
    xx2 = np.linspace(output_lbound, output_rbound, n_f_sample)
    y = data.reshape((n_t_sample, n_f_sample, 1, 1, 1))

    assert np.allclose(spline2d(x1, x2, y, xx1, xx2), np.ones(data.shape))


# -----------------------------------spline2d----------------------------------


@pytest.mark.parametrize("input_lbound, input_rbound", ((2, 8),))
@pytest.mark.parametrize("output_lbound, output_rbound",
                         ((2, 8), (1, 9), (3, 7)))
@pytest.mark.parametrize("n_t_sample", (5, 10))
@pytest.mark.parametrize("n_f_sample", (5, 10))
def test_csaps2d(input_lbound,
                  input_rbound,
                  output_lbound,
                  output_rbound,
                  n_t_sample,
                  n_f_sample):

    data = np.ones((n_t_sample, n_f_sample))

    x1 = np.linspace(input_lbound, input_rbound, n_t_sample)
    x2 = np.linspace(input_lbound, input_rbound, n_f_sample)
    xx1 = np.linspace(output_lbound, output_rbound, n_t_sample)
    xx2 = np.linspace(output_lbound, output_rbound, n_f_sample)
    y = data.reshape((n_t_sample, n_f_sample, 1, 1, 1))

    assert np.allclose(csaps2d(x1, x2, y, xx1, xx2), np.ones(data.shape))


# -----------------------------------------------------------------------------


