import pytest
import dask.array as da
import numpy as np
from quartical.interpolation.interpolants import (_interpolate_missing,
                                                  spline2d)


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

def test_gpr():
    from quartical.interpolation.interpolants import matern52
    from quartical.interpolation.interpolants import kron_matvec
    from quartical.interpolation.interpolants import _interp_gpr

    nt = 50
    ntp = 50
    nf = 100
    nfp = 100
    nant = 1
    ndir = 1
    ncorr = 1
    sigmaf = 0.25
    lt = 0.1
    lf = 0.1

    t = np.linspace(0, 1, nt)
    tp = np.linspace(0, 1, ntp)
    f = np.linspace(0, 1, nf)
    fp = np.linspace(0, 1, nfp)

    K = matern52(t, t, sigmaf, lt)
    Lt = np.linalg.cholesky(K +1e-14*np.eye(nt))

    K = matern52(f, f, 1.0, lf)
    Lf = np.linalg.cholesky(K + 1e-14*np.eye(nf))

    L = (Lt, Lf)

    smooth_gains = np.zeros((nt, nf, nant, ndir, ncorr), dtype=np.complex128)
    noisy_gains = np.zeros((nt, nf, nant, ndir, ncorr), dtype=np.complex128)
    jhj = np.zeros((nt, nf, nant, ndir, ncorr), dtype=np.complex128)
    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                xif = (np.random.randn(nt, nf) +
                       1.0j*np.random.randn(nt, nf))
                smooth_gains[:, :, p, d, c] = kron_matvec(L, xif)

                Sigma = 1e-12
                # xin = (np.random.randn(nt, nf) +
                #        1.0j*np.random.randn(nt, nf)) * np.sqrt(Sigma/2.0)
                jhj[:, :, p, d, c] = 1.0/Sigma

                noisy_gains[:, :, p, d, c] = smooth_gains[:, :, p, d, c] # + xin

    rec_gains = _interp_gpr(t, f, noisy_gains, jhj, tp, fp, sigmaf, lt, lf)

    # interpolate smooth gains for comparsion
    from scipy.interpolate import RectBivariateSpline
    import matplotlib.pyplot as plt
    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                sgor = RectBivariateSpline(t, f, smooth_gains[:, :, p, d, c].real)
                sgoi = RectBivariateSpline(t, f, smooth_gains[:, :, p, d, c].imag)

                # print(np.abs(sgp-rec_gains[:, :, p, d, c]).max())

                # plt.figure(1)
                # plt.imshow(sgor(tp, fp) - rec_gains[:, :, p, d, c].real)
                # plt.colorbar()

                # plt.figure(2)
                # plt.imshow(sgoi(tp, fp) - rec_gains[:, :, p, d, c].imag)
                # plt.colorbar()

                # plt.show()

                real_diff = sgor(tp, fp) - rec_gains[:, :, p, d, c].real
                imag_diff = sgoi(tp, fp) - rec_gains[:, :, p, d, c].imag
                # print(np.abs(real_diff).max())
                # print(np.abs(real_diff).max())
                assert np.allclose(real_diff, 0.0, atol=1e-3)
                assert np.allclose(imag_diff, 0.0, atol=1e-3)

test_gpr()
