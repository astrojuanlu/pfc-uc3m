from astropy import units as u

from numpy.testing import assert_almost_equal

from poliastro.bodies import Earth

from edelbaum import extra_quantities


def test_leo_geo_time_and_delta_v():
    a_0 = 7000.0  # km
    a_f = 42166.0  # km
    i_f = 0.0  # rad
    i_0 = (28.5 * u.deg).to(u.rad).value  # rad
    f = 3.5e-7  # km / s2

    k = Earth.k.decompose([u.km, u.s]).value

    expected_t_f = 191.26295  # s
    expected_delta_V = 5.78378  # km / s

    delta_V, t_f = extra_quantities(k, a_0, a_f, i_f - i_0, f)

    assert_almost_equal(t_f / 86400, expected_t_f, decimal=2)
    assert_almost_equal(delta_V, expected_delta_V, decimal=4)
