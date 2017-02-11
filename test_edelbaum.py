import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from poliastro.twobody.propagation import cowell

from edelbaum import guidance_law, extra_quantities


def test_leo_geo_time_and_delta_v():
    a_0 = 7000.0  # km
    a_f = 42166.0  # km
    inc_0 = (28.5 * u.deg).to(u.rad).value  # rad
    i_f = 0.0  # rad
    f = 3.5e-7  # km / s2

    k = Earth.k.decompose([u.km, u.s]).value

    expected_t_f = 191.26295  # days
    expected_delta_V = 5.78378  # km / s

    delta_V, t_f = extra_quantities(k, a_0, a_f, inc_0, i_f, f)

    assert_almost_equal(delta_V, expected_delta_V, decimal=4)
    assert_almost_equal(t_f / 86400, expected_t_f, decimal=2)


@pytest.mark.parametrize("inc_0", [np.radians(28.5), np.radians(90.0)])
def test_leo_geo_numerical(inc_0):
    a_0 = 7000.0  # km
    a_f = 42166.0  # km
    i_f = 0.0  # deg
    f = 3.5e-7  # km / s2

    k = Earth.k.decompose([u.km, u.s]).value

    edelbaum_accel = guidance_law(k, a_0, a_f, inc_0, i_f, f)

    _, t_f = extra_quantities(k, a_0, a_f, inc_0, i_f, f)

    # Retrieve r and v from initial orbit
    s0 = Orbit.circular(Earth, a_0 * u.km - Earth.R, inc_0 * u.rad)
    r0, v0 = s0.rv()

    # Propagate orbit
    r, v = cowell(k,
                  r0.to(u.km).value,
                  v0.to(u.km / u.s).value,
                  t_f,
                  ad=edelbaum_accel,
                  nsteps=1000000)

    sf = Orbit.from_vectors(Earth,
                            r * u.km,
                            v * u.km / u.s,
                            s0.epoch + t_f * u.s)

    assert_almost_equal(sf.a.to(u.km).value, a_f, decimal=0)
    assert_almost_equal(sf.ecc.value, 0.0, decimal=2)
    assert_almost_equal(sf.inc.to(u.rad).value, i_f, decimal=2)
