import pytest
import numpy as np

from numpy.testing import assert_allclose

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from poliastro.twobody.propagation import cowell

from combined_ei import guidance_law, extra_quantities


# Data extracted from plots using
# http://arohatgi.info/WebPlotDigitizer/app/
@pytest.mark.parametrize("ecc_0,inc_f,expected_beta,expected_delta_V", [
    [0.1, 20.0, 83.043, 1.6789],
    [0.2, 20.0, 76.087, 1.6890],
    [0.4, 20.0, 61.522, 1.7592],
    [0.6, 16.0, 40.0, 1.7241],
    [0.8, 10.0, 16.304, 1.9799]
])
def test_geo_cases_beta_and_delta_v(ecc_0, inc_f, expected_beta, expected_delta_V):
    a = 42164  # km
    inc_0 = 0.0  # rad, dummy value
    ecc_f = 0.0
    argp = 0.0  # rad, the method is efficient for 0 and 180
    f = 1e-7  # km / s2, unused

    k = Earth.k.decompose([u.km, u.s]).value

    inc_f = np.radians(inc_f)
    expected_beta = np.radians(expected_beta)

    delta_V, beta, _ = extra_quantities(k, a, ecc_0, ecc_f, inc_0, inc_f, argp, f)

    assert_allclose(delta_V, expected_delta_V, rtol=1e-2)
    assert_allclose(beta, expected_beta, rtol=1e-2)


@pytest.mark.skip
def test_geo_cases_numerical():
    a_0 = Earth.R.to(u.km).value + 900  # km
    ecc_0 = 0.0
    ecc_f = 0.1245  # Reverse-engineered from results
    f = 2.4e-7  # km / s2, assumed constant

    k = Earth.k.decompose([u.km, u.s]).value

    optimal_accel = guidance_law(f)

    _, t_f = extra_quantities(k, a_0, ecc_0, ecc_f, f)

    # Retrieve r and v from initial orbit
    s0 = Orbit.circular(Earth, 900 * u.km)
    r0, v0 = s0.rv()

    # Propagate orbit
    r, v = cowell(k,
                  r0.to(u.km).value,
                  v0.to(u.km / u.s).value,
                  t_f,
                  ad=optimal_accel,
                  nsteps=1000000)

    sf = Orbit.from_vectors(Earth,
                            r * u.km,
                            v * u.km / u.s,
                            s0.epoch + t_f * u.s)

    assert_allclose(sf.ecc.value, ecc_f)
