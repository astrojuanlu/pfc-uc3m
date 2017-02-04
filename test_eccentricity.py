from numpy.testing import assert_almost_equal

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from poliastro.twobody.propagation import cowell

from eccentricity_quasioptimal import guidance_law, extra_quantities


def test_sso_disposal_time_and_delta_v():
    a_0 = Earth.R.to(u.km).value + 900  # km
    e_0 = 0.0
    e_f = 0.1245  # Reverse-engineered from results
    f = 2.4e-7  # km / s2, assumed constant

    k = Earth.k.decompose([u.km, u.s]).value

    expected_t_f = 29.697  # days, reverse-engineered
    expected_delta_V = 0.6158  # km / s, lower than actual result

    delta_V, t_f = extra_quantities(k, a_0, e_0, e_f, f)

    assert_almost_equal(t_f / 86400, expected_t_f, decimal=2)
    assert_almost_equal(delta_V, expected_delta_V, decimal=4)


def test_ruggiero_case():
    a_0 = Earth.R.to(u.km).value + 900  # km
    e_0 = 0.0
    e_f = 0.1245  # Reverse-engineered from results
    f = 2.4e-7  # km / s2, assumed constant

    k = Earth.k.decompose([u.km, u.s]).value

    optimal_accel = guidance_law(f)

    _, t_f = extra_quantities(k, a_0, e_0, e_f, f)

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

    assert_almost_equal(sf.ecc.value, e_f, decimal=4)
