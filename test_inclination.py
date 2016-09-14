import numpy as np

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.util import norm


def test_inclination():
    a_0 = 7000.0
    i_0 = (28.5 * u.deg).to(u.rad).value
    f = 3.5e-7

    k = Earth.k.decompose([u.km, u.s]).value

    def a_d(t0, u_, _):
        r, v = u_[:3], u_[3:]

        beta = np.pi / 2 * np.sign(r[0])  # Change with out-of-plane velocity
        #beta = np.pi / 2 * np.sign(r[1])  # Change at node crossing

        # DEBUG
        #ss = Orbit.from_vectors(Earth, r * u.km, v * u.km / u.s)
        #print(beta, ss.inc.to("deg"))
        # END DEBUG

        w_ = np.cross(r, v) / norm(np.cross(r, v))
        accel_v = f * np.sin(beta) * w_
        return accel_v

    # Retrieve r and v from initial orbit
    s0 = Orbit.circular(Earth, a_0 * u.km - Earth.R, i_0 * u.rad)
    r0, v0 = s0.rv()

    tf = 150 * s0.period

    # Propagate orbit
    r, v = cowell(k,
                  r0.to(u.km).value,
                  v0.to(u.km / u.s).value,
                  tf.to(u.s).value,
                  ad=a_d,
                  nsteps=100000)

    sf = Orbit.from_vectors(Earth,
                            r * u.km,
                            v * u.km / u.s,
                            s0.epoch + tf)

    print(sf.a.to(u.km))
    print(sf.ecc)
    print(sf.inc.to("deg"))
