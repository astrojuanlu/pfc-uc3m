"""Argument of perigee change, with formulas developed by Pollard.

References
----------

* Pollard, J. E. "Simplified Approach for Assessment of Low-Thrust
  Elliptical Orbit Transfers", 1997.
* Pollard, J. E. "Evaluation of Low-Thrust Orbital Maneuvers", 1998.

"""
import numpy as np

from poliastro.twobody.decorators import state_from_vector
from poliastro.util import norm, circular_velocity


def apsidal_precession(ss, J2):
    return (
        3 * ss.n * ss.attractor.R ** 2 * J2 * (4 - 5 * np.sin(ss.inc) ** 2) /
        (4 * ss.a ** 2 * (1 - ss.ecc ** 2) ** 2)
    )


def guidance_law(f):
    """Guidance law from the model.

    Thrust is aligned with an inertially fixed direction perpendicular to the
    semimajor axis of the orbit.

    Parameters
    ----------
    f : float
        Magnitude of constant acceleration

    """
    @state_from_vector
    def a_d(t0, ss):
        r = ss.r.value
        v = ss.v.value
        nu = ss.nu.value

        alpha_ = nu - np.pi / 2

        r_ = r / norm(r)
        w_ = np.cross(r, v) / norm(np.cross(r, v))
        s_ = np.cross(w_, r_)
        accel_v = f * (
            np.cos(alpha_) * s_ +
            np.sin(alpha_) * r_
        )
        return accel_v

    return a_d


def delta_V(V, ecc, argp_0, argp_f, f, A):
    """Compute required increment of velocity.

    """
    delta_argp = argp_f - argp_0
    return delta_argp / (3 * np.sign(delta_argp) / 2 * np.sqrt(1 - ecc ** 2) / ecc / V + A / f)


def extra_quantities(k, a, ecc, argp_0, argp_f, f, A=0.0):
    """Extra quantities given by the model.

    """
    V = circular_velocity(k, a)
    delta_V_ = delta_V(V, ecc, argp_0, argp_f, f, A)
    t_f_ = delta_V_ / f

    return delta_V_, t_f_


if __name__ == '__main__':
    from poliastro.examples import iss
    J2_EARTH = 0.0010826359
    print(apsidal_precession(iss, J2_EARTH))


