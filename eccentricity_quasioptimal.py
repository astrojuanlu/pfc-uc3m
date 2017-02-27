"""Quasi optimal eccentricity-only change, with formulas developed by Pollard.

References
----------

* Pollard, J. E. "Simplified Approach for Assessment of Low-Thrust
  Elliptical Orbit Transfers", 1997.

"""
import numpy as np

from poliastro.twobody.decorators import state_from_vector
from poliastro.util import norm, circular_velocity


def guidance_law(ecc_0, ecc_f, f):
    """Guidance law from the model.

    Thrust is aligned with an inertially fixed direction perpendicular to the
    semimajor axis of the orbit.

    Parameters
    ----------
    ecc_0 : float
        Initial eccentricity.
    ecc_f : float
        Final eccentricity.
    f : float
        Magnitude of constant acceleration

    """
    delta_alpha = 0.0 if ecc_f - ecc_0 > 0 else np.pi

    @state_from_vector
    def a_d(t0, ss):
        r = ss.r.value
        v = ss.v.value
        nu = ss.nu.value

        alpha_ = nu + delta_alpha

        r_ = r / norm(r)
        w_ = np.cross(r, v) / norm(np.cross(r, v))
        s_ = np.cross(w_, r_)
        accel_v = f * (
            np.cos(alpha_) * s_ +
            np.sin(alpha_) * r_
        )
        return accel_v

    return a_d


def delta_V(V_0, ecc_0, ecc_f):
    """Compute required increment of velocity.

    """
    return 2 / 3 * V_0 * np.abs(np.arcsin(ecc_0) - np.arcsin(ecc_f))


def extra_quantities(k, a, ecc_0, ecc_f, f):
    """Extra quantities given by the model.

    """
    V_0 = circular_velocity(k, a)
    delta_V_ = delta_V(V_0, ecc_0, ecc_f)
    t_f_ = delta_V_ / f

    return delta_V_, t_f_

