"""Simultaneous eccentricity and inclination changes.

References
----------

* Pollard, J. E. "Simplified Analysis of Low-Thrust Orbital Maneuvers", 2000.

"""
import numpy as np

from poliastro.twobody.decorators import state_from_vector
from poliastro.util import norm, circular_velocity


def guidance_law(ecc_0, ecc_f, inc_0, inc_f, argp, f):
    """Guidance law from the model.

    Thrust is aligned with an inertially fixed direction perpendicular to the
    semimajor axis of the orbit.

    Parameters
    ----------
    ecc_0 : float
        Initial eccentricity.
    ecc_f : float
        Final eccentricity.
    inc_0 : float
        Initial inclination.
    inc_f : float
        Final inclination.
    argp : float
        Argument of perigee.
    f : float
        Magnitude of constant acceleration.

    """
    beta_ = beta(ecc_0, ecc_f, inc_0, inc_f, argp)

    @state_from_vector
    def a_d(t0, ss):
        r = ss.r.value
        v = ss.v.value
        nu = ss.nu.value

        alpha_ = nu

        r_ = r / norm(r)
        w_ = np.cross(r, v) / norm(np.cross(r, v))
        s_ = np.cross(w_, r_)
        accel_v = f * (
            np.cos(alpha_) * s_ +
            np.sin(alpha_) * r_
        )
        return accel_v

    return a_d


def beta(ecc_0, ecc_f, inc_0, inc_f, argp):
    # Note: "The argument of perigee co will vary during the orbit transfer
    # due to the natural drift and because e may approach zero.
    # However, [the equation] still gives a good estimate of the desired
    # thrust angle."
    return np.arctan(abs(3 * np.pi * (inc_f - inc_0) / (4 * np.cos(argp) * (ecc_0 - ecc_f + np.log(
        (1 + ecc_f) * (-1 + ecc_0) / ((1 + ecc_0) * (-1 + ecc_f)))))))


def delta_V(V_0, ecc_0, ecc_f, beta_):
    """Compute required increment of velocity.

    """
    return 2 * V_0 * np.abs(np.arcsin(ecc_0) - np.arcsin(ecc_f)) / (3 * np.cos(beta_))


def extra_quantities(k, a, ecc_0, ecc_f, inc_0, inc_f, argp, f):
    """Extra quantities given by the model.

    """
    beta_ = beta(ecc_0, ecc_f, inc_0, inc_f, argp)
    V_0 = circular_velocity(k, a)
    delta_V_ = delta_V(V_0, ecc_0, ecc_f, beta_)
    t_f_ = delta_V_ / f

    return delta_V_, beta_, t_f_

