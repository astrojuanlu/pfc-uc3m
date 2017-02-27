"""Quasi optimal eccentricity-only change, with formulas developed by Pollard.

References
----------

* Pollard, J. E. "Simplified Approach for Assessment of Low-Thrust
  Elliptical Orbit Transfers", 1997.

"""
import numpy as np

from poliastro.twobody.decorators import state_from_vector
from poliastro.util import norm, circular_velocity


def guidance_law(ss_0, ecc_f, f):
    """Guidance law from the model.

    Thrust is aligned with an inertially fixed direction perpendicular to the
    semimajor axis of the orbit.

    Parameters
    ----------
    ss_0 : Orbit
        Initial orbit, containing all the information.
    ecc_f : float
        Final eccentricity.
    f : float
        Magnitude of constant acceleration

    """
    # We fix the inertial direction at the beginning
    ecc_0 = ss_0.ecc.value
    if ecc_0 > 0.001:  # Arbitrary tolerance
        ref_vec = ss_0.ecc_vec / ecc_0
    else:
        ref_vec = ss_0.r / norm(ss_0.r)

    h_unit = ss_0.h_vec / norm(ss_0.h_vec)
    thrust_unit = np.cross(h_unit, ref_vec) * np.sign(ecc_f - ecc_0)

    @state_from_vector
    def a_d(t0, ss):
        accel_v = f * thrust_unit
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

