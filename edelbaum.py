"""Edelbaum theory, reformulated by Kéchichian.

References
----------

* Edelbaum, T. N. "Propulsion Requirements for Controllable
  Satellites", 1961.
* Kéchichian, J. A. "Reformulation of Edelbaum's Low-Thrust
  Transfer Problem Using Optimal Control Theory", 1997.

"""
import numpy as np

from astropy import units as u

from poliastro.twobody.decorators import state_from_vector
from poliastro.util import norm


def _compute_parameters(k, a_0, a_f, i_0, i_f):
    """Compute parameters of the model.

    """
    delta_inc = abs(i_f - i_0)
    V_0 = np.sqrt(k / a_0)
    V_f = np.sqrt(k / a_f)
    beta_0 = np.arctan2(
        np.sin(np.pi / 2 * delta_inc),
        V_0 / V_f - np.cos(np.pi / 2 * delta_inc)
    )
    return V_0, beta_0, delta_inc


def guidance_law(k, a_0, a_f, i_0, i_f, f):
    """Guidance law from the model.

    Parameters
    ----------
    k : float
        Gravitational parameter.
    a_0 : float
        Initial semimajor axis.
    a_f : float
        Final semimajor axis.
    i_0 : float
        Initial inclination.
    i_f : float
        Final inclination.
    f : float
        Magnitude of constant acceleration

    """
    # TODO: Check documentation nomenclature
    V_0, beta_0, _ = _compute_parameters(k, a_0, a_f, i_0, i_f)

    @state_from_vector
    def a_d(t0, ss):
        r = ss.r.to(u.km).value
        v = ss.v.to(u.km / u.s).value

        beta = np.arctan2(
            V_0 * np.sin(beta_0),
            V_0 * np.cos(beta_0) - f * t0
        ) * np.sign(r[0] * (i_f - i_0))  # Change sign of beta with the out-of-plane velocity

        # DEBUG
        #print(beta, ss.inc.to("deg"))
        # END DEBUG

        t_ = v / norm(v)
        w_ = np.cross(r, v) / norm(np.cross(r, v))
        #n_ = np.cross(t_, w_)
        accel_v = f * (
            np.cos(beta) * t_ +
            np.sin(beta) * w_
        )
        return accel_v

    return a_d


def extra_quantities(k, a_0, a_f, i_0, i_f, f):
    """Extra quantities given by the model.

    """
    # Extra interesting quantities
    V_0, beta_0, delta_inc = _compute_parameters(k, a_0, a_f, i_0, i_f)
    delta_V = (
        V_0 * np.cos(beta_0) -
        V_0 * np.sin(beta_0) / np.tan(np.pi / 2 * delta_inc + beta_0)
    )
    t_f = delta_V / f

    return delta_V, t_f
