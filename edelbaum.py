"""Edelbaum theory, reformulated by Kéchichian.

References
----------

* Edelbaum, T. N. "Propulsion Requirements for Controllable
  Satellites", 1961.
* Kéchichian, J. A. "Reformulation of Edelbaum's Low-Thrust
  Transfer Problem Using Optimal Control Theory", 1997.

"""
# TODO: Reformat references
import numpy as np

from poliastro.util import norm


def _compute_parameters(k, a_0, a_f, delta_inc):
    """Compute parameters of the model.

    """
    V_0 = np.sqrt(k / a_0)
    V_f = np.sqrt(k / a_f)
    beta_0 = np.arctan2(
        np.sin(np.pi / 2 * delta_inc),
        V_0 / V_f - np.cos(np.pi / 2 * delta_inc)
    )
    return V_0, beta_0


def guidance_law(k, a_0, a_f, delta_inc, f):
    """Guidance law from the model.

    Parameters
    ----------
    k : float
        Gravitational parameter.
    a_0 : float
        Initial semimajor axis.
    a_f : float
        Final semimajor axis.
    delta_inc : float
        Change in inclination.
    f : float
        Magnitude of constant acceleration

    """
    # TODO: Check documentation nomenclature
    V_0, beta_0 = _compute_parameters(k, a_0, a_f, delta_inc)

    def a_d(t0, u, _):
        # TODO: Is k needed in a general case?
        beta = np.arctan2(
            V_0 * np.sin(beta_0),
            V_0 * np.cos(beta_0) - f * t0
        )
        r, v = u[:3], u[3:]
        t_ = v; t_ /= norm(t_)  # TODO: Rewrite
        w_ = np.cross(r, v); w_ /= norm(w_)  # TODO: Rewrite
        #n_ = np.cross(t_, w_)
        accel_v = f * (
            np.cos(beta) * t_ +
            np.sin(beta) * w_
        )
        return accel_v

    return a_d


def extra_quantities(k, a_0, a_f, delta_inc, f):
    """Extra quantities given by the model.

    """
    # Extra interesting quantities
    V_0, beta_0 = _compute_parameters(k, a_0, a_f, delta_inc)
    delta_V = (
        V_0 * np.cos(beta_0) -
        V_0 * np.sin(beta_0) / np.tan(np.pi / 2 * delta_inc + beta_0)
    )
    t_f = delta_V / f

    return delta_V, t_f
