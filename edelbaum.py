"""Edelbaum theory, reformulated by Kéchichian.

References
----------

* Edelbaum, T. N. "Propulsion Requirements for Controllable
  Satellites", 1961.
* Kéchichian, J. A. "Reformulation of Edelbaum's Low-Thrust
  Transfer Problem Using Optimal Control Theory", 1997.

"""
import numpy as np

from poliastro.twobody.decorators import state_from_vector
from poliastro.util import norm


def _compute_parameters(k, a_0, a_f, i_0, i_f):
    """Compute parameters of the model.

    """
    delta_inc = abs(i_f - i_0)
    V_0 = circular_velocity(k, a_0)
    V_f = circular_velocity(k, a_f)
    beta_0_ = beta_0(V_0, V_f, i_0, i_f)

    return V_0, beta_0_, delta_inc


def circular_velocity(k, a):
    """Compute circular velocity for a given body (k) and semimajor axis (a).

    """
    return np.sqrt(k / a)


def beta_0(V_0, V_f, i_0, i_f):
    """Compute initial yaw angle (β) as a function of the problem parameters.

    """
    delta_i_f = abs(i_f - i_0)
    return np.arctan2(
        np.sin(np.pi / 2 * delta_i_f),
        V_0 / V_f - np.cos(np.pi / 2 * delta_i_f)
    )


def beta(t, *, V_0, f, beta_0):
    """Compute yaw angle (β) as a function of time and the problem parameters.

    """
    return np.arctan2(V_0 * np.sin(beta_0), V_0 * np.cos(beta_0) - f * t)


def delta_V(V_0, beta_0, i_0, i_f):
    """Compute required increment of velocity.

    """
    delta_i_f = abs(i_f - i_0)
    return V_0 * np.cos(beta_0) - V_0 * np.sin(beta_0) / np.tan(np.pi / 2 * delta_i_f + beta_0)


def t_f(V_0, f, beta_0, i_0, i_f):
    """Compute required time of flight.

    """
    delta_V_ = delta_V(V_0, beta_0, i_0, i_f)
    return delta_V_ / f


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
    V_0, beta_0_, _ = _compute_parameters(k, a_0, a_f, i_0, i_f)

    @state_from_vector
    def a_d(t0, ss):
        r = ss.r.value
        v = ss.v.value

        # Change sign of beta with the out-of-plane velocity
        beta_ = beta(t0, V_0=V_0, f=f, beta_0=beta_0_) * np.sign(r[0] * (i_f - i_0))

        t_ = v / norm(v)
        w_ = np.cross(r, v) / norm(np.cross(r, v))
        # n_ = np.cross(t_, w_)
        accel_v = f * (
            np.cos(beta_) * t_ +
            np.sin(beta_) * w_
        )
        return accel_v

    return a_d


def extra_quantities(k, a_0, a_f, i_0, i_f, f):
    """Extra quantities given by the model.

    """
    V_0, beta_0_, _ = _compute_parameters(k, a_0, a_f, i_0, i_f)
    delta_V_ = delta_V(V_0, beta_0_, i_0, i_f)
    t_f_ = t_f(V_0, f, beta_0_, i_0, i_f)

    return delta_V_, t_f_
