"""Reproduces plots from original Kéchichian 1997 paper.

"""
import numpy as np
import matplotlib.pyplot as plt

from poliastro.util import circular_velocity

from edelbaum import beta_0, beta


def V(t, *, V_0, f, beta_0):
    """Compute velocity (V) as a function of time and the problem parameters.

    """
    return np.sqrt(V_0 ** 2 - 2 * V_0 * f * t * np.cos(beta_0) + f ** 2 * t ** 2)


def a(t, *, k, V_0, f, beta_0):
    """Compute semimajor axis (a) as a function of time, a given body (k) and the problem parameters.

    """
    return k / V(t, V_0=V_0, f=f, beta_0=beta_0) ** 2


def delta_inc(t, *, V_0, f, beta_0):
    """Difference of inclination (Δi) as a function of time and the problem parameters.

    """
    return 2 / np.pi * (np.arctan((f * t - V_0 * np.cos(beta_0)) / (V_0 * np.sin(beta_0))) + np.pi / 2 - beta_0)


def plot_problem(k=398600.0, a_0=7000.0, a_f=42166.0, inc_0=np.radians(28.5), i_f=0.0, f=3.5e-7, t_f=200):
    """Plot interesting quantities.

    """
    t_domain = np.linspace(0, t_f, num=2000) * 86400  # s

    V_0 = circular_velocity(k, a_0)
    V_f = circular_velocity(k, a_f)

    beta_0_ = beta_0(V_0, V_f, inc_0, i_f)

    _, ax_l1 = plt.subplots()
    ax_l1.set_xlabel("Time, days")

    ax_l1.plot(t_domain / 86400, np.degrees(beta(
        t_domain,
        V_0=V_0,
        f=f,
        beta_0=beta_0_
    )), color='k', linestyle='solid')
    ax_l1.set_ylabel("Yaw, degrees")

    ax_r1 = ax_l1.twinx()
    ax_r1.plot(t_domain / 86400, 1e-3 * a(
        t_domain,
        k=k,
        V_0=V_0,
        f=f,
        beta_0=beta_0_
    ), color='k', linestyle='dashed')
    ax_r1.set_ylabel("Semimajor axis, km (thousands)")

    _, ax_l2 = plt.subplots()
    ax_l2.set_xlabel("Time, days")

    ax_l2.plot(t_domain / 86400, V(
        t_domain,
        V_0=V_0,
        f=f,
        beta_0=beta_0_
    ), color='k', linestyle='solid')
    ax_l2.set_ylabel("Velocity, km/s")

    ax_r2 = ax_l2.twinx()
    ax_r2.plot(t_domain / 86400, np.degrees(inc_0 - delta_inc(
        t_domain,
        V_0=V_0,
        f=f,
        beta_0=beta_0_
    )), color='k', linestyle='solid')
    ax_r2.set_ylabel("Inclination, degrees")

    return ax_l1, ax_r1, ax_l2, ax_r2


if __name__ == '__main__':
    ax_l1, ax_r1, ax_l2, ax_r2 = plot_problem(inc_0=np.radians(28.5))
    ax_l1.set_ylim(20, 75)
    ax_r1.set_ylim(0, 45)
    ax_l2.set_ylim(3, 8)
    ax_r2.set_ylim(0, 30)
    plt.show()

    ax_l1, ax_r1, ax_l2, ax_r2 = plot_problem(inc_0=np.radians(90), t_f=330)
    ax_l1.set_ylim(10, 150)
    ax_r1.set_ylim(0, 200)
    ax_l2.set_ylim(1, 8)
    ax_r2.set_ylim(0, 90)
    plt.show()

    ax_l1, ax_r1, ax_l2, ax_r2 = plot_problem(inc_0=2, t_f=350)
    ax_l1.set_ylim(0, 180)
    ax_r1.set_ylim(0, 500)
    ax_l2.set_ylim(0, 8)
    ax_r2.set_ylim(-10, 120)
    plt.show()
