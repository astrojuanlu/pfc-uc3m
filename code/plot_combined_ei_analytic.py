"""Reproduces plots from original Pollard 2000 paper.

"""
import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt

from poliastro.util import circular_velocity

from combined_ei import beta, delta_V


def main():
    # http://matplotlib.org/users/pgf.html#custom-preamble
    # http://sbillaudelle.de/2015/02/23/seamlessly-embedding-matplotlib-output-into-latex.html
    rc("pgf", rcfonts=False)
    rc("text", usetex=True)

    k = 398600
    a = 42164
    inc_domain = np.radians(np.linspace(0, 30))

    fig1, ax = plt.subplots(figsize=(6, 6))

    eccentricities = 0.1, 0.2, 0.4, 0.6, 0.8
    beta_data = np.zeros((inc_domain.shape[0], len(eccentricities)))
    for ii, ecc in enumerate(eccentricities):
        beta_data[:, ii] = beta(ecc, 0.0, 0.0, inc_domain, 0.0)
        ax.plot(np.degrees(inc_domain), np.degrees(beta_data[:, ii]), label="$e = %.1f$" % ecc)

    ax.set_xlabel("Inclination change (deg)")
    ax.set_ylabel(r"Yaw angle $|\beta| (deg)$")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 90)
    ax.grid(True)
    ax.legend()

    fig1.savefig("combined_ei/chart_beta.pgf")

    fig2, ax = plt.subplots(figsize=(6, 6))

    eccentricities = 0.1, 0.2, 0.4, 0.6, 0.8
    delta_V_data = np.zeros((inc_domain.shape[0], len(eccentricities)))
    for ii, ecc in enumerate(eccentricities):
        delta_V_data[:, ii] = delta_V(circular_velocity(k, a), ecc, 0.0, beta_data[:, ii])
        ax.plot(np.degrees(inc_domain), delta_V_data[:, ii], label="$e = %.1f$" % ecc)

    ax.set_xlabel("Inclination change (deg)")
    ax.set_ylabel(r"$\Delta V$ (km/s)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 2.5)
    ax.grid(True)
    ax.legend()

    fig2.savefig("combined_ei/chart_dV.pgf")

    return fig1, fig2


if __name__ == '__main__':
    main()
    plt.show()
