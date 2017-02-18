import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt


def dadt_f2(a_a0, e_0):
    return 2 * a_a0 ** 1.5 * (1 - e_0**2 / a_a0**1.5)**.5


def dadt_fT(a_a0, e_0):
    return 2 * a_a0 ** 1.5 * (1 - .25 * e_0**2 / a_a0)


def main():
    # http://matplotlib.org/users/pgf.html#custom-preamble
    # http://sbillaudelle.de/2015/02/23/seamlessly-embedding-matplotlib-output-into-latex.html
    rc("pgf", rcfonts=False)
    rc("text", usetex=True)

    fig, ax = plt.subplots()

    a_a0_range = np.logspace(-1, 1)
    for e_0 in 0.02, 0.05, 0.07:
        ax.semilogx(a_a0_range, dadt_fT(a_a0_range, e_0) - dadt_f2(a_a0_range, e_0), color='k')

    # Manually place labels
    ax.text(0.3, 0.00045, fr"$e_0 = 0.02$", rotation=-1)
    ax.text(0.4, 0.0019, fr"$e_0 = 0.05$", rotation=-13)
    ax.text(0.6, 0.0032, fr"$e_0 = 0.07$", rotation=-30)

    ax.axvline(4, color='k', linewidth=1, linestyle='--')

    ax.set_xlabel(r"$a / a_0$")

    fig.savefig("burt_difference.pgf")


if __name__ == '__main__':
    main()
