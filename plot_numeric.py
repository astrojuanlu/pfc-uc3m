import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from poliastro.twobody.propagation import cowell
from poliastro.twobody.rv import RVState

from edelbaum import guidance_law, extra_quantities


def _compute_results_array(a_0, a_f, inc_0, i_f, f):
    k = Earth.k.decompose([u.km, u.s]).value

    edelbaum_accel = guidance_law(k, a_0, a_f, inc_0, i_f, f)
    _, t_f = extra_quantities(k, a_0, a_f, inc_0, i_f, f)

    # Retrieve r and v from initial orbit
    s0 = Orbit.circular(Earth, a_0 * u.km - Earth.R, inc_0 * u.rad)
    r0, v0 = s0.rv()

    results = []

    def callback(pb):
        # https://pypi.python.org/pypi/tqdm#hooks-and-callbacks
        last_t = 0.0

        def inner(t, u_):
            nonlocal last_t
            results.append([t] + list(u_))
            pb.update(t - last_t)
            last_t = t

        return inner

    # Propagate orbit
    from tqdm import tqdm
    with tqdm(total=t_f, unit="s") as progress:
        cowell(
            k, r0.to(u.km).value, v0.to(u.km / u.s).value, t_f,
            ad=edelbaum_accel,
            callback=callback(progress),
            nsteps=1000000
        )

    results_np = np.array(results)  # ~70 k rows, 7 columns, 3 MB in memory

    t_domain = results_np[:, 0]
    r_vectors = results_np[:, 1:4]
    v_vectors = results_np[:, 4:]

    return t_domain, r_vectors, v_vectors


def _extract_arrays(t_domain, r_vectors, v_vectors):
    a_values = np.zeros_like(t_domain)
    inc_values = np.zeros_like(t_domain)
    v_values = norm(v_vectors, axis=1)
    for ii in range(len(t_domain)):
        r = r_vectors[ii]
        v = v_vectors[ii]
        if r.any():
            ss = RVState(Earth, r * u.km, v * u.km / u.s)
            a_values[ii] = ss.a.to(u.km).value
            inc_values[ii] = ss.inc.to(u.rad).value

    return a_values, inc_values, v_values


def _plot_quantities(t_domain, a_values, inc_values, v_values):
    # TODO: Plotting 70k rows is extremely slow, consider subsampling
    _, ax_r1 = plt.subplots()
    ax_r1.set_xlabel("Time, days")

    ax_r1.plot(t_domain / 86400, 1e-3 * a_values, color='k', linestyle='dashed')
    ax_r1.set_ylabel("Semimajor axis, km (thousands)")

    _, ax_l2 = plt.subplots()
    ax_l2.set_xlabel("Time, days")

    ax_l2.plot(t_domain / 86400, v_values, color='k', linestyle='solid')
    ax_l2.set_ylabel("Velocity, km/s")

    ax_r2 = ax_l2.twinx()
    ax_r2.plot(t_domain / 86400, np.degrees(inc_values), color='k', linestyle='solid')
    ax_r2.set_ylabel("Inclination, degrees")

    return ax_r1, ax_r1, ax_l2, ax_r2


def plot_edelbaum_case(inc_0):
    a_0 = 7000.0  # km
    a_f = 42166.0  # km
    i_f = 0.0  # deg
    f = 3.5e-7  # km / s2

    t_domain, r_vectors, v_vectors = _compute_results_array(a_0, a_f, inc_0, i_f, f)
    a_values, inc_values, v_values = _extract_arrays(t_domain, r_vectors, v_vectors)
    _plot_quantities(t_domain, a_values, inc_values, v_values)

    plt.show()


if __name__ == '__main__':
    plot_edelbaum_case(np.radians(90.0))
