from tqdm import tqdm

import numpy as np


class ResultsCallback:
    """Object to store results of an odeint integration.

    """
    def __init__(self):
        self._results = []

    def __call__(self, t, u_):
        self._results.append([t] + list(u_))

    def get_results(self):
        res = np.array(self._results)

        t_domain = res[:, 0]
        r_vectors = res[:, 1:4]
        v_vectors = res[:, 4:]

        return t_domain, r_vectors, v_vectors


class ProgressResultsCallback(ResultsCallback):
    def __init__(self, total_time):
        super().__init__()

        self._last_t = 0.0
        self._progress = tqdm(total=total_time, unit="s")

    def __enter__(self):
        self._progress.__enter__()
        return self

    def __call__(self, t, u_):
        super().__call__(t, u_)

        self._progress.update(t - self._last_t)
        self._last_t = t

    def __exit__(self, *args):
        self._progress.__exit__(*args)
