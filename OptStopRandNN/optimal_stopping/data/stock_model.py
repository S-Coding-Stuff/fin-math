"""Underlying stochastic models used by OptStopRandNN."""

import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
try:
    from fbm import FBM
except ModuleNotFoundError:  # pragma: no cover - optional for non-fBM runs
    FBM = None

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
from engines.random_numbers import SobolNormalGenerator

NB_JOBS_PATH_GEN = 1
PATH_SAMPLERS = (
    "mc",
    "mc_antithetic",
    "sobol",
    "sobol_seq",
    "sobol_bb",
    "sobol_scrambled",
    "sobol_scrambled_seq",
    "sobol_scrambled_bb",
)

def _apply_brownian_bridge(normals: np.ndarray, maturity: float) -> np.ndarray:
    """Apply Brownian bridge construction along the time axis.

    normals is expected to have shape (nb_paths, nb_stocks, nb_dates).
    """
    nb_paths, nb_stocks, nb_dates = normals.shape
    if nb_dates <= 1:
        return normals

    dt = maturity / nb_dates
    times = np.linspace(0.0, maturity, nb_dates + 1)
    flat_normals = normals.transpose(2, 0, 1).reshape(nb_dates, nb_paths * nb_stocks)
    bridged = np.empty_like(flat_normals)

    for path_idx in range(flat_normals.shape[1]):
        path = np.zeros(nb_dates + 1, dtype=float)
        path[0] = 0.0
        path[-1] = np.sqrt(maturity) * flat_normals[0, path_idx]
        path_flags = np.zeros(nb_dates + 1, dtype=bool)
        path_flags[0] = True
        path_flags[-1] = True

        def fill(left: int, right: int, cursor: int) -> int:
            if right - left <= 1 or cursor >= nb_dates:
                return cursor
            mid = (left + right) // 2
            if path_flags[mid]:
                cursor = fill(left, mid, cursor)
                return fill(mid, right, cursor)

            t_left, t_mid, t_right = times[left], times[mid], times[right]
            weight_left = (t_right - t_mid) / (t_right - t_left)
            weight_right = (t_mid - t_left) / (t_right - t_left)
            variance = (t_mid - t_left) * (t_right - t_mid) / (t_right - t_left)

            path[mid] = weight_left * path[left] + weight_right * path[right]
            if variance > 0.0:
                path[mid] += np.sqrt(variance) * flat_normals[cursor, path_idx]
            path_flags[mid] = True
            cursor += 1

            cursor = fill(left, mid, cursor)
            return fill(mid, right, cursor)

        fill(0, nb_dates, 1)
        bridged[:, path_idx] = np.diff(path) / np.sqrt(dt)

    return bridged.reshape(nb_dates, nb_paths, nb_stocks).transpose(1, 2, 0)

class Model:
    def __init__(
        self,
        drift,
        dividend,
        volatility,
        spot,
        nb_stocks,
        nb_paths,
        nb_dates,
        maturity,
        name,
        path_sampler="mc",
        sampler_seed=None,
        **keywords,
    ):
        del keywords
        if path_sampler not in PATH_SAMPLERS:
            raise ValueError(
                f"path_sampler must be one of {PATH_SAMPLERS}, received {path_sampler!r}."
            )
        self.name = name
        self.drift = drift - dividend
        self.rate = drift
        self.dividend = dividend
        self.volatility = volatility
        self.spot = spot
        self.nb_stocks = nb_stocks
        self.nb_paths = nb_paths
        self.nb_dates = nb_dates
        self.maturity = maturity
        self.dt = self.maturity / self.nb_dates
        self.df = math.exp(-self.rate * self.dt)
        self.return_var = False
        self.path_sampler = path_sampler
        self.sampler_seed = sampler_seed

    def disc_factor(self, date_begin, date_end):
        time = (date_end - date_begin) * self.dt
        return math.exp(-self.drift * time)

    def drift_fct(self, x, t):
        raise NotImplemented()

    def diffusion_fct(self, x, t, v=0):
        raise NotImplemented()

    def generate_one_path(self):
        raise NotImplemented()

    def generate_paths(self, nb_paths=None):
        """Returns a nparray (nb_paths * nb_stocks * nb_dates+1) with prices."""
        nb_paths = nb_paths or self.nb_paths
        if NB_JOBS_PATH_GEN > 1:
            return np.array(
                joblib.Parallel(n_jobs=NB_JOBS_PATH_GEN, prefer="threads")(
                    joblib.delayed(self.generate_one_path)() for _ in range(nb_paths)
                )
            ), None
        return np.array([self.generate_one_path() for _ in range(nb_paths)]), None

class BlackScholes(Model):
    def __init__(
        self,
        drift,
        volatility,
        nb_paths,
        nb_stocks,
        nb_dates,
        spot,
        maturity,
        dividend=0,
        path_sampler="mc",
        sampler_seed=None,
        **keywords,
    ):
        super(BlackScholes, self).__init__(
            drift=drift,
            dividend=dividend,
            volatility=volatility,
            nb_stocks=nb_stocks,
            nb_paths=nb_paths,
            nb_dates=nb_dates,
            spot=spot,
            maturity=maturity,
            name="BlackScholes",
            path_sampler=path_sampler,
            sampler_seed=sampler_seed,
            **keywords,
        )

    def drift_fct(self, x, t):
        del t
        return self.drift * x

    def diffusion_fct(self, x, t, v=0):
        del t
        del v
        return self.volatility * x

    def _draw_standard_normals(self, nb_paths: int, nb_dates: int) -> np.ndarray:
        if self.path_sampler == "mc":
            return np.random.normal(0.0, 1.0, (nb_paths, self.nb_stocks, nb_dates))
        if self.path_sampler == "mc_antithetic":
            count = (nb_paths + 1) // 2
            normals = np.random.normal(0.0, 1.0, (count, self.nb_stocks, nb_dates))
            return np.concatenate((normals, -normals), axis=0)[:nb_paths]

        dimension = self.nb_stocks * nb_dates
        scrambled = self.path_sampler in (
            "sobol_scrambled",
            "sobol_scrambled_seq",
            "sobol_scrambled_bb",
        )
        generator = SobolNormalGenerator(
            dimension=dimension,
            scramble=scrambled,
            seed=self.sampler_seed,
        )
        normals = generator.standard_normal((dimension, nb_paths)).T
        normals = normals.reshape(nb_paths, self.nb_stocks, nb_dates)
        if self.path_sampler in ("sobol_bb", "sobol_scrambled_bb"):
            normals = _apply_brownian_bridge(normals, maturity=self.maturity)
        return normals

    def generate_paths(self, nb_paths=None, return_dW=False, dW=None, X0=None, nb_dates=None):
        """Returns a nparray (nb_paths * nb_stocks * nb_dates) with prices."""
        nb_paths = nb_paths or self.nb_paths
        nb_dates = nb_dates or self.nb_dates
        spot_paths = np.empty((nb_paths, self.nb_stocks, nb_dates + 1))
        if X0 is None:
            spot_paths[:, :, 0] = self.spot
        else:
            spot_paths[:, :, 0] = X0
        if dW is None:
            random_numbers = self._draw_standard_normals(nb_paths=nb_paths, nb_dates=nb_dates)
            dW = random_numbers * np.sqrt(self.dt)
        drift = self.drift
        r = np.repeat(
            np.repeat(np.repeat(np.reshape(drift, (-1, 1, 1)), nb_paths, axis=0), self.nb_stocks, axis=1),
            nb_dates,
            axis=2,
        )
        sig = np.repeat(
            np.repeat(
                np.repeat(np.reshape(self.volatility, (-1, 1, 1)), nb_paths, axis=0),
                self.nb_stocks,
                axis=1,
            ),
            nb_dates,
            axis=2,
        )
        spot_paths[:, :, 1:] = np.repeat(spot_paths[:, :, 0:1], nb_dates, axis=2) * np.exp(
            np.cumsum(r * self.dt - (sig ** 2) * self.dt / 2 + sig * dW, axis=2)
        )
        if return_dW:
            return spot_paths, None, dW
        return spot_paths, None

    def generate_paths_with_alternatives(self, nb_paths=None, nb_alternatives=1, nb_dates=None):
        """Returns a nparray (nb_paths * nb_stocks * nb_dates) with prices."""
        nb_paths = nb_paths or self.nb_paths
        nb_dates = nb_dates or self.nb_dates
        total_nb_paths = nb_paths + nb_paths * nb_alternatives * nb_dates
        spot_paths = np.empty((total_nb_paths, self.nb_stocks, nb_dates + 1))
        spot_paths[:, :, 0] = self.spot
        random_numbers = self._draw_standard_normals(nb_paths=total_nb_paths, nb_dates=nb_dates)
        mult = nb_alternatives * nb_paths
        for i in range(nb_dates - 1):
            random_numbers[
                nb_paths + i * mult : nb_paths + (i + 1) * mult, :, : nb_dates - i - 1
            ] = np.tile(random_numbers[:nb_paths, :, : nb_dates - i - 1], reps=(nb_alternatives, 1, 1))
        dW = random_numbers * np.sqrt(self.dt)
        drift = self.drift
        r = np.repeat(
            np.repeat(
                np.repeat(np.reshape(drift, (-1, 1, 1)), total_nb_paths, axis=0),
                self.nb_stocks,
                axis=1,
            ),
            nb_dates,
            axis=2,
        )
        sig = np.repeat(
            np.repeat(
                np.repeat(np.reshape(self.volatility, (-1, 1, 1)), total_nb_paths, axis=0),
                self.nb_stocks,
                axis=1,
            ),
            nb_dates,
            axis=2,
        )
        spot_paths[:, :, 1:] = np.repeat(spot_paths[:, :, 0:1], nb_dates, axis=2) * np.exp(
            np.cumsum(r * self.dt - (sig ** 2) * self.dt / 2 + sig * dW, axis=2)
        )
        return spot_paths, None

class FractionalBlackScholes(Model):
    def __init__(
        self,
        drift,
        volatility,
        hurst,
        nb_paths,
        nb_stocks,
        nb_dates,
        spot,
        maturity,
        dividend=0,
        **keywords,
    ):
        super(FractionalBlackScholes, self).__init__(
            drift=drift,
            dividend=dividend,
            volatility=volatility,
            nb_stocks=nb_stocks,
            nb_paths=nb_paths,
            nb_dates=nb_dates,
            spot=spot,
            maturity=maturity,
            name="FractionalBlackScholes",
            **keywords,
        )
        if FBM is None:
            raise ImportError("FractionalBlackScholes requires the optional 'fbm' package.")
        self.hurst = hurst
        self.fBM = FBM(n=nb_dates, hurst=self.hurst, length=maturity, method="cholesky")

    def drift_fct(self, x, t):
        del t
        return self.drift * x

    def diffusion_fct(self, x, t, v=0):
        del t
        del v
        return self.volatility * x

    def generate_one_path(self):
        """Returns a nparray (nb_stocks * nb_dates) with prices."""
        path = np.empty((self.nb_stocks, self.nb_dates + 1))
        fracbm_noise = np.empty((self.nb_stocks, self.nb_dates))
        path[:, 0] = self.spot
        for stock in range(self.nb_stocks):
            fracbm_noise[stock, :] = self.fBM.fgn()
        for k in range(1, self.nb_dates + 1):
            previous_spots = path[:, k - 1]
            diffusion = self.diffusion_fct(previous_spots, k * self.dt)
            path[:, k] = (
                previous_spots
                + self.drift_fct(previous_spots, k * self.dt) * self.dt
                + np.multiply(diffusion, fracbm_noise[:, k - 1])
            )
        return path


class FBMH1:
    """fractional Brownian Motion for hurst H=1"""

    def __init__(self, n, length):
        self.n = n
        self.length = length

    def fbm(self):
        return np.linspace(0, self.length, self.n + 1) * np.random.randn(1)

class FractionalBrownianMotion(Model):
    def __init__(
        self,
        drift,
        volatility,
        hurst,
        nb_paths,
        nb_stocks,
        nb_dates,
        spot,
        maturity,
        dividend=0,
        **keywords,
    ):
        super(FractionalBrownianMotion, self).__init__(
            drift=drift,
            dividend=dividend,
            volatility=volatility,
            nb_stocks=nb_stocks,
            nb_paths=nb_paths,
            nb_dates=nb_dates,
            spot=spot,
            maturity=maturity,
            name="FractionalBrownianMotion",
            **keywords,
        )
        self.hurst = hurst
        if self.hurst == 1:
            self.fBM = FBMH1(n=nb_dates, length=maturity)
        else:
            if FBM is None:
                raise ImportError("FractionalBrownianMotion requires the optional 'fbm' package.")
            self.fBM = FBM(n=nb_dates, hurst=hurst, length=maturity, method="cholesky")
        self._nb_stocks = self.nb_stocks

    def _generate_one_path(self):
        """Returns a nparray (nb_stocks * nb_dates) with prices."""
        path = np.empty((self._nb_stocks, self.nb_dates + 1))
        for stock in range(self._nb_stocks):
            path[stock, :] = self.fBM.fbm() + self.spot
        return path

    def generate_one_path(self):
        return self._generate_one_path()

class FractionalBrownianMotionPathDep(FractionalBrownianMotion):
    def __init__(
        self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot, maturity, dividend=0, **keywords
    ):
        assert nb_stocks == 1
        assert spot == 0
        super(FractionalBrownianMotionPathDep, self).__init__(
            drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot, maturity, dividend=0, **keywords
        )
        self.nb_stocks = nb_dates + 1
        self._nb_stocks = 1

    def generate_one_path(self):
        """Returns a nparray (nb_stocks * nb_dates) with prices."""
        path_raw = self._generate_one_path()
        path = np.zeros((self.nb_stocks, self.nb_dates + 1))
        for i in range(self.nb_dates + 1):
            path[: i + 1, i] = np.flip(path_raw[0, : i + 1])
        return path, None

STOCK_MODELS = {
    "BlackScholes": BlackScholes,
    "FractionalBlackScholes": FractionalBlackScholes,
    "FractionalBrownianMotion": FractionalBrownianMotion,
    "FractionalBrownianMotionPathDep": FractionalBrownianMotionPathDep,
}

hyperparam_test_stock_models = {
    "drift": 0.2,
    "volatility": 0.3,
    "mean": 0.5,
    "speed": 0.5,
    "hurst": 0.05,
    "correlation": 0.5,
    "nb_paths": 1,
    "nb_dates": 100,
    "maturity": 1.0,
    "nb_stocks": 10,
    "spot": 100,
}

def draw_stock_model(stock_model_name):
    hyperparam_test_stock_models["model_name"] = stock_model_name
    stockmodel = STOCK_MODELS[stock_model_name](**hyperparam_test_stock_models)
    stock_paths, _ = stockmodel.generate_paths()
    filename = "{}.pdf".format(stock_model_name)

    one_path = stock_paths[0, 0, :]
    dates = np.array([i for i in range(len(one_path))])
    plt.plot(dates, one_path, label="stock path")
    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    draw_stock_model("BlackScholes")
