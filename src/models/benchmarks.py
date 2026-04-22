import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


class HAROLSVolatility:
    """HAR-OLS (Corsi 2009): OLS on 1d/5d/22d log-RV lags."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HAROLSVolatility":
        self._model = LinearRegression().fit(X, y)
        self.intercept_ = float(self._model.intercept_)
        self.coef_ = self._model.coef_.tolist()
        self.sigma_: float | None = None
        return self

    def calibrate_sigma(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        residuals = y_val - self._model.predict(X_val)
        self.sigma_ = float(np.std(residuals, ddof=0))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = self._model.predict(X)
        sigma = np.full_like(mu, fill_value=self.sigma_)
        return mu, sigma


class EWMAVolatility:
    """EWMA (RiskMetrics): sigma2_t = lam*sigma2_{t-1} + (1-lam)*r_t^2."""

    def __init__(self, lam: float = 0.94):
        self.lam = lam
        self.sigma_: float | None = None
        self._sigma2_last: float | None = None
        self._last_r: float | None = None

    def fit(self, r: np.ndarray) -> "EWMAVolatility":
        s2 = float(np.var(r, ddof=0))
        for rt in r:
            s2 = self.lam * s2 + (1.0 - self.lam) * rt ** 2
        self._sigma2_last = s2
        self._last_r = float(r[-1])
        return self

    def forecast_rolling(self, r: np.ndarray) -> np.ndarray:
        sigma2 = np.empty(len(r))
        s2 = self._sigma2_last
        for t, rt in enumerate(r):
            s2 = self.lam * s2 + (1.0 - self.lam) * rt ** 2
            sigma2[t] = s2
        self._sigma2_last = s2
        self._last_r = float(r[-1])
        return sigma2

    def calibrate_sigma(self, y_val: np.ndarray, mu_val: np.ndarray) -> None:
        self.sigma_ = float(np.std(y_val - mu_val, ddof=0))

    def predict_from_variance(self, sigma2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.log(sigma2 + 1e-12)
        sigma = np.full_like(mu, fill_value=self.sigma_)
        return mu, sigma


class GARCH11Volatility:
    """GARCH(1,1): sigma2_t = omega + alpha*r_{t-1}^2 + beta*sigma2_{t-1}."""

    def __init__(self):
        self.omega: float | None = None
        self.alpha: float | None = None
        self.beta: float | None = None
        self._converged = False
        self._nll: float | None = None
        self.sigma_: float | None = None
        self._sigma2_last: float | None = None
        self._last_r: float | None = None

    def _sigma2_series(self, r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
        n = len(r)
        s2 = np.empty(n)
        s2[0] = float(np.var(r, ddof=0))
        for t in range(1, n):
            s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        return s2

    def _nll_fn(self, params: np.ndarray, r: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        s2 = self._sigma2_series(r, omega, alpha, beta)
        if np.any(s2 <= 0):
            return 1e10
        return float(0.5 * np.sum(np.log(s2) + r ** 2 / s2))

    def fit(self, r: np.ndarray) -> "GARCH11Volatility":
        var_r = float(np.var(r, ddof=0))
        x0 = np.array([var_r * 0.05, 0.10, 0.85])
        bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999)]
        res = minimize(self._nll_fn, x0, args=(r,), method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 1000, "ftol": 1e-9})
        self.omega, self.alpha, self.beta = float(res.x[0]), float(res.x[1]), float(res.x[2])
        self._converged = bool(res.success)
        self._nll = float(res.fun)
        s2_train = self._sigma2_series(r, self.omega, self.alpha, self.beta)
        self._sigma2_last = float(s2_train[-1])
        self._last_r = float(r[-1])
        return self

    def forecast_rolling(self, r: np.ndarray) -> np.ndarray:
        sigma2 = np.empty(len(r))
        s2, r_prev = self._sigma2_last, self._last_r
        for t, rt in enumerate(r):
            s2 = self.omega + self.alpha * r_prev ** 2 + self.beta * s2
            sigma2[t] = s2
            r_prev = rt
        self._sigma2_last = s2
        self._last_r = float(r[-1])
        return sigma2

    def calibrate_sigma(self, y_val: np.ndarray, mu_val: np.ndarray) -> None:
        self.sigma_ = float(np.std(y_val - mu_val, ddof=0))

    def predict_from_variance(self, sigma2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.log(sigma2 + 1e-12)
        sigma = np.full_like(mu, fill_value=self.sigma_)
        return mu, sigma


class GJRGARCH11Volatility:
    """GJR-GARCH(1,1): adds gamma*r_{t-1}^2*I(r<0) for asymmetric leverage."""

    def __init__(self):
        self.omega: float | None = None
        self.alpha: float | None = None
        self.gamma: float | None = None
        self.beta: float | None = None
        self._converged = False
        self._nll: float | None = None
        self.sigma_: float | None = None
        self._sigma2_last: float | None = None
        self._last_r: float | None = None

    def _sigma2_series(self, r: np.ndarray, omega: float, alpha: float,
                       gamma: float, beta: float) -> np.ndarray:
        n = len(r)
        s2 = np.empty(n)
        s2[0] = float(np.var(r, ddof=0))
        for t in range(1, n):
            ind = 1.0 if r[t - 1] < 0 else 0.0
            s2[t] = omega + (alpha + gamma * ind) * r[t - 1] ** 2 + beta * s2[t - 1]
        return s2

    def _nll_fn(self, params: np.ndarray, r: np.ndarray) -> float:
        omega, alpha, gamma, beta = params
        if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
            return 1e10
        if alpha + beta + 0.5 * gamma >= 1:
            return 1e10
        s2 = self._sigma2_series(r, omega, alpha, gamma, beta)
        if np.any(s2 <= 0):
            return 1e10
        return float(0.5 * np.sum(np.log(s2) + r ** 2 / s2))

    def fit(self, r: np.ndarray) -> "GJRGARCH11Volatility":
        var_r = float(np.var(r, ddof=0))
        bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999), (1e-8, 0.999)]
        starting_points = [
            [var_r * 0.05, 0.05,  0.10, 0.85],
            [var_r * 0.10, 0.08,  0.06, 0.82],
            [var_r * 0.02, 0.03,  0.15, 0.80],
            [var_r * 0.01, 0.10,  0.05, 0.75],
            [var_r * 0.08, 0.12,  0.08, 0.78],
        ]
        best_res = None
        for x0 in starting_points:
            try:
                res = minimize(self._nll_fn, x0, args=(r,), method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": 2000, "ftol": 1e-10})
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            except Exception:
                continue
        self.omega  = float(best_res.x[0])
        self.alpha  = float(best_res.x[1])
        self.gamma  = float(best_res.x[2])
        self.beta   = float(best_res.x[3])
        self._converged = bool(best_res.success)
        self._nll = float(best_res.fun)
        s2_train = self._sigma2_series(r, self.omega, self.alpha, self.gamma, self.beta)
        self._sigma2_last = float(s2_train[-1])
        self._last_r = float(r[-1])
        return self

    def forecast_rolling(self, r: np.ndarray) -> np.ndarray:
        sigma2 = np.empty(len(r))
        s2, r_prev = self._sigma2_last, self._last_r
        for t, rt in enumerate(r):
            ind = 1.0 if r_prev < 0 else 0.0
            s2 = self.omega + (self.alpha + self.gamma * ind) * r_prev ** 2 + self.beta * s2
            sigma2[t] = s2
            r_prev = rt
        self._sigma2_last = s2
        self._last_r = float(r[-1])
        return sigma2

    def calibrate_sigma(self, y_val: np.ndarray, mu_val: np.ndarray) -> None:
        self.sigma_ = float(np.std(y_val - mu_val, ddof=0))

    def predict_from_variance(self, sigma2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.log(sigma2 + 1e-12)
        sigma = np.full_like(mu, fill_value=self.sigma_)
        return mu, sigma
