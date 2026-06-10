import numpy as np
from qsim.lin_alg import Operator
from qsim.state import QuantumState

from qres.regression import LinearRegression


class QELM:

    def __init__(self, reservoir_func, heisenberg_func):
        self._res_func = reservoir_func
        self._heisenberg_func = heisenberg_func
        self._reg = LinearRegression()
        self._fitted = False
        self._condition_number = None

    @property
    def condition_number(self) -> float | None:
        return self._condition_number

    def map(self, state: QuantumState) -> list[float]:
        return self._res_func(state)

    def mapOperators(self, ops: list[Operator]) -> list[Operator]:
        return self._heisenberg_func(ops)

    def fit(
        self,
        x_train: list[QuantumState],
        y_train: np.ndarray[float],
        fit_intercept: bool = True,
        r: int | None = None,
    ) -> None:
        self._x_train = x_train
        self._d_train = np.array([self.map(state) for state in x_train])
        if fit_intercept:
            S = np.linalg.svd(
                np.hstack([self._d_train, np.ones((self._d_train.shape[0], 1))])
            ).S[:r]
            self._condition_number = S[0] / S[-1]
        else:
            S = np.linalg.svd(self._d_train).S[:r]
            self._condition_number = S[0] / S[-1]
        self._y_train = y_train
        self._reg.fit(self._d_train, y_train, fit_intercept, r)
        self._fitted = True

    def predict(self, x_test: list[QuantumState]) -> np.ndarray[float]:
        if not self._fitted:
            raise ValueError("Can't make predictions with an unfitted QELM")

        d_test = np.array([self.map(state) for state in x_test])
        return self._reg.predict(d_test)

    def score(
        self, x_test: list[QuantumState], y_test: np.ndarray[float]
    ) -> np.ndarray[float]:
        y_est = self.predict(x_test)

        if len(y_test.shape) == 1:
            y_test = y_test.reshape(y_test.shape[0], -1)
            y_est = y_est.reshape(y_est.shape[0], -1)

        mses = []
        for i in range(y_est.shape[1]):
            mses.append(np.mean((y_test[:, i] - y_est[:, i]) ** 2))
        return np.array(mses)

    def generateEffectiveOperators(self, ops: list[Operator]) -> list[Operator]:
        eff_ops = self.mapOperators(ops)

        total_ops = []

        for coeffs, c in zip(self._reg.coeff.T, self._reg.c):
            total_ops.append(
                sum([coeff * op for coeff, op in zip(coeffs, eff_ops)])
                + c * Operator(np.eye(eff_ops[0].dim))
            )

        return total_ops
