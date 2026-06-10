import numpy as np
import pytest

from qres.regression import LinearRegression


def test_no_intercept():
    reg = LinearRegression()
    reg.fit(np.random.rand(10, 2), np.random.rand(10))
    assert len(reg.coeff) == 2
    assert reg.c == 0


def test_intercept():
    reg = LinearRegression()
    reg.fit(np.random.rand(10, 2), np.random.rand(10), fit_intercept=True)
    assert len(reg.coeff) == 2
    assert reg.c != 0
