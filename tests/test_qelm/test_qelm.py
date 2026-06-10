import math

import numpy as np
import pytest
from qsim.dynamics import Dynamics, ExponentialPropagator, HamiltonianGenerator
from qsim.lin_alg import I, Operator, sigmaMinus, sigmaPlus, sigmaZ
from qsim.state import DensityMatrix, QuantumState

from qres.qelm import QELM


@pytest.fixture
def trial_reservoir():
    H = HamiltonianGenerator((sigmaPlus ^ sigmaMinus) + (sigmaMinus ^ sigmaPlus))
    prop = ExponentialPropagator()
    dynam = Dynamics(prop, H)

    def reservoir_func(state: DensityMatrix):
        updated_state = dynam.evolve(
            state ^ DensityMatrix(np.array([[1, 0], [0, 0]])), ts=[math.pi / 2]
        )
        return np.array([(updated_state @ (I(2) ^ sigmaZ)).trace()])

    def heisenberg_func(op: Operator):
        updated_op = dynam.evolveOperator(I(2) ^ op, ts=[0], t0=math.pi / 2)
        return (
            (
                DensityMatrix(np.array([[1, 0], [0, 1]]))
                ^ DensityMatrix(np.array([[1, 0], [0, 0]]))
            )
            @ updated_op
        ).partialTrace((2, 2), (0,))

    return QELM(reservoir_func, heisenberg_func)


@pytest.fixture
def fitted_reservoir(trial_reservoir):
    trial_reservoir.fit(
        [
            DensityMatrix(np.array([[0, 0], [0, 1]])),
            DensityMatrix(np.array([[1, 0], [0, 0]])),
        ],
        y_train=np.array([-1, 1]),
    )
    return trial_reservoir


def test_qelm_map_state(trial_reservoir):
    output = trial_reservoir.map(DensityMatrix(np.array([[0, 0], [0, 1]])))
    assert output == -1


def test_qelm_heisenberg(trial_reservoir):
    output = trial_reservoir.mapOperators(sigmaZ)
    assert pytest.approx(output.matrix) == sigmaZ.matrix


def test_train(fitted_reservoir):

    assert fitted_reservoir._reg is not None
    assert pytest.approx(fitted_reservoir._reg.coeff) == np.array([1])
    assert pytest.approx(fitted_reservoir._reg.c) == 0


def test_score(fitted_reservoir):
    assert fitted_reservoir.score(
        [DensityMatrix(np.array([[0.5, 0], [0, 0.5]]))], np.array([0])
    ) == pytest.approx(np.array([0]))


def test_condition_number(fitted_reservoir):
    assert fitted_reservoir.condition_number is not None
