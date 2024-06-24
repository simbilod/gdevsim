import numpy as np

from gdevsim.utils.operations import identity, signed_log


def test_identity():
    data = np.array([1, 2, 3])
    assert np.array_equal(identity(data), data)

def test_signed_log():
    data = np.array([-100, 0, 100])
    expected = np.array([-2, np.nan, 2])
    result = signed_log(data)
    assert np.allclose(result, expected, equal_nan=True)
