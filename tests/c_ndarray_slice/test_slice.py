from slice.c_ndarray_slice import c_ndarray_slice
import numpy as np


x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

def test_slice_axis_0():
    y1 = c_ndarray_slice(x, 0, 1)
    assert (y1 == x[0:1, :]).all()


def test_slice_axis_0_slice_to_end_of_axis():
    y2 = c_ndarray_slice(x, 1, 5)
    assert (y2 == x[1:5, :]).all()
