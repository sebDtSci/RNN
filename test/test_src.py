from src.Utils.tensorialAlgebra import *

def test_tensor():
    assert shape([[1, 2], [3, 4]]) == (2, 2)
    # assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]
    assert shape(random_normal(2, 3, mean=10)) == (2,3)
    assert tensor_shape([[1, 2], [3, 4]]) == [2, 2]
