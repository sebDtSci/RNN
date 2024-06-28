from src.Utils.tensorialAlgebra import *

def test_tensor():
    assert shape(random_tensor(3, 4)) == [3, 4]
    assert tensor_shape([[1, 2], [3, 4]]) == [2, 2]
    