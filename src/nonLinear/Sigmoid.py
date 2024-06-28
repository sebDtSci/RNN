from src.neural_network_utils.activation import sigmoid
from src.baseArchi.Layer import Layer

Tensor = list

class Sigmoid(Layer):
    def forward(self, input:Tensor) -> Tensor:
        self.sigmoids = tensor_apply(sigmoid, input)
        return self.sigmoids
    
    def backward(self, grad:Tensor) -> Tensor:
        return tensor_combine(lambda sigmoids, grad: sigmoids * (1 - sigmoids) * grad, self.sigmoids, grad)