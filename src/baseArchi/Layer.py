Tensor = list

class Layer:
    """
    Pas de typage sur les variables d'entré, on reste le plus souple possible pour prendre en charge un maximum d'architectures.
    """
    def forward(self, input):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError
    def params(self) -> list[Tensor]:
        return ()
    def grads(self) -> list[Tensor]:
        return()