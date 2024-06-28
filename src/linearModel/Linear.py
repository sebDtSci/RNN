from src.Utils.tensorialAlgebra import dot, random_tensor, Tensor
from src.baseArchi.Layer import Layer

class Linear(Layer):
    def _init_(self, input_dim:int, output_dim:int, init:str = 'Xavier')-> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = random_tensor(input_dim, output_dim, init=init)
        self.b = random_tensor(output_dim, init=init)
    
    def __str__(self) -> str:
        return f'Linear({self.input_dim}, {self.output_dim}, {self.w}, {self.b})'
    
    def forward(self, input:Tensor) -> Tensor:
        """
        enregistre l'entrée à utiliser dans la passe vers l'arrière
        """
        self.input = input
        return [dot(input, self.w[o]) + self.b[o] for o in range(self.output_dim)]
    
    def backward(self, grad:Tensor) -> Tensor:
        """
        On ajoute chaque b[o] sur chaque output[o] donc le gradient de b est le même que celui de sortie
        Input[i]*output[o]
        """
        self.grad_b = grad
        self.grad_w = [[self.input[i]*grad[o] for i in range(self.input_dim)] for o in range(self.output_dim)]
        return [sum(self.w[o][i] * grad[o] for o in range(self.output_dim)) for i in range(self.input_dim)]
    
    def params(self) -> list[Tensor]:
        return [self.w, self.b]
    
    def grads(self) -> list[Tensor]:
        return [self.grad_w, self.grad_b]
        