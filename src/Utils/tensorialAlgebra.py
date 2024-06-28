# -*- coding: utf-8 -*-
# Version 0.0.1
# Copyright © 2024 Tadiello Stadiello.
# All rights reserved.

from src.Utils.probability import inverse_normal_cdf, normal_cdf, random_uniform, random_normal

Matrix = list[list[float]]
Tensor = list
Vector = list[float]

# Vecteur
def dot(vect1:Vector, vect2:Vector) -> float:
    assert len(vect1) == len(vect2), f"Vecteurs de tailles différentes : {len(vect1)} et {len(vect2)}"
    return sum(vect1_i * vect2_i for vect1_i, vect2_i in zip(vect1, vect2))

# Matrice
def shape(M: Matrix) -> tuple[int,int]:
    numRows = len(M)
    numsColumns = len(M[0]) if M else 0
    return (numRows, numsColumns)

def get_row(M: Matrix, i:int) -> list[float]:
    return M[i]

def get_column(M: Matrix, j:int) -> list[float]:
    return [M_i[j] for M_i in M] # elem j de la ligne i pour ligne de M_i

# Tensor

def tensor_shape(tensor:Tensor) -> list[int]:
    sizes:list[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes

def random_tensor(*dims:int, init:str = 'normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    # elif init == 'Xavier':
    #     variance = len(dims)/sum(dims)
    #     return random_normal(*dims, variance=variance)
    elif init == 'Xavier':
        if len(dims) == 2:
            variance = 2 / (dims[0] + dims[1])
        elif len(dims) == 1:
            variance = 1 / dims[0]  # Ajustement pour éviter les petites valeurs
        else:
            variance = 2 / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError('init must be normal, uniform or Xavier')

# def tensor_apply(f:callable[[float],[float]], tensor:Tensor) -> Tensor:
    