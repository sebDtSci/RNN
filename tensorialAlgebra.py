import random 
Tensor = list

def random_uniform(*dims:int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]
    
def ramdom_normal(*dims:int, mean:float = 0.0, variance:float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * random.random() for _ in range(dims[0])] ................


def random_tensor(*dims:int, init:str = 'normal') -> Tensor:
    if init == 'normal':
        return ra
    
    