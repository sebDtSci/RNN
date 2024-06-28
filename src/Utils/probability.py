import math
import random

Tensor = list
"""
Distribution Normale : Se réfère généralement à la fonction de densité de probabilité qui décrit comment les valeurs sont distribuées.
Répartition Normale : Se réfère plus spécifiquement à la fonction de répartition cumulative, qui décrit la probabilité cumulée jusqu'à une certaine valeur.
"""

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculate normal cdf (fonction de répartition de la loi normale)
    """
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p: float, mu: float = 0, sigma: float = 1, tolerance: float = 0.00001) -> float:
    """
    Trouver l'inverse approximatif de la fonction normal par une recherche dicotomique.
    """
    if mu!= 0 or sigma!= 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance = tolerance)
    low_z = -10.0
    hi_z = 10.0

    while hi_z - low_z > tolerance:
        
        mid_z = (low_z + hi_z) / 2 # considérer le point médian
        
        # moyenne
        mid_p = normal_cdf(mid_z) # la valeur de cdf ici !
    
        if mid_p < p:
            low_z = mid_z # poit median encore trp bas
        else:
            hi_z = mid_z # point median encore trp haut
    return mid_z

"""
Pour initialiser les couche de neurone avec des paramètres avec des valeurs aleatoires.
Ici cette génération répond à trois point pour optimiser l'apprentissage:
- choix dans une distribution uniforme aléatoire [0,1]
- choix dans une distribution gaussienne (normale0)
- choix par tirage au sort dans une distribution normal d'espérance 0 et de variance `2 / (num_inputs + num_outputs)`. -> technique Xavier
"""

def random_uniform(*dims:int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]
    
def random_normal(*dims:int, mean:float = 0.0, variance:float = 1.0) -> Tensor:
    if len(dims) == 1:
        # return [mean + variance * random.random() for _ in range(dims[0])] 
        return [mean + variance * inverse_normal_cdf(random.random()) for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance) for _ in range(dims[0])] 