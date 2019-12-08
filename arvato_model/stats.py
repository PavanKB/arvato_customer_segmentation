import numpy as np
from collections import Counter


def calc_proportion(x):
    """
    Calculates the proportion for each unique value in list
    :params x: A list of values
    """
    p = Counter(x)
    p = {key: (float(val) / len(x)) for key, val in p.items()}

    return p


def simpson_diversity_index(x):
    """
    Calculates the Simpson's diversity index
    http://www.tiem.utk.edu/~gross/bioed/bealsmodules/simpsonDI.html
    :params x: A list of values
    """

    p = calc_proportion(x)
    p = list(p.values())

    d = [i**2 for i in p]
    d = 1/sum(d)

    return d


def shannon_diversity_index(x):
    """
    Calculates the Shannon's diversity index
    http://www.tiem.utk.edu/~gross/bioed/bealsmodules/shannonDI.html
    :params x: A list of values
    """
    p = calc_proportion(x)
    p = list(p.values())

    h = [i * np.log(i) for i in p]
    h = -sum(h)

    return h
