import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import chi2_contingency


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


def calc_chi2_stats(df_popl, df_cust, attributes, alpha=0.05, exclude_na=True, use_perc=True):
    """
    Calculates the chi-sq stays for the attribute to compare the
    similarities between population and customer
    ref: https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
    :params df_popl: The popl df
    :params df_popl: The cust df
    :params attributes: List of attributes to compare
    :params alpha: The
    :return: The stats as tuple
    """

    if not isinstance(attributes, list):
        attributes = [attributes]

    stats = {}
    for attribute in attributes:

        if exclude_na:
            expected = df_popl.loc[:, attribute].dropna()
            observed = df_cust.loc[:, attribute].dropna()
        else:
            expected = df_popl.loc[:, attribute]
            observed = df_cust.loc[:, attribute]

        if use_perc:
            expected = expected.value_counts(normalize=True) * 100
            observed = observed.value_counts(normalize=True) * 100
        else:
            expected = expected.value_counts(normalize=False)
            observed = observed.value_counts(normalize=False)

        contingency_table = pd.DataFrame.from_dict({'Popl': expected,
                                                    'Cust': observed},
                                                   orient='index').values

        stat, p, dof, _ = chi2_contingency(contingency_table)
        stats[attribute] = {'stat': stat, 'p': p, 'dof': dof, 'similar': p > alpha}

    return pd.DataFrame.from_dict(stats, orient='index')

