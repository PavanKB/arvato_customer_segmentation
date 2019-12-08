import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data_dist(data, col_idx, n_col=4, n_row=4):
    fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=[25, 20])

    n_plot = n_row * n_col
    plot_idx = [x for x in range(0, n_plot)]

    i = [x for x in col_idx]

    if len(i) > n_plot:
        print('more cols than plots')

    mapping = dict(zip(i[0:n_plot], plot_idx))

    for i in col_idx:
        c = data.columns[i]
        plot_data = data[[c]].copy(deep=True)  # doing this to avoid the error
        plot_data.fillna('NA', inplace=True)
        sns.countplot(x=c, data=plot_data, ax=axes.flatten()[mapping[i]])
    fig.show()
