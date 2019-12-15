import pandas as pd
import numpy as np
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


def plot_data_count_comp(df_popl, df_cust, col_names, n_col=4, n_row=4, fig_size=(25, 20)):
    df_p = df_popl.loc[:, col_names]
    df_p['Type'] = 'Popl'
    df_c = df_cust.loc[:, col_names]
    df_c['Type'] = 'Cust'

    data = pd.concat([df_p, df_c], axis=0)

    n_plot = n_row * n_col
    plot_idx = [x for x in range(0, n_plot)]

    if len(col_names) > n_plot:
        print('more cols than plots')

    col_idx = [i for i in range(len(col_names))]

    mapping = dict(zip(col_idx[0:n_plot], plot_idx))

    fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)
    for i in col_idx[0:n_plot]:
        c = data.columns[i]
        plot_data = data.loc[:, [c, 'Type']]  # doing this to avoid the error
        plot_data.fillna('NA', inplace=True)
        sns.countplot(x=c, hue='Type', data=plot_data, ax=axes.flatten()[mapping[i]])
    fig.show()


def plot_data_dist_comp(df_popl, df_cust, col_names, n_col=4, n_row=4, fig_size=(25, 20)):
    # https://stackoverflow.com/a/53214367/6931113

    n_plot = n_row * n_col
    plot_idx = [x for x in range(0, n_plot)]

    if len(col_names) > n_plot:
        print('more cols than plots')

    col_idx = [i for i in range(len(col_names))]

    mapping = dict(zip(col_idx[0:n_plot], plot_idx))

    fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=fig_size)

    for i in col_idx:

        col = col_names[i]
        df_x = df_popl.loc[:, [col]]
        df_x['Src'] = 'Popl'
        df_y = df_cust.loc[:, [col]]
        df_y['Src'] = 'Cust'
        df_z = pd.concat([df_x, df_y], axis=0)

        df_z[col].groupby(df_z['Src']).value_counts(normalize=True).rename('Proportion').reset_index().pipe(
            (sns.barplot, "data"), x=col, y='Proportion', hue='Src', hue_order=['Popl', 'Cust'],
            ax=axes.flatten()[mapping[i]])


def plot_pca_expl_var(pca, expl_var=0.95):
    """
    Plots the pca var explanation curve
    :params pca: The fitted pca object
    :expl_var: How much of the var is to be explained.
    """
    cm_expl_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = (cm_expl_var <= expl_var).sum()

    fig = plt.figure()
    fig.suptitle(f'PCA {expl_var}, n={n_components}')
    plt.plot(cm_expl_var)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.axhline(y=expl_var)
    plt.axvline(x=n_components)


def plot_elbow(k_means):
    """
    Plot the elbow plot
    :params k_means: A list of the k_means model with different clusters
    """
    k_means_dist = pd.DataFrame({'Dist': [k.inertia_ / k.n_clusters for k in k_means],
                                'Clusters': [k.n_clusters for k in k_means]})

    sns.pointplot(x='Clusters', y="Dist", data=k_means_dist)
