import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


def get_na_summary(df, axis=0):
    """
    Calculates the na values in the dataframe and returns the summary.
    axis
    :params df: The dataframe to analyse
    :params axis:  0 for columns, 1 for rows
    :return: summary as dataframe
    """
    na_count = df.isna().sum(axis=axis).sort_values(ascending=False)

    n_tot = df.shape[0] if axis == 0 else df.shape[1]

    na_perc = na_count / n_tot

    return pd.DataFrame({'na_count': na_count,
                         'na_perc': na_perc})


def read_value_meta_data(attr_file_path='data/DIAS Attributes - Values 2017.xlsx',
                         attr_sheet_name='Tabelle1',
                         info_file_path='data/DIAS Information Levels - Attributes 2017.xlsx',
                         info_sheet_name='Komplett'
                         ):
    """
    Read the meta data file
    :params file_path: The path of the excel file with meta data
    :params sheet_name: Sheet to read
    :return: The meta data as the data frame. All columns are read as string.
    """
    meta_data = pd.read_excel(attr_file_path, sheet_name=attr_sheet_name, header=1, dtype=str).iloc[:, 1:]
    meta_data.fillna(method='ffill', inplace=True)

    # We only need the information level and the attribute
    info_data = pd.read_excel(info_file_path, sheet_name=info_sheet_name, header=1, dtype=str).iloc[:, 1:3]
    info_data.fillna(method='ffill', inplace=True)

    # Bug fixes since the excel file is not properly formatted.

    # Tag to Household
    untagged_attrb = ['D19_BANKEN_ANZ_12', 'D19_BANKEN_ANZ_24', 'D19_GESAMT_ANZ_12', 'D19_GESAMT_ANZ_24',
                      'D19_TELKO_ANZ_12', 'D19_TELKO_ANZ_24', 'D19_VERSAND_ANZ_12', 'D19_VERSAND_ANZ_24',
                      'D19_VERSI_ANZ_12', 'D19_VERSI_ANZ_24']

    info_data = pd.concat([info_data,
                           pd.DataFrame({'Information level': ['Household'] * len(untagged_attrb),
                                         'Attribute': untagged_attrb}
                                        )
                           ])
    # Clear up the malformed values from the meta data
    attr_idx = info_data.loc[:, 'Attribute'].isin(
        ['D19_GESAMT_ANZ_12                                    D19_GESAMT_ANZ_24',
         'D19_BANKEN_ ANZ_12             D19_BANKEN_ ANZ_24',
         'D19_TELKO_ ANZ_12                  D19_TELKO_ ANZ_24',
         'D19_VERSI_ ANZ_12                                       D19_VERSI_ ANZ_24',
         'D19_VERSAND_ ANZ_12          D19_VERSAND_ ANZ_24']
    )
    info_data.drop(info_data.loc[attr_idx, :].index, inplace=True)

    # Tag to PLZ8
    untagged_attrb = ['KBA13_CCM_3000', 'KBA13_CCM_3001']
    info_data = pd.concat([info_data,
                           pd.DataFrame({'Information level': ['PLZ8'] * len(untagged_attrb),
                                         'Attribute': untagged_attrb}
                                        )
                           ])

    # Other tags
    info_data = pd.concat([info_data,
                           pd.DataFrame({'Information level': ['Building', '125m x 125m Grid'],
                                         'Attribute': ['BIP_FLAG', 'D19_LOTTO_RZ']}
                                        )
                           ])

    info_data.loc[info_data.Attribute == 'AGER_TYP', 'Information level'] = 'Person'

    # # Drop the rows that don't have a information level or malformed attributes
    # na_info_idx = info_data.loc[:, 'Information level'].isna()
    # info_data.drop(info_data.loc[na_info_idx, :].index, inplace=True)

    meta_data = meta_data.merge(info_data, how='outer', on=['Attribute'])

    return meta_data.loc[:, ['Information level', 'Attribute', 'Description', 'Value', 'Meaning']]


def get_values_missing(meta_data, sep=', '):
    """
    Read the meta data and for each column return the values used to indicate missing data.
    :params meta_data: data frame with the meta data of values
    :params sep: The separator of values
    :return: a dictionary where the keys are column names and values are a list
    """
    x = meta_data.loc[meta_data.Meaning == 'unknown', ['Attribute', 'Value']]
    x.loc[:, 'Value'] = x.loc[:, 'Value'].str.split(sep)
    na_dict = x.set_index(['Attribute']).loc[:, 'Value'].to_dict()
    return na_dict


def get_values_missing_count(df, na_dict):
    """
    Returns a summary of how much data has been marked missing for each column.
    :params df: data frame to be analysed
    :params na_dict:
    :return: A data frame with summary of how many have been tagged as missing values.
    """
    missing_col_count_summary = {}

    for col in df.columns:
        if col in na_dict:
            missing_col_count_summary[col] = df.loc[df.loc[:, col].isin(na_dict[col]), :].shape[0]
        else:
            missing_col_count_summary[col] = 0

    missing_col_count_summary = pd.DataFrame.from_dict(missing_col_count_summary,
                                                       orient='index',
                                                       columns=['missing_count'])
    missing_col_count_summary['missing_perc'] = missing_col_count_summary['missing_count']/df.shape[0] * 100

    missing_col_count_summary.index.names = ['Attribute']
    missing_col_count_summary.reset_index(inplace=True)
    missing_col_count_summary.sort_values(by='missing_count', ascending=False, inplace=True)

    return missing_col_count_summary


def read_data(file_name, meta_data):
    """
    Performs all the data clean up in one function.

    1. Set 'LNR' as index
    1. Encode OST_WEST_KZ - {'O': 0, 'W': 1}
    1. Drop 'EINGEFUEGT_AM'
    1. Replace missing values with NA,
    1. Replace GEBURTSJAHR 0 with NA
    1. Encode ANREDE_KZ, VERS_TYP  {2: 0}

    :params df: The data frame to be cleaned
    :params meta_data: Meta data of `df` that describes missing values
    :return: Cleaned up data
    """

    na_dict = get_values_missing(meta_data, sep=', ')

    na_dict['CAMEO_DEUG_2015'] += ['X', 'XX']
    na_dict['CAMEO_DEU_2015'] = ['X', 'XX']
    na_dict['CAMEO_INTL_2015'] = ['X', 'XX']
    na_dict['GEBURTSJAHR'] = ['0']

    df = pd.read_csv(file_name,
                     sep=';',
                     na_values=na_dict,
                     index_col='LNR',
                     )

    df.replace({'OST_WEST_KZ': {'O': 0, 'W': 1}}, inplace=True)

    # remove the indexing column and the time inserted column
    # Due to the amount of processing needed
    # 'LP_LEBENSPHASE_FEIN', 'LP_STATUS_FEIN', 'PRAEGENDE_JUGENDJAHRE'
    #
    # Drop columns that don't have information about the attrb values
    # 'D19_VERSI_OFFLINE_DATUM', 'D19_VERSI_ONLINE_DATUM', 'D19_VERSI_DATUM', 'ARBEIT'
    #
    # Too many values: 'CAMEO_DEU_2015'
    #
    # Drop high NA columns : 'TITEL_KZ', 'KBA05_BAUMAX', 'AGER_TYP', 'GEBURTSJAHR'
    # Drop column which is timestmap of data entry : 'MIN_GEBAEUDEJAHR'
    df.drop(['EINGEFUEGT_AM', 'LP_LEBENSPHASE_FEIN', 'LP_STATUS_FEIN',
             'PRAEGENDE_JUGENDJAHRE',
             'D19_VERSI_OFFLINE_DATUM', 'D19_VERSI_ONLINE_DATUM', 'D19_VERSI_DATUM', 'ARBEIT'
             'CAMEO_DEU_2015',
             'TITEL_KZ', 'KBA05_BAUMAX', 'AGER_TYP', 'GEBURTSJAHR',
             'MIN_GEBAEUDEJAHR'
             ], axis=1, inplace=True)

    # convert the binary values to 0, 1 so that we don't need one-hot encoding
    # we do this here after the missing value -> NA as 0 is a missing value for ANREDE_KZ
    df.replace({'ANREDE_KZ': {2: 0}}, inplace=True)
    df.replace({'VERS_TYP': {2: 0}}, inplace=True)

    return df


def get_pca_component_wts(component_matrix, pc, sort_by='magnitude'):
    label = 'PC-' + str(pc)
    summary = pd.DataFrame(component_matrix.loc[label, :])

    if sort_by == '+':
        return summary.sort_values(label, ascending=False).loc[:, label]
    elif sort_by == '-':
        # we are just sorting ascending hoping that negative components show up.
        # But there may be no negative components.
        return summary.sort_values(label, ascending=True).loc[:, label]
    elif sort_by == 'magnitude':
        summary['abs'] = summary[label].abs()
        return summary.sort_values('abs', ascending=False).loc[:, label]
    else:
        raise ValueError(f'sort_by={sort_by} not recognised. Choose +, -, magnitude')


def perform_segmentation(df, meta_data, info_level, non_categorical, pca_var=0.90, USE_MODEL_CACHE=False):

    print('{}: {} - Getting the attributes'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
    info_lvl_attributes = meta_data.loc[meta_data.loc[:, 'Information level'] == info_level, 'Attribute'].unique()
    info_lvl_attributes = list(set(info_lvl_attributes).intersection(set(df.columns)))

    X = df.loc[:, info_lvl_attributes]

    one_hot_encode_cols = X.columns[~X.columns.isin(non_categorical)]
    # X = pd.get_dummies(X, columns=one_hot_encode_cols)

    if USE_MODEL_CACHE:
        print('{}: {} - Loading the enc from cache'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
        enc = load('data/model/{}'.format(f'{info_level}_one_hot_enc.joblib'))
    else:
        print('{}: {} - Performing one hot encode'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
        # list the possible values for each attribute as an integer.
        enc_categories = [list(meta_data.loc[(meta_data.Attribute == attrb) & (meta_data.Meaning != 'unknown'), 'Value'].astype(
            float).sort_values()) for attrb in one_hot_encode_cols]
        enc = OneHotEncoder(categories=enc_categories, handle_unknown='ignore')
        enc.fit(X.loc[:, one_hot_encode_cols])
        dump(enc, 'data/model/{}'.format(f'{info_level}_one_hot_enc.joblib'))

    # reverse engineer the names
    encoded_col_names = []
    for i in enc.get_feature_names():
        name_idx, val = i.replace('x', '').split('_')
        encoded_col_names.append(one_hot_encode_cols[int(name_idx)] + '_' + val)

    # build the encoded data frame
    encoded_data = pd.DataFrame(enc.transform(X.loc[:, one_hot_encode_cols]).toarray(),
                                index=X.index, columns=encoded_col_names)
    X.drop(one_hot_encode_cols, axis=1, inplace=True)
    if X.empty:
        X = encoded_data
    else:
        X = X.merge(encoded_data, left_index=True, right_index=True, how='outer')

    cols_to_scale = set(non_categorical).intersection(set(info_lvl_attributes))
    print('Cols to scale', cols_to_scale)
    scaler = None
    if cols_to_scale:
        scaler = StandardScaler()
        if USE_MODEL_CACHE:
            print('{}: {} - Loading the scaler from cache'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                  info_level))
            scaler = load('data/model/{}'.format(f'{info_level}_scaler.joblib'))
            X.loc[:, cols_to_scale] = scaler.transform(X.loc[:, cols_to_scale])
        else:
            print('{}: {} - Fitting the scaler'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
            scaler.fit(X.loc[:, cols_to_scale])
            X.loc[:, cols_to_scale] = scaler.transform(X.loc[:, cols_to_scale])
            dump(scaler, 'data/model/{}'.format(f'{info_level}_scaler.joblib'))

    # Do pca
    if USE_MODEL_CACHE:
        print('{}: {} - Loading the pca from cache'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
        pca = load('data/model/{}'.format(f'{info_level}_pca.joblib'))
        X_txf = pca.fit_transform(X)
    else:
        print('{}: {} - Running the pca'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
        start_time = time.perf_counter()
        pca = PCA()
        pca.fit(X)
        cm_expl_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = (cm_expl_var <= pca_var).sum()
        pca = PCA(n_components)
        pca.fit(X)
        X_txf = pca.fit_transform(X)
        dump(pca, 'data/model/{}'.format(f'{info_level}_pca.joblib'))
        print('PCA completed in {:0.2f} min.'.format((time.perf_counter() - start_time) / 60))

    if USE_MODEL_CACHE:
        print('{}: {} - Loading the K_means from cache'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
        k_means = load('data/model/{}'.format(f'{info_level}_k_means.joblib'))
    else:
        print('{}: {} - Running the K_means '.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
        start_time = time.perf_counter()
        k_means = [KMeans(n_clusters=i, random_state=0, n_jobs=4, precompute_distances=True).fit(X_txf) for i in
                   range(5, 35, 5)]
        dump(k_means, 'data/model/{}'.format(f'{info_level}_k_means.joblib'))
        print('K_means completed in {:0.2f} min.'.format((time.perf_counter() - start_time)/60))

    return enc, scaler, pca, k_means


def fit_k_means_final(df, meta_data, info_level, non_categorical, enc, scaler, pca, n_cluster=10, USE_MODEL_CACHE=False):

    if USE_MODEL_CACHE:
        print('{}: {} - Loading the K_means from cache'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
        k_means_final = load('data/model/{}'.format(f'{info_level}_k_means_final.joblib'))
        return k_means_final

    print('{}: {} - Getting the attributes'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
    info_lvl_attributes = meta_data.loc[meta_data.loc[:, 'Information level'] == info_level, 'Attribute'].unique()
    info_lvl_attributes = list(set(info_lvl_attributes).intersection(set(df.columns)))

    X = df.loc[:, info_lvl_attributes]

    print('{}: {} - Performing one hot encode'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
    one_hot_encode_cols = X.columns[~X.columns.isin(non_categorical)]
    # X = pd.get_dummies(X, columns=one_hot_encode_cols)

    # reverse engineer the names
    encoded_col_names = []
    for i in enc.get_feature_names():
        name_idx, val = i.replace('x', '').split('_')
        encoded_col_names.append(one_hot_encode_cols[int(name_idx)] + '_' + val)

    # build the encoded data frame
    encoded_data = pd.DataFrame(enc.transform(X.loc[:, one_hot_encode_cols]).toarray(),
                                index=X.index, columns=encoded_col_names)
    X.drop(one_hot_encode_cols, axis=1, inplace=True)
    if X.empty:
        X = encoded_data
    else:
        X = X.merge(encoded_data, left_index=True, right_index=True, how='outer')

    cols_to_scale = set(non_categorical).intersection(set(info_lvl_attributes))
    if cols_to_scale:
        print('{}: {} - Transform using the scaler'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
        X.loc[:, cols_to_scale] = scaler.transform(X.loc[:, cols_to_scale])

    # Do pca
    print('{}: {} - Transform using the pca'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
    X_txf = pca.transform(X)
    
    print('{}: {} - Running the K_means '.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
    start_time = time.perf_counter()
    k_means_final = KMeans(n_clusters=n_cluster, random_state=0, n_jobs=4, precompute_distances=True).fit(X_txf)
    dump(k_means_final, 'data/model/{}'.format(f'{info_level}_k_means_final.joblib'))
    print('K_means completed in {:0.2f} min.'.format((time.perf_counter() - start_time)/60))

    return k_means_final


def predict_pca_kmeans(df, meta_data, info_level, non_categorical, enc, scaler, pca, k_means):
    print('{}: {} - Getting the attributes'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
    info_lvl_attributes = meta_data.loc[meta_data.loc[:, 'Information level'] == info_level, 'Attribute'].unique()
    info_lvl_attributes = list(set(info_lvl_attributes).intersection(set(df.columns)))

    X = df.loc[:, info_lvl_attributes]
    print('{}: {} - Performing one hot encode'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), info_level))
    one_hot_encode_cols = X.columns[~X.columns.isin(non_categorical)]
    # X = pd.get_dummies(X, columns=one_hot_encode_cols)

    # reverse engineer the names
    encoded_col_names = []
    for i in enc.get_feature_names():
        name_idx, val = i.replace('x', '').split('_')
        encoded_col_names.append(one_hot_encode_cols[int(name_idx)] + '_' + val)

    # build the encoded data frame
    encoded_data = pd.DataFrame(enc.transform(X.loc[:, one_hot_encode_cols]).toarray(),
                                index=X.index, columns=encoded_col_names)
    X.drop(one_hot_encode_cols, axis=1, inplace=True)
    if X.empty:
        X = encoded_data
    else:
        X = X.merge(encoded_data, left_index=True, right_index=True, how='outer')

    cols_to_scale = set(non_categorical).intersection(set(info_lvl_attributes))

    if cols_to_scale:
        scaler.fit(X.loc[:, cols_to_scale])
        X.loc[:, cols_to_scale] = scaler.transform(X.loc[:, cols_to_scale])

    X_txf = pca.transform(X)
    X_txf_k = k_means.predict(X_txf)

    return X_txf_k


def get_encoded_col_names(col_names, info_level, meta_data, non_categorical, enc):

    info_lvl_attributes = meta_data.loc[meta_data.loc[:, 'Information level'] == info_level, 'Attribute'].unique()
    info_lvl_attributes = list(set(info_lvl_attributes).intersection(set(col_names)))

    one_hot_encode_cols = list(set(info_lvl_attributes) - set(non_categorical))

    # reverse engineer the names
    encoded_col_names = []
    for i in enc.get_feature_names():
        name_idx, val = i.replace('x', '').split('_')
        encoded_col_names.append(one_hot_encode_cols[int(name_idx)] + '_' + val)

    return encoded_col_names


def get_scaler_col_names(col_names, info_level, meta_data, non_categorical):
    info_lvl_attributes = meta_data.loc[meta_data.loc[:, 'Information level'] == info_level, 'Attribute'].unique()
    info_lvl_attributes = list(set(info_lvl_attributes).intersection(set(col_names)))

    return list(set(non_categorical).intersection(set(info_lvl_attributes)))


def choose_cluster(popl_k_means, customer_k_means, n=2):
    """
    Given two k mean results, compare the counts for each cluster and return the ones with largest diff
    """
    popl_perc = pd.Series(popl_k_means).value_counts() / popl_k_means.shape[0]
    cust_perc = pd.Series(customer_k_means).value_counts() / customer_k_means.shape[0]

    return abs(popl_perc - cust_perc).sort_values(ascending=False).index.tolist()[0:n]


def view_centroid_info(centroid, cluster=0, n=10):
    x = centroid.loc[[cluster], :].transpose()
    x.reset_index(inplace=True)
    x.rename(columns={'index': 'Attribute'}, inplace=True)
    return x.sort_values(by=cluster, ascending=False).head(n)
