import pandas as pd
import numpy as np
from . import stats


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


def data_clean_up(df, meta_data, na_col_thold=None, na_row_thold=None, diversity_thold=None):
    """
    Performs all the data clean up in one function.

    NOTE: This function modifies the input df. no copies are made.

    1. Set 'LNR' as index
    1. Encode OST_WEST_KZ - {'O': 0, 'W': 1}
    1. Drop 'EINGEFUEGT_AM'
    1. Drop columns that don't have meta data
    1. Replace missing values with NA,
    1. Replace GEBURTSJAHR 0 with NA
    1. Encode ANREDE_KZ, VERS_TYP  {2: 0}

    1. Find NA rows and cols based on the threshold
    1. Find columns with low diversity

    :params df: The data frame to be cleaned
    :params meta_data: Meta data of `df` that describes missing values
    :params na_col_thold: NA % threshold for columns
    :params na_row_thold: NA % threshold for rows
    :params diversity_thold: diversity % threshold for columns
    :return: Cleaned up data, names of columns dropped for NA, names of columns dropped for diversity,
    """

    df.replace({'OST_WEST_KZ': {'O': 0, 'W': 1}}, inplace=True)

    # remove the indexing column and the time inserted column
    df.drop(['EINGEFUEGT_AM'], axis=1, inplace=True)

    # drop columns with not meta data
    no_meta_cols = set(df.columns) - set(meta_data.Attribute)
    df.drop(no_meta_cols, axis=1, inplace=True)

    # Get the missing values and mark them as NA
    na_dict = get_values_missing(meta_data, sep=', ')
    na_dict = {key: [float(i) for i in val] for key, val in na_dict.items()}
    df.replace(na_dict, pd.np.nan, inplace=True)

    # Birthday cant be 0
    df.replace({'GEBURTSJAHR': 0}, pd.np.nan, inplace=True)

    # convert the binary values to 0, 1 so that we don't need one-hot encoding
    # we do this here after the missing value -> NA as 0 is a missing value for ANREDE_KZ
    df.replace({'ANREDE_KZ': {2: 0}}, inplace=True)
    df.replace({'VERS_TYP': {2: 0}}, inplace=True)

    # Remove cols with missing data
    cols_to_drop = None
    if na_col_thold:
        col_na_summary = get_na_summary(df)
        cols_to_drop = col_na_summary.loc[col_na_summary.na_perc > na_col_thold, :].index

    # Remove rows with missing data
    rows_to_drop = None
    if na_row_thold:
        row_na_summary = get_na_summary(df, axis=1)
        rows_to_drop = row_na_summary.loc[row_na_summary.na_perc > na_row_thold, :].index

    # Remove low diversity columns - USE WITH CARE - takes forever on large data set!
    low_div_col = None
    if diversity_thold:
        col_uq = df.nunique()
        shannon_idx = df.apply(stats.shannon_diversity_index)
        shannon_idx = shannon_idx / np.log(col_uq)
        low_div_col = shannon_idx.loc[shannon_idx <= diversity_thold].index

    return df, no_meta_cols, cols_to_drop, rows_to_drop, low_div_col


def data_one_hot_encode(df, uq_val_thold=None):
    """
    This does the one hot encoding of the data
    :params df: Data frame to be encoded
    :params uq_val_thold: The unique value threshold for a column, beyond which it will be ignored/dropped
    :return: The modified data frame and list of dropped columns if any
    """
    no_encode_cols = ['ANZ_HAUSHALTE_AKTIV', 'ANZ_HH_TITEL', 'ANZ_PERSONEN', 'ANZ_TITEL',
                      'KBA13_ANZAHL_PKW', 'ANREDE_KZ', 'BIP_FLAG', 'GREEN_AVANTGARDE',
                      'KBA05_SEG6', 'OST_WEST_KZ', 'VERS_TYP']
    one_hot_encode_cols = df.columns[~df.columns.isin(no_encode_cols)]

    col_to_drop = None
    if uq_val_thold:
        col_n_uq = df.loc[:, one_hot_encode_cols].nunique()
        col_to_drop = col_n_uq[col_n_uq >= uq_val_thold]

    return df, col_to_drop


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
        return summary.loc[:, label]