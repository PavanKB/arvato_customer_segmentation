import pandas as pd


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

    na_perc = na_count / n_tot * 100

    return pd.DataFrame({'na_count': na_count,
                         'na_perc': na_perc})


def read_value_meta_data(file_path='data/DIAS Attributes - Values 2017.xlsx', sheet_name='Tabelle1'):
    """
    Read the meta data file
    :params file_path: The path of the excel file with meta data
    :params sheet_name: Sheet to read
    :return: The meta data as the data frame. All columns are read as string.
    """
    meta_data = pd.read_excel(file_path, sheet_name=sheet_name, header=1, dtype=str).iloc[:, 1:]
    meta_data.fillna(method='ffill', inplace=True)

    return meta_data


def get_values_missing(meta_data, sep=','):
    """
    Read the meta data and for each column return the values used to indicate missing data.
    :params meta_data: data frame with the meta data of values
    :params sep: The separator of values in `missing_val_col`
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
        missing_col_count_summary[col] = df.loc[df[col].isin(na_dict[col]), :].shape[0]
        
    missing_col_count_summary = pd.DataFrame.from_dict(missing_col_count_summary,
                                                       orient='index',
                                                       columns=['missing_count'])
    missing_col_count_summary.index.names = ['attribute']
    missing_col_count_summary.reset_index(inplace=True)

    return missing_col_count_summary

