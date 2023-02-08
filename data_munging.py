import pandas as pd
import os
import itertools

def read_data(file_list: list) -> list:
    """
    read data reads in list of provided data

    Params:
    file_list: list of files contained wthin 
    a directory 
    Returns:
    list of data frames for merging 
    
    """
    df_list = [pd.read_csv("data/" + file) 
                for file in file_list]

    return df_list 

def prelim_analysis(df_list: list) -> None:
    """
    prelim analysis looks to do a quick analysis
    of the commonality and completeness acorss the
    data sets

    Params:
    df_list: list data frames read in 

    Returns: 
    None
    """

    df_list = df_list.copy()
    df_list = [df.drop(["Unnamed: 0"], axis=1) if "Unnamed: 0" in  list(df.columns) else df for df in df_list ]
    colnames = [list(df.columns) for df in df_list]
    flatten_cols = [subcols  for cols in colnames for subcols in cols]
    hero_col_names = [col  for col in flatten_cols if "hero" in col.lower() or "name" in col.lower()]
    
    
unmerged_dfs = read_data(os.listdir("data/"))
prelim_analysis(unmerged_dfs)
