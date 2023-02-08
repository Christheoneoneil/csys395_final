import pandas as pd
import os
import itertools
from functools import reduce

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
    heros = [pd.DataFrame(df[hero_col]).rename(columns={hero_col:"name"}) 
            for df,hero_col in zip(df_list, hero_col_names)]
    common_heros = reduce(lambda df1,df2: pd.merge(df1,df2,on='name'), heros)
    print(len(common_heros))

    
unmerged_dfs = read_data(os.listdir("data/"))
prelim_analysis(unmerged_dfs)
