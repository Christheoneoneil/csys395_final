import pandas as pd
import os
import missingno as msno
from functools import reduce
import matplotlib.pyplot as plt

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


def prelim_analysis(df_list: list, possible_hero_col: list) -> None:
    """
    prelim analysis looks to do a quick analysis
    of the commonality and completeness acorss the
    data sets

    Params:
    df_list: list data frames read in 
    possible_hero_cols: list of column names
    that are hero columns across data sets

    Returns: 
    None
    """

    df_list = df_list.copy()
    df_list = [df.drop(["Unnamed: 0"], axis=1) if "Unnamed: 0" in list(df.columns) else df for df in df_list ]
    colnames = [list(df.columns) for df in df_list]
    flatten_cols = [subcols  for cols in colnames for subcols in cols]
    
    hero_col_names = [col for col in flatten_cols if col.lower() in possible_hero_col]
    renamed_heros_col = possible_hero_col[0]
    heros = [pd.DataFrame(df).rename(columns={hero_col:renamed_heros_col}) 
            for df, hero_col in zip(df_list, hero_col_names)]
    
    common_heros = reduce(lambda df1,df2: pd.merge(df1,df2,on=possible_hero_col[0]), heros)
    print("Number of Common Heros:", len(common_heros[renamed_heros_col]))
    print(common_heros.info())

    msno.matrix(msno.nullity_sort(common_heros))
    plt.show()

unmerged_dfs = read_data(os.listdir("data/"))
prelim_analysis(unmerged_dfs, ["hero", "name"])
