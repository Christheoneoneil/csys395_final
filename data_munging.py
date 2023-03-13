import pandas as pd
import os
import missingno as msno
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np


def read_data(file_list: list) -> list:
    """
    read data reads in list of provided data

    Params:
    file_list: list of files contained wthin 
    a directory 
    
    Returns:
    list of data frames for merging 
    
    """
    df_list = [pd.read_csv("data/" + file).replace("-", np.nan) 
                for file in file_list if file != "merged_dat.csv"]

    return df_list 


def prelim_analysis(df_list: list, possible_hero_col: list, fig_title: str) -> None:
    """
    prelim analysis looks to do a quick analysis
    of the commonality and completeness across the
    data sets

    Params:
    df_list: list data frames read in 
    possible_hero_cols: list of column names
    that are hero columns across data sets
    fig_title: title for matrix plot

    Returns: 
    merged data
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
    plt.savefig(fig_title)

    return common_heros


def clean_data(merged_data_set: pd.DataFrame, cols_to_drop: list, prelim: pd.DataFrame):
    """
    clean data takes in raw uncleaned data,
    and drops and cleans for feature engineering

    Params:
    merged_data_set: data frame of raw uncleaned data
    cols_to_drop: columns that are redundent and or
    to sparse for usage and will be dropped
    prelim: analysis function to give basic missingness stats

    Returns: 
    None 
    """

    merged_data_set = merged_data_set.copy()
    merged_data_set.drop(cols_to_drop, axis="columns", inplace=True)
    merged_data_set.columns = map(str.lower, merged_data_set.columns)
    merged_data_set.columns = merged_data_set.columns.str.rstrip('_x')
    merged_data_set.drop_duplicates(inplace=True)
    
    num_cols = list(merged_data_set.select_dtypes(include="float64").columns)
    merged_data_set[num_cols] = merged_data_set[num_cols].apply(np.abs)
    print(merged_data_set[num_cols].describe())
    
    cat_cols = list(merged_data_set.select_dtypes(include="object").columns)
    print(merged_data_set[cat_cols].describe())

    prelim([merged_data_set], ["hero"], "missing_mat_non_red")
    
    merged_data_set.to_csv("data/merged_dat.csv")
    

unmerged_dfs = read_data(file_list=os.listdir("data/"))
merged_data = prelim_analysis(df_list=unmerged_dfs, 
                              possible_hero_col=["hero", "name"], 
                              fig_title="raw_missing_mat")
drop_cols = ["Identity", "Status", "Race_y", "Gender_y", 
             "Alignment_y", "Height_y", "Weight_y", 
             "EyeColor", "SkinColor", "Publisher_y",
             "Year", "Appearances", "FirstAppearance", 
             "AdditionalData", "Total", "Alignment"]
clean_data(merged_data_set=merged_data, cols_to_drop=drop_cols, 
           prelim=prelim_analysis) 
