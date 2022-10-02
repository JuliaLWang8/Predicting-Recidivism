# Initialization
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import chain
from datetime import datetime as dt

pd.set_option('display.max_columns', None)
#torch.set_default_tensor_type("torch.cuda.FloatTensor")
# COMPAS dataset
#raw_df = pd.read_csv('https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv')
#raw_df.head()
    

def process_dataframe(raw_df, 
    columns_to_view=[
        #"id",
        "sex",
        "age",
        "age_cat",
        "race_other", 
        "race_african-american", 
        "race_caucasian",
        "race_hispanic", 
        "race_native-american",
        "race_asian",
        "jail_time",
        "juv_fel_count",
        #"decile_score", 
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        # "is_recid",
        # "is_violent_recid",
        #"two_year_recid",
        #"encoder_value"
    ],
    encode_dict={
        "sex": {"Male": 0, "Female": 1},
        "age_cat": {"Less than 25": 0, "25 - 45": 1, "Greater than 45": 2},
        "race": {"Other": 0, "African-American": 1, "Caucasian": 2, "Hispanic": 3, "Native American": 4, "Asian": 5},
        "c_charge_degree": {"M": 0, "F": 1}
    }):

    df = raw_df.copy()

    if ("c_jail_in" in df.columns) & ("c_jail_out" in df.columns):
        for io in ["in", "out"]:
            df[f"c_jail_{io}"] = pd.to_datetime(df[f"c_jail_{io}"])
        df["jail_time"] = (abs(df["c_jail_out"] - df["c_jail_in"]).round("D") / np.timedelta64(1, "D")).fillna(0)
    else:
        columns_to_view.drop(columns="jail_time")

    df = df.replace(encode_dict)

    df["encoder_value"] = df["sex"]*6 + df["race"]

    #print(df.columns)

    df = df.replace({"race": {0: "other", 1: "african-american", 2: "caucasian", 3: "hispanic", 4: "native-american", 5: "asian"}})

    races = pd.get_dummies(df.race, prefix='race')

    df = pd.merge(df, races, left_index=True, right_index=True)

    df_inpt = df[columns_to_view], torch.float32
    df_sens = df["encoder_value"], torch.int64
    df_outpt = df["two_year_recid"], torch.float32

    return [torch.tensor(d.values, dtype=enc) for d,enc in [df_inpt, df_sens, df_outpt]]


def split_temporal(df, time='compas_screening_date', train_split=0.8, val_split=None):
    df[time] = pd.to_datetime(df[time])
    df = df.sort_values(time)

    train_ind = int(np.round(len(df)*train_split))
    train_df = df.iloc[:train_ind]

    if val_split == None:
        test_df = df.iloc[train_ind:]
        return train_df, test_df
    else:
        val_ind = int(np.round(len(df)*(val_split + train_split)))
        val_df = df.iloc[train_ind:val_ind]
        test_df = df.iloc[val_ind:]
        
    return train_df, val_df, test_df

