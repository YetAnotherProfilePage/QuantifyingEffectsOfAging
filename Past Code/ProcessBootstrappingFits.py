# Want to drop the worst half of the bootstrapping fits.

## Tasks
"""
#Load files
#Sort rows by RMSLE
#Keep only top 200.
#Export to BootstrappingFits
"""
NUM_SAMPLES =250

import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

import math

from GroupColors import *

Fits_Folder = "UnprocessedBootstrappingFits/"
Destination_Folder = "BootstrappingFits/"


import itertools

ModelGroupLabels = ["MA","MB","MC","MD"]
ModelMechLabels = ["1","2","3","4"]
DataGroups = ["Adult","Aged"]

test =itertools.product(ModelGroupLabels,ModelMechLabels,DataGroups)

file_names = []
for i in test:
    file_name = i[0]+i[1]+"_"+i[2]+"_Bootstrapping.csv"
    print(file_name)
    file_names.append(file_name)

print(file_names)

if __name__ == "__main__":
    for file_name in file_names:
        df = pd.read_csv(Fits_Folder+file_name)
        sorted_df = df.sort_values(by=["RMSLE"],inplace=False)
        #print(df["RMSLE"])
        print(sorted_df.iloc[:NUM_SAMPLES]["RMSLE"])

        print((sorted_df.iloc[:NUM_SAMPLES].shape))
        print(sorted_df.iloc[:NUM_SAMPLES].index)
        #print(df['RMSLE'])
        df = df.iloc[sorted_df.iloc[:NUM_SAMPLES].index]
        df = df.sample(frac=1).reset_index(drop=True)
        df[df.columns[2:]].to_csv(f"{Destination_Folder}{file_name}")


