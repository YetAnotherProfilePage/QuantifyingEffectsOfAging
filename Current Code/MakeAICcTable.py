import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from GroupColors import *

from Models.ModelA1 import *
from Models.ModelA2 import *
from Models.ModelA3 import *
from Models.ModelA4 import *

from Models.ModelB1 import *
from Models.ModelB2 import *
from Models.ModelB3 import *
from Models.ModelB4 import *

from Models.ModelC1 import *
from Models.ModelC2 import *
from Models.ModelC3 import *
from Models.ModelC4 import *

from Models.ModelD1 import *
from Models.ModelD2 import *
from Models.ModelD3 import *
from Models.ModelD4 import *

CD8_Data = pd.read_csv("ExperimentalData/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("ExperimentalData/ReorganizedData_ViralTiter.csv")

#TODO: Make it so that replacement occurs for DPI > 5
Viral_Data['Viral Titer (Pfu/ml)'] = Viral_Data['Viral Titer (Pfu/ml)'].replace(0.0,1e-8).copy()
Viral_Data.loc[Viral_Data['DPI']==0,'Viral Titer (Pfu/ml)'] = 25.0

Adult_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Adult']
Aged_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Aged']

Adult_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Adult']
Aged_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Aged']

ALPHA = 0.5
t = np.linspace(0,19,190)
t_vals = [i for i in range(1,20)]

## Set up dataframe columns and rows
index_labels = [
    "MA1",
    "MA2",
    "MA3",
    "MA4",
    "MB1",
    "MB2",
    "MB3",
    "MB4",
    "MC1",
    "MC2",
    "MC3",
    "MC4",
    "MD1",
    "MD2",
    "MD3",
    "MD4"
]

cols = ["Model", "Adult", "Aged"]

### Load model fits

## Group A

adult_modelA1_params = np.loadtxt("ModelFits/adult_modelA1_params.txt")
aged_modelA1_params = np.loadtxt("ModelFits/aged_modelA1_params.txt")

adult_modelA2_params = np.loadtxt("ModelFits/adult_modelA2_params.txt")
aged_modelA2_params = np.loadtxt("ModelFits/aged_modelA2_params.txt")

adult_modelA3_params = np.loadtxt("ModelFits/adult_modelA3_params.txt")
aged_modelA3_params = np.loadtxt("ModelFits/aged_modelA3_params.txt")

adult_modelA4_params = np.loadtxt("ModelFits/adult_modelA4_params.txt")
aged_modelA4_params = np.loadtxt("ModelFits/aged_modelA4_params.txt")

## GROUP B

adult_modelB1_params = np.loadtxt("ModelFits/adult_modelB1_params.txt")
aged_modelB1_params = np.loadtxt("ModelFits/aged_modelB1_params.txt")

adult_modelB2_params = np.loadtxt("ModelFits/adult_modelB2_params.txt")
aged_modelB2_params = np.loadtxt("ModelFits/aged_modelB2_params.txt")

adult_modelB3_params = np.loadtxt("ModelFits/adult_modelB3_params.txt")
aged_modelB3_params = np.loadtxt("ModelFits/aged_modelB3_params.txt")

adult_modelB4_params = np.loadtxt("ModelFits/adult_modelB4_params.txt")
aged_modelB4_params = np.loadtxt("ModelFits/aged_modelB4_params.txt")

## Group C

adult_modelC1_params = np.loadtxt("ModelFits/adult_modelC1_params.txt")
aged_modelC1_params = np.loadtxt("ModelFits/aged_modelC1_params.txt")

adult_modelC2_params = np.loadtxt("ModelFits/adult_modelC2_params.txt")
aged_modelC2_params = np.loadtxt("ModelFits/aged_modelC2_params.txt")

adult_modelC3_params = np.loadtxt("ModelFits/adult_modelC3_params.txt")
aged_modelC3_params = np.loadtxt("ModelFits/aged_modelC3_params.txt")

adult_modelC4_params = np.loadtxt("ModelFits/adult_modelC4_params.txt")
aged_modelC4_params = np.loadtxt("ModelFits/aged_modelC4_params.txt")

## GROUP D

adult_modelD1_params = np.loadtxt("ModelFits/adult_modelD1_params.txt")
aged_modelD1_params = np.loadtxt("ModelFits/aged_modelD1_params.txt")

adult_modelD2_params = np.loadtxt("ModelFits/adult_modelD2_params.txt")
aged_modelD2_params = np.loadtxt("ModelFits/aged_modelD2_params.txt")

adult_modelD3_params = np.loadtxt("ModelFits/adult_modelD3_params.txt")
aged_modelD3_params = np.loadtxt("ModelFits/aged_modelD3_params.txt")

adult_modelD4_params = np.loadtxt("ModelFits/adult_modelD4_params.txt")
aged_modelD4_params = np.loadtxt("ModelFits/aged_modelD4_params.txt")

if __name__ == "__main__":
    ## GROUP A AICC
    adult_modelA1_AICc = ModelA1_AICc(adult_modelA1_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=1)
    aged_modelA1_AICc = ModelA1_AICc(aged_modelA1_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=1)

    adult_modelA2_AICc = ModelA2_AICc(adult_modelA2_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=1)
    aged_modelA2_AICc = ModelA2_AICc(aged_modelA2_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=1)

    adult_modelA3_AICc = ModelA3_AICc(adult_modelA3_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=0)
    aged_modelA3_AICc = ModelA3_AICc(aged_modelA3_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=0)
    
    adult_modelA4_AICc = ModelA4_AICc(adult_modelA4_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=0)
    aged_modelA4_AICc = ModelA4_AICc(aged_modelA4_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=0)

    ## GROUP B AICC
    adult_modelB1_AICc = ModelB1_AICc(adult_modelB1_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=1)
    aged_modelB1_AICc = ModelB1_AICc(aged_modelB1_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=1)

    adult_modelB2_AICc = ModelB2_AICc(adult_modelB2_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=1)
    aged_modelB2_AICc = ModelB2_AICc(aged_modelB2_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=1)

    adult_modelB3_AICc = ModelB3_AICc(adult_modelB3_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=0)
    aged_modelB3_AICc = ModelB3_AICc(aged_modelB3_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=0)
    
    adult_modelB4_AICc = ModelB4_AICc(adult_modelB4_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=0)
    aged_modelB4_AICc = ModelB4_AICc(aged_modelB4_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=0)

    ## GROUP C AICC
    adult_modelC1_AICc = ModelC1_AICc(adult_modelC1_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=1)
    aged_modelC1_AICc = ModelC1_AICc(aged_modelC1_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=1)

    adult_modelC2_AICc = ModelC2_AICc(adult_modelC2_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=1)
    aged_modelC2_AICc = ModelC2_AICc(aged_modelC2_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=1)

    adult_modelC3_AICc = ModelC3_AICc(adult_modelC3_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=0)
    aged_modelC3_AICc = ModelC3_AICc(aged_modelC3_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=0)
    
    adult_modelC4_AICc = ModelC4_AICc(adult_modelC4_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=0)
    aged_modelC4_AICc = ModelC4_AICc(aged_modelC4_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=0)

    ## GROUP D AICC
    adult_modelD1_AICc = ModelD1_AICc(adult_modelD1_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=1)
    aged_modelD1_AICc = ModelD1_AICc(aged_modelD1_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=1)

    adult_modelD2_AICc = ModelD2_AICc(adult_modelD2_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=1)
    aged_modelD2_AICc = ModelD2_AICc(aged_modelD2_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=1)

    adult_modelD3_AICc = ModelD3_AICc(adult_modelD3_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=0)
    aged_modelD3_AICc = ModelD3_AICc(aged_modelD3_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=0)
    
    adult_modelD4_AICc = ModelD4_AICc(adult_modelD4_params,19,1000,Adult_Viral_Data,Adult_CD8_Data, param_modifier=0)
    aged_modelD4_AICc = ModelD4_AICc(aged_modelD4_params,19,1000,Aged_Viral_Data,Aged_CD8_Data, param_modifier=0)

    Vals = [
        [adult_modelA1_AICc,aged_modelA1_AICc],
        [adult_modelA2_AICc,aged_modelA2_AICc],
        [adult_modelA3_AICc,aged_modelA3_AICc],
        [adult_modelA4_AICc,aged_modelA4_AICc],
        [adult_modelB1_AICc,aged_modelB1_AICc],
        [adult_modelB2_AICc,aged_modelB2_AICc],
        [adult_modelB3_AICc,aged_modelB3_AICc],
        [adult_modelB4_AICc,aged_modelB4_AICc],
        [adult_modelC1_AICc,aged_modelC1_AICc],
        [adult_modelC2_AICc,aged_modelC2_AICc],
        [adult_modelC3_AICc,aged_modelC3_AICc],
        [adult_modelC4_AICc,aged_modelC4_AICc],
        [adult_modelD1_AICc,aged_modelD1_AICc],
        [adult_modelD2_AICc,aged_modelD2_AICc],
        [adult_modelD3_AICc,aged_modelD3_AICc],
        [adult_modelD4_AICc,aged_modelD4_AICc]
        ]

    df = pd.DataFrame(data = Vals, columns = ["Adult AICc", "Aged AICc"])
    df["Model"] = index_labels

    df = df.set_index('Model')
    print(df)

    df.to_csv("ModelAICc.csv")

