import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

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

#TODO: Some of the following boilerplate can be cleaned up.
CD8_Data = pd.read_csv("ExperimentalData/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("ExperimentalData/ReorganizedData_ViralTiter.csv")

#TODO: Make it so that replacement occurs for DPI > 5
Viral_Data['Viral Titer (Pfu/ml)'] = Viral_Data['Viral Titer (Pfu/ml)'].replace(0.0,1e-8).copy()
Viral_Data.loc[Viral_Data['DPI']==0,'Viral Titer (Pfu/ml)'] = 25.0


Adult_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Adult']
Aged_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Aged']

Adult_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Adult']
Aged_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Aged']

T_0_Adult = np.mean(Adult_CD8_Data[Adult_CD8_Data['DPI']==0]['CD8+ per g/tissue'])
T_0_Aged = np.mean(Aged_CD8_Data[Aged_CD8_Data['DPI']==0]['CD8+ per g/tissue'])

t = np.linspace(0,19,190)

#Note: We do not distinguish k_V between Aged and Adult.
k_V = np.max(Viral_Data['Viral Titer (Pfu/ml)']) #k_V = 1.2e6


adult_modelA1_params = np.loadtxt("ModelFits/adult_modelA1_params.txt")
aged_modelA1_params = np.loadtxt("ModelFits/aged_modelA1_params.txt")

adult_modelA2_params = np.loadtxt("ModelFits/adult_modelA2_params.txt")
aged_modelA2_params = np.loadtxt("ModelFits/aged_modelA2_params.txt")

adult_modelA3_params = np.loadtxt("ModelFits/adult_modelA3_params.txt")
aged_modelA3_params = np.loadtxt("ModelFits/aged_modelA3_params.txt")

adult_modelA4_params = np.loadtxt("ModelFits/adult_modelA4_params.txt")
aged_modelA4_params = np.loadtxt("ModelFits/aged_modelA4_params.txt")


adult_modelB1_params = np.loadtxt("ModelFits/adult_modelB1_params.txt")
aged_modelB1_params = np.loadtxt("ModelFits/aged_modelB1_params.txt")

adult_modelB2_params = np.loadtxt("ModelFits/adult_modelB2_params.txt")
aged_modelB2_params = np.loadtxt("ModelFits/aged_modelB2_params.txt")

adult_modelB3_params = np.loadtxt("ModelFits/adult_modelB3_params.txt")
aged_modelB3_params = np.loadtxt("ModelFits/aged_modelB3_params.txt")

adult_modelB4_params = np.loadtxt("ModelFits/adult_modelB4_params.txt")
aged_modelB4_params = np.loadtxt("ModelFits/aged_modelB4_params.txt")


adult_modelC1_params = np.loadtxt("ModelFits/adult_modelC1_params.txt")
aged_modelC1_params = np.loadtxt("ModelFits/aged_modelC1_params.txt")

adult_modelC2_params = np.loadtxt("ModelFits/adult_modelC2_params.txt")
aged_modelC2_params = np.loadtxt("ModelFits/aged_modelC2_params.txt")

adult_modelC3_params = np.loadtxt("ModelFits/adult_modelC3_params.txt")
aged_modelC3_params = np.loadtxt("ModelFits/aged_modelC3_params.txt")

adult_modelC4_params = np.loadtxt("ModelFits/adult_modelC4_params.txt")
aged_modelC4_params = np.loadtxt("ModelFits/aged_modelC4_params.txt")


adult_modelD1_params = np.loadtxt("ModelFits/adult_modelD1_params.txt")
aged_modelD1_params = np.loadtxt("ModelFits/aged_modelD1_params.txt")

adult_modelD2_params = np.loadtxt("ModelFits/adult_modelD2_params.txt")
aged_modelD2_params = np.loadtxt("ModelFits/aged_modelD2_params.txt")

adult_modelD3_params = np.loadtxt("ModelFits/adult_modelD3_params.txt")
aged_modelD3_params = np.loadtxt("ModelFits/aged_modelD3_params.txt")

adult_modelD4_params = np.loadtxt("ModelFits/adult_modelD4_params.txt")
aged_modelD4_params = np.loadtxt("ModelFits/aged_modelD4_params.txt")

"""
Here, we will compute the statistics
"""

"""
#print(adult_modelA1_params)
print(ModelA1_ViralClearanceTime(adult_modelA1_params,19,1000,threshhold = 10.0))
print(ModelA1_ViralClearanceTime(aged_modelA1_params,19,1000,threshhold = 10.0))

print(ModelA2_ViralClearanceTime(adult_modelA2_params,19,1000,threshhold = 10.0))
print(ModelA2_ViralClearanceTime(aged_modelA2_params,19,1000,threshhold = 10.0))

print(ModelA3_ViralClearanceTime(adult_modelA3_params,19,1000,threshhold = 10.0))
print(ModelA3_ViralClearanceTime(aged_modelA3_params,19,1000,threshhold = 10.0))

print(ModelA4_ViralClearanceTime(adult_modelA4_params,19,1000,threshhold = 10.0))
print(ModelA4_ViralClearanceTime(aged_modelA4_params,19,1000,threshhold = 10.0))

##
print(ModelB1_ViralClearanceTime(adult_modelB1_params,19,1000,threshhold = 10.0))
print(ModelB1_ViralClearanceTime(aged_modelB1_params,19,1000,threshhold = 10.0))

print(ModelB2_ViralClearanceTime(adult_modelB2_params,19,1000,threshhold = 10.0))
print(ModelB2_ViralClearanceTime(aged_modelB2_params,19,1000,threshhold = 10.0))

print(ModelB3_ViralClearanceTime(adult_modelB3_params,19,1000,threshhold = 10.0))
print(ModelB3_ViralClearanceTime(aged_modelB3_params,19,1000,threshhold = 10.0))

print(ModelB4_ViralClearanceTime(adult_modelB4_params,19,1000,threshhold = 10.0))
print(ModelB4_ViralClearanceTime(aged_modelB4_params,19,1000,threshhold = 10.0))

##
print(ModelC1_ViralClearanceTime(adult_modelC1_params,19,1000,threshhold = 10.0))
print(ModelC1_ViralClearanceTime(aged_modelC1_params,19,1000,threshhold = 10.0))

print(ModelC2_ViralClearanceTime(adult_modelC2_params,19,1000,threshhold = 10.0))
print(ModelC2_ViralClearanceTime(aged_modelC2_params,19,1000,threshhold = 10.0))

print(ModelC3_ViralClearanceTime(adult_modelC3_params,19,1000,threshhold = 10.0))
print(ModelC3_ViralClearanceTime(aged_modelC3_params,19,1000,threshhold = 10.0))

print(ModelC4_ViralClearanceTime(adult_modelC4_params,19,1000,threshhold = 10.0))
print(ModelC4_ViralClearanceTime(aged_modelC4_params,19,1000,threshhold = 10.0))

##
print(ModelD1_ViralClearanceTime(adult_modelD1_params,19,1000,threshhold = 10.0))
print(ModelD1_ViralClearanceTime(aged_modelD1_params,19,1000,threshhold = 10.0))

print(ModelD2_ViralClearanceTime(adult_modelD2_params,19,1000,threshhold = 10.0))
print(ModelD2_ViralClearanceTime(aged_modelD2_params,19,1000,threshhold = 10.0))

print(ModelD3_ViralClearanceTime(adult_modelD3_params,19,1000,threshhold = 10.0))
print(ModelD3_ViralClearanceTime(aged_modelD3_params,19,1000,threshhold = 10.0))

print(ModelD4_ViralClearanceTime(adult_modelD4_params,19,1000,threshhold = 10.0))
print(ModelD4_ViralClearanceTime(aged_modelD4_params,19,1000,threshhold = 10.0))
"""

###
"""
print(ModelA1_TotalViralLoad(adult_modelA1_params,19,1000,threshhold = 10.0))
print(ModelA1_TotalViralLoad(aged_modelA1_params,19,1000,threshhold = 10.0))

print(ModelA2_TotalViralLoad(adult_modelA2_params,19,1000,threshhold = 10.0))
print(ModelA2_TotalViralLoad(aged_modelA2_params,19,1000,threshhold = 10.0))

print(ModelA3_TotalViralLoad(adult_modelA3_params,19,1000,threshhold = 10.0))
print(ModelA3_TotalViralLoad(aged_modelA3_params,19,1000,threshhold = 10.0))

print(ModelA4_TotalViralLoad(adult_modelA4_params,19,1000,threshhold = 10.0))
print(ModelA4_TotalViralLoad(aged_modelA4_params,19,1000,threshhold = 10.0))

##
print()

print(ModelB1_TotalViralLoad(adult_modelB1_params,19,1000,threshhold = 10.0))
print(ModelB1_TotalViralLoad(aged_modelB1_params,19,1000,threshhold = 10.0))

print(ModelB2_TotalViralLoad(adult_modelB2_params,19,1000,threshhold = 10.0))
print(ModelB2_TotalViralLoad(aged_modelB2_params,19,1000,threshhold = 10.0))

print(ModelB3_TotalViralLoad(adult_modelB3_params,19,1000,threshhold = 10.0))
print(ModelB3_TotalViralLoad(aged_modelB3_params,19,1000,threshhold = 10.0))

print(ModelB4_TotalViralLoad(adult_modelB4_params,19,1000,threshhold = 10.0))
print(ModelB4_TotalViralLoad(aged_modelB4_params,19,1000,threshhold = 10.0))

##
print()

print(ModelC1_TotalViralLoad(adult_modelC1_params,19,1000,threshhold = 10.0))
print(ModelC1_TotalViralLoad(aged_modelC1_params,19,1000,threshhold = 10.0))

print(ModelC2_TotalViralLoad(adult_modelC2_params,19,1000,threshhold = 10.0))
print(ModelC2_TotalViralLoad(aged_modelC2_params,19,1000,threshhold = 10.0))

print(ModelC3_TotalViralLoad(adult_modelC3_params,19,1000,threshhold = 10.0))
print(ModelC3_TotalViralLoad(aged_modelC3_params,19,1000,threshhold = 10.0))

print(ModelC4_TotalViralLoad(adult_modelC4_params,19,1000,threshhold = 10.0))
print(ModelC4_TotalViralLoad(aged_modelC4_params,19,1000,threshhold = 10.0))

##
print()

print(ModelD1_TotalViralLoad(adult_modelD1_params,19,1000,threshhold = 10.0))
print(ModelD1_TotalViralLoad(aged_modelD1_params,19,1000,threshhold = 10.0))

print(ModelD2_TotalViralLoad(adult_modelD2_params,19,1000,threshhold = 10.0))
print(ModelD2_TotalViralLoad(aged_modelD2_params,19,1000,threshhold = 10.0))

print(ModelD3_TotalViralLoad(adult_modelD3_params,19,1000,threshhold = 10.0))
print(ModelD3_TotalViralLoad(aged_modelD3_params,19,1000,threshhold = 10.0))

print(ModelD4_TotalViralLoad(adult_modelD4_params,19,1000,threshhold = 10.0))
print(ModelD4_TotalViralLoad(aged_modelD4_params,19,1000,threshhold = 10.0))
"""

###
"""
print(ModelA1_ExcessCTL(adult_modelA1_params,19,1000,threshhold = 10.0))
print(ModelA1_ExcessCTL(aged_modelA1_params,19,1000,threshhold = 10.0))

print(ModelA2_ExcessCTL(adult_modelA2_params,19,1000,threshhold = 10.0))
print(ModelA2_ExcessCTL(aged_modelA2_params,19,1000,threshhold = 10.0))

print(ModelA3_ExcessCTL(adult_modelA3_params,19,1000,threshhold = 10.0))
print(ModelA3_ExcessCTL(aged_modelA3_params,19,1000,threshhold = 10.0))

print(ModelA4_ExcessCTL(adult_modelA4_params,19,1000,threshhold = 10.0))
print(ModelA4_ExcessCTL(aged_modelA4_params,19,1000,threshhold = 10.0))
"""

##
"""
print(ModelB1_ExcessCTL(adult_modelB1_params,19,1000,threshhold = 10.0))
print(ModelB1_ExcessCTL(aged_modelB1_params,19,1000,threshhold = 10.0))

print(ModelB2_ExcessCTL(adult_modelB2_params,19,1000,threshhold = 10.0))
print(ModelB2_ExcessCTL(aged_modelB2_params,19,1000,threshhold = 10.0))

print(ModelB3_ExcessCTL(adult_modelB3_params,19,1000,threshhold = 10.0))
print(ModelB3_ExcessCTL(aged_modelB3_params,19,1000,threshhold = 10.0))

print(ModelB4_ExcessCTL(adult_modelB4_params,19,1000,threshhold = 10.0))
print(ModelB4_ExcessCTL(aged_modelB4_params,19,1000,threshhold = 10.0))
"""
#
"""
print(ModelC1_ExcessCTL(adult_modelC1_params,19,1000,threshhold = 10.0))
print(ModelC1_ExcessCTL(aged_modelC1_params,19,1000,threshhold = 10.0))

print(ModelC2_ExcessCTL(adult_modelC2_params,19,1000,threshhold = 10.0))
print(ModelC2_ExcessCTL(aged_modelC2_params,19,1000,threshhold = 10.0))

print(ModelC3_ExcessCTL(adult_modelC3_params,19,1000,threshhold = 10.0))
print(ModelC3_ExcessCTL(aged_modelC3_params,19,1000,threshhold = 10.0))

print(ModelC4_ExcessCTL(adult_modelC4_params,19,1000,threshhold = 10.0))
print(ModelC4_ExcessCTL(aged_modelC4_params,19,1000,threshhold = 10.0))
"""
#
"""
print(ModelD1_ExcessCTL(adult_modelD1_params,19,1000,threshhold = 10.0))
print(ModelD1_ExcessCTL(aged_modelD1_params,19,1000,threshhold = 10.0))

print(ModelD2_ExcessCTL(adult_modelD2_params,19,1000,threshhold = 10.0))
print(ModelD2_ExcessCTL(aged_modelD2_params,19,1000,threshhold = 10.0))

print(ModelD3_ExcessCTL(adult_modelD3_params,19,1000,threshhold = 10.0))
print(ModelD3_ExcessCTL(aged_modelD3_params,19,1000,threshhold = 10.0))

print(ModelD4_ExcessCTL(adult_modelD4_params,19,1000,threshhold = 10.0))
print(ModelD4_ExcessCTL(aged_modelD4_params,19,1000,threshhold = 10.0))
"""