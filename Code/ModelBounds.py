import pandas as pd
import numpy as np

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

V_0 = 25.0

t = np.linspace(0,19,190)

#Note: We do not distinguish k_V between Aged and Adult.
k_V = np.max(Viral_Data['Viral Titer (Pfu/ml)']) #k_V = 1.2e6

# Load Models
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

### Now define parameter bounds again

"""
Adult_ModelA1_Parameter_Bounds = [
    [1e-2,1e2], #p
    [k_V,k_V], #k_V #1.2e6
    [1e-8,1e-4], #c_V
    #(r,r), #r
    [1e-4,1e2],
    [1e1,1e8],
    [1e-4,1], #d_T
    [T_0_Adult,T_0_Adult],
    [V_0,V_0], #V_0
    ]

Aged_ModelA1_Parameter_Bounds = [
    [1e-2,1e2], #p
    [k_V,k_V], #k_V #1.2e6
    [1e-8,1e-4], #c_V
    #(r,r), #r
    [1e-4,1e2],
    [1e1,1e8],
    [1e-4,1], #d_T
    [T_0_Aged,T_0_Aged],
    [V_0,V_0], #V_0
    ]

Adult_ModelA2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

Adult_ModelA3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #d_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #c_T
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

Adult_ModelA4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]
"""
Adult_ModelA1_Parameter_Bounds = [
    [1e-2,1e2], #p
    [k_V,k_V], #k_V #1.2e6
    [1e-8,1e-4], #c_V
    #(r,r), #r
    [1e-4,1e2],
    [1e1,1e8],
    [1e-4,1], #d_T
    [T_0_Adult,T_0_Adult],
    [V_0,V_0], #V_0
    ]

Aged_ModelA1_Parameter_Bounds = [
    [1e-2,1e2], #p
    [k_V,k_V], #k_V #1.2e6
    [1e-8,1e-4], #c_V
    #(r,r), #r
    [1e-4,1e2],
    [1e1,1e8],
    [1e-4,1], #d_T
    [T_0_Aged,T_0_Aged],
    [V_0,V_0], #V_0
    ]

Adult_ModelA2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (0,1), #c_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (0,1), #c_T
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

Adult_ModelA3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #d_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #d_T
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

Adult_ModelA4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #d_T
    (0,1), #c_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #d_T
    (0,1), #c_T
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

Adult_ModelB1_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]
#Param values largely taken from Esteban's paper 2014

Aged_ModelB1_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]


#beta,d_I,p,c,d_T,r,k_T,c_T,U_0,I_0,V_0,T_0
#Param values largely taken from Esteban's paper 2014
Adult_ModelB2_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (0,1e0),#c_T
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]


#Param values largely taken from Esteban's paper 2014

Aged_ModelB2_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (0,1e0),#c_T
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]

#beta,d_I,p,c,d_T,r,U_0,I_0,V_0,T_0
#Param values largely taken from Esteban's paper 2014
Adult_ModelB3_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]

#beta,d_I,p,c,d_T,r,U_0,I_0,V_0,T_0
#Param values largely taken from Esteban's paper 2014
Aged_ModelB3_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]

#Param values largely taken from Esteban's paper 2014
Adult_ModelB4_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0), #p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (0,1e0),#c_T
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]
#Param values largely taken from Esteban's paper 2014

Aged_ModelB4_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (0,1e0),#c_T
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]


Adult_ModelC1_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (0,1e-4), #K
    (T_0_Adult,T_0_Adult),
    (V_0,V_0) #V_0
    ]

Aged_ModelC1_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (0,1e-4), #K
    (T_0_Aged,T_0_Aged),
    (V_0,V_0) #V_0
    ]

Adult_ModelC2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (0,1), #c_T
    (0,1e-4), #K
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelC2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (0,1), #c_T
    (0,1e-4), #K
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

Adult_ModelC3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #c_T
    (0,1e-4), #K
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelC3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #c_T
    (0,1e-4), #K
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

Adult_ModelC4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #d_T
    (0,1), #c_T
    (0,1e-4), #K
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelC4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #d_T
    (0,1), #c_T
    (0,1e-4), #K
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]


Adult_ModelD1_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (0,1e-4), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]
#Param values largely taken from Esteban's paper 2014

Aged_ModelD1_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (0,1e-4), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]


#beta,d_I,p,c,d_T,r,k_T,c_T,U_0,I_0,V_0,T_0
#Param values largely taken from Esteban's paper 2014
Adult_ModelD2_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (0,1e0),#c_T
    (0,1e-4), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]
#Param values largely taken from Esteban's paper 2014

Aged_ModelD2_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (0,1e0),#c_T
    (0,1e-4), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]

#beta,d_I,p,c,d_T,r,U_0,I_0,V_0,T_0
#Param values largely taken from Esteban's paper 2014
Adult_ModelD3_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (0,1e-4), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]

#beta,d_I,p,c,d_T,r,U_0,I_0,V_0,T_0
#Param values largely taken from Esteban's paper 2014
Aged_ModelD3_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (0,1e-4), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]

#Param values largely taken from Esteban's paper 2014
Adult_ModelD4_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0), #p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (0,1e0),#c_T
    (0,1e-4), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]
#Param values largely taken from Esteban's paper 2014

Aged_ModelD4_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (0,1e0),#c_T
    (0,1e-4), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]
