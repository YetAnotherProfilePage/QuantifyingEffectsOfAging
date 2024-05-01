import pandas as pd
import numpy as numpy

from ModelBounds import *

Models = ["MA1","MA2","MA3","MA4","MB1","MB2","MB3","MB4","MC1","MC2","MC3","MC4","MD1","MD2","MD3","MD4"]

Model_Bounds = {
    "MA1_Adult": Adult_ModelA1_Parameter_Bounds,
    "MA1_Aged": Aged_ModelA1_Parameter_Bounds,
    "MA2_Adult": Adult_ModelA2_Parameter_Bounds,
    "MA2_Aged": Aged_ModelA2_Parameter_Bounds,
    "MA3_Adult": Adult_ModelA3_Parameter_Bounds,
    "MA3_Aged": Aged_ModelA3_Parameter_Bounds,
    "MA4_Adult": Adult_ModelA4_Parameter_Bounds,
    "MA4_Aged": Aged_ModelA4_Parameter_Bounds,

    "MB1_Adult": Adult_ModelB1_Parameter_Bounds,
    "MB1_Aged": Aged_ModelB1_Parameter_Bounds,
    "MB2_Adult": Adult_ModelB2_Parameter_Bounds,
    "MB2_Aged": Aged_ModelB2_Parameter_Bounds,
    "MB3_Adult": Adult_ModelB3_Parameter_Bounds,
    "MB3_Aged": Aged_ModelB3_Parameter_Bounds,
    "MB4_Adult": Adult_ModelB4_Parameter_Bounds,
    "MB4_Aged": Aged_ModelB4_Parameter_Bounds,

    "MC1_Adult": Adult_ModelC1_Parameter_Bounds,
    "MC1_Aged": Aged_ModelC1_Parameter_Bounds,
    "MC2_Adult": Adult_ModelC2_Parameter_Bounds,
    "MC2_Aged": Aged_ModelC2_Parameter_Bounds,
    "MC3_Adult": Adult_ModelC3_Parameter_Bounds,
    "MC3_Aged": Aged_ModelC3_Parameter_Bounds,
    "MC4_Adult": Adult_ModelC4_Parameter_Bounds,
    "MC4_Aged": Aged_ModelC4_Parameter_Bounds,

    "MD1_Adult": Adult_ModelD1_Parameter_Bounds,
    "MD1_Aged": Aged_ModelD1_Parameter_Bounds,
    "MD2_Adult": Adult_ModelD2_Parameter_Bounds,
    "MD2_Aged": Aged_ModelD2_Parameter_Bounds,
    "MD3_Adult": Adult_ModelD3_Parameter_Bounds,
    "MD3_Aged": Aged_ModelD3_Parameter_Bounds,
    "MD4_Adult": Adult_ModelD4_Parameter_Bounds,
    "MD4_Aged": Aged_ModelD4_Parameter_Bounds,
}

# These are currently somewhat incorrect. I've fixed the cases of MA3 and MB1
Models_Inputs = {
    "MA1":['p','k_V','c_V','r','k_T','d_T','T_0','V_0'],
    "MA2":['p','k_V','c_V','r','k_T','d_T','c_T','T_0','V_0'],
    "MA3":['p','k_V','c_V','r','k_T','d_T','T_0','V_0'],
    "MA4":['p','k_V','c_V','r','k_T','d_T','c_T','T_0','V_0'],
    "MB1":['beta','d_I','p','c','d_T','r','k_T','U_0', 'I_0', 'V_0', 'T_0'],
    "MB2":['beta','d_I','p','c','d_T','r','k_T','c_T','U_0', 'I_0', 'V_0', 'T_0'],
    "MB3":['beta','d_I','p','c','d_T','r','U_0', 'I_0', 'V_0', 'T_0'],
    "MB4":['beta','d_I','p','c','d_T','r','c_T','U_0', 'I_0', 'V_0', 'T_0'],
    "MC1":['p','k_V','c_V','r','k_T','d_T','K','T_0','V_0'],
    "MC2":['p','k_V','c_V','r','k_T','d_T','c_T','K','T_0','V_0'],
    "MC3":['p','k_V','c_V','r','d_T','K','T_0','V_0'],
    "MC4":['p','k_V','c_V','r','d_T','c_T','K','T_0','V_0'],
    "MD1":['beta','d_I','p','c','d_T','r','k_T','K','U_0', 'I_0', 'V_0', 'T_0'],
    "MD2":['beta','d_I','p','c','d_T','r','k_T','c_T','K','U_0', 'I_0', 'V_0', 'T_0'],
    "MD3":['beta','d_I','p','c','d_T','r','K','U_0', 'I_0', 'V_0', 'T_0'],
    "MD4":['beta','d_I','p','c','d_T','r','c_T','K','U_0', 'I_0', 'V_0', 'T_0'],
}

## Define useful lookup tables.

Models_RMSLE_Dict = {
    "MA1":ModelA1_RMSLE,
    "MA2":ModelA2_RMSLE,
    "MA3":ModelA3_RMSLE,
    "MA4":ModelA4_RMSLE,
    "MB1":ModelB1_RMSLE,
    "MB2":ModelB2_RMSLE,
    "MB3":ModelB3_RMSLE,
    "MB4":ModelB4_RMSLE,
    "MC1":ModelC1_RMSLE,
    "MC2":ModelC2_RMSLE,
    "MC3":ModelC3_RMSLE,
    "MC4":ModelC4_RMSLE,
    "MD1":ModelD1_RMSLE,
    "MD2":ModelD2_RMSLE,
    "MD3":ModelD3_RMSLE,
    "MD4":ModelD4_RMSLE,
}

Models_ViralClearanceTime_Dict = {
    "MA1":ModelA1_ViralClearanceTime,
    "MA2":ModelA2_ViralClearanceTime,
    "MA3":ModelA3_ViralClearanceTime,
    "MA4":ModelA4_ViralClearanceTime,
    "MB1":ModelB1_ViralClearanceTime,
    "MB2":ModelB2_ViralClearanceTime,
    "MB3":ModelB3_ViralClearanceTime,
    "MB4":ModelB4_ViralClearanceTime,
    "MC1":ModelC1_ViralClearanceTime,
    "MC2":ModelC2_ViralClearanceTime,
    "MC3":ModelC3_ViralClearanceTime,
    "MC4":ModelC4_ViralClearanceTime,
    "MD1":ModelD1_ViralClearanceTime,
    "MD2":ModelD2_ViralClearanceTime,
    "MD3":ModelD3_ViralClearanceTime,
    "MD4":ModelD4_ViralClearanceTime,
}

Models_TotalViralLoad_Dict = {
    "MA1":ModelA1_TotalViralLoad,
    "MA2":ModelA2_TotalViralLoad,
    "MA3":ModelA3_TotalViralLoad,
    "MA4":ModelA4_TotalViralLoad,
    "MB1":ModelB1_TotalViralLoad,
    "MB2":ModelB2_TotalViralLoad,
    "MB3":ModelB3_TotalViralLoad,
    "MB4":ModelB4_TotalViralLoad,
    "MC1":ModelC1_TotalViralLoad,
    "MC2":ModelC2_TotalViralLoad,
    "MC3":ModelC3_TotalViralLoad,
    "MC4":ModelC4_TotalViralLoad,
    "MD1":ModelD1_TotalViralLoad,
    "MD2":ModelD2_TotalViralLoad,
    "MD3":ModelD3_TotalViralLoad,
    "MD4":ModelD4_TotalViralLoad,
}

Models_ExcessCTL_Dict = {
    "MA1":ModelA1_ExcessCTL,
    "MA2":ModelA2_ExcessCTL,
    "MA3":ModelA3_ExcessCTL,
    "MA4":ModelA4_ExcessCTL,
    "MB1":ModelB1_ExcessCTL,
    "MB2":ModelB2_ExcessCTL,
    "MB3":ModelB3_ExcessCTL,
    "MB4":ModelB4_ExcessCTL,
    "MC1":ModelC1_ExcessCTL,
    "MC2":ModelC2_ExcessCTL,
    "MC3":ModelC3_ExcessCTL,
    "MC4":ModelC4_ExcessCTL,
    "MD1":ModelD1_ExcessCTL,
    "MD2":ModelD2_ExcessCTL,
    "MD3":ModelD3_ExcessCTL,
    "MD4":ModelD4_ExcessCTL,
}
