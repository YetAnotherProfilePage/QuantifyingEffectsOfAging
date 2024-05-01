#Note: Run this once.


RESAMPLES = 500

import numpy as np
import pandas as pd
import scipy as sp


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

def resample_measurements_across_time(dataframe, time_column,val_column,num_resamples):
    dataframe = dataframe.reset_index()


    resamples = []
    time_values = np.array(dataframe[time_column])

    for i in np.unique(dataframe[time_column]):
        source= np.array(dataframe[dataframe[time_column]==i].index)

        indices = np.random.choice(source,[len(source),num_resamples])

        resamples.append(indices)

    resamples = np.concatenate(resamples,axis=0)
    resampled_values = []

    for i in range(num_resamples):
        vals = np.array(dataframe.iloc[resamples[:,i]][val_column])
        resampled_values.append(vals)
    
    resampled_values = np.array(resampled_values)
    
    res = {}
    res[time_column] = dataframe[time_column]
    for i in range(num_resamples):
        res['Resample'+str(i)]=resampled_values[i]
    
    return pd.DataFrame(res,index=None)

if __name__ == "__main__":
    print(Adult_CD8_Data.columns)
    print(Adult_Viral_Data.columns)
    Adult_CD8_bootstrapped = resample_measurements_across_time(Adult_CD8_Data, 'DPI', 'CD8+ per g/tissue', RESAMPLES)
    Aged_CD8_bootstrapped = resample_measurements_across_time(Aged_CD8_Data, 'DPI', 'CD8+ per g/tissue', RESAMPLES)
    Adult_Virus_bootstrapped = resample_measurements_across_time(Adult_Viral_Data, 'DPI', 'Viral Titer (Pfu/ml)', RESAMPLES)
    Aged_Virus_bootstrapped = resample_measurements_across_time(Aged_Viral_Data, 'DPI', 'Viral Titer (Pfu/ml)', RESAMPLES)

    Adult_CD8_bootstrapped.to_csv('BootstrappingData/Adult_CD8_Resamples.csv')
    Aged_CD8_bootstrapped.to_csv('BootstrappingData/Aged_CD8_Resamples.csv')
    Adult_Virus_bootstrapped.to_csv('BootstrappingData/Adult_Viral_Resamples.csv')
    Aged_Virus_bootstrapped.to_csv('BootstrappingData/Aged_Viral_Resamples.csv')