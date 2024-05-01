import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from GroupColors import *

from Models.ModelC1 import *

from Models.ModelD1 import *

CD8_Data = pd.read_csv("ExperimentalData/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("ExperimentalData/ReorganizedData_ViralTiter.csv")

#TODO: Make it so that replacement occurs for DPI > 5
Viral_Data['Viral Titer (Pfu/ml)'] = Viral_Data['Viral Titer (Pfu/ml)'].replace(0.0,1e-8).copy()
Viral_Data.loc[Viral_Data['DPI']==0,'Viral Titer (Pfu/ml)'] = 25.0

Adult_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Adult']
Aged_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Aged']

Adult_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Adult']
Aged_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Aged']


# Adult viral data is only valid through day 9, where it drops below measurement
Plottable_Adult_Viral_Data = Adult_Viral_Data[Adult_Viral_Data["DPI"]<=9]

# Aged viral data is only valid through day 11, where it drops below measurement
Plottable_Aged_Viral_Data = Aged_Viral_Data[Aged_Viral_Data["DPI"]<=11]

ALPHA = 0.5
t = np.linspace(0,19,190)
t_vals = [i for i in range(0,20)]


adult_ymin = min(Adult_CD8_Data['CD8+ per g/tissue'])
adult_ymax = max(Adult_CD8_Data['CD8+ per g/tissue'])

aged_ymin = min(Adult_CD8_Data['CD8+ per g/tissue'])
aged_ymax = max(Adult_CD8_Data['CD8+ per g/tissue'])    

adult_params = []
aged_params = []

RESAMPLES = 100000

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

Adult_CD8_bootstrapped = resample_measurements_across_time(Adult_CD8_Data, 'DPI', 'CD8+ per g/tissue', RESAMPLES)
Aged_CD8_bootstrapped = resample_measurements_across_time(Aged_CD8_Data, 'DPI', 'CD8+ per g/tissue', RESAMPLES)

Adult_CD8_bootstrapped = Adult_CD8_bootstrapped[Adult_CD8_bootstrapped['DPI']>=9]
Aged_CD8_bootstrapped = Aged_CD8_bootstrapped[Aged_CD8_bootstrapped['DPI']>=11]

print(Adult_CD8_bootstrapped)
print(Aged_CD8_bootstrapped) 

# I still need to trim to past the correct DPI.   

for i in range(RESAMPLES):
    Y_adult = np.log(Adult_CD8_bootstrapped[f"Resample{i}"].to_numpy())
    Y_aged = np.log(Aged_CD8_bootstrapped[f"Resample{i}"].to_numpy())

    X_adult = Adult_CD8_bootstrapped['DPI'].to_numpy()
    X_aged = Aged_CD8_bootstrapped['DPI'].to_numpy()

    X_adult = sm.add_constant(X_adult)
    X_aged = sm.add_constant(X_aged)

    c1, m1 = np.linalg.lstsq(X_adult, Y_adult, rcond=None)[0]
    c2, m2 = np.linalg.lstsq(X_aged, Y_aged, rcond=None)[0]

    #print(f"{c1},{m1}")
    adult_params.append([c1,m1])
    aged_params.append([c2,m2])

adult_slopes = [i[1] for i in adult_params]
aged_slopes = [i[1] for i in aged_params]

adult_slopes = np.array(adult_slopes)
aged_slopes = np.array(aged_slopes)

## Need to compute credible intervals for the slope

## Need to evaluate the posterior probability of adult slopes being beneath aged slopes.
# Equivalent to probability that X in adult slope < Y in aged slope
MONTE_CARLO_SAMPLES = 100000
successes = 0
drawsA = np.random.randint(0,len(adult_slopes),MONTE_CARLO_SAMPLES)
drawsB = np.random.randint(0,len(aged_slopes),MONTE_CARLO_SAMPLES)

adult_samples = adult_slopes[drawsA]
aged_samples = aged_slopes[drawsB]
outcomes = [1 if adult_samples[i]<aged_samples[i] else 0 for i in range(MONTE_CARLO_SAMPLES)]
## Need to add lines to the image, along with the mean prediction.

print(f'The probability that the Adult Downregulation is less than the aged Downregulation is: {sum(outcomes)/len(outcomes)}')

#These indicate that the slopes computed for the adult case are more negative than the aged case.
print(min(adult_slopes))
print(max(adult_slopes))
print(min(aged_slopes))
print(max(aged_slopes))
# So, in fact, the resamples indicate that the probability is nearly 100% that adult CTL declines faster than aged CTL post infection.

## I should make a histogram of the slopes.

if __name__ == "__main__":
    ## First: Plot the predicted slopes for adult and aged.
    plt.figure()

    fig, axs = plt.subplots(2, 1,figsize=(8,6),sharex='col',sharey='row')

    axs[0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    
    #axs[0].plot(t, adult_modelC1_predictions[:-1,0],label="MC1 Prediction",color=MC1_COLOR)
    #axs[0].plot(Adult_Virus_times, adult_modelC1_predictions[AdultC1_valid_indices,0][0],label="MC1 Prediction",color=MC1_COLOR)

    axs[0].set_yscale('log')
    axs[0].set_ylabel('CD8+ T Cells',fontsize=12)
    axs[0].set_title('Adult CTL Measurements',fontsize=16)
    axs[0].vlines(x=9,ymin=adult_ymin,ymax=adult_ymax,ls=':',color='red', alpha=0.5)

    all_preds = np.zeros((1000,100))

    for i in range(1000):
        preds = adult_params[i][0] + adult_params[i][1]*t[90:]
        all_preds[i,:]=preds
        axs[0].plot(t[90:],np.exp(preds),color='lightblue',alpha=0.01)

    #print(all_preds)
    mean_traj = np.mean(all_preds, axis=0)
    axs[0].plot(t[90:],np.exp(mean_traj),color='black')

    axs[1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1].plot(t, adult_modelC1_predictions[:-1,1],label="MC1 Prediction",color=MC1_COLOR)
    
    axs[1].set_ylabel('CD8+ T Cells',fontsize=12)
    axs[1].set_xlabel('Days Post Infection',fontsize=12)
    axs[1].set_yscale('log')
    axs[1].set_xticks(t_vals)
    axs[1].set_title('Aged CTL Measurements',fontsize=16)
    axs[1].vlines(x=11,ymin=aged_ymin,ymax=aged_ymax,ls=':',color='red', alpha=0.5)
    
    all_preds = np.zeros((1000,80))

    for i in range(1000):
        preds = aged_params[i][0] + aged_params[i][1]*t[110:]
        all_preds[i,:]=preds
        axs[1].plot(t[110:],np.exp(preds),color='lightblue',alpha=0.01)

    #print(all_preds)
    mean_traj = np.mean(all_preds, axis=0)

    axs[1].plot(t[110:],np.exp(mean_traj),color='black')
    plt.savefig("MiscPlots/CTL_Retraction_Analysis.png")

    ## Next: Create a histogram of the adult and aged slopes.

    plt.figure()
    fig, ax = plt.subplots()

    nbins = 100

    weights = np.ones_like(adult_slopes)/float(len(adult_slopes))

    ax.set_title('Bootstrap Distributions of CTL Downregulation Rates',fontsize=16)
    b, bins, patches = ax.hist(adult_slopes,nbins, density=True,label='Adult $\\beta_1$ Posterior',weights=weights)
    b2, bins2, patches2 = ax.hist(aged_slopes,nbins, density=True,label='Aged $\\beta_1$ Posterior',weights=weights)
    ax.set_xlabel('$\\beta_1$ : CTL downregulation rate')
    ax.set_ylabel('Frequency')

    plt.legend()
    fig.canvas.draw()

    plt.savefig("MiscPlots/CTL_Retraction_Analysis_Histogram.png")

    print("Here")

    print(b)
    print(bins)

