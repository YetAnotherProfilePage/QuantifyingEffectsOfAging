import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from GroupColors import *

DESTINATION = "ParameterComparisonPlots"

## This file considers Models MC1 and MD1
MD1_Aged = pd.read_csv("BootstrappingFits/MD1_Aged_Bootstrapping.csv")
MC1_Adult = pd.read_csv("BootstrappingFits/MC1_Adult_Bootstrapping.csv")

print(MD1_Aged)
print(MC1_Adult)

# We are concerned primarily with parameters r, k_T, and d_T.

## Plot posterior of the comparable parameters between the two models


print(MD1_Aged['r'])
print(MC1_Adult['r'])

nbins = 30

plt.figure()
fig, ax = plt.subplots()

weights = np.ones_like(MD1_Aged['r'])/float(len(MD1_Aged['r']))

ax.set_title('Bootstrap Distributions of CTL Proliferation Rates',fontsize=16)
b, bins, patches = ax.hist(MC1_Adult['r'],nbins, density=True,label='Adult $r$',weights=weights,alpha=0.8)
b2, bins2, patches2 = ax.hist(MD1_Aged['r'],nbins, density=True,label='Aged $r$',weights=weights,alpha=0.8)
ax.set_xlabel('$r$ : CTL proliferation rate')
ax.set_ylabel('Frequency')

plt.legend()
fig.canvas.draw()

plt.savefig(f"{DESTINATION}/CTL-Proliferation-r-Histogram.png")


plt.figure()
fig, ax = plt.subplots()

nbins = 30

weights = np.ones_like(MD1_Aged['r'])/float(len(MD1_Aged['r']))

ax.set_title('Bootstrap Distributions of CTL Half Saturation Constants',fontsize=16)
b, bins, patches = ax.hist(MC1_Adult['k_T'],nbins, density=True,label='Adult $k_T$',weights=weights,alpha=0.8)
b2, bins2, patches2 = ax.hist(MD1_Aged['k_T'],nbins, density=True,label='Aged $k_T$',weights=weights,alpha=0.8)
ax.set_xlabel('$k_T$ : CTL Half Saturation Constant')
ax.set_ylabel('log Frequency')
ax.set_yscale('log')

plt.legend()
fig.canvas.draw()

plt.savefig(f"{DESTINATION}/CTL-Half_Saturation-k_T-Histogram.png")


plt.figure()
fig, ax = plt.subplots()

nbins = 30

weights = np.ones_like(MD1_Aged['r'])/float(len(MD1_Aged['r']))

ax.set_title('Bootstrap Distributions of CTL Clearance Rate',fontsize=16)
b, bins, patches = ax.hist(MC1_Adult['d_T'],nbins, density=True,label='Adult $d_T$',weights=weights,alpha=0.8)
b2, bins2, patches2 = ax.hist(MD1_Aged['d_T'],nbins, density=True,label='Aged $d_T$',weights=weights,alpha=0.8)
ax.set_xlabel('$d_T$ : CTL Clearance Rate')
ax.set_ylabel('Frequency')

plt.legend()
fig.canvas.draw()

plt.savefig(f"{DESTINATION}/CTL-Clearance-d_T-Histogram.png")

## Compute probability comparisons on the posterior distributions

r_adult = list(MC1_Adult['r'])
r_aged = list(MD1_Aged['r'])
outcomes = [1 if r_aged[i]<r_adult[i] else 0 for i in range(len(r_adult))]
print(f"Probability that r is less for aged than adult is {sum(outcomes)/len(outcomes)}")

k_T_adult = list(MC1_Adult['k_T'])
k_T_aged = list(MD1_Aged['k_T'])
outcomes = [1 if k_T_aged[i]<k_T_adult[i] else 0 for i in range(len(r_adult))]
print(f"Probability that k_T is less for aged than adult is {sum(outcomes)/len(outcomes)}")

d_T_adult = list(MC1_Adult['d_T'])
d_T_aged = list(MD1_Aged['d_T'])
outcomes = [1 if d_T_aged[i]<d_T_adult[i] else 0 for i in range(len(r_adult))]
print(f"Probability that d_T is less for aged than adult is {sum(outcomes)/len(outcomes)}")

## Identify the meaningful differences from our analysis.
# -> Reduced clearance rate d_T for aged vs adult
# -> Reduced viral proliferation rate