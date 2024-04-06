import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import math

PRECISION = 3

def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)

Models_Params_Labels = {
    "MA1":['p','k_V','c_V','r','k_T','d_T'],
    "MA2":['p','k_V','c_V','r','k_T','d_T','c_T'],
    "MA3":['p','k_V','c_V','r','d_T'],
    "MA4":['p','k_V','c_V','r','d_T','c_T'],
    "MB1":['beta','d_I','p','c','d_T','r','k_T'],
    "MB2":['beta','d_I','p','c','d_T','r','k_T','c_T'],
    "MB3":['beta','d_I','p','c','d_T','r'],
    "MB4":['beta','d_I','p','c','d_T','r','c_T'],
    "MC1":['p','k_V','c_V','r','k_T','d_T','K'],
    "MC2":['p','k_V','c_V','r','k_T','d_T','c_T','K'],
    "MC3":['p','k_V','c_V','r','d_T','K'],
    "MC4":['p','k_V','c_V','r','d_T','c_T','K'],
    "MD1":['beta','d_I','p','c','d_T','r','k_T','K'],
    "MD2":['beta','d_I','p','c','d_T','r','k_T','c_T','K'],
    "MD3":['beta','d_I','p','c','d_T','r','K'],
    "MD4":['beta','d_I','p','c','d_T','r','c_T','K'],
}

Models_Params_Indices = {
    "MA1":[0,2,3,4,5],
    "MA2":[0,2,3,4,5,6],
    "MA3":[0,2,3,4],
    "MA4":[0,2,3,4,5],
    "MB1":[0,1,3,4,5,6],
    "MB2":[0,1,3,4,5,6,7],
    "MB3":[0,1,3,4,5],
    "MB4":[0,1,3,4,5,6],
    "MC1":[0,2,3,4,5,6],
    "MC2":[0,2,3,4,5,6,7],
    "MC3":[0,2,3,4,5],
    "MC4":[0,2,3,4,5,6],
    "MD1":[0,1,3,4,5,6,7],
    "MD2":[0,1,3,4,5,6,7,8],
    "MD3":[0,1,3,4,5,6],
    "MD4":[0,1,3,4,5,6,7],
}

Bootstrap_Fitted_Folder = "BootstrappingFits/"
Main_Fits_Folder = "ModelFits/"
Destination_Folder = "ConfidenceIntervals/"

import itertools

ModelGroupLabels = ["MA","MB","MC","MD"]
ModelMechLabels = ["1","2","3","4"]
DataGroups = ["Adult","Aged"]

test =list(itertools.product(ModelGroupLabels,ModelMechLabels,DataGroups))

test2 = list(itertools.product(ModelGroupLabels,ModelMechLabels))

bootstrapping_file_names = []
estimates_file_names = []
fitted_loss_file_names = []

for i in test:
    bootstrapping_file_name = i[0]+i[1]+"_"+i[2]+"_Bootstrapping.csv"
    #print(bootstrapping_file_name)
    bootstrapping_file_names.append(bootstrapping_file_name)

for i in test:
    estimates_file_name = i[2].lower() + "_model"+i[0][1]+i[1]+"_params.txt"
    fitted_loss_file_name = i[2].lower() + "_model"+i[0][1]+i[1]+"_fit.txt"
    #print(estimates_file_name)
    estimates_file_names.append(estimates_file_name)
    fitted_loss_file_names.append(fitted_loss_file_name)

#print(file_names)

"""
We want to save to .csv file in format: columns : [Param, Adult Estimate, Aged Estimate, Search Bounds, Units], rows: [param1,...]. With each entry of form {estimate (lower, upper)}
"""
if __name__ == "__main__":

    for i in range(len(test)):

        bootstrap_samples = pd.read_csv(Bootstrap_Fitted_Folder+bootstrapping_file_names[i])
        estimates = np.loadtxt(Main_Fits_Folder+estimates_file_names[i])
        fitted_loss = np.loadtxt(Main_Fits_Folder+fitted_loss_file_names[i])

        quants = bootstrap_samples.quantile([0.025,0.975])

        model_params_labels = Models_Params_Labels[bootstrapping_file_names[i][0:3]]
        fitted_params_indices = Models_Params_Indices[bootstrapping_file_names[i][0:3]]

        row = []

        for j in fitted_params_indices:
            par = model_params_labels[j]
            print(par)
            quants_temp = list(quants[par])
            quants_temp = [to_precision(bound,PRECISION) for bound in quants_temp]
            fit = to_precision(estimates[j],PRECISION)
            print(estimates)
            print(model_params_labels)

            print(fit)
            print(quants_temp)

            row.append(f"{fit} ({quants_temp[0]},{quants_temp[1]})")

        print(row)
        print(fitted_params_indices)

        print(len(row))
        print(len(fitted_params_indices))

        df = pd.DataFrame(data=[row],columns=[model_params_labels[i] for i in fitted_params_indices])
        names = bootstrapping_file_names[i].split("_")
        df.to_csv(Destination_Folder+f"{names[0]}_{names[1]}_ConfidenceIntervals.csv")

        #print()

    """

    redos = []
    for i in range(len(test)):
        #print(bootstrapping_file_names[i])
        #print(estimates_file_names[i])
        #print(fitted_loss_file_names[i])

        bootstrap_samples = pd.read_csv(Bootstrap_Fitted_Folder+bootstrapping_file_names[i])
        estimates = np.loadtxt(Main_Fits_Folder+estimates_file_names[i])
        fitted_loss = np.loadtxt(Main_Fits_Folder+fitted_loss_file_names[i])

        #print(estimates)
        #print(fitted_loss)
        #print(bootstrap_samples)

        quants = bootstrap_samples.quantile([0.025,0.975])

        # From here, we should determine for which models either:
        # 1. fitted losses are not in the bootstrap samples
        # 2. parameters are not in the bootstrap quantiles

        quants_temp = list(quants['RMSLE'])
        #print(quants_temp)

        # Actually, this might be fine. More data represents a more challenging fit. We would expect RMSLE to be higher for the real dataset.
        '''

        if (fitted_loss > quants_temp[1]) or (fitted_loss < quants_temp[0]):
            print(fitted_loss)
            print(quants_temp)
            print(bootstrapping_file_names[i])
            print(bootstrapping_file_names[i][0:3])
            print(bootstrapping_file_names[i][4:8])
        '''
        model_params_labels = Models_Params_Labels[bootstrapping_file_names[i][0:3]]
        fitted_params_indices = Models_Params_Indices[bootstrapping_file_names[i][0:3]]

        redo = False

        for j in fitted_params_indices:
            par = model_params_labels[j]
            quants_temp = list(quants[par])
            quants_temp = [float(to_precision(bound,PRECISION)) for bound in quants_temp]
            fit = float(to_precision(estimates[j],PRECISION))
            print(estimates)
            print(model_params_labels)

            print(fit)
            print(quants_temp)
            if (fit > quants_temp[1]) or (fit < quants_temp[0]):
                #print(fit)
                #print(quants_temp)
                #print(par)
                #print(bootstrapping_file_names[i][0:3])
                print(bootstrapping_file_names[i][0:3]+ " " + bootstrapping_file_names[i][4:8])
                redo = True
                print(f"Problem in {par} for {bootstrapping_file_names[i][0:3]} {bootstrapping_file_names[i][4:8]}")
                print()

            print()

        if redo == True:
            redos.append(bootstrapping_file_names[i][0:3]+ " " + bootstrapping_file_names[i][4:8])

    print(redos)

    print(bootstrapping_file_names)

    print(type(to_precision(12054321,5)))
    print(to_precision(12054321,5))
    print(float(to_precision(12054321,5)))

    """

