import numpy as np
import math
from sklearn.linear_model import LinearRegression
import pandas as pd

temperature = pd.read_csv("data\\temperature_5minute.csv")["temp (C)"].to_numpy()

def sensitivity_voltage(soc, p_max, delta = 0.1):
    
    direc = "data\\voltage_" + str(p_max) +'.npy'
    data_voltage = np.load(direc)
    
    soc_up = soc + 0.1
    soc_down = soc - 0.1
    soc_up = min(soc_up, 1.0)
    soc_down = max(soc_down, 0.0)
        
    l = []
    for i in range(1,len(data_voltage)):
        if data_voltage[i,0] <= soc_up and data_voltage[i,0] >= soc_down and abs(data_voltage[i,1])>=1e-4:
            sens = (data_voltage[i,2]-data_voltage[i-1,2])/data_voltage[i,1]
            l.append(sens)
            
    return max(l)


def sensitivity_thermal(p_max):
     
    direc = "data\\temperature_" + str(p_max) +'.npy'
    data_temp = np.load(direc)
    
    power = data_temp[1:,1]
    power = power**2
    temp_delta = data_temp[1:,2] - data_temp[:-1,2]
    temp_ambient = np.tile(temperature,7)
    temp_delta_ambient = data_temp[:-1,2] - temp_ambient[:-1]
    
    X = np.concatenate((temp_delta_ambient.reshape(-1, 1), power.reshape(-1, 1)),axis=1)
    y = temp_delta
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    return model.coef_