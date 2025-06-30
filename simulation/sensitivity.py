import numpy as np
import math

def sensitivity_voltage(soc, p_max):
    
    direc = "data\\voltage_" + str(p_max) +'.npy'
    data_voltage = np.load(direc)
    
    soc_up = math.ceil(soc*10)/10
    soc_down = math.floor(soc*10)/10
    
    if soc_up == soc_down:
        soc_up += 0.1
        soc_down -= 0.1
        soc_up = min(soc_up, 1.0)
        soc_down = max(soc_down, 0.0)
        
    l = np.array([])
        
    for soc in np.arange(soc_down, soc_up+0.1, 0.1):
        soc = np.round(soc,1)
    
        profile = data_voltage[data_voltage[:,1]==soc]
        idx = np.where(np.abs(profile[:,0]) <= 1e-6)
        profile[idx,0] = 1E-6
        sens = (profile[:,2] - profile[idx,2]) / profile[:,0]
        sens = np.delete(sens,idx)
    
        l = np.concatenate((l, sens))
    
    return np.max(l)


def sensitivity_thermal(soc, p_max):
     
    direc = "data\\temp_" + str(p_max) +'.npy'
    data_temp = np.load(direc)

    return data_temp