import sys
sys.path.append('../battery')
from battery import Battery
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool

np.random.seed(42)

temperature = pd.read_csv("data\\temperature_5minute.csv")["temp (C)"].to_numpy()

def simulate_battery(args):
    p_max = args
    e_max = 2 * p_max    
    bat = Battery(p_max=p_max, e_max=e_max)
    bat.update_heat_transfer_coefficient(0.005,0.5)
    l_power = []
    
    phase_length = 72
    
    for t in tqdm(range(288*7)):
        power_max = min(e_max*(1-bat.soc) * 60/5, p_max)
        power_min = max(-e_max*bat.soc * 60/5, -p_max)
        
        phase = (t // phase_length) % 2
        if phase == 0:
            power = np.random.uniform(0, power_max)  # charging
            power = max(power, 0.1)
            if bat.soc >= 1.0:
                power = 0.0
        else:
            power = np.random.uniform(power_min, 0)  # discharging
            power = min(power, -0.1)
            if bat.soc <= 0.0:
                power = 0.0
        bat.charge(power=power, length_t = 5/60, temp_ambient=temperature[t%288])
        l_power.append(power)
    bat.hist_power = np.array(l_power)
    return bat

if __name__ == "__main__":
    
    n_storage = 36
    df_storage = pd.read_excel('data/storage.xlsx', index_col = 0).iloc[0:n_storage,:]
    s_storage = df_storage.sn_mva.to_numpy() * 1E3
    s_unique = list(dict.fromkeys(s_storage))
    idx_storage = [s_unique.index(i) for i in s_storage]
    
    for s in s_unique:
        battery = simulate_battery(s)
        voltage_save = np.concatenate((battery.hist_soc[:-1].reshape(-1, 1), battery.hist_power[:].reshape(-1, 1), 
                                    battery.hist_voltage[1:].reshape(-1, 1)),axis=1)
        np.save('data\\voltage_'+str(s)+'.npy',voltage_save)
        temp_save = np.concatenate((battery.hist_soc[:-1].reshape(-1, 1), battery.hist_power[:].reshape(-1, 1), 
                                    battery.hist_temp_cell[1:].reshape(-1, 1)),axis=1)
        np.save('data\\temperature_'+str(s)+'.npy',temp_save)
        