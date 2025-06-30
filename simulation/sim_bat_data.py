import sys
sys.path.append('../battery')
from battery import Battery
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
import matplotlib.pyplot as plt

# Function to simulate battery behavior
def simulate_battery(args, temp_cell = 25):
    power, soc, p_max = args
    e_max = 2 * p_max
    bat = Battery(p_max=p_max, e_max=e_max, temp_cell=temp_cell, soc=soc)
    bat.update_heat_transfer_coefficient(0.01)
    bat.charge(power=power, length_t = 5/60)
    temp = bat.temp_cell
    voltage = bat.voltage
    if voltage <= 1.5:
        return None
    
    return (power, soc, p_max, temp), (power, soc, p_max, voltage)

def plot_3d_surface(data, z_label):
    power_vals = np.array([x[0] for x in data])
    soc_vals = np.array([x[1] for x in data])
    z_vals = np.array([x[2] for x in data])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(soc_vals, power_vals, z_vals, cmap='viridis', edgecolor='none')
    
    ax.set_ylabel('Power (kW)')
    ax.set_xlabel('SoC')
    ax.set_zlabel(z_label)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, cax=cbar_ax)
    plt.show()


if __name__ == "__main__":

    n_storage = 36
    df_storage = pd.read_excel('data/storage.xlsx', index_col = 0).iloc[0:n_storage,:]
    s_storage = df_storage.sn_mva.to_numpy() * 1E3
    e_storage = df_storage.max_e_mwh.to_numpy() * 1E3
    
    s_unique = list(dict.fromkeys(s_storage))
    idx_storage = [s_unique.index(i) for i in s_storage]
    
    
    # Generate (power, soc, p_max) combinations
    param_grid = [(power, np.round(soc,3), p_max) for soc in np.arange(0.0, 1.1, 0.1) for p_max in s_unique for power in np.linspace(-p_max, p_max, 21)]
    
    # Create a multiprocessing Pool
    with Pool() as pool:
        results = pool.map(simulate_battery, param_grid)
        
    filtered_results = [x for x in results if x is not None]
    
    # Unpack results
    list_temp, list_voltage = zip(*filtered_results)
    list_temp = list(list_temp)
    list_voltage = list(list_voltage)

    grouped = defaultdict(list)
    for item in list_voltage:
        item_new = item[:2] + item[3:]
        grouped[item[2]].append(item_new)
        
    for s in s_unique:
        np.save('data\\voltage_'+str(s)+'.npy', grouped[s])
        
    grouped2 = defaultdict(list)
    for item in list_temp:
        item_new = item[:2] + item[3:]
        grouped2[item[2]].append(item_new)
        
    for s in s_unique:
        np.save('data\\temp_'+str(s)+'.npy', grouped2[s])
        
    # visualization
    plot_3d_surface(grouped[6.8], z_label='Cell voltage (V)')
    plot_3d_surface(grouped2[6.8], z_label='Temperature (C)')

    grouped3 = defaultdict(list)
    for item in grouped[34.6]:
        item_new = item[:1] + item[2:]
        grouped3[item[1]].append(item_new)
        
    for soc in np.arange(0.0, 1.1, 0.1):
        soc = np.round(soc,3)
        plt.plot(np.array(grouped3[soc])[:,0],np.array(grouped3[soc])[:,1],label=soc)
    plt.legend()
    plt.show()
    
    l_sens = []
    for soc in np.arange(0.0, 1.1, 0.1):
        soc = np.round(soc,3)
        idx = np.where(np.abs(np.array(grouped3[soc])[:,0]) <= 1e-6)
        p_temp = (np.array(grouped3[soc])[:,0])
        p_temp[idx] = 1E-6
        sens = (np.array(grouped3[soc])[:,1]-np.array(grouped3[soc])[idx,1])/p_temp
        sens = np.delete(sens,idx)
        l_sens.append(sens)

    # Create boxplots
    plt.boxplot(l_sens)    
    plt.grid(True)
    plt.show()