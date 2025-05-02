import sys
import numpy as np
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt

sys.path.append('../meas_opt')
from battery import Battery

# Fixed parameters
p_max = 10
e_max = 10
temp_cell = 25

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

# Function to simulate battery behavior
def simulate_battery(args):
    power, soc = args
    print(f"PID {os.getpid()} - Power {power} kW, SoC {soc}.")
    bat = Battery(p_max=p_max, e_max=e_max, temp_cell=temp_cell, soc=soc)
    bat.initialize()
    bat.charge(power=power, length_t = 5/60)
    temp = bat.temp_cell
    voltage = bat.hist_dt_voltage[-1]
    if voltage <= 1.5:
        return None
    return (power, soc, temp), (power, soc, voltage)

if __name__ == '__main__':
    # Generate (power, soc) combinations
    param_grid = [(power, soc) for soc in np.arange(0.0, 1.1, 0.1) for power in np.arange(-10, 11, 1.0)]

    # Create a multiprocessing Pool
    with Pool() as pool:
        results = pool.map(simulate_battery, param_grid)
        
    filtered_results = [x for x in results if x is not None]

    # Unpack results
    list_temp, list_voltage = zip(*filtered_results)
    list_temp = list(list_temp)
    list_voltage = list(list_voltage)
    
    # Plot temperature surface
    plot_3d_surface(list_temp, z_label='Temperature (°C)')
    
    # Plot voltage surface
    plot_3d_surface(list_voltage, z_label='Cell voltage (V)')
    
    