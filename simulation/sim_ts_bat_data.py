import sys
sys.path.append('../battery')
from battery import Battery
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(42)

def simulate_battery(args):
    p_max = args
    e_max = 2 * p_max    
    bat = Battery(p_max=p_max, e_max=e_max)
    l_power = []
    
    phase_length = 72
    
    for t in tqdm(range(288*7)):
        power_max = min(e_max*(1-bat.soc) * 60/5, p_max)
        power_min = max(-e_max*bat.soc * 60/5, -p_max)
        
        phase = (t // phase_length) % 2
        if phase == 0:
            power = np.random.uniform(0, power_max)  # charging
        else:
            power = np.random.uniform(power_min, 0)  # discharging
        #power = np.random.uniform(power_min, power_max)
        bat.charge(power=power, length_t = 5/60)
        l_power.append(power)
    bat.hist_power = np.array(l_power)
    return bat

if __name__ == "__main__":
    bat = simulate_battery(6.8)
    
    plt.plot(bat.hist_soc)
    plt.show()
    
    plt.plot(bat.hist_voltage)
    plt.axhline(y=4.2, color='r', linestyle='--', label='Voltage Limit (4.2V)')
    plt.axhline(y=2.5, color='b', linestyle='--', label='Voltage Limit (2.5V)')
    plt.legend()
    plt.show()
    
    plt.plot(bat.hist_power)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(bat.hist_soc[:-1], bat.hist_power, bat.hist_voltage[1:], cmap='viridis', edgecolor='none')   
    ax.set_ylabel('Power (kW)')
    ax.set_xlabel('SoC')
    ax.set_zlabel('Voltage (V)')
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, cax=cbar_ax)
    plt.show()