import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


temperature = pd.read_csv("data\\temperature_5minute.csv")["temp (C)"].to_numpy()

def plot_3d_surface(data, z_label, x_label = 'SoC', save = None):
    power_vals = np.array([x[1] for x in data])
    soc_vals = np.array([x[0] for x in data])
    z_vals = np.array([x[2] for x in data])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(soc_vals, power_vals, z_vals, cmap='viridis', edgecolor='none',alpha=1.0)
    
    ax.set_ylabel('Power (kW)')
    ax.set_xlabel(x_label)
    ax.set_zlabel(z_label)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, cax=cbar_ax)
    if save is not None:
        plt.savefig(save, bbox_inches = 'tight')
    plt.show()

capacity = 6.8
data_voltage = np.load("data\\voltage_" + str(capacity) + ".npy")
data_voltage_rev = np.concatenate((data_voltage[:,0].reshape(-1, 1),data_voltage[:,1].reshape(-1, 1),
                                   (data_voltage[:,2]-data_voltage[:,3]).reshape(-1, 1)),axis=1)
plot_3d_surface(data_voltage, "Voltage (V)", save="..\\..\\paper\\voltage3d.pdf")

data_temperature = np.load("data\\temperature_" + str(capacity) + ".npy")
power = data_temperature[1:,1]
power = power
temp_delta = data_temperature[1:,2] - data_temperature[:-1,2]
temp_ambient = np.tile(temperature,7)
temp_delta_ambient = data_temperature[:-1,2] - temp_ambient[:-1]

data_temp_new = np.concatenate((temp_delta_ambient.reshape(-1, 1), power.reshape(-1, 1), 
                            temp_delta.reshape(-1, 1)),axis=1)

plot_3d_surface(data_temp_new, "Temperature change (°C)", 
                x_label = "Temperature diff. to\n ambient (°C)", save = "..\\..\\paper\\temperature3d.pdf")