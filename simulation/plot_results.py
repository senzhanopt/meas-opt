import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

daily_temperature = pd.read_csv('data/temperature_5minute.csv')["temp (C)"].to_numpy()

voltage = pd.read_csv("result\mat_voltage_lyapunov_voltage_thermal.csv", index_col=0)
voltage_nc = pd.read_csv("result\mat_voltage_lyapunov.csv", index_col=0)

max_voltage = voltage.max(axis=1)
min_voltage = voltage.min(axis=1)
max_voltage_nc = voltage_nc.max(axis=1)
min_voltage_nc = voltage_nc.min(axis=1)

# Time axis
time = max_voltage.index

plt.figure(figsize=(6,3))

# --- With voltage & thermal control ---
plt.plot(time, max_voltage, color='red', label='Max (w/ constraints)')
plt.plot(time, min_voltage, color='blue', label='Min (w/ constraints)')
#plt.fill_between(time, min_voltage, max_voltage,
#                 color='orange', alpha=0.2,
#                 label='Range (Voltage + Thermal)')

# --- Without thermal control (Lyapunov only) ---
plt.plot(time, max_voltage_nc, color='darkred', linestyle='--', label='Max (w/o constraints)')
plt.plot(time, min_voltage_nc, color='navy', linestyle='--', label='Min (w/o constraints)')
#plt.fill_between(time, min_voltage_nc, max_voltage_nc,
#                 color='gray', alpha=0.2,
#                 label='Range (Lyapunov Only)')

# Optional voltage limit lines
plt.axhline(4.2, color='black', linestyle='--', alpha=0.6, label='Upper limit (4.2 V)')
plt.axhline(2.5, color='black', linestyle=':', alpha=0.6, label='Lower limit (2.5 V)')

tick_positions = np.arange(0, 289, 24)  # 0, 12, 24, ..., 288
tick_labels = [f"{int(pos/12)}:00" for pos in tick_positions]
plt.xticks(tick_positions, tick_labels)

plt.ylabel("Cell voltage (V)")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("..\\..\\paper\\voltagecomp.pdf")
plt.show()

temp_cell = pd.read_csv("result\mat_temp_cell_lyapunov_voltage_thermal.csv", index_col=0)
temp_cell_nc = pd.read_csv("result\mat_temp_cell_lyapunov.csv", index_col=0)

max_temp_cell = temp_cell.max(axis=1)
min_temp_cell = temp_cell.min(axis=1)
max_temp_cell_nc = temp_cell_nc.max(axis=1)
min_temp_cell_nc = temp_cell_nc.min(axis=1)

# Time axis
time = max_temp_cell.index

plt.figure(figsize=(6,3))

# --- With temp_cell & thermal control ---
plt.plot(time, max_temp_cell, color='red', label='Max (w/ constraints)')
#plt.plot(time, min_temp_cell, color='blue', label='Min (temp_cell + Thermal)')
#plt.fill_between(time, min_temp_cell, max_temp_cell,
#                 color='orange', alpha=0.2,
#                 label='Range (temp_cell + Thermal)')

# --- Without thermal control (Lyapunov only) ---
plt.plot(time, max_temp_cell_nc, color='darkred', linestyle='--', label='Max (w/o constraints)')
#plt.plot(time, min_temp_cell_nc, color='navy', linestyle='--', label='Min (Lyapunov Only)')
#plt.fill_between(time, min_temp_cell_nc, max_temp_cell_nc,
#                 color='gray', alpha=0.2,
#                 label='Range (Lyapunov Only)')

# Optional temp_cell limit lines
plt.axhline(45, color='black', linestyle='--', alpha=0.6, label='Upper limit (45 °C)')
plt.plot(daily_temperature, color='orange', linestyle=':', label='Ambient temperature')
tick_positions = np.arange(0, 289, 24)  # 0, 12, 24, ..., 288
tick_labels = [f"{int(pos/12)}:00" for pos in tick_positions]
plt.xticks(tick_positions, tick_labels)

plt.ylabel("Cell temperature (°C)")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("..\\..\\paper\\temp_cellcomp.pdf")
plt.show()

mat_v = pd.read_csv("result\mat_v_lyapunov_voltage_thermal.csv", index_col=0).to_numpy()
mat_v0 = pd.read_csv("result\mat_v0.csv", index_col=0).to_numpy()
list_bus_visual = [55,66]
v_upp = 1.05
v_low = 0.95
plt.figure(figsize=(6,3))
colorset = [["red","darkred"], ["b","navy"]]
for i,b in enumerate(list_bus_visual):
    plt.plot(mat_v[:,b-1], label = f'Bus {b}', color = colorset[i][0], linestyle = "-")
    plt.plot(mat_v0[:,b-1], label = f'Bus {b} w/o control', color = colorset[i][1], linestyle = "--")
plt.xticks(tick_positions, tick_labels)
plt.ylabel("Distribution grid voltage (pu)")
plt.axhline(1.05, color='black', linestyle='--', alpha=0.6, label='Upper limit (1.05 pu)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("..\\..\\paper\\voltage.pdf")
plt.show() 


mat_loadingt = pd.read_csv("result\mat_loading_lyapunov_voltage_thermal_tight.csv", index_col=0).to_numpy()
mat_loading = pd.read_csv("result\mat_loading_lyapunov_voltage_thermal.csv", index_col=0).to_numpy()
mat_loading0 = pd.read_csv("result\mat_loading0.csv", index_col=0).to_numpy()
plt.figure(figsize=(6,3))
plt.plot(mat_loading, label = 'w/ control', color = 'r', linestyle = "-")
plt.plot(mat_loading0, label = 'w/o control', color = 'darkred', linestyle = "--")
plt.plot(mat_loadingt, label = 'w/ tighter limit', color = 'b', alpha=0.6, linestyle = "-")
plt.xticks(tick_positions, tick_labels)
plt.ylabel("Transformer loading")
plt.axhline(1.0, color='black', linestyle='--', alpha=0.6, label='Upper limit (1.0)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("..\\..\\paper\\loading.pdf")
plt.show() 

sgen = pd.read_csv("data\sgen.csv",index_col=0).to_numpy()
load = pd.read_csv("data\load_p.csv",index_col=0).to_numpy()
sgen = np.repeat(sgen, repeats = 3, axis = 0)
load = np.repeat(load, repeats = 3, axis = 0)
sgen_real = pd.read_csv("result\mat_p_pv_lyapunov_voltage_thermal.csv",index_col=0).to_numpy()
storage_real = pd.read_csv("result\mat_p_storage_lyapunov_voltage_thermal.csv",index_col=0).to_numpy()
plt.figure(figsize=(6,3))
plt.plot(sgen.sum(axis=1), label = 'Max generation', color = 'r', linestyle = "-")
plt.plot(sgen_real.sum(axis=1), label = 'Real generation', color = 'darkred', linestyle = "--")
plt.plot(-storage_real.sum(axis=1), label = 'Storage', color = 'b', alpha=0.6, linestyle = "-")
plt.plot(-load.sum(axis=1), label = 'Load', color = 'orange', alpha=0.6, linestyle = "-")
plt.xticks(tick_positions, tick_labels)
plt.ylabel("Power (kW)")
#plt.axhline(1.0, color='black', linestyle='--', alpha=0.6, label='Upper limit (1.0)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("..\\..\\paper\\powergeneration.pdf")
plt.show()


storage_real = pd.read_csv("result\mat_p_storage_lyapunov_voltage_thermal.csv",index_col=0).to_numpy()
storage_real1 = pd.read_csv("result\mat_p_storage_lyapunov_voltage_thermal_alpha0.1.csv",index_col=0).to_numpy()
storage_real2 = pd.read_csv("result\mat_p_storage_lyapunov_voltage_thermal_alpha5.0.csv",index_col=0).to_numpy()
storage_real3 = pd.read_csv("result\mat_p_storage_lyapunov_voltage_thermal_gamma0.01.csv",index_col=0).to_numpy()
storage_real4 = pd.read_csv("result\mat_p_storage_lyapunov_voltage_thermal_gamma0.1.csv",index_col=0).to_numpy()
plt.figure(figsize=(6,3))
plt.plot(-storage_real.sum(axis=1), label = r'$\alpha = 0.5, \gamma = 0.05$')
plt.plot(-storage_real1.sum(axis=1), label = r'$\alpha = 0.1, \gamma = 0.05$')
plt.plot(-storage_real2.sum(axis=1), label = r'$\alpha = 5.0, \gamma = 0.05$')
plt.plot(-storage_real3.sum(axis=1), label = r'$\alpha = 0.5, \gamma = 0.01$')
plt.plot(-storage_real4.sum(axis=1), label = r'$\alpha = 0.5, \gamma = 0.1$')
plt.xticks(tick_positions, tick_labels)
plt.ylabel("Power (kW)")
#plt.axhline(1.0, color='black', linestyle='--', alpha=0.6, label='Upper limit (1.0)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("..\\..\\paper\\sensitivityanalysis.pdf")
plt.show()

pv_real = pd.read_csv("result\mat_p_pv_lyapunov_voltage_thermal.csv",index_col=0).to_numpy()
pv_real1 = pd.read_csv("result\mat_p_pv_lyapunov_voltage_thermal_alpha0.1.csv",index_col=0).to_numpy()
pv_real2 = pd.read_csv("result\mat_p_pv_lyapunov_voltage_thermal_alpha5.0.csv",index_col=0).to_numpy()
pv_real10 = pd.read_csv("result\mat_p_pv_lyapunov_voltage_thermal_alpha1.0.csv",index_col=0).to_numpy()
pv_real3 = pd.read_csv("result\mat_p_pv_lyapunov_voltage_thermal_gamma0.01.csv",index_col=0).to_numpy()
pv_real4 = pd.read_csv("result\mat_p_pv_lyapunov_voltage_thermal_gamma0.1.csv",index_col=0).to_numpy()
plt.figure(figsize=(6,3))
plt.plot(pv_real.sum(axis=1), label = r'$\alpha = 0.5, \gamma = 0.05$')
plt.plot(pv_real1.sum(axis=1), label = r'$\alpha = 0.1, \gamma = 0.05$')
#plt.plot(pv_real10.sum(axis=1), label = r'$\alpha = 1.0, \gamma = 0.05$')
plt.plot(pv_real2.sum(axis=1), label = r'$\alpha = 5.0, \gamma = 0.05$')
plt.plot(pv_real3.sum(axis=1), label = r'$\alpha = 0.5, \gamma = 0.01$')
plt.plot(pv_real4.sum(axis=1), label = r'$\alpha = 0.5, \gamma = 0.1$')
plt.xticks(tick_positions, tick_labels)
plt.ylabel("Power (kW)")
#plt.axhline(1.0, color='black', linestyle='--', alpha=0.6, label='Upper limit (1.0)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("..\\..\\paper\\sensitivityanalysis2.pdf")
plt.show()

