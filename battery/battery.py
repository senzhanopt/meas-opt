import pybamm
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Battery:
    
    def __init__(self, p_max = 15.0, e_max = 10.0, soc = 0.5, temp_cell = 25, 
                 cell_model = "dfn", thermal_model = "lumped", 
                 params_cell = "Chen2020", operating_mode = "power",
                 soc_min = 0.0, soc_max = 1.0):
        self.p_max = p_max
        self.e_max = e_max
        self.soc = soc
        self.soc_init = soc
        self.temp_cell = temp_cell
        self.cell_model = cell_model
        self.thermal_model = thermal_model
        self.params_cell = params_cell
        self.operating_mode = operating_mode
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.hist_soc = np.array([soc])
        self.hist_temp_cell = np.array([temp_cell])
        self.hist_voltage = np.array([])
        self.sol = None
        self.initialize()
        
    def initialize(self, power_cell = 9.0):    
        
        '''
        determine number of cells and efficiencies
        
        '''

        options = {"cell geometry": "pouch", "thermal": self.thermal_model,
                   "operating mode": self.operating_mode}
        if self.cell_model == "dfn":
            self.model = pybamm.lithium_ion.DFN(options)
        else:
            raise Exception("Cell model is not DFN.")
        self.parameter_values = pybamm.ParameterValues(self.params_cell)
        self.v_low_cut = self.parameter_values["Lower voltage cut-off [V]"]
        self.v_upp_cut = self.parameter_values["Upper voltage cut-off [V]"]
        #self.cap_nominal = self.parameter_values["Nominal cell capacity [A.h]"]  
        self.parameter_values.update({"Power function [W]": pybamm.InputParameter("Power [W]")}, 
                                     check_already_exists=False)
        sim = pybamm.Simulation(self.model, parameter_values = self.parameter_values)
        
        # simulate a charge/discharge cycle
        sim.build(initial_soc=0.0)
        sol = sim.step(dt=1E5,starting_solution=None, inputs={"Power [W]": -power_cell})
        sol.termination = "final time" # trick!
        end_time = sol["Time [h]"].data[-1]
        total_energy_ch = power_cell * end_time
        print(f"Total energy charge is: {total_energy_ch:.2f} [W.h]")
        sol = sim.step(dt=1E5,starting_solution=self.sol, inputs={"Power [W]":power_cell})
        total_energy_dis = power_cell * (sol["Time [h]"].data[-1] - end_time)
        print(f"Total energy discharge is: {total_energy_dis:.2f} [W.h]")
        sol.plot(["Power [W]", "Voltage [V]", "Current [A]"])
        
        # determine battery pack parameters
        eta_round_trip = total_energy_dis / total_energy_ch
        self.eta_ch, self.eta_dis = eta_round_trip**0.5, eta_round_trip**0.5
        self.n_cell = math.ceil(self.e_max*1E3*2/(total_energy_dis+total_energy_ch))
        
    def update_heat_transfer_coefficient(self, val_new):
        '''
        update parameters for simulation

        '''
        print(f"By default: {self.parameter_values['Total heat transfer coefficient [W.m-2.K-1]']}")
        self.parameter_values["Total heat transfer coefficient [W.m-2.K-1]"] = val_new
        
    def charge(self, power = 0.0, length_t = 0.25, temp_ambient = 25):
        '''
        compute temperature and soc from charging power
        
        '''
        power_cell = power / self.n_cell * 1E3 # in Watt
        if power > 0:
            experiment = pybamm.Experiment([f"Charge at {power_cell} W for {length_t*60} minutes"])
        elif power < 0:
            experiment = pybamm.Experiment([f"Discharge at {-power_cell} W for {length_t*60} minutes"])
        else:
            experiment = pybamm.Experiment([f"Rest for {length_t*60} minutes"])
        self.experiment = experiment
        self.parameter_values["Ambient temperature [K]"] = temp_ambient + 273.15    
        self.parameter_values["Initial temperature [K]"] = self.temp_cell + 273.15  
        sim = pybamm.Simulation(self.model, parameter_values = self.parameter_values, experiment = experiment)
        self.sim = sim
        try:
            if self.sol == None:
                sim.build_for_experiment(initial_soc = self.soc)  
                sol = sim.solve()
                self.hist_voltage = np.append(self.hist_voltage, sol["Voltage [V]"].entries[0])
            else:
                sol = sim.solve(starting_solution = self.sol)
            self.sol = sol
            traj_temp = sol["Cell temperature [C]"].entries[0,:]
            traj_soc = self.soc_init - sol["Discharge capacity [A.h]"].data \
                / self.parameter_values["Nominal cell capacity [A.h]"]  
            traj_voltage = sol["Voltage [V]"].entries
            self.temp_cell = traj_temp[-1]
            self.soc = traj_soc[-1]
            self.voltage = traj_voltage[-1]
            self.hist_soc = np.append(self.hist_soc, self.soc)
            self.hist_voltage = np.append(self.hist_voltage, self.voltage)
            self.hist_temp_cell = np.append(self.hist_temp_cell, self.temp_cell)
            self.hist_dt_temp_cell = traj_temp   
            self.hist_dt_soc = traj_soc       
            self.hist_dt_power = sol["Power [W]"].entries     
            self.hist_dt_current = sol["Current [A]"].entries      
            self.hist_dt_voltage = traj_voltage
        except:
            print("Solver fails to converge.")

    
    def charge_ts(self, arr_power, arr_temp_ambient, length_t = 0.25):
        if len(arr_power) != len(arr_temp_ambient) or len(arr_temp_ambient) <= 1:
            raise Exception("Check lengths of arrays.")
        
        for i in tqdm(range(len(arr_power))):
            power, temp_ambient = arr_power[i], arr_temp_ambient[i]
            self.charge(power = power, temp_ambient = temp_ambient, length_t = length_t)
        
if __name__ == "__main__":
    bat1 = Battery(soc = 0.2, params_cell="Chen2020")

    #traj_temp1, traj_soc1 = bat1.charge(power = 10)
    '''
    bat1.charge_ts(arr_power = [5,6,5,6,5,6], arr_temp_ambient=[25,25,25,25,25,25])
    
    plt.figure()
    plt.plot(bat1.hist_soc)
    plt.show()
    
    plt.figure()
    plt.plot(bat1.hist_dt_temp_cell)
    plt.show()
    
    plt.figure()
    plt.plot(bat1.hist_dt_voltage)
    plt.show()
    '''
    
    

    
    