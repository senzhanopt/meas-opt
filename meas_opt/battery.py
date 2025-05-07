import pybamm
import math
import matplotlib.pyplot as plt
import numpy as np

class Battery:
    
    def __init__(self, p_max = 15.0, e_max = 10.0, soc = 0.5, temp_cell = 25, 
                 cell_model = "dfn", thermal_model = "lumped", 
                 params_cell = "Chen2020"):
        self.p_max = p_max
        self.e_max = e_max
        self.soc = soc
        self.soc_init = soc
        self.temp_cell = temp_cell
        self.cell_model = cell_model
        self.thermal_model = thermal_model
        self.params_cell = params_cell
        self.hist_soc = np.array([soc])
        self.hist_temp_cell = np.array([temp_cell])
        self.sol = None
        self.initialize()
        
    def initialize(self):    
        
        '''
        determine number of cells and efficiencies
        
        '''
        # charging experiment
        options = {"cell geometry": "pouch", "thermal": self.thermal_model}
        if self.cell_model == "dfn":
            self.model = pybamm.lithium_ion.DFN(options)
        else:
            raise Exception("Cell model is not DFN.")
        self.parameter_values = pybamm.ParameterValues(self.params_cell)
        self.v_low_cut = self.parameter_values["Lower voltage cut-off [V]"]
        self.v_upp_cut = self.parameter_values["Upper voltage cut-off [V]"]
        experiment_ch = pybamm.Experiment([f"Charge at C/2 until {self.v_upp_cut} V", 
                                           f"Hold at {self.v_upp_cut} V until C/50"])
        sim_ch = pybamm.Simulation(self.model, parameter_values = 
                                   self.parameter_values, experiment = 
                                   experiment_ch)
        sim_ch.build_for_experiment(initial_soc = 0.0)
        sol_ch = sim_ch.solve()
        total_energy_ch = -sol_ch["Power [W]"].data.mean() * sol_ch["Time [h]"].data[-1]
        #print(f"Total energy charge is: {total_energy_ch:.2f} [W.h]")
        
        # discharging experiment
        experiment_dis = pybamm.Experiment([f"Discharge at C/2 until {self.v_low_cut} V"])
        sim_dis = pybamm.Simulation(self.model, parameter_values = 
                                   self.parameter_values, experiment = 
                                   experiment_dis)
        sim_dis.build_for_experiment(initial_soc = 1.0)
        sol_dis = sim_dis.solve()
        total_energy_dis = sol_dis["Power [W]"].data.mean() * sol_dis["Time [h]"].data[-1]
        #print(f"Total energy discharge is: {total_energy_dis:.2f} [W.h]") 
        
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
        self.parameter_values["Ambient temperature [K]"] = temp_ambient + 273.15    
        self.parameter_values["Initial temperature [K]"] = self.temp_cell + 273.15  
        sim = pybamm.Simulation(self.model, parameter_values = self.parameter_values, experiment = experiment)
        try:
            if self.sol == None:
                sim.build_for_experiment(initial_soc = self.soc)  
                sol = sim.solve()
            else:
                sol = sim.solve(starting_solution = self.sol)
            self.sol = sol
            traj_temp = sol["Cell temperature [C]"].entries[0,:]
            traj_soc = self.soc_init - sol["Discharge capacity [A.h]"].data \
                / self.parameter_values["Nominal cell capacity [A.h]"]           
            self.temp_cell = traj_temp[-1]
            self.soc = traj_soc[-1]
            self.hist_soc = np.append(self.hist_soc, self.soc)
            self.hist_temp_cell = np.append(self.hist_temp_cell, self.temp_cell)
            self.hist_dt_temp_cell = traj_temp   
            self.hist_dt_soc = traj_soc       
            self.hist_dt_power = sol["Power [W]"].entries     
            self.hist_dt_current = sol["Current [A]"].entries      
            self.hist_dt_voltage = sol["Voltage [V]"].entries
        except:
            print("Solver fails to converge.")

    
    def charge_ts(self, arr_power, arr_temp_ambient, length_t = 0.25):
        if len(arr_power) != len(arr_temp_ambient) or len(arr_temp_ambient) <= 1:
            raise Exception("Check lengths of arrays.")
        
        for power, temp_ambient in zip(arr_power, arr_temp_ambient):
            self.charge(power = power, temp_ambient = temp_ambient, length_t = length_t)
        
if __name__ == "__main__":
    bat1 = Battery(soc = 0.2, params_cell="Chen2020")
    bat1.initialize()
    #traj_temp1, traj_soc1 = bat1.charge(power = 10)
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
    
    
    

    
    