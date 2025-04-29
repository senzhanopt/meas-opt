import pybamm
import matplotlib.pyplot as plt

# include a thermal model
options = {"cell geometry": "pouch", "thermal": "lumped"}
model = pybamm.lithium_ion.DFN(options)

# set parameter values
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values["Ambient temperature [K]"] = 308.15
#parameter_values["Current function [A]"] = 0.68

# set experiment
experiment = pybamm.Experiment(["Charge at 1 A until 4.2 V", "Hold at 4.2 V until 50 mA"])

# set simulation 
sim = pybamm.Simulation(model, parameter_values = parameter_values, experiment = experiment)
sim.build_for_experiment(initial_soc = 0.0)

# extract result
sim.solve([0, 3600])
sol = sim.solution

output_variables = ["Power [W]", "Voltage [V]", "Cell temperature [C]", "Current [A]"]
sim.plot(output_variables=output_variables)

cell_temperature = sol["Cell temperature [C]"].entries
plt.figure()
plt.plot(cell_temperature[0,:])
plt.title("Cell temperature [C]")
plt.xlabel("Time [h]")
plt.show()

soc = 0.0 - sol["Discharge capacity [A.h]"].data[-1] / parameter_values["Nominal cell capacity [A.h]"]
print(f"End SoC is: {soc:.3f}")
