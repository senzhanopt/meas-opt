import pybamm

options = {"thermal": "lumped", "operating mode": "power"}
model = pybamm.lithium_ion.DFN(options=options)
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update({"Power function [W]": pybamm.InputParameter("Power [W]")}, check_already_exists=False)


solver = pybamm.IDAKLUSolver()
sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)

sol = sim.step(dt=1, starting_solution=None, inputs={"Power [W]": 15})
sol = sim.step(dt=1, starting_solution=sol, inputs={"Power [W]": 10})

sol.plot(["Power [W]", "Voltage [V]", "Current [A]"])