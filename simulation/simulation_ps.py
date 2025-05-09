import sys
sys.path.append('../meas_opt')
from battery import Battery
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import cvxpy as cp

def func_ps(bat, load):
    
    # parameters
    p_max = bat.p_max
    e_max = bat.e_max
    n_step = len(load)
    eta_ch = bat.eta_ch
    eta_dis = bat.eta_dis
    soc = bat.soc
    
    # variables
    p_ch = cp.Variable(n_step, nonneg = True)
    p_dis = cp.Variable(n_step, nonneg = True)
    e = cp.Variable(n_step+1)
    delta = cp.Variable(n_step, boolean = True)
    
    # constraints
    cons = []
    cons += [p_ch <= p_max * delta]
    cons += [p_dis <= p_max * delta]
    cons += [e[0] == e_max * soc]
    for t in range(n_step):
        cons += [e[t+1] == e[t] + eta_ch * p_ch[t] - p_dis[t] / eta_dis]
    cons += [e[1:] <= bat.soc_max * e_max]
    cons += [e[1:] >= bat.soc_min * e_max]
    # objective
    load_net = load + p_ch - p_dis
    obj = cp.Minimize(cp.sum_squares(load_net))
    prob = cp.Problem(obj, cons)
    prob.solve()

    return p_ch.value, p_dis.value, e.value

n_step = 24
bat = Battery()
bat.update_heat_transfer_coefficient(0.0)
load = 10 * np.array([
    0.41, 0.36, 0.34, 0.33, 0.36, 0.49, 0.69, 0.84, 0.96, 1.01, 0.94, 0.89,
    0.84, 0.82, 0.86, 0.91, 1.03, 0.96, 0.81, 0.72, 0.61, 0.56, 0.49, 0.46
])

p_ch, p_dis, e = func_ps(bat, load)

plt.figure(figsize = (5,3))
plt.plot(load, label = "load")
plt.plot(p_ch-p_dis, label = "charge")
plt.plot(load + p_ch-p_dis, label = "net")
plt.legend()
plt.grid()
plt.show()



summer_temp_profile = np.array([
    22, 21, 20, 19, 18, 18, 19, 21, 24, 27, 30, 32,
    34, 36, 36, 34, 33, 31, 29, 27, 25, 24, 23, 22
])

bat.charge_ts(p_ch - p_dis, summer_temp_profile, length_t = 1.0)

plt.figure(figsize = (5,3))
plt.plot(bat.hist_dt_voltage, label = "voltage")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize = (5,3))
plt.plot(bat.hist_dt_power, label = "power")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize = (5,3))
plt.plot(bat.hist_soc, label = "soc-pybamm")
plt.plot(e/bat.e_max, label = "soc")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize = (5,3))
plt.plot(summer_temp_profile, label = "ambient temperature")
plt.plot(bat.hist_temp_cell, label = "temperature")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize = (5,3))
plt.plot(bat.hist_dt_temp_cell, label = "temperature")
plt.legend()
plt.grid()
plt.show()