import sys
sys.path.append('../battery')
from battery import Battery
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import simbench as sb
import pandapower as pp
import pandas as pd
import copy
from power_grid_model_io.converters import PandaPowerConverter
from power_grid_model import ComponentType, DatasetType, initialize_array, PowerGridModel
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import cvxpy as cp
from sensitivity import sensitivity_voltage, sensitivity_thermal


def powerflow(p_load, q_load, p_sgen, q_sgen, p_storage, q_storage):
    update_data['sym_load']['p_specified'] = np.concatenate((p_load, p_storage))*1E3
    update_data['sym_load']['q_specified'] = np.concatenate((q_load, q_storage))*1E3
    update_data['sym_gen']['p_specified'] = p_sgen * 1E3
    update_data['sym_gen']['q_specified'] = q_sgen * 1E3
    pgm.update(update_data = update_data)
    output_data = pgm.calculate_power_flow(symmetric=True)
    dict_return = {}
    dict_return["v"] = output_data['node']['u_pu'][1:]
    dict_return["loading"] = output_data['transformer']['loading'][0]
    dict_return["p_trafo"] = output_data['transformer']['p_from']*1E-3
    dict_return["q_trafo"] = output_data['transformer']['q_from']*1E-3
    dict_return["p_net"] = output_data['node']['p']*1E-3
    dict_return["q_net"] = output_data['node']['q']*1E-3
    return dict_return

def prob_lyapunov(alpha = 0.5, gamma = 0.05):
    
    # cp parameters, can be updated without rebuilding optimizaiton problem
    v_meas = cp.Parameter(n_bus-1) # exclude slack bus
    soc_storage = cp.Parameter(n_storage)
    p_ch_last = cp.Parameter(n_storage)
    q_ch_last = cp.Parameter(n_storage)
    p_max_pv = cp.Parameter(n_pv)
    p_pv_last = cp.Parameter(n_pv)
    q_pv_last = cp.Parameter(n_pv)
    p_trafo = cp.Parameter(1)
    q_trafo = cp.Parameter(1)
    
    v_cell = cp.Parameter(len(idx_storage))
    temp_cell = cp.Parameter(len(idx_storage))
    temp_ambient = cp.Parameter(1)
    voltage_sensitivity = cp.Parameter(len(idx_storage))
     
    # variables
    p_ch = cp.Variable(n_storage)
    q_ch = cp.Variable(n_storage)
    p_pv = cp.Variable(n_pv, nonneg = True)
    q_pv = cp.Variable(n_pv)
    
    # constraints
    cons = []
    for i in range(n_storage):
        cons += [cp.square(p_ch[i]) + cp.square(q_ch[i]) <= s_storage[i]**2]
        cons += [p_ch[i] * itr_length * eta_ch <= (soc_max-soc_storage[i])*e_storage[i]]
        cons += [p_ch[i] * itr_length >= eta_dis * (soc_min-soc_storage[i])*e_storage[i]]
    
    for i in range(n_pv):
        cons += [cp.square(p_pv[i]) + cp.square(q_pv[i]) <= s_pv[i]**2]
    cons += [p_pv <= p_max_pv]
        
    cons += [v_meas + mat_R_storage @ (p_ch - p_ch_last)
                    + mat_X_storage @ (q_ch - q_ch_last)
                    + mat_R_pv @ (p_pv - p_pv_last)
                    + mat_X_pv @ (q_pv - q_pv_last) <= v_upp*np.ones(n_bus-1)]
    cons += [v_meas + mat_R_storage @ (p_ch - p_ch_last)
                    + mat_X_storage @ (q_ch - q_ch_last)
                    + mat_R_pv @ (p_pv - p_pv_last)
                    + mat_X_pv @ (q_pv - q_pv_last) >= v_low**np.ones(n_bus-1)]

    #cons += [cp.square(p_trafo+cp.sum(p_ch)-cp.sum(p_ch_last)-cp.sum(p_pv)+cp.sum(p_pv_last)) 
    #         + cp.square(q_trafo+cp.sum(q_ch)-cp.sum(q_ch_last)-cp.sum(q_pv)+cp.sum(q_pv_last)) <= scale**2 * s_trafo**2 ]
    
    cons += [ cp.SOC(scale*s_trafo, cp.hstack([ p_trafo+cp.sum(p_ch)-cp.sum(p_ch_last)-cp.sum(p_pv)+cp.sum(p_pv_last),
                                   q_trafo+cp.sum(q_ch)-cp.sum(q_ch_last)-cp.sum(q_pv)+cp.sum(q_pv_last) ]))]
    
    if control_voltage:
        for i,j in enumerate(idx_storage):
            cons += [v_cell[i] + voltage_sensitivity[i] * p_ch[j] <= v_cell_max]
            cons += [v_cell[i] + voltage_sensitivity[i] * p_ch[j] >= v_cell_min]
            
    if control_thermal:
        for i,j in enumerate(idx_storage):
            coeff1, coeff2 = sensitivity_thermal(s_storage[j])
            cons += [temp_cell[i] + (temp_cell[i]-temp_ambient)*coeff1 + cp.square(p_ch[j])*coeff2 <= temp_cell_max]
                
    # objective
    #obj = cp.Minimize(cp.sum_squares(p_max_pv-p_pv) + 0.1 * cp.sum_squares(q_pv)
    #                  + 0.1 * cp.sum_squares(p_ch) + 0.1 * cp.sum_squares(q_ch) 
    #                  + 0.05 * cp.sum(cp.multiply(cp.multiply(soc_storage-soc_init,e_storage),p_ch)))
    
    grad_p_pv = 1.0 * (p_pv_last - p_max_pv)
    grad_q_pv = 0.1 * q_pv_last
    grad_p_ch = 0.1 * p_ch_last
    grad_q_ch = 0.1 * q_ch_last
    grad_p_ch += gamma * cp.multiply(soc_storage-soc_init,e_storage)
    
    obj = cp.Minimize( 
          cp.sum_squares(p_pv-(p_pv_last - alpha*grad_p_pv))
        + cp.sum_squares(q_pv-(q_pv_last - alpha*grad_q_pv))
        + cp.sum_squares(p_ch-(p_ch_last - alpha*grad_p_ch))
        + cp.sum_squares(q_ch-(q_ch_last - alpha*grad_q_ch)) 
        )
    
    prob = cp.Problem(obj, cons)
    
    return {"prob": prob,
            "p_ch": p_ch,
            "q_ch": q_ch,
            "p_pv": p_pv,
            "q_pv": q_pv,
            "v_meas": v_meas,
            "v_cell": v_cell,
            "temp_cell": temp_cell,
            "temp_ambient": temp_ambient,
            "voltage_sensitivity": voltage_sensitivity,
            "p_trafo": p_trafo,
            "q_trafo": q_trafo,
            "soc_storage": soc_storage,
            "p_max_pv": p_max_pv,
            "p_ch_last": p_ch_last,
            "q_ch_last": q_ch_last,
            "p_pv_last": p_pv_last,
            "q_pv_last": q_pv_last
            }

def solve_prob_lyapunov(v_meas, v_cell, voltage_sensitivity, temp_cell,temp_ambient, p_trafo, q_trafo, soc_storage, p_max_pv,
                        p_ch_last, q_ch_last, p_pv_last, q_pv_last):
    prob_lyapunov["v_meas"].value = v_meas
    prob_lyapunov["v_cell"].value = v_cell
    prob_lyapunov["temp_cell"].value = temp_cell
    prob_lyapunov["temp_ambient"].value = temp_ambient
    prob_lyapunov["voltage_sensitivity"].value = voltage_sensitivity
    prob_lyapunov["p_trafo"].value = p_trafo
    prob_lyapunov["q_trafo"].value = q_trafo
    prob_lyapunov["soc_storage"].value = soc_storage
    prob_lyapunov["p_max_pv"].value = p_max_pv
    prob_lyapunov["p_ch_last"].value = p_ch_last
    prob_lyapunov["q_ch_last"].value = q_ch_last
    prob_lyapunov["p_pv_last"].value = p_pv_last
    prob_lyapunov["q_pv_last"].value = q_pv_last
    try:
        prob_lyapunov["prob"].solve(solver = cp.GUROBI)
    except:
        return p_ch_last, q_ch_last, p_pv_last, q_pv_last
    else:
        if prob_lyapunov["prob"].status == "optimal":
            return prob_lyapunov["p_ch"].value, prob_lyapunov["q_ch"].value, \
            prob_lyapunov["p_pv"].value, prob_lyapunov["q_pv"].value
        else:            
            return p_ch_last, q_ch_last, p_pv_last, q_pv_last


#%% similation
control_voltage, control_thermal = [True] * 2
tight = False
alpha = 5.0
gamma = 0.05
# simulation parameters
start, end = 224, 225
n_timesteps = (end-start)*96
n_itr = 3 # 5-minute control resolution
itr_length = 0.25 / n_itr # in hour

# grid parameters
n_pv = 54
n_storage = 36
list_bus_visual = [55,66,59,70]
v_upp = 1.05
v_low = 0.95

# read pp net object from simbench
net = sb.get_simbench_net('1-LV-rural2--2-no_sw')
net.ext_grid.vm_pu = 1.0
net.trafo.tap_pos = 0
net.trafo.sn_mva *= 1.6 # increase the trafo capacity from 250 to 400 kVA
net.sgen = pd.read_excel('data/sgen.xlsx', index_col = 0).iloc[0:n_pv,:]
net.storage = pd.read_excel('data/storage.xlsx', index_col = 0).iloc[0:n_storage,:]
#pp.plotting.to_html(net, 'grid.html')
n_bus = len(net.bus) # 97 bus
n_load = len(net.load)
n_load_storage = n_storage + n_load
s_pv = net.sgen.sn_mva.to_numpy() * 1E3
s_storage = net.storage.sn_mva.to_numpy() * 1E3
e_storage = net.storage.max_e_mwh.to_numpy() * 1E3
s_trafo = net.trafo.sn_mva[0] * 1E3 
soc_min, soc_max, soc_init = 0.0, 1.0, 0.0
eta_ch, eta_dis = 0.97, 0.97
v_cell_min, v_cell_max = 2.5, 4.2
temp_cell_max = 45

# read load profiles
prof = sb.get_absolute_values(net, 1)
load_p = prof[('load', 'p_mw')].iloc[start*96:end*96,:].to_numpy() * 1E3
load_q = prof[('load', 'q_mvar')].iloc[start*96:end*96,:].to_numpy() * 1E3
# match sgen and storage profiles
sgen_p = sb.get_absolute_profiles_from_relative_profiles(net, 'sgen', 'sn_mva').iloc[start*96:end*96,:].to_numpy() * 1E3
storage_p = sb.get_absolute_profiles_from_relative_profiles(net, 'storage', 'sn_mva').iloc[start*96:end*96,:].to_numpy() * 1E3

# electrochemical simulation of batteries
daily_temperature = pd.read_csv('data/temperature_5minute.csv')["temp (C)"].to_numpy()

# use pgm
net_pgm = copy.deepcopy(net)
net_pgm["storage"] = net_pgm["storage"].iloc[:0] #DELETE STORAGE because PGM does not support storage
for i in range(n_storage):
    pp.create_load(net_pgm, bus = net.storage.bus[i], p_mw = 0.0, q_mvar = 0.0)
converter = PandaPowerConverter()
input_data, extra_info = converter.load_input_data(net_pgm)
pgm = PowerGridModel(input_data)
update_sym_load = initialize_array(DatasetType.update, ComponentType.sym_load, n_load_storage )
update_sym_load["id"] = input_data["sym_load"]['id'][:n_load_storage ]  # same ID
update_sym_gen = initialize_array(DatasetType.update, ComponentType.sym_gen, n_pv )
update_sym_gen["id"] = input_data["sym_gen"]['id']  # same ID
update_data = {ComponentType.sym_load: update_sym_load, ComponentType.sym_gen: update_sym_gen}

# read constant network sensitivities
mat_R_storage = pd.read_csv('data/mat_R_storage.csv', index_col = 0).to_numpy()
mat_X_storage = pd.read_csv('data/mat_X_storage.csv', index_col = 0).to_numpy()
mat_R_pv = pd.read_csv('data/mat_R_pv.csv', index_col = 0).to_numpy()
mat_X_pv = pd.read_csv('data/mat_X_pv.csv', index_col = 0).to_numpy()


# store all iterates
mat_p_storage = np.zeros((n_timesteps*n_itr, n_storage))
mat_q_storage = np.zeros((n_timesteps*n_itr, n_storage))
mat_soc_storage = np.zeros((n_timesteps*n_itr+1, n_storage))
mat_p_pv = np.zeros((n_timesteps*n_itr, n_pv))
mat_q_pv = np.zeros((n_timesteps*n_itr, n_pv))
mat_v = np.ones((n_timesteps*n_itr, n_bus-1))
mat_loading = np.zeros(n_timesteps*n_itr)
mat_p_trafo = np.zeros(n_timesteps*n_itr)


# initialize
p_pv = copy.deepcopy(sgen_p[0,:]) 
q_pv = np.zeros(n_pv) 
p_storage = np.zeros(n_storage) 
q_storage = np.zeros(n_storage)
soc = soc_init * np.ones(n_storage)
mat_soc_storage[0,:] = soc

idx_storage = range(n_storage)
mat_temp_cell = np.zeros((n_timesteps*n_itr+1, len(idx_storage)))
mat_voltage = np.zeros((n_timesteps*n_itr+1, len(idx_storage)))
l_storage = []
for i in idx_storage:
    battery = Battery(p_max=s_storage[i],e_max=e_storage[i],soc=soc_init,temp_cell=daily_temperature[0])
    battery.update_heat_transfer_coefficient(0.005,0.5)
    l_storage.append(battery)

# build optimization problem instance
if tight:
    scale = 0.98
else:
    scale = 1.0
prob_lyapunov = prob_lyapunov(alpha = alpha, gamma = gamma)

# iterating process
for itr in tqdm(range(n_timesteps * n_itr)):
    if itr % n_itr == 0:
        load_p_current = load_p[itr//n_itr,:]
        load_q_current = load_q[itr//n_itr,:]
        sgen_p_current = sgen_p[itr//n_itr,:]
    
    # pv and storage setpoint projection
    p_pv = np.minimum(p_pv, sgen_p_current)
    if itr <= 108 or itr >= 192: # PV control between 9-16 hr
        p_pv = sgen_p_current * 1.00
    for i in range(n_storage):
        if p_storage[i] >= 0:
            p_max_ch = (soc_max-soc[i])*e_storage[i]/(itr_length*eta_ch)
            p_storage[i] = min(p_storage[i], p_max_ch)
        else:
            p_max_dis = (soc[i]-soc_min)*e_storage[i]*eta_dis/itr_length
            p_storage[i] = max(p_storage[i], -p_max_dis)
            
    for j,i in enumerate(idx_storage):
        l_storage[j].charge(p_storage[i],itr_length,daily_temperature[itr])
    v_cell = [l_storage[i].voltage for i in range(len(idx_storage))]
    temp_cell = [l_storage[i].temp_cell for i in range(len(idx_storage))]
    temp_ambient = np.array([daily_temperature[itr]])
    voltage_sensitivity = [sensitivity_voltage(l_storage[i].soc, l_storage[i].p_max) for i in range(len(idx_storage))]

    # save iterates
    mat_p_storage[itr,:] = p_storage
    mat_q_storage[itr,:] = q_storage
    soc += (eta_ch*p_storage*(p_storage>=0)+p_storage*(p_storage<=0)/eta_dis)*itr_length/e_storage
    mat_soc_storage[itr+1,:] = soc
    mat_p_pv[itr,:] = p_pv
    mat_q_pv[itr,:] = q_pv

    # grid calculation
    dict_return = powerflow(load_p_current, load_q_current, p_pv, q_pv, p_storage, q_storage)
    v = dict_return["v"]
    loading = dict_return["loading"]
    p_trafo = dict_return["p_trafo"]
    q_trafo = dict_return["q_trafo"]
    pf_trafo = p_trafo / np.sqrt(p_trafo**2+q_trafo**2)
    rpf_trafo = q_trafo / np.sqrt(p_trafo**2+q_trafo**2)
    mat_v[itr, :] = v
    mat_loading[itr] = loading
    mat_p_trafo[itr] = p_trafo
    
    # call optimization
    p_storage, q_storage, p_pv, q_pv = solve_prob_lyapunov(v,v_cell,voltage_sensitivity,temp_cell,temp_ambient,
                                                           p_trafo, q_trafo, soc, sgen_p_current,
                                                           p_storage, q_storage, p_pv, q_pv)
    
    

for j in range(len(idx_storage)):
    mat_voltage[:,j] = l_storage[j].hist_voltage
    mat_temp_cell[:,j] = l_storage[j].hist_temp_cell
    
if False:
    mat_v0 = np.ones((n_timesteps*n_itr, n_bus-1))
    mat_loading0 = np.zeros(n_timesteps*n_itr)
    for itr in tqdm(range(n_timesteps * n_itr)):
        if itr % n_itr == 0:
            load_p_current = load_p[itr//n_itr,:]
            load_q_current = load_q[itr//n_itr,:]
            sgen_p_current = sgen_p[itr//n_itr,:]
        dict_return0 = powerflow(load_p_current, load_q_current, sgen_p_current,
                                np.zeros(n_pv), np.zeros(n_storage), np.zeros(n_storage) )
        v0 = dict_return0["v"]
        loading0 = dict_return0["loading"]
        mat_v0[itr, :] = v0
        mat_loading0[itr] = loading0
    pd.DataFrame(mat_v0).to_csv('result/' + 'mat_v0' +'.csv')
    pd.DataFrame(mat_loading0).to_csv('result/' + 'mat_loading0' +'.csv')

if True:
    # visualization
    for b in list_bus_visual:
        plt.plot(mat_v[:,b-1], label = f'bus {b}')
    plt.legend()
    plt.title("voltage")
    plt.show()    
    
    plt.plot(mat_loading[:], label = "trafo")
    plt.legend()
    plt.title("loading")
    plt.show()
    
    plt.plot(mat_p_trafo[:], label = "trafo")
    plt.legend()
    plt.title("p trafo")
    plt.show()
    
    for i in range(4):
        plt.plot(mat_p_pv[:,i], label = f'pv {i}')
    plt.legend()
    plt.title("pv")
    plt.show()    
    
    for i in range(4):
        plt.plot(mat_p_storage[:,i], label = f'storage {i}')
    plt.legend()
    plt.title("storage")
    plt.show()    
    
    
    for i in range(4):
        plt.plot(mat_soc_storage[:,i], label = f'storage {i}')
    plt.legend()
    plt.title("soc")
    plt.show()
    
    for i in range(len(idx_storage)):
        plt.plot(mat_voltage[:, i], label = f'storage {i}')
    plt.legend()
    plt.title("voltage")
    plt.axhline(y=4.2, color='r', linestyle='--', linewidth=2)
    plt.ylim(2.4,4.3)
    plt.show()        

    for i in range(len(idx_storage)):
        plt.plot(mat_temp_cell[:, i], label = f'storage {i}')
    plt.plot(daily_temperature, label = "ambient")
    plt.axhline(y=45, color='r', linestyle='--', linewidth=2)
    plt.legend()
    plt.title("temperature")
    plt.show()        
        
        
        
    name = "_lyapunov"
    if control_voltage:
        name += '_voltage'
    if control_thermal:
        name += '_thermal'
    if tight:
        name += '_tight'
    if alpha != 0.5:
        name += f"_alpha{alpha}"
    if gamma != 0.05:
        name += f"_gamma{gamma}"
    pd.DataFrame(mat_p_pv).to_csv('result/' + 'mat_p_pv' + name +'.csv')
    pd.DataFrame(mat_q_pv).to_csv('result/' + 'mat_q_pv' + name +'.csv')
    pd.DataFrame(mat_v).to_csv('result/' + 'mat_v' + name +'.csv')
    pd.DataFrame(mat_voltage).to_csv('result/' + 'mat_voltage' + name +'.csv')
    pd.DataFrame(mat_temp_cell).to_csv('result/' + 'mat_temp_cell' + name +'.csv')
    pd.DataFrame(mat_loading).to_csv('result/' + 'mat_loading' + name +'.csv')
    pd.DataFrame(mat_p_storage).to_csv('result/' + 'mat_p_storage' + name +'.csv')
    pd.DataFrame(mat_q_storage).to_csv('result/' + 'mat_q_storage' + name +'.csv')















