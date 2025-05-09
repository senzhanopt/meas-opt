import sys
sys.path.append('../meas_opt')
from battery import Battery
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
plt.style.use(['science', 'nature'])
import simbench as sb
import pandapower as pp
import pandas as pd
import copy
from power_grid_model_io.converters import PandaPowerConverter
from power_grid_model import ComponentType, DatasetType, initialize_array, PowerGridModel
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import time

def powerflow(p_load, q_load, p_sgen, q_sgen, p_storage, q_storage, update_data, pgm):
    update_data['sym_load']['p_specified'] = np.concatenate((p_load, p_storage))*1E3
    update_data['sym_load']['q_specified'] = np.concatenate((q_load, q_storage))*1E3
    update_data['sym_gen']['p_specified'] = p_sgen * 1E3
    update_data['sym_gen']['q_specified'] = q_sgen * 1E3
    pgm.update(update_data = update_data)
    output_data = pgm.calculate_power_flow(symmetric=True)
    dict_return = {}
    dict_return["v"] = output_data['node']['u_pu'][1:]
    dict_return["loading"] = output_data['transformer']['loading'][0]
    dict_return["P_trafo"] = output_data['transformer']['p_from']*1E-3
    dict_return["Q_trafo"] = output_data['transformer']['q_from']*1E-3
    dict_return["p_net"] = output_data['node']['p']*1E-3
    dict_return["q_net"] = output_data['node']['q']*1E-3
    return dict_return

# simulation parameters
start, end = 224, 252
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
net.sgen = pd.read_excel('sgen.xlsx', index_col = 0).iloc[0:n_pv,:]
net.storage = pd.read_excel('storage.xlsx', index_col = 0).iloc[0:n_storage,:]
#pp.plotting.to_html(net, 'grid.html')
n_bus = len(net.bus) # 97 bus
n_load = len(net.load)
n_load_storage = n_storage + n_load
s_pv = net.sgen.sn_mva.to_numpy() * 1E3
s_storage = net.storage.sn_mva.to_numpy() * 1E3
e_storage = net.storage.max_e_mwh.to_numpy() * 1E3
s_trafo = net.trafo.sn_mva[0] * 1E3 

# read load profiles
prof = sb.get_absolute_values(net, 1)
load_p = prof[('load', 'p_mw')].iloc[start*96:end*96,:].to_numpy() * 1E3
load_q = prof[('load', 'q_mvar')].iloc[start*96:end*96,:].to_numpy() * 1E3
# match sgen and storage profiles
sgen_p = sb.get_absolute_profiles_from_relative_profiles(net, 'sgen', 'sn_mva').iloc[start*96:end*96,:].to_numpy() * 1E3
storage_p = sb.get_absolute_profiles_from_relative_profiles(net, 'storage', 'sn_mva').iloc[start*96:end*96,:].to_numpy() * 1E3

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



