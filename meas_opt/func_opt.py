import cvxpy as cp

def func_ps(bat, load, soc_min = 0.0, soc_max = 1.0):
    
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
    cons += [e[1:] <= soc_max * e_max]
    cons += [e[1:] >= soc_min * e_max]
    # objective
    load_net = load + p_ch - p_dis
    obj = cp.Minimize(cp.sum_squares(load_net))
    prob = cp.Problem(obj, cons)
    prob.solve()

    return p_ch.value, p_dis.value, e.value

def opt_determ():
    pass

def opt_chance():
    pass