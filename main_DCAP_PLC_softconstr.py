import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time    # [ADDED: ML4SDDP] for solve-time reporting

# Notes:
# 1. The optimization problem should have a minimization orientation.

# [MODIFIED: ML4SDDP] This file now uses the DCAP model variant from our
# ML4SDDP project (binary variant of Ahmed & Garcia 2003):
#   - first stage: x_{jt} in {0,1} (open facility j in period t at cost a_{jt});
#     the old continuous capacity x and the binary indicator u are removed
#   - capacity when open is data b_{jt}; the cumulative capacity at (j,t) is
#     sum_{tau<=t} b_{j,tau} x_{j,tau}
#   - the Lagrangian copy variable z_{jt} is BINARY (binary coupling -> zero
#     integrality gap of the Lagrangian cuts, SDDiP vertex argument)
# All modified/added lines are tagged with [MODIFIED: ML4SDDP] or
# [ADDED: ML4SDDP]. The Level Set inner machinery (solve_lag_dual and its
# auxiliary problems) is untouched.

# [ADDED: ML4SDDP] heuristic top-k scenario selection (Heur-k), see
# utils_heuristic_k.py; enabled/disabled by USE_HEURISTIC_K in __main__
from utils_heuristic_k import HeuristicKSelector

# [ADDED: ML4SDDP] cheap certified Lagrangian multipliers (sb / probe), see
# utils_sb_probe.py; selected by MULTIPLIER_ORACLE in __main__
from utils_sb_probe import SBProbeOracle

# Create a new Gurobi environment
env = gp.Env(empty=True)
env.setParam('LogFile', 'gurobi.log')
env.start()

# [MODIFIED: ML4SDDP] signature changed: b is now the capacity data b_{jt}
# (capacity added when facility j opens in period t); B and u_option are
# removed since the first stage has a single binary variable x_{jt}
def build_extensive_form(omega, a, b, c, d, p, I_len, J_len, T_len):
    # construct the extensive formulation
    extensive_prob = gp.Model("extensive_form")

    # obtain problem dimensions
    J = range(J_len)
    I = range(I_len)
    T = range(T_len)

    # set up the decision variables
    # [MODIFIED: ML4SDDP] x is binary (open facility j in period t); u removed
    x = extensive_prob.addVars(J_len, T_len, vtype=GRB.BINARY, name="x")
    y = extensive_prob.addVars(omega, I_len, J_len, T_len, vtype=GRB.BINARY, name="y")
    s = extensive_prob.addVars(omega, J_len, T_len, vtype=GRB.CONTINUOUS, lb=0, name="s")

    # set up the objective function
    # [MODIFIED: ML4SDDP] first-stage cost is a_{jt} x_{jt} only (u term removed)
    extensive_prob.setObjective(gp.quicksum(a[j][t] * x[j,t] for j in J for t in T) + 1/omega * gp.quicksum(\
            gp.quicksum(p[j][t] * s[o,j,t] + gp.quicksum(c[o][i][j][t] * y[o,i,j,t] for i in I) for j in J for t in T)
         for o in range(omega)), GRB.MINIMIZE)
    # set up the structural constraints
    # [MODIFIED: ML4SDDP] the constraint x <= B*u is removed; the right-hand
    # side is now the cumulative capacity sum_{tau<=t} b_{j,tau} x_{j,tau}
    extensive_prob.addConstrs((gp.quicksum(d[o,i,t] * y[o,i,j,t] for i in I) - s[o,j,t] <= gp.quicksum(b[j][tau] * x[j,tau] for tau in range(t+1))\
                                for j in J for t in T for o in range(omega)),name = "flow_cons")
    extensive_prob.addConstrs((gp.quicksum(y[o,i,j,t] for j in J) == 1 for i in I for t in T for o in range(omega)), name = "demand_cons")
    extensive_prob.update()
    return extensive_prob

# [MODIFIED: ML4SDDP] signature changed: b, B and u_option removed — the
# master only carries the binary x and the scenario value approximations theta
def build_masterproblem(omega, a, J_len, T_len, prob_lb=-100000):
    # construct the master program
    master_prob = gp.Model("masterproblem")
    master_prob.Params.OutputFlag = 0

    # obtain problem dimensions
    J = range(J_len)
    T = range(T_len)

    # set up the decision variables
    # [MODIFIED: ML4SDDP] x is binary; u removed
    x = master_prob.addVars(J_len, T_len, vtype=GRB.BINARY, name="x")
    theta = master_prob.addVars(omega, vtype=GRB.CONTINUOUS, lb=prob_lb, name="theta")
    # [MODIFIED: ML4SDDP] first-stage cost is a_{jt} x_{jt} only; the
    # constraint x <= B*u is removed (no structural first-stage constraints)
    master_prob.setObjective(gp.quicksum(a[j][t] * x[j,t] for j in J for t in T) + 1/omega * gp.quicksum(\
        theta[o] for o in range(omega)), GRB.MINIMIZE)
    master_prob.update()
    return master_prob

# [MODIFIED: ML4SDDP] signature changed: capacity data b (b_{jt}) added
def build_subproblem(o, b, co, do, p, I_len, J_len, T_len, x_value):
    # input:
    # o - index of the scenario, do - processing requirement, co - processing cost, p - penalty cost,
    # b - capacity data b_{jt},                                                   [MODIFIED: ML4SDDP]
    # I_len - number of tasks, J_len - number of resources, T_len - number of time periods,
    # x_value - the optimal solution of the master problem

    sub_prob = gp.Model("subproblem_" + str(o))
    sub_prob.Params.OutputFlag = 0

    # obtain problem dimensions
    J = range(J_len)
    I = range(I_len)
    T = range(T_len)

    # set up the decision variables
    y = sub_prob.addVars(I_len, J_len, T_len, vtype=GRB.BINARY, name="y")
    s = sub_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0, name="s")

    # set up the objective function
    sub_prob.setObjective(gp.quicksum(p[j][t] * s[j,t] + gp.quicksum(co[i][j][t] * y[i,j,t] for i in I) for j in J for t in T), GRB.MINIMIZE)

    # set up the structural constraints
    # [MODIFIED: ML4SDDP] cumulative capacity sum_{tau<=t} b_{j,tau} x_{j,tau}
    sub_prob.addConstrs((gp.quicksum(do[i,t] * y[i,j,t] for i in I) - s[j,t] <= gp.quicksum(b[j][tau] * x_value[j,tau] for tau in range(t+1))\
                                for j in J for t in T),name = "flow_cons")
    sub_prob.addConstrs((gp.quicksum(y[i,j,t] for j in J) == 1 for i in I for t in T), name = "demand_cons")

    sub_prob.update()
    return sub_prob

# [MODIFIED: ML4SDDP] signature changed: capacity data b (b_{jt}) replaces
# the scalar capacity bound B
def build_subproblem_lag(o, b, co, do, p, I_len, J_len, T_len, pi_value):
    # input:
    # o - index of the scenario, do - processing requirement, co - processing cost, p - penalty cost,
    # b - capacity data b_{jt},                                                   [MODIFIED: ML4SDDP]
    # I_len - number of tasks, J_len - number of resources, T_len - number of time periods,
    # pi_value - the Lagrangian dual multipliers

    sub_prob_lag = gp.Model("subproblem_lag_" + str(o))
    sub_prob_lag.Params.OutputFlag = 0

    # obtain problem dimensions
    J = range(J_len)
    I = range(I_len)
    T = range(T_len)

    # set up the auxiliary variables z (copy of x)
    # [MODIFIED: ML4SDDP] z is BINARY (copy of the binary x) — binary coupling
    # gives zero integrality gap of the Lagrangian cuts (SDDiP vertex argument)
    z = sub_prob_lag.addVars(J_len, T_len, vtype=GRB.BINARY, name="z")

    # set up the decision variables
    y = sub_prob_lag.addVars(I_len, J_len, T_len, vtype=GRB.BINARY, name="y")
    s = sub_prob_lag.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0, name="s")

    # set up the objective function with Lagrangian penalty term
    sub_prob_lag.setObjective(gp.quicksum(p[j][t] * s[j,t] + gp.quicksum(co[i][j][t] * y[i,j,t] for i in I) - pi_value[j,t] * z[j,t] for j in J for t in T), GRB.MINIMIZE)
    
    # set up the structural constraints
    # [MODIFIED: ML4SDDP] cumulative capacity sum_{tau<=t} b_{j,tau} z_{j,tau}
    sub_prob_lag.addConstrs((gp.quicksum(do[i,t] * y[i,j,t] for i in I) - s[j,t] <= gp.quicksum(b[j][tau] * z[j,tau] for tau in range(t+1))\
                                for j in J for t in T), name = "flow_cons")
    sub_prob_lag.addConstrs((gp.quicksum(y[i,j,t] for j in J) == 1 for i in I for t in T), name = "demand_cons")

    sub_prob_lag.update()
    return sub_prob_lag

def update_subproblem_lag(sub_prob_lag, co, p, I_len, J_len, T_len, pi_value):
    # obtain problem dimensions
    J = range(J_len)
    I = range(I_len)
    T = range(T_len)

    # set up the objective function with Lagrangian penalty term
    sub_prob_lag.setObjective(gp.quicksum(p[j][t] * sub_prob_lag.getVarByName("s[{},{}]".format(j,t)) + 
                        gp.quicksum(co[i][j][t] * sub_prob_lag.getVarByName("y[{},{},{}]".format(i,j,t)) for i in I) -
                        pi_value[j,t] * sub_prob_lag.getVarByName("z[{},{}]".format(j,t)) for j in J for t in T), GRB.MINIMIZE)
    
    sub_prob_lag.update()
    return sub_prob_lag

# Build the level set lower bound problem
def build_ls_lb_problem(J_len, T_len, x_value, L_value, cutList, x_tilde, prob_lb=-100000, prob_ub=100000):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    lb_prob = gp.Model("lb_problem")
    lb_prob.Params.OutputFlag = 0

    # set up the dual variables pi and auxiliary variables theta
    pi = lb_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=prob_lb, ub=prob_ub, name="pi")
    theta = lb_prob.addVar(vtype=GRB.CONTINUOUS, lb=10000*prob_lb, name="theta")

    # set up the objective function with Lagrangian penalty term
    lb_prob.setObjective(-gp.quicksum(pi[j,t] * x_tilde[j,t] for j in range(J_len) for t in range(T_len)) - theta, GRB.MINIMIZE)

    # set up the structural constraints
    lb_prob.addConstrs((gp.quicksum(cutList[k][0][j,t] * pi[j,t] for j in range(J_len) for t in range(T_len)) + cutList[k][1] >= theta
                                for k in range(len(cutList))), name = "cuts")
    lb_prob.addConstr(gp.quicksum(pi[j,t] * x_value[j,t] for j in range(J_len) for t in range(T_len)) + theta >= L_value, name = "cons")
    lb_prob.update()
    return lb_prob

# update the level set lower bound problem
def update_ls_lb_problem(lb_prob, J_len, T_len, x_tilde, cutList, update_ind_range):
    # input: 
    # lb_prob - the level set lower bound problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    for k in update_ind_range:
        lb_prob.addConstr(gp.quicksum(cutList[k][0][j,t] * lb_prob.getVarByName("pi[{},{}]".format(j,t)) for j in range(J_len) for t in range(T_len)) + \
                          cutList[k][1] >= lb_prob.getVarByName("theta"), name = "cuts[{}]".format(k))
        
    # set up the objective function
    lb_prob.setObjective(-gp.quicksum(lb_prob.getVarByName("pi[{},{}]".format(j,t)) * x_tilde[j,t] for j in range(J_len) for t in range(T_len)) - 
                        lb_prob.getVarByName("theta"), GRB.MINIMIZE)
    
    lb_prob.update()
    return lb_prob

# Build the level set lower bound problem
def build_next_pi_problem(J_len, T_len, level, alpha, x_value, L_value, cutList, x_tilde, prob_lb=-100000, prob_ub=100000):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)
    next_pi_prob = gp.Model("next_pi_prob")
    next_pi_prob.Params.OutputFlag = 0
    # set up the dual variables pi and auxiliary variables theta
    pi = next_pi_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=prob_lb, ub=prob_ub, name="pi")
    theta = next_pi_prob.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="theta")
    pi_obj_abs = next_pi_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=np.maximum(np.abs(prob_ub),np.abs(prob_lb)), name="pi_obj")

    # set up the structural constraints
    next_pi_prob.addConstrs((gp.quicksum(cutList[k][0][j,t] * pi[j,t] for j in range(J_len) for t in range(T_len)) + cutList[k][1] >= theta
                                for k in range(len(cutList))), name = "cuts")
    next_pi_prob.addConstr(alpha * (-gp.quicksum(pi[j,t] * x_tilde[j,t] for j in range(J_len) for t in range(T_len)) - theta) + 
                           (1 - alpha) * (L_value - gp.quicksum(pi[j,t] * x_value[j,t] for j in range(J_len) for t in range(T_len)) - theta) <= level, name = "level_cons")

    # set up the objective function absolute value term
    next_pi_prob.addConstrs((pi_obj_abs[j,t] - pi[j,t] >= 0 for j in range(J_len) for t in range(T_len)), name = "pi_obj_pos")
    next_pi_prob.addConstrs((pi_obj_abs[j,t] + pi[j,t] >= 0 for j in range(J_len) for t in range(T_len)), name = "pi_obj_neg")

    # set up the objective function
    next_pi_prob.setObjective(gp.quicksum(pi_obj_abs[j,t] for j in range(J_len) for t in range(T_len)), GRB.MINIMIZE)

    next_pi_prob.update()
    return next_pi_prob

def update_next_pi_problem(next_pi_prob, J_len, T_len, cutList, update_ind_range, alpha, x_value, L_value, pi_bar_value, level, x_tilde):
    # input: 
    # next_pi_prob - the next pi problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    next_pi_prob.remove(next_pi_prob.getConstrByName("level_cons"))
    next_pi_prob.addConstr(alpha * (-gp.quicksum(next_pi_prob.getVarByName("pi[{},{}]".format(j,t)) * x_tilde[j,t] for j in range(J_len) for t in range(T_len)) - next_pi_prob.getVarByName("theta")) + 
                (1 - alpha) * (L_value - gp.quicksum(next_pi_prob.getVarByName("pi[{},{}]".format(j,t)) * x_value[j,t] for j in range(J_len) for t in range(T_len)) - 
                next_pi_prob.getVarByName("theta")) <= level, name = "level_cons")

    for k in update_ind_range:
        next_pi_prob.addConstr(gp.quicksum(cutList[k][0][j,t] * next_pi_prob.getVarByName("pi[{},{}]".format(j,t)) for j in range(J_len) for t in range(T_len)) + \
                          cutList[k][1] >= next_pi_prob.getVarByName("theta"), name = "cuts[{}]".format(k))

    # set up the objective function absolute value rhs term
    for j in range(J_len):
        for t in range(T_len):
            pos_constr = next_pi_prob.getConstrByName("pi_obj_pos[{},{}]".format(j,t))
            neg_constr = next_pi_prob.getConstrByName("pi_obj_neg[{},{}]".format(j,t))
            next_pi_prob.setAttr("RHS", pos_constr, -pi_bar_value[j,t])
            next_pi_prob.setAttr("RHS", neg_constr, pi_bar_value[j,t])

    next_pi_prob.update()
    return next_pi_prob

def obtain_alpha_bounds_opt(J_len, T_len, pi_list, L_value, x_value, v_underbar, V_list, x_tilde, prob_lb=-100000):
    # input: 
    # alpha_prob - the alpha problem with piecewise linear objective function
    # return the upper and lower bounds of alpha
    alpha_min = 0
    alpha_max = 1

    # set up the alpha problem
    alpha_prob = gp.Model("alpha_problem")
    alpha_prob.Params.OutputFlag = 0
    alpha = alpha_prob.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="alpha")
    alpha_obj = alpha_prob.addVar(vtype=GRB.CONTINUOUS, lb=prob_lb, name="alpha_obj")
    alpha_prob.addConstr(alpha_obj >= 0, name="alpha_obj_lb")
    alpha_prob.addConstrs((alpha_obj <= alpha * ((-sum(pi_list[k][j,t] * x_tilde[j,t] for j in range(J_len) for t in range(T_len)) - V_list[k]) - v_underbar) + \
                            (1 - alpha) * (L_value - np.inner(pi_list[k].flatten(), x_value.flatten()) - V_list[k])
                            for k in range(len(pi_list))), name="alpha_obj_constr")

    alpha_prob.setObjective(alpha, GRB.MINIMIZE)
    alpha_prob.update()
    alpha_prob.optimize()
    alpha_min = np.round(alpha_prob.ObjVal, 5)

    alpha_prob.setObjective(alpha, GRB.MAXIMIZE)
    alpha_prob.update()
    alpha_prob.optimize()
    alpha_max = np.round(alpha_prob.ObjVal, 5)

    # output the Delta
    alpha_prob.setObjective(alpha_obj, GRB.MAXIMIZE)
    alpha_prob.update()
    alpha_prob.optimize()
    Delta = alpha_prob.ObjVal

    return alpha_max, alpha_min, Delta


def maximize_lower_envelope(gamma, eta):
    """
    Find the maximum of the lower envelope (minimum over all lines) of piecewise linear functions.
    
    For each line k: f_k(x) = gamma[k] * x + eta[k]
    The lower envelope is: F(x) = min_k f_k(x)
    This function finds x* that maximizes F(x) on [0, 1].
    
    Parameters:
    -----------
    gamma : numpy array
        Array of slopes for each line
    eta : numpy array
        Array of intercepts for each line
    
    Returns:
    --------
    best_x : float
        The x value that maximizes the lower envelope
    best_val : float
        The maximum value of the lower envelope
    """
    n = len(gamma)
    assert n == len(eta), "gamma and eta must have same length"
    
    # If only one line, trivial: maximize gamma*x + eta on [0,1]
    if n == 1:
        if gamma[0] > 0:
            return 1.0, gamma[0] * 1.0 + eta[0]
        else:
            return 0.0, eta[0]
    
    # Collect candidate x-values
    xs = [0.0, 1.0]  # boundaries always candidates
    
    # Compute pairwise intersections
    for j in range(n):
        for k in range(j + 1, n):
            gj, gk = gamma[j], gamma[k]
            ej, ek = eta[j], eta[k]
            
            if gj != gk:
                x_int = (ek - ej) / (gj - gk)
                if 0.0 <= x_int <= 1.0:
                    xs.append(x_int)
    
    # Remove duplicates and sort
    xs = sorted(list(set(xs)))
    
    # Evaluate envelope on all candidates
    best_x = 0.0
    best_val = float('-inf')
    
    for x in xs:
        # Compute minimum over all lines at x
        v = np.min(gamma * x + eta)
        if v > best_val:
            best_val = v
            best_x = x
    
    return best_x, best_val

def obtain_alpha_bounds(J_len, T_len, pi_list, L_value, x_value, v_underbar, V_list, x_tilde):
    # algebraic way to calculate alpha_max and alpha_min
    alpha_underbar = []
    alpha_bar = []
    gamma_list = {}
    eta_list = {}

    for k in range(len(pi_list)):
        gamma_list[k] = ((-sum(pi_list[k][j,t] * x_tilde[j,t] for j in range(J_len) for t in range(T_len)) - V_list[k]) - v_underbar) - \
                (L_value - np.inner(pi_list[k].flatten(), x_value.flatten()) - V_list[k])
        eta_list[k] = (L_value - np.inner(pi_list[k].flatten(), x_value.flatten()) - V_list[k])
        if gamma_list[k] >= 0:
            alpha_bar.append(1)
            if eta_list[k] >= 0:
                alpha_underbar.append(0)
            else:
                if -eta_list[k] / gamma_list[k] <= 1:
                    alpha_underbar.append(-eta_list[k] / gamma_list[k])
                else:
                    ValueError("alpha_underbar is not feasible")
        else:
            alpha_underbar.append(0)
            if eta_list[k] >= 0:
                if -eta_list[k] / gamma_list[k] < 1:
                    alpha_bar.append(-eta_list[k] / gamma_list[k])
                else:
                    alpha_bar.append(1)
            else:
                ValueError("alpha_bar is not feasible")
    alpha_min = np.round(np.max(alpha_underbar),7)
    alpha_max = np.round(np.min(alpha_bar),7)

    # algebraic way to calculate Delta using maximize_lower_envelope
    gamma_array = np.array([gamma_list[k] for k in range(len(pi_list))])
    eta_array = np.array([eta_list[k] for k in range(len(pi_list))])
    alpha_star, Delta = maximize_lower_envelope(gamma_array, eta_array)

    return alpha_max, alpha_min, Delta

# procedure to solve the Lagrangian dual problem
# [MODIFIED: ML4SDDP] signature changed: capacity data b (b_{jt}) replaces the
# scalar capacity bound B; the Level Set algorithm itself is untouched
def solve_lag_dual(o, b, co, do, p, I_len, J_len, T_len, x_value, L_value, lambda_level, mu_level, x_tilde, tol = 1e-2, cutList = [], sub_lb = -100000, sub_ub = 100000):
    # input: 
    # o - index of the scenario, do - processing requirement, co - processing cost, p - penalty cost,
    # I_len - number of tasks, J_len - number of resources, T_len - number of time periods,
    # x_value - the optimal solution of the master problem
    # cutList - the list of cuts generated for the inner minimization problem so far, 
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # obtain problem dimensions
    J = range(J_len)
    I = range(I_len)
    T = range(T_len)

    # initialize the Lagrangian dual multipliers and cut list for the level set problem
    pi_value = np.zeros((J_len, T_len))
    v_value = 0
    alpha_min = 0
    alpha_max = 1
    alpha = (alpha_max + alpha_min) / 2
    pi_list = []
    V_list = []

    # set up the lower bound problem
    lb_prob = build_ls_lb_problem(J_len, T_len, x_value, L_value, cutList, x_tilde)
    # set up the auxiliary problem to find the next pi_value
    level = sub_ub
    next_pi_prob = build_next_pi_problem(J_len, T_len, level, alpha, x_value, L_value, cutList, x_tilde)

    # build the inner min subproblem with Lagrangian penalty term
    sub_prob = build_subproblem_lag(o, b, co, do, p, I_len, J_len, T_len, pi_value)   # [MODIFIED: ML4SDDP] pass capacity data b
    # solve the subproblem
    sub_prob.optimize()

    # initialize the termination criterion
    cont_bool = True
    counter = 0

    # loop until the termination criterion
    while cont_bool:
        # record the pi_value
        pi_list.append(pi_value)

        # obtain the inner minimization problem's optimal solution
        z_value = np.zeros((J_len, T_len))
        for j in range(J_len):
            for t in range(T_len):
                z_value[j,t] = sub_prob.getVarByName("z[{},{}]".format(j,t)).X

        # add the cut to the cut list
        Vj = sub_prob.ObjVal        # V(\pi_j)
        V_list.append(Vj)
        if len(cutList) > 0:
            theta_j = np.min([np.inner(cutList[k][0].flatten(), pi_value.flatten()) + cutList[k][1] for k in range(len(cutList))])
        else:
            theta_j = np.inf
        # only generate the cut if Vj > theta_j
        if Vj < theta_j - 1e-4:
            cutList.append((-z_value, Vj + np.inner(z_value.flatten(), pi_value.flatten())))
            cutList_update_start_ind = len(cutList) - 1
            cutList_update_end_ind = len(cutList)
        else:
            cutList_update_start_ind = len(cutList)
            cutList_update_end_ind = len(cutList)

        # update and solve the lower bound problem
        lb_prob = update_ls_lb_problem(lb_prob, J_len, T_len, x_tilde, cutList, range(cutList_update_start_ind, cutList_update_end_ind))
        lb_prob.optimize()
        if lb_prob.Status != GRB.OPTIMAL:
            # update the L_value and resolve lb_prob
            L_test_bool = True
            L_value_lb = sub_lb
            L_value_ub = L_value
            while L_test_bool:
                L_value = (L_value_lb + L_value_ub) / 2
                lb_prob.remove(lb_prob.getConstrByName("cons"))
                lb_prob.addConstr(gp.quicksum(lb_prob.getVarByName("pi[{},{}]".format(j,t)) * x_value[j,t] for j in range(J_len) for t in range(T_len)) + 
                                  lb_prob.getVarByName("theta") >= L_value, name = "cons")
                lb_prob.update()
                lb_prob.optimize()
                if lb_prob.Status == GRB.OPTIMAL:
                    L_value_lb = L_value
                    if abs(L_value_ub - L_value_lb) < 1e-2:
                        L_test_bool = False
                else:
                    L_value_ub = L_value

        # update the alpha
        alpha_max, alpha_min, Delta = obtain_alpha_bounds(J_len, T_len, pi_list, L_value, x_value, lb_prob.ObjVal, V_list, x_tilde)
        if counter == 0:
            alpha = (alpha_max + alpha_min) / 2
        else:
            if ((alpha - alpha_min)/(alpha_max - alpha_min) < mu_level/2)|((alpha - alpha_min)/(alpha_max - alpha_min) > 1 - mu_level/2):
                alpha = (alpha_min + alpha_max) / 2

        # update the termination indicator
        if Delta < tol:
            cont_bool = False
        else:
            if counter > 200:
                cont_bool = False
            else:
                counter += 1
                # update the level
                v_bar_list = [alpha * (-sum(pi_list[pi_k][j,t] * x_tilde[j,t] for j in J for t in T) - V_list[pi_k]) + \
                              (1 - alpha) * (L_value - np.inner(pi_list[pi_k].flatten(), x_value.flatten()) - V_list[pi_k]) for pi_k in range(len(pi_list))]
                v_bar = np.min(v_bar_list)
                v_underbar = alpha * lb_prob.ObjVal
                level = lambda_level * v_bar + (1 - lambda_level) * v_underbar

                # solve for the next pi_value
                next_pi_prob = update_next_pi_problem(next_pi_prob, J_len, T_len, cutList, range(cutList_update_start_ind, cutList_update_end_ind), alpha, x_value, L_value, pi_value, level, x_tilde)
                next_pi_prob.optimize()
                # obtain the next pi_value
                # [MODIFIED: ML4SDDP] guard the case where next_pi_prob is not
                # solved to optimality (e.g. the level constraint is
                # infeasible): the original code printed a message but still
                # read pi.X, which raises an AttributeError. Now the inner
                # loop stops gracefully and keeps the last pi_value — the
                # returned Lagrangian cut is valid for ANY pi by weak duality.
                if next_pi_prob.Status != GRB.OPTIMAL:
                    print("next_pi_prob is not optimal, stop the inner loop")
                    cont_bool = False
                else:
                    pi_value = np.zeros((J_len, T_len))
                    for j in range(J_len):
                        for t in range(T_len):
                            pi_value[j,t] = next_pi_prob.getVarByName("pi[{},{}]".format(j,t)).X
                    # update the subproblem and solve it
                    sub_prob = update_subproblem_lag(sub_prob, co, p, I_len, J_len, T_len, pi_value)
                    sub_prob.optimize()

                # output the current status
        # print("Iteration: {}, V(pi_j): {}, Delta: {}".format(counter, sub_prob.ObjVal, Delta))

    # obtain the intercept of the Lagrangian cut
    v_value = sub_prob.ObjVal
    return pi_value, v_value, cutList

if __name__ == "__main__":
    # [ADDED: ML4SDDP] fix the random seed so that USE_HEURISTIC_K = True and
    # False run on the SAME instance (comment out for a fresh random instance)
    np.random.seed(42)

    # initialize the data
    omega = 50          # number of scenarios
    J_len = 5           # [MODIFIED: ML4SDDP] 3 -> 5, ML4SDDP paper instance dimensions
    T_len = 5
    I_len = 8           # [MODIFIED: ML4SDDP] 4 -> 8, ML4SDDP paper instance dimensions

    # [ADDED: ML4SDDP] switch for the heuristic top-k scenario selection
    # (Heur-k, see utils_heuristic_k.py). With USE_HEURISTIC_K = False the
    # loop below behaves exactly like the original code (every scenario is
    # processed in every outer iteration).
    USE_HEURISTIC_K = False
    HEUR_K_RATIO = 0.5        # kappa: fraction of scenarios solved per selective iteration
    HEUR_K_SAFETY = 5         # N_full: every N_full-th iteration solves ALL scenarios

    # [ADDED: ML4SDDP] Lagrangian multiplier oracle, see utils_sb_probe.py:
    #   "smc"   - the original level set only (the loop behaves like the original code)
    #   "sb"    - strengthened-Benders multiplier + 1 certification MIP, else the level set
    #   "probe" - flip-probe multiplier + 1 certification MIP, else the level set
    # Combines with USE_HEURISTIC_K: the oracle runs for the selected scenarios only.
    MULTIPLIER_ORACLE = "smc"

    # [ADDED: ML4SDDP] safety cap for the outer cutting-plane loop: the
    # original loop had no cap, so it can repeat identical iterations forever
    # if no violated cut is found while the gap is still above the tolerance
    MAX_OUTER_ITER = 200

    # [MODIFIED: ML4SDDP] data generation follows our ML4SDDP DCAP instances:
    #   a_{jt} - opening cost, b_{jt} - capacity when open,
    #   p_{jt} - overflow penalty, d^omega_{it} - demand,
    #   c^omega_{ijt} - assignment cost
    # (u_option and the scalar capacity bound B are removed together with u)
    a = np.round(np.random.uniform(1.0, 5.0, (J_len, T_len)), 5)
    b = np.round(np.random.uniform(5.0, 15.0, (J_len, T_len)), 5)
    c = np.round(np.random.uniform(1.0, 10.0, (omega, I_len, J_len, T_len)), 5)
    p = np.round(np.random.uniform(50.0, 100.0, (J_len, T_len)), 5)
    d = np.round(np.random.uniform(1.0, 5.0, (omega, I_len, T_len)), 5)

    LB = -np.inf
    UB = np.inf
    lambda_level = 0.5
    mu_level = 0.6
    iter_bool = True
    # initialize the dictionary to store the Lagrangian cuts for the subproblems' convex envelope
    cut_Dict = {}
    for o in range(omega):
        cut_Dict[o] = []

    # build the extensive form and solve it
    extensive_prob = build_extensive_form(omega, a, b, c, d, p, I_len, J_len, T_len)   # [MODIFIED: ML4SDDP] updated call signature
    ef_start_time = time.time()    # [ADDED: ML4SDDP] time the extensive form solve
    extensive_prob.optimize()
    ef_time = time.time() - ef_start_time    # [ADDED: ML4SDDP]
    # obtain the extensive form solution/optimal value
    x_opt_value = np.zeros((J_len,T_len))
    for j in range(J_len):
        for t in range(T_len):
            x_opt_value[j,t] = extensive_prob.getVarByName("x[{},{}]".format(j,t)).X
    opt_value = extensive_prob.ObjVal

    # build the master problem
    master_prob = build_masterproblem(omega, a, J_len, T_len)   # [MODIFIED: ML4SDDP] updated call signature
    x_best = np.zeros((J_len,T_len))
    # [MODIFIED: ML4SDDP] u_best removed (no u variable in the new model)

    # [ADDED: ML4SDDP] set up the Heur-k selector and the outer iteration
    # counter (the counter drives the full-iteration schedule)
    if USE_HEURISTIC_K:
        heur_selector = HeuristicKSelector(omega, solve_ratio=HEUR_K_RATIO, safety_interval=HEUR_K_SAFETY)
    else:
        heur_selector = None
    outer_iter = 0

    # [ADDED: ML4SDDP] set up the multiplier oracle (None = original level set only)
    if MULTIPLIER_ORACLE == "smc":
        oracle = None
    else:
        oracle = SBProbeOracle(MULTIPLIER_ORACLE)

    # [ADDED: ML4SDDP] time the whole cutting-plane algorithm (this is the
    # number to compare between USE_HEURISTIC_K = True / False)
    algo_start_time = time.time()

    # iteration of the cutting plane algorithm
    while iter_bool:
        # [ADDED: ML4SDDP] advance the outer iteration counter
        outer_iter += 1

        # solve the master problem
        master_prob.optimize()
        LB = master_prob.ObjVal

        # obtain the master soluton/optimal value & update the lower bound
        # [MODIFIED: ML4SDDP] u_value removed (no u variable in the new model)
        x_value = np.zeros((J_len,T_len))
        for j in range(J_len):
            for t in range(T_len):
                x_value[j,t] = master_prob.getVarByName("x[{},{}]".format(j,t)).X

        # obtain x_tilde
        # [MODIFIED: ML4SDDP] x is binary, so the reference point is 0.5
        # (midpoint of {0,1}) instead of 0.5 * B
        x_tilde = np.zeros((J_len,T_len))
        for j in range(J_len):
            for t in range(T_len):
                x_tilde[j,t] = 0.5

        # [ADDED: ML4SDDP] Heur-k scenario selection.
        # On "full" iterations (every HEUR_K_SAFETY-th one) ALL scenarios are
        # solved — these are the only iterations where V_bar is a valid upper
        # bound and the stopping criterion may be certified. On the other
        # ("selective") iterations only the top-k scenarios ranked by
        # score = L_tilde - theta_hat are solved, where L_tilde is the last
        # OBSERVED recourse value (lazy protocol: no extra MIP is solved just
        # for scoring). With USE_HEURISTIC_K = False every iteration processes
        # all scenarios, exactly as in the original code.
        if USE_HEURISTIC_K:
            theta_hat = [master_prob.getVarByName("theta[{}]".format(o)).X for o in range(omega)]
            solve_set = heur_selector.select_scenarios(outer_iter, theta_hat)
            full_iter_bool = heur_selector.is_full_iteration(outer_iter)
        else:
            solve_set = set(range(omega))
            full_iter_bool = True

        # iterate over the subproblem
        # [MODIFIED: ML4SDDP] first-stage cost term is a^T x only (u removed)
        V_bar = sum((a[j][t] * x_value[j,t] for j in range(J_len) for t in range(T_len)))
        for o in range(omega):
            # [ADDED: ML4SDDP] skip the scenarios not selected by Heur-k
            # (solve_set contains all scenarios when USE_HEURISTIC_K = False
            # or on full iterations)
            if o not in solve_set:
                continue

            # obtain the subproblem value & update the upper bound
            sub_prob = build_subproblem(o, b, c[o], d[o], p, I_len, J_len, T_len, x_value)   # [MODIFIED: ML4SDDP] pass capacity data b
            sub_prob.optimize()
            # obtain the subproblem solution/optimal value and update the upper bound
            L_value = sub_prob.ObjVal
            V_bar += L_value / omega
            # [ADDED: ML4SDDP] record the observed recourse value — it feeds
            # the Heur-k score of the following iterations
            if USE_HEURISTIC_K:
                heur_selector.update_observation(o, L_value)

            # [ADDED: ML4SDDP] try the cheap certified multiplier first; None means
            # "not certified" (always None for "smc") and the original level set runs
            cheap_cut = None
            if oracle is not None:
                cheap_cut = oracle.certified_cut(o, lambda pi_value: build_subproblem_lag(o, b, c[o], d[o], p, I_len, J_len, T_len, pi_value),
                                                 x_value, L_value, cut_Dict[o])
            if cheap_cut is not None:
                pi_value_o, v_value_o = cheap_cut
            else:
                # [MODIFIED: ML4SDDP] the original level-set call, unchanged but
                # indented into the fallback branch
                # generate the Lagrangian cuts
                pi_value_o, v_value_o, cutList_o = solve_lag_dual(o, b, c[o], d[o], p, I_len, J_len, T_len, x_value, L_value, lambda_level, mu_level, x_tilde, 1e-3, cut_Dict[o])   # [MODIFIED: ML4SDDP] pass capacity data b instead of B
                cut_Dict[o] = cutList_o
            # update the master problem with the Lagrangian cuts
            if v_value_o + np.inner(pi_value_o.flatten(), x_value.flatten()) > master_prob.getVarByName("theta[{}]".format(o)).X:
                master_prob.addConstr(v_value_o + gp.quicksum(pi_value_o[j,t] * master_prob.getVarByName("x[{},{}]".format(j,t)) for j in range(J_len) for t in range(T_len)) <= \
                                    master_prob.getVarByName("theta[{}]".format(o)))
        
        # [MODIFIED: ML4SDDP] V_bar is a valid upper bound only when the
        # recourse problem of EVERY scenario was evaluated at this x_value,
        # i.e. on full iterations (full_iter_bool is always True when
        # USE_HEURISTIC_K = False, recovering the original behavior)
        if full_iter_bool and V_bar < UB:
            UB = V_bar
            # record the best solution
            # [MODIFIED: ML4SDDP] u_best removed (no u variable)
            for j in range(J_len):
                for t in range(T_len):
                    x_best[j,t] = x_value[j,t]

        # [ADDED: ML4SDDP] progress output of the outer loop
        print("Outer iteration: {}, LB: {:.4f}, UB: {:.4f}, scenarios solved: {}/{}".format(
            outer_iter, LB, UB, len(solve_set), omega))

        # check the stopping criterion
        # [MODIFIED: ML4SDDP] under Heur-k the criterion is only certified on
        # full iterations (full_iter_bool is always True when
        # USE_HEURISTIC_K = False, recovering the original behavior)
        if full_iter_bool and abs((UB - LB)/UB) < 1e-2:
            iter_bool = False
        # [ADDED: ML4SDDP] stop when the safety cap on outer iterations is hit
        elif outer_iter >= MAX_OUTER_ITER:
            print("Maximum number of outer iterations ({}) reached, stop.".format(MAX_OUTER_ITER))
            iter_bool = False
        else:
            # update the master problem
            master_prob.update()

    # [ADDED: ML4SDDP] final summary
    algo_time = time.time() - algo_start_time
    print("Extensive form optimal value: {:.4f} (solve time: {:.2f}s)".format(opt_value, ef_time))
    print("Final LB: {:.4f}, UB: {:.4f}, gap: {:.4f}%".format(LB, UB, abs((UB - LB)/UB)*100))
    print("Total cutting-plane solve time: {:.2f}s ({} outer iterations)".format(algo_time, outer_iter))
    # [ADDED: ML4SDDP] multiplier oracle setting and its statistics
    print("Multiplier oracle: {}, Heur-k: {}".format(MULTIPLIER_ORACLE, USE_HEURISTIC_K))
    if oracle is not None:
        print(oracle.summary())