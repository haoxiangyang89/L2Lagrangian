import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Notes:
# 1. The optimization problem should have a minimization orientation.

# Create a new Gurobi environment
env = gp.Env(empty=True)
env.setParam('LogFile', 'gurobi.log')
env.start()

def build_extensive_form(omega, a, b, c, d, p, B, I_len, J_len, T_len, u_option=0):
    # construct the extensive formulation
    extensive_prob = gp.Model("extensive_form")

    # obtain problem dimensions
    J = range(J_len)
    I = range(I_len)
    T = range(T_len)

    # set up the decision variables
    x = extensive_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0, name="x")
    if u_option == 0:
        u = extensive_prob.addVars(J_len, T_len, vtype=GRB.BINARY, name="u")
    else:
        u = extensive_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="u")
    y = extensive_prob.addVars(omega, I_len, J_len, T_len, vtype=GRB.BINARY, name="y")
    s = extensive_prob.addVars(omega, J_len, T_len, vtype=GRB.CONTINUOUS, lb=0, name="s")

    # set up the objective function
    extensive_prob.setObjective(gp.quicksum(a[j][t] * x[j,t] + b[j][t] * u[j,t] for j in J for t in T) + 1/omega * gp.quicksum(\
            gp.quicksum(p[j][t] * s[o,j,t] + gp.quicksum(c[o][i][j][t] * y[o,i,j,t] for i in I) for j in J for t in T)
         for o in range(omega)), GRB.MINIMIZE)
    # set up the structural constraints
    extensive_prob.addConstrs((x[j,t] <= B * u[j,t] for j in J for t in T), name = "capacity_cons")
    extensive_prob.addConstrs((gp.quicksum(d[o,i,t] * y[o,i,j,t] for i in I) - s[o,j,t] <= gp.quicksum(x[j,tau] for tau in range(t+1))\
                                for j in J for t in T for o in range(omega)),name = "flow_cons")
    extensive_prob.addConstrs((gp.quicksum(y[o,i,j,t] for j in J) == 1 for i in I for t in T for o in range(omega)), name = "demand_cons")
    extensive_prob.update()
    return extensive_prob

def build_masterproblem(omega, a, b, B, J_len, T_len, u_option, prob_lb=-100000):
    # construct the master program
    master_prob = gp.Model("masterproblem")
    master_prob.Params.OutputFlag = 0

    # obtain problem dimensions
    J = range(J_len)
    T = range(T_len)

    # set up the decision variables
    x = master_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0, name="x")
    if u_option == 0:
        u = master_prob.addVars(J_len, T_len, vtype=GRB.BINARY, name="u")
    else:
        u = master_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="u")
    theta = master_prob.addVars(omega, vtype=GRB.CONTINUOUS, lb=prob_lb, name="theta")
    master_prob.setObjective(gp.quicksum(a[j][t] * x[j,t] + b[j][t] * u[j,t] for j in J for t in T) + 1/omega * gp.quicksum(\
        theta[o] for o in range(omega)), GRB.MINIMIZE)
    
    # set up the structural constraints
    master_prob.addConstrs((x[j,t] <= B * u[j,t] for j in J for t in T), name = "capacity_cons")
    master_prob.update()
    return master_prob

def build_subproblem(o, co, do, p, I_len, J_len, T_len, x_value):
    # input: 
    # o - index of the scenario, do - processing requirement, co - processing cost, p - penalty cost,
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
    sub_prob.addConstrs((gp.quicksum(do[i,t] * y[i,j,t] for i in I) - s[j,t] <= gp.quicksum(x_value[j,tau] for tau in range(t+1))\
                                for j in J for t in T),name = "flow_cons")
    sub_prob.addConstrs((gp.quicksum(y[i,j,t] for j in J) == 1 for i in I for t in T), name = "demand_cons")

    sub_prob.update()
    return sub_prob

def build_subproblem_lag(o, B, co, do, p, I_len, J_len, T_len, pi_value):
    # input: 
    # o - index of the scenario, do - processing requirement, co - processing cost, p - penalty cost,
    # I_len - number of tasks, J_len - number of resources, T_len - number of time periods,
    # pi_value - the Lagrangian dual multipliers

    sub_prob_lag = gp.Model("subproblem_lag_" + str(o))
    sub_prob_lag.Params.OutputFlag = 0

    # obtain problem dimensions
    J = range(J_len)
    I = range(I_len)
    T = range(T_len)

    # set up the auxiliary variables z (copy of x)
    z = sub_prob_lag.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=B, name="z")

    # set up the decision variables
    y = sub_prob_lag.addVars(I_len, J_len, T_len, vtype=GRB.BINARY, name="y")
    s = sub_prob_lag.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0, name="s")

    # set up the objective function with Lagrangian penalty term
    sub_prob_lag.setObjective(gp.quicksum(p[j][t] * s[j,t] + gp.quicksum(co[i][j][t] * y[i,j,t] for i in I) - pi_value[j,t] * z[j,t] for j in J for t in T), GRB.MINIMIZE)
    
    # set up the structural constraints
    sub_prob_lag.addConstrs((gp.quicksum(do[i,t] * y[i,j,t] for i in I) - s[j,t] <= gp.quicksum(z[j,tau] for tau in range(t+1))\
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
def build_ls_lb_problem(J_len, T_len, x_value, L_value, cutList, norm_option, prob_lb=-100000, prob_ub=100000):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    lb_prob = gp.Model("lb_problem")
    lb_prob.Params.OutputFlag = 0

    # set up the dual variables pi and auxiliary variables theta
    pi = lb_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=prob_lb, ub=prob_ub, name="pi")
    theta = lb_prob.addVar(vtype=GRB.CONTINUOUS, lb=prob_lb, name="theta")

    # set up the objective function with Lagrangian penalty term
    if norm_option == 0:
        # L2 norm
        lb_prob.setObjective(gp.quicksum(pi[j,t] * pi[j,t] for j in range(J_len) for t in range(T_len)), GRB.MINIMIZE)
    else:
        # L1 norm
        pi_abs = lb_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=np.maximum(np.abs(prob_ub),np.abs(prob_lb)), name="pi_abs")
        lb_prob.setObjective(gp.quicksum(pi_abs[j,t] for j in range(J_len) for t in range(T_len)), GRB.MINIMIZE)
        lb_prob.addConstrs((pi[j,t] <= pi_abs[j,t] for j in range(J_len) for t in range(T_len)), name = "pi_abs_pos")
        lb_prob.addConstrs((-pi[j,t] <= pi_abs[j,t] for j in range(J_len) for t in range(T_len)), name = "pi_abs_neg")

    # set up the structural constraints
    lb_prob.addConstrs((gp.quicksum(cutList[k][0][j,t] * pi[j,t] for j in range(J_len) for t in range(T_len)) + cutList[k][1] >= theta
                                for k in range(len(cutList))), name = "cuts")
    lb_prob.addConstr(gp.quicksum(pi[j,t] * x_value[j,t] for j in range(J_len) for t in range(T_len)) + theta >= L_value, name = "cons")
    lb_prob.update()
    return lb_prob

# update the level set lower bound problem
def update_ls_lb_problem(lb_prob, J_len, T_len, cutList, update_ind_range):
    # input: 
    # lb_prob - the level set lower bound problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    for k in update_ind_range:
        lb_prob.addConstr(gp.quicksum(cutList[k][0][j,t] * lb_prob.getVarByName("pi[{},{}]".format(j,t)) for j in range(J_len) for t in range(T_len)) + \
                          cutList[k][1] >= lb_prob.getVarByName("theta"), name = "cuts[{}]".format(k))
    lb_prob.update()
    return lb_prob

# Build the level set lower bound problem
def build_next_pi_problem(J_len, T_len, level, alpha, x_value, L_value, cutList, norm_option, prob_lb=-100000, prob_ub=100000):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)
    next_pi_prob = gp.Model("next_pi_prob")
    next_pi_prob.Params.OutputFlag = 0
    # set up the dual variables pi and auxiliary variables theta
    pi = next_pi_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=prob_lb, ub=prob_ub, name="pi")
    pi_abs = next_pi_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=np.maximum(np.abs(prob_ub),np.abs(prob_lb)), name="pi_abs")
    theta = next_pi_prob.addVar(vtype=GRB.CONTINUOUS, lb=prob_lb, name="theta")
    pi_obj_abs = next_pi_prob.addVars(J_len, T_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=np.maximum(np.abs(prob_ub),np.abs(prob_lb)), name="pi_obj")

    # set up the structural constraints
    next_pi_prob.addConstrs((gp.quicksum(cutList[k][0][j,t] * pi[j,t] for j in range(J_len) for t in range(T_len)) + cutList[k][1] >= theta
                                for k in range(len(cutList))), name = "cuts")
    if norm_option == 0:
        # L2 norm
        next_pi_prob.addConstr(alpha * gp.quicksum(pi[j,t] * pi[j,t] for j in range(J_len) for t in range(T_len)) + 
                           (1 - alpha) * (L_value - gp.quicksum(pi[j,t] * x_value[j,t] for j in range(J_len) for t in range(T_len)) - theta) <= level, name = "level_cons")
    else:
        # L1 norm
        next_pi_prob.addConstr(alpha * gp.quicksum(pi_abs[j,t] for j in range(J_len) for t in range(T_len)) + 
                           (1 - alpha) * (L_value - gp.quicksum(pi[j,t] * x_value[j,t] for j in range(J_len) for t in range(T_len)) - theta) <= level, name = "level_cons")
        next_pi_prob.addConstrs((pi[j,t] <= pi_abs[j,t] for j in range(J_len) for t in range(T_len)), name = "pi_abs_pos")
        next_pi_prob.addConstrs((-pi[j,t] <= pi_abs[j,t] for j in range(J_len) for t in range(T_len)), name = "pi_abs_neg")

    # set up the objective function absolute value term
    next_pi_prob.addConstrs((pi_obj_abs[j,t] - pi[j,t] >= 0 for j in range(J_len) for t in range(T_len)), name = "pi_obj_pos")
    next_pi_prob.addConstrs((pi_obj_abs[j,t] + pi[j,t] >= 0 for j in range(J_len) for t in range(T_len)), name = "pi_obj_neg")

    # set up the objective function
    next_pi_prob.setObjective(gp.quicksum(pi_obj_abs[j,t] for j in range(J_len) for t in range(T_len)), GRB.MINIMIZE)

    next_pi_prob.update()
    return next_pi_prob

def update_next_pi_problem(next_pi_prob, J_len, T_len, cutList, update_ind_range, alpha, x_value, L_value, pi_bar_value, level, norm_option):
    # input: 
    # next_pi_prob - the next pi problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    if norm_option == 0:
        next_pi_prob.remove(next_pi_prob.getQConstrs()[0])
        next_pi_prob.addConstr(alpha * gp.quicksum(next_pi_prob.getVarByName("pi[{},{}]".format(j,t)) * next_pi_prob.getVarByName("pi[{},{}]".format(j,t)) for j in range(J_len) for t in range(T_len)) + 
                        (1 - alpha) * (L_value - gp.quicksum(next_pi_prob.getVarByName("pi[{},{}]".format(j,t)) * x_value[j,t] for j in range(J_len) for t in range(T_len)) - 
                        next_pi_prob.getVarByName("theta")) <= level, name = "level_cons")
    else:
        next_pi_prob.remove(next_pi_prob.getConstrByName("level_cons"))
        next_pi_prob.addConstr(alpha * gp.quicksum(next_pi_prob.getVarByName("pi_abs[{},{}]".format(j,t)) for j in range(J_len) for t in range(T_len)) + 
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

def obtain_alpha_bounds_opt(J_len, T_len, pi_list, L_value, x_value, v_underbar, V_list, norm_option, prob_lb=-100000):
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
    if norm_option == 0:
        # L2 norm
        alpha_prob.addConstrs((alpha_obj <= alpha * (np.inner(pi_list[k].flatten(), pi_list[k].flatten()) - v_underbar) + \
                            (1 - alpha) * (L_value - np.inner(pi_list[k].flatten(), x_value.flatten()) - V_list[k])
                            for k in range(len(pi_list))), name="alpha_obj_constr")
    else:
        # L1 norm
        alpha_prob.addConstrs((alpha_obj <= alpha * (sum(np.abs(pi_list[k][j,t]) for j in range(J_len) for t in range(T_len)) - v_underbar) + \
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

def obtain_alpha_bounds(J_len, T_len, pi_list, L_value, x_value, v_underbar, V_list, norm_option):
    # algebraic way to calculate alpha_max and alpha_min
    alpha_underbar = []
    alpha_bar = []
    gamma_list = {}
    eta_list = {}

    for k in range(len(pi_list)):
        if norm_option == 0:
            # L2 norm
            gamma_list[k] = (np.inner(pi_list[k].flatten(), pi_list[k].flatten()) - v_underbar) - \
                (L_value - np.inner(pi_list[k].flatten(), x_value.flatten()) - V_list[k])
            eta_list[k] = (L_value - np.inner(pi_list[k].flatten(), x_value.flatten()) - V_list[k])
        else:
            # L1 norm
            gamma_list[k] = (np.sum(np.abs(pi_list[k])) - v_underbar) - \
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

    # algebraic way to calculate Delta
    alpha_star = 0
    Delta = np.min([eta_list[k] for k in range(len(pi_list))])
    k_mark = {}
    for k in range(len(pi_list)):
        k_mark[k] = 0
    k_star = np.argmin([eta_list[k] for k in range(len(pi_list))])
    k_mark[k_star] = 1

    if gamma_list[k_star] >= 0:
        cont_bool = True
    else:
        cont_bool = False
    while cont_bool:
        alpha_candidate = []
        k_candidate = []
        for k in range(len(pi_list)):
            if k_mark[k] == 0:
                alpha_k = (eta_list[k] - eta_list[k_star]) / (gamma_list[k_star] - gamma_list[k])
                if (alpha_k > alpha_star)&(alpha_k <= 1)&(alpha_k >= 0):
                    alpha_candidate.append(alpha_k)
                    k_candidate.append(k)
        if len(alpha_candidate) > 0:
            alpha_star = np.min(alpha_candidate)
            k_star_new = k_candidate[np.argmin(alpha_candidate)]
            if (gamma_list[k_star_new] < 0)&(gamma_list[k_star] >= 0):
                cont_bool = False
            else:
                k_star = k_star_new
                k_mark[k_star] = 1
        else:
            alpha_star = 1
            cont_bool = False

    Delta = np.min([gamma_list[k] * alpha_star + eta_list[k] for k in range(len(pi_list))])

    return alpha_max, alpha_min, Delta

# procedure to solve the Lagrangian dual problem
def solve_lag_dual(o, B, co, do, p, I_len, J_len, T_len, x_value, L_value, lambda_level, mu_level, norm_option, tol = 1e-2, cutList = [], sub_lb = -100000, sub_ub = 100000):
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
    lb_prob = build_ls_lb_problem(J_len, T_len, x_value, L_value, cutList, norm_option)
    # set up the auxiliary problem to find the next pi_value
    level = sub_ub
    next_pi_prob = build_next_pi_problem(J_len, T_len, level, alpha, x_value, L_value, cutList, norm_option)

    # build the inner min subproblem with Lagrangian penalty term
    sub_prob = build_subproblem_lag(o, B, co, do, p, I_len, J_len, T_len, pi_value)
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
            theta_j = np.Infinity
        # only generate the cut if Vj > theta_j
        if Vj < theta_j - 1e-4:
            cutList.append((-z_value, Vj + np.inner(z_value.flatten(), pi_value.flatten())))
            cutList_update_start_ind = len(cutList) - 1
            cutList_update_end_ind = len(cutList)
        else:
            cutList_update_start_ind = len(cutList)
            cutList_update_end_ind = len(cutList)

        # update and solve the lower bound problem
        lb_prob = update_ls_lb_problem(lb_prob, J_len, T_len, cutList, range(cutList_update_start_ind, cutList_update_end_ind))
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
        alpha_max, alpha_min, Delta = obtain_alpha_bounds(J_len, T_len, pi_list, L_value, x_value, lb_prob.ObjVal, V_list, norm_option)
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
                if norm_option == 0:
                    v_bar_list = [alpha * np.inner(pi_list[pi_k].flatten(), pi_list[pi_k].flatten()) + (1 - alpha) * (L_value - np.inner(pi_list[pi_k].flatten(), x_value.flatten()) - V_list[pi_k]) for pi_k in range(len(pi_list))]
                else:
                    v_bar_list = [alpha * np.sum(np.abs(pi_list[pi_k])) + (1 - alpha) * (L_value - np.inner(pi_list[pi_k].flatten(), x_value.flatten()) - V_list[pi_k]) for pi_k in range(len(pi_list))]
                v_bar = np.min(v_bar_list)
                v_underbar = alpha * lb_prob.ObjVal
                level = lambda_level * v_bar + (1 - lambda_level) * v_underbar

                # solve for the next pi_value
                next_pi_prob = update_next_pi_problem(next_pi_prob, J_len, T_len, cutList, range(cutList_update_start_ind, cutList_update_end_ind), alpha, x_value, L_value, pi_value, level, norm_option)
                next_pi_prob.optimize()
                # obtain the next pi_value
                pi_value = np.zeros((J_len, T_len))
                for j in range(J_len):
                    for t in range(T_len):
                        if next_pi_prob.Status != GRB.OPTIMAL:
                            print("next_pi_prob is not optimal")
                            cont_bool = False
                        pi_value[j,t] = next_pi_prob.getVarByName("pi[{},{}]".format(j,t)).X
                # update the subproblem and solve it
                sub_prob = update_subproblem_lag(sub_prob, co, p, I_len, J_len, T_len, pi_value)
                sub_prob.optimize()

                # output the current status
        print("Iteration: {}, V(pi_j): {}, Delta: {}".format(counter, sub_prob.ObjVal, Delta))

    # obtain the intercept of the Lagrangian cut
    v_value = sub_prob.ObjVal
    return pi_value, v_value, cutList

if __name__ == "__main__":
    # initialize the data
    omega = 300          # number of scenarios
    # norm_option = 0       # 0 represents the L2 norm
    norm_option = 1         # 1 represents the L1 norm
    # u_option = 0
    u_option = 1

    J_len = 3
    T_len = 5
    I_len = 4

    a = np.round(np.random.uniform(5.0,7.0, (J_len, T_len)), 5)
    b = np.round(np.random.uniform(15.0, 35.0, (J_len, T_len)), 5)
    c = np.round(np.random.uniform(5.0, 10.0, (omega, I_len, J_len, T_len)), 5)
    p = np.round(np.random.uniform(500.0, 1000.0, (J_len, T_len)), 5)
    d = np.round(np.random.uniform(0.5, 1.5, (omega, I_len, T_len)), 5)
    B = 1.0

    LB = -np.Infinity
    UB = np.Infinity
    lambda_level = 0.5
    mu_level = 0.6
    iter_bool = True
    # initialize the dictionary to store the Lagrangian cuts for the subproblems' convex envelope
    cut_Dict = {}
    for o in range(omega):
        cut_Dict[o] = []

    # build the extensive form and solve it
    extensive_prob = build_extensive_form(omega, a, b, c, d, p, B, I_len, J_len, T_len, u_option)
    extensive_prob.optimize()
    # obtain the extensive form solution/optimal value
    x_opt_value = np.zeros((J_len,T_len))
    for j in range(J_len):
        for t in range(T_len):
            x_opt_value[j,t] = extensive_prob.getVarByName("x[{},{}]".format(j,t)).X
    opt_value = extensive_prob.ObjVal

    # build the master problem
    master_prob = build_masterproblem(omega, a, b, B, J_len, T_len, u_option)
    x_best = np.zeros((J_len,T_len))
    u_best = np.zeros((J_len,T_len))

    # iteration of the cutting plane algorithm
    while iter_bool:
        # solve the master problem
        master_prob.optimize()
        LB = master_prob.ObjVal

        # obtain the master soluton/optimal value & update the lower bound
        x_value = np.zeros((J_len,T_len))
        u_value = np.zeros((J_len,T_len))
        for j in range(J_len):
            for t in range(T_len):
                x_value[j,t] = master_prob.getVarByName("x[{},{}]".format(j,t)).X
                u_value[j,t] = master_prob.getVarByName("u[{},{}]".format(j,t)).X

        # iterate over the subproblem
        V_bar = sum((a[j][t] * x_value[j,t] + b[j][t] * u_value[j,t] for j in range(J_len) for t in range(T_len)))
        for o in range(omega):
            # obtain the subproblem value & update the upper bound
            sub_prob = build_subproblem(o, c[o], d[o], p, I_len, J_len, T_len, x_value)
            sub_prob.optimize()
            # obtain the subproblem solution/optimal value and update the upper bound
            L_value = sub_prob.ObjVal 
            V_bar += L_value / omega

            # generate the Lagrangian cuts
            pi_value_o, v_value_o, cutList_o = solve_lag_dual(o, B, c[o], d[o], p, I_len, J_len, T_len, x_value, L_value, lambda_level, mu_level, norm_option, 1e-3, cut_Dict[o])
            cut_Dict[o] = cutList_o
            # update the master problem with the Lagrangian cuts
            if v_value_o + np.inner(pi_value_o.flatten(), x_value.flatten()) > master_prob.getVarByName("theta[{}]".format(o)).X:
                master_prob.addConstr(v_value_o + gp.quicksum(pi_value_o[j,t] * master_prob.getVarByName("x[{},{}]".format(j,t)) for j in range(J_len) for t in range(T_len)) <= \
                                    master_prob.getVarByName("theta[{}]".format(o)))
        
        if V_bar < UB:
            UB = V_bar
            # record the best solution
            for j in range(J_len):
                for t in range(T_len):
                    x_best[j,t] = x_value[j,t]
                    u_best[j,t] = u_value[j,t]
        
        # check the stopping criterion
        if abs((UB - LB)/UB) < 1e-2:
            iter_bool = False
        else:
            # update the master problem
            master_prob.update()