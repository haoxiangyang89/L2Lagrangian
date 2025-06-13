import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Notes:
# 1. The optimization problem should have a minimization orientation.

# Create a new Gurobi environment
env = gp.Env(empty=True)
env.setParam('LogFile', 'gurobi.log')
env.start()

def build_extensive_form(omega, h, T, W, c, c1, y_option):
    # obtain the dimension of the problem
    J_len = W.shape[0]  # number of first-stage decision variables
    I_len = W.shape[1]  # number of second-stage decision variables

    # construct the extensive formulation
    extensive_prob = gp.Model("extensive_form")
    x = extensive_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=0, ub=5, name="x")
    x[0].Start = 0.0
    x[1].Start = 4.0
    if y_option == 0:
        y = extensive_prob.addVars(omega, I_len, vtype=GRB.BINARY, name="y")
    else:
        y = extensive_prob.addVars(omega, I_len, vtype=GRB.INTEGER, lb=0, name="y")
    # set up the objective function
    extensive_prob.setObjective(gp.quicksum(c1[j] * x[j] for j in range(J_len)) + 1/omega * gp.quicksum(gp.quicksum(c[i] * y[o,i] for i in range(I_len)) for o in range(omega)), GRB.MINIMIZE)
    # set up the structural constraints
    extensive_prob.addConstrs((gp.quicksum(W[j,i] * y[o,i] for i in range(I_len)) <= h[o,j] -
                                gp.quicksum(T[j,k] * x[k] for k in range(J_len)) for j in range(J_len) for o in range(omega)), name = "cons")
    extensive_prob.update()
    return extensive_prob

def build_masterproblem(omega, c1, xub = 5, prob_lb=-10000):
    # obtain the dimension of the problem
    J_len = len(c1)  # number of first-stage decision variables

    # construct the master program
    master_prob = gp.Model("masterproblem")
    master_prob.Params.OutputFlag = 0
    x = master_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=0, ub=xub, name="x")
    theta = master_prob.addVars(omega, vtype=GRB.CONTINUOUS, lb=prob_lb, name="theta")
    master_prob.setObjective(gp.quicksum(c1[j] * x[j] for j in range(J_len)) + 1/omega * gp.quicksum(theta[o] for o in range(omega)), GRB.MINIMIZE)
    master_prob.update()
    return master_prob

def build_subproblem(o, ho, T, W, c, y_option, x_value):
    # input: 
    # o - index of the scenario, ho - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # x_value - the optimal solution of the master problem

    # obtain the dimension of the problem
    J_len = W.shape[0]  # number of first-stage decision variables
    I_len = W.shape[1]  # number of second-stage decision variables

    sub_prob = gp.Model("subproblem_" + str(o))
    sub_prob.Params.OutputFlag = 0

    # set up the decision variables y, allowing two types, binary and integers
    if y_option == 0:
        y = sub_prob.addVars(4, vtype=GRB.BINARY, name="y")
    else:
        y = sub_prob.addVars(4, vtype=GRB.INTEGER, lb=0, name="y")

    # set up the objective function
    sub_prob.setObjective(gp.quicksum(c[i] * y[i] for i in range(I_len)), GRB.MINIMIZE)

    # set up the structural constraints
    cons = sub_prob.addConstrs((gp.quicksum(W[j,i] * y[i] for i in range(I_len)) <= ho[j] - 
                                gp.quicksum(T[j,k] * x_value[k] for k in range(J_len)) for j in range(J_len)), name = "cons")
    sub_prob.update()
    return sub_prob

def build_subproblem_lag(o, ho, T, W, c, y_option, pi_value, xub = 5):
    # input: 
    # o - index of the scenario, ho - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # pi_value - the Lagrangian dual multipliers

    # obtain the dimension of the problem
    J_len = W.shape[0]  # number of first-stage decision variables
    I_len = W.shape[1]  # number of second-stage decision variables

    sub_prob_lag = gp.Model("subproblem_lag_" + str(o))
    sub_prob_lag.Params.OutputFlag = 0

    # set up the auxiliary variables z (copy of x)
    z = sub_prob_lag.addVars(J_len, vtype=GRB.CONTINUOUS, lb=0, ub=xub, name="z")

    # set up the decision variables y, allowing two types, binary and integers
    if y_option == 0:
        y = sub_prob_lag.addVars(I_len, vtype=GRB.BINARY, name="y")
    else:
        y = sub_prob_lag.addVars(I_len, vtype=GRB.INTEGER, lb=0, name="y")

    # set up the objective function with Lagrangian penalty term
    sub_prob_lag.setObjective(gp.quicksum(c[i] * y[i] for i in range(I_len)) -
                              gp.quicksum(pi_value[k] * z[k] for k in range(J_len)), GRB.MINIMIZE)
    
    # set up the structural constraints
    cons = sub_prob_lag.addConstrs((gp.quicksum(W[j,i] * y[i] for i in range(I_len)) <= ho[j] - 
                                gp.quicksum(T[j,k] * z[k] for k in range(J_len)) for j in range(J_len)), name = "cons")

    sub_prob_lag.update()
    return sub_prob_lag

def update_subproblem_lag(sub_prob_lag, c, pi_value):
    # obtain the dimension of the problem
    I_len = len(c)
    J_len = len(pi_value)

    # set up the objective function with Lagrangian penalty term
    sub_prob_lag.setObjective(gp.quicksum(c[i] * sub_prob_lag.getVarByName("y[{}]".format(i)) for i in range(I_len)) -
                              gp.quicksum(pi_value[k] * sub_prob_lag.getVarByName("z[{}]".format(k)) for k in range(J_len)), GRB.MINIMIZE)
    sub_prob_lag.update()
    return sub_prob_lag

# Build the level set lower bound problem
def build_ls_lb_problem(x_value, L_value, cutList, norm_option, prob_lb=-10000, prob_ub=10000):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # obtain the dimension of the problem
    J_len = len(x_value)  # number of first-stage decision variables

    lb_prob = gp.Model("lb_problem")
    lb_prob.Params.OutputFlag = 0

    # set up the dual variables pi and auxiliary variables theta
    pi = lb_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=prob_lb, ub=prob_ub, name="pi")
    theta = lb_prob.addVar(vtype=GRB.CONTINUOUS, lb=prob_lb, name="theta")

    # set up the objective function with Lagrangian penalty term
    if norm_option == 0:
        # L2 norm
        lb_prob.setObjective(gp.quicksum(pi[i] * pi[i] for i in range(J_len)), GRB.MINIMIZE)
    else:
        # L1 norm
        pi_abs = lb_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=np.maximum(np.abs(prob_ub),np.abs(prob_lb)), name="pi_abs")
        lb_prob.setObjective(gp.quicksum(pi_abs[i] for i in range(J_len)), GRB.MINIMIZE)
        lb_prob.addConstrs((pi[i] <= pi_abs[i] for i in range(J_len)), name = "pi_abs_pos")
        lb_prob.addConstrs((-pi[i] <= pi_abs[i] for i in range(J_len)), name = "pi_abs_neg")

    # set up the structural constraints
    lb_prob.addConstrs((gp.quicksum(cutList[j][0][i] * pi[i] for i in range(J_len)) + cutList[j][1] >= theta
                                for j in range(len(cutList))), name = "cuts")
    lb_prob.addConstr(gp.quicksum(pi[i] * x_value[i] for i in range(J_len)) + theta >= L_value, name = "cons")
    lb_prob.update()
    return lb_prob

# update the level set lower bound problem
def update_ls_lb_problem(lb_prob, J_len, cutList, update_ind_range):
    # input: 
    # lb_prob - the level set lower bound problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    for j in update_ind_range:
        lb_prob.addConstr(gp.quicksum(cutList[j][0][i] * lb_prob.getVarByName("pi[{}]".format(i)) for i in range(J_len)) + \
                          cutList[j][1] >= lb_prob.getVarByName("theta"), name = "cuts[{}]".format(j))
    lb_prob.update()
    return lb_prob

# Build the level set lower bound problem
def build_next_pi_problem(level, alpha, x_value, L_value, cutList, norm_option, prob_lb=-10000, prob_ub=10000):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # obtain the dimension of the problem
    J_len = len(x_value)  # number of first-stage decision variables

    next_pi_prob = gp.Model("next_pi_prob")
    next_pi_prob.Params.OutputFlag = 0
    # set up the dual variables pi and auxiliary variables theta
    pi = next_pi_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=prob_lb, ub=prob_ub, name="pi")
    pi_abs = next_pi_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=np.maximum(np.abs(prob_ub),np.abs(prob_lb)), name="pi_abs")
    theta = next_pi_prob.addVar(vtype=GRB.CONTINUOUS, lb=prob_lb, name="theta")
    pi_obj_abs = next_pi_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=np.maximum(np.abs(prob_ub),np.abs(prob_lb)), name="pi_obj")

    # set up the structural constraints
    next_pi_prob.addConstrs((gp.quicksum(cutList[j][0][i] * pi[i] for i in range(J_len)) + cutList[j][1] >= theta
                                for j in range(len(cutList))), name = "cuts")
    if norm_option == 0:
        # L2 norm
        next_pi_prob.addConstr(alpha * gp.quicksum(pi[i] * pi[i] for i in range(J_len)) + 
                           (1 - alpha) * (L_value - gp.quicksum(pi[i] * x_value[i] for i in range(J_len)) - theta) <= level, name = "level_cons")
    else:
        # L1 norm
        next_pi_prob.addConstr(alpha * gp.quicksum(pi_abs[i] for i in range(J_len)) + 
                           (1 - alpha) * (L_value - gp.quicksum(pi[i] * x_value[i] for i in range(J_len)) - theta) <= level, name = "level_cons")
        next_pi_prob.addConstrs((pi[i] <= pi_abs[i] for i in range(J_len)), name = "pi_abs_pos")
        next_pi_prob.addConstrs((-pi[i] <= pi_abs[i] for i in range(J_len)), name = "pi_abs_neg")

    # set up the objective function absolute value term
    next_pi_prob.addConstrs((pi_obj_abs[i] - pi[i] >= 0 for i in range(J_len)), name = "pi_obj_pos")
    next_pi_prob.addConstrs((pi_obj_abs[i] + pi[i] >= 0 for i in range(J_len)), name = "pi_obj_neg")

    # set up the objective function
    next_pi_prob.setObjective(gp.quicksum(pi_obj_abs[i] for i in range(J_len)), GRB.MINIMIZE)

    next_pi_prob.update()
    return next_pi_prob

def update_next_pi_problem(next_pi_prob, J_len, cutList, update_ind_range, alpha, x_value, L_value, pi_bar_value, level, norm_option):
    # input: 
    # next_pi_prob - the next pi problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    if norm_option == 0:
        next_pi_prob.remove(next_pi_prob.getQConstrs()[0])
        next_pi_prob.addConstr(alpha * gp.quicksum(next_pi_prob.getVarByName("pi[{}]".format(i)) * next_pi_prob.getVarByName("pi[{}]".format(i)) for i in range(J_len)) + 
                        (1 - alpha) * (L_value - gp.quicksum(next_pi_prob.getVarByName("pi[{}]".format(i)) * x_value[i] for i in range(J_len)) - next_pi_prob.getVarByName("theta")) <= level, name = "level_cons")
    else:
        next_pi_prob.remove(next_pi_prob.getConstrByName("level_cons"))
        next_pi_prob.addConstr(alpha * gp.quicksum(next_pi_prob.getVarByName("pi_abs[{}]".format(i)) for i in range(J_len)) + 
                    (1 - alpha) * (L_value - gp.quicksum(next_pi_prob.getVarByName("pi[{}]".format(i)) * x_value[i] for i in range(J_len)) - next_pi_prob.getVarByName("theta")) <= level, name = "level_cons")

    for j in update_ind_range:
        next_pi_prob.addConstr(gp.quicksum(cutList[j][0][i] * next_pi_prob.getVarByName("pi[{}]".format(i)) for i in range(J_len)) + \
                          cutList[j][1] >= next_pi_prob.getVarByName("theta"), name = "cuts[{}]".format(j))

    # set up the objective function absolute value rhs term
    for i in range(J_len):
        pos_constr = next_pi_prob.getConstrByName("pi_obj_pos[{}]".format(i))
        neg_constr = next_pi_prob.getConstrByName("pi_obj_neg[{}]".format(i))
        next_pi_prob.setAttr("RHS", pos_constr, -pi_bar_value[i])
        next_pi_prob.setAttr("RHS", neg_constr, pi_bar_value[i])

    next_pi_prob.update()
    return next_pi_prob

def obtain_alpha_bounds_opt(pi_list, L_value, x_value, v_underbar, V_list, norm_option, prob_lb=-10000):
    # input: 
    # alpha_prob - the alpha problem with piecewise linear objective function
    # return the upper and lower bounds of alpha
    alpha_min = 0
    alpha_max = 1

    # obtain the dimension of the problem
    J_len = len(x_value)  # number of first-stage decision variables

    # set up the alpha problem
    alpha_prob = gp.Model("alpha_problem")
    alpha_prob.Params.OutputFlag = 0
    alpha = alpha_prob.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="alpha")
    alpha_obj = alpha_prob.addVar(vtype=GRB.CONTINUOUS, lb=prob_lb, name="alpha_obj")
    alpha_prob.addConstr(alpha_obj >= 0, name="alpha_obj_lb")
    if norm_option == 0:
        # L2 norm
        alpha_prob.addConstrs((alpha_obj <= alpha * (np.inner(pi_list[j], pi_list[j]) - v_underbar) + \
                            (1 - alpha) * (L_value - np.inner(pi_list[j], x_value) - V_list[j])
                            for j in range(len(pi_list))), name="alpha_obj_constr")
    else:
        # L1 norm
        alpha_prob.addConstrs((alpha_obj <= alpha * (gp.quicksum(np.abs(pi_list[j][i]) for i in range(J_len)) - v_underbar) + \
                            (1 - alpha) * (L_value - np.inner(pi_list[j], x_value) - V_list[j])
                            for j in range(len(pi_list))), name="alpha_obj_constr")

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

def obtain_alpha_bounds(pi_list, L_value, x_value, v_underbar, V_list, norm_option):
    # algebraic way to calculate alpha_max and alpha_min
    alpha_underbar = []
    alpha_bar = []
    gamma_list = {}
    eta_list = {}

    for k in range(len(pi_list)):
        if norm_option == 0:
            # L2 norm
            gamma_list[k] = (np.inner(pi_list[k], pi_list[k]) - v_underbar) - \
                (L_value - np.inner(pi_list[k], x_value) - V_list[k])
            eta_list[k] = (L_value - np.inner(pi_list[k], x_value) - V_list[k])
        else:
            # L1 norm
            gamma_list[k] = (np.sum(np.abs(pi_list[k])) - v_underbar) - \
                (L_value - np.inner(pi_list[k], x_value) - V_list[k])
            eta_list[k] = (L_value - np.inner(pi_list[k], x_value) - V_list[k])
            if abs(eta_list[k]) <= 1e-7:
                eta_list[k] = 0.0
        if gamma_list[k] >= 0:
            alpha_bar.append(1)
            if eta_list[k] >= 0:
                alpha_underbar.append(0)
            else:
                if -eta_list[k] / gamma_list[k] <= 1 + 1e-5:
                    alpha_underbar.append(-eta_list[k] / gamma_list[k])
                else:
                    ValueError("alpha_underbar is not feasible")
        else:
            alpha_underbar.append(0)
            if eta_list[k] >= 0:
                if -eta_list[k] / gamma_list[k] < 1 + 1e-5:
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
def solve_lag_dual(o, ho, T, W, c, y_option, x_value, L_value, lambda_level, mu_level, norm_option, tol = 1e-2, cutList = [], sub_lb = -10000, sub_ub = 100000, init_pt = 1):
    # input: 
    # o - index of the scenario, ho - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # x_value - the optimal solution of the master problem
    # cutList - the list of cuts generated for the inner minimization problem so far, 
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # obtain the dimension of the problem
    J_len = W.shape[0]  # number of first-stage decision variables
    I_len = W.shape[1]  # number of second-stage decision variables

    # initialize the cut list for the level set problem
    v_value = 0
    alpha_min = 0
    alpha_max = 1
    alpha = (alpha_max + alpha_min) / 2
    pi_list = []
    V_list = []

    # set up the lower bound problem
    lb_prob = build_ls_lb_problem(x_value, L_value, cutList, norm_option)
    if init_pt == 0:
        # initialize the Lagrangian dual multipliers with zeros
        pi_value = np.zeros(J_len)
    else:
        # initialize the Lagrangian dual multipliers with the lb solution
        lb_prob.optimize()
        if lb_prob.Status != GRB.OPTIMAL:
            # update the L_value and resolve lb_prob
            L_test_bool = True
            L_value_lb = sub_lb
            L_value_ub = L_value
            while L_test_bool:
                L_value = (L_value_lb + L_value_ub) / 2
                lb_prob.remove(lb_prob.getConstrByName("cons"))
                lb_prob.addConstr(gp.quicksum(lb_prob.getVarByName("pi[{}]".format(i)) * x_value[i] for i in range(J_len)) + lb_prob.getVarByName("theta") >= L_value, name = "cons")
                lb_prob.update()
                lb_prob.optimize()
                if lb_prob.Status == GRB.OPTIMAL:
                    L_value_lb = L_value
                    if abs(L_value_ub - L_value_lb) < 1e-2:
                        L_test_bool = False
                else:
                    L_value_ub = L_value
        pi_value = np.zeros(J_len)
        for i in range(J_len):
            pi_value[i] = lb_prob.getVarByName("pi[{}]".format(i)).X
    # set up the auxiliary problem to find the next pi_value
    level = sub_ub
    next_pi_prob = build_next_pi_problem(level, alpha, x_value, L_value, cutList, norm_option)

    # build the inner min subproblem with Lagrangian penalty term
    sub_prob = build_subproblem_lag(o, ho, T, W, c, y_option, pi_value)
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
        z_value = np.zeros(J_len)
        for i in range(J_len):
            z_value[i] = sub_prob.getVarByName("z[{}]".format(i)).X

        # add the cut to the cut list
        Vj = sub_prob.ObjVal        # V(\pi_j)
        V_list.append(Vj)
        if len(cutList) > 0:
            theta_j = np.min([np.inner(cutList[j][0], pi_value) + cutList[j][1] for j in range(len(cutList))])
        else:
            theta_j = np.Infinity
        # only generate the cut if Vj > theta_j
        if Vj < theta_j - 1e-4:
            cutList.append((-z_value, Vj + np.inner(z_value, pi_value)))
            cutList_update_start_ind = len(cutList) - 1
            cutList_update_end_ind = len(cutList)
        else:
            cutList_update_start_ind = len(cutList)
            cutList_update_end_ind = len(cutList)

        # update and solve the lower bound problem
        lb_prob = update_ls_lb_problem(lb_prob, J_len, cutList, range(cutList_update_start_ind, cutList_update_end_ind))
        lb_prob.optimize()
        if lb_prob.Status != GRB.OPTIMAL:
            # update the L_value and resolve lb_prob
            L_test_bool = True
            L_value_lb = sub_lb
            L_value_ub = L_value
            while L_test_bool:
                L_value = (L_value_lb + L_value_ub) / 2
                lb_prob.remove(lb_prob.getConstrByName("cons"))
                lb_prob.addConstr(gp.quicksum(lb_prob.getVarByName("pi[{}]".format(i)) * x_value[i] for i in range(J_len)) + lb_prob.getVarByName("theta") >= L_value, name = "cons")
                lb_prob.update()
                lb_prob.optimize()
                if lb_prob.Status == GRB.OPTIMAL:
                    L_value_lb = L_value
                    if abs(L_value_ub - L_value_lb) < 1e-2:
                        L_test_bool = False
                else:
                    L_value_ub = L_value

        # update the alpha
        alpha_max, alpha_min, Delta = obtain_alpha_bounds(pi_list, L_value, x_value, lb_prob.ObjVal, V_list, norm_option)
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
                    v_bar_list = [alpha * np.inner(pi_list[pi_j_ind], pi_list[pi_j_ind]) + (1 - alpha) * (L_value - np.inner(pi_list[pi_j_ind], x_value) - V_list[pi_j_ind]) for pi_j_ind in range(len(pi_list))]
                else:
                    v_bar_list = [alpha * np.sum(np.abs(pi_list[pi_j_ind])) + (1 - alpha) * (L_value - np.inner(pi_list[pi_j_ind], x_value) - V_list[pi_j_ind]) for pi_j_ind in range(len(pi_list))]
                v_bar = np.min(v_bar_list)
                v_underbar = alpha * lb_prob.ObjVal
                level = lambda_level * v_bar + (1 - lambda_level) * v_underbar

                # solve for the next pi_value
                next_pi_prob = update_next_pi_problem(next_pi_prob, J_len, cutList, range(cutList_update_start_ind, cutList_update_end_ind), alpha, x_value, L_value, pi_value, level, norm_option)
                next_pi_prob.optimize()
                # obtain the next pi_value
                pi_value = np.zeros(J_len)
                for i in range(J_len):
                    if next_pi_prob.Status != GRB.OPTIMAL:
                        print("next_pi_prob is not optimal")
                        cont_bool = False
                    pi_value[i] = next_pi_prob.getVarByName("pi[{}]".format(i)).X
                # update the subproblem and solve it
                sub_prob = update_subproblem_lag(sub_prob, c, pi_value)
                sub_prob.optimize()

                # output the current status
        print("Iteration: {}, V(pi_j): {}, Delta: {}".format(counter, sub_prob.ObjVal, Delta))

    # obtain the intercept of the Lagrangian cut
    v_value = sub_prob.ObjVal
    return pi_value, v_value, cutList

if __name__ == "__main__":
    # initialize the data
    omega = 50          # number of scenarios
    J_len = 2        # number of first-stage decision variables
    I_len = 4        # number of second-stage decision variables
    h = np.round(np.random.uniform(5,15,[omega,J_len]),5)
    T = np.array([[1,0],[0,1]])
    # T = np.array([[2/3,1/3],[1/3,2/3]])
    y_option = 0            # 0 represents binary
    # y_option = 1          # 1 represents general integers
    # norm_option = 0       # 0 represents the L2 norm
    norm_option = 1         # 1 represents the L1 norm
    W = np.array([[2,3,4,5],[6,1,3,2]])
    c = np.array([-16,-19,-23,-28])
    c1 = np.array([-3/2,-4])
    LB = -np.Infinity
    UB = np.Infinity
    lambda_level = 0.5
    mu_level = 0.6
    iter_bool = True
    # initialize the dictionary to store the Lagrangian cuts for the subproblems' convex envelope
    cut_Dict = {}
    for o in range(omega):
        cut_Dict[o] = []

    # # build the extensive form and solve it
    extensive_prob = build_extensive_form(omega, h, T, W, c, c1, y_option)
    extensive_prob.optimize()
    # obtain the extensive form solution/optimal value
    x_opt_value = np.zeros(J_len)
    for j in range(J_len):
        x_opt_value[j] = extensive_prob.getVarByName("x[" + str(j) + "]").X
    opt_value = extensive_prob.ObjVal

    # build the master problem
    master_prob = build_masterproblem(omega, c1)
    x_best = np.zeros(J_len)

    # iteration of the cutting plane algorithm
    while iter_bool:
        # solve the master problem
        master_prob.optimize()
        LB = master_prob.ObjVal

        # obtain the master soluton/optimal value & update the lower bound
        x_value = np.zeros(J_len)
        for j in range(J_len):
            x_value[j] = master_prob.getVarByName("x[" + str(j) + "]").X

        # iterate over the subproblem
        V_bar = np.inner(c1, x_value)
        for o in range(omega):
            # obtain the subproblem value & update the upper bound
            sub_prob = build_subproblem(o, h[o], T, W, c, y_option, x_value)
            sub_prob.optimize()
            # obtain the subproblem solution/optimal value and update the upper bound
            L_value = sub_prob.ObjVal 
            V_bar += L_value / omega

            # generate the Lagrangian cuts
            pi_value_o, v_value_o, cutList_o = solve_lag_dual(o, h[o], T, W, c, y_option, x_value, L_value, lambda_level, mu_level, norm_option, 1e-3, cut_Dict[o])
            cut_Dict[o] = cutList_o
            # update the master problem with the Lagrangian cuts
            if v_value_o + np.inner(pi_value_o, x_value) > master_prob.getVarByName("theta[{}]".format(o)).X:
                master_prob.addConstr(v_value_o + gp.quicksum(pi_value_o[j] * master_prob.getVarByName("x[{}]".format(j)) for j in range(J_len)) <= \
                                    master_prob.getVarByName("theta[{}]".format(o)))
        
        if V_bar < UB:
            UB = V_bar
            # record the best solution
            for j in range(J_len):
                x_best[j] = x_value[j]
        
        # check the stopping criterion
        if abs((UB - LB)/UB) < 1e-2:
            iter_bool = False
        else:
            # update the master problem
            master_prob.update()