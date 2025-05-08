import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Notes:
# 1. The optimization problem should have a minimization orientation.

# Create a new Gurobi environment
env = gp.Env(empty=True)
env.setParam('LogFile', 'gurobi.log')
env.start()

def build_extensive_form(omega, c, v, u, d, h, q0, q):
    # construct the extensive formulation
    extensive_prob = gp.Model("extensive_form")

    # obtain problem dimensions
    J_len = len(c)
    J = range(J_len)
    I_len = len(h[0])
    I = range(I_len)

    # set up the decision variables
    x = extensive_prob.addVars(J_len, vtype=GRB.BINARY, name="x")
    y = extensive_prob.addVars(omega, I_len, J_len, vtype=GRB.BINARY, name="y")
    y0 = extensive_prob.addVars(omega, J_len, vtype=GRB.CONTINUOUS, lb=0, name="y0")

    # set up the objective function
    extensive_prob.setObjective(gp.quicksum(c[j] * x[j] for j in J) + 1/omega * gp.quicksum(\
            gp.quicksum(q0[o][j] * y0[o,j] + gp.quicksum(q[o][i][j] * y[o,i,j] for i in I) for j in J)
         for o in range(omega)), GRB.MINIMIZE)
    # set up the structural constraints
    extensive_prob.addConstr((gp.quicksum(x[j] for j in J) <= v), name = "server_no_cons")
    extensive_prob.addConstrs((gp.quicksum(d[i,j] * y[o,i,j] for i in I) - y0[o,j] <= u * x[j] for j in J for o in range(omega)),name = "capacity_cons")
    extensive_prob.addConstrs((gp.quicksum(y[o,i,j] for j in J) == h[o][i] for i in I for o in range(omega)), name = "demand_cons")
    extensive_prob.update()
    return extensive_prob

def build_masterproblem(omega, c, v, prob_lb=-100000):
    # construct the master program
    master_prob = gp.Model("masterproblem")
    master_prob.Params.OutputFlag = 0
    J_len = len(c)
    J = range(J_len)
    x = master_prob.addVars(J_len, vtype=GRB.BINARY, name="x")
    theta = master_prob.addVars(omega, vtype=GRB.CONTINUOUS, lb=prob_lb, name="theta")
    master_prob.setObjective(gp.quicksum(c[j] * x[j] for j in J) + 1/omega * gp.quicksum(theta[o] for o in range(omega)), GRB.MINIMIZE)
    # set up the structural constraints
    master_prob.addConstr((gp.quicksum(x[j] for j in J) <= v), name = "server_no_cons")
    master_prob.update()
    return master_prob

def build_subproblem(o, c, u, do, ho, q0o, qo, x_value):
    # input: 
    # o - index of the scenario, do - customer demand, ho - customer availability,
    # q0o - unit penalty cost, qo - unit sales revenue, u - capacity of each server
    # x_value - the optimal solution of the master problem

    sub_prob = gp.Model("subproblem_" + str(o))
    sub_prob.Params.OutputFlag = 0

    # obtain problem dimensions
    J_len = len(c)
    J = range(J_len)
    I_len = len(ho)
    I = range(I_len)

    # set up the decision variables
    y = sub_prob.addVars(I_len, J_len, vtype=GRB.BINARY, name="y")
    y0 = sub_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=0, name="y0")

    # set up the objective function
    sub_prob.setObjective(gp.quicksum(q0o[j] * y0[j] + gp.quicksum(qo[i][j] * y[i,j] for i in I) for j in J), GRB.MINIMIZE)

    # set up the structural constraints
    sub_prob.addConstrs((gp.quicksum(do[i,j] * y[i,j] for i in I) - y0[j] <= u * x_value[j] for j in J),name = "capacity_cons")
    sub_prob.addConstrs((gp.quicksum(y[i,j] for j in J) == ho[i] for i in I), name = "demand_cons")

    sub_prob.update()
    return sub_prob

def build_subproblem_lag(o, c, u, do, ho, q0o, qo, pi_value):
    # input: 
    # o - index of the scenario, do - customer demand, ho - customer availability,
    # q0o - unit penalty cost, qo - unit sales revenue, u - capacity of each server
    # pi_value - the Lagrangian dual multipliers

    sub_prob_lag = gp.Model("subproblem_lag_" + str(o))
    sub_prob_lag.Params.OutputFlag = 0

    # obtain problem dimensions
    J_len = len(c)
    J = range(J_len)
    I_len = len(ho)
    I = range(I_len)

    # set up the auxiliary variables z (copy of x)
    z = sub_prob_lag.addVars(J_len, vtype=GRB.BINARY, name="z")

    # set up the decision variables
    y = sub_prob_lag.addVars(I_len, J_len, vtype=GRB.BINARY, name="y")
    y0 = sub_prob_lag.addVars(J_len, vtype=GRB.CONTINUOUS, lb=0, name="y0")

    # set up the objective function with Lagrangian penalty term
    sub_prob_lag.setObjective(gp.quicksum(q0o[j] * y0[j] + gp.quicksum(qo[i][j] * y[i,j] for i in I) for j in J) -
                              gp.quicksum(pi_value[j] * z[j] for j in J), GRB.MINIMIZE)
    
    # set up the structural constraints
    sub_prob_lag.addConstrs((gp.quicksum(do[i,j] * y[i,j] for i in I) - y0[j] <= u * z[j] for j in J),name = "capacity_cons")
    sub_prob_lag.addConstrs((gp.quicksum(y[i,j] for j in J) == ho[i] for i in I), name = "demand_cons")

    sub_prob_lag.update()
    return sub_prob_lag

def update_subproblem_lag(sub_prob_lag, I_len, J_len, q0o, qo, pi_value):
    # set up the objective function with Lagrangian penalty term
    sub_prob_lag.setObjective(gp.quicksum(q0o[j] * sub_prob_lag.getVarByName("y0[{}]".format(j)) + 
                                          gp.quicksum(qo[i][j] * sub_prob_lag.getVarByName("y[{},{}]".format(i,j)) for i in range(I_len)) for j in range(J_len)) -
                              gp.quicksum(pi_value[j] * sub_prob_lag.getVarByName("z[{}]".format(j)) for j in range(J_len)), GRB.MINIMIZE)
    sub_prob_lag.update()
    return sub_prob_lag

# Build the level set lower bound problem
def build_ls_lb_problem(J_len, x_value, L_value, cutList, x_tilde, prob_lb=-2000, prob_ub=2000):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    lb_prob = gp.Model("lb_problem")
    lb_prob.Params.OutputFlag = 0

    # set up the dual variables pi and auxiliary variables theta
    pi = lb_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=prob_lb, ub=prob_ub, name="pi")
    theta = lb_prob.addVar(vtype=GRB.CONTINUOUS, lb=1000*prob_lb, name="theta")

    # set up the objective function with Lagrangian penalty term
    lb_prob.setObjective(-gp.quicksum(pi[j] * x_tilde[j] for j in range(J_len)) - theta, GRB.MINIMIZE)

    # set up the structural constraints
    lb_prob.addConstrs((gp.quicksum(cutList[k][0][j] * pi[j] for j in range(J_len)) + cutList[k][1] >= theta
                                for k in range(len(cutList))), name = "cuts")
    lb_prob.addConstr(gp.quicksum(pi[j] * x_value[j] for j in range(J_len)) + theta >= L_value, name = "cons")
    lb_prob.update()
    return lb_prob

# update the level set lower bound problem
def update_ls_lb_problem(lb_prob, J_len, x_tilde, cutList, update_ind_range):
    # input: 
    # lb_prob - the level set lower bound problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    for k in update_ind_range:
        lb_prob.addConstr(gp.quicksum(cutList[k][0][j] * lb_prob.getVarByName("pi[{}]".format(j)) for j in range(J_len)) + \
                          cutList[k][1] >= lb_prob.getVarByName("theta"), name = "cuts[{}]".format(k))
    # set up the objective function with Lagrangian penalty term
    lb_prob.setObjective(-gp.quicksum(lb_prob.getVarByName("pi[{}]".format(j)) * x_tilde[j] for j in range(J_len)) - lb_prob.getVarByName("theta"), GRB.MINIMIZE)

    lb_prob.update()
    return lb_prob

# Build the level set lower bound problem
def build_next_pi_problem(J_len, level, alpha, x_value, L_value, cutList, x_tilde, prob_lb=-10000, prob_ub=10000):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)
    next_pi_prob = gp.Model("next_pi_prob")
    next_pi_prob.Params.OutputFlag = 0
    # set up the dual variables pi and auxiliary variables theta
    pi = next_pi_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=prob_lb, ub=prob_ub, name="pi")
    theta = next_pi_prob.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="theta")
    pi_obj_abs = next_pi_prob.addVars(J_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=np.maximum(np.abs(prob_ub),np.abs(prob_lb)), name="pi_obj")

    # set up the structural constraints
    next_pi_prob.addConstrs((gp.quicksum(cutList[k][0][j] * pi[j] for j in range(J_len)) + cutList[k][1] >= theta
                                for k in range(len(cutList))), name = "cuts")
    next_pi_prob.addConstr(-alpha * (gp.quicksum(pi[j] * x_tilde[j] for j in range(J_len)) + theta) + 
                        (1 - alpha) * (L_value - gp.quicksum(pi[j] * x_value[j] for j in range(J_len)) - theta) <= level, name = "level_cons")

    # set up the objective function absolute value term
    next_pi_prob.addConstrs((pi_obj_abs[j] - pi[j] >= 0 for j in range(J_len)), name = "pi_obj_pos")
    next_pi_prob.addConstrs((pi_obj_abs[j] + pi[j] >= 0 for j in range(J_len)), name = "pi_obj_neg")

    # set up the objective function
    next_pi_prob.setObjective(gp.quicksum(pi_obj_abs[j] for j in range(J_len)), GRB.MINIMIZE)

    next_pi_prob.update()
    return next_pi_prob

def update_next_pi_problem(next_pi_prob, J_len, cutList, update_ind_range, alpha, x_value, L_value, pi_bar_value, level, x_tilde):
    # input: 
    # next_pi_prob - the next pi problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    next_pi_prob.remove(next_pi_prob.getConstrByName("level_cons"))
    next_pi_prob.addConstr(-alpha * (gp.quicksum(next_pi_prob.getVarByName("pi[{}]".format(j)) * x_tilde[j] for j in range(J_len)) + next_pi_prob.getVarByName("theta")) + 
                (1 - alpha) * (L_value - gp.quicksum(next_pi_prob.getVarByName("pi[{}]".format(j)) * x_value[j] for j in range(J_len)) - next_pi_prob.getVarByName("theta")) <= level, name = "level_cons")

    for k in update_ind_range:
        next_pi_prob.addConstr(gp.quicksum(cutList[k][0][j] * next_pi_prob.getVarByName("pi[{}]".format(j)) for j in range(J_len)) + \
                          cutList[k][1] >= next_pi_prob.getVarByName("theta"), name = "cuts[{}]".format(k))

    # set up the objective function absolute value rhs term
    for j in range(J_len):
        pos_constr = next_pi_prob.getConstrByName("pi_obj_pos[{}]".format(j))
        neg_constr = next_pi_prob.getConstrByName("pi_obj_neg[{}]".format(j))
        next_pi_prob.setAttr("RHS", pos_constr, -pi_bar_value[j])
        next_pi_prob.setAttr("RHS", neg_constr, pi_bar_value[j])

    next_pi_prob.update()
    return next_pi_prob

def obtain_alpha_bounds_opt(J_len, pi_list, L_value, x_value, v_underbar, V_list, x_tilde, prob_lb=-100000):
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
    alpha_prob.addConstrs((alpha_obj <= alpha * (-(np.inner(pi_list[k], x_tilde) + V_list[k]) - v_underbar) + \
                                (1 - alpha) * (L_value - np.inner(pi_list[k], x_value) - V_list[k])
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

def obtain_alpha_bounds(J_len, pi_list, L_value, x_value, v_underbar, V_list, x_tilde):
    # algebraic way to calculate alpha_max and alpha_min
    alpha_underbar = []
    alpha_bar = []
    gamma_list = {}
    eta_list = {}

    for k in range(len(pi_list)):
        gamma_list[k] = (-(np.inner(pi_list[k], x_tilde) + V_list[k]) - v_underbar) - \
                (L_value - np.inner(pi_list[k], x_value) - V_list[k])
        eta_list[k] = (L_value - np.inner(pi_list[k], x_value) - V_list[k])
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
def solve_lag_dual(o, c, u, do, ho, q0o, qo, x_value, L_value, lambda_level, mu_level, x_tilde, tol = 1e-2, cutList = [], sub_lb = -10000, sub_ub = 100000):
    # input: 
    # o - index of the scenario, do - customer demand, ho - customer availability,
    # q0o - unit penalty cost, qo - unit sales revenue, u - capacity of each server
    # x_value - the optimal solution of the master problem
    # cutList - the list of cuts generated for the inner minimization problem so far, 
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # obtain problem dimensions
    J_len = len(c)
    J = range(J_len)
    I_len = len(ho)
    I = range(I_len)

    # initialize the Lagrangian dual multipliers and cut list for the level set problem
    pi_value = np.zeros(J_len)
    v_value = 0
    alpha_min = 0
    alpha_max = 1
    alpha = (alpha_max + alpha_min) / 2
    pi_list = []
    V_list = []

    # set up the lower bound problem
    lb_prob = build_ls_lb_problem(J_len, x_value, L_value, cutList, x_tilde)
    # set up the auxiliary problem to find the next pi_value
    level = sub_ub
    next_pi_prob = build_next_pi_problem(J_len, level, alpha, x_value, L_value, cutList, x_tilde)

    # build the inner min subproblem with Lagrangian penalty term
    sub_prob = build_subproblem_lag(o, c, u, do, ho, q0o, qo, pi_value)
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
        for j in range(J_len):
            z_value[j] = sub_prob.getVarByName("z[{}]".format(j)).X

        # add the cut to the cut list
        Vj = sub_prob.ObjVal        # V(\pi_j)
        V_list.append(Vj)
        if len(cutList) > 0:
            theta_j = np.min([np.inner(cutList[k][0], pi_value) + cutList[k][1] for k in range(len(cutList))])
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
        lb_prob = update_ls_lb_problem(lb_prob, J_len, x_tilde, cutList, range(cutList_update_start_ind, cutList_update_end_ind))
        lb_prob.optimize()
        if lb_prob.Status != GRB.OPTIMAL:
            # update the L_value and resolve lb_prob
            L_test_bool = True
            L_value_lb = sub_lb
            L_value_ub = L_value
            while L_test_bool:
                L_value = (L_value_lb + L_value_ub) / 2
                lb_prob.remove(lb_prob.getConstrByName("cons"))
                lb_prob.addConstr(gp.quicksum(lb_prob.getVarByName("pi[{}]".format(j)) * x_value[j] for j in range(J_len)) + lb_prob.getVarByName("theta") >= L_value, name = "cons")
                lb_prob.update()
                lb_prob.optimize()
                if lb_prob.Status == GRB.OPTIMAL:
                    L_value_lb = L_value
                    if abs(L_value_ub - L_value_lb) < 1e-2:
                        L_test_bool = False
                else:
                    L_value_ub = L_value

        # update the alpha
        alpha_max, alpha_min, Delta = obtain_alpha_bounds(J_len, pi_list, L_value, x_value, lb_prob.ObjVal, V_list, x_tilde)
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
                v_bar_list = [-alpha * (np.inner(pi_list[pi_k], x_tilde) + V_list[pi_k]) + (1 - alpha) * (L_value - np.inner(pi_list[pi_k], x_value) - V_list[pi_k]) for pi_k in range(len(pi_list))]                
                v_bar = np.min(v_bar_list)
                v_underbar = alpha * lb_prob.ObjVal
                level = lambda_level * v_bar + (1 - lambda_level) * v_underbar

                # solve for the next pi_value
                next_pi_prob = update_next_pi_problem(next_pi_prob, J_len, cutList, range(cutList_update_start_ind, cutList_update_end_ind), alpha, x_value, L_value, pi_value, level, x_tilde)
                next_pi_prob.optimize()
                # obtain the next pi_value
                pi_value = np.zeros(J_len)
                for j in range(J_len):
                    if next_pi_prob.Status != GRB.OPTIMAL:
                        print("next_pi_prob is not optimal")
                        cont_bool = False
                    pi_value[j] = next_pi_prob.getVarByName("pi[{}]".format(j)).X
                # update the subproblem and solve it
                sub_prob = update_subproblem_lag(sub_prob, I_len, J_len, q0o, qo, pi_value)
                sub_prob.optimize()

                # output the current status
        print("Iteration: {}, V(pi_j): {}, Delta: {}".format(counter, sub_prob.ObjVal, Delta))

    # obtain the intercept of the Lagrangian cut
    v_value = sub_prob.ObjVal
    return pi_value, v_value, cutList

if __name__ == "__main__":
    # initialize the data
    omega = 50          # number of scenarios

    J_len = 10
    I_len = 50
    c = np.round(np.random.uniform(40, 80, J_len), 4)
    v = 10
    u = 120
    d = np.round(np.random.uniform(0, 25, (I_len, J_len)), 4)
    r = np.round(v * u / sum(np.max(d[i]) for i in range(I_len)),4)
    h = np.random.randint(0, 2, (omega, I_len))
    q = np.ones([omega, I_len, J_len])
    q0 = np.ones([omega, J_len]) * 1000

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
    extensive_prob = build_extensive_form(omega, c, v, u, d, h, q0, q)
    extensive_prob.optimize()
    # obtain the extensive form solution/optimal value
    x_opt_value = np.zeros(J_len)
    for j in range(J_len):
        x_opt_value[j] = extensive_prob.getVarByName("x[" + str(j) + "]").X
    opt_value = extensive_prob.ObjVal

    # build the master problem
    master_prob = build_masterproblem(omega, c, v)
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

        # obtain x_tilde
        x_tilde = np.zeros(J_len)
        for j in range(J_len):
            x_tilde[j] = 0.5

        # iterate over the subproblem
        V_bar = sum(c[j] * x_value[j] for j in range(J_len))
        for o in range(omega):
            # obtain the subproblem value & update the upper bound
            sub_prob = build_subproblem(o, c, u, d, h[o], q0[o], q[o], x_value)
            sub_prob.optimize()
            # obtain the subproblem solution/optimal value and update the upper bound
            L_value = sub_prob.ObjVal 
            V_bar += L_value / omega

            # generate the Lagrangian cuts
            pi_value_o, v_value_o, cutList_o = solve_lag_dual(o, c, u, d, h[o], q0[o], q[o], x_value, L_value, lambda_level, mu_level, x_tilde, 1e-3, cut_Dict[o])
            cut_Dict[o] = cutList_o
            # update the master problem with the Lagrangian cuts
            if v_value_o + np.inner(pi_value_o, x_value) > master_prob.getVarByName("theta[{}]".format(o)).X:
                master_prob.addConstr(v_value_o + gp.quicksum(pi_value_o[i] * master_prob.getVarByName("x[{}]".format(i)) for i in range(J_len)) <= \
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