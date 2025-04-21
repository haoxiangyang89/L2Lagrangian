import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Notes:
# 1. The optimization problem should have a minimization orientation.

# Create a new Gurobi environment
env = gp.Env(empty=True)
env.setParam('LogFile', 'gurobi.log')
env.start()

def build_extensive_form(omega, h, T, W, c, y_option):
    # construct the extensive formulation
    extensive_prob = gp.Model("extensive_form")
    x = extensive_prob.addVars(2, vtype=GRB.CONTINUOUS, lb=0, ub=5, name="x")
    x[0].Start = 0.0
    x[1].Start = 4.0
    if y_option == 0:
        y = extensive_prob.addVars(omega, 4, vtype=GRB.BINARY, name="y")
    else:
        y = extensive_prob.addVars(omega, 4, vtype=GRB.INTEGER, lb=0, name="y")
    # set up the objective function
    extensive_prob.setObjective(-3/2 * x[0] - 4 * x[1] + 1/omega * gp.quicksum(gp.quicksum(c[i] * y[o,i] for i in range(4)) for o in range(omega)), GRB.MINIMIZE)
    # set up the structural constraints
    extensive_prob.addConstrs((gp.quicksum(W[j,i] * y[o,i] for i in range(4)) <= h[o,j] -
                                gp.quicksum(T[j,k] * x[k] for k in range(2)) for j in range(2) for o in range(omega)), name = "cons")
    extensive_prob.update()
    return extensive_prob

def build_masterproblem(omega):
    # construct the master program
    master_prob = gp.Model("masterproblem")
    x = master_prob.addVars(2, vtype=GRB.CONTINUOUS, lb=0, ub=5, name="x")
    theta = master_prob.addVars(omega, vtype=GRB.CONTINUOUS, lb=-10000, name="theta")
    master_prob.setObjective(-3/2 * x[0] - 4*x[1] + 1/omega * gp.quicksum(theta[o] for o in range(omega)), GRB.MINIMIZE)
    master_prob.update()
    return master_prob

def build_subproblem(o, ho, T, W, c, y_option, x_value):
    # input: 
    # o - index of the scenario, ho - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # x_value - the optimal solution of the master problem

    sub_prob = gp.Model("subproblem_" + str(o))
    sub_prob.Params.OutputFlag = 0

    # set up the decision variables y, allowing two types, binary and integers
    if y_option == 0:
        y = sub_prob.addVars(4, vtype=GRB.BINARY, name="y")
    else:
        y = sub_prob.addVars(4, vtype=GRB.INTEGER, lb=0, name="y")

    # set up the objective function
    sub_prob.setObjective(gp.quicksum(c[i] * y[i] for i in range(4)), GRB.MINIMIZE)

    # set up the structural constraints
    cons = sub_prob.addConstrs((gp.quicksum(W[j,i] * y[i] for i in range(4)) <= ho[j] - 
                                gp.quicksum(T[j,k] * x_value[k] for k in range(2)) for j in range(2)), name = "cons")
    sub_prob.update()
    return sub_prob

def build_subproblem_lag(o, ho, T, W, c, y_option, pi_value):
    # input: 
    # o - index of the scenario, ho - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # pi_value - the Lagrangian dual multipliers

    sub_prob_lag = gp.Model("subproblem_lag_" + str(o))
    sub_prob_lag.Params.OutputFlag = 0

    # set up the auxiliary variables z (copy of x)
    z = sub_prob_lag.addVars(2, vtype=GRB.CONTINUOUS, lb=0, ub=5, name="z")

    # set up the decision variables y, allowing two types, binary and integers
    if y_option == 0:
        y = sub_prob_lag.addVars(4, vtype=GRB.BINARY, name="y")
    else:
        y = sub_prob_lag.addVars(4, vtype=GRB.INTEGER, lb=0, name="y")

    # set up the objective function with Lagrangian penalty term
    sub_prob_lag.setObjective(gp.quicksum(c[i] * y[i] for i in range(4)) -
                              gp.quicksum(pi_value[k] * z[k] for k in range(2)), GRB.MINIMIZE)
    
    # set up the structural constraints
    cons = sub_prob_lag.addConstrs((gp.quicksum(W[j,i] * y[i] for i in range(4)) <= ho[j] - 
                                gp.quicksum(T[j,k] * z[k] for k in range(2)) for j in range(2)), name = "cons")

    sub_prob_lag.update()
    return sub_prob_lag

def update_subproblem_lag(sub_prob_lag, c, pi_value):
    # set up the objective function with Lagrangian penalty term
    sub_prob_lag.setObjective(gp.quicksum(c[i] * sub_prob_lag.getVarByName("y[{}]".format(i)) for i in range(4)) -
                              gp.quicksum(pi_value[k] * sub_prob_lag.getVarByName("z[{}]".format(k)) for k in range(2)), GRB.MINIMIZE)
    sub_prob_lag.update()
    return sub_prob_lag

# Build the level set lower bound problem
def build_ls_lb_problem(x_value, L_value, cutList):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    lb_prob = gp.Model("lb_problem")
    lb_prob.Params.OutputFlag = 0

    # set up the dual variables pi and auxiliary variables theta
    pi = lb_prob.addVars(2, vtype=GRB.CONTINUOUS, lb=-10000, ub=10000, name="pi")
    theta = lb_prob.addVar(vtype=GRB.CONTINUOUS, lb=-10000, name="theta")

    # set up the objective function with Lagrangian penalty term
    lb_prob.setObjective(gp.quicksum(pi[i] * pi[i] for i in range(2)), GRB.MINIMIZE)

    # set up the structural constraints
    lb_prob.addConstrs((gp.quicksum(cutList[j][0][i] * pi[i] for i in range(2)) + cutList[j][1] >= theta
                                for j in range(len(cutList))), name = "cuts")
    lb_prob.addConstr(gp.quicksum(pi[i] * x_value[i] for i in range(2)) + theta >= L_value, name = "cons")
    lb_prob.update()
    return lb_prob

# update the level set lower bound problem
def update_ls_lb_problem(lb_prob, cutList, update_ind_range):
    # input: 
    # lb_prob - the level set lower bound problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    for j in update_ind_range:
        lb_prob.addConstr(gp.quicksum(cutList[j][0][i] * lb_prob.getVarByName("pi[{}]".format(i)) for i in range(2)) + \
                          cutList[j][1] >= lb_prob.getVarByName("theta"), name = "cuts[{}]".format(j))
    lb_prob.update()
    return lb_prob

# Build the level set lower bound problem
def build_next_pi_problem(level, alpha, x_value, L_value, cutList):
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)
    next_pi_prob = gp.Model("next_pi_prob")
    next_pi_prob.Params.OutputFlag = 0
    # set up the dual variables pi and auxiliary variables theta
    pi = next_pi_prob.addVars(2, vtype=GRB.CONTINUOUS, lb=-10000, ub=10000, name="pi")
    theta = next_pi_prob.addVar(vtype=GRB.CONTINUOUS, lb=-10000, name="theta")

    # set up the structural constraints
    next_pi_prob.addConstrs((gp.quicksum(cutList[j][0][i] * pi[i] for i in range(2)) + cutList[j][1] >= theta
                                for j in range(len(cutList))), name = "cuts")
    next_pi_prob.addConstr(alpha * gp.quicksum(pi[i] * pi[i] for i in range(2)) + 
                           (1 - alpha) * (L_value - gp.quicksum(pi[i] * x_value[i] for i in range(2)) - theta) <= level, name = "level_cons")

    next_pi_prob.update()
    return next_pi_prob

def update_next_pi_problem(next_pi_prob, cutList, update_ind_range, alpha, x_value, L_value, pi_bar_value, level):
    # input: 
    # next_pi_prob - the next pi problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    next_pi_prob.remove(next_pi_prob.getQConstrs()[0])
    next_pi_prob.addConstr(alpha * gp.quicksum(next_pi_prob.getVarByName("pi[{}]".format(i)) * next_pi_prob.getVarByName("pi[{}]".format(i)) for i in range(2)) + 
                           (1 - alpha) * (L_value - gp.quicksum(next_pi_prob.getVarByName("pi[{}]".format(i)) * x_value[i] for i in range(2)) - next_pi_prob.getVarByName("theta")) <= level, name = "level_cons")
    for j in update_ind_range:
        next_pi_prob.addConstr(gp.quicksum(cutList[j][0][i] * next_pi_prob.getVarByName("pi[{}]".format(i)) for i in range(2)) + \
                          cutList[j][1] >= next_pi_prob.getVarByName("theta"), name = "cuts[{}]".format(j))
    # set up the objective function
    next_pi_prob.setObjective(gp.quicksum((next_pi_prob.getVarByName("pi[{}]".format(i)) - pi_bar_value[i]) ** 2 for i in range(2)), GRB.MINIMIZE)

    next_pi_prob.update()
    return next_pi_prob

def obtain_alpha_bounds(pi_list, L_value, x_value, v_underbar, V_list):
    # input: 
    # alpha_prob - the alpha problem with piecewise linear objective function
    # return the upper and lower bounds of alpha
    alpha_min = 0
    alpha_max = 1

    # set up the alpha problem
    alpha_prob = gp.Model("alpha_problem")
    alpha_prob.Params.OutputFlag = 0
    alpha = alpha_prob.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="alpha")
    alpha_obj = alpha_prob.addVar(vtype=GRB.CONTINUOUS, lb=-10000, name="alpha_obj")
    alpha_prob.addConstr(alpha_obj >= 0, name="alpha_obj_lb")
    alpha_prob.addConstrs((alpha_obj <= alpha * (np.inner(pi_list[j], pi_list[j]) - v_underbar) + \
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

# procedure to solve the Lagrangian dual problem
def solve_lag_dual(o, ho, T, W, c, y_option, x_value, L_value, lambda_level, tol = 1e-2, cutList = [], ub = 100000):
    # input: 
    # o - index of the scenario, ho - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # x_value - the optimal solution of the master problem
    # cutList - the list of cuts generated for the inner minimization problem so far, 
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # initialize the Lagrangian dual multipliers and cut list for the level set problem
    pi_value = np.zeros(2)
    v_value = 0
    alpha_min = 0
    alpha_max = 1
    alpha = (alpha_max + alpha_min) / 2
    pi_list = []
    V_list = []

    # set up the lower bound problem
    lb_prob = build_ls_lb_problem(x_value, L_value, cutList)
    # set up the auxiliary problem to find the next pi_value
    level = ub
    next_pi_prob = build_next_pi_problem(level, alpha, x_value, L_value, cutList)

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
        z_value = np.zeros(2)
        for i in range(2):
            z_value[i] = sub_prob.getVarByName("z[{}]".format(i)).X

        # add the cut to the cut list
        Vj = sub_prob.ObjVal        # V(\pi_j)
        V_list.append(Vj)
        if len(cutList) > 0:
            theta_j = np.max([np.inner(cutList[j][0], pi_value) + cutList[j][1] for j in range(len(cutList))])
        else:
            theta_j = np.Infinity
        # only generate the cut if Vj > theta_j
        if Vj < theta_j:
            cutList.append((-z_value, Vj + np.inner(z_value, pi_value)))
            cutList_update_start_ind = len(cutList) - 1
            cutList_update_end_ind = len(cutList)
        else:
            cutList_update_start_ind = len(cutList)
            cutList_update_end_ind = len(cutList)

        # update and solve the lower bound problem
        lb_prob = update_ls_lb_problem(lb_prob, cutList, range(cutList_update_start_ind, cutList_update_end_ind))
        lb_prob.optimize()
        if lb_prob.Status != GRB.OPTIMAL:
            # update the L_value and resolve lb_prob
            L_test_bool = True
            L_value_lb = -10000
            L_value_ub = L_value
            while L_test_bool:
                L_value = (L_value_lb + L_value_ub) / 2
                lb_prob.remove(lb_prob.getConstrByName("cons"))
                lb_prob.addConstr(gp.quicksum(lb_prob.getVarByName("pi[{}]".format(i)) * x_value[i] for i in range(2)) + lb_prob.getVarByName("theta") >= L_value, name = "cons")
                lb_prob.update()
                lb_prob.optimize()
                if lb_prob.Status == GRB.OPTIMAL:
                    L_value_lb = L_value
                    if abs(L_value_ub - L_value_lb) < 1e-2:
                        L_test_bool = False
                else:
                    L_value_ub = L_value

        # update the alpha
        alpha_max, alpha_min, Delta = obtain_alpha_bounds(pi_list, L_value, x_value, lb_prob.ObjVal, V_list)
        alpha = (alpha_max + alpha_min) / 2

        # update the termination indicator
        if Delta < tol:
            cont_bool = False
        else:
            if counter > 200:
                cont_bool = False
            else:
                counter += 1
                # update the level
                v_bar_list = [alpha * np.inner(pi_list[pi_j_ind], pi_list[pi_j_ind]) + (1 - alpha) * (L_value - np.inner(pi_list[pi_j_ind], x_value) - V_list[pi_j_ind]) for pi_j_ind in range(len(pi_list))]
                v_bar = np.min(v_bar_list)
                v_underbar = alpha * lb_prob.ObjVal
                level = lambda_level * v_bar + (1 - lambda_level) * v_underbar

                # solve for the next pi_value
                next_pi_prob = update_next_pi_problem(next_pi_prob, cutList, range(cutList_update_start_ind, cutList_update_end_ind), alpha, x_value, L_value, pi_value, level)
                next_pi_prob.optimize()
                # obtain the next pi_value
                pi_value = np.zeros(2)
                for i in range(2):
                    if next_pi_prob.Status != GRB.OPTIMAL:
                        print("next_pi_prob is not optimal")
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
    omega = 10
    #h = np.random.uniform(5,15,[omega,2])
    h = np.round(np.array([[ 8.93541994, 11.34106653],
       [ 6.95796047,  9.05425548],
       [12.23509333, 14.82091246],
       [13.01313764, 12.59155558],
       [ 5.49651947,  8.81261555],
       [ 6.63768811,  9.94772575],
       [14.15786958, 13.48730714],
       [ 9.63203678,  7.00587213],
       [ 8.6734546 ,  9.32073894],
       [11.60052146,  6.61219049]]),5)
    T = np.array([[1,0],[0,1]])
    # T = np.array([[2/3,1/3],[1/3,2/3]])
    y_option = 0            # 0 represents binary
    # y_option = 1          # 1 represents general integers
    W = np.array([[2,3,4,5],[6,1,3,2]])
    c = np.array([-16,-19,-23,-28])
    LB = -np.Infinity
    UB = np.Infinity
    lambda_level = 0.5
    iter_bool = True
    cut_Dict = {}
    for o in range(omega):
        cut_Dict[o] = []

    # # build the extensive form and solve it
    extensive_prob = build_extensive_form(omega, h, T, W, c, y_option)
    extensive_prob.optimize()
    # obtain the extensive form solution/optimal value
    x_opt_value = np.zeros(2)
    for j in range(2):
        x_opt_value[j] = extensive_prob.getVarByName("x[" + str(j) + "]").X
    opt_value = extensive_prob.ObjVal

    # build the master problem
    master_prob = build_masterproblem(omega)

    # iteration of the cutting plane algorithm
    
    while iter_bool:
        # solve the master problem
        master_prob.optimize()
        LB = master_prob.ObjVal

        # obtain the master soluton/optimal value & update the lower bound
        x_value = np.zeros(2)
        for j in range(2):
            x_value[j] = master_prob.getVarByName("x[" + str(j) + "]").X

        # iterate over the subproblem
        V_bar = -3/2 * x_value[0] - 4 * x_value[1]
        for o in range(omega):
            # obtain the subproblem value & update the upper bound
            sub_prob = build_subproblem(o, h[o], T, W, c, y_option, x_value)
            sub_prob.optimize()
            # obtain the subproblem solution/optimal value and update the upper bound
            L_value = sub_prob.ObjVal 
            V_bar += L_value / omega

            # generate the Lagrangian cuts
            pi_value_o, v_value_o, cutList_o = solve_lag_dual(o, h[o], T, W, c, y_option, x_value, L_value, lambda_level, 1e-3, cut_Dict[o])
            cut_Dict[o] = cutList_o
            # update the master problem with the Lagrangian cuts
            if v_value_o + np.inner(pi_value_o, x_value) > master_prob.getVarByName("theta[{}]".format(o)).X:
                master_prob.addConstr(v_value_o + gp.quicksum(pi_value_o[i] * master_prob.getVarByName("x[{}]".format(i)) for i in range(2)) <= \
                                    master_prob.getVarByName("theta[{}]".format(o)))
        
        if V_bar < UB:
            UB = V_bar
        
        # check the stopping criterion
        if abs((UB - LB)/UB) < 1e-2:
            iter_bool = False
        else:
            # update the master problem
            master_prob.update()