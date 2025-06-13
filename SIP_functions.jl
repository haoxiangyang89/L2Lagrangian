function build_extensive_form(omega, h, T, W, c, c1, y_option, start_value = [0.0, 4.0])
    # obtain the dimension of the problem
    J_len = length(W);  # number of first-stage decision variables
    I_len = length(W[1]);  # number of second-stage decision variables

    # construct the extensive formulation
    extensive_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV)));
    @variable(extensive_prob, 0 <= x[j in 1:J_len] <= 5, start = start_value[j]);
    if y_option == 0
        @variable(extensive_prob, y[o in 1:omega, i in 1:I_len], Bin);
    else
        @variable(extensive_prob, y[o in 1:omega, i in 1:I_len] >= 0, Int);
    end
    # set up the objective function
    @objective(extensive_prob, Min, sum(c1[j] * x[j] for j in 1:J_len) + 1/omega * sum(sum(c[i] * y[o,i] for i in 1:I_len) for o in 1:omega));
    # set up the structural constraints
    @constraint(extensive_prob, cons[j in 1:J_len, o in 1:omega], sum(W[j][i] * y[o,i] for i in 1:I_len) <= h[o,j] - sum(T[j][k] * x[k] for k in 1:J_len));
    return extensive_prob;
end

function build_masterproblem(omega, c1, xub = 5, prob_lb=-10000)
    # obtain the dimension of the problem
    J_len = length(c1)  # number of first-stage decision variables

    # construct the master program
    master_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0));
    @variable(master_prob, 0 <= x[j in 1:J_len] <= xub);
    @variable(master_prob, theta[o in 1:omega] >= prob_lb);
    # set up the objective function
    @objective(master_prob, Min, sum(c1[j] * x[j] for j in 1:J_len) + 1/omega * sum(theta[o] for o in 1:omega));
    return master_prob;
end

function build_subproblem(o, ho, T, W, c, y_option, x_value)
    # input: 
    # o - index of the scenario, ho - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # x_value - the optimal solution of the master problem

    # obtain the dimension of the problem
    J_len = length(W);  # number of first-stage decision variables
    I_len = length(W[1]);  # number of second-stage decision variables

    # create a subproblem model
    sub_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0));

    # set up the decision variables y, allowing two types, binary and integers
    if y_option == 0
        # binary decision variables
        @variable(sub_prob, y[i in 1:I_len], Bin);
    else
        # integer decision variables
        @variable(sub_prob, y[i in 1:I_len] >= 0, Int);
    end

    # set up the objective function
    @objective(sub_prob, Min, sum(c[i] * y[i] for i in 1:I_len));

    # set up the structural constraints
    @constraint(sub_prob, cons[j in 1:J_len], sum(W[j][i] * y[i] for i in 1:I_len) <= ho[j] - sum(T[j][k] * x_value[k] for k in 1:J_len));
    return sub_prob;
end

function build_subproblem_lag(o, ho, T, W, c, y_option, pi_value, xub = 5)
    # input: 
    # o - index of the scenario, ho - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # pi_value - the Lagrangian dual multipliers

    # obtain the dimension of the problem
    J_len = length(W);  # number of first-stage decision variables
    I_len = length(W[1]);  # number of second-stage decision variables

    # create a subproblem Lagrangian relaxation model
    sub_prob_lag = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0));

    # set up the auxiliary variables z (copy of x)
    @variable(sub_prob_lag, 0.0 <= z[k in 1:J_len] <= xub);

    # set up the decision variables y, allowing two types, binary and integers
    if y_option == 0
        # binary decision variables
        @variable(sub_prob_lag, y[i in 1:I_len], Bin);
    else
        # integer decision variables
        @variable(sub_prob_lag, y[i in 1:I_len] >= 0, Int);
    end

    # set up the objective function with Lagrangian penalty term
    @objective(sub_prob_lag, Min, 
        sum(c[i] * y[i] for i in 1:I_len) - 
        sum(pi_value[k] * z[k] for k in 1:J_len));
    
    # set up the structural constraints
    @constraint(sub_prob_lag, cons[j in 1:J_len], 
        sum(W[j][i] * y[i] for i in 1:I_len) <= ho[j] - 
        sum(T[j][k] * z[k] for k in 1:J_len));

    return sub_prob_lag;
end

function update_subproblem_lag(sub_prob_lag, c, pi_value)
    # obtain the dimension of the problem
    I_len = length(c);
    J_len = length(pi_value);

    # set up the objective function with Lagrangian penalty term
    @objective(sub_prob_lag, Min, 
        sum(c[i] * sub_prob_lag[:y][i] for i in 1:I_len) - 
        sum(pi_value[k] * sub_prob_lag[:z][k] for k in 1:J_len));
    return sub_prob_lag;
end

# Build the level set lower bound problem
function build_ls_lb_problem(x_value, L_value, cutList, norm_option, prob_lb=-10000, prob_ub=10000)
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # obtain the dimension of the problem
    J_len = length(x_value);  # number of first-stage decision variables

    # create a level set lower bound problem
    lb_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0));

    # set up the dual variables pi and auxiliary variables theta
    @variable(lb_prob, prob_lb <= pi_var[i in 1:J_len] <= prob_ub);
    @variable(lb_prob, theta >= prob_lb);

    # set up the objective function with Lagrangian penalty term
    if norm_option == 0
        # L2 norm
        @objective(lb_prob, Min, 
            sum(pi_var[i] * pi_var[i] for i in 1:J_len));
    else
        # L1 norm
        @variable(lb_prob, 0.0 <= pi_abs[i in 1:J_len] <= max(abs(prob_ub), abs(prob_lb)));
        @objective(lb_prob, Min, sum(pi_abs[i] for i in 1:J_len));
        @constraint(lb_prob, pi_abs_pos[i in 1:J_len], pi_var[i] <= pi_abs[i]);
        @constraint(lb_prob, pi_abs_neg[i in 1:J_len], -pi_var[i] <= pi_abs[i]);
    end

    # set up the structural constraints
    @constraint(lb_prob, [j in 1:length(cutList)], sum(cutList[j][1][i] * pi_var[i] for i in 1:J_len) + cutList[j][2] >= theta);
    @constraint(lb_prob, cons, sum(pi_var[i] * x_value[i] for i in 1:J_len) + theta >= L_value);
    return lb_prob;
end

# update the level set lower bound problem
function update_ls_lb_problem(lb_prob, J_len, cutList, update_ind_range)
    # input: 
    # lb_prob - the level set lower bound problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    for j in update_ind_range
        @constraint(lb_prob, sum(cutList[j][1][i] * lb_prob[:pi_var][i] for i in 1:J_len) + cutList[j][2] >= lb_prob[:theta]);
    end
    return lb_prob;
end

# Build the level set lower bound problem
function build_next_pi_problem(level, alpha, x_value, L_value, cutList, norm_option, prob_lb=-10000, prob_ub=10000)
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # obtain the dimension of the problem
    J_len = length(x_value);  # number of first-stage decision variables

    # create a next pi problem
    next_pi_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0));
    # set up the dual variables pi and auxiliary variables theta
    @variable(next_pi_prob, prob_lb <= pi_var[i in 1:J_len] <= prob_ub);
    @variable(next_pi_prob, 0.0 <= pi_abs[i in 1:J_len] <= max(abs(prob_ub), abs(prob_lb)));
    @variable(next_pi_prob, theta >= prob_lb);
    @variable(next_pi_prob, 0.0 <= pi_obj_abs[i in 1:J_len] <= max(abs(prob_ub), abs(prob_lb)));

    # set up the structural constraints
    @constraint(next_pi_prob, [j in 1:length(cutList)], sum(cutList[j][1][i] * pi_var[i] for i in 1:J_len) + cutList[j][2] >= theta);
    if norm_option == 0
        # L2 norm
        @constraint(next_pi_prob, level_cons, 
            alpha * sum(pi_var[i] * pi_var[i] for i in 1:J_len) + 
            (1 - alpha) * (L_value - sum(pi_var[i] * x_value[i] for i in 1:J_len) - theta) <= level);
    else
        # L1 norm
        @constraint(next_pi_prob, level_cons, 
            alpha * sum(pi_abs[i] for i in 1:J_len) + 
            (1 - alpha) * (L_value - sum(pi_var[i] * x_value[i] for i in 1:J_len) - theta) <= level);
        @constraint(next_pi_prob, pi_abs_pos[i in 1:J_len], pi_var[i] <= pi_abs[i]);
        @constraint(next_pi_prob, pi_abs_neg[i in 1:J_len], -pi_var[i] <= pi_abs[i]);
    end

    # set up the objective function absolute value term
    @constraint(next_pi_prob, pi_obj_pos[i in 1:J_len], pi_obj_abs[i] - pi_var[i] >= 0);
    @constraint(next_pi_prob, pi_obj_neg[i in 1:J_len], pi_obj_abs[i] + pi_var[i] >= 0);

    # set up the objective function
    @objective(next_pi_prob, Min, sum(pi_obj_abs[i] for i in 1:J_len));

    return next_pi_prob;
end

function update_next_pi_problem(next_pi_prob, J_len, cutList, update_ind_range, alpha, x_value, L_value, pi_bar_value, level, norm_option)
    # input: 
    # next_pi_prob - the next pi problem
    # cutList - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    if norm_option == 0
        delete(next_pi_prob, next_pi_prob[:level_cons]);
        unregister(next_pi_prob, :level_cons);
        @constraint(next_pi_prob, level_cons, 
                        alpha * sum(next_pi_prob[:pi_var][i] * next_pi_prob[:pi_var][i] for i in 1:J_len) + 
                        (1 - alpha) * (L_value - sum(next_pi_prob[:pi_var][i] * x_value[i] for i in 1:J_len) - 
                        next_pi_prob[:theta]) <= level);
    else
        delete(next_pi_prob, next_pi_prob[:level_cons]);
        unregister(next_pi_prob, :level_cons);
        @constraint(next_pi_prob, level_cons, 
                        alpha * sum(next_pi_prob[:pi_abs][i] for i in 1:J_len) + 
                        (1 - alpha) * (L_value - sum(next_pi_prob[:pi_var][i] * x_value[i] for i in 1:J_len) - 
                        next_pi_prob[:theta]) <= level);
    end

    # update the cuts
    for j in update_ind_range
        @constraint(next_pi_prob, sum(cutList[j][1][i] * next_pi_prob[:pi_var][i] for i in 1:J_len) + cutList[j][2] >= next_pi_prob[:theta]);
    end

    # set up the objective function absolute value rhs term
    for i in 1:J_len
        set_normalized_rhs(next_pi_prob[:pi_obj_pos][i], -pi_bar_value[i]);
        set_normalized_rhs(next_pi_prob[:pi_obj_neg][i], pi_bar_value[i]);
    end

    return next_pi_prob;
end

function obtain_alpha_bounds(pi_list, L_value, x_value, v_underbar, V_list, norm_option)
    # algebraic way to calculate alpha_max and alpha_min
    alpha_underbar = [];
    alpha_bar = [];
    gamma_list = Dict();
    eta_list = Dict();

    for k in eachindex(pi_list)
        if norm_option == 0
            # L2 norm
            gamma_list[k] = (pi_list[k]'pi_list[k] - v_underbar) - (L_value - pi_list[k]'x_value - V_list[k]);
            eta_list[k] = (L_value - pi_list[k]'x_value - V_list[k]);
        else
            # L1 norm
            gamma_list[k] = (sum(abs.(pi_list[k])) - v_underbar) - (L_value - pi_list[k]'x_value - V_list[k]);
            eta_list[k] = (L_value - pi_list[k]'x_value - V_list[k]);
            if abs(eta_list[k]) <= 1e-7
                eta_list[k] = 0.0
            end
        end
        if gamma_list[k] >= 0.0
            push!(alpha_bar, 1.0)
            if eta_list[k] >= 0.0
                push!(alpha_underbar, 0.0)
            else
                if -eta_list[k] / gamma_list[k] <= 1 + 1e-5
                    push!(alpha_underbar, -eta_list[k] / gamma_list[k])
                else
                    throw("alpha_underbar is not feasible")
                end
            end
        else
            push!(alpha_underbar, 0.0);
            if eta_list[k] >= 0.0
                if -eta_list[k] / gamma_list[k] < 1 + 1e-5
                    push!(alpha_bar, -eta_list[k] / gamma_list[k]);
                else
                    push!(alpha_bar, 1.0);
                end
            else
                throw("alpha_bar is not feasible");
            end
        end
    end
    alpha_min = round(maximum(alpha_underbar), digits=7);
    alpha_max = round(minimum(alpha_bar), digits=7);

    # algebraic way to calculate Delta
    alpha_star = 0.0;
    Delta = minimum([eta_list[k] for k in eachindex(pi_list)]);
    k_mark = Dict();
    for k in eachindex(pi_list)
        k_mark[k] = 0;
    end
    k_star = argmin([eta_list[k] for k in eachindex(pi_list)]);
    k_mark[k_star] = 1;

    if gamma_list[k_star] >= 0
        cont_bool = true;
    else
        cont_bool = false;
    end
    while cont_bool
        alpha_candidate = [];
        k_candidate = [];
        for k in eachindex(pi_list)
            if k_mark[k] == 0
                alpha_k = (eta_list[k] - eta_list[k_star]) / (gamma_list[k_star] - gamma_list[k]);
                if (alpha_k > alpha_star)&(alpha_k <= 1)&(alpha_k >= 0)
                    push!(alpha_candidate, alpha_k);
                    push!(k_candidate, k);
                end
            end
        end
        if length(alpha_candidate) > 0
            alpha_star = minimum(alpha_candidate);
            k_star_new = k_candidate[argmin(alpha_candidate)];
            if (gamma_list[k_star_new] < 0)&(gamma_list[k_star] >= 0)
                cont_bool = false;
            else
                k_star = k_star_new;
                k_mark[k_star] = 1;
            end
        else
            alpha_star = 1;
            cont_bool = false;
        end
    end

    Delta = minimum([gamma_list[k] * alpha_star + eta_list[k] for k in eachindex(pi_list)]);

    return alpha_max, alpha_min, Delta;
end

# procedure to solve the Lagrangian dual problem
function solve_lag_dual(o, ho, T, W, c, y_option, x_value, L_value, lambda_level, mu_level, norm_option, tol = 1e-2, cutList = [], sub_lb = -10000, sub_ub = 100000, init_pt = 1, iter_limit = 200)
    # input: 
    # o - index of the scenario, ho - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # x_value - the optimal solution of the master problem
    # cutList - the list of cuts generated for the inner minimization problem so far, 
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # obtain the dimension of the problem
    J_len = length(W);  # number of first-stage decision variables
    I_len = length(W[1]);  # number of second-stage decision variables

    # initialize the cut list for the level set problem
    v_value = 0;
    alpha_min = 0;
    alpha_max = 1;
    alpha = (alpha_max + alpha_min) / 2;
    pi_list = [];
    V_list = [];

    # set up the lower bound problem
    lb_prob = build_ls_lb_problem(x_value, L_value, cutList, norm_option);
    if init_pt == 0
        # initialize the Lagrangian dual multipliers with zeros
        pi_value = zeros(J_len);
    else
        # initialize the Lagrangian dual multipliers with the lb solution
        optimize!(lb_prob);
        if termination_status(lb_prob) != OPTIMAL
            # update the L_value and resolve lb_prob
            L_test_bool = true;
            L_value_lb = sub_lb;
            L_value_ub = L_value;
            while L_test_bool
                L_value = (L_value_lb + L_value_ub) / 2;
                delete(lb_prob, lb_prob[:cons]);
                unregister(lb_prob, :cons);
                @constraint(lb_prob, cons, sum(lb_prob[:pi_var][i] * x_value[i] for i in 1:J_len) + lb_prob[:theta] >= L_value);
                optimize!(lb_prob);
                if termination_status(lb_prob) == OPTIMAL
                    L_value_lb = L_value;
                    if abs(L_value_ub - L_value_lb) < 1e-2
                        L_test_bool = false;
                    end
                else
                    L_value_ub = L_value;
                end
            end
        end
        pi_value = value.(lb_prob[:pi_var]);
    end
    # set up the auxiliary problem to find the next pi_value
    level = sub_ub;
    next_pi_prob = build_next_pi_problem(level, alpha, x_value, L_value, cutList, norm_option);

    # build the inner min subproblem with Lagrangian penalty term
    sub_prob = build_subproblem_lag(o, ho, T, W, c, y_option, pi_value);
    # solve the subproblem
    optimize!(sub_prob);

    # initialize the termination criterion
    cont_bool = true;
    counter = 0;

    # loop until the termination criterion
    while cont_bool
        # record the pi_value
        push!(pi_list, pi_value);

        # obtain the inner minimization problem's optimal solution
        z_value = value.(sub_prob[:z]);

        # add the cut to the cut list
        Vj = objective_value(sub_prob);        # V(\pi_j)
        push!(V_list, Vj);
        if length(cutList) > 0
            theta_j = minimum([cutList[j][1]'pi_value + cutList[j][2] for j in eachindex(cutList)]);
        else
            theta_j = Inf;
        end
        # only generate the cut if Vj > theta_j
        if Vj < theta_j - 1e-4
            push!(cutList, (-z_value, Vj + z_value'pi_value))
            cutList_update_start_ind = length(cutList);
            cutList_update_end_ind = length(cutList);
        else
            cutList_update_start_ind = length(cutList) + 1;
            cutList_update_end_ind = length(cutList);
        end

        # update and solve the lower bound problem
        lb_prob = update_ls_lb_problem(lb_prob, J_len, cutList, cutList_update_start_ind:cutList_update_end_ind);
        optimize!(lb_prob);
        if termination_status(lb_prob) != OPTIMAL
            # update the L_value and resolve lb_prob
            L_test_bool = true;
            L_value_lb = sub_lb;
            L_value_ub = L_value;
            while L_test_bool
                L_value = (L_value_lb + L_value_ub) / 2
                delete(lb_prob, lb_prob[:cons]);
                unregister(lb_prob, :cons);
                @constraint(lb_prob, cons, sum(lb_prob[:pi_var][i] * x_value[i] for i in 1:J_len) + lb_prob[:theta] >= L_value);
                optimize!(lb_prob);
                if termination_status(lb_prob) == OPTIMAL
                    L_value_lb = L_value;
                    if abs(L_value_ub - L_value_lb) < 1e-2
                        L_test_bool = false;
                    end
                else
                    L_value_ub = L_value;
                end
            end
        end

        # update the alpha
        alpha_max, alpha_min, Delta = obtain_alpha_bounds(pi_list, L_value, x_value, objective_value(lb_prob), V_list, norm_option);
        if counter == 0
            alpha = (alpha_max + alpha_min) / 2;
        else
            if ((alpha - alpha_min)/(alpha_max - alpha_min) < mu_level/2)|((alpha - alpha_min)/(alpha_max - alpha_min) > 1 - mu_level/2)
                alpha = (alpha_min + alpha_max) / 2;
            end
        end

        # update the termination indicator
        if Delta < tol
            cont_bool = false;
        else
            if counter > iter_limit
                cont_bool = false;
            else
                counter += 1
                # update the level
                if norm_option == 0
                    v_bar_list = [alpha * pi_list[pi_j_ind]'pi_list[pi_j_ind] + (1 - alpha) * (L_value - pi_list[pi_j_ind]'x_value - V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                else
                    v_bar_list = [alpha * sum(abs.(pi_list[pi_j_ind])) + (1 - alpha) * (L_value - pi_list[pi_j_ind]'x_value - V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                end
                v_bar = minimum(v_bar_list);
                v_underbar = alpha * objective_value(lb_prob);
                level = lambda_level * v_bar + (1 - lambda_level) * v_underbar;

                # solve for the next pi_value
                next_pi_prob = update_next_pi_problem(next_pi_prob, J_len, cutList, cutList_update_start_ind:cutList_update_end_ind, alpha, x_value, L_value, pi_value, level, norm_option);
                optimize!(next_pi_prob);
                # obtain the next pi_value
                if termination_status(next_pi_prob) != OPTIMAL
                    throw("next_pi_prob is not optimal");
                else
                    pi_value = value.(next_pi_prob[:pi_var]);
                end
                # update the subproblem and solve it
                sub_prob = update_subproblem_lag(sub_prob, c, pi_value);
                optimize!(sub_prob);
            end
        end

        # output the current status
        println("Iteration: $(counter), V(pi_j): $(objective_value(sub_prob)), Delta: $(Delta)");
    end

    # obtain the intercept of the Lagrangian cut
    v_value = objective_value(sub_prob);
    return pi_value, v_value, cutList;
end

# procedure to solve the subproblem in parallel
function sub_routine(o, h, T, W, c, y_option, x_value, lambda_level, mu_level, norm_option, tol = 1e-2, cut_Dict = Dict())
    # input: 
    # o - index of the scenario, h - the right-hand side of the structural constraints,
    # T - the coefficient matrix of x variables in sub, W - the coefficient matrix of y variables in sub,
    # c - the objective function coefficients, y_option - the type of the decision variables: binary or integer,
    # x_value - the optimal solution of the master problem
    # cut_Dict - the dictionary to store the Lagrangian cuts for the subproblems' convex envelope

    # obtain the subproblem value & update the upper bound
    sub_prob = build_subproblem(o, h[o,:], T, W, c, y_option, x_value);
    optimize!(sub_prob);
    # obtain the subproblem solution/optimal value and update the upper bound
    L_value = objective_value(sub_prob); 

    # generate the Lagrangian cuts
    pi_value_o, v_value_o, cutList_o = solve_lag_dual(o, h[o,:], T, W, c, y_option, x_value, L_value, lambda_level, mu_level, norm_option, tol, cut_Dict[o]);
    return o, objective_value(sub_prob), pi_value_o, v_value_o, cutList_o;
end