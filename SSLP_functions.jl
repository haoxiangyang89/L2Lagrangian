function build_extensive_form(omega, c, v, u, d, h, q0, q)
    # obtain the dimension of the problem
    J_len = length(c);  # number of servers
    I_len = size(h, 2);  # number of customers

    # construct the extensive formulation
    extensive_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV)));

    # set up the decision variables
    @variable(extensive_prob, x[j in 1:J_len], Bin);
    @variable(extensive_prob, y[o in 1:omega, i in 1:I_len, j in 1:J_len], Bin);
    @variable(extensive_prob, y0[o in 1:omega, j in 1:J_len] >= 0);

    # set up the objective function
    @objective(extensive_prob, Min,
        sum(c[j] * x[j] for j in 1:J_len) +
        1 / omega * sum(
            sum(q0[o, j] * y0[o, j] + sum(q[o, i, j] * y[o, i, j] for i in 1:I_len) for j in 1:J_len)
            for o in 1:omega
        )
    );

    # set up the structural constraints
    @constraint(extensive_prob, server_no_cons, sum(x[j] for j in 1:J_len) <= v);
    @constraint(extensive_prob, capacity_cons[o in 1:omega, j in 1:J_len],
        sum(d[i, j] * y[o, i, j] for i in 1:I_len) - y0[o, j] <= u * x[j]);
    @constraint(extensive_prob, demand_cons[o in 1:omega, i in 1:I_len],
        sum(y[o, i, j] for j in 1:J_len) == h[o, i]);

    return extensive_prob;
end

function build_masterproblem(omega, c, v, prob_lb=-100000)
    # obtain the dimension of the problem
    J_len = length(c);  # number of servers

    # construct the master program
    master_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Threads" => 1));
    @variable(master_prob, x[j in 1:J_len], Bin);
    @variable(master_prob, theta[o in 1:omega] >= prob_lb);

    # set up the objective function
    @objective(master_prob, Min,
        sum(c[j] * x[j] for j in 1:J_len) +
        1 / omega * sum(theta[o] for o in 1:omega));

    # set up the structural constraints
    @constraint(master_prob, server_no_cons, sum(x[j] for j in 1:J_len) <= v);

    return master_prob;
end

function build_subproblem(o, c, u, d, ho, q0o, qo, x_value)
    # input:
    # o - index of the scenario, d - customer demand, ho - customer availability,
    # q0o - unit penalty cost, qo - unit sales revenue, u - capacity of each server,
    # x_value - the optimal solution of the master problem

    # obtain the dimension of the problem
    J_len = length(c);  # number of servers
    I_len = length(ho);  # number of customers

    # construct the subproblem
    sub_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Threads" => 1));
    @variable(sub_prob, y[i in 1:I_len, j in 1:J_len], Bin);
    @variable(sub_prob, y0[j in 1:J_len] >= 0);

    # set up the objective function
    @objective(sub_prob, Min,
        sum(q0o[j] * y0[j] + sum(qo[i, j] * y[i, j] for i in 1:I_len) for j in 1:J_len));

    # set up the structural constraints
    @constraint(sub_prob, capacity_cons[j in 1:J_len],
        sum(d[i, j] * y[i, j] for i in 1:I_len) - y0[j] <= u * x_value[j]);
    @constraint(sub_prob, demand_cons[i in 1:I_len],
        sum(y[i, j] for j in 1:J_len) == ho[i]);

    return sub_prob;
end

function build_subproblem_lag(o, c, u, v, d, ho, q0o, qo, pi_value)
    # input:
    # o - index of the scenario, d - customer demand, ho - customer availability,
    # q0o - unit penalty cost, qo - unit sales revenue, u - capacity of each server,
    # pi_value - the Lagrangian dual multipliers

    # obtain the dimension of the problem
    J_len = length(c);  # number of servers
    I_len = length(ho);  # number of customers

    # construct the Lagrangian subproblem
    sub_prob_lag = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Threads" => 1));

    # set up the auxiliary variables z (copy of x)
    @variable(sub_prob_lag, z[j in 1:J_len], Bin);
    @constraint(sub_prob_lag, server_no_cons, sum(z[j] for j in 1:J_len) <= v);

    # set up the second-stage decision variables
    @variable(sub_prob_lag, y[i in 1:I_len, j in 1:J_len], Bin);
    @variable(sub_prob_lag, y0[j in 1:J_len] >= 0);

    # set up the objective function with Lagrangian penalty term
    @objective(sub_prob_lag, Min,
        sum(q0o[j] * y0[j] + sum(qo[i, j] * y[i, j] for i in 1:I_len) for j in 1:J_len) -
        sum(pi_value[j] * z[j] for j in 1:J_len));

    # set up the structural constraints
    @constraint(sub_prob_lag, capacity_cons[j in 1:J_len],
        sum(d[i, j] * y[i, j] for i in 1:I_len) - y0[j] <= u * z[j]);
    @constraint(sub_prob_lag, demand_cons[i in 1:I_len],
        sum(y[i, j] for j in 1:J_len) == ho[i]);

    return sub_prob_lag;
end

function update_subproblem_lag(sub_prob_lag, q0o, qo, pi_value)
    # obtain the dimension of the problem
    I_len, J_len = size(qo);

    # update the objective function with the new Lagrangian multipliers
    @objective(sub_prob_lag, Min,
        sum(q0o[j] * sub_prob_lag[:y0][j] +
            sum(qo[i, j] * sub_prob_lag[:y][i, j] for i in 1:I_len) for j in 1:J_len) -
        sum(pi_value[j] * sub_prob_lag[:z][j] for j in 1:J_len));

    return sub_prob_lag;
end

function build_levelset_model(levelset_solver=:Gurobi)
    if levelset_solver == :Gurobi
        return Model(optimizer_with_attributes(
            () -> Gurobi.Optimizer(GUROBI_ENV),
            "OutputFlag" => 0,
            "Threads" => 1,
            "NumericFocus" => 2,
            "FeasibilityTol" => 1e-8
        ));
    elseif levelset_solver == :COPT
        if !isdefined(@__MODULE__, :COPT)
            throw(ArgumentError("COPT is not loaded. Run `using COPT` on every worker before selecting :COPT."));
        end
        copt_module = getfield(@__MODULE__, :COPT);
        return Model(optimizer_with_attributes(
            getfield(copt_module, :Optimizer),
            "TuneOutputLevel" => 0,
            "Threads" => 1,
            "LogToConsole" => 0
        ));
    end
    throw(ArgumentError("levelset_solver must be :Gurobi or :COPT"));
end

# Build the level set lower bound problem
function build_ls_lb_problem(x_value, L_value, cutList, norm_option, prob_lb=-100000, prob_ub=100000;
    levelset_solver=:Gurobi)
    # input:
    # x_value - the optimal solution of the master problem,
    # L_value - the optimal value of the subproblem evaluated at x_value,
    # cutList - cuts for the inner minimization problem

    J_len = length(x_value);
    lb_prob = build_levelset_model(levelset_solver);

    @variable(lb_prob, prob_lb <= pi_var[j in 1:J_len] <= prob_ub);
    @variable(lb_prob, theta >= prob_lb);

    if norm_option == 0
        # L2 norm
        @objective(lb_prob, Min, sum(pi_var[j] * pi_var[j] for j in 1:J_len));
    else
        # L1 norm
        @variable(lb_prob, 0.0 <= pi_abs[j in 1:J_len] <= max(abs(prob_ub), abs(prob_lb)));
        @objective(lb_prob, Min, sum(pi_abs[j] for j in 1:J_len));
        @constraint(lb_prob, pi_abs_pos[j in 1:J_len], pi_var[j] <= pi_abs[j]);
        @constraint(lb_prob, pi_abs_neg[j in 1:J_len], -pi_var[j] <= pi_abs[j]);
    end

    @constraint(lb_prob, [k in 1:length(cutList)],
        sum(cutList[k][1][j] * pi_var[j] for j in 1:J_len) + cutList[k][2] >= theta);
    @constraint(lb_prob, cons,
        sum(pi_var[j] * x_value[j] for j in 1:J_len) + theta >= L_value);

    return lb_prob;
end

# Update the level set lower bound problem
function update_ls_lb_problem(lb_prob, J_len, cutList, update_ind_range)
    for k in update_ind_range
        @constraint(lb_prob,
            sum(cutList[k][1][j] * lb_prob[:pi_var][j] for j in 1:J_len) + cutList[k][2] >= lb_prob[:theta]);
    end
    return lb_prob;
end

# Build the auxiliary problem used to find the next pi value
function build_next_pi_problem(level, alpha, x_value, L_value, cutList, norm_option, prob_lb=-100000, prob_ub=100000;
    levelset_solver=:Gurobi)
    J_len = length(x_value);
    next_pi_prob = build_levelset_model(levelset_solver);

    @variable(next_pi_prob, prob_lb <= pi_var[j in 1:J_len] <= prob_ub);
    @variable(next_pi_prob, theta >= prob_lb);

    @constraint(next_pi_prob, [k in 1:length(cutList)],
        sum(cutList[k][1][j] * pi_var[j] for j in 1:J_len) + cutList[k][2] >= theta);
    if norm_option == 0
        # Keep the level constraint linear by modeling the squared L2 norm
        # with supporting hyperplanes added in update_next_pi_problem.
        @variable(next_pi_prob, norm_epigraph);
        @constraint(next_pi_prob, level_cons,
            alpha * norm_epigraph +
            (1 - alpha) * (L_value - sum(pi_var[j] * x_value[j] for j in 1:J_len) - theta) <= level);
        @objective(next_pi_prob, Min, sum(pi_var[j] * pi_var[j] for j in 1:J_len));
    else
        # L1 norm
        @variable(next_pi_prob, 0.0 <= pi_abs[j in 1:J_len] <= max(abs(prob_ub), abs(prob_lb)));
        @variable(next_pi_prob, 0.0 <= pi_obj_abs[j in 1:J_len] <= max(abs(prob_ub), abs(prob_lb)));
        @constraint(next_pi_prob, level_cons,
            alpha * sum(pi_abs[j] for j in 1:J_len) +
            (1 - alpha) * (L_value - sum(pi_var[j] * x_value[j] for j in 1:J_len) - theta) <= level);
        @constraint(next_pi_prob, pi_abs_pos[j in 1:J_len], pi_var[j] <= pi_abs[j]);
        @constraint(next_pi_prob, pi_abs_neg[j in 1:J_len], -pi_var[j] <= pi_abs[j]);
        @constraint(next_pi_prob, pi_obj_pos[j in 1:J_len], pi_obj_abs[j] - pi_var[j] >= 0);
        @constraint(next_pi_prob, pi_obj_neg[j in 1:J_len], pi_obj_abs[j] + pi_var[j] >= 0);
        @objective(next_pi_prob, Min, sum(pi_obj_abs[j] for j in 1:J_len));
    end

    return next_pi_prob;
end

function update_next_pi_problem(next_pi_prob, J_len, cutList, update_ind_range, alpha, x_value, L_value, pi_bar_value, level, norm_option)
    # update the level constraint
    delete(next_pi_prob, next_pi_prob[:level_cons]);
    unregister(next_pi_prob, :level_cons);
    if norm_option == 0
        # Tangent to ||pi||_2^2 at pi_bar_value:
        # z_norm >= ||pi_bar||^2 + 2*pi_bar' * (pi - pi_bar).
        @constraint(next_pi_prob,
            next_pi_prob[:norm_epigraph] >=
            2 * sum(pi_bar_value[j] * next_pi_prob[:pi_var][j] for j in 1:J_len) -
            sum(pi_bar_value[j] * pi_bar_value[j] for j in 1:J_len));
        @constraint(next_pi_prob, level_cons,
            alpha * next_pi_prob[:norm_epigraph] +
            (1 - alpha) * (L_value - sum(next_pi_prob[:pi_var][j] * x_value[j] for j in 1:J_len) -
            next_pi_prob[:theta]) <= level);
        @objective(next_pi_prob, Min,
            sum((next_pi_prob[:pi_var][j] - pi_bar_value[j])^2 for j in 1:J_len));
    else
        @constraint(next_pi_prob, level_cons,
            alpha * sum(next_pi_prob[:pi_abs][j] for j in 1:J_len) +
            (1 - alpha) * (L_value - sum(next_pi_prob[:pi_var][j] * x_value[j] for j in 1:J_len) -
            next_pi_prob[:theta]) <= level);
    end

    # add new cuts
    for k in update_ind_range
        @constraint(next_pi_prob,
            sum(cutList[k][1][j] * next_pi_prob[:pi_var][j] for j in 1:J_len) +
            cutList[k][2] >= next_pi_prob[:theta]);
    end

    # center the L1 objective at pi_bar_value
    if norm_option != 0
        for j in 1:J_len
            set_normalized_rhs(next_pi_prob[:pi_obj_pos][j], -pi_bar_value[j]);
            set_normalized_rhs(next_pi_prob[:pi_obj_neg][j], pi_bar_value[j]);
        end
    end

    return next_pi_prob;
end

function has_acceptable_solution(model)
    status = termination_status(model);
    return (status == OPTIMAL || status == LOCALLY_SOLVED) && has_values(model);
end

# Optimization-based calculation of Delta and the feasible alpha interval.
function obtain_alpha_bounds_opt(pi_list, L_value, x_value, v_underbar, V_list, norm_option;
    levelset_solver=:Gurobi)

    alpha_prob = build_levelset_model(levelset_solver);
    @variable(alpha_prob, 0 <= alpha <= 1);
    @variable(alpha_prob, alpha_obj);

    if norm_option == 0
        @constraint(alpha_prob, alpha_obj_constr[k in eachindex(pi_list)],
            alpha_obj <= alpha * (pi_list[k]'pi_list[k] - v_underbar) +
            (1 - alpha) * (L_value - pi_list[k]'x_value - V_list[k]));
    else
        @constraint(alpha_prob, alpha_obj_constr[k in eachindex(pi_list)],
            alpha_obj <= alpha * (sum(abs.(pi_list[k])) - v_underbar) +
            (1 - alpha) * (L_value - pi_list[k]'x_value - V_list[k]));
    end

    # First maximize the lower envelope to obtain Delta.
    @objective(alpha_prob, Max, alpha_obj);
    optimize!(alpha_prob);
    if !has_acceptable_solution(alpha_prob)
        return 1.0, 0.0, Inf, false;
    end
    Delta = objective_value(alpha_prob);

    # Alpha is admissible where the lower envelope is nonnegative.
    @constraint(alpha_prob, alpha_obj_nonnegative, alpha_obj >= 0);
    @objective(alpha_prob, Min, alpha);
    optimize!(alpha_prob);
    if !has_acceptable_solution(alpha_prob)
        return 1.0, 0.0, Delta, false;
    end
    alpha_min = value(alpha);

    @objective(alpha_prob, Max, alpha);
    optimize!(alpha_prob);
    if !has_acceptable_solution(alpha_prob)
        return 1.0, 0.0, Delta, false;
    end
    alpha_max = value(alpha);

    return clamp(alpha_max, 0.0, 1.0), clamp(alpha_min, 0.0, 1.0), Delta, true;
end

function maximize_lower_envelope(gamma, eta)
    n = length(gamma);
    @assert n == length(eta) "gamma and eta must have the same length";

    if n == 1
        if gamma[1] > 0
            return 1.0, gamma[1] + eta[1];
        else
            return 0.0, eta[1];
        end
    end

    candidates = Float64[0.0, 1.0];
    for j in 1:n, k in j+1:n
        if gamma[j] != gamma[k]
            alpha_intersection = (eta[k] - eta[j]) / (gamma[j] - gamma[k]);
            if 0.0 <= alpha_intersection <= 1.0
                push!(candidates, alpha_intersection);
            end
        end
    end

    best_alpha = 0.0;
    best_value = -Inf;
    for alpha in unique(candidates)
        value = minimum(gamma .* alpha .+ eta);
        if value > best_value
            best_value = value;
            best_alpha = alpha;
        end
    end

    return best_alpha, best_value;
end

function obtain_alpha_bounds(pi_list, L_value, x_value, v_underbar, V_list, norm_option)
    # algebraic way to calculate alpha_max and alpha_min
    alpha_underbar = [];
    alpha_bar = [];
    gamma_list = Dict();
    eta_list = Dict();
    alpha_valid = true;

    for k in eachindex(pi_list)
        eta_list[k] = L_value - pi_list[k]'x_value - V_list[k];
        if norm_option == 0
            gamma_list[k] = (pi_list[k]'pi_list[k] - v_underbar) - eta_list[k];
        else
            gamma_list[k] = (sum(abs.(pi_list[k])) - v_underbar) - eta_list[k];
            if abs(eta_list[k]) <= 1e-7
                eta_list[k] = 0.0;
            end
        end

        if gamma_list[k] >= 0.0
            push!(alpha_bar, 1.0);
            if eta_list[k] >= 0.0
                push!(alpha_underbar, 0.0);
            elseif -eta_list[k] / gamma_list[k] <= 1 + 1e-5
                push!(alpha_underbar, -eta_list[k] / gamma_list[k]);
            else
                #throw(ArgumentError("alpha_underbar is not feasible"));
                alpha_valid = false;
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
                #throw(ArgumentError("alpha_bar is not feasible"));
                alpha_valid = false;
            end
        end
    end

    alpha_min = round(maximum(alpha_underbar), digits=7);
    alpha_max = round(minimum(alpha_bar), digits=7);

    # calculate Delta from the maximum of the lower envelope
    gamma_values = [gamma_list[k] for k in eachindex(pi_list)];
    eta_values = [eta_list[k] for k in eachindex(pi_list)];
    _, Delta = maximize_lower_envelope(gamma_values, eta_values);

    return alpha_max, alpha_min, Delta, alpha_valid;
end

# Procedure to solve the Lagrangian dual problem
function solve_lag_dual(o, c, u, v, d, ho, q0o, qo, x_value, L_value, lambda_level, mu_level, norm_option,
    tol=1e-2, cutList=[], sub_lb=-10000, sub_ub=100000, iter_limit=200, levelset_solver=:Gurobi)

    # obtain the dimension of the problem
    J_len = length(c);

    # initialize the Lagrangian dual multipliers and level-set data
    if isempty(cutList)
        pi_value = zeros(J_len);
    else
        initial_lb_prob = build_ls_lb_problem(
            x_value, L_value, cutList, norm_option;
            levelset_solver=levelset_solver);
        optimize!(initial_lb_prob);
        if has_acceptable_solution(initial_lb_prob)
            pi_value = value.(initial_lb_prob[:pi_var]);
        else
            @warn "Unable to warm-start pi from the lower-bound model; using zeros." scenario=o status=termination_status(initial_lb_prob);
            pi_value = zeros(J_len);
        end
    end
    v_value = 0;
    alpha_min = 0;
    alpha_max = 1;
    alpha = (alpha_max + alpha_min) / 2;
    pi_list = [];
    V_list = [];

    # set up the lower-bound and next-pi problems
    lb_prob = build_ls_lb_problem(
        x_value, L_value, cutList, norm_option;
        levelset_solver=levelset_solver);
    level = sub_ub;
    next_pi_prob = build_next_pi_problem(
        level, alpha, x_value, L_value, cutList, norm_option;
        levelset_solver=levelset_solver);

    # build and solve the inner minimization problem
    sub_prob = build_subproblem_lag(o, c, u, v, d, ho, q0o, qo, pi_value);
    optimize!(sub_prob);
    if !has_acceptable_solution(sub_prob)
        throw(ErrorException("Lagrangian subproblem failed for scenario $(o): $(termination_status(sub_prob))"));
    end
    last_valid_pi = copy(pi_value);
    last_valid_v = objective_value(sub_prob);

    cont_bool = true;
    counter = 0;

    while cont_bool
        push!(pi_list, copy(pi_value));

        # obtain the inner minimization problem's optimal solution
        z_value = value.(sub_prob[:z]);

        # add a cut to the inner approximation when violated
        Vj = objective_value(sub_prob);
        push!(V_list, Vj);
        if length(cutList) > 0
            theta_j = minimum(cutList[k][1]'pi_value + cutList[k][2] for k in eachindex(cutList));
        else
            theta_j = Inf;
        end

        if Vj < theta_j - 1e-4
            push!(cutList, (-z_value, Vj + z_value'pi_value));
            cutList_update_start_ind = length(cutList);
            cutList_update_end_ind = length(cutList);
        else
            cutList_update_start_ind = length(cutList) + 1;
            cutList_update_end_ind = length(cutList);
        end

        # update and solve the lower-bound problem
        lb_prob = update_ls_lb_problem(lb_prob, J_len, cutList, cutList_update_start_ind:cutList_update_end_ind);
        optimize!(lb_prob);
        if !has_acceptable_solution(lb_prob)
            lb_status = termination_status(lb_prob);
            if lb_status != INFEASIBLE && lb_status != INFEASIBLE_OR_UNBOUNDED
                @warn "Lower-bound model stopped without a usable solution." scenario=o status=lb_status;
                cont_bool = false;
                break;
            end

            # The target L_value can be structurally too high for the current
            # cutting-plane model. Only in that case do we lower it by bisection.
            L_value_lb = sub_lb;
            L_value_ub = L_value;
            recovery_failed = false;
            while abs(L_value_ub - L_value_lb) >= 1e-2
                L_value_test = (L_value_lb + L_value_ub) / 2;
                delete(lb_prob, lb_prob[:cons]);
                unregister(lb_prob, :cons);
                @constraint(lb_prob, cons,
                    sum(lb_prob[:pi_var][j] * x_value[j] for j in 1:J_len) + lb_prob[:theta] >= L_value_test);
                optimize!(lb_prob);
                if has_acceptable_solution(lb_prob)
                    L_value_lb = L_value_test;
                elseif termination_status(lb_prob) == INFEASIBLE ||
                    termination_status(lb_prob) == INFEASIBLE_OR_UNBOUNDED
                    L_value_ub = L_value_test;
                else
                    @warn "Numerical failure while recovering the lower-bound model." scenario=o status=termination_status(lb_prob);
                    recovery_failed = true;
                    break;
                end
            end
            if recovery_failed
                cont_bool = false;
                break;
            end

            L_value = L_value_lb;
            delete(lb_prob, lb_prob[:cons]);
            unregister(lb_prob, :cons);
            @constraint(lb_prob, cons,
                sum(lb_prob[:pi_var][j] * x_value[j] for j in 1:J_len) + lb_prob[:theta] >= L_value);
            optimize!(lb_prob);
            if !has_acceptable_solution(lb_prob)
                @warn "Recovered lower-bound target still has no usable solution." scenario=o status=termination_status(lb_prob);
                cont_bool = false;
                break;
            end
        end

        # update alpha
        lb_objective = objective_value(lb_prob);
        if norm_option == 0
            alpha_max, alpha_min, Delta, alpha_valid = obtain_alpha_bounds_opt(
                pi_list, L_value, x_value, lb_objective, V_list, norm_option;
                levelset_solver=levelset_solver);
        else
            alpha_max, alpha_min, Delta, alpha_valid = obtain_alpha_bounds(
                pi_list, L_value, x_value, lb_objective, V_list, norm_option);
        end

        if !alpha_valid || isapprox(alpha_max, alpha_min; atol=1e-8, rtol=1e-8)
            alpha = (alpha_max + alpha_min) / 2;
        elseif counter == 0
            alpha = (alpha_max + alpha_min) / 2;
        elseif alpha_max > alpha_min
            alpha_position = (alpha - alpha_min) / (alpha_max - alpha_min);
            if (alpha_position < mu_level / 2) || (alpha_position > 1 - mu_level / 2)
                alpha = (alpha_min + alpha_max) / 2;
            end
        end

        # update the termination indicator
        if Delta < tol
            cont_bool = false;
        elseif counter >= iter_limit
            cont_bool = false;
        else
            counter += 1;

            # update the level
            if norm_option == 0
                v_bar_list = [
                    alpha * pi_list[k]'pi_list[k] +
                    (1 - alpha) * (L_value - pi_list[k]'x_value - V_list[k])
                    for k in eachindex(pi_list)
                ];
            else
                v_bar_list = [
                    alpha * sum(abs.(pi_list[k])) +
                    (1 - alpha) * (L_value - pi_list[k]'x_value - V_list[k])
                    for k in eachindex(pi_list)
                ];
            end
            v_bar = minimum(v_bar_list);
            v_underbar = alpha * lb_objective;
            level = lambda_level * v_bar + (1 - lambda_level) * v_underbar;

            # solve for the next pi value
            next_pi_prob = update_next_pi_problem(
                next_pi_prob, J_len, cutList,
                cutList_update_start_ind:cutList_update_end_ind,
                alpha, x_value, L_value, pi_value, level, norm_option);
            optimize!(next_pi_prob);

            # The reference implementation retries numerical failures with a
            # looser level close to the upper model value.
            if !has_acceptable_solution(next_pi_prob)
                first_status = termination_status(next_pi_prob);
                safe_level = v_underbar + 0.99 * (v_bar - v_underbar);
                set_normalized_rhs(next_pi_prob[:level_cons], safe_level);
                optimize!(next_pi_prob);
                if !has_acceptable_solution(next_pi_prob)
                    @warn "Next-pi model failed after safe-level retry." scenario=o initial_status=first_status retry_status=termination_status(next_pi_prob);
                    cont_bool = false;
                end
            end

            next_pi_value = nothing;
            if cont_bool
                try
                    next_pi_value = value.(next_pi_prob[:pi_var]);
                catch error
                    @warn "Unable to extract pi from the next-pi model." scenario=o exception=error;
                    cont_bool = false;
                end
            end

            if cont_bool
                pi_value = next_pi_value;
                sub_prob = update_subproblem_lag(sub_prob, q0o, qo, pi_value);
                optimize!(sub_prob);
                if !has_acceptable_solution(sub_prob)
                    @warn "Updated Lagrangian subproblem has no usable solution." scenario=o status=termination_status(sub_prob);
                    pi_value = copy(last_valid_pi);
                    cont_bool = false;
                else
                    last_valid_pi = copy(pi_value);
                    last_valid_v = objective_value(sub_prob);
                end
            end
        end

        current_v = has_acceptable_solution(sub_prob) ? objective_value(sub_prob) : last_valid_v;
        println("Iteration: $(counter), V(pi_j): $(current_v), Delta: $(Delta)");
    end

    return last_valid_pi, last_valid_v, cutList;
end

# Procedure to solve one scenario in parallel
function sub_routine(o, c, u, v, d, h, q0, q, x_value, lambda_level, mu_level, norm_option,
    tol=1e-2, cut_Dict=Dict(), levelset_solver=:Gurobi)

    ho = h[o, :];
    q0o = q0[o, :];
    qo = q[o, :, :];

    # obtain the subproblem value and update the upper bound
    sub_prob = build_subproblem(o, c, u, d, ho, q0o, qo, x_value);
    optimize!(sub_prob);
    if !has_acceptable_solution(sub_prob)
        throw(ErrorException("Subproblem failed for scenario $(o): $(termination_status(sub_prob))"));
    end
    L_value = objective_value(sub_prob);

    # generate the Lagrangian cut
    pi_value_o, v_value_o, cutList_o = solve_lag_dual(
        o, c, u, v, d, ho, q0o, qo, x_value, L_value,
        lambda_level, mu_level, norm_option, tol, cut_Dict[o],
        -10000, 100000, 200, levelset_solver);

    return o, objective_value(sub_prob), pi_value_o, v_value_o, cutList_o;
end
