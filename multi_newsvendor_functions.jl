# decomposition functions for the multi-stage, multi-product capacitated newsvendor problem
# Arslan, Dowson, and Morton (2024)
using Distributions;
using Base.Iterators

function generate_scenario_index(scenTree, previous_scen, t)
    # input:
    # scenTree: the scenario tree object
    # previous_scen: the scenario index at time t-1
    # t: current time period

    # output:
    # scen_ind: the scenario index at time t

    # obtain the scenario index at time t
    p = scenTree[t - 1][previous_scen].probability;
    cat_dist = Categorical(p);
    scen_ind = rand(cat_dist);
    return scen_ind;
end

function build_stage_problem(probData, t, node, spatial_structure_t, cut_lag, x_prev)
    # input:
    # t: current stage
    # node: the current node index
    # spatial_structure_t: the spatial branching structure at time t
    # cut_lag: the Lagrangian cuts for all nodes in the scenario tree
    # x_prev: the incumbent decisions from the previous stage

    # output:
    # stage_problem: the JuMP model for the current stage problem

    stage_problem = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Threads" => 1, "MIPGap" => 1e-6));

    # define decision variables
    @variable(stage_problem, 0 <= x[i in probData.I] <= probData.xub[i]);                        # inventory
    @variable(stage_problem, us[i in probData.I] >= 0);                       # sales
    @variable(stage_problem, ub[i in probData.I] >= 0);                       # purchase
    @variable(stage_problem, z[k in eachindex(probData.K)], Bin);             # marketing
    @variable(stage_problem, theta[ω in node.successor] >= -sum(probData.xub[i] * probData.s[i] * 50 for i in probData.I));              # value function approximation

    # constraints
    @constraint(stage_problem, sales_bound1[i in probData.I], us[i] <= node.d[i]);
    @constraint(stage_problem, sales_bound2[i in probData.I], us[i] <= x_prev["x"][i]);
    @constraint(stage_problem, inventory_balance[i in probData.I], x[i] == x_prev["x"][i] - us[i] + ub[i]);
    @constraint(stage_problem, capacity_constraint, sum(ub[i] for i in probData.I) <= probData.Capacity);
    @constraint(stage_problem, marketing_constraint, sum(z[k] for k in eachindex(probData.K)) == 1);

    # objective function
    probability_changed = Dict();
    for ω in node.successor
        for k in eachindex(probData.K)
            # calculate the probability after the marketing strategy
            probList = [];
            for i in probData.I
                if node.sign[ω][i] == 1
                    if probData.K[k][i] == 1
                        push!(probList, probData.p_up[i]);
                    else
                        push!(probList, probData.p_norm[i]);
                    end
                else
                    if probData.K[k][i] == 1
                        push!(probList, probData.p_down[i]);
                    else
                        push!(probList, probData.p_norm[i]);
                    end
                end
            end
            probability_changed[ω,k] = prod(probList);
        end
    end
    @objective(stage_problem, Min, sum(probData.b[i] * ub[i] + probData.h[i] * x[i] - probData.s[i] * us[i] for i in probData.I) +
                                   sum(probability_changed[ω,k] * z[k] * theta[ω] for k in eachindex(probData.K) for ω in node.successor) + 
                                   sum(z[k] * probData.f[k] for k in eachindex(probData.K)));
    @expression(stage_problem, stage_cost, sum(probData.b[i] * ub[i] + probData.h[i] * x[i] - probData.s[i] * us[i] for i in probData.I) + 
        sum(z[k] * probData.f[k] for k in eachindex(probData.K)));

    # add spatial partitioning structure constraints
    if spatial_structure_t == Dict()
        # no branching
        # add Lagrangian cuts
        for ω in node.successor
            for j in eachindex(cut_lag[t+1][ω])
                @constraint(stage_problem, sum(cut_lag[t+1][ω][j][1][i] * x[i] for i in probData.I) + cut_lag[t+1][ω][j][2] <= theta[ω]);
            end
        end
    else
        # add the branching variables
        @variable(stage_problem, w[i in keys(spatial_structure_t), part_ind in 1:(length(spatial_structure_t[i])+1)], Bin);
        # add the spatial partitioning constraints
        @constraint(stage_problem, w_1[i in keys(spatial_structure_t)], sum(w[i, part_ind] for part_ind in 1:(length(spatial_structure_t[i]) + 1)) == 1);
        part_lb = Dict();
        part_ub = Dict();
        for i in keys(spatial_structure_t)
            for part_ind in 1:(length(spatial_structure_t[i]) + 1)
                if part_ind == 1
                    part_lb[i, part_ind] = probData.xlb[i];
                    part_ub[i, part_ind] = spatial_structure_t[i][1];
                elseif part_ind == (length(spatial_structure_t[i]) + 1)
                    part_lb[i, part_ind] = spatial_structure_t[i][length(spatial_structure_t[i])];
                    part_ub[i, part_ind] = probData.xub[i];
                else
                    part_lb[i, part_ind] = spatial_structure_t[i][part_ind - 1];
                    part_ub[i, part_ind] = spatial_structure_t[i][part_ind];
                end
            end
        end
        @constraint(stage_problem, x_part_ub[i in keys(spatial_structure_t)], x[i] <= sum(part_ub[i, part_ind] * w[i, part_ind] for part_ind in 1:(length(spatial_structure_t[i]) + 1)));
        @constraint(stage_problem, x_part_lb[i in keys(spatial_structure_t)], x[i] >= sum(part_lb[i, part_ind] * w[i, part_ind] for part_ind in 1:(length(spatial_structure_t[i]) + 1)));

        # add Lagrangian cuts
        for ω in node.successor
            for j in eachindex(cut_lag[t+1][ω])
                if spatial_structure_t != Dict()
                    x_value_w_keys = Set([(i,part_ind) for (i, part_ind) in keys(cut_lag[t+1][ω][j][3])]);
                    spatial_t_keys = Set([(i,part_ind) for i in keys(spatial_structure_t) for part_ind in 1:(length(spatial_structure_t[i]) + 1)]);
                    if x_value_w_keys != spatial_t_keys
                        println("ERROR in build_stage_problem: Key mismatch!");
                        println("  cut_lag[\"w\"] keys: $x_value_w_keys");
                        println("  spatial_structure_t keys: $spatial_t_keys");
                    end
                end
                @constraint(stage_problem, sum(cut_lag[t+1][ω][j][1][i] * x[i] for i in probData.I) + 
                    sum(cut_lag[t+1][ω][j][3][i, part_ind] * w[i, part_ind] for i in keys(spatial_structure_t) for part_ind in 1:(length(spatial_structure_t[i]) + 1)) + 
                    cut_lag[t+1][ω][j][2] <= theta[ω]);
            end
        end
    end

    return stage_problem;
end

function update_spatial_structure(spatial_structure_t, x_partition)
    # input:
    # spatial_structure_t: the current spatial branching structure at time t
    # x_partition: the x value used to create a new partition

    # output:
    # updated_spatial_structure_t: the updated spatial branching structure at time t

    # update the spatial branching structure based on x_partition
    spatial_structure_t_new = deepcopy(spatial_structure_t);
    index_partition = Dict();
    for i in keys(x_partition)
        if i in keys(spatial_structure_t)
            push!(spatial_structure_t_new[i], x_partition[i]);
            sort!(spatial_structure_t_new[i]);
            index_partition[i] = indexin(x_partition[i], spatial_structure_t_new[i])[1];
        else
            spatial_structure_t_new[i] = [x_partition[i]];
            index_partition[i] = 1;
        end
    end
    return spatial_structure_t_new, index_partition;
end

function update_lag_cut(node, spatial_structure_t, index_partition_node, cut_lag_node)
    # input:
    # t: current stage
    # node_ind: the current node index
    # spatial_structure_t: the spatial branching structure at time t
    # cut_lag_node: the current Lagrangian cuts at the node

    # output:
    # updated_cut_lag_node: the updated Lagrangian cuts at the node

    # update the Lagrangian cuts based on updated spatial_structure_t
    for j in eachindex(cut_lag_node)
        if length(cut_lag_node[j]) == 2
            # if the cut does not contain w information, pad 0
            pi_w = Dict();
            for i in keys(spatial_structure_t)
                for part_ind in 1:(length(spatial_structure_t[i]) + 1)
                    pi_w[i,part_ind] = 0.0;
                end
            end
            push!(cut_lag_node[j], pi_w);
        else
            # if the cut already contains w information, copy that info
            pi_w = Dict();
            for i in keys(spatial_structure_t)
                if i in keys(index_partition_node)
                    # if we need to expand this i-dimension
                    current_part = 1;
                    for part_ind in 1:(length(spatial_structure_t[i]))
                        if part_ind != index_partition_node[i]
                            pi_w[i, current_part] = cut_lag_node[j][3][i,part_ind];
                            current_part += 1;
                        else
                            # we need to make pi_{ij_1} and pi_{ij_1+1} the same
                            pi_w[i, current_part] = cut_lag_node[j][3][i,part_ind];
                            current_part += 1;
                            pi_w[i, current_part] = cut_lag_node[j][3][i,part_ind];
                            current_part += 1;
                        end
                    end
                else
                    for part_ind in 1:(length(spatial_structure_t[i]) + 1)
                        pi_w[i,part_ind] = cut_lag_node[j][3][i, part_ind];
                    end
                end
            end
            cut_lag_node[j][3] = pi_w;
        end
    end
    return cut_lag_node;
end

function build_ls_lb_problem(probData, x_prev, spatial_structure_prev, L_value, norm_option, cut_Dict_node, prob_lb=-1e5, prob_ub=1e5)
    # input: 
    # x_prev - predecessor's x_value
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # spatial_structure_prev - the predecessor-specific spatial branching structure,
    # cut_Dict_node - the list of cuts generated for the inner minimization problem so far at node
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # create a level set lower bound problem
    lb_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Threads" => 1));

    # set up the dual variables pi and auxiliary variables theta
    @variable(lb_prob, prob_lb <= pi_var_x[i in probData.I] <= prob_ub);
    if spatial_structure_prev != Dict()
        @variable(lb_prob, prob_lb <= pi_var_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i]) + 1)] <= prob_ub);
    end
    @variable(lb_prob, theta >= prob_lb);

    if norm_option == 0
        # L2 norm
        if spatial_structure_prev != Dict()
            @objective(lb_prob, Min, sum(pi_var_x[i] * pi_var_x[i] for i in probData.I));
        else
            @objective(lb_prob, Min, sum(pi_var_x[i] * pi_var_x[i] for i in probData.I) + 
                sum(pi_var_w[i, part_ind] * pi_var_w[i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)));
        end
    else
        # L1 norm
        if spatial_structure_prev != Dict()
            @variable(lb_prob, 0.0 <= pi_abs_x[i in probData.I] <= max(abs(prob_ub), abs(prob_lb)));
            @variable(lb_prob, 0.0 <= pi_abs_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i]) + 1)] <= max(abs(prob_ub), abs(prob_lb)))
            @objective(lb_prob, Min, sum(pi_abs_x[i] for i in probData.I) + 
                sum(pi_abs_w[i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)));
            @constraint(lb_prob, pi_abs_pos_x[i in probData.I], pi_var_x[i] <= pi_abs_x[i]);
            @constraint(lb_prob, pi_abs_neg_x[i in probData.I], -pi_var_x[i] <= pi_abs_x[i]);
            @constraint(lb_prob, pi_abs_pos_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i]) + 1)],
                pi_var_w[i, part_ind] <= pi_abs_w[i, part_ind]);
            @constraint(lb_prob, pi_abs_neg_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i]) + 1)],
                -pi_var_w[i, part_ind] <= pi_abs_w[i, part_ind]);
        else
            @variable(lb_prob, 0.0 <= pi_abs_x[i in probData.I] <= max(abs(prob_ub), abs(prob_lb)));
            @objective(lb_prob, Min, sum(pi_abs_x[i] for i in probData.I));
            @constraint(lb_prob, pi_abs_pos_x[i in probData.I], pi_var_x[i] <= pi_abs_x[i]);
            @constraint(lb_prob, pi_abs_neg_x[i in probData.I], -pi_var_x[i] <= pi_abs_x[i]);
        end
    end

    # set up the structural constraints
    if spatial_structure_prev != Dict()
        @constraint(lb_prob, [j in 1:length(cut_Dict_node)], sum(cut_Dict_node[j][1][i] * pi_var_x[i] for i in probData.I) +
            sum(cut_Dict_node[j][3][i, part_ind] * pi_var_w[i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)) +
            cut_Dict_node[j][2] >= theta);
        @constraint(lb_prob, cons, sum(pi_var_x[i] * x_prev["x"][i] for i in probData.I) + 
            sum(pi_var_w[i, part_ind] * x_prev["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)) + 
            theta >= L_value);
    else
        @constraint(lb_prob, [j in 1:length(cut_Dict_node)], sum(cut_Dict_node[j][1][i] * pi_var_x[i] for i in probData.I) + cut_Dict_node[j][2] >= theta);
        @constraint(lb_prob, cons, sum(pi_var_x[i] * x_prev["x"][i] for i in probData.I) + theta >= L_value);
    end
    return lb_prob;
end

function update_ls_lb_problem(lb_prob, probData, spatial_structure_prev, cut_Dict_node, update_ind_range)
    # input: 
    # lb_prob - the level set lower bound problem
    # cut_Dict_node - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
        # add new cuts to the level set lower bound problem
    for j in update_ind_range
        if spatial_structure_prev != Dict()
            @constraint(lb_prob, sum(cut_Dict_node[j][1][i] * lb_prob[:pi_var_x][i] for i in probData.I) +
                sum(cut_Dict_node[j][3][i, part_ind] * lb_prob[:pi_var_w][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)) +
                cut_Dict_node[j][2] >= lb_prob[:theta]);
        else
            @constraint(lb_prob, sum(cut_Dict_node[j][1][i] * lb_prob[:pi_var_x][i] for i in probData.I) + cut_Dict_node[j][2] >= lb_prob[:theta]);
        end
    end
    return lb_prob;
end

# Build the level set lower bound problem
function build_next_pi_problem(probData, spatial_structure_prev, level, alpha, x_value, L_value, cut_Dict_node, norm_option, prob_lb=-1e5, prob_ub=1e5)
    # input: 
    # x_value - the optimal solution of the master problem, pi_value - the Lagrangian dual multipliers,
    # L_value - the optimal value of Lagrangian function evaluated at x_value,
    # cut_Dict_node - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # create a next pi problem
    next_pi_prob = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Threads" => 1));
    # set up the dual variables pi and auxiliary variables theta
    if spatial_structure_prev != Dict()
        @variable(next_pi_prob, prob_lb <= pi_var_x[i in probData.I] <= prob_ub);
        @variable(next_pi_prob, 0.0 <= pi_abs_x[i in probData.I] <= max(abs(prob_ub), abs(prob_lb)));
        @variable(next_pi_prob, theta >= prob_lb);
        @variable(next_pi_prob, 0.0 <= pi_obj_abs_x[i in probData.I] <= max(abs(prob_ub), abs(prob_lb)));
        @variable(next_pi_prob, prob_lb <= pi_var_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i])+1)] <= prob_ub);
        @variable(next_pi_prob, 0.0 <= pi_abs_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i])+1)] <= max(abs(prob_ub), abs(prob_lb)));
        @variable(next_pi_prob, 0.0 <= pi_obj_abs_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i])+1)] <= max(abs(prob_ub), abs(prob_lb)));

        # set up the structural constraints
        @constraint(next_pi_prob, [j in 1:length(cut_Dict_node)], sum(cut_Dict_node[j][1][i] * pi_var_x[i] for i in probData.I) + 
            sum(cut_Dict_node[j][3][i, part_ind] * pi_var_w[i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1)) + 
            cut_Dict_node[j][2] >= theta);
        if norm_option == 0
            # L2 norm
            @constraint(next_pi_prob, level_cons, 
                alpha * (sum(pi_var_x[i] * pi_var_x[i] for i in probData.I) + 
                    sum(pi_var_w[i, part_ind] * pi_var_w[i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1))) + 
                (1 - alpha) * (L_value - sum(pi_var_x[i] * x_value["x"][i] for i in probData.I) - 
                    sum(pi_var_w[i, part_ind] * x_value["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1)) - theta) <= level);
        else
            # L1 norm
            @constraint(next_pi_prob, level_cons, 
                alpha * (sum(pi_abs_x[i] for i in probData.I) + 
                    sum(pi_abs_w[i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1))) + 
                (1 - alpha) * (L_value - sum(pi_var_x[i] * x_value["x"][i] for i in probData.I) -  
                    sum(pi_var_w[i, part_ind] * x_value["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1)) - 
                    theta) <= level);
            @constraint(next_pi_prob, pi_abs_pos_x[i in probData.I], pi_var_x[i] <= pi_abs_x[i]);
            @constraint(next_pi_prob, pi_abs_neg_x[i in probData.I], -pi_var_x[i] <= pi_abs_x[i]);
            @constraint(next_pi_prob, pi_abs_pos_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i])+1)], pi_var_w[i, part_ind] <= pi_abs_w[i, part_ind]);
            @constraint(next_pi_prob, pi_abs_neg_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i])+1)], -pi_var_w[i, part_ind] <= pi_abs_w[i, part_ind]);
        end

        # set up the objective function absolute value term
        @constraint(next_pi_prob, pi_obj_pos_x[i in probData.I], pi_obj_abs_x[i] - pi_var_x[i] >= 0);
        @constraint(next_pi_prob, pi_obj_neg_x[i in probData.I], pi_obj_abs_x[i] + pi_var_x[i] >= 0);

        # set up the objective function absolute value term
        @constraint(next_pi_prob, pi_obj_pos_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i])+1)], pi_obj_abs_w[i, part_ind] - pi_var_w[i, part_ind] >= 0);
        @constraint(next_pi_prob, pi_obj_neg_w[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i])+1)], pi_obj_abs_w[i, part_ind] + pi_var_w[i, part_ind] >= 0);

        # set up the objective function
        @objective(next_pi_prob, Min, sum(pi_obj_abs_x[i] for i in probData.I) + 
            sum(pi_obj_abs_w[i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1)));
    else
        @variable(next_pi_prob, prob_lb <= pi_var_x[i in probData.I] <= prob_ub);
        @variable(next_pi_prob, 0.0 <= pi_abs_x[i in probData.I] <= max(abs(prob_ub), abs(prob_lb)));
        @variable(next_pi_prob, theta >= prob_lb);
        @variable(next_pi_prob, 0.0 <= pi_obj_abs_x[i in probData.I] <= max(abs(prob_ub), abs(prob_lb)));

        # set up the structural constraints
        @constraint(next_pi_prob, [j in 1:length(cut_Dict_node)], sum(cut_Dict_node[j][1][i] * pi_var_x[i] for i in probData.I) + cut_Dict_node[j][2] >= theta);
        if norm_option == 0
            # L2 norm
            @constraint(next_pi_prob, level_cons, 
                alpha * sum(pi_var_x[i] * pi_var_x[i] for i in probData.I) + 
                (1 - alpha) * (L_value - sum(pi_var_x[i] * x_value["x"][i] for i in probData.I) - theta) <= level);
        else
            # L1 norm
            @constraint(next_pi_prob, level_cons, 
                alpha * sum(pi_abs_x[i] for i in probData.I) + 
                (1 - alpha) * (L_value - sum(pi_var_x[i] * x_value["x"][i] for i in probData.I) - theta) <= level);
            @constraint(next_pi_prob, pi_abs_pos_x[i in probData.I], pi_var_x[i] <= pi_abs_x[i]);
            @constraint(next_pi_prob, pi_abs_neg_x[i in probData.I], -pi_var_x[i] <= pi_abs_x[i]);
        end

        # set up the objective function absolute value term
        @constraint(next_pi_prob, pi_obj_pos_x[i in probData.I], pi_obj_abs_x[i] - pi_var_x[i] >= 0);
        @constraint(next_pi_prob, pi_obj_neg_x[i in probData.I], pi_obj_abs_x[i] + pi_var_x[i] >= 0);

        # set up the objective function
        @objective(next_pi_prob, Min, sum(pi_obj_abs_x[i] for i in probData.I));
    end

    return next_pi_prob;
end

function update_next_pi_problem(next_pi_prob, probData, spatial_structure_prev, cut_Dict_node, update_ind_range, alpha, x_value, L_value, pi_bar_value, level, norm_option)
    # input: 
    # next_pi_prob - the next pi problem
    # cut_Dict_node - the list of cuts generated for the inner minimization problem so far
    #           each element is a tuple with two elements: (cut_coeffs for pi, cut intercept)

    # add new cuts to the level set lower bound problem
    if spatial_structure_prev != Dict()
        if norm_option == 0
            # L2 norm
            delete(next_pi_prob, next_pi_prob[:level_cons]);
            unregister(next_pi_prob, :level_cons);
            @constraint(next_pi_prob, level_cons, 
                alpha * (sum(next_pi_prob[:pi_var_x][i] * next_pi_prob[:pi_var_x][i] for i in probData.I) + 
                    sum(next_pi_prob[:pi_var_w][i, part_ind] * next_pi_prob[:pi_var_w][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1))) + 
                (1 - alpha) * (L_value - sum(next_pi_prob[:pi_var_x][i] * x_value["x"][i] for i in probData.I) - 
                    sum(next_pi_prob[:pi_var_w][i, part_ind] * x_value["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1)) - 
                    next_pi_prob[:theta]) <= level);
        else
            # L1 norm
            delete(next_pi_prob, next_pi_prob[:level_cons]);
            unregister(next_pi_prob, :level_cons);
            @constraint(next_pi_prob, level_cons, 
                alpha * (sum(next_pi_prob[:pi_abs_x][i] for i in probData.I) + 
                    sum(next_pi_prob[:pi_abs_w][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1))) + 
                (1 - alpha) * (L_value - sum(next_pi_prob[:pi_var_x][i] * x_value["x"][i] for i in probData.I) - 
                    sum(next_pi_prob[:pi_var_w][i, part_ind] * x_value["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1)) - 
                    next_pi_prob[:theta]) <= level);
        end

        # update the cuts
        for j in update_ind_range
            @constraint(next_pi_prob, sum(cut_Dict_node[j][1][i] * next_pi_prob[:pi_var_x][i] for i in probData.I) + 
                sum(cut_Dict_node[j][3][i, part_ind] * next_pi_prob[:pi_var_w][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1)) + 
                cut_Dict_node[j][2] >= next_pi_prob[:theta]);
        end

        # set up the objective function absolute value rhs term
        for i in probData.I
            set_normalized_rhs(next_pi_prob[:pi_obj_pos_x][i], -pi_bar_value["x"][i]);
            set_normalized_rhs(next_pi_prob[:pi_obj_neg_x][i], pi_bar_value["x"][i]);
        end
        for i in keys(spatial_structure_prev)
            for part_ind in 1:(length(spatial_structure_prev[i])+1)
                set_normalized_rhs(next_pi_prob[:pi_obj_pos_w][i, part_ind], -pi_bar_value["w"][i, part_ind]);
                set_normalized_rhs(next_pi_prob[:pi_obj_neg_w][i, part_ind], pi_bar_value["w"][i, part_ind]);
            end
        end
    else
        if norm_option == 0
            # L2 norm
            delete(next_pi_prob, next_pi_prob[:level_cons]);
            unregister(next_pi_prob, :level_cons);
            @constraint(next_pi_prob, level_cons, 
                alpha * sum(next_pi_prob[:pi_var_x][i] * next_pi_prob[:pi_var_x][i] for i in probData.I) + 
                (1 - alpha) * (L_value - sum(next_pi_prob[:pi_var_x][i] * x_value["x"][i] for i in probData.I) - next_pi_prob[:theta]) <= level);
        else
            # L1 norm
            delete(next_pi_prob, next_pi_prob[:level_cons]);
            unregister(next_pi_prob, :level_cons);
            @constraint(next_pi_prob, level_cons, 
                alpha * sum(next_pi_prob[:pi_abs_x][i] for i in probData.I) + 
                (1 - alpha) * (L_value - sum(next_pi_prob[:pi_var_x][i] * x_value["x"][i] for i in probData.I) - next_pi_prob[:theta]) <= level);
        end

        # update the cuts
        for j in update_ind_range
            @constraint(next_pi_prob, sum(cut_Dict_node[j][1][i] * next_pi_prob[:pi_var_x][i] for i in probData.I) + cut_Dict_node[j][2] >= next_pi_prob[:theta]);
        end

        # set up the objective function absolute value rhs term
        for i in probData.I
            set_normalized_rhs(next_pi_prob[:pi_obj_pos_x][i], -pi_bar_value["x"][i]);
            set_normalized_rhs(next_pi_prob[:pi_obj_neg_x][i], pi_bar_value["x"][i]);
        end
    end

    return next_pi_prob;
end

function maximize_lower_envelope(gamma, eta)
    n = length(gamma)
    @assert n == length(eta) "gamma and eta must have same length"

    # If only one line, trivial: maximize gamma*x + eta on [0,1]
    if n == 1
        if gamma[1] > 0
            return 1.0, gamma[1]*1 + eta[1]
        else
            return 0.0, eta[1]
        end
    end

    # Collect candidate x-values
    xs = Float64[0.0, 1.0]   # boundaries always candidates

    # Compute pairwise intersections
    for j in 1:n, k in j+1:n
        gj, gk = gamma[j], gamma[k]
        ej, ek = eta[j], eta[k]

        if gj != gk
            x_int = (ek - ej) / (gj - gk)
            if 0.0 <= x_int <= 1.0
                push!(xs, x_int)
            end
        end
    end

    # Remove duplicates
    xs = unique(xs)

    # Evaluate envelope on all candidates
    best_x = 0.0
    best_val = -Inf

    for x in xs
        v = minimum(gamma .* x .+ eta)
        if v > best_val
            best_val = v
            best_x = x
        end
    end

    return best_x, best_val;
end

function obtain_alpha_bounds(pi_list, spatial_structure_prev, L_value, x_value, v_underbar, V_list, norm_option)
    # algebraic way to calculate alpha_max and alpha_min
    alpha_underbar = [];
    alpha_bar = [];
    gamma_list = Dict();
    eta_list = Dict();

    if spatial_structure_prev != Dict()
        x_value_w_keys = Set([i for (i, part_ind) in keys(x_value["w"])]);
        spatial_prev_keys = Set(keys(spatial_structure_prev));
        if x_value_w_keys != spatial_prev_keys
            println("ERROR in obtain_alpha_bounds: Key mismatch!");
            println("  x_value[\"w\"] keys: $x_value_w_keys");
            println("  spatial_structure_prev keys: $spatial_prev_keys");
        end
    end

    for k in eachindex(pi_list)
        if spatial_structure_prev != Dict()
            if norm_option == 0
                # L2 norm
                gamma_list[k] = (pi_list[k]["x"]'pi_list[k]["x"] + sum(pi_list[k]["w"][i, part_ind] * pi_list[k]["w"][i, part_ind] for (i,part_ind) in keys(pi_list[k]["w"])) - v_underbar) - 
                    (L_value - pi_list[k]["x"]'x_value["x"] - sum(pi_list[k]["w"][i,part_ind] * x_value["w"][i,part_ind] for (i,part_ind) in keys(x_value["w"])) - V_list[k]);
                eta_list[k] = (L_value - pi_list[k]["x"]'x_value["x"] - sum(pi_list[k]["w"][i,part_ind] * x_value["w"][i,part_ind] for (i,part_ind) in keys(x_value["w"])) - V_list[k]);
            else
                # L1 norm
                gamma_list[k] = (sum(abs.(pi_list[k]["x"])) + sum(abs(pi_list[k]["w"][i,part_ind]) for (i,part_ind) in keys(pi_list[k]["w"])) - v_underbar) - 
                    (L_value - pi_list[k]["x"]'x_value["x"] - sum(pi_list[k]["w"][i,part_ind] * x_value["w"][i,part_ind] for (i,part_ind) in keys(x_value["w"])) - V_list[k]);
                eta_list[k] = (L_value - pi_list[k]["x"]'x_value["x"] - sum(pi_list[k]["w"][i,part_ind] * x_value["w"][i,part_ind] for (i,part_ind) in keys(x_value["w"])) - V_list[k]);
                if abs(eta_list[k]) <= 1e-7
                    eta_list[k] = 0.0
                end
            end
        else
            if norm_option == 0
                # L2 norm
                gamma_list[k] = (pi_list[k]["x"]'pi_list[k]["x"] - v_underbar) - 
                    (L_value - pi_list[k]["x"]'x_value["x"] - V_list[k]);
                eta_list[k] = (L_value - pi_list[k]["x"]'x_value["x"] - V_list[k]);
            else
                # L1 norm
                gamma_list[k] = (sum(abs.(pi_list[k]["x"])) - v_underbar) - (L_value - pi_list[k]["x"]'x_value["x"] - V_list[k]);
                eta_list[k] = (L_value - pi_list[k]["x"]'x_value["x"] - V_list[k]);
                if abs(eta_list[k]) <= 1e-7
                    eta_list[k] = 0.0
                end
            end
        end
        if gamma_list[k] >= 0.0
            push!(alpha_bar, 1.0)
            if eta_list[k] >= 0.0
                push!(alpha_underbar, 0.0)
            else
                if -eta_list[k] / gamma_list[k] <= 1
                    push!(alpha_underbar, -eta_list[k] / gamma_list[k])
                elseif -eta_list[k] / gamma_list[k] >= 0
                    push!(alpha_underbar, 1.0);
                    println("Warning: alpha_underbar is set to 1.0 since -eta/gamma = $(-eta_list[k] / gamma_list[k]) > 1")
                else
                    throw("alpha_underbar is not feasible")
                end
            end
        else
            push!(alpha_underbar, 0.0);
            if eta_list[k] >= 0.0
                if -eta_list[k] / gamma_list[k] <= 1 + 1e-5
                    push!(alpha_bar, -eta_list[k] / gamma_list[k]);
                else
                    push!(alpha_bar, 1.0);
                end
            else
                println("eta = $(eta_list[k]), gamma = $(gamma_list[k]), eta/gamma = $(-eta_list[k] / gamma_list[k])");
                # throw("alpha_bar is not feasible");
            end
        end
    end
    alpha_min = round(maximum(alpha_underbar), digits=7);
    alpha_max = round(minimum(alpha_bar), digits=7);

    # algebraic way to calculate Delt
    alpha_maximizer, Delta = maximize_lower_envelope(collect(values(gamma_list)), collect(values(eta_list)));

    return alpha_max, alpha_min, Delta;
end

function build_subproblem_lag(probData, t, node, spatial_structure_prev, spatial_structure_t, cut_lag, x_prev, pi_value)
    # build the Lagrangian relaxation problem
    # input:
    # t: current stage
    # node: the current node index
    # spatial_structure_t: the spatial branching structure at time t
    # cut_lag: the Lagrangian cuts for all nodes in the scenario tree
    # x_prev: the incumbent decisions from the previous stage

    # output:
    # stage_problem: the JuMP model for the current stage problem

    stage_problem = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0, "Threads" => 1, "MIPGap" => 1e-6));

    # define decision variables
    @variable(stage_problem, 0 <= x[i in probData.I] <= probData.xub[i]);                        # inventory
    @variable(stage_problem, 0 <= xcp[i in probData.I] <= probData.xub[i]);                      # inventory copy for the last stage
    @variable(stage_problem, us[i in probData.I] >= 0);                       # sales
    @variable(stage_problem, ub[i in probData.I] >= 0);                       # purchase
    @variable(stage_problem, z[k in eachindex(probData.K)], Bin);             # marketing
    @variable(stage_problem, theta[ω in node.successor] >= -sum(probData.xub[i] * probData.s[i] * 50 for i in probData.I));              # value function approximation

    # constraints
    @constraint(stage_problem, sales_bound1[i in probData.I], us[i] <= node.d[i]);
    @constraint(stage_problem, sales_bound2[i in probData.I], us[i] <= xcp[i]);
    @constraint(stage_problem, inventory_balance[i in probData.I], x[i] == xcp[i] - us[i] + ub[i]);
    @constraint(stage_problem, capacity_constraint, sum(ub[i] for i in probData.I) <= probData.Capacity);
    @constraint(stage_problem, marketing_constraint, sum(z[k] for k in eachindex(probData.K)) <= 1);

    # objective function
    probability_changed = Dict();
    for ω in node.successor
        for k in eachindex(probData.K)
            # calculate the probability after the marketing strategy
            probList = [];
            for i in probData.I
                if node.sign[ω][i] == 1
                    if probData.K[k][i] == 1
                        push!(probList, probData.p_up[i]);
                    else
                        push!(probList, probData.p_norm[i]);
                    end
                else
                    if probData.K[k][i] == 1
                        push!(probList, probData.p_down[i]);
                    else
                        push!(probList, probData.p_norm[i]);
                    end
                end
            end
            probability_changed[ω,k] = prod(probList);
        end
    end
    @expression(stage_problem, stage_cost, sum(probData.b[i] * ub[i] + probData.h[i] * x[i] - probData.s[i] * us[i] for i in probData.I) +
        sum(z[k] * probData.f[k] for k in eachindex(probData.K)));
    @expression(stage_problem, expected_future_cost, sum(probability_changed[ω,k] * z[k] * theta[ω] for k in eachindex(probData.K) for ω in node.successor));

    # add predecessor's spatial partitioning structure
    if spatial_structure_prev != Dict()
        # add the branching variables
        @variable(stage_problem, wcp[i in keys(spatial_structure_prev), part_ind in 1:(length(spatial_structure_prev[i])+1)], Bin);
        # add the spatial partitioning constraints
        @constraint(stage_problem, wcp_1[i in keys(spatial_structure_prev)], sum(wcp[i, part_ind] for part_ind in 1:(length(spatial_structure_prev[i]) + 1)) == 1);
        part_lb_prev = Dict();
        part_ub_prev = Dict();
        for i in keys(spatial_structure_prev)
            for part_ind in 1:(length(spatial_structure_prev[i]) + 1)
                if part_ind == 1
                    part_lb_prev[i, part_ind] = probData.xlb[i];
                    part_ub_prev[i, part_ind] = spatial_structure_prev[i][1];
                elseif part_ind == (length(spatial_structure_prev[i]) + 1)
                    part_lb_prev[i, part_ind] = spatial_structure_prev[i][length(spatial_structure_prev[i])];
                    part_ub_prev[i, part_ind] = probData.xub[i];
                else
                    part_lb_prev[i, part_ind] = spatial_structure_prev[i][part_ind - 1];
                    part_ub_prev[i, part_ind] = spatial_structure_prev[i][part_ind];
                end
            end
        end
        @constraint(stage_problem, x_part_ub_prev[i in keys(spatial_structure_prev)], xcp[i] <= sum(part_ub_prev[i, part_ind] * wcp[i, part_ind] for part_ind in 1:(length(spatial_structure_prev[i]) + 1)));
        @constraint(stage_problem, x_part_lb_prev[i in keys(spatial_structure_prev)], xcp[i] >= sum(part_lb_prev[i, part_ind] * wcp[i, part_ind] for part_ind in 1:(length(spatial_structure_prev[i]) + 1)));
        @objective(stage_problem, Min, sum(probData.b[i] * ub[i] + probData.h[i] * x[i] - probData.s[i] * us[i] for i in probData.I) +
                        sum(probability_changed[ω,k] * z[k] * theta[ω] for k in eachindex(probData.K) for ω in node.successor) + 
                        sum(z[k] * probData.f[k] for k in eachindex(probData.K)) - 
                        sum(pi_value["x"][i] * xcp[i] for i in probData.I) - 
                        sum(pi_value["w"][i, part_ind] * wcp[i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)));
    else
        @objective(stage_problem, Min, sum(probData.b[i] * ub[i] + probData.h[i] * x[i] - probData.s[i] * us[i] for i in probData.I) +
                        sum(probability_changed[ω,k] * z[k] * theta[ω] for k in eachindex(probData.K) for ω in node.successor) + 
                        sum(z[k] * probData.f[k] for k in eachindex(probData.K)) - 
                        sum(pi_value["x"][i] * xcp[i] for i in probData.I));
    end

    # add spatial partitioning structure constraints
    if spatial_structure_t == Dict()
        # no branching
        # add Lagrangian cuts
        for ω in node.successor
            for j in eachindex(cut_lag[t+1][ω])
                @constraint(stage_problem, sum(cut_lag[t+1][ω][j][1][i] * x[i] for i in probData.I) + cut_lag[t+1][ω][j][2] <= theta[ω]);
            end
        end
    else
        # add the branching variables
        @variable(stage_problem, w[i in keys(spatial_structure_t), part_ind in 1:(length(spatial_structure_t[i]) + 1)], Bin);
        # add the spatial partitioning constraints
        @constraint(stage_problem, w_1[i in keys(spatial_structure_t)], sum(w[i, part_ind] for part_ind in 1:(length(spatial_structure_t[i]) + 1)) == 1);
        part_lb = Dict();
        part_ub = Dict();
        for i in keys(spatial_structure_t)
            for part_ind in 1:(length(spatial_structure_t[i]) + 1)
                if part_ind == 1
                    part_lb[i, part_ind] = probData.xlb[i];
                    part_ub[i, part_ind] = spatial_structure_t[i][1];
                elseif part_ind == (length(spatial_structure_t[i]) + 1)
                    part_lb[i, part_ind] = spatial_structure_t[i][length(spatial_structure_t[i])];
                    part_ub[i, part_ind] = probData.xub[i];
                else
                    part_lb[i, part_ind] = spatial_structure_t[i][part_ind - 1];
                    part_ub[i, part_ind] = spatial_structure_t[i][part_ind];
                end
            end
        end
        @constraint(stage_problem, x_part_ub[i in keys(spatial_structure_t)], x[i] <= sum(part_ub[i, part_ind] * w[i, part_ind] for part_ind in 1:(length(spatial_structure_t[i]) + 1)));
        @constraint(stage_problem, x_part_lb[i in keys(spatial_structure_t)], x[i] >= sum(part_lb[i, part_ind] * w[i, part_ind] for part_ind in 1:(length(spatial_structure_t[i]) + 1)));

        # add Lagrangian cuts
        for ω in node.successor
            for j in eachindex(cut_lag[t+1][ω])
                @constraint(stage_problem, sum(cut_lag[t+1][ω][j][1][i] * x[i] for i in probData.I) + 
                    sum(cut_lag[t+1][ω][j][3][i, part_ind] * w[i, part_ind] for i in keys(spatial_structure_t) for part_ind in 1:(length(spatial_structure_t[i]) + 1)) + 
                    cut_lag[t+1][ω][j][2] <= theta[ω]);
            end
        end
    end

    return stage_problem;
end

function update_subproblem_lag(sub_prob_lag, probData, node, spatial_structure_prev, pi_value)
    # update the Lagrangian relaxation problem with the objective function including Lagrangian penalty term
    if spatial_structure_prev != Dict()
        @objective(sub_prob_lag, Min, sub_prob_lag[:stage_cost] + sub_prob_lag[:expected_future_cost] -
                sum(pi_value["x"][i] * sub_prob_lag[:xcp][i] for i in probData.I) - 
                sum(pi_value["w"][i, part_ind] * sub_prob_lag[:wcp][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)));
    else
        @objective(sub_prob_lag, Min, sub_prob_lag[:stage_cost] + sub_prob_lag[:expected_future_cost] - 
                sum(pi_value["x"][i] * sub_prob_lag[:xcp][i] for i in probData.I));
    end
    return sub_prob_lag;
end

function solve_lag_dual(probData, t, node, spatial_structure_prev, spatial_structure_t, x_prev, L_value, lambda_level, mu_level, norm_option, tol, 
        cut_lag, cut_Dict_node, sub_lb = -1e5, sub_ub = 1e5, iter_limit = 100)
    # solve the Lagrangian dual problem

    # initialize the cut list for the level set problem
    v_value = 0;
    alpha_min = 0;
    alpha_max = 1;
    alpha = (alpha_max + alpha_min) / 2;
    pi_list = [];
    V_list = [];
    pi_value = Dict();

    if spatial_structure_prev != Dict()
        x_value_w_keys = Set([i for (i, part_ind) in keys(x_prev["w"])]);
        spatial_prev_keys = Set(keys(spatial_structure_prev));
        if x_value_w_keys != spatial_prev_keys
            println("ERROR in obtain_alpha_bounds: Key mismatch!");
            println("  x_value[\"w\"] keys: $x_value_w_keys");
            println("  spatial_structure_prev keys: $spatial_prev_keys");
        end
    end

    # set up the lower bound problem
    lb_prob = build_ls_lb_problem(probData, x_prev, spatial_structure_prev, L_value, norm_option, cut_Dict_node);
    # initialize the Lagrangian dual multipliers with the lb solution
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
            if spatial_structure_prev != Dict()
                @constraint(lb_prob, cons, sum(lb_prob[:pi_var_x][i] * x_prev["x"][i] for i in probData.I) + 
                    sum(lb_prob[:pi_var_w][i, part_ind] * x_prev["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)) + 
                    lb_prob[:theta] >= L_value);
            else
                @constraint(lb_prob, cons, sum(lb_prob[:pi_var_x][i] * x_prev["x"][i] for i in probData.I) + lb_prob[:theta] >= L_value);
            end
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
    pi_value["x"] = value.(lb_prob[:pi_var_x]);
    if spatial_structure_prev != Dict()
        pi_value["w"] = Dict();
        for i in keys(spatial_structure_prev)
            for part_ind in 1:(length(spatial_structure_prev[i]) + 1)
                pi_value["w"][i, part_ind] = value(lb_prob[:pi_var_w][i, part_ind]);
            end
        end
    end

    # set up the auxiliary problem to find the next pi_value
    level = sub_ub;
    next_pi_prob = build_next_pi_problem(probData, spatial_structure_prev, level, alpha, x_prev, L_value, cut_Dict_node, norm_option);

    # build the inner min subproblem with Lagrangian penalty term
    sub_prob = build_subproblem_lag(probData, t, node, spatial_structure_prev, 
        spatial_structure_t, cut_lag, x_prev, pi_value);
    # solve the subproblem
    optimize!(sub_prob);

    # initialize the termination criterion
    cont_bool = true;
    counter = 0;
    Delta_list = [];

    # loop until the termination criterion
    while cont_bool
        # record the pi_value
        push!(pi_list, deepcopy(pi_value));

        # obtain the inner minimization problem's optimal solution
        xcp_value = value.(sub_prob[:xcp]);
        if spatial_structure_prev != Dict()
            wcp_value = value.(sub_prob[:wcp]);
        end
        # add the cut to the cut list
        Vj = objective_value(sub_prob);        # V(\pi_j)
        push!(V_list, Vj);
        if length(cut_Dict_node) > 0
            if spatial_structure_prev != Dict()
                theta_j = minimum([sum(cut_Dict_node[j][1][i] * pi_value["x"][i] for i in probData.I) + 
                    sum(cut_Dict_node[j][3][i, part_ind] * pi_value["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)) + 
                    cut_Dict_node[j][2] for j in eachindex(cut_Dict_node)]);
            else
                theta_j = minimum([sum(cut_Dict_node[j][1][i] * pi_value["x"][i] for i in probData.I) + cut_Dict_node[j][2] for j in eachindex(cut_Dict_node)]);
            end
        else
            theta_j = Inf;
        end

        # only generate the cut if Vj > theta_j
        if Vj < theta_j - 1e-4
            if spatial_structure_prev != Dict()
                push!(cut_Dict_node, [-xcp_value, Vj + sum(xcp_value[i] * pi_value["x"][i] for i in probData.I) +
                    sum(wcp_value[i, part_ind] * pi_value["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)), 
                    -wcp_value]);
            else
                push!(cut_Dict_node, [-xcp_value, Vj + sum(xcp_value[i] * pi_value["x"][i] for i in probData.I)]);
            end
            cutList_update_start_ind = length(cut_Dict_node);
            cutList_update_end_ind = length(cut_Dict_node);
        else
            cutList_update_start_ind = length(cut_Dict_node) + 1;
            cutList_update_end_ind = length(cut_Dict_node);
        end

        # update and solve the lower bound problem
        lb_prob = update_ls_lb_problem(lb_prob, probData, spatial_structure_prev, cut_Dict_node, cutList_update_start_ind:cutList_update_end_ind);
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
                if spatial_structure_prev != Dict()
                    @constraint(lb_prob, cons, sum(lb_prob[:pi_var_x][i] * x_prev["x"][i] for i in probData.I) + 
                        sum(lb_prob[:pi_var_w][i, part_ind] * x_prev["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i]) + 1)) + 
                        lb_prob[:theta] >= L_value);
                else
                    @constraint(lb_prob, cons, sum(lb_prob[:pi_var_x][i] * x_prev["x"][i] for i in probData.I) + lb_prob[:theta] >= L_value);
                end
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
        alpha_max, alpha_min, Delta = obtain_alpha_bounds(pi_list, spatial_structure_prev, L_value, x_prev, objective_value(lb_prob), V_list, norm_option);
        push!(Delta_list, Delta);
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
            if length(Delta_list) >= 5
                if abs(mean(Delta_list[length(Delta_list)-4:length(Delta_list)]) - Delta_list[length(Delta_list)]) < 1e-5
                    cont_bool = false;
                else
                    if counter > iter_limit
                        cont_bool = false;
                    else
                        counter += 1
                        if spatial_structure_prev != Dict()
                            # update the level
                            if norm_option == 0
                                v_bar_list = [alpha * (pi_list[pi_j_ind]["x"]'pi_list[pi_j_ind]["x"] + 
                                                sum(pi_list[pi_j_ind]["w"][i,part_ind] * pi_list[pi_j_ind]["w"][i,part_ind]) for (i,part_ind) in keys(pi_list[pi_j_ind]["w"])) + 
                                    (1 - alpha) * (L_value - pi_list[pi_j_ind]["x"]'x_prev["x"] - 
                                        sum(pi_list[pi_j_ind]["w"][i,part_ind] * x_prev["w"][i,part_ind] for (i,part_ind) in keys(x_prev["w"])) - V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                            else
                                v_bar_list = [alpha * sum(abs.(pi_list[pi_j_ind]["x"])) + sum(abs(pi_list[pi_j_ind]["w"][i,part_ind]) for (i,part_ind) in keys(pi_list[pi_j_ind]["w"])) + 
                                    (1 - alpha) * (L_value - pi_list[pi_j_ind]["x"]'x_prev["x"] - 
                                    sum(pi_list[pi_j_ind]["w"][i,part_ind] * x_prev["w"][i,part_ind] for (i,part_ind) in keys(x_prev["w"])) -  V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                            end
                        else
                            # update the level
                            if norm_option == 0
                                v_bar_list = [alpha * pi_list[pi_j_ind]["x"]'pi_list[pi_j_ind]["x"] + 
                                    (1 - alpha) * (L_value - pi_list[pi_j_ind]["x"]'x_prev["x"] - V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                            else
                                v_bar_list = [alpha * sum(abs.(pi_list[pi_j_ind]["x"])) + 
                                    (1 - alpha) * (L_value - pi_list[pi_j_ind]["x"]'x_prev["x"] - V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                            end
                        end
                        v_bar = minimum(v_bar_list);
                        v_underbar = alpha * objective_value(lb_prob);
                        level = lambda_level * v_bar + (1 - lambda_level) * v_underbar;

                        # solve for the next pi_value
                        next_pi_prob = update_next_pi_problem(next_pi_prob, probData, spatial_structure_prev, cut_Dict_node, cutList_update_start_ind:cutList_update_end_ind, alpha, x_prev, L_value, pi_value, level, norm_option);
                        optimize!(next_pi_prob);
                        # obtain the next pi_value
                        if termination_status(next_pi_prob) != OPTIMAL
                            #throw("next_pi_prob is not optimal");
                            cont_bool = false;
                            println("next_pi_prob is not optimal, it is $(termination_status(next_pi_prob)) at iteration $(counter)");
                        else
                            pi_value["x"] = value.(next_pi_prob[:pi_var_x]);
                            if spatial_structure_prev != Dict()
                                pi_value["w"] = Dict();
                                for i in keys(spatial_structure_prev)
                                    for part_ind in 1:(length(spatial_structure_prev[i]) + 1)
                                        pi_value["w"][i, part_ind] = value(next_pi_prob[:pi_var_w][i, part_ind]);
                                    end
                                end
                            end
                        end
                        # update the subproblem and solve it
                        sub_prob = update_subproblem_lag(sub_prob, probData, node, spatial_structure_prev, pi_value);
                        optimize!(sub_prob);
                    end
                end
            else
                if counter > iter_limit
                    cont_bool = false;
                else
                    counter += 1
                    if spatial_structure_prev != Dict()
                        # update the level
                        if norm_option == 0
                            v_bar_list = [alpha * (pi_list[pi_j_ind]["x"]'pi_list[pi_j_ind]["x"] + 
                                            sum(pi_list[pi_j_ind]["w"][i,part_ind] * pi_list[pi_j_ind]["w"][i,part_ind]) for (i,part_ind) in keys(pi_list[pi_j_ind]["w"])) + 
                                (1 - alpha) * (L_value - pi_list[pi_j_ind]["x"]'x_prev["x"] - 
                                    sum(pi_list[pi_j_ind]["w"][i,part_ind] * x_prev["w"][i,part_ind] for (i,part_ind) in keys(x_prev["w"])) - V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                        else
                            v_bar_list = [alpha * sum(abs.(pi_list[pi_j_ind]["x"])) + sum(abs(pi_list[pi_j_ind]["w"][i,part_ind]) for (i,part_ind) in keys(pi_list[pi_j_ind]["w"])) + 
                                (1 - alpha) * (L_value - pi_list[pi_j_ind]["x"]'x_prev["x"] - 
                                sum(pi_list[pi_j_ind]["w"][i,part_ind] * x_prev["w"][i,part_ind] for (i,part_ind) in keys(x_prev["w"])) -  V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                        end
                    else
                        # update the level
                        if norm_option == 0
                            v_bar_list = [alpha * pi_list[pi_j_ind]["x"]'pi_list[pi_j_ind]["x"] + 
                                (1 - alpha) * (L_value - pi_list[pi_j_ind]["x"]'x_prev["x"] - V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                        else
                            v_bar_list = [alpha * sum(abs.(pi_list[pi_j_ind]["x"])) + 
                                (1 - alpha) * (L_value - pi_list[pi_j_ind]["x"]'x_prev["x"] - V_list[pi_j_ind]) for pi_j_ind in eachindex(pi_list)];
                        end
                    end
                    v_bar = minimum(v_bar_list);
                    v_underbar = alpha * objective_value(lb_prob);
                    level = lambda_level * v_bar + (1 - lambda_level) * v_underbar;

                    # solve for the next pi_value
                    next_pi_prob = update_next_pi_problem(next_pi_prob, probData, spatial_structure_prev, cut_Dict_node, cutList_update_start_ind:cutList_update_end_ind, alpha, x_prev, L_value, pi_value, level, norm_option);
                    optimize!(next_pi_prob);
                    # obtain the next pi_value
                    if termination_status(next_pi_prob) != OPTIMAL
                        throw("next_pi_prob is not optimal");
                    else
                        pi_value["x"] = value.(next_pi_prob[:pi_var_x]);
                        if spatial_structure_prev != Dict()
                            pi_value["w"] = Dict();
                            for i in keys(spatial_structure_prev)
                                for part_ind in 1:(length(spatial_structure_prev[i]) + 1)
                                    pi_value["w"][i, part_ind] = value(next_pi_prob[:pi_var_w][i, part_ind]);
                                end
                            end
                        end
                    end
                    # update the subproblem and solve it
                    sub_prob = update_subproblem_lag(sub_prob, probData, node, spatial_structure_prev, pi_value);
                    optimize!(sub_prob);
                end
            end
        end

        # output the current status
        println("Iteration: $(counter), V(pi_j): $(objective_value(sub_prob)), Delta: $(Delta)");
    end

    # obtain the intercept of the Lagrangian cut
    sub_prob = update_subproblem_lag(sub_prob, probData, node, spatial_structure_prev, pi_value);
    optimize!(sub_prob);
    if spatial_structure_prev != Dict()
        Q_value = objective_value(sub_prob) + sum(pi_value["x"][i] * x_prev["x"][i] for i in probData.I) + sum(pi_value["w"][i, part_ind] * x_prev["w"][i, part_ind] for i in keys(spatial_structure_prev) for part_ind in 1:(length(spatial_structure_prev[i])+1));
    else
        Q_value = objective_value(sub_prob) + sum(pi_value["x"][i] * x_prev["x"][i] for i in probData.I);
    end
    v_value = objective_value(sub_prob);

    return pi_value, v_value, cut_Dict_node, Q_value;
end

function sub_routine(probData, t, node, x_prev, lambda_level, mu_level, norm_option, tol, 
        spatial_structure_prev, spatial_structure_t, cut_lag, cut_Dict_node)
    # input:
    # t: current stage
    # scen_ind: the current scenario index
    # x_prev: the incumbent decisions from the previous stage
    # lambda_level: the risk level for the CVaR constraint on the Lagrangian multipliers
    # mu_level: the risk level for the CVaR constraint on the recourse function value
    # norm_option: the choice of norm for the regularization term
    # tol: the tolerance for convergence
    # spatial_structure_t: the spatial branching structure at time t
    # cut_lag_node: the Lagrangian cuts for the node

    # output:
    # pi_value_o: the dual values associated with the first-stage decisions
    # v_value_o: the intercept of the Lagrangian cut
    # cut_Dict_node: the updated list of cuts for the node's Lagrangian dual problem

    # obtain the subproblem value & update the upper bound
    sub_prob = build_stage_problem(probData, t, node, spatial_structure_t, cut_lag, x_prev);
    optimize!(sub_prob);
    # obtain the subproblem solution/optimal value and update the upper bound
    L_value = objective_value(sub_prob); 

    if spatial_structure_prev != Dict()
        x_value_w_keys = Set([i for (i, part_ind) in keys(x_prev["w"])]);
        spatial_prev_keys = Set(keys(spatial_structure_prev));
        if x_value_w_keys != spatial_prev_keys
            println("ERROR in sub_routine: Key mismatch!");
            println("  x_value[\"w\"] keys: $x_value_w_keys");
            println("  spatial_structure_prev keys: $spatial_prev_keys");
        end
    end
    # generate the Lagrangian cuts
    pi_value_node, v_value_node, cut_Dict_node, Q_value_node = solve_lag_dual(probData, t, node, spatial_structure_prev,
        spatial_structure_t, x_prev, L_value, lambda_level, mu_level, norm_option, tol, cut_lag, cut_Dict_node);

    return pi_value_node, v_value_node, cut_Dict_node, Q_value_node, L_value;
end

function forward_single(m, probData, T, scenTree, spatial_structure, cut_lag, x_star_0, theta_star_0, given_path = [])
    # solve a single scenario path in the forward pass
    VList = [];
    # generate the scenario path
    x_star = Dict();
    theta_star = Dict();
    x_star[1] = x_star_0;
    # record the scenario path
    scen_star = Dict();
    scen_star[1] = 1;
    theta_star[1] = theta_star_0;

    # obtain the scenario path solutions
    for t in 2:T
        # obtain the scenario index at time t
        previous_scen = scen_star[t-1];
        if length(given_path) >= m
            scen_ind = given_path[m][t];
        else
            scen_ind = generate_scenario_index(scenTree, previous_scen, t);
        end
        scen_star[t] = scen_ind;
        stage_problem = build_stage_problem(probData, t, scenTree[t][scen_ind], spatial_structure[t], cut_lag, x_star[t - 1]);
        optimize!(stage_problem);

        # obtain the decision at time t
        x_star[t] = Dict();
        x_star[t]["x"] = value.(stage_problem[:x]);
        if spatial_structure[t] != Dict()
            x_star[t]["w"] = Dict();
            for i in keys(spatial_structure[t])
                for part_ind in 1:(length(spatial_structure[t][i]) + 1)
                    x_star[t]["w"][i, part_ind] = value(stage_problem[:w][i, part_ind]);
                end
            end
        end
        theta_star[t] = value.(stage_problem[:theta]);
        # calculate the current stage cost
        push!(VList, value(stage_problem[:stage_cost]));
    end

    return sum(VList), x_star, scen_star, theta_star;
end

# forward pass function
function forward_pass(probData, T, M, scenTree, spatial_structure, cut_lag, para_option = 0, given_path = Dict())
    # input:
    # T: the time horizon
    # M: number of sample paths
    # scenTree: the scenario tree object
    # spatial_structure: the spatial branching structure
    # cut_lag: the Lagrangian cuts for each node in the scenario tree

    x_star = Dict();
    scen_star = Dict();
    theta_star = Dict();
    x0 = Dict("x" => Dict());
    for i in probData.I
        x0["x"][i] = 0.0;
    end
    first_stage_problem = build_stage_problem(probData, 1, scenTree[1][1], spatial_structure[1], cut_lag, x0);
    optimize!(first_stage_problem);
    # obtain the lower bound
    Vstar = objective_value(first_stage_problem);
    V1 = value(first_stage_problem[:stage_cost]);
    # obtain the first-stage decision
    x_star[0] = Dict();
    x_star[0]["x"] = value.(first_stage_problem[:x]);
    if spatial_structure[1] != Dict()
        x_star[0]["w"] = Dict();
        for i in keys(spatial_structure[1])
            for part_ind in 1:(length(spatial_structure[1][i]) + 1)
                x_star[0]["w"][i, part_ind] = value(first_stage_problem[:w][i, part_ind]);
            end
        end
    end
    theta_star[0] = value.(first_stage_problem[:theta]);

    VbarList = [];
    if para_option == 0
        for m in 1:M
            VList = [];
            # generate the scenario path
            x_star[m] = Dict();
            x_star[m][1] = x_star[0];
            # record the scenario path
            scen_star[m] = Dict();
            scen_star[m][1] = 1;
            theta_star[m] = Dict();
            theta_star[m][1] = theta_star[0];
            # record the first-stage cost
            push!(VList, value(first_stage_problem[:stage_cost]));

            # obtain the scenario path solutions
            for t in 2:T
                # obtain the scenario index at time t
                previous_scen = scen_star[m][t-1];
                if m in keys(given_path)
                    scen_ind = given_path[m][t];    
                else
                    scen_ind = generate_scenario_index(scenTree, previous_scen, t);
                end
                scen_star[m][t] = scen_ind;
                stage_problem = build_stage_problem(probData, t, scenTree[t][scen_ind], spatial_structure[t], cut_lag, x_star[m][t - 1]);
                optimize!(stage_problem);

                # obtain the decision at time t
                x_star[m][t] = Dict();
                x_star[m][t]["x"] = value.(stage_problem[:x]);
                if spatial_structure[t] != Dict()
                    for i in keys(spatial_structure[t])
                        for part_ind in 1:(length(spatial_structure[t][i]) + 1)
                            x_star[m][t]["w"][i, part_ind] = value(stage_problem[:w][i, part_ind]);
                        end
                    end
                end
                theta_star[m][t] = value.(stage_problem[:theta]);
                # calculate the current stage cost
                push!(VList, value(stage_problem[:stage_cost]));
            end
            # calculate the total cost for the scenario path
            push!(VbarList, sum(VList));
        end
    elseif para_option == 1
        forward_results = pmap(m -> forward_single(m, probData, T, scenTree, spatial_structure, cut_lag, x_star[0], theta_star[0], given_path), 1:M);
        VbarList = [forward_results[m][1] + V1 for m in 1:M];
        for m in 1:M
            x_star[m] = forward_results[m][2];
            scen_star[m] = forward_results[m][3];
            theta_star[m] = forward_results[m][4];
        end
    end

    return Vstar, x_star, VbarList, scen_star, theta_star;
end

function backward_single(probData, m, scen_ind, t, T, node, x_prev, lambda_level, mu_level, 
    norm_option, tol, spatial_structure_prev, spatial_structure_t, cut_lag, cut_Dict_node)
    # solve a single node in the backward pass to generate a cut
    pi_value, v_value, cut_Dict_node_new, Q_value_node, L_value = sub_routine(probData, t, node, x_prev, lambda_level, mu_level, norm_option, tol, spatial_structure_prev, spatial_structure_t, cut_lag, cut_Dict_node);
    if t == T
        return m, scen_ind, pi_value, v_value, cut_Dict_node_new;
    else
        return m, scen_ind, pi_value, v_value, [];
    end
end

# backward pass function
function backward_pass(probData, T, M, scenTree, x_star, scen_star, theta_star, cut_lag, cut_Dict, spatial_structure, lambda_level, mu_level, norm_option, partition_option, tol = 1e-3, para_option = 0)
    # input:
    # T: the time horizon
    # M: number of sample paths
    # scenTree: the scenario tree object
    # x_star: the first-stage decisions from the forward pass
    # cut_lag: the Lagrangian cuts for each node in the scenario tree
    # cut_Dict: the overall list of level-set cuts
    # spatial_structure: the spatial branching structure

    # generate the cut for stage t from T back to 2
    for t in T-1:-1:1
        # we have obtained M paths of solutions x_star[m][t-1]
        if partition_option == 1
            # partition based on the solutions hitting each node at time t
            for node in scenTree[t]
                node_ind = node.index;
                # for every node at time t, obtain the x_star hitting this node
                x_star_node = [];
                for m in 1:M
                    if scen_star[m][t] == node_ind
                        x_added = Dict();
                        for i in probData.I
                            if spatial_structure[t] != Dict()
                                if i in keys(spatial_structure[t])
                                    if !(round(x_star[m][t]["x"][i], digits = 0) in spatial_structure[t][i])
                                        x_added[i] = round(x_star[m][t]["x"][i], digits = 0);
                                    end
                                else
                                    x_added[i] = round(x_star[m][t]["x"][i], digits = 0);
                                end
                            else
                                x_added[i] = round(x_star[m][t]["x"][i], digits = 0);
                            end
                        end
                        push!(x_star_node, x_added);
                    end
                end
                if x_star_node != []
                    # update the spatial branching structure at time t
                    for x_ind in eachindex(x_star_node)
                        spatial_structure[t], index_partition_node = update_spatial_structure(spatial_structure[t], x_star_node[x_ind]);
                        for ω in node.successor
                            cut_lag[t+1][ω] = update_lag_cut(scenTree[t+1][ω], spatial_structure[t], index_partition_node, cut_lag[t+1][ω]);
                        end
                    end

                    # required to update the x_star
                    for m in 1:M
                        # for each of the previous x
                        if !("w" in keys(x_star[m][t]))
                            # if there was a partition before
                            x_star[m][t]["w"] = Dict();
                        end
                        for i in keys(spatial_structure[t])
                            # Initialize all w values to 0
                            for part_ind in 1:(length(spatial_structure[t][i]) + 1)
                                x_star[m][t]["w"][i, part_ind] = 0;
                            end

                            # Find the correct partition
                            found_part = false;
                            for part_ind in 1:length(spatial_structure[t][i])
                                if x_star[m][t]["x"][i] <= spatial_structure[t][i][part_ind]
                                    found_part = true;
                                    x_star[m][t]["w"][i,part_ind] = 1;
                                    break;
                                end 
                            end
                            if !(found_part)
                                x_star[m][t]["w"][i,length(spatial_structure[t][i])+1] = 1;
                            end
                        end
                    end
                end
            end
            # temporary update the cut_Dict to []
            for node in scenTree[T]
                node_ind = node.index;
                cut_Dict[T][node_ind] = [];
            end
        end

        if para_option == 0
            # execute the backward pass in a serial manner
            for m in 1:M
                # for every scenario index at time t
                current_node_ind = scen_star[m][t];
                for scen_ind in scenTree[t][current_node_ind].successor
                    # solve the Lagrangian dual problem at node scen_ind and return the dual values
                    if t + 1 == T
                        pi_value, v_value, cut_Dict[t+1][scen_ind], Q_value, L_value = sub_routine(probData, t+1, scenTree[t+1][scen_ind], x_star[m][t], lambda_level, mu_level, norm_option, tol, spatial_structure[t], spatial_structure[t+1], cut_lag, cut_Dict[t+1][scen_ind]);
                    else
                        pi_value, v_value, cut_Dict[t+1][scen_ind], Q_value, L_value = sub_routine(probData, t+1, scenTree[t+1][scen_ind], x_star[m][t], lambda_level, mu_level, norm_option, tol, spatial_structure[t], spatial_structure[t+1], cut_lag, []);
                    end
                    # update cut_lag and cut_Dict
                    if Q_value > theta_star[m][t][scen_ind] + 1e-4
                        if "w" in keys(pi_value)
                            # if we indeed cut off some space
                            push!(cut_lag[t+1][scen_ind], [pi_value["x"], v_value, pi_value["w"]]);
                        else
                            push!(cut_lag[t+1][scen_ind], [pi_value["x"], v_value]);
                        end
                    end
                end
            end
        elseif para_option == 1
            # execute the backward pass in a parallel manner
            iter_list = [(m, scen_ind) for m in 1:M for scen_ind in scenTree[t][scen_star[m][t]].successor];
            if t + 1 == T
                backward_results = pmap(((m, scen_ind),) -> backward_single(probData, m, scen_ind, t+1, T, scenTree[t+1][scen_ind], x_star[m][t], lambda_level, mu_level, norm_option, tol, spatial_structure[t], spatial_structure[t+1], cut_lag, cut_Dict[t+1][scen_ind]), iter_list);
            else
                backward_results = pmap(((m, scen_ind),) -> backward_single(probData, m, scen_ind, t+1, T, scenTree[t+1][scen_ind], x_star[m][t], lambda_level, mu_level, norm_option, tol, spatial_structure[t], spatial_structure[t+1], cut_lag, []), iter_list);
            end
            for item in backward_results
                m = item[1];
                scen_ind = item[2];
                pi_value = item[3];
                v_value = item[4];
                cut_Dict_node_new = item[5];
                # update cut_lag and cut_Dict
                if "w" in keys(pi_value)
                    push!(cut_lag[t+1][scen_ind], [pi_value["x"], v_value, pi_value["w"]]);
                else
                    push!(cut_lag[t+1][scen_ind], [pi_value["x"], v_value]);
                end
                if t + 1 == T
                    cut_Dict[t+1][scen_ind] = cut_Dict_node_new;
                end
            end
        end
    end
    return cut_lag, cut_Dict, spatial_structure;
end

# The main SDDLP process
function sddLp_process(probData, T, M, scenTree, lambda_level, mu_level, norm_option, partition_option, tol = 1e-3, iter_limit = 300, bound_threshold = 1e-2, para_option = 1, given_paths = Dict())
    # input:
    # probData: the problem data object
    # T: the time horizon
    # M: number of sample paths
    # scenTree: the scenario tree object

    # initialization
    keep_iter = true;
    LB = -Inf;
    UB = Inf;
    counter = 0;
    LB_list = [];
    UB_list = [];
    time_list = [];
    scen_star_list = Dict();

    # initialize the spatial branching structure
    spatial_structure = Dict();
    cut_lag = Dict();
    cut_Dict = Dict();
    for t in 1:T
        spatial_structure[t] = Dict();
        cut_lag[t] = Dict();
        cut_Dict[t] = Dict();
        for node in scenTree[t]
            node_ind = node.index;
            cut_lag[t][node_ind] = [];
            cut_Dict[t][node_ind] = [];
        end
    end

    # while stopping criteria not met
    while keep_iter
        if mod(counter, 10) == 0
            partition_option = 1;
        else
            partition_option = 0;
        end
        counter += 1;
        start_time = time();
        # run the forward pass
        if given_paths == Dict()
            Vstar, x_star, VbarList, scen_star, theta_star = forward_pass(probData, T, M, scenTree, spatial_structure, cut_lag, para_option);
            scen_star_list[counter] = scen_star;
        else
            Vstar, x_star, VbarList, scen_star, theta_star = forward_pass(probData, T, M, scenTree, spatial_structure, cut_lag, para_option, given_paths[counter]);
        end
        LB = max(LB, Vstar);

        # calculate the statistical upper bound
        UB = mean(VbarList) + 1.96 * std(VbarList) / sqrt(M);
        println("Iteration: $(counter): LB = $(LB), Statistical UB = $(UB)");
        push!(LB_list, LB);
        push!(UB_list, UB);

        # reset the cut_Dict
        for t in 1:(T-1)
            for node in scenTree[t]
                node_ind = node.index;
                cut_Dict[t][node_ind] = [];
            end
        end

        # run the backward pass
        cut_lag, cut_Dict, spatial_structure = backward_pass(probData, T, M, scenTree, x_star, scen_star, theta_star, cut_lag, cut_Dict, spatial_structure, lambda_level, mu_level, norm_option, partition_option, tol, para_option);

        # check the stopping criteria
        # one possibility: iteration limit
        if counter >= iter_limit
            keep_iter = false;
        end
        elapsed_time = time() - start_time;
        push!(time_list, elapsed_time);
        # second possibility: convergence of bounds
        # if abs(UB - LB) / (1e-5 + abs(LB)) <= bound_threshold
        #     keep_iter = false;
        # end
    end
    Vstar = LB_list[length(LB_list)];

    return Vstar, cut_lag, cut_Dict, LB_list, UB_list, time_list, spatial_structure;
end

function forward_pass_path(probData, T, scenTree, scen_star, spatial_structure, cut_lag, x_star_0, thetaList)
        # solve a single scenario path in the forward pass
    VList = [];
    # generate the scenario path
    x_star = Dict();
    x_star[1] = x_star_0;

    # obtain the scenario path solutions
    for t in 2:T
        # obtain the scenario index at time t
        scen_ind = scen_star[t];
        stage_problem = build_stage_problem(probData, t, scenTree[t][scen_ind], spatial_structure[t], cut_lag, x_star[t - 1]);
        optimize!(stage_problem);
        if t <= T-1
            thetaList[t] = value.(stage_problem[:theta]);
        end

        # obtain the decision at time t
        x_star[t] = Dict();
        x_star[t]["x"] = value.(stage_problem[:x]);
        if spatial_structure[t] != Dict()
            x_star[t]["w"] = value.(stage_problem[:w]);
        end
        x_star[t]["z"] = value.(stage_problem[:z]);
        # calculate the current stage cost
        push!(VList, value(stage_problem[:stage_cost]));
    end

    return sum(VList), x_star, thetaList, VList;
end

function gen_all_scen(T, scenTree)
    # generate all scenarios in the scenario tree
    lists = [[item.index for item in scenTree[t]] for t in 1:T];
    combinations = [];
    for item in Iterators.product(lists...)
        push!(combinations, item);
    end
    return combinations;
end

function sample_scen(T, scenTree, M)
    # generate M scenarios
    scenList = [];
    for m in 1:M
        scen = [1];
        for t in 2:T
            previous_scen = scen[t-1];
            scen_ind = generate_scenario_index(scenTree, previous_scen, t);
            push!(scen, scen_ind)
        end
        push!(scenList, scen);
    end
    return scenList;
end

function eval_sddLp_all(probData, T, scenTree, spatial_structure, cut_lag)
    # evaluate the SDDLP solution
    x_star = Dict();
    x0 = Dict("x" => Dict());
    for i in probData.I
        x0["x"][i] = 0.0;
    end

    first_stage_problem = build_stage_problem(probData, 1, scenTree[1][1], spatial_structure[1], cut_lag, x0);
    optimize!(first_stage_problem);
    # obtain the lower bound
    Vstar = objective_value(first_stage_problem);
    V1 = value(first_stage_problem[:stage_cost]);
    # obtain the first-stage decision
    x_star[0] = Dict();
    x_star[0]["x"] = value.(first_stage_problem[:x]);
    if spatial_structure[1] != Dict()
        x_star[0]["w"] = value.(first_stage_problem[:w]);
    end
    x_star[0]["z"] = value.(first_stage_problem[:z]);

    VbarList = [];
    scenList = [];
    thetaList = Dict();
    thetaList[1] = value.(first_stage_problem[:theta]);

    scenList = gen_all_scen(T, scenTree);

    eval_results = pmap(m -> forward_pass_path(probData, T, scenTree, scenList[m], spatial_structure, cut_lag, x_star[0], thetaList), 1:length(scenList));
    VbarList = [eval_results[m][1] + V1 for m in 1:length(scenList)];
    probList = [];
    for m in eachindex(scenList)
        x_star[m] = eval_results[m][2];
        prob_m = 1.0;
        # calculate the probability of the scenario path
        for t in 1:T-1
            node = scenTree[t][scenList[m][t]];
            probability_changed = Dict();
            for ω in node.successor
                for k in eachindex(probData.K)
                    # calculate the probability after the marketing strategy
                    probList_inner = [];
                    for i in probData.I
                        if node.sign[ω][i] == 1
                            if probData.K[k][i] == 1
                                push!(probList_inner, probData.p_up[i]);
                            else
                                push!(probList_inner, probData.p_norm[i]);
                            end
                        else
                            if probData.K[k][i] == 1
                                push!(probList_inner, probData.p_down[i]);
                            else
                                push!(probList_inner, probData.p_norm[i]);
                            end
                        end
                    end
                    probability_changed[ω,k] = prod(probList_inner);
                end
            end
            prob_m *= sum(probability_changed[scenList[m][t+1],k] * x_star[m][t]["z"][k] for k in eachindex(probData.K));
        end
        push!(probList, prob_m);
    end

    return Vstar, VbarList, probList, x_star, scenList;
end

function forward_pass_sim(probData, T, scenTree, spatial_structure, cut_lag, x_star_0, m)
    # generate a single scenario path
    VList = [];
    # generate the scenario path
    x_star = Dict();
    x_star[1] = x_star_0;
    current_scen = 1;
    scenList = [current_scen];

    # obtain the scenario path solutions
    for t in 2:T
        # obtain the scenario index at time t
        node = scenTree[t-1][current_scen];
        prob_t = Float64[];
        for ω in node.successor
            k_star = 0;
            for k in eachindex(probData.K)
                if abs(x_star[t-1]["z"][k] - 1) < 1e-6
                    k_star = k;
                    break;
                end
            end
            # calculate the probability after the marketing strategy
            probList_inner = [];
            for i in probData.I
                if node.sign[ω][i] == 1
                    if probData.K[k_star][i] == 1
                        push!(probList_inner, probData.p_up[i]);
                    else
                        push!(probList_inner, probData.p_norm[i]);
                    end
                else
                    if probData.K[k_star][i] == 1
                        push!(probList_inner, probData.p_down[i]);
                    else
                        push!(probList_inner, probData.p_norm[i]);
                    end
                end
            end
            push!(prob_t, round(prod(probList_inner), digits=6));
        end
        cat_dist = Categorical(prob_t);
        scen_ind = rand(cat_dist);
        current_scen = scen_ind;
        push!(scenList, current_scen);

        # build the stage problem
        stage_problem = build_stage_problem(probData, t, scenTree[t][scen_ind], spatial_structure[t], cut_lag, x_star[t - 1]);
        optimize!(stage_problem);

        # obtain the decision at time t
        x_star[t] = Dict();
        x_star[t]["x"] = value.(stage_problem[:x]);
        if spatial_structure[t] != Dict()
            x_star[t]["w"] = value.(stage_problem[:w]);
        end
        x_star[t]["z"] = value.(stage_problem[:z]);
        # calculate the current stage cost
        push!(VList, value(stage_problem[:stage_cost]));
    end
 
    return sum(VList), x_star, scenList;
end

function eval_sddLp_path(probData, T, scenTree, spatial_structure, cut_lag, M)
    # evaluate the SDDLP solution
    x_star = Dict();
    x0 = Dict("x" => Dict());
    for i in probData.I
        x0["x"][i] = 0.0;
    end

    first_stage_problem = build_stage_problem(probData, 1, scenTree[1][1], spatial_structure[1], cut_lag, x0);
    optimize!(first_stage_problem);
    # obtain the lower bound
    Vstar = objective_value(first_stage_problem);
    V1 = value(first_stage_problem[:stage_cost]);
    # obtain the first-stage decision
    x_star[0] = Dict();
    x_star[0]["x"] = value.(first_stage_problem[:x]);
    if spatial_structure[1] != Dict()
        x_star[0]["w"] = value.(first_stage_problem[:w]);
    end
    x_star[0]["z"] = value.(first_stage_problem[:z]);
    
    # sampling should be integrated with the forward pass
    sim_results = pmap(m -> forward_pass_sim(probData, T, scenTree, spatial_structure, cut_lag, x_star[0],m), 1:M);
    VbarList = [sim_results[m][1] + V1 for m in 1:M];
    scenList = [sim_results[m][3] for m in 1:M];
    return Vstar, VbarList, scenList;
end