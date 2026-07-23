# Julia implementation of the stochastic server location problem with Lagrangian cuts
using Distributed;
addprocs(10);
@everywhere using JuMP, Gurobi, LinearAlgebra, Random;
@everywhere using JLD, HDF5;
@everywhere const GUROBI_ENV = Gurobi.Env();

# Solver used by the level-set lower-bound, alpha, and next-pi models.
# Gurobi is the default; set this to :COPT to use the optional COPT backend.
levelset_solver = :Gurobi;
if levelset_solver == :COPT
    @everywhere using COPT;
end

@everywhere include("./SSLP_functions.jl");

##--------------------------------------------------------------------------------------------------------
# input the parameters
omega = 200;          # number of scenarios
norm_option = 1;     # 0 represents the L2 norm, 1 represents the L1 norm

J_len = 30;          # number of servers
I_len = 50;          # number of customers
c = round.(rand(J_len) .* 40 .+ 40, digits=4);
v = 10;              # maximum number of servers
u = 120;             # capacity of each server
d = round.(rand(I_len, J_len) .* 25, digits=4);
r = v * u / sum(maximum(d[i, :]) for i in 1:I_len);
h = rand(0:1, omega, I_len);
q = ones(omega, I_len, J_len);
q0 = ones(omega, J_len) .* 1000;

LB = -Inf;
UB = Inf;
cut_inherit = true;
iter_limit = 50;
time_list = [];
LB_List = [];
UB_List = [];

lambda_level = 0.5;
mu_level = 0.6;
iter_bool = true;
iter_num = 1;

# initialize the dictionary to store the Lagrangian cuts for the subproblems' convex envelope
cut_Dict = Dict();
for o in 1:omega
    cut_Dict[o] = [];
end

# solve the problem via the extensive form
extensive_prob = build_extensive_form(omega, c, v, u, d, h, q0, q);
optimize!(extensive_prob);
x_opt_value = value.(extensive_prob[:x]);
opt_value = objective_value(extensive_prob);

# build the master problem
master_prob = build_masterproblem(omega, c, v);
x_best = zeros(J_len);

while iter_bool
    start_time = time();

    # solve the master problem
    optimize!(master_prob);
    LB = objective_value(master_prob);

    # obtain the master solution and lower-bound values
    x_value = value.(master_prob[:x]);
    theta_values = value.(master_prob[:theta]);

    # iterate over the subproblems in parallel
    V_bar = c'x_value;
    sub_opt_results = pmap(
        o -> sub_routine(
            o, c, u, v, d, h, q0, q, x_value,
            lambda_level, mu_level, norm_option, 1e-3, cut_Dict, levelset_solver
        ),
        1:omega
    );

    for o_ind in 1:omega
        o = sub_opt_results[o_ind][1];
        sub_value_o = sub_opt_results[o_ind][2];
        pi_value_o = sub_opt_results[o_ind][3];
        v_value_o = sub_opt_results[o_ind][4];
        cutList_o = sub_opt_results[o_ind][5];

        # update the evaluation at the current solution
        V_bar += sub_value_o / omega;

        # update the master problem with a violated Lagrangian cut
        if v_value_o + pi_value_o'x_value > theta_values[o]
            @constraint(master_prob,
                v_value_o +
                sum(pi_value_o[j] * master_prob[:x][j] for j in 1:J_len) <=
                master_prob[:theta][o]);
        end

        # update the inner cuts
        if cut_inherit
            cut_Dict[o] = cutList_o;
        else
            cut_Dict[o] = [];
        end
    end

    end_time = time();
    push!(time_list, end_time - start_time);

    if V_bar < UB
        UB = V_bar;
        # record the best solution
        for j in 1:J_len
            x_best[j] = x_value[j];
        end
    end

    # check the stopping criterion
    if (abs((UB - LB) / UB) < 1e-2) || (iter_num >= iter_limit)
        iter_bool = false;
    end

    iter_num += 1;
    push!(LB_List, LB);
    push!(UB_List, UB);
end

time_list_true = deepcopy(time_list);
LB_List_true = deepcopy(LB_List);
UB_List_true = deepcopy(UB_List);

#--------------------------------------------------------------------------------------------------------
# test the false option for inherit
norm_option = 1;     # 0 represents the L2 norm, 1 represents the L1 norm
LB = -Inf;
UB = Inf;
cut_inherit = false;
iter_limit = 50;
time_list = [];
LB_List = [];
UB_List = [];

lambda_level = 0.5;
mu_level = 0.6;
iter_bool = true;
iter_num = 1;

# initialize the dictionary to store the Lagrangian cuts for the subproblems' convex envelope
cut_Dict = Dict();
for o in 1:omega
    cut_Dict[o] = [];
end

# build the master problem
master_prob = build_masterproblem(omega, c, v);
x_best = zeros(J_len);

while iter_bool
    start_time = time();

    # solve the master problem
    optimize!(master_prob);
    LB = objective_value(master_prob);

    # obtain the master solution and lower-bound values
    x_value = value.(master_prob[:x]);
    theta_values = value.(master_prob[:theta]);

    # iterate over the subproblems in parallel
    V_bar = c'x_value;
    sub_opt_results = pmap(
        o -> sub_routine(
            o, c, u, v, d, h, q0, q, x_value,
            lambda_level, mu_level, norm_option, 1e-3, cut_Dict, levelset_solver
        ),
        1:omega
    );

    for o_ind in 1:omega
        o = sub_opt_results[o_ind][1];
        sub_value_o = sub_opt_results[o_ind][2];
        pi_value_o = sub_opt_results[o_ind][3];
        v_value_o = sub_opt_results[o_ind][4];
        cutList_o = sub_opt_results[o_ind][5];

        # update the evaluation at the current solution
        V_bar += sub_value_o / omega;

        # update the master problem with a violated Lagrangian cut
        if v_value_o + pi_value_o'x_value > theta_values[o]
            @constraint(master_prob,
                v_value_o +
                sum(pi_value_o[j] * master_prob[:x][j] for j in 1:J_len) <=
                master_prob[:theta][o]);
        end

        # update the inner cuts
        if cut_inherit
            cut_Dict[o] = cutList_o;
        else
            cut_Dict[o] = [];
        end
    end

    end_time = time();
    push!(time_list, end_time - start_time);

    if V_bar < UB
        UB = V_bar;
        # record the best solution
        for j in 1:J_len
            x_best[j] = x_value[j];
        end
    end

    # check the stopping criterion
    if (abs((UB - LB) / UB) < 1e-2) || (iter_num >= iter_limit)
        iter_bool = false;
    end

    iter_num += 1;
    push!(LB_List, LB);
    push!(UB_List, UB);
end

time_list_false = deepcopy(time_list);
LB_List_false = deepcopy(LB_List);
UB_List_false = deepcopy(UB_List);

test_instance = (omega, c, v, u, d, h, q0, q);
save("SSLP_results.jld", "test_instance", test_instance, "time_list_true", time_list_true, "LB_List_true", LB_List_true, "UB_List_true", UB_List_true, "time_list_false", time_list_false, "LB_List_false", LB_List_false, "UB_List_false", UB_List_false);

# --------------------------------------------------------------------------------------------------------
# test the L2 norm option
omega = 200;          # number of scenarios
norm_option = 0;     # 0 represents the L2 norm, 1 represents the L1 norm

LB = -Inf;
UB = Inf;
cut_inherit = false;
iter_limit = 50;
time_list = [];
LB_List = [];
UB_List = [];

lambda_level = 0.5;
mu_level = 0.6;
iter_bool = true;
iter_num = 1;

# initialize the dictionary to store the Lagrangian cuts for the subproblems' convex envelope
cut_Dict = Dict();
for o in 1:omega
    cut_Dict[o] = [];
end

# build the master problem
master_prob = build_masterproblem(omega, c, v);
x_best = zeros(J_len);

while iter_bool
    start_time = time();

    # solve the master problem
    optimize!(master_prob);
    LB = objective_value(master_prob);

    # obtain the master solution and lower-bound values
    x_value = value.(master_prob[:x]);
    theta_values = value.(master_prob[:theta]);

    # iterate over the subproblems in parallel
    V_bar = c'x_value;
    sub_opt_results = pmap(
        o -> sub_routine(
            o, c, u, v, d, h, q0, q, x_value,
            lambda_level, mu_level, norm_option, 1e-2, cut_Dict, levelset_solver
        ),
        1:omega
    );

    for o_ind in 1:omega
        o = sub_opt_results[o_ind][1];
        sub_value_o = sub_opt_results[o_ind][2];
        pi_value_o = sub_opt_results[o_ind][3];
        v_value_o = sub_opt_results[o_ind][4];
        cutList_o = sub_opt_results[o_ind][5];

        # update the evaluation at the current solution
        V_bar += sub_value_o / omega;

        # update the master problem with a violated Lagrangian cut
        if v_value_o + pi_value_o'x_value > theta_values[o]
            @constraint(master_prob,
                v_value_o +
                sum(pi_value_o[j] * master_prob[:x][j] for j in 1:J_len) <=
                master_prob[:theta][o]);
        end

        # update the inner cuts
        if cut_inherit
            cut_Dict[o] = cutList_o;
        else
            cut_Dict[o] = [];
        end
    end

    end_time = time();
    push!(time_list, end_time - start_time);

    if V_bar < UB
        UB = V_bar;
        # record the best solution
        for j in 1:J_len
            x_best[j] = x_value[j];
        end
    end

    # check the stopping criterion
    if (abs((UB - LB) / UB) < 1e-2) || (iter_num >= iter_limit)
        iter_bool = false;
    end

    iter_num += 1;
    push!(LB_List, LB);
    push!(UB_List, UB);
end

time_list_false_L2 = deepcopy(time_list);
LB_List_false_L2 = deepcopy(LB_List);
UB_List_false_L2 = deepcopy(UB_List);

# --------------------------------------------------------------------------------------------------------
# test the inherit option for L2 norm
LB = -Inf;
UB = Inf;
cut_inherit = true;
iter_limit = 50;
time_list = [];
LB_List = [];
UB_List = [];

lambda_level = 0.5;
mu_level = 0.6;
iter_bool = true;
iter_num = 1;

# initialize the dictionary to store the Lagrangian cuts for the subproblems' convex envelope
cut_Dict = Dict();
for o in 1:omega
    cut_Dict[o] = [];
end

# build the master problem
master_prob = build_masterproblem(omega, c, v);
x_best = zeros(J_len);

while iter_bool
    start_time = time();

    # solve the master problem
    optimize!(master_prob);
    LB = objective_value(master_prob);

    # obtain the master solution and lower-bound values
    x_value = value.(master_prob[:x]);
    theta_values = value.(master_prob[:theta]);

    # iterate over the subproblems in parallel
    V_bar = c'x_value;
    sub_opt_results = pmap(
        o -> sub_routine(
            o, c, u, v, d, h, q0, q, x_value,
            lambda_level, mu_level, norm_option, 1e-2, cut_Dict, levelset_solver
        ),
        1:omega
    );

    for o_ind in 1:omega
        o = sub_opt_results[o_ind][1];
        sub_value_o = sub_opt_results[o_ind][2];
        pi_value_o = sub_opt_results[o_ind][3];
        v_value_o = sub_opt_results[o_ind][4];
        cutList_o = sub_opt_results[o_ind][5];

        # update the evaluation at the current solution
        V_bar += sub_value_o / omega;

        # update the master problem with a violated Lagrangian cut
        if v_value_o + pi_value_o'x_value > theta_values[o]
            @constraint(master_prob,
                v_value_o +
                sum(pi_value_o[j] * master_prob[:x][j] for j in 1:J_len) <=
                master_prob[:theta][o]);
        end

        # update the inner cuts
        if cut_inherit
            cut_Dict[o] = cutList_o;
        else
            cut_Dict[o] = [];
        end
    end

    end_time = time();
    push!(time_list, end_time - start_time);

    if V_bar < UB
        UB = V_bar;
        # record the best solution
        for j in 1:J_len
            x_best[j] = x_value[j];
        end
    end

    # check the stopping criterion
    if (abs((UB - LB) / UB) < 1e-2) || (iter_num >= iter_limit)
        iter_bool = false;
    end

    iter_num += 1;
    push!(LB_List, LB);
    push!(UB_List, UB);
end

time_list_true_L2 = deepcopy(time_list);
LB_List_true_L2 = deepcopy(LB_List);
UB_List_true_L2 = deepcopy(UB_List);

