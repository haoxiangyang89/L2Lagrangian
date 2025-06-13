# Julia implementation of two-stage stochastic mip with Lagrangian cuts
using Distributed;
addprocs(4);
@everywhere using JuMP, Gurobi, LinearAlgebra, Random;
@everywhere const GUROBI_ENV = Gurobi.Env();
@everywhere include("./SIP_functions.jl");

##--------------------------------------------------------------------------------------------------------
# input the parameters
omega = 50          # number of scenarios
J_len = 2        # number of first-stage decision variables
I_len = 4        # number of second-stage decision variables
h = round.(rand(omega, J_len) * 10 .+ 5, digits=5);  # random right-hand side values
h = reduce(vcat, h');
T = [[1,0],[0,1]];  # coefficients of first-stage decision variables in second-stage constraints
y_option = 0;
norm_option = 1;
W = [[2,3,4,5],[6,1,3,2]];
c = [-16,-19,-23,-28];
c1 = [-3/2,-4];
LB = -Inf;
UB = Inf;

lambda_level = 0.5;
mu_level = 0.6;
iter_bool = true;

# initialize the dictionary to store the Lagrangian cuts for the subproblems' convex envelope
cut_Dict = Dict();
for o in 1:omega
    cut_Dict[o] = [];
end

# solve the problem via the extensive form
extensive_prob = build_extensive_form(omega, h, T, W, c, c1, y_option);
optimize!(extensive_prob);
x_value = value.(extensive_prob[:x]);
opt_value = objective_value(extensive_prob);

# build the master problem
master_prob = build_masterproblem(omega, c1);
x_best = zeros(J_len);

while iter_bool
    # solve the master problem
    optimize!(master_prob);
    LB = objective_value(master_prob);

    # obtain the master soluton/optimal value & update the lower bound
    x_value = value.(master_prob[:x]);
    theta_values = value.(master_prob[:theta]);

    # iterate over the subproblem
    V_bar = c1'x_value;

    sub_opt_results = pmap(o -> sub_routine(o, h, T, W, c, y_option, x_value, lambda_level, mu_level, norm_option, 1e-3, cut_Dict), 1:omega);

    for o_ind in 1:omega
        o = sub_opt_results[o_ind][1];
        sub_value_o = sub_opt_results[o_ind][2];
        pi_value_o = sub_opt_results[o_ind][3];
        v_value_o = sub_opt_results[o_ind][4];
        cutList_o = sub_opt_results[o_ind][5];

        # update the evaluation at the current solution
        V_bar += sub_value_o / omega;

        # update the master problem with the Lagrangian cuts
        if v_value_o + pi_value_o'x_value > theta_values[o]
            @constraint(master_prob, v_value_o + sum(pi_value_o[j] * master_prob[:x][j] for j in 1:J_len) <= master_prob[:theta][o]);
        end
            
        # update the inner cuts
        cut_Dict[o] = cutList_o;
    end

    if V_bar < UB
        UB = V_bar;
        # record the best solution
        for j in 1:J_len
            x_best[j] = x_value[j];
        end
    end
    
    # check the stopping criterion
    if abs((UB - LB)/UB) < 1e-2
        iter_bool = false;
    end
end
