using Distributed;
addprocs(8);
@everywhere using JuMP, Gurobi, LinearAlgebra, Random, Statistics;
@everywhere const GUROBI_ENV = Gurobi.Env();
@everywhere include("./multi_def.jl");
@everywhere include("./multi_newsvendor_functions.jl");

##--------------------------------------------------------------------------------------------------------
# input the parameters
I = [1,2,3];                  # set of products
K = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]];   # set of marketing strategies
T = 10;                         # time horizon
b = [2,2,2];                    # purchase cost
s = [8,10,12];                  # selling price
h = [0.1,0.1,0.1];              # holding cost
Capacity = 125.0;               # purchase capacity
p_up = [0.55, 0.55, 0.55];      # upward demand change probability
p_down = [0.45, 0.45, 0.45];    # downward demand change probability
p_norm = [0.5, 0.5, 0.5];       # normal demand change probability
xlb = [0.0, 0.0, 0.0];          # lower bound of first-stage decision variables (inventory)
xub = [1000.0, 1000.0, 1000.0]; # upper bound of first-stage decision variables (inventory)
f = [sum(kitem .* 5) for kitem in K];
probData = problemData(I, K, Dict(i => p_up[i] for i in I), Dict(i => p_norm[i] for i in I), 
    Dict(i => p_down[i] for i in I), Dict(i => s[i] for i in I), Dict(i => b[i] for i in I), 
    Dict(i => h[i] for i in I), Capacity, Dict(i => xlb[i] for i in I), Dict(i => xub[i] for i in I), f);

# build the scenario tree
dList = [[20, 50], [10, 60], [5, 65]];
scenTree = Dict();
# first-stage node is just to build an inventory
node_1 = nodeType(1, 1, 0, Dict(i => 0.0 for i in I), [node_ind for node_ind in 1:8], [0.125 for node_ind in 1:8],
     Dict(node_ind+1 => digits(node_ind, base=2, pad=3) for node_ind in 0:7));
scenTree[1] = Any[node_1,];
for t in 2:T
    scenTree[t] = [];
    if t < T
        for node_ind in 0:7
            # create the node
            node_xi = digits(node_ind, base=2, pad=3);
            d_node = Dict(i => dList[i][node_xi[i] + 1] for i in I);
            node = nodeType(t, node_ind + 1, 0, d_node, [node_ind_t for node_ind_t in 1:8], [0.125 for node_ind_t in 1:8], 
                Dict(node_ind_t+1 => digits(node_ind_t, base=2, pad=3) for node_ind_t in 0:7));
            push!(scenTree[t], node);
        end
    else
        for node_ind in 0:7
            # create the node
            node_xi = digits(node_ind, base=2, pad=3);
            d_node = Dict(i => dList[i][node_xi[i] + 1] for i in I);
            node = nodeType(t, node_ind + 1, 0, d_node, [], [], Dict());
            push!(scenTree[t], node);
        end
    end
end

lambda_level = 0.5;
mu_level = 0.6;
norm_option = 1;
partition_option = 1;
tol = 1e-3;
para_option = 1;
M = 2;
iter_limit = 300;

Vstar, cut_lag, cut_Dict, LB_list, UB_list, time_list, spatial_structure = sddLp_process(probData, T, M, scenTree, lambda_level, mu_level, norm_option, partition_option, tol, iter_limit, 1e-2, para_option);
Vstar_UB, VbarList, probList, x_star, scenList = eval_sddLp_all(probData, T, scenTree, spatial_structure, cut_lag);
Vstar_UB, VbarList, scenList = eval_sddLp_path(probData, T, scenTree, spatial_structure, cut_lag, 1000);
