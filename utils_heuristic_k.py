"""
utils_heuristic_k.py — heuristic top-k scenario selection ("Heur-k").

[ADDED: ML4SDDP] This whole file is new. It implements the history-based
scenario-selection heuristic from our ML4SDDP project (paper name "Heur-k").
Idea: in each outer iteration of the Lagrangian cutting-plane algorithm,
instead of solving the (expensive) Lagrangian dual for EVERY scenario, rank
the scenarios by a heuristic score and only solve the top k = ceil(kappa*|Omega|):

    score_omega = L_tilde_omega - theta_hat_omega

where
    L_tilde_omega   = most recently OBSERVED recourse value Q_omega(x_hat)
                      (0.0 before the first observation — "lazy" protocol:
                      no recourse MIP is ever solved just to compute a score),
    theta_hat_omega = current master value of theta_omega.

A large score means the master approximation theta_omega is far below the last
observed recourse cost, i.e. the scenario's Lagrangian cut is likely to be
violated — those scenarios are the most valuable ones to process.

Safety schedule ("full iterations"): every `safety_interval`-th outer iteration
solves ALL scenarios. These are the only iterations where
  (a) V_bar = a^T x_hat + (1/|Omega|) * sum_omega Q_omega(x_hat) is a valid
      upper bound (every recourse problem evaluated at the same x_hat), and
  (b) the stopping criterion gap = |UB - LB| / |UB| may be certified.
"""

import numpy as np


class HeuristicKSelector:
    """History-based top-k scenario selector (Heur-k).

    Tracks the last observed recourse value L_tilde_omega of every scenario
    and, on selective iterations, returns the k scenarios with the largest
    gap proxy L_tilde_omega - theta_hat_omega.
    """

    def __init__(self, num_scenarios, solve_ratio=0.5, safety_interval=5):
        # num_scenarios   - |Omega|
        # solve_ratio     - kappa in (0, 1]: fraction of scenarios solved per
        #                   selective iteration (0.5 = top 50%)
        # safety_interval - N_full: every N_full-th outer iteration is a full
        #                   iteration (all scenarios solved)
        if not (0.0 < solve_ratio <= 1.0):
            raise ValueError("solve_ratio (kappa) must be in (0, 1], got {}".format(solve_ratio))
        if int(safety_interval) < 1:
            raise ValueError("safety_interval (N_full) must be >= 1, got {}".format(safety_interval))
        self.num_scenarios = num_scenarios
        self.solve_ratio = solve_ratio
        self.safety_interval = safety_interval
        # last observed recourse value per scenario;
        # 0.0 before the first observation (iteration-1 default)
        self.L_tilde = np.zeros(num_scenarios)

    def is_full_iteration(self, iteration):
        # full (safety) iteration: solve ALL scenarios; upper bound and
        # termination are only certified here.
        # NOTE: iteration 1 is also treated as full so that every scenario's
        # bundle cuts and L_tilde are initialized at the first master point
        # before any selective skipping — a scenario whose first Lagrangian
        # dual call happens at a later x_hat with an EMPTY bundle makes the
        # Level Set inner solver numerically unstable (the lower-bound problem
        # is then only bounded by the pi box, so the level constraint can
        # become infeasible).
        return iteration == 1 or iteration % self.safety_interval == 0

    def compute_scores(self, theta_hat_values):
        # score_omega = L_tilde_omega - theta_hat_omega (larger = more urgent)
        return self.L_tilde - np.asarray(theta_hat_values, dtype=float)

    def select_scenarios(self, iteration, theta_hat_values):
        # return the set of scenario indices to solve in this outer iteration
        if self.is_full_iteration(iteration):
            return set(range(self.num_scenarios))
        scores = self.compute_scores(theta_hat_values)
        # k = ceil(kappa * |Omega|); stable sort breaks score ties by lower
        # scenario index (deterministic)
        k = max(1, int(np.ceil(self.solve_ratio * self.num_scenarios)))
        top_k_indices = np.argsort(-scores, kind="stable")[:k]
        return set(top_k_indices.tolist())

    def update_observation(self, o, L_value):
        # record the recourse value observed for scenario o at the current x_hat
        self.L_tilde[o] = float(L_value)
