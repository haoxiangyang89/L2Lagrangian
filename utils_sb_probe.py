"""
utils_sb_probe.py — cheap certified Lagrangian multipliers ("sb" and "probe").

[ADDED: ML4SDDP] This whole file is new. It ports two multiplier oracles from
our ML4SDDP project. For one scenario at the master point x_hat, both build a
cheap multiplier pi and accept it only if ONE Lagrangian MIP certifies that its
cut is tight at x_hat; otherwise the caller falls back to the original
level-set solve_lag_dual, which is left unchanged:

    sb     strengthened-Benders multiplier: the dual of the constraint z = x_hat
           in the LP relaxation of the Lagrangian subproblem, projected onto the
           sign cone K(x_hat) = {pi : d_j * pi_j >= 0}, d = 2 * x_hat - 1
           (1 LP + 1 certification MIP)
    probe  flip-probe multiplier pi = d * m with
               m_j = [Q(x_hat) - Q(x_hat with coordinate j flipped)]^+,
           Q(z) = recourse value with the copy variable fixed to z; a coordinate
           whose single flip is infeasible takes the best pair-flip gain net of
           its partner's weight (dim(x) fixed-z MIPs, cached per scenario for
           the whole run, + 1 certification MIP)

Conventions (the same as the level set in the main scripts):
    V(pi) = min g(y) - pi^T z over the Lagrangian subproblem,
    Lagrangian cut: theta >= V(pi) + pi^T x,
    bundle cut (coeffs, intercept) = (-z, g(y)) of a feasible (z, y), meaning
    V(pi) <= intercept + coeffs^T pi.

Certification: solve the Lagrangian MIP at pi and take its dual bound
V_lb = ObjBound <= V(pi) as the cut intercept, so the cut is valid by weak
duality whatever pi is. Accept iff the cut is tight at x_hat:
    L - (V_lb + pi^T x_hat) <= CERT_TOL_REL * max(1, |L|),
where L is the recourse value the main loop computed at x_hat. L comes from a
MIP solved to Gurobi's default MIPGap 1e-4, so it may exceed Q(x_hat) by
1e-4 * |L|. Each MIP may use a quarter of the tolerance (as in ML4SDDP), hence
CERT_TOL_REL = 1e-4 / 0.25 = 4e-4, far below the 1% gap of the outer loop.

Only for binary first-stage x and binary copy variables z (SSLP and the binary
DCAP variant): the sign cone, the flips and the zero-gap certificate all need
it. Every probe and certification incumbent is added to the scenario's bundle
cut list, so a fallback level set starts from it.
"""

import time

import numpy as np
from gurobipy import GRB

# tightness a certified cut must meet at x_hat, relative to max(1, |L|)
CERT_TOL_REL = 4e-4
# share of that tolerance the certification MIP may leave as its own gap
MIP_SLACK_FRAC = 0.25


def sign_vector(x_hat):
    # d = 2 x_hat - 1: +1 where x_hat = 1, -1 where x_hat = 0
    return 2.0 * np.round(x_hat) - 1.0


def project_sign_cone(pi, d):
    # projection onto K(x_hat) = {pi : d_j pi_j >= 0}: zero the entries of the wrong sign
    return np.where(d * pi < 0, 0.0, pi)


def z_variables(model, shape):
    # copy variables z of a Lagrangian model in np.ndindex(shape) order,
    # named "z[j]" (SSLP) or "z[j,t]" (DCAP) as in the main scripts
    var_by_name = {var.VarName: var for var in model.getVars()}
    return [var_by_name["z[{}]".format(",".join(str(i) for i in idx))] for idx in np.ndindex(shape)]


def add_bundle_cut(cut_list, z, value):
    # add the bundle cut V(pi) <= value - z^T pi of a feasible (z, y) with g(y) = value;
    # a cut with the same z only lowers the existing intercept, so repeated
    # probes do not grow the list
    coeff = -np.asarray(z, dtype=float)
    for k in range(len(cut_list)):
        if np.array_equal(cut_list[k][0], coeff):
            if value < cut_list[k][1]:
                cut_list[k] = (coeff, value)
            return
    cut_list.append((coeff, value))


def sb_multiplier(build_lag, x_hat):
    # strengthened-Benders multiplier: dual of z = x_hat in the LP relaxation of the
    # Lagrangian subproblem at pi = 0, projected onto K(x_hat); None if the LP fails
    relaxed = build_lag(np.zeros(x_hat.shape)).relax()
    relaxed.Params.OutputFlag = 0
    fix_cons = []
    for var, x_j in zip(z_variables(relaxed, x_hat.shape), np.round(x_hat).flatten()):
        # free bounds, so the whole sensitivity sits in the dual of z = x_hat
        var.LB, var.UB = -GRB.INFINITY, GRB.INFINITY
        fix_cons.append(relaxed.addConstr(var == x_j))
    relaxed.optimize()
    if relaxed.Status != GRB.OPTIMAL:
        return None
    pi = np.array([con.Pi for con in fix_cons]).reshape(x_hat.shape)
    return project_sign_cone(pi, sign_vector(x_hat))


class FixedZEvaluator:
    """Q(z) = min{g(y) : (z, y) feasible}, the Lagrangian model at pi = 0 with z fixed; None if z is infeasible.

    One per scenario for the whole run: Q does not change across outer
    iterations, so every value is cached per z.
    """

    def __init__(self, build_lag, shape):
        self.model = build_lag(np.zeros(shape))
        self.model.Params.OutputFlag = 0
        self.z = z_variables(self.model, shape)
        self.cache = {}
        self.n_solves = 0

    def __call__(self, z):
        key = np.round(z).astype(np.int8).tobytes()
        if key not in self.cache:
            for var, z_j in zip(self.z, np.round(z).flatten()):
                var.LB = var.UB = z_j
            self.model.optimize()
            self.n_solves += 1
            self.cache[key] = self.model.ObjVal if self.model.Status == GRB.OPTIMAL else None
        return self.cache[key]


def probe_multiplier(qz, x_hat, max_pairs):
    # flip-probe weights m >= 0 (shaped like x_hat) and the probed (z, Q(z)) pairs
    # with flat z; None if Q(x_hat) fails
    x_flat = np.round(x_hat).flatten()
    q0 = qz(x_flat)
    if q0 is None:
        return None
    n = x_flat.size
    m = np.zeros(n)
    single_ok = np.zeros(n, dtype=bool)
    probed = []
    for j in range(n):
        z = x_flat.copy()
        z[j] = 1.0 - z[j]
        q = qz(z)
        if q is None:
            continue
        single_ok[j] = True
        probed.append((z, q))
        m[j] = max(q0 - q, 0.0)
    # a coordinate whose single flip is infeasible: best pair-flip gain net of the
    # partner's weight, at most max_pairs pair probes in total
    budget = max_pairs
    for j in np.where(~single_ok)[0]:
        best = 0.0
        for i in np.where(single_ok)[0]:
            if budget <= 0:
                break
            z = x_flat.copy()
            z[j] = 1.0 - z[j]
            z[i] = 1.0 - z[i]
            q = qz(z)
            budget -= 1
            if q is None:
                continue
            probed.append((z, q))
            best = max(best, q0 - q - m[i])
        m[j] = max(m[j], best)
    return m.reshape(x_hat.shape), probed


def certify(build_lag, pi, x_hat, L_value, cut_list, cert_tol_rel=CERT_TOL_REL):
    # solve the Lagrangian MIP at pi; return (V_lb, tight): V_lb = ObjBound is the
    # intercept of the valid cut theta >= V_lb + pi^T x, tight says whether that
    # cut is within the tolerance of L at x_hat
    tol = cert_tol_rel * max(1.0, abs(L_value))
    model = build_lag(pi)
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = MIP_SLACK_FRAC * tol
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return -np.inf, False
    z = np.array([var.X for var in z_variables(model, x_hat.shape)]).reshape(x_hat.shape)
    # the incumbent (z, y) is feasible: its bundle cut has g(y) = ObjVal + pi^T z
    add_bundle_cut(cut_list, z, model.ObjVal + np.inner(z.flatten(), pi.flatten()))
    v_lb = model.ObjBound
    return v_lb, L_value - (v_lb + np.inner(pi.flatten(), x_hat.flatten())) <= tol


class SBProbeOracle:
    """Cheap certified multiplier ("sb" or "probe") for one scenario per call.

    certified_cut returns (pi_value, v_value) of a certified tight cut, shaped
    like x_value, or None: then the caller runs the original level set.
    """

    def __init__(self, method, max_pairs_factor=4, cert_tol_rel=CERT_TOL_REL):
        # method           - "sb" or "probe"
        # max_pairs_factor - probe: pair-flip probes per call, as a multiple of dim(x)
        # cert_tol_rel     - relative tightness a certified cut must meet at x_hat
        if method not in ("sb", "probe"):
            raise ValueError("method must be 'sb' or 'probe', got {}".format(method))
        self.method = method
        self.max_pairs_factor = max_pairs_factor
        self.cert_tol_rel = cert_tol_rel
        self.fixed_z = {}     # scenario -> FixedZEvaluator (probe)
        self.stats = {"calls": 0, "certified": 0, "fallback": 0, "lp_solves": 0,
                      "probe_mips": 0, "cert_mips": 0, "time": 0.0}

    def certified_cut(self, o, build_lag, x_value, L_value, cut_list):
        # o         - scenario index (key of the probe cache)
        # build_lag - pi -> Gurobi model of scenario o's Lagrangian subproblem
        #             (the main script's build_subproblem_lag with the data bound)
        # x_value   - master solution x_hat, L_value - recourse value at x_hat
        # cut_list  - the scenario's bundle cut list cut_Dict[o]; every probe and
        #             certification incumbent is added to it
        start_time = time.time()
        self.stats["calls"] += 1
        x_hat = np.asarray(x_value, dtype=float)
        pi = None
        if self.method == "sb":
            pi = sb_multiplier(build_lag, x_hat)
            self.stats["lp_solves"] += 1
        else:
            if o not in self.fixed_z:
                self.fixed_z[o] = FixedZEvaluator(build_lag, x_hat.shape)
            qz = self.fixed_z[o]
            n_before = qz.n_solves
            probe = probe_multiplier(qz, x_hat, self.max_pairs_factor * x_hat.size)
            self.stats["probe_mips"] += qz.n_solves - n_before
            if probe is not None:
                m, probed = probe
                for z, q in probed:
                    add_bundle_cut(cut_list, z.reshape(x_hat.shape), q)
                pi = sign_vector(x_hat) * m
        cut = None
        if pi is not None:
            v_lb, tight = certify(build_lag, pi, x_hat, L_value, cut_list, self.cert_tol_rel)
            self.stats["cert_mips"] += 1
            if tight:
                cut = (pi, v_lb)
        self.stats["certified" if cut is not None else "fallback"] += 1
        self.stats["time"] += time.time() - start_time
        return cut

    def summary(self):
        s = self.stats
        return ("Multiplier oracle '{}': {} calls, {} certified, {} fell back to the level set; "
                "{} LPs, {} probe MIPs, {} certification MIPs, {:.2f}s in the oracle").format(
                    self.method, s["calls"], s["certified"], s["fallback"], s["lp_solves"],
                    s["probe_mips"], s["cert_mips"], s["time"])
