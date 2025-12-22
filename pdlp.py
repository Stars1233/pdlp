import torch

"""
Consider the LP.

  min_x   c^T x
  s.t.    Gx >= h
          Ax  = b
          l <= x <= u

G: (m1,n), A: (m2,n), c: (n,), h: (m1,), b: (m2,)
K = [G; A]  (m1+m2, n)
q = [h; b]  (m1+m2,)
X = {x : l <= x <= u}
Y = {y : y[:m1] >= 0} (y[m1:] free)
"""


def solve(
    G: torch.Tensor,
    A: torch.Tensor,
    c: torch.Tensor,
    h: torch.Tensor,
    b: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    *,
    MAX_OUTER_ITERS: int = 100,
    MAX_INNER_ITERS: int = 100,
    MAX_BACKTRACK: int = 50,
    theta: float = 0.5,
):
    # -----------------------------
    # Shape checks / setup
    # -----------------------------
    assert G.ndim == 2 and A.ndim == 2
    assert c.ndim == h.ndim == b.ndim == l.ndim == u.ndim == 1
    assert G.shape[0] == h.shape[0]
    assert A.shape[0] == b.shape[0]
    assert G.shape[1] == A.shape[1] == c.shape[0] == l.shape[0] == u.shape[0]

    device = c.device
    dtype = c.dtype
    eps_zero = 1e-12

    m1, n = G.shape
    m2 = A.shape[0]
    m = m1 + m2

    # Stack constraints
    K = torch.cat([G, A], dim=0) # (m, n)
    q = torch.cat([h, b], dim=0) # (m,)

    # -----------------------------
    # Projections
    # -----------------------------
    def proj_X(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, l, u)

    def proj_Y(y: torch.Tensor) -> torch.Tensor:
        y2 = y.clone()
        y2[:m1] = torch.clamp(y2[:m1], min=0.0)
        return y2

    # -----------------------------
    # Helper functions
    # -----------------------------
    @torch.no_grad()
    def initialize_primal_weight() -> torch.Tensor:
        return (torch.linalg.norm(c) / torch.linalg.norm(q)).clamp_min(eps_zero)

    @torch.no_grad()
    def primal_weight_update(
        x_new: torch.Tensor, y_new: torch.Tensor,
        x_old: torch.Tensor, y_old: torch.Tensor,
        w_old: torch.Tensor,
    ) -> torch.Tensor:
        dx = torch.linalg.norm(x_new - x_old)
        dy = torch.linalg.norm(y_new - y_old)

        if (dx > eps_zero) and (dy > eps_zero):
            ratio = (dy / dx).clamp_min(eps_zero)
            w_new = torch.exp(theta * torch.log(ratio) + (1.0 - theta) * w_old)
            return w_new.clamp_min(eps_zero)
        return w_old

    def get_restart_candidate(
        x_cur: torch.Tensor, y_cur: torch.Tensor,
        x_avg: torch.Tensor, y_avg: torch.Tensor,
    ):
        # simplest valid: restart from averaged iterate
        # TODO
        return x_avg, y_avg

    def should_restart(
        x_cur: torch.Tensor, y_cur: torch.Tensor,
        x_avg: torch.Tensor, y_avg: torch.Tensor,
        x_cand: torch.Tensor, y_cand: torch.Tensor,
        w_val: torch.Tensor,
        t: int, k: int,
    ) -> bool:
        # simplest: never early restart; only restart when inner loop budget ends
        # TODO
        return False

    def termination_criteria(x_cur: torch.Tensor, y_cur: torch.Tensor, k: int) -> bool:
        return False

    @torch.no_grad()
    def adaptive_step_pdhg(
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        eta_hat: torch.Tensor,
        k: int,
    ):
        """
        One PDHG step
        """
        eta = eta_hat

        kp1 = k + 1
        fac1 = 1.0 - (kp1 ** -0.3)
        fac2 = 1.0 + (kp1 ** -0.6)

        for _ in range(MAX_BACKTRACK):
            x_p = proj_X(x - (eta / w) * (c - (K.T @ y)))
            y_p = proj_Y(y + (eta * w) * (q - K @ (2.0 * x_p - x)))

            dx = x_p - x
            dy = y_p - y
            num = w * (dx @ dx) + (1.0 / w) * (dy @ dy)
            denom = 2.0 * (dy @ (K @ dx))
            eta_bar =  num / denom.clamp_min(eps_zero)

            eta_p = min(fac1*eta_bar, fac2*eta)

            if eta <= eta_bar:
                return x_p, y_p, eta, eta_p

            eta = eta_p

        # if we run out of backtracking steps, return last proposal
        return x_p, y_p, eta, eta_p


    # -----------------------------
    # Main Algorithm
    # -----------------------------

    # Initializations

    # safe average of l and u to initialize x
    finite = torch.isfinite(l) & torch.isfinite(u)
    x_mid = 0.5 * (l + u)
    x0 = torch.where(finite, x_mid, proj_X(torch.zeros(n, device=device, dtype=dtype)))
    x0 = proj_X(x0)

    # trivially in Y
    y0 = torch.zeros(m, device=device, dtype=dtype)

    # step size 1/||K||_inf
    K_inf = K.abs().sum(dim=1).max().clamp_min(eps_zero)
    eta_hat = (1.0 / K_inf).to(device=device, dtype=dtype)

    w = initialize_primal_weight()

    x, y = x0.clone(), y0.clone() # current iterate
    x_prev, y_prev = x.clone(), y.clone() # past iterate

    k_global = 0 # global step counter

    for n_outer in range(MAX_OUTER_ITERS):
        # reset averaging at start of each outer loop
        eta_sum = 0.0
        x_bar, y_bar = x.clone(), y.clone()

        for n_inner in range(MAX_INNER_ITERS):
            if termination_criteria(x, y, k_global):
                return x, y

            x, y, eta_used, eta_hat = adaptive_step_pdhg(x, y, w, eta_hat, k_global)

            # online weighted average
            eta_sum += float(eta_used)
            alpha = float(eta_used) / eta_sum
            x_bar = x_bar + alpha * (x - x_bar)
            y_bar = y_bar + alpha * (y - y_bar)

            # restart candidate
            x_c, y_c = get_restart_candidate(x, y, x_bar, y_bar)

            k_global += 1

            if should_restart(x, y, x_bar, y_bar, x_c, y_c, w, t, k_global):
                break

        # restart from candidate
        x, y = x_c, y_c

        # primal weight update
        w = primal_weight_update(x, y, x_prev, y_prev, w)

        # store previous restart start
        x_prev, y_prev = x.clone(), y.clone()

    return x, y