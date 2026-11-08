"""TriPlast: frozen population core + three forward-only plastic stores (docs/research_plan.md §7.3).

    logits = (W + P + F) u + b,   u = G(z) = g ⊙ (z − λ ⊙ m),   g = (1 − λ) + λ / sqrt(v + ε)

z: standardized value-domain features (population scaler). W, b: population readout (frozen).
G  label-free: per-feature EMA mean m and variance v of the stream windows (rate α); λ ∈ [0, 1] per feature group.
F  fast: on a confirmation (u, y): F ← F + η (e_y − softmax(ℓ)) uᵀ / (‖u‖² + 1)  (normalized delta rule, error-gated);
   F decays by (1 − δ) at every stream window.
P  slow: per activity a spacing counter; a confirmation of y counts if ≥ Δ stream windows passed since y's last
   counted confirmation; at 2 counts, row y moves from F to P: P_y += κ F_y, F_y ← (1 − κ) F_y, counter reset.
θ = {λ (3 groups), α, η, δ, κ, Δ}: 8 dataset-agnostic scalars, meta-learned. With soft=True the two gates are
sigmoids (meta-training); deployment uses the hard gates (soft=False).

Variants (for ablations and ML twins; flags of Theta):
  no_g (λ = 0), no_f (η = 0), no_p (κ = 0), no_spacing (Δ = 0: every confirmation counts), hebbian (F update e_y uᵀ
  instead of the error e_y − softmax(ℓ)), fixed (θ not learned: hand-set values).
  bias (iteration 3): the stores act on [u, 1], so F and P carry a bias column like the delta rule's b += g; the
  constant input belongs to a fourth feature group with λ = 0, and the step normalization ‖[u, 1]‖² equals the
  original ‖u‖² + 1. The batch is augmented by with_bias_input().

The stream of a person: its calibration-pool windows in time order; the first k pool windows of each activity are
confirmations (labelled); all pool windows update G. After the stream the state is frozen and every scored window is
classified (no adaptation on scored windows). A batch holds the persons of one population model (same W, b).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

EPS = 1e-4
HAND_SET = {"lam": 0.0, "alpha": 0.1, "eta": 0.5, "delta": 0.0, "kappa": 0.0, "Delta": 0.0}


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _softplus_inv(y: float) -> float:
    return math.log(math.expm1(max(y, 1e-6)))


class Theta(nn.Module):
    """The 8 plasticity parameters, with constraints, and the variant flags."""

    def __init__(self, *, no_g=False, no_f=False, no_p=False, no_spacing=False, hebbian=False, fixed=False,
                 rehearse: int = 0, pseudo: bool = False, bias: bool = False, init: dict | None = None):
        super().__init__()
        init = {"lam": 0.05, "alpha": 0.1, "eta": 2.0, "delta": 0.01, "kappa": 0.5, "Delta": 3.0, "tau": 0.9,
                "omega": 0.3, **(init or {})}
        self.flags = dict(no_g=no_g, no_f=no_f, no_p=no_p, no_spacing=no_spacing, hebbian=hebbian, fixed=fixed,
                          rehearse=rehearse, pseudo=pseudo, bias=bias)
        if fixed:
            init = {**init, **HAND_SET}
        self.lam_raw = nn.Parameter(torch.full((3,), _logit(init["lam"])))
        self.alpha_raw = nn.Parameter(torch.tensor(_logit(init["alpha"])))
        self.eta_raw = nn.Parameter(torch.tensor(_softplus_inv(init["eta"])))
        self.delta_raw = nn.Parameter(torch.tensor(_logit(init["delta"])))
        self.kappa_raw = nn.Parameter(torch.tensor(_logit(init["kappa"])))
        self.Delta_raw = nn.Parameter(torch.tensor(_softplus_inv(init["Delta"])))
        self.tau_raw = nn.Parameter(torch.tensor(_logit(init["tau"])))      # pseudo-label confidence threshold
        self.omega_raw = nn.Parameter(torch.tensor(_logit(init["omega"])))  # weight of a pseudo-labelled window
        if fixed:
            for p in self.parameters():
                p.requires_grad_(False)

    def values(self) -> dict:
        f = self.flags
        if f["fixed"]:  # exact hand-set values (the raw parametrization cannot reach 0 exactly)
            v = {k: torch.tensor(float(x)) for k, x in HAND_SET.items() if k != "lam"}
            v["lam"] = torch.zeros(3)
            v["tau"], v["omega"] = torch.tensor(1.0), torch.tensor(0.0)
        else:
            v = {"lam": torch.sigmoid(self.lam_raw), "alpha": torch.sigmoid(self.alpha_raw),
                 "eta": nn.functional.softplus(self.eta_raw), "delta": torch.sigmoid(self.delta_raw),
                 "kappa": torch.sigmoid(self.kappa_raw), "Delta": nn.functional.softplus(self.Delta_raw),
                 "tau": torch.sigmoid(self.tau_raw), "omega": torch.sigmoid(self.omega_raw)}
        dev = self.lam_raw.device
        v = {k: x.to(dev) for k, x in v.items()}
        if f["no_g"]:
            v["lam"] = torch.zeros_like(v["lam"])
        if f["no_f"]:
            v["eta"] = torch.zeros_like(v["eta"])
        if f["no_p"]:
            v["kappa"] = torch.zeros_like(v["kappa"])
        if f["no_spacing"]:
            v["Delta"] = torch.zeros_like(v["Delta"])
        if f.get("bias"):   # group 3: the constant input of the bias column, never gain-adjusted
            v["lam"] = torch.cat([v["lam"], torch.zeros(1, device=dev)])
        return v

    def describe(self) -> dict:
        return {k: (x.detach().cpu().tolist()) for k, x in self.values().items()} | {"flags": self.flags}


@dataclass
class Batch:
    """Persons of one population model. Streams padded to T; scored windows padded to S."""
    W: torch.Tensor          # (C, D), or (B, C, D) when persons come from different population models
    b: torch.Tensor          # (C,) or (B, C)
    groups: torch.Tensor     # (D,) feature group ids 0..2
    pool_z: torch.Tensor     # (B, T, D)
    pool_y: torch.Tensor     # (B, T) long
    pool_valid: torch.Tensor  # (B, T) bool
    conf: dict               # k -> (B, T) bool, the k shots per activity
    sc_z: torch.Tensor       # (B, S, D)
    sc_y: torch.Tensor       # (B, S) long
    sc_valid: torch.Tensor   # (B, S) bool


def readout(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Population readout per person: W (B, C, D), b (B, C)."""
    B = batch.pool_z.shape[0]
    W = batch.W if batch.W.dim() == 3 else batch.W[None].expand(B, -1, -1)
    b = batch.b if batch.b.dim() == 2 else batch.b[None].expand(B, -1)
    return W, b


def _sweeps(F, P, W0, b0, S, N, eta, eye, R, c, plus: float = 1.0):
    """R error-gated delta sweeps over the engram class means S / N, weighted by their shares (persons with c = 1)."""
    present = (N > 0).float()
    mu = S / N.clamp(min=1e-6)[:, :, None] * present[:, :, None]
    w = N / N.sum(-1, keepdim=True).clamp(min=1e-6)
    norm = ((mu * mu).sum(-1) + plus).clamp(min=1e-6)   # plus = 0 with the bias column ([mu, 1])
    for _ in range(R):
        lg = torch.einsum("bkd,bcd->bkc", mu, W0 + P + F) + b0[:, None, :]
        err = (eye[None] - torch.softmax(lg, -1)) * present[:, :, None]
        F = F + (c * eta)[:, None, None] * torch.einsum("bk,bkc,bkd->bcd", w / norm, err, mu)
    return F


def _pseudo(S, N, u, logits, mask, tau, omega, soft):
    """Iteration 2: unlabelled windows (mask) whose readout confidence passes tau are added to the engram of their
    predicted class with weight omega. u, logits: (B, L, D), (B, L, C); mask: (B, L)."""
    p = torch.softmax(logits, -1)
    conf, yhat = p.max(-1)
    gate = torch.sigmoid((conf - tau) * 20.0) if soft else (conf >= tau).float()
    wgt = omega * gate * mask.float()
    onehot = nn.functional.one_hot(yhat, logits.shape[-1]).float()
    S = S + torch.einsum("bl,blc,bld->bcd", wgt, onehot, u)
    N = N + torch.einsum("bl,blc->bc", wgt, onehot)
    return S, N


def _write(F, P, W0, b0, u, e_y, c, eta, eye, theta, S, N):
    """F's write at a confirmation (c: (B,) 1 where the person confirms). Without rehearsal: one error-gated
    delta step on the confirmed window. With rehearsal R (iteration 1): the window is added to the per-class
    engram sums S (B, C, D) and counts N (B, C) - feature x label outer products, no window is stored - and R
    error-gated delta sweeps run over the class means S / N, each class weighted by its share of confirmations."""
    R = theta.flags["rehearse"]
    plus = 0.0 if theta.flags.get("bias") else 1.0
    if not R:
        logits = torch.einsum("bcd,bd->bc", W0 + P + F, u) + b0
        err = e_y if theta.flags["hebbian"] else e_y - torch.softmax(logits, -1)
        F = F + (c * eta)[:, None, None] * err[:, :, None] * u[:, None, :] / ((u * u).sum(-1) + plus)[:, None, None]
        return F, S, N
    S = S + c[:, None, None] * e_y[:, :, None] * u[:, None, :]
    N = N + c[:, None] * e_y
    return _sweeps(F, P, W0, b0, S, N, eta, eye, R, c, plus), S, N


def _ema_all(x: torch.Tensor, alpha: torch.Tensor, init: float) -> torch.Tensor:
    """All states of s_t = (1 - α) s_{t-1} + α x_t (s_{-1} = init) for t = 0..T-1, as one matrix product: the
    recursion is linear, so it needs no loop. x: (B, T, D) -> (B, T, D)."""
    T = x.shape[1]
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    lag = t[:, None] - t[None, :]
    w = torch.where(lag >= 0, alpha * torch.pow(1 - alpha, lag.clamp(min=0)), torch.zeros((), device=x.device))
    return torch.einsum("ts,bsd->btd", w, x) + torch.pow(1 - alpha, t + 1)[None, :, None] * init


def run_stream(theta: Theta, batch: Batch, k: int, soft: bool) -> dict:
    """Final state (m, v, F, P) per person after the stream. Equal to run_stream_reference (tests/test_triplast.py)
    but faster: G's EMAs are computed for all windows at once, F's per-window decay is applied as a power between
    confirmations, and the sequential loop visits only the windows where some person confirms."""
    v_ = theta.values()
    B, T, D = batch.pool_z.shape
    C = batch.W.shape[-2]
    dev = batch.pool_z.device
    W0, b0 = readout(batch)
    lam = v_["lam"][batch.groups]                          # (D,)
    alpha, eta, delta, kappa, Delta = v_["alpha"], v_["eta"], v_["delta"], v_["kappa"], v_["Delta"]
    z_all = batch.pool_z
    m_all = _ema_all(z_all, alpha, 0.0)
    v_all = _ema_all((z_all - m_all) ** 2, alpha, 1.0)
    length = batch.pool_valid.sum(1)                       # (B,) stream lengths (padding is at the end)
    last_t = (length - 1).clamp(min=0)
    pick = last_t[:, None, None].expand(B, 1, D)
    has = (length > 0).float()[:, None]
    m = m_all.gather(1, pick)[:, 0] * has
    v = v_all.gather(1, pick)[:, 0] * has + (1 - has)
    F = torch.zeros(B, C, D, device=dev)
    P = torch.zeros(B, C, D, device=dev)
    cnt = torch.zeros(B, C, device=dev)
    last = torch.full((B, C), -1e6, device=dev)
    eye = torch.eye(C, device=dev)
    S_eng = torch.zeros(B, C, D, device=dev)
    N_eng = torch.zeros(B, C, device=dev)
    conf = (batch.conf[k] & batch.pool_valid) if k > 0 else torch.zeros(B, T, dtype=torch.bool, device=dev)
    steps = torch.nonzero(conf.any(0)).flatten().tolist()
    e_prev = torch.zeros(B, device=dev)
    pseudo = theta.flags["pseudo"]
    unl = batch.pool_valid & ~conf                          # unlabelled windows of the stream
    prev = -1

    def pseudo_block(a, b_, F_, S_, N_, e_ref):
        """Pseudo-label windows a..b_-1 with F decayed per window since e_ref (F only changes at confirmations)."""
        if b_ <= a:
            return S_, N_
        idx = torch.arange(a, b_, device=dev)
        z, mm, vv = z_all[:, a:b_], m_all[:, a:b_], v_all[:, a:b_]
        uu = ((1 - lam) + lam / torch.sqrt(vv + EPS)) * (z - lam * mm)
        e_s = torch.minimum(idx[None, :].float(), last_t[:, None].float())
        fac = torch.pow(1 - delta, (e_s - e_ref[:, None]).clamp(min=0))
        lg = torch.einsum("bld,bcd->blc", uu, W0 + P) + fac[..., None] * torch.einsum("bld,bcd->blc", uu, F_) \
            + b0[:, None, :]
        return _pseudo(S_, N_, uu, lg, unl[:, a:b_], v_["tau"], v_["omega"], soft)

    for t in steps:
        if pseudo:
            # windows prev+1..t: at t only the persons that do not confirm (mask), whose F is unchanged at t
            S_eng, N_eng = pseudo_block(prev + 1, t + 1, F, S_eng, N_eng, e_prev)
        prev = t
        e_t = torch.minimum(torch.full_like(e_prev, float(t)), last_t.float())
        F = F * torch.pow(1 - delta, e_t - e_prev)[:, None, None]
        e_prev = e_t
        c = conf[:, t].float()
        z, m_t, v_t = z_all[:, t], m_all[:, t], v_all[:, t]
        u = ((1 - lam) + lam / torch.sqrt(v_t + EPS)) * (z - lam * m_t)
        e_y = eye[batch.pool_y[:, t]]
        F, S_eng, N_eng = _write(F, P, W0, b0, u, e_y, c, eta, eye, theta, S_eng, N_eng)
        gap = t - (last * e_y).sum(-1)
        counted = (torch.sigmoid(gap - Delta) if soft else (gap >= Delta).float()) * c
        cnt = cnt + counted[:, None] * e_y
        last = last + (counted * (t - (last * e_y).sum(-1)))[:, None] * e_y
        cnt_y = (cnt * e_y).sum(-1)
        transfer = (torch.sigmoid((cnt_y - 1.5) * 8.0) if soft else (cnt_y >= 2).float()) * c
        move = (kappa * transfer)[:, None, None] * e_y[:, :, None] * F
        P = P + move
        F = F - move
        cnt = cnt * (1 - transfer[:, None] * e_y)
    if pseudo:
        S_eng, N_eng = pseudo_block(prev + 1, T, F, S_eng, N_eng, e_prev)
    F = F * torch.pow(1 - delta, last_t.float() - e_prev)[:, None, None]
    if pseudo and theta.flags["rehearse"]:  # final sweeps: engrams written after the last confirmation count too
        F = _sweeps(F, P, W0, b0, S_eng, N_eng, eta, eye, theta.flags["rehearse"],
                    (N_eng.sum(-1) > 0).float(), 0.0 if theta.flags.get("bias") else 1.0)
    return {"m": m, "v": v, "F": F, "P": P, "lam": lam, "S": S_eng, "N": N_eng}


def run_stream_reference(theta: Theta, batch: Batch, k: int, soft: bool) -> dict:
    """Unroll the stream window by window (the definition; kept as the reference for run_stream)."""
    v_ = theta.values()
    B, T, D = batch.pool_z.shape
    C = batch.W.shape[-2]
    dev = batch.pool_z.device
    W0, b0 = readout(batch)
    lam = v_["lam"][batch.groups]                          # (D,)
    alpha, eta, delta, kappa, Delta = v_["alpha"], v_["eta"], v_["delta"], v_["kappa"], v_["Delta"]
    m = torch.zeros(B, D, device=dev)
    v = torch.ones(B, D, device=dev)
    F = torch.zeros(B, C, D, device=dev)
    P = torch.zeros(B, C, D, device=dev)
    cnt = torch.zeros(B, C, device=dev)
    last = torch.full((B, C), -1e6, device=dev)
    S_eng = torch.zeros(B, C, D, device=dev)
    N_eng = torch.zeros(B, C, device=dev)
    conf = batch.conf.get(k, torch.zeros(B, T, dtype=torch.bool, device=dev)) if k > 0 else \
        torch.zeros(B, T, dtype=torch.bool, device=dev)
    eye = torch.eye(C, device=dev)
    for t in range(T):
        valid = batch.pool_valid[:, t].float()[:, None]     # (B, 1)
        z = batch.pool_z[:, t]                               # (B, D)
        m_new = (1 - alpha) * m + alpha * z
        v_new = (1 - alpha) * v + alpha * (z - m_new) ** 2
        m = valid * m_new + (1 - valid) * m
        v = valid * v_new + (1 - valid) * v
        F = F * (1 - delta * valid[:, :, None])
        c = (conf[:, t] & batch.pool_valid[:, t]).float()   # (B,)
        if theta.flags["pseudo"]:
            g_ = (1 - lam) + lam / torch.sqrt(v + EPS)
            u_ = g_ * (z - lam * m)
            lg = torch.einsum("bcd,bd->bc", W0 + P + F, u_) + b0
            mask = (batch.pool_valid[:, t] & ~conf[:, t])[:, None]
            S_eng, N_eng = _pseudo(S_eng, N_eng, u_[:, None], lg[:, None], mask, v_["tau"], v_["omega"], soft)
        if c.sum() == 0:
            continue
        g = (1 - lam) + lam / torch.sqrt(v + EPS)
        u = g * (z - lam * m)                                # (B, D)
        y = batch.pool_y[:, t]
        e_y = eye[y]                                         # (B, C)
        F, S_eng, N_eng = _write(F, P, W0, b0, u, e_y, c, eta, eye, theta, S_eng, N_eng)
        gap = t - (last * e_y).sum(-1)                       # (B,) steps since y's last counted confirmation
        counted = torch.sigmoid(gap - Delta) if soft else (gap >= Delta).float()
        counted = counted * c
        cnt = cnt + counted[:, None] * e_y
        last = last + (counted * (t - (last * e_y).sum(-1)))[:, None] * e_y
        cnt_y = (cnt * e_y).sum(-1)
        transfer = (torch.sigmoid((cnt_y - 1.5) * 8.0) if soft else (cnt_y >= 2).float()) * c
        move = (kappa * transfer)[:, None, None] * e_y[:, :, None] * F   # rows y only
        P = P + move
        F = F - move
        cnt = cnt * (1 - transfer[:, None] * e_y)
    if theta.flags["pseudo"] and theta.flags["rehearse"]:
        F = _sweeps(F, P, W0, b0, S_eng, N_eng, eta, eye, theta.flags["rehearse"], (N_eng.sum(-1) > 0).float(),
                    0.0 if theta.flags.get("bias") else 1.0)
    return {"m": m, "v": v, "F": F, "P": P, "lam": lam}


def with_bias_input(batch: Batch) -> Batch:
    """The batch with a constant-1 input appended (feature group 3) and the readout's bias moved into W's last
    column (b = 0), for the `bias` variant: the stores then act on [u, 1]."""
    def one(x):
        return torch.cat([x, torch.ones_like(x[..., :1])], -1)
    W = torch.cat([batch.W, batch.b[..., None]], -1)
    return Batch(W=W, b=torch.zeros_like(batch.b), groups=torch.cat([batch.groups, batch.groups.new_full((1,), 3)]),
                 pool_z=one(batch.pool_z), pool_y=batch.pool_y, pool_valid=batch.pool_valid, conf=batch.conf,
                 sc_z=one(batch.sc_z), sc_y=batch.sc_y, sc_valid=batch.sc_valid)


def score(theta: Theta, batch: Batch, state: dict, z: torch.Tensor | None = None) -> torch.Tensor:
    """Logits of the scored windows (B, S, C) with the frozen final state."""
    z = batch.sc_z if z is None else z
    lam = state["lam"]
    g = (1 - lam) + lam / torch.sqrt(state["v"] + EPS)
    u = g[:, None, :] * (z - lam * state["m"][:, None, :])
    W0, b0 = readout(batch)
    return torch.einsum("bsd,bcd->bsc", u, W0 + state["P"] + state["F"]) + b0[:, None, :]


def balanced_ce(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per person, cross-entropy with each present class weighted equally (macro-F1 treats classes equally);
    mean over persons."""
    B, S, C = logits.shape
    ce = nn.functional.cross_entropy(logits.reshape(-1, C), y.reshape(-1), reduction="none").reshape(B, S)
    onehot = nn.functional.one_hot(y, C).float() * valid[..., None].float()   # (B, S, C)
    per_class = onehot.sum(1)                                                   # (B, C)
    w = (onehot / per_class.clamp(min=1)[:, None, :]).sum(-1)                   # (B, S)
    n_cls = (per_class > 0).sum(-1).clamp(min=1)
    return ((ce * w).sum(1) / n_cls).mean()
