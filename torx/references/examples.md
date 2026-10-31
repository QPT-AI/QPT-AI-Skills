# Torx recipes

Patterns distilled from the Torx example gallery (notebooks 01-16), each written as a
complete script that runs as-is and asserts its own correctness against an exact
reference. All of them were run against torx 0.0.2 (JAX 0.11, Equinox 0.13, CPU) with
every assertion passing. Sizes are small so each finishes in seconds to about a minute
on CPU; scale them up once the pattern works.

The four core workflows (exact versus sampled readouts with parameter-shift training,
Trotterized graph heat flow, Gaussian state-space smoothing, and custom factors with a
DFG and the composites) are the worked examples at the top of `source.md`; they are
not repeated here.

Contents:
1. Chemical reaction networks with PJUMP
2. Single-site Gibbs updates with a one-gate PNOT circuit (and why per-edge PISING is not Gibbs)
3. Posterior per-pixel sampling with PNOT
4. Shared-kernel stochastic CNN (parameter tying + parameter shift)
5. MaxCut: exact density + REINFORCE over PISING sweeps
6. Regime-switching diffusion (hybrid scan)
7. Gaussian mixtures from a discrete control
8. Custom continuous gate: Metropolis-adjusted Langevin
9. Custom affine-Gaussian gate: an exact Ornstein-Uhlenbeck step
10. Factor-graph Gibbs sweep: TiledFactor + DeterministicFactor + ChainFactor
11. Injecting a distribution provider (AbstractSampler)

---

## 1. Chemical reaction networks with PJUMP

`PJUMP([src, dst])` moves a token from `src` to an empty `dst` and leaves every other
pair alone, so with exactly one token it is a single-walker jump on a directed edge with
`p = 1 - exp(-rate * dt)`. A chemical reaction network becomes such a walk on its
reachable configurations: enumerate them by breadth-first search, put one pbit on each,
and add one `PJUMP` per reaction edge with the mass-action propensity
`c * prod_s comb(n_s, nu_s)` as its rate. Check against `expm(Q t)` of the master
equation, and assert conservation laws on every edge (they catch enumeration bugs).
Circuit width equals the number of reachable configurations, which grows quickly with
the number of independent reaction extents.

```python
import jax
import jax.numpy as jnp
import numpy as np
from collections import deque
from math import comb
from jax.scipy.special import logit
from scipy.linalg import expm
from torx.psc import BranchingSimulator, DiscretePCircuit, PJUMP

# A + B <-> C with mass-action rates; species order (A, B, C).
reactions = [  # (reactants, products, rate constant)
    ({0: 1, 1: 1}, {2: 1}, 1.2),
    ({2: 1}, {0: 1, 1: 1}, 0.5),
]
start = (3, 2, 0)

# 1. Enumerate reachable configurations by breadth-first search.
configs, index, queue = [start], {start: 0}, deque([start])
edges = []                                          # (src, dst, propensity)
while queue:
    n = queue.popleft()
    for reac, prod, c in reactions:
        if all(n[s] >= v for s, v in reac.items()):
            m = list(n)
            for s, v in reac.items():
                m[s] -= v
            for s, v in prod.items():
                m[s] += v
            m = tuple(m)
            if m not in index:
                index[m] = len(configs)
                configs.append(m)
                queue.append(m)
            propensity = c * np.prod([comb(n[s], v) for s, v in reac.items()])
            edges.append((index[n], index[m], propensity))
configs = np.array(configs)
for i, j, _ in edges:                               # conservation laws hold on every edge
    assert configs[i][0] + configs[i][2] == configs[j][0] + configs[j][2]
    assert configs[i][1] + configs[i][2] == configs[j][1] + configs[j][2]

# 2. One pbit per configuration, one PJUMP per reaction edge: p = 1 - exp(-rate * dt).
T_END, M = 2.0, 200
dt = T_END / M
gates = [PJUMP([int(i), int(j)]) for i, j, _ in edges]
thetas = [jnp.array([logit(1.0 - np.exp(-r * dt))]) for _, _, r in edges]
circuit = DiscretePCircuit(gates, reps=M)

N = 20_000
sim = BranchingSimulator(num_samples=N)
x0 = jnp.zeros(len(configs), jnp.int32).at[0].set(1)   # one token on the start configuration
occupancy = np.asarray(sim.expval_all(sim.build_circuit(circuit, thetas), x0, jax.random.key(0)))

# 3. Exact reference: the master equation dp/dt = Q p.
Q = np.zeros((len(configs), len(configs)))
for i, j, r in edges:
    Q[j, i] += r
    Q[i, i] -= r
p_exact = expm(Q * T_END) @ np.eye(len(configs))[0]

assert abs(occupancy.sum() - 1.0) < 1e-6           # exactly one token survives
assert np.max(np.abs(occupancy - p_exact)) < 0.03  # MC error plus O(dt) splitting bias
expected_counts = occupancy @ configs
print("configurations", configs.tolist())
print("occupancy", np.round(occupancy, 3), "exact", np.round(p_exact, 3))
print("expected counts (A, B, C):", np.round(expected_counts, 3))
```

## 2. Single-site Gibbs updates with a one-gate PNOT circuit

The Gibbs conditional of an Ising spin is Bernoulli with logit `2 * beta * l_i`, where
`l_i = h_i + J * sum of neighbor spins`. A one-gate `PNOT` circuit started from an exact
zero draws it without any `PReset` leak; swapping the logit into a precompiled template
and vmapping draws a whole color class in one call. The second half measures the
pitfall: a sweep of one `PISING` per edge converges to the wrong law for the whole
graph, because each bond kernel ignores its endpoints' other neighbors. For serious
Gibbs sampling use thrml.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from torx.psc import BranchingSimulator, DiscretePCircuit, PISING, PNOT, StateVectorSimulator

# A 6-site Ising ring, spins s = 2b - 1, P(s) ~ exp(beta * (sum J s_i s_j + sum h_i s_i)).
N, BETA, J = 6, 0.6, 1.0
h = np.array([0.3, -0.2, 0.0, 0.1, -0.4, 0.2])
nbrs = [((i - 1) % N, (i + 1) % N) for i in range(N)]
colors = [np.arange(0, N, 2), np.arange(1, N, 2)]       # even/odd sites never share an edge

# Exact reference by enumeration, site 0 the most significant bit.
bits = (np.arange(2 ** N)[:, None] >> np.arange(N - 1, -1, -1)) & 1
spins = 2 * bits - 1
logw = BETA * (J * sum(spins[:, i] * spins[:, (i + 1) % N] for i in range(N)) + spins @ h)
p_exact = np.exp(logw - logw.max()); p_exact /= p_exact.sum()

# One PNOT started from an exact 0 draws Bernoulli(sigmoid(logit)): no PReset leak.
sim = BranchingSimulator(num_samples=1)
base = sim.build_circuit(DiscretePCircuit([PNOT(0)]), [jnp.array([0.0])])


def sample_bit(logit_value, key):
    compiled = eqx.tree_at(lambda c: c.thetas, base, base.thetas.at[0, 0].set(logit_value))
    return sim.sample(compiled, jnp.zeros(1, jnp.int32), key)[0, 0]


sample_bits = jax.vmap(sample_bit)                     # one batched call per color class
nbr_idx = jnp.array(nbrs)


@eqx.filter_jit
def gibbs(key, b, sweeps=60):
    def sweep(b, k):
        for color, kc in zip(colors, jax.random.split(k, 2)):
            s = 2 * b - 1
            local = jnp.asarray(h)[color] + J * s[nbr_idx[color]].sum(-1)
            b = b.at[color].set(sample_bits(2 * BETA * local, jax.random.split(kc, len(color))))
        return b, None
    return jax.lax.scan(sweep, b, jax.random.split(key, sweeps))[0]


chains = 8_000
keys = jax.random.split(jax.random.key(0), chains)
init = jax.random.bernoulli(jax.random.key(1), 0.5, (chains, N)).astype(jnp.int32)
final = np.asarray(jax.vmap(gibbs)(keys, init))
emp = np.bincount(final @ (1 << np.arange(N - 1, -1, -1)), minlength=2 ** N) / chains
tv_gibbs = 0.5 * np.abs(emp - p_exact).sum()

# Pitfall: one PISING per edge, swept repeatedly, is NOT a Gibbs sampler of the ring.
# Each bond kernel ignores its endpoints' other neighbor, so the fixed point differs.
ising = DiscretePCircuit([PISING([i, (i + 1) % N]) for i in range(N)], reps=200)
bond = [jnp.array([J, 0.0, 0.0, BETA, 0.5]) for _ in range(N)]
sv = StateVectorSimulator()
p_bonds = np.asarray(eqx.filter_jit(sv.density)(sv.build_circuit(ising, bond), jnp.ones(2 ** N) / 2 ** N))
p_zero_field = np.exp(BETA * J * sum(spins[:, i] * spins[:, (i + 1) % N] for i in range(N)))
p_zero_field /= p_zero_field.sum()
tv_bonds = 0.5 * np.abs(p_bonds - p_zero_field).sum()

assert tv_gibbs < 0.06, tv_gibbs
assert tv_bonds > 3 * tv_gibbs, (tv_bonds, tv_gibbs)
print(f"chromatic Gibbs TV to exact: {tv_gibbs:.3f}; per-edge PISING sweep TV: {tv_bonds:.3f}")
```

## 3. Posterior per-pixel sampling with PNOT

Given model probabilities `p_hat_i = P(x0_i = 1 | x_t)`, one `PNOT` per pixel with flip
probability `p_hat` (current bit 0) or `1 - p_hat` (current bit 1) produces an exact
`Bernoulli(p_hat)` draw. Clip before `logit`. Report one genuine draw separately from a
thresholded many-draw average, which is a mode estimator, not a sample.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logit
from torx.psc import BranchingSimulator, DiscretePCircuit, PNOT

# Given model probabilities p_hat_i = P(x0_i = 1 | x_t), one PNOT per pixel turns the
# current bit into an exact Bernoulli(p_hat_i) draw: flip with the right probability.
n_pixels, num_draws, batch = 16, 4_000, 3
gates = [PNOT(i) for i in range(n_pixels)]
sim = BranchingSimulator(num_samples=num_draws)
template = sim.build_circuit(DiscretePCircuit(gates), [jnp.zeros((1,)) for _ in gates])


def sample_row(x_row, p_hat, key):                    # x_row: int (n_pixels,)
    p_hat = jnp.clip(p_hat, 1e-6, 1 - 1e-6)           # keep logits finite
    p_flip = jnp.where(x_row == 1, 1.0 - p_hat, p_hat)
    compiled = eqx.tree_at(lambda c: c.thetas, template, logit(p_flip)[:, None])
    return sim.sample(compiled, x_row, key)           # (num_draws, n_pixels)


sample_batch = eqx.filter_jit(jax.vmap(sample_row))
rng = np.random.default_rng(0)
x_t = jnp.asarray(rng.integers(0, 2, (batch, n_pixels)), jnp.int32)
p_hat = jnp.asarray(rng.uniform(0, 1, (batch, n_pixels)), jnp.float32)
draws = np.asarray(sample_batch(x_t, p_hat, jax.random.split(jax.random.key(0), batch)))

assert draws.shape == (batch, num_draws, n_pixels)
np.testing.assert_allclose(draws.mean(axis=1), np.asarray(p_hat), atol=6 * 0.5 / np.sqrt(num_draws))
one_sample = draws[:, 0]                              # a genuine posterior draw
mode_estimate = (draws.mean(axis=1) > 0.5).astype(int)   # thresholded average: NOT a sample
print("max |mean - p_hat|:", float(np.abs(draws.mean(axis=1) - np.asarray(p_hat)).max()))
```

## 4. Shared-kernel stochastic CNN

Tie many physical gates to a few logits with an integer index map, read a few sites,
and train with parameter shift. The module stores the trainable arrays plus integer
structure arrays; `eqx.filter(model, eqx.is_inexact_array)` keeps the integers away
from the optimizer, and the gather `gate_logits[param_map]` is what accumulates the
gradients of all 60 gates into 5 logits. `param_shift_inf` requires `reps == 1`. The
script checks one logit's parameter-shift gradient against a central finite
difference on the same batch and key, then checks that the gate logits actually moved,
since otherwise only the linear head learned. This one takes about a minute on CPU.

```python
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import logit
from torx.psc import BranchingSimulator, DiscretePCircuit, PCNOT, PJUMP, PNOT, PReset

N_SIDE = 4


def bars_and_stripes(n):
    images, labels, seen = [], [], set()
    for label, axis in ((0, 1), (1, 0)):
        for line in itertools.product([0, 1], repeat=n):
            img = np.repeat(np.array(line)[:, None] if axis else np.array(line)[None, :], n, axis=axis)
            if tuple(img.ravel()) not in seen:
                seen.add(tuple(img.ravel())); images.append(img.ravel()); labels.append(label)
    return np.array(images, np.int32), np.array(labels, np.float32)


def build_scnn_circuit():
    """Shared-kernel convolution + PCNOT pooling: 60 physical gates, 5 shared logits."""
    gates, param_map = [], []
    for r in range(N_SIDE):
        for c in range(N_SIDE - 1):
            i = r * N_SIDE + c
            j = i + 1
            gates += [PJUMP([i, j]), PReset(j), PNOT(j), PJUMP([j, i])]
            param_map += [0, 1, 2, 0]
    pooled = []
    for r in range(N_SIDE):
        for c in range(0, N_SIDE, 2):
            src = r * N_SIDE + c
            gates.append(PCNOT([src, src + 1])); param_map.append(3); pooled.append(src + 1)
    readout = []
    for c in range(2):
        for r in range(0, N_SIDE, 2):
            src, sink = pooled[r * 2 + c], pooled[(r + 1) * 2 + c]
            gates.append(PCNOT([src, sink])); param_map.append(4); readout.append(sink)
    return DiscretePCircuit(gates), np.asarray(param_map), readout


circuit, param_map, readout_sites = build_scnn_circuit()
assert len(circuit.gates) == 60
sim = BranchingSimulator(diff_method="param_shift_inf", num_samples=256)   # reps == 1
template = sim.build_circuit(circuit, [jnp.zeros((1,)) for _ in circuit.gates])


class SCNN(eqx.Module):
    gate_logits: jax.Array            # (5,) trained
    linear: eqx.nn.Linear             # trained
    _param_map: jax.Array             # (60,) int: not inexact, so the optimizer skips it
    _readout_sites: jax.Array         # (4,) int

    def __call__(self, x, key):
        full = self.gate_logits[self._param_map][:, None]       # gather: grads accumulate
        compiled = eqx.tree_at(lambda c: c.thetas, template, full)
        features = sim.expval_all(compiled, x, key)[self._readout_sites]
        return jax.nn.sigmoid(jnp.squeeze(self.linear(features)))


def bce(model, xb, yb, key):
    preds = jax.vmap(model)(xb, jax.random.split(key, len(xb)))
    preds = jnp.clip(preds, 1e-6, 1 - 1e-6)
    return jnp.mean(-(yb * jnp.log(preds) + (1 - yb) * jnp.log(1 - preds)))


x, y = map(jnp.asarray, bars_and_stripes(N_SIDE))
init_logits = jnp.asarray([logit(p) for p in (0.34, 0.08, 0.06, 0.55, 0.60)])
model = SCNN(init_logits, eqx.nn.Linear(4, 1, key=jax.random.key(0)),
             jnp.asarray(param_map), jnp.asarray(readout_sites))
optimizer = optax.adam(5e-2)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


@eqx.filter_jit
def train_step(model, opt_state, key):
    loss, grads = eqx.filter_value_and_grad(bce)(model, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


# Gradient check on one shared logit: parameter shift vs central finite difference.
key = jax.random.key(1)
g_shift = eqx.filter_grad(bce)(model, x, y, key).gate_logits[0]
eps = 0.05
bump = lambda d: eqx.tree_at(lambda m: m.gate_logits, model, model.gate_logits.at[0].add(d))
g_fd = (bce(bump(eps), x, y, key) - bce(bump(-eps), x, y, key)) / (2 * eps)
assert abs(g_shift - g_fd) < 0.3 * abs(g_fd) + 2e-3, (g_shift, g_fd)

for step in range(25):
    key, sub = jax.random.split(key)
    model, opt_state, loss = train_step(model, opt_state, sub)
shift = jnp.abs(model.gate_logits - init_logits)
assert float(shift.max()) > 0.05                     # the shared kernel itself trained
print(f"grad check: shift {float(g_shift):.4f} vs fd {float(g_fd):.4f}; final loss {float(loss):.3f}")
print("logit movement", np.round(np.asarray(shift), 3))
```

## 5. MaxCut: exact density + REINFORCE over PISING sweeps

One `PISING` per edge, swept `REPS` times, with trainable couplings; the exact
distribution comes from `StateVectorSimulator`. The score-function estimator samples
indices from `stop_gradient(logp)`, weights `logp[idx]` by the centered reward, and
multiplies by `M / (M - 1)` to remove the same-batch-baseline bias. The exact expected
cut is tracked as a diagnostic only. On a circuit this small the exact gradient is also
available, and the script checks the two point the same way. Keep the site count away
from 8 on CPU (see the compile note in `source.md`, `statevector.py`).

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torx.psc import DiscretePCircuit, PISING, StateVectorSimulator

# MaxCut on a 6-node graph: sequential PISING bond updates, repeated, from uniform.
N, BETA, DT, REPS, M = 6, 1.5, 0.5, 20, 512
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4)]
bits = (np.arange(2 ** N)[:, None] >> np.arange(N - 1, -1, -1)) & 1   # site 0 = MSB
spins = jnp.asarray(2 * bits - 1)
cut_values = jnp.asarray(sum((bits[:, i] != bits[:, j]) for i, j in edges), jnp.float32)

sv = StateVectorSimulator()
circuit = DiscretePCircuit([PISING([i, j]) for i, j in edges], reps=REPS)
x0 = jnp.ones(2 ** N) / 2 ** N


def density(J):                                       # J: (num_edges,) trainable couplings
    thetas = [jnp.stack([J[e], 0.0, 0.0, BETA, DT]) for e in range(len(edges))]
    return sv.density(sv.build_circuit(circuit, thetas), x0)


def surrogate(J, key):
    """Score-function (REINFORCE) loss on exact log-probabilities of sampled states."""
    logp = jnp.log(density(J))
    idx = jax.random.categorical(key, jax.lax.stop_gradient(logp), shape=(M,))
    cuts = cut_values[idx]
    centered = jax.lax.stop_gradient(cuts - cuts.mean())
    loss = -(M / (M - 1)) * jnp.mean(centered * logp[idx])   # M/(M-1): same-batch baseline
    return loss, cuts.mean()


grad_fn = jax.jit(jax.value_and_grad(surrogate, has_aux=True))
expected_cut = jax.jit(lambda J: density(J) @ cut_values)   # exact diagnostic only

optimizer = optax.adam(0.1)
J = jnp.full((len(edges),), -0.2)                   # small antiferromagnetic start
opt_state = optimizer.init(J)
start_value = float(expected_cut(J))
key = jax.random.key(0)
for _ in range(120):
    key, sub = jax.random.split(key)
    (_, batch_mean), grads = grad_fn(J, sub)
    updates, opt_state = optimizer.update(grads, opt_state)
    J = optax.apply_updates(J, updates)
end_value = float(expected_cut(J))

# On a circuit this small the exact expected cut is differentiable too: same direction.
J0 = jnp.full((len(edges),), -0.2)
g_exact = -jax.jit(jax.grad(lambda J: density(J) @ cut_values))(J0)
g_reinforce = jnp.mean(jnp.stack([grad_fn(J0, k)[1] for k in jax.random.split(jax.random.key(1), 32)]), axis=0)
cos = float(g_exact @ g_reinforce / (jnp.linalg.norm(g_exact) * jnp.linalg.norm(g_reinforce)))

assert end_value > start_value + 0.5, (start_value, end_value)
assert cos > 0.9, cos
print(f"expected cut {start_value:.3f} -> {end_value:.3f} (max {float(cut_values.max()):.0f}); "
      f"cos(REINFORCE, exact grad) = {cos:.3f}")
```

## 6. Regime-switching diffusion

A latent regime on a pdit picks the drift and volatility of a position. One step is
`PditCycle` (regime first) then `MixtureGaussianGate` (increment under the new regime);
`PditCycle` theta is the log-odds of moving forward or backward relative to staying, and
the mixture theta holds per-regime means and log-variances. Trajectories come from
`jax.lax.scan` over `one_step.sample`, never from a large `reps` (direct sampling
unrolls `reps` into DFG sites). Checks: per-regime increments against
`N(mu_k dt, sigma_k**2 dt)` with a KS test, and the terminal mean against the exact mean
of the same discretized chain, so Monte Carlo error is separated from timestep error.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from torx.psc import HybridPCircuit, MixtureGaussianGate, PditCycle

# A latent regime S on a 3-state pdit picks the drift and volatility of position X.
# Lie-Trotter order inside one step: update the regime, then move X under the new regime.
K, T_END, STEPS = 3, 2.0, 200
dt = T_END / STEPS
lam_fwd, lam_back = 0.8, 0.4                         # ring switching rates
mu = np.array([0.5, -1.0, 0.0])
sigma = np.array([0.2, 0.6, 1.0])
p_fwd, p_back = lam_fwd * dt, lam_back * dt          # first-order switching probabilities
p_stay = 1.0 - p_fwd - p_back

one_step = HybridPCircuit([
    PditCycle(sites=0, dims=K),                                            # discrete site 0
    MixtureGaussianGate(sites=(0, 0), dims=(1,), num_components=K),        # disc 0 -> cont 0
])
thetas = [jnp.log(jnp.array([p_fwd / p_stay, p_back / p_stay])),
          {"means": jnp.asarray(mu * dt)[:, None], "log_vars": jnp.log(jnp.asarray(sigma ** 2 * dt))[:, None]}]


@eqx.filter_jit
def simulate(keys):
    def one_path(key):
        init = {"discrete": jnp.array([0], jnp.int32), "continuous": jnp.zeros(1)}

        def step(state, k):
            new = one_step.sample(k, state, thetas)
            return new, (new["discrete"][0], new["continuous"][0])

        _, path = jax.lax.scan(step, init, jax.random.split(key, STEPS))
        return path

    return jax.vmap(one_path)(keys)


regimes, positions = map(np.asarray, simulate(jax.random.split(jax.random.key(0), 20_000)))

# Check 1: conditioned on the new regime, one-step increments are N(mu_k dt, sigma_k^2 dt).
inc = np.diff(positions, axis=1, prepend=0.0)
for k in range(K):
    sel = inc[regimes == k][:5_000]
    z = (sel - mu[k] * dt) / (sigma[k] * np.sqrt(dt))
    assert stats.kstest(z, "norm").statistic < 5 / np.sqrt(len(z)), k

# Check 2: terminal mean against the exact mean of the same discretized chain.
P = p_stay * np.eye(K) + p_fwd * np.roll(np.eye(K), 1, axis=0) + p_back * np.roll(np.eye(K), -1, axis=0)
pi, mean_x = np.eye(K)[0], 0.0
for _ in range(STEPS):
    pi = P @ pi                                       # regime law after this step's switch
    mean_x += dt * mu @ pi
se = positions[:, -1].std() / np.sqrt(len(positions))
assert abs(positions[:, -1].mean() - mean_x) < 5 * se
print(f"terminal mean {positions[:, -1].mean():.4f} vs discretized exact {mean_x:.4f} (SE {se:.4f})")
print("final regime occupancy", np.round(np.bincount(regimes[:, -1], minlength=K) / len(regimes), 3),
      "vs", np.round(pi, 3))
```

## 7. Gaussian mixtures from a discrete control

Mixture weights live in the control site's distribution, not in
`MixtureGaussianGate`'s theta. Starting the control at 0, `PditCycle` with
`theta = log(pi[1:] / pi[0])` sets three weights in one step (`PditShift` with
`theta = [log(w1 / w0)]` does two). Starting the continuous site at 0 makes
`x = mu_k + noise`. Responsibilities for recovery must keep the log-variance term
whenever component covariances differ.

```python
import jax
import jax.numpy as jnp
import numpy as np
from torx.psc import HybridPCircuit, MixtureGaussianGate, PditCycle

# Mixture weights are not in MixtureGaussianGate's theta: they come from the control's
# distribution. A PditCycle started from control 0 sets them in one step:
# softmax([0, t0, t1]) = pi, so theta = log(pi[1:] / pi[0]).
K, d = 3, 2
pi = np.array([0.5, 0.3, 0.2])
means = jnp.array([[-2.0, 0.0], [2.0, 1.0], [0.0, -2.5]])
log_vars = jnp.log(jnp.array([[0.3, 0.2], [0.2, 0.4], [0.5, 0.1]]))

gen = HybridPCircuit([PditCycle(sites=0, dims=K),
                      MixtureGaussianGate(sites=(0, 0), dims=(d,), num_components=K)])
thetas = [jnp.log(jnp.asarray(pi[1:] / pi[0])), {"means": means, "log_vars": log_vars}]
init = {"discrete": jnp.array([0], jnp.int32), "continuous": jnp.zeros(d)}   # x = mu_k + noise
N = 30_000
s = gen.sample_multiple(jax.random.key(0), init, thetas, n_samples=N)
labels, x = np.asarray(s["discrete"][:, 0]), np.asarray(s["continuous"])

np.testing.assert_allclose(np.bincount(labels, minlength=K) / N, pi, atol=0.015)
for k in range(K):
    np.testing.assert_allclose(x[labels == k].mean(0), np.asarray(means[k]), atol=0.03)


# Recovery: responsibilities keep the log-variance term because covariances differ.
def responsibilities(v):
    quad = ((v[:, None, :] - means[None]) ** 2 / jnp.exp(log_vars)[None]).sum(-1)
    return jax.nn.softmax(jnp.log(jnp.asarray(pi)) - 0.5 * (quad + log_vars.sum(-1)[None]), axis=-1)


accuracy = float((np.asarray(responsibilities(jnp.asarray(x))).argmax(-1) == labels).mean())
assert accuracy > 0.97, accuracy
print(f"label frequencies {np.round(np.bincount(labels, minlength=K) / N, 3)}; MAP accuracy {accuracy:.3f}")
```

## 8. Custom continuous gate: Metropolis-adjusted Langevin

For an update with no closed form, subclass `AbstractContinuousGate`. It declares no
fields, so declare `sites` (with a converter to the `HybridSites` dict) and `dims`
yourself; implement `init_params` and `sample`. Store configuration such as the
Metropolis switch as a static field so a Python `if` on it is legal; derive the drift
with `jax.grad` of the same energy that defines the target. Validate on a 1-D target
against quadrature with a KS distance. To honour a caller's distribution provider, draw
noise with `info.normal(key, shape)` when `info` is not `None`.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, stats
from torx.psc import AbstractContinuousGate, HybridPCircuit


def to_hybrid_sites(x):
    """Accept 0, [0, 1] or a HybridSites dict, like the built-in continuous gates."""
    if isinstance(x, dict):
        return x
    return {"discrete": [], "continuous": list(x) if isinstance(x, (list, tuple)) else [x]}


class LangevinGate(AbstractContinuousGate):
    """One (Metropolis-adjusted) Langevin step on the target exp(-V(x) / T).

    AbstractContinuousGate declares no fields: a custom gate declares `sites` and `dims`
    itself. Concrete inherited methods are final, so only abstract ones are implemented.
    """

    sites: dict = eqx.field(converter=to_hybrid_sites)
    dims: tuple[int, ...] = eqx.field(converter=tuple)
    metropolis: bool = eqx.field(static=True, default=False)   # static: Python `if` is fine

    def init_params(self, key):
        return {"a": jnp.array(1.0), "tilt": jnp.array(0.3),
                "step": jnp.array(0.05), "temperature": jnp.array(1.0)}

    def energy(self, x, theta):                       # a tilted double well per coordinate
        return jnp.sum(theta["a"] * (x ** 2 - 1.0) ** 2 - theta["tilt"] * x)

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        x, eps, T = inputs["continuous"], params["step"], params["temperature"]
        grad_x = jax.grad(self.energy)(x, params)    # drift from the same potential
        k_noise, k_accept = jax.random.split(key)
        noise = jax.random.normal(k_noise, x.shape, dtype=x.dtype)
        prop = x - eps * grad_x + jnp.sqrt(2.0 * T * eps) * noise
        if not self.metropolis:
            out = prop
        else:
            grad_p = jax.grad(self.energy)(prop, params)
            log_target = -(self.energy(prop, params) - self.energy(x, params)) / T
            fwd = prop - (x - eps * grad_x)
            bwd = x - (prop - eps * grad_p)
            log_q = (jnp.sum(fwd ** 2) - jnp.sum(bwd ** 2)) / (4.0 * T * eps)
            accept = jax.random.bernoulli(k_accept, jnp.exp(jnp.minimum(0.0, log_target + log_q)))
            out = jnp.where(accept, prop, x)
        return (out, None) if return_aux else out


gate = LangevinGate(sites=0, dims=(1,), metropolis=True)
theta = gate.init_params(jax.random.key(0))
one_step = HybridPCircuit([gate])                    # reps=1: scan it, do not unroll reps=400


@eqx.filter_jit
def run_chains(key, n_chains=8_000, n_steps=400):
    def chain(k):
        init = {"discrete": jnp.zeros(0, jnp.int32), "continuous": jnp.zeros(1)}
        step = lambda state, kk: (one_step.sample(kk, state, [theta]), None)
        return jax.lax.scan(step, init, jax.random.split(k, n_steps))[0]["continuous"]
    return jax.vmap(chain)(jax.random.split(key, n_chains))


x = np.asarray(run_chains(jax.random.key(1))[:, 0])

# Exact 1-D reference by quadrature of exp(-V / T).
V = lambda u: float(theta["a"]) * (u ** 2 - 1) ** 2 - float(theta["tilt"]) * u
Z = integrate.quad(lambda u: np.exp(-V(u)), -4, 4)[0]
cdf = lambda u: integrate.quad(lambda w: np.exp(-V(w)) / Z, -4, u)[0]
ks = stats.kstest(x, np.vectorize(cdf)).statistic
assert ks < 5 / np.sqrt(len(x)), ks
print(f"MALA vs quadrature: KS = {ks:.4f} (threshold {5 / np.sqrt(len(x)):.4f})")
```

## 9. Custom affine-Gaussian gate: an exact Ornstein-Uhlenbeck step

When the update is affine-Gaussian, subclass `AbstractAffineGaussianGate` and implement
only `affine_parameters(theta) -> (A, b, log_var)` and `init_params`; the inherited
`sample` is final. `AffineGaussianSimulator` then gives exact, differentiable moments
(and conditioning) for any stack of such gates, looping `reps` internally.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from torx.psc import AbstractAffineGaussianGate, AffineGaussianSimulator, HybridPCircuit


def to_hybrid_sites(x):
    if isinstance(x, dict):
        return x
    return {"discrete": [], "continuous": list(x) if isinstance(x, (list, tuple)) else [x]}


class OUStep(AbstractAffineGaussianGate):
    """Exact Ornstein-Uhlenbeck step: x' = a x + sqrt(1 - a^2) noise, a = exp(-rate dt).

    Implement affine_parameters and init_params only; the inherited sample is final,
    and AffineGaussianSimulator gets exact moments and conditioning for free.
    """

    sites: dict = eqx.field(converter=to_hybrid_sites)
    dims: tuple[int, ...] = eqx.field(converter=tuple)

    def init_params(self, key):
        return {"rate": jnp.array(1.0), "dt": jnp.array(0.1)}

    def affine_parameters(self, theta):
        d = sum(self.dims)
        a = jnp.exp(-theta["rate"] * theta["dt"])
        return a * jnp.eye(d), jnp.zeros(d), jnp.full((d,), jnp.log1p(-a ** 2))


gate = OUStep(sites=0, dims=(1,))
theta = {"rate": jnp.array(0.7), "dt": jnp.array(0.1)}
n = 30
circuit = HybridPCircuit([gate], reps=n)             # the simulator loops reps internally
sim = AffineGaussianSimulator()
mom = sim.propagate(sim.build_circuit(circuit, [theta]), jnp.array([2.0]))

a = np.exp(-0.7 * 0.1)
np.testing.assert_allclose(float(mom.mean[0]), 2.0 * a ** n, rtol=1e-5)
np.testing.assert_allclose(float(mom.covariance[0, 0]), 1 - a ** (2 * n), rtol=1e-4)

# Exact moments are differentiable: d mean / d rate in closed form.
dmean = jax.grad(lambda r: sim.propagate(sim.build_circuit(circuit, [{"rate": r, "dt": jnp.array(0.1)}]),
                                         jnp.array([2.0])).mean[0])(jnp.array(0.7))
np.testing.assert_allclose(float(dmean), -2.0 * n * 0.1 * a ** n, rtol=1e-4)
print(f"OU after {n} steps: mean {float(mom.mean[0]):.4f}, var {float(mom.covariance[0, 0]):.4f}, "
      f"d mean / d rate {float(dmean):.4f}")
```

## 10. Factor-graph Gibbs sweep

One spin conditional as a factor, tiled over a checkerboard color, two color blocks plus
a deterministic reassembly as one `DFG` sweep, repeated by a weight-tied `ChainFactor`
(a `lax.scan`, so 150 sweeps compile once). Both block sites share
`param_key="coupling"`; only `spins` feeds back while `drive` is broadcast unchanged
every step. Checked spin by spin against enumeration of all `2**16` states.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from torx import (AbstractHasLogProbability, AbstractReferenceFactor, ChainFactor, DFG,
                  DeterministicFactor, Site, TiledFactor)

# A 4x4 Ising torus: one spin conditional as a factor, tiled over each checkerboard color,
# two color blocks plus a reassembly as one DFG sweep, repeated by a weight-tied chain.
G, BETA, J = 4, 0.45, 1.0
N = G * G
nbrs = [[((r - 1) % G) * G + c, ((r + 1) % G) * G + c, r * G + (c - 1) % G, r * G + (c + 1) % G]
        for r in range(G) for c in range(G)]
color_a = [i for i in range(N) if (i // G + i % G) % 2 == 0]
color_b = [i for i in range(N) if (i // G + i % G) % 2 == 1]
pos_a = {s: t for t, s in enumerate(color_a)}

SPIN = jax.ShapeDtypeStruct((), jnp.int32)
NEIGHBORS = jax.ShapeDtypeStruct((4,), jnp.int32)
DRIVE = jax.ShapeDtypeStruct((), jnp.float32)
SPINS = jax.ShapeDtypeStruct((N,), jnp.int32)
DRIVES = jax.ShapeDtypeStruct((N,), jnp.float32)
BLOCK_SPINS = jax.ShapeDtypeStruct((len(color_a),), jnp.int32)


class SpinUpdate(AbstractReferenceFactor, AbstractHasLogProbability):
    """P(s_i = +1 | neighbors, drive) = sigmoid(2 * (drive + j * sum(neighbors)))."""

    input_ports: dict = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports = {"neighbors": NEIGHBORS, "drive": DRIVE}
        self.output_spec = SPIN

    def init_params(self, key):
        return {"j": jnp.zeros((), jnp.float32)}

    def _field(self, inputs, params):
        return inputs["drive"] + params["j"] * inputs["neighbors"].sum()

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        up = jax.random.bernoulli(key, jax.nn.sigmoid(2 * self._field(inputs, params)))
        spin = 2 * up.astype(jnp.int32) - 1
        return (spin, None) if return_aux else spin

    def log_probability(self, inputs, outputs, params, info=None, site_info=None, return_aux=False):
        valid = (outputs == 1) | (outputs == -1)
        lp = jnp.where(valid, jax.nn.log_sigmoid(2 * self._field(inputs, params) * outputs), -jnp.inf)
        return (lp, None) if return_aux else lp


A_IDX, B_IDX = jnp.array(color_a), jnp.array(color_b)
A_NBRS = jnp.array([nbrs[i] for i in color_a])                       # read the incoming state
B_NBRS = jnp.array([[pos_a[j] for j in nbrs[i]] for i in color_b])   # read the fresh A block


def gather_a(parent_outputs):
    spins, drives = parent_outputs
    return {"neighbors": spins[A_NBRS], "drive": drives[A_IDX]}


def gather_b(parent_outputs):
    block_a, drives = parent_outputs
    return {"neighbors": block_a[B_NBRS], "drive": drives[B_IDX]}


def scatter(inputs, site_info):
    return jnp.zeros(N, jnp.int32).at[A_IDX].set(inputs["block_a"]).at[B_IDX].set(inputs["block_b"])


block = TiledFactor(SpinUpdate(), len(color_a), weight_tied=True)
sweep = DFG(
    sites=(
        Site("block_a", block, ("spins", "drive"), gather_a, "coupling", info_key=None, site_info=None),
        Site("block_b", block, ("block_a", "drive"), gather_b, "coupling", info_key=None, site_info=None),
        Site("state", DeterministicFactor(scatter, {"block_a": BLOCK_SPINS, "block_b": BLOCK_SPINS}, SPINS),
             ("block_a", "block_b"), ("block_a", "block_b"), None, info_key=None, site_info=None),
    ),
    input_ports={"spins": SPINS, "drive": DRIVES},
    output_name="state",
)
chain = ChainFactor(sweep, n_steps=150, feedback_porting_fn="spins", weight_tied=True)  # a scan

field = jnp.asarray(np.random.default_rng(0).normal(0, 0.6, N), jnp.float32)
params = {"coupling": {"j": jnp.asarray(BETA * J, jnp.float32)}}   # both blocks share it


@eqx.filter_jit
def run(keys, inits):
    return jax.vmap(lambda k, x: chain.sample(k, {"spins": x, "drive": BETA * field}, params))(keys, inits)


n_chains = 4_000
inits = 2 * jax.random.bernoulli(jax.random.key(0), 0.5, (n_chains, N)).astype(jnp.int32) - 1
samples = np.asarray(run(jax.random.split(jax.random.key(1), n_chains), inits))

# Exact reference by enumerating all 2**16 states.
all_s = 2 * ((np.arange(2 ** N)[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(np.int8) - 1
edges = sorted({tuple(sorted((i, j))) for i in range(N) for j in nbrs[i]})
logw = BETA * (J * sum(all_s[:, i] * all_s[:, j] for i, j in edges) + all_s @ np.asarray(field))
p = np.exp(logw - logw.max()); p /= p.sum()
exact_mean = p @ all_s
err = float(np.abs(samples.mean(0) - exact_mean).max())
assert err < 0.06, err
print(f"{len(edges)} edges; max per-spin error vs exact enumeration: {err:.4f}")
```

## 11. Injecting a distribution provider

Every random draw in Torx goes through an `AbstractSampler` (`bernoulli`, `categorical`,
`normal`); `None` means `JaxPRNGSampler`. Simulators take `sampler=`; inside a circuit's
DFG walk, gates read it from `info`, so pass it once as
`DFGInfo(expose_site_outputs=False, default_site_info=sampler)`. A deterministic
"mode" provider like the one below is a quick way to debug wiring: the circuit follows
its most likely branch with zero noise.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logit
from torx import AbstractSampler, DFGInfo, JaxPRNGSampler
from torx.psc import (AffineGaussianGate, BranchingSimulator, DiscretePCircuit, HybridPCircuit,
                      PCNOT, PNOT, PditCycle, MixtureGaussianGate)


class ModeSampler(AbstractSampler):
    """Deterministic provider: always the most likely outcome, zero noise.

    Useful to debug wiring: the circuit follows its most probable branch everywhere.
    """

    def bernoulli(self, key, p, shape=None):
        p = jnp.asarray(p)
        return jnp.broadcast_to(p > 0.5, p.shape if shape is None else shape).astype(jnp.int32)

    def categorical(self, key, logits, axis=-1, shape=None):
        idx = jnp.argmax(logits, axis=axis)
        return idx if shape is None else jnp.broadcast_to(idx, shape)

    def normal(self, key, shape=(), dtype=float):
        return jnp.zeros(shape, dtype)


circuit = DiscretePCircuit([PNOT(0), PCNOT([0, 1])])
thetas = [jnp.array([logit(0.9)]), jnp.array([logit(0.8)])]
sim = BranchingSimulator(num_samples=5)
compiled = sim.build_circuit(circuit, thetas)
start = jnp.array([0, 0], jnp.int32)

# Simulators take sampler=...; None means JaxPRNGSampler, so these two are identical.
a = sim.sample(compiled, start, jax.random.key(0))
b = sim.sample(compiled, start, jax.random.key(0), sampler=JaxPRNGSampler())
np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
mode = np.asarray(sim.sample(compiled, start, jax.random.key(0), sampler=ModeSampler()))
assert (mode == [1, 1]).all()                         # both gates take their likely branch

# Inside a circuit's DFG walk, gates read the provider from info: pass it once as the
# default for every site.
hybrid = HybridPCircuit([PditCycle(sites=0, dims=3),
                         MixtureGaussianGate(sites=(0, 0), dims=(1,), num_components=3),
                         AffineGaussianGate(sites=[0], dims=(1,))])
h_thetas = [jnp.array([2.0, -1.0]),
            {"means": jnp.array([[0.0], [5.0], [-5.0]]), "log_vars": jnp.zeros((3, 1))},
            {"A": jnp.eye(1) * 2.0, "b": jnp.ones(1), "log_var": jnp.zeros(1)}]
state = {"discrete": jnp.array([0], jnp.int32), "continuous": jnp.zeros(1)}
out = hybrid.sample(jax.random.key(1), state, h_thetas,
                    info=DFGInfo(expose_site_outputs=False, default_site_info=ModeSampler()))
assert int(out["discrete"][0]) == 1                   # forward is the likeliest branch
np.testing.assert_allclose(np.asarray(out["continuous"]), [2.0 * 5.0 + 1.0])  # noise-free path
print("mode run:", np.asarray(mode[0]), {k: np.asarray(v) for k, v in out.items()})
```

