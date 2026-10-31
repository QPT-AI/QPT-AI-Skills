# Torx source, annotated

The annotated library source of **Torx 0.0.2** (`pip install extro-torx`, module
`torx`), by Extropic, distributed under the Apache License 2.0 (full text in
`LICENSE-torx.txt` next to this file). The package ships no tests; the ASCII
renderer's grid internals are omitted. Everything else is verbatim.

Use this file as the ground truth for any exact signature, default, shape, or
numerically subtle rule. Every module is prefaced by a note on how it composes with
the rest. Four worked examples come first; each is a complete script that was run
against torx 0.0.2 (JAX 0.11, Equinox 0.13, CPU) with every assertion passing.

If the installed version differs (`python -c "import torx; print(torx.__version__)"`),
trust the installed code over this file and say so.

## Contents

- Worked example 1: one discrete circuit, three readouts, one trained gate
- Worked example 2: graph heat flow as a Trotterized PSWAP circuit
- Worked example 3: a linear-Gaussian state-space model and its exact smoother
- Worked example 4: custom factors, a DFG, and the two composites
- Module map and import surface
- `torx/factor.py` - the factor contract
- `torx/tractable_prob_factors.py` - log-probability mixins, matrix and deterministic factors
- `torx/dfg.py` - `Site`, `DFG`, `DFGInfo`
- `torx/composite_factors.py` - `TiledFactor`, `ChainFactor`
- `torx/_sampler.py`, `torx/_utils.py` - distribution providers, spec comparison
- `torx/psc/gates/_base.py` - the gate hierarchy
- `torx/psc/gates/_binary.py` - pbit gates
- `torx/psc/gates/_pdit.py` - pdit gates
- `torx/psc/gates/_generator.py` - `PISING`
- `torx/psc/gates/_continuous.py` - pmode and hybrid gates
- `torx/psc/_circuit.py` - `DiscretePCircuit`, `HybridPCircuit`
- `torx/psc/simulation/base.py` - simulator and compiled-circuit bases
- `torx/psc/simulation/statevector.py` - `StateVectorSimulator`
- `torx/psc/simulation/sampled.py` - `BranchingSimulator` and parameter shift
- `torx/psc/simulation/gaussian.py` - `AffineGaussianSimulator`
- `torx/psc/visualization/text.py` - `draw_circuit`

Search tip: every module section starts with a heading of the form
`` ## `torx/...` ``, and every class appears verbatim as `class Name(`.

---

## Worked example 1: one discrete circuit, three readouts, one trained gate

The canonical discrete workflow. One circuit, read three ways: the exact density from
`StateVectorSimulator` (float distribution in), raw bitstrings from
`BranchingSimulator` (int basis state in), and the circuit sampling itself as a DFG
(input port `"in"`). All three agree within the binomial standard error. Then the three
logits are trained by parameter shift on a precompiled template; the sampled gradient
is checked against exact autodiff of the same objective before training.

```python
# we1_discrete.py
"""Worked example 1: one discrete circuit, three readouts, one trained gate."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import logit
from torx.psc import BranchingSimulator, DiscretePCircuit, PCNOT, PNOT, StateVectorSimulator

# Structure: which gate touches which pbit, in order. No numbers live here.
circuit = DiscretePCircuit([PNOT(0), PCNOT([0, 1]), PNOT(1)])
# Data: one theta per gate, aligned with circuit.gates. Two-branch gates take a (1,) logit.
thetas = [jnp.array([logit(p)]) for p in (0.38, 0.70, 0.25)]
assert len(thetas) == len(circuit.gates)

start = jnp.array([0, 0], jnp.int32)                     # basis state |00)
flat = int(start[0]) * 2 + int(start[1])                  # big-endian: site 0 is the high digit

# 1. Exact: StateVectorSimulator takes a float distribution over prod(dims) = 4 states.
sv = StateVectorSimulator()
exact = sv.density(sv.build_circuit(circuit, thetas), jnp.zeros(4).at[flat].set(1.0))
np.testing.assert_allclose(float(exact.sum()), 1.0, atol=1e-6)

# 2. Sampled: BranchingSimulator takes an int basis state and returns raw bitstrings.
N = 20_000
bs = BranchingSimulator(num_samples=N)
samples = np.asarray(bs.sample(bs.build_circuit(circuit, thetas), start, jax.random.key(0)))
empirical = np.bincount(samples[:, 0] * 2 + samples[:, 1], minlength=4) / N

# 3. Direct: a circuit is a DFG, so it samples itself (input port "in", int32 (num_pdits,)).
direct = np.asarray(circuit.sample_multiple(jax.random.key(1), {"in": start}, thetas, n_samples=N))
direct_emp = np.bincount(direct[:, 0] * 2 + direct[:, 1], minlength=4) / N

se = 0.5 / np.sqrt(N)                                     # worst-case binomial standard error
np.testing.assert_allclose(empirical, exact, atol=6 * se)
np.testing.assert_allclose(direct_emp, exact, atol=6 * se)

# Train the three logits so the final marginals hit a target, by parameter shift.
target = jnp.array([0.2, 0.9])
train_sim = BranchingSimulator(diff_method="param_shift_inf", num_samples=2_048)
template = train_sim.build_circuit(circuit, [jnp.zeros((1,)) for _ in circuit.gates])


def sampled_loss(rows, key):                              # rows: (num_gates, max_branches - 1)
    compiled = eqx.tree_at(lambda c: c.thetas, template, rows)
    return jnp.sum((train_sim.expval_all(compiled, start, key) - target) ** 2)


def exact_loss(rows):                                     # same objective, exact
    compiled = sv.build_circuit(circuit, [r for r in rows])
    means = sv.expval_all(compiled, jnp.zeros(4).at[flat].set(1.0))
    return jnp.sum((means - target) ** 2)


# Gradient check: parameter shift vs exact autodiff at the starting point.
rows0 = jnp.stack(thetas)                                 # (3, 1)
g_shift = jnp.mean(jnp.stack([jax.grad(sampled_loss)(rows0, k)
                              for k in jax.random.split(jax.random.key(2), 8)]), axis=0)
g_exact = jax.grad(exact_loss)(rows0)
np.testing.assert_allclose(g_shift, g_exact, atol=0.02)

optimizer = optax.adam(5e-2)
opt_state = optimizer.init(rows0)


@jax.jit
def train_step(rows, opt_state, key):
    value, grads = jax.value_and_grad(sampled_loss)(rows, key)
    updates, opt_state = optimizer.update(grads, opt_state, rows)
    return optax.apply_updates(rows, updates), opt_state, value


rows, key = rows0, jax.random.key(3)
for _ in range(300):
    key, sub = jax.random.split(key)                      # fresh key every step
    rows, opt_state, _ = train_step(rows, opt_state, sub)

final_exact = float(exact_loss(rows))
assert final_exact < 1e-3, final_exact
print("exact density     ", np.round(np.asarray(exact), 4))
print("branching sampled ", np.round(empirical, 4))
print("direct DFG sampled", np.round(direct_emp, 4))
print("param-shift grad  ", np.round(np.asarray(g_shift).ravel(), 4))
print("exact grad        ", np.round(np.asarray(g_exact).ravel(), 4))
print(f"trained exact loss {final_exact:.2e}")
```

## Worked example 2: graph heat flow as a Trotterized PSWAP circuit

One PSWAP per edge is the exact two-node propagator; a layer of them is one Lie-Trotter
step of `exp(Q t)`, and `reps=m` repeats the layer. The script separates the two error
sources: sampled-versus-scheme is pure Monte Carlo error, and scheme-versus-`expm` is
splitting bias, which falls as `1/m` for the first-order sweep and `1/m**2` for the
symmetric Strang sweep. The exact state vector over all `2**10` bitstrings reproduces
the deterministic scheme, and is jitted once because eager simulator calls recompile.

```python
# we2_heat_flow.py
"""Worked example 2: heat flow on a graph as a Trotterized PSWAP circuit."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logit
from scipy.linalg import expm
from torx.psc import BranchingSimulator, DiscretePCircuit, PSWAP, StateVectorSimulator

# A 10-node ring with five chords; one pbit per node marks where the walker is.
N_NODES, RATE, T = 10, 0.7, 1.4
edges = [(i, (i + 1) % N_NODES) for i in range(N_NODES)] + [(i, i + 5) for i in range(5)]
p0 = np.zeros(N_NODES)
p0[[0, 1, 9]] = [0.70, 0.20, 0.10]

# Exact continuous-time reference: dp/dt = Q p with Q = -rate * Laplacian.
Q = np.zeros((N_NODES, N_NODES))
for i, j in edges:
    Q[i, j] += RATE
    Q[j, i] += RATE
Q -= np.diag(Q.sum(axis=0))
exact = expm(Q * T) @ p0


def edge_swap_probability(rate, dt):
    """Exact two-node propagator over dt: the walker crosses with this probability."""
    return 0.5 * (1.0 - np.exp(-2.0 * rate * dt))


def trotter_circuit(m, order=1):
    """m Trotter layers; order=2 is the symmetric Strang sweep (forward, then reversed)."""
    if order == 1:
        layer, dt = list(edges), T / m
    else:
        layer, dt = list(edges) + list(reversed(edges)), T / m / 2
    gates = [PSWAP([int(i), int(j)]) for i, j in layer]
    thetas = [jnp.array([logit(edge_swap_probability(RATE, dt))]) for _ in layer]
    return DiscretePCircuit(gates, reps=m), thetas


def deterministic_scheme(m, order=1):
    """The same ordered product applied to the occupancy vector, with no sampling."""
    layer = list(edges) if order == 1 else list(edges) + list(reversed(edges))
    s = edge_swap_probability(RATE, T / m if order == 1 else T / m / 2)
    p = p0.copy()
    for _ in range(m):
        for a, b in layer:
            p[a], p[b] = (1 - s) * p[a] + s * p[b], s * p[a] + (1 - s) * p[b]
    return p


# One walker is a one-hot bitstring. Heat flow is linear in p0, so run each occupied
# source separately and recombine with its weight.
def sampled_occupancy(circuit, thetas, num_samples, key):
    sim = BranchingSimulator(num_samples=num_samples)
    compiled = sim.build_circuit(circuit, thetas)
    total = np.zeros(N_NODES)
    for src in np.flatnonzero(p0):
        key, sub = jax.random.split(key)
        start = jnp.zeros(N_NODES, jnp.int32).at[src].set(1)
        total += p0[src] * np.asarray(sim.expval_all(compiled, start, sub))
    return total


M, NUM_SAMPLES = 16, 12_000
circuit, thetas = trotter_circuit(M)
sampled = sampled_occupancy(circuit, thetas, NUM_SAMPLES, jax.random.key(0))
scheme = deterministic_scheme(M)

# The exact state vector over 2**10 bitstrings reproduces the deterministic scheme.
sv = StateVectorSimulator()
x0 = jnp.zeros(2 ** N_NODES)
for src in np.flatnonzero(p0):                            # mixture of one-hot walkers
    x0 = x0.at[1 << (N_NODES - 1 - src)].add(p0[src])     # site 0 is the most significant bit
exact_occupancy = eqx.filter_jit(sv.expval_all)           # jit once; eager calls recompile
sv_occ = np.asarray(exact_occupancy(sv.build_circuit(circuit, thetas), x0))
np.testing.assert_allclose(sv_occ, scheme, atol=1e-5)

mc_floor = 0.5 / np.sqrt(NUM_SAMPLES)
np.testing.assert_allclose(sampled.sum(), 1.0, atol=4 * mc_floor)
assert np.max(np.abs(sampled - scheme)) < 5 * mc_floor            # Monte Carlo error only
bias_1 = [np.linalg.norm(deterministic_scheme(m) - exact) for m in (4, 8, 16, 32)]
bias_2 = [np.linalg.norm(deterministic_scheme(m, order=2) - exact) for m in (4, 8, 16, 32)]
assert all(np.diff(bias_1) < 0) and all(np.diff(bias_2) < 0)
ratio_1 = bias_1[0] / bias_1[-1]                                    # ~8 for 1/m
ratio_2 = bias_2[0] / bias_2[-1]                                    # ~64 for 1/m**2
assert 5 < ratio_1 < 12 and ratio_2 > 30, (ratio_1, ratio_2)

print("exact      ", np.round(exact, 3))
print("scheme m=16", np.round(scheme, 3))
print("sampled    ", np.round(sampled, 3))
print("first-order bias", np.round(bias_1, 5), f"(x{ratio_1:.1f} from m=4 to 32)")
print("Strang bias     ", np.round(bias_2, 5), f"(x{ratio_2:.1f} from m=4 to 32)")
```

## Worked example 3: a linear-Gaussian state-space model and its exact smoother

An unrolled state-space model built from `AffineGaussianGate`s that read one site and
write the next (the write-to-a-new-site idiom, with `log_var = -inf` for exact
passthrough). `AffineGaussianSimulator.propagate` gives the exact joint prior;
`condition` gives the smoothed posterior over the latent sites. Both are checked
against a hand-built dense covariance and Schur complement, and the circuit's own
samples are checked against the propagated moments.

```python
# we3_gaussian_smoother.py
"""Worked example 3: a linear-Gaussian state-space model and its exact smoother."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from torx.psc import AffineGaussianGate, AffineGaussianSimulator, HybridPCircuit

T, d, k = 12, 2, 3                                         # steps, latent dim, channels
rng = np.random.default_rng(0)
A = np.array([[0.95, 0.10], [-0.10, 0.90]])
C = rng.normal(size=(k, d))
m0, P0 = np.array([1.0, -0.5]), np.array([0.30, 0.30])
Qd, Rd = np.array([0.05, 0.04]), np.array([0.10, 0.12, 0.08])

# Simulate observations from the model itself.
z = np.zeros((T, d)); y = np.zeros((T, k))
z[0] = m0 + rng.normal(size=d) * np.sqrt(P0)
for t in range(T):
    if t:
        z[t] = A @ z[t - 1] + rng.normal(size=d) * np.sqrt(Qd)
    y[t] = C @ z[t] + rng.normal(size=k) * np.sqrt(Rd)

# Latent sites 0..T-1 (dim d), observation sites T..2T-1 (dim k). Separate namespaces
# from any discrete sites. Multi-site gates act on the concatenated local state.
det = lambda n: jnp.full((n,), -jnp.inf)                  # exact, noise-free coordinates
I, Zd = jnp.eye(d), jnp.zeros((d, d))

initial = AffineGaussianGate(sites=[0], dims=(d,))
initial_theta = {"A": jnp.zeros((d, d)), "b": jnp.asarray(m0), "log_var": jnp.log(jnp.asarray(P0))}

# Write-to-a-new-site: read z_{t-1}, keep it unchanged, write z_t = A z_{t-1} + noise.
transitions = [AffineGaussianGate(sites=[t - 1, t], dims=(d, d)) for t in range(1, T)]
transition_theta = {"A": jnp.block([[I, Zd], [jnp.asarray(A), Zd]]),
                    "b": jnp.zeros(2 * d),
                    "log_var": jnp.concatenate([det(d), jnp.log(jnp.asarray(Qd))])}

emissions = [AffineGaussianGate(sites=[t, T + t], dims=(d, k)) for t in range(T)]
emission_theta = {"A": jnp.block([[I, jnp.zeros((d, k))], [jnp.asarray(C), jnp.zeros((k, k))]]),
                  "b": jnp.zeros(d + k),
                  "log_var": jnp.concatenate([det(d), jnp.log(jnp.asarray(Rd))])}

circuit = HybridPCircuit([initial, *transitions, *emissions])
thetas = [initial_theta, *[transition_theta] * (T - 1), *[emission_theta] * T]
assert circuit.continuous_state_dim == T * d + T * k

sim = AffineGaussianSimulator()
compiled = sim.build_circuit(circuit, thetas)
x0 = jnp.zeros(circuit.continuous_state_dim)
prior = sim.propagate(compiled, x0)                        # exact joint moments
post = sim.condition(compiled,
                     observations={T + t: jnp.asarray(y[t]) for t in range(T)},
                     initial_continuous=x0,
                     query_sites=list(range(T)))           # site sets are static Python
smoothed = np.asarray(post.mean).reshape(T, d)

# Dense reference: build the joint covariance by hand and apply the Schur complement.
D = T * d + T * k
F = np.zeros((D, D)); mu = np.zeros(D)
for t in range(T):
    for s in range(t + 1):
        F[t * d:(t + 1) * d, s * d:(s + 1) * d] = np.linalg.matrix_power(A, t - s)
noise = np.zeros(D)
noise[:d] = P0
for t in range(1, T):
    noise[t * d:(t + 1) * d] = Qd
for t in range(T):
    mu[t * d:(t + 1) * d] = np.linalg.matrix_power(A, t) @ m0
lat_cov = F[:T * d, :T * d] @ np.diag(noise[:T * d]) @ F[:T * d, :T * d].T
H = np.kron(np.eye(T), C)
mu[T * d:] = H @ mu[:T * d]
Sigma = np.zeros((D, D))
Sigma[:T * d, :T * d] = lat_cov
Sigma[T * d:, :T * d] = H @ lat_cov
Sigma[:T * d, T * d:] = lat_cov @ H.T
Sigma[T * d:, T * d:] = H @ lat_cov @ H.T + np.diag(np.tile(Rd, T))
np.testing.assert_allclose(np.asarray(prior.mean), mu, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(np.asarray(prior.covariance), Sigma, rtol=1e-4, atol=1e-5)

q, o = np.arange(T * d), np.arange(T * d, D)
gain = Sigma[np.ix_(q, o)] @ np.linalg.inv(Sigma[np.ix_(o, o)])
dense_mean = mu[q] + gain @ (y.reshape(-1) - mu[o])
dense_cov = Sigma[np.ix_(q, q)] - gain @ Sigma[np.ix_(o, q)]
np.testing.assert_allclose(np.asarray(post.mean), dense_mean, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(np.asarray(post.covariance), dense_cov, rtol=1e-3, atol=1e-5)

# The same circuit also samples directly; its moments match propagate within MC error.
N = 20_000
init = {"discrete": jnp.zeros(0, jnp.int32), "continuous": x0}
draws = eqx.filter_jit(circuit.sample_multiple)(jax.random.key(0), init, thetas, n_samples=N)
draws = np.asarray(draws["continuous"])
se = np.sqrt(np.diag(np.asarray(prior.covariance)) / N)
assert np.all(np.abs(draws.mean(0) - np.asarray(prior.mean)) < 6 * se)

rmse_smooth = np.sqrt(np.mean((smoothed - z) ** 2))
rmse_prior = np.sqrt(np.mean((np.asarray(prior.mean[:T * d]).reshape(T, d) - z) ** 2))
assert rmse_smooth < rmse_prior
print(f"smoother RMSE {rmse_smooth:.3f} vs prior-mean RMSE {rmse_prior:.3f}")
print("site 3 posterior mean, cov:", post.site_moments(3)[0], np.round(post.site_moments(3)[1], 4).tolist())
```

## Worked example 4: custom factors, a DFG, and the two composites

The lower layer. Three custom factors (one with an analytic `log_probability`), a
matrix factor with an explicit table, a two-site `DFG` with a callable porting function,
`DFGInfo(expose_site_outputs=True)` to read every site, and `TiledFactor` /
`ChainFactor` checked draw-for-draw against a manual `vmap` and a manual loop under the
same keys.

```python
# we4_factors_dfg.py
"""Worked example 4: custom factors, a DFG, and the two composites."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from torx import (AbstractHasLogProbability, AbstractMatrixFactor, AbstractReferenceFactor,
                  ChainFactor, DFG, DFGInfo, DeterministicFactor, Site, TiledFactor)

BIT = jax.ShapeDtypeStruct((), jnp.int32)                 # port specs: shape + dtype, no data
DRIVE = jax.ShapeDtypeStruct((), jnp.float32)


class CoinFactor(AbstractReferenceFactor):
    """P(coin): no inputs, 1 with probability sigmoid(bias)."""

    input_ports: dict[str, jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports = {}
        self.output_spec = BIT

    def init_params(self, key):
        return {"bias": jnp.array(0.4)}

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        out = jax.random.bernoulli(key, jax.nn.sigmoid(params["bias"])).astype(jnp.int32)
        return (out, None) if return_aux else out


class ConditionalBit(AbstractReferenceFactor, AbstractHasLogProbability):
    """P(out | drive) = Bernoulli(sigmoid(w * drive + b)), with an exact log_probability."""

    input_ports: dict[str, jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports = {"drive": DRIVE}
        self.output_spec = BIT

    def init_params(self, key):
        return {"w": jnp.array(2.0), "b": jnp.array(-1.0)}

    def _logit(self, inputs, params):
        return params["w"] * inputs["drive"] + params["b"]

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        out = jax.random.bernoulli(key, jax.nn.sigmoid(self._logit(inputs, params))).astype(jnp.int32)
        return (out, None) if return_aux else out

    def log_probability(self, inputs, outputs, params, info=None, site_info=None, return_aux=False):
        z = self._logit(inputs, params)
        lp = jnp.where(outputs == 1, jax.nn.log_sigmoid(z), jax.nn.log_sigmoid(-z))
        return (lp, None) if return_aux else lp


class TinyCategorical(AbstractMatrixFactor):
    """A 3-state pdit with no inputs that reports its exact table. Rows = inputs."""

    input_states: dict[str, jax.Array]
    output_states: dict[str, jax.Array]

    def __init__(self):
        self.input_states = {}                            # no inputs: one row
        self.output_states = {"x": jnp.arange(3)}

    def init_params(self, key):
        return {"logits": jnp.array([0.2, 1.0, -0.5])}

    def get_log_probability_matrix(self, params, info=None, site_info=None):
        logits = params["logits"]
        return (logits - jax.scipy.special.logsumexp(logits))[None, :]

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        row = self.get_log_probability_matrix(params, info, site_info)[0]
        out = self.get_nth_output_state(jax.random.categorical(key, row))
        return (out, None) if return_aux else out


class FlipFactor(AbstractReferenceFactor):
    """P(next | state): a two-state telegraph step, int in and int out."""

    input_ports: dict[str, jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports = {"state": BIT}
        self.output_spec = BIT

    def init_params(self, key):
        return {"w": jnp.array(1.5), "b": jnp.array(0.0)}

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        gamma = params["b"] + params["w"] * (2 * inputs["state"] - 1)
        out = jax.random.bernoulli(key, jax.nn.sigmoid(gamma)).astype(jnp.int32)
        return (out, None) if return_aux else out


# 1. A DFG: coin -> cast -> conditional bit. Parents are addresses; porting routes them.
def to_drive(parent_outputs):                              # receives a list, in parent order
    return {"drive": parent_outputs[0].astype(jnp.float32)}


sites = (
    Site("coin", CoinFactor(), (), (), "src", info_key=None, site_info=None),
    Site("out", ConditionalBit(), ("coin",), to_drive, "cond", info_key=None, site_info=None),
)
graph = DFG(sites, {}, "out")                              # (sites, input_ports, output_name)
params = graph.init_params(jax.random.key(0))              # {"src": ..., "cond": ...}
assert set(params) == {"src", "cond"}

N = 60_000
draws = eqx.filter_jit(lambda ks: jax.vmap(lambda k: graph.sample(k, {}, params))(ks))(
    jax.random.split(jax.random.key(1), N))
p_coin = jax.nn.sigmoid(params["src"]["bias"])
w, b = params["cond"]["w"], params["cond"]["b"]
p_exact = float((1 - p_coin) * jax.nn.sigmoid(b) + p_coin * jax.nn.sigmoid(w + b))
assert abs(float(draws.mean()) - p_exact) < 0.01

# Read every site's output through DFGInfo(expose_site_outputs=True).
out, (site_outputs, per_site_aux) = graph.sample(
    jax.random.key(2), {}, params, info=DFGInfo(expose_site_outputs=True), return_aux=True)
assert set(site_outputs) == {"coin", "out"} and int(site_outputs["out"]) == int(out)

# The analytic conditional normalizes.
cond = ConditionalBit()
lp = [cond.log_probability({"drive": jnp.float32(0.5)}, jnp.int32(v), params["cond"]) for v in (0, 1)]
np.testing.assert_allclose(float(jnp.exp(lp[0]) + jnp.exp(lp[1])), 1.0, atol=1e-6)

# 2. A matrix factor: sampling and the table describe the same distribution.
cat = TinyCategorical()
cat_params = cat.init_params(jax.random.key(3))
table = np.asarray(jnp.exp(cat.get_log_probability_matrix(cat_params))[0])
cat_draws = cat.sample_multiple(jax.random.key(4), {}, cat_params, n_samples=50_000)["x"]
np.testing.assert_allclose(np.bincount(np.asarray(cat_draws), minlength=3) / 50_000, table, atol=0.01)
np.testing.assert_allclose(float(cat.log_probability({}, {"x": jnp.array(1)}, cat_params)),
                           np.log(table[1]), atol=1e-5)

# 3. Composites. Same keys in, same draws out, as their manual JAX equivalents.
flip = FlipFactor()
flip_params = flip.init_params(jax.random.key(5))

tiled = TiledFactor(flip, n_tiles=8, weight_tied=True)     # inputs gain a leading 8 axis
key = jax.random.key(6)
tiled_out = tiled.sample(key, {"state": jnp.zeros(8, jnp.int32)}, flip_params)
manual = jax.vmap(lambda k: flip.sample(k, {"state": jnp.array(0, jnp.int32)}, flip_params))(
    jax.random.split(key, 8))
np.testing.assert_array_equal(np.asarray(tiled_out), np.asarray(manual))

chain = ChainFactor(flip, n_steps=5, feedback_porting_fn="state", weight_tied=True)
key = jax.random.key(7)
chain_out = chain.sample(key, {"state": jnp.array(0, jnp.int32)}, flip_params)
state = jnp.array(0, jnp.int32)
for step_key in jax.random.split(key, 5):
    state = flip.sample(step_key, {"state": state}, flip_params)
assert int(chain_out) == int(state)

# Untied composites carry one parameter set per tile (or step) on a leading axis.
untied = TiledFactor(flip, n_tiles=4, weight_tied=False)
assert untied.init_params(jax.random.key(8))["w"].shape == (4,)

# 4. A DeterministicFactor is glue: exact log-probability 0 on its output, -inf elsewhere.
double = DeterministicFactor(lambda inputs, site_info: 2 * inputs["x"], {"x": BIT}, BIT)
assert float(double.log_probability({"x": jnp.int32(3)}, jnp.int32(6), None)) == 0.0
assert float(double.log_probability({"x": jnp.int32(3)}, jnp.int32(5), None)) == -np.inf

print(f"DFG P(out=1): sampled {float(draws.mean()):.4f}, exact {p_exact:.4f}")
print("categorical table", np.round(table, 3))
print("tiled == manual vmap, chain == manual loop: OK")
```

---

## Module map and import surface

**How it composes.** Two public namespaces:

| Import from | What |
|-------------|------|
| `torx` | the factor-graph core: `AbstractFactor`, `AbstractReferenceFactor`, the tractable mixins, `AbstractMatrixFactor`, `DeterministicFactor`, `Site`, `DFG`, `DFGInfo`, `DFGParams`, `TiledFactor`, `ChainFactor` (and their `Abstract*` bases), `AbstractSampler`, `JaxPRNGSampler`, the `PortSpec` / `ParamsTree` / `InfoTree` aliases, `__version__` |
| `torx.psc` | everything circuit-shaped: every gate and gate base, `HybridSites`, `DiscretePCircuit`, `HybridPCircuit`, `HybridState`, the three simulators and their compiled circuits, `GaussianMoments`, `draw_circuit`, `TextDrawing` |

Never import from a private module (`torx.psc.gates._base`, `torx._sampler`): every
public name is re-exported from one of the two namespaces above. Dependencies: JAX,
Equinox, jaxtyping, and `ihoop` (which supplies the strict module base).

**The strict-module rules.** `AbstractFactor`, the simulators and the compiled
circuits derive from `ihoop.eqx.AbstractStrictModule`. Its metaclass enforces, at class
definition time:

1. **Concrete classes are final.** `class MyGate(PNOT)` or `class MyTile(TiledFactor)`
   raises `TypeError`. Subclass an `Abstract*` base instead.
2. **Concrete inherited methods are final.** You may implement abstract methods and
   always `__init__`, but overriding an inherited concrete method (for example
   `sample` of `AbstractAffineGaussianGate`, `sample_multiple`, or
   `sample_with_references` of `AbstractReferenceFactor`) raises `TypeError`.
3. **Names follow abstractness.** A class with unresolved abstract members must be
   named `Abstract...`; a fully concrete class must not be.
4. **`eqx.AbstractVar` fields must be resolved** by real fields (or properties) in the
   concrete class, for example `input_ports` / `output_spec` on a custom factor and
   `sites` / `dims` on a custom gate.

### Source: `torx/__init__.py`

```python
import importlib.metadata

from . import (
    psc as psc,
)
from ._sampler import (
    AbstractSampler as AbstractSampler,
    JaxPRNGSampler as JaxPRNGSampler,
)
from .composite_factors import (
    AbstractChainFactor as AbstractChainFactor,
    AbstractTiledFactor as AbstractTiledFactor,
    ChainFactor as ChainFactor,
    TiledFactor as TiledFactor,
)
from .dfg import (
    AbstractDFG as AbstractDFG,
    DFG as DFG,
    DFGInfo as DFGInfo,
    DFGParams as DFGParams,
    Site as Site,
)
from .factor import (
    AbstractFactor as AbstractFactor,
    AbstractReferenceFactor as AbstractReferenceFactor,
    InfoTree as InfoTree,
    ParamsTree as ParamsTree,
    PortSpec as PortSpec,
)
from .tractable_prob_factors import (
    AbstractEnumerableOutputFactor as AbstractEnumerableOutputFactor,
    AbstractFiniteStateSpaceFactor as AbstractFiniteStateSpaceFactor,
    AbstractHasExplicitOutputDistribution as AbstractHasExplicitOutputDistribution,
    AbstractHasLogProbability as AbstractHasLogProbability,
    AbstractMatrixFactor as AbstractMatrixFactor,
    DeterministicFactor as DeterministicFactor,
)

__version__ = importlib.metadata.version("extro-torx")
```

### Source: `torx/psc/__init__.py`

```python
"""Probabilistic Circuits in JAX."""

import importlib.metadata

from ._circuit import (
    AbstractPCircuit as AbstractPCircuit,
    DiscretePCircuit as DiscretePCircuit,
    HybridPCircuit as HybridPCircuit,
    HybridState as HybridState,
)
from .gates import (
    AbstractAffineGaussianGate as AbstractAffineGaussianGate,
    AbstractContinuousGate as AbstractContinuousGate,
    AbstractControlledContinuousGate as AbstractControlledContinuousGate,
    AbstractDiscreteGate as AbstractDiscreteGate,
    AbstractGeneratorGate as AbstractGeneratorGate,
    AbstractHybridGate as AbstractHybridGate,
    AbstractKBranchGate as AbstractKBranchGate,
    AbstractMultiBinaryPGate as AbstractMultiBinaryPGate,
    AbstractMultiPditGate as AbstractMultiPditGate,
    AbstractPGate as AbstractPGate,
    AbstractSingleBinaryPGate as AbstractSingleBinaryPGate,
    AbstractSinglePditGate as AbstractSinglePditGate,
    AffineGaussianGate as AffineGaussianGate,
    Diffuse as Diffuse,
    Displace as Displace,
    GaussianNoiseGate as GaussianNoiseGate,
    HybridSites as HybridSites,
    JumpDiffusionGate as JumpDiffusionGate,
    Mix as Mix,
    MixtureGaussianGate as MixtureGaussianGate,
    PCNOT as PCNOT,
    PCopy as PCopy,
    PCSWAP as PCSWAP,
    PDEMUX as PDEMUX,
    PditCycle as PditCycle,
    PditShift as PditShift,
    PditSWAP as PditSWAP,
    PISING as PISING,
    PJUMP as PJUMP,
    PMultiCNOT as PMultiCNOT,
    PNOT as PNOT,
    POR as POR,
    PReset as PReset,
    PSWAP as PSWAP,
    Scale as Scale,
)
from .simulation import (
    AbstractCompiledPCircuit as AbstractCompiledPCircuit,
    AbstractSimulator as AbstractSimulator,
    AffineGaussianSimulator as AffineGaussianSimulator,
    BranchingSimulator as BranchingSimulator,
    CompiledAffineGaussianPCircuit as CompiledAffineGaussianPCircuit,
    CompiledBranchingPCircuit as CompiledBranchingPCircuit,
    CompiledStateVectorPCircuit as CompiledStateVectorPCircuit,
    GaussianMoments as GaussianMoments,
    StateVectorSimulator as StateVectorSimulator,
)
from .visualization import draw_circuit as draw_circuit, TextDrawing as TextDrawing

__version__ = importlib.metadata.version("extro-torx")
```

---

## `torx/factor.py`

**How it composes.** The one contract everything honours. A factor is a directed
conditional `P(output | inputs)` that stores structure only; `params` and `info` are
passed on every call. `sample_multiple` splits the key `n_samples` ways and vmaps
`sample` (concrete and final). `AbstractReferenceFactor` supplies
`sample_with_references` by drawing `n_references + 1` samples through
`sample_multiple` and returning the first as the main one; it is the usual base for
custom factors. A custom factor resolves `input_ports` (a dict of
`jax.ShapeDtypeStruct` pytrees) and `output_spec` as `eqx.field(static=True)` fields
and implements `init_params` and `sample`; `sample` returns `(output, aux)` when
`return_aux=True`, usually `(output, None)`.

### Source: `torx/factor.py`

```python
"""Abstract base classes for probabilistic factors."""

from abc import abstractmethod
from typing import Any, Mapping, TypeAlias

import equinox as eqx
import jax
from ihoop.eqx import AbstractStrictModule
from jaxtyping import Array, Key, PyTree

PortSpec: TypeAlias = PyTree[jax.ShapeDtypeStruct]
ParamsTree: TypeAlias = PyTree[Array]
InfoTree: TypeAlias = PyTree
_SampleOutput: TypeAlias = PyTree[Array] | tuple[PyTree[Array], PyTree[Array]]


class AbstractFactor(AbstractStrictModule):
    """Base class for probabilistic factors.

    A directed factor is a conditional distribution
    $P(\\text{output} \\mid \\text{inputs})$.

    Factor methods operate with the following:

    - `params`: the factor's parameter pytree.
    - `info`: optional runtime auxiliary info.
    - `site_info`: optional per-site metadata supplied by the
        surrounding `Site`.
    - `return_aux`: when `True`, methods additionally return a
        factor-defined `aux` pytree.
    """

    input_ports: eqx.AbstractVar[Mapping[str, PortSpec]]
    output_spec: eqx.AbstractVar[PortSpec]

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Draw a single sample from this factor.

        **Arguments:**

        - `key`: PRNG key.
        - `inputs`: Per-port pytree inputs. Keys must equal the
            `self.input_ports` keys.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info.
        - `site_info`: Static per-site metadata supplied by the
            surrounding `Site`.
        - `return_aux`: If `True`, return `(output, aux)`.

        **Returns:**

        The sampled output (a pytree matching `output_spec`), or
        `(output, aux)` when `return_aux=True`.
        """
        raise NotImplementedError

    @abstractmethod
    def init_params(self, key: Key[Array, ""]) -> ParamsTree:
        """Return a freshly-initialised parameter pytree.

        **Arguments:**

        - `key`: PRNG key.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        r"""Draw a "main" sample plus `n_references` additional samples.

        Always returns `(main_output, aux)`. `main_output` is a single
        sample. Every `aux` leaf gains a leading axis of size
        `n_references + 1`: position `0` is the main sample's
        aux, positions `1 .. n_references` are the references' auxes.

        For a sample with no references and no aux, use `sample`.

        **Arguments:**

        - `key`: PRNG key.
        - `inputs`: Per-port pytree inputs, as in `sample`.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.
        - `n_references`: Number of reference samples to draw alongside
            the main sample.

        **Returns:**

        `(main_output, aux)`.
        """
        raise NotImplementedError

    def sample_multiple(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        n_samples: int = 1,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Draw `n_samples` samples by `vmap`-ing `sample`.

        **Arguments:**

        - `key`: PRNG key, split `n_samples` ways.
        - `inputs`: Per-port pytree inputs, as in `sample`.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.
        - `n_samples`: Number of independent samples to draw.
        - `return_aux`: Whether to return the `aux` pytree, as in `sample`.

        **Returns:**

        The stacked outputs, or `(outputs, auxes)` when `return_aux=True`,
        each with a leading axis of size `n_samples`.
        """
        keys = jax.random.split(key, n_samples)

        def _single(
            k: Key[Array, ""],
        ) -> _SampleOutput:
            return self.sample(
                k, inputs, params, info, site_info, return_aux=return_aux
            )

        return jax.vmap(_single)(keys)


class AbstractReferenceFactor(AbstractFactor):
    """Abstract factor + a concrete `sample_with_references`."""

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Generic reference-sampling default.

        Draws `n_references + 1` samples via `sample_multiple` and returns
        the first as the main sample, with all of their auxes stacked.
        """
        samples, aux = self.sample_multiple(
            key,
            inputs,
            params,
            info,
            site_info,
            n_samples=n_references + 1,
            return_aux=True,
        )
        main = jax.tree.map(lambda x: x[0], samples)
        return main, aux
```

---

## `torx/tractable_prob_factors.py`

**How it composes.** Opt-in capabilities layered on the contract:

| Class | Adds | You implement |
|-------|------|---------------|
| `AbstractHasLogProbability` | `log_probability(inputs, outputs, params, info=None, site_info=None, return_aux=False)` | the scalar log-probability |
| `AbstractEnumerableOutputFactor` | finite outputs: `n_output_states`, `get_nth_output_state`, `output_state_to_index` | the ordering |
| `AbstractFiniteStateSpaceFactor` | finite inputs too: `n_input_states`, `get_nth_input_state`, `input_state_to_index` | the ordering |
| `AbstractHasExplicitOutputDistribution` | `get_log_output_distribution(...) -> (n_output_states,)`; `log_probability` indexes it (final) | the log-prob vector |
| `AbstractMatrixFactor` | all of the above from stacked state tables | `input_states`, `output_states`, `get_log_probability_matrix(params, info=None, site_info=None)`, `init_params`, **and `sample`** |
| `DeterministicFactor(fn, input_ports, output_spec)` | `fn(inputs, site_info)`; log-probability 0 on its output, `-inf` elsewhere; `init_params` returns `None` | nothing, it is concrete |

Matrix-factor conventions: `input_states` is a dict of arrays and `output_states` a
pytree of arrays, each with a leading `n_states` axis (an empty dict means one empty
input state). The matrix is `(n_input_states, n_output_states)` with
`[i, j] = log P(output_states[j] | input_states[i])`: **rows are inputs**, the transpose
of a gate's `get_matrix`. A state that is not in the table maps to index `NaN`, which
`log_probability` turns into `-inf`. `AbstractMatrixFactor` does not define `sample`;
draw with `jax.random.categorical` on the row and map back with
`get_nth_output_state` (worked example 4).

### Source: `torx/tractable_prob_factors.py`

```python
from abc import abstractmethod
from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key, PyTree

from ._utils import same_pytree_spec
from .factor import (
    _SampleOutput,
    AbstractFactor,
    AbstractReferenceFactor,
    InfoTree,
    ParamsTree,
    PortSpec,
)


def _state_to_index(
    query: PyTree[Array],
    stacked_states: PyTree[Array],
    spec: PortSpec,
) -> Float[Array, ""]:
    """Index of the first state equal to ``query``.

    ``stacked_states`` is the canonical ordering with a leading axis of size
    ``n_states``; ``query`` is a single state. Returns the matching
    index, or `NaN` if ``query`` is not among the canonical states.
    """
    if not same_pytree_spec(query, spec):
        raise ValueError(f"query {query} does not match the expected spec {spec}.")
    s_leaves = jax.tree.leaves(stacked_states)
    if len(s_leaves) == 0:
        return jnp.array(0.0)
    n = s_leaves[0].shape[0]
    q_leaves = jax.tree.leaves(query)
    matches = jnp.ones((n,), dtype=bool)
    for q, s in zip(q_leaves, s_leaves):
        matches = matches & (s == q[None]).reshape(n, -1).all(axis=1)
    return jnp.where(jnp.any(matches), jnp.argmax(matches).astype(jnp.float32), jnp.nan)


def _n_states_from_stacked(stacked: PyTree[Array], name: str) -> int:
    """Leading-axis size shared by every leaf of a stacked state pytree."""
    leaves = jax.tree.leaves(stacked)
    if len(leaves) == 0:
        return 1
    n = leaves[0].shape[0]
    for i, leaf in enumerate(leaves):
        if leaf.shape[0] != n:
            raise ValueError(
                f"{name} leaf {i} has shape {leaf.shape}, expected leading dim {n}"
            )
    return n


def _state_spec(leaf: Array) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape=leaf.shape[1:], dtype=leaf.dtype)


class AbstractHasLogProbability(AbstractFactor):
    """Capability mixin for factors with a tractable, analytic
    `log_probability(outputs | inputs)`.
    """

    @abstractmethod
    def log_probability(
        self,
        inputs: Mapping[str, PyTree[Array]],
        outputs: PyTree[Array],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Float[Array, ""] | tuple[Float[Array, ""], PyTree[Array]]:
        """Scalar log-probability of `outputs` given `inputs`.

        **Arguments:**

        - `inputs`: Per-port pytree inputs, as in `sample`.
        - `outputs`: An output pytree matching `output_spec`.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.
        - `return_aux`: Whether to additionally return a factor-defined
            `aux` pytree.

        **Returns:**

        The scalar `log P(outputs | inputs)`, or `(log_p, aux)` when
        `return_aux=True`.
        """
        raise NotImplementedError


class AbstractEnumerableOutputFactor(AbstractFactor):
    """Factor whose output state space is finite, with a canonical ordering."""

    @property
    @abstractmethod
    def n_output_states(self) -> int:
        """Number of output states in the canonical ordering."""
        raise NotImplementedError

    @abstractmethod
    def get_nth_output_state(self, n: int | Int[Array, ""]) -> PyTree[Array]:
        """Return the `n`-th output state in the canonical ordering.

        The returned pytree has the same structure as the output of
        [`torx.AbstractFactor.sample`][].
        """
        raise NotImplementedError

    @abstractmethod
    def output_state_to_index(self, outputs: PyTree[Array]) -> Float[Array, ""]:
        """Index of `outputs` in the canonical ordering, as a scalar float."""
        raise NotImplementedError


class AbstractFiniteStateSpaceFactor(AbstractEnumerableOutputFactor):
    """Factor whose input and output state spaces are both finite."""

    @property
    @abstractmethod
    def n_input_states(self) -> int:
        """Number of input states in the canonical ordering."""
        raise NotImplementedError

    @abstractmethod
    def get_nth_input_state(
        self, n: int | Int[Array, ""]
    ) -> Mapping[str, PyTree[Array]]:
        """Return the `n`-th input state in the canonical ordering.

        The returned pytree has the same structure as the `inputs` argument
        to [`torx.AbstractFactor.sample`][].
        """
        raise NotImplementedError

    @abstractmethod
    def input_state_to_index(
        self, inputs: Mapping[str, PyTree[Array]]
    ) -> Float[Array, ""]:
        """Index of `inputs` in the canonical ordering, as a scalar float."""
        raise NotImplementedError


class AbstractHasExplicitOutputDistribution(
    AbstractEnumerableOutputFactor, AbstractHasLogProbability
):
    """Factor whose conditional is available in closed form as an explicit
    log-prob vector over the enumerable output states.
    """

    @abstractmethod
    def get_log_output_distribution(
        self,
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> (
        Float[Array, " n_output_states"]
        | tuple[Float[Array, " n_output_states"], PyTree[Array]]
    ):
        """Log-probabilities of all `n_output_states` outputs given `inputs`.

        Returns a vector `v` of shape `(n_output_states,)` with
        `v[j] = log P(get_nth_output_state(j) | inputs)`.

        **Arguments:**

        - `inputs`: Per-port pytree inputs, as in `sample`.
        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.
        - `return_aux`: Whether to additionally return a factor-defined `aux`.

        **Returns:**

        The length-`n_output_states` log-probability vector, or `(v, aux)` when
        `return_aux=True`.
        """
        raise NotImplementedError

    def log_probability(
        self,
        inputs: Mapping[str, PyTree[Array]],
        outputs: PyTree[Array],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Float[Array, ""] | tuple[Float[Array, ""], PyTree[Array]]:
        """Index `get_log_output_distribution` at the queried `outputs`."""
        log_dist, aux = self.get_log_output_distribution(
            inputs, params, info, site_info, return_aux=True
        )

        n_out = self.output_state_to_index(outputs)
        out_valid = ~jnp.isnan(n_out)
        j = jnp.where(out_valid, n_out, 0.0).astype(jnp.int32)
        log_prob = jnp.where(out_valid, log_dist[j], -jnp.inf)
        log_prob = jnp.where(jnp.all(jnp.isnan(log_dist)), jnp.nan, log_prob)

        if return_aux:
            return log_prob, aux
        return log_prob


class AbstractMatrixFactor(
    AbstractHasExplicitOutputDistribution,
    AbstractFiniteStateSpaceFactor,
    AbstractReferenceFactor,
):
    """A finite-state factor whose conditional is given explicitly as a matrix."""

    input_states: eqx.AbstractVar[dict[str, PyTree[Array]]]
    output_states: eqx.AbstractVar[PyTree[Array]]

    @property
    def input_ports(self) -> Mapping[str, PortSpec]:  # type: ignore[override]
        return jax.tree.map(_state_spec, self.input_states)

    @property
    def output_spec(self) -> PortSpec:
        return jax.tree.map(_state_spec, self.output_states)

    @property
    def n_input_states(self) -> int:
        return _n_states_from_stacked(self.input_states, "input_states")

    @property
    def n_output_states(self) -> int:
        return _n_states_from_stacked(self.output_states, "output_states")

    def get_nth_input_state(
        self, n: int | Int[Array, ""]
    ) -> Mapping[str, PyTree[Array]]:
        """See [`torx.AbstractFiniteStateSpaceFactor.get_nth_input_state`][]."""
        return jax.tree.map(lambda x: x[n], self.input_states)

    def get_nth_output_state(self, n: int | Int[Array, ""]) -> PyTree[Array]:
        """See [`torx.AbstractFiniteStateSpaceFactor.get_nth_output_state`][]."""
        return jax.tree.map(lambda x: x[n], self.output_states)

    def input_state_to_index(
        self, inputs: Mapping[str, PyTree[Array]]
    ) -> Float[Array, ""]:
        """See [`torx.AbstractFiniteStateSpaceFactor.input_state_to_index`][]."""
        return _state_to_index(inputs, self.input_states, self.input_ports)

    def output_state_to_index(self, outputs: PyTree[Array]) -> Float[Array, ""]:
        """See [`torx.AbstractEnumerableOutputFactor.output_state_to_index`][]."""
        return _state_to_index(outputs, self.output_states, self.output_spec)

    @abstractmethod
    def get_log_probability_matrix(
        self,
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
    ) -> Float[Array, "n_input_states n_output_states"]:
        """Return the `(n_input_states, n_output_states)` log-probability matrix.

        Entry `[i, j]` is `log P(output_states[j] | input_states[i])`.

        **Arguments:**

        - `params`: Parameter pytree for this factor.
        - `info`: Runtime auxiliary info, as in `sample`.
        - `site_info`: Static per-site metadata, as in `sample`.

        **Returns:**

        The matrix of log-probabilities.
        """
        raise NotImplementedError

    def get_log_output_distribution(
        self,
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> (
        Float[Array, " n_output_states"]
        | tuple[Float[Array, " n_output_states"], PyTree[Array]]
    ):
        """The matrix row selected by the input's canonical index."""
        n_in = self.input_state_to_index(inputs)
        matrix = self.get_log_probability_matrix(params, info, site_info)

        in_valid = ~jnp.isnan(n_in)
        i = jnp.where(in_valid, n_in, 0.0).astype(jnp.int32)
        log_dist = jnp.where(in_valid, matrix[i], jnp.nan)

        if return_aux:
            return log_dist, None
        return log_dist


class DeterministicFactor(AbstractHasLogProbability):
    """A factor whose output is a deterministic function of its inputs.

    **Arguments:**

    - `fn`: A pure function `fn(inputs, site_info) -> output` mapping the
        per-port `inputs` dict (and the surrounding `Site`'s static
        `site_info`) to an output pytree matching `output_spec`.
    - `input_ports`: The factor's input-port specs.
    - `output_spec`: The spec of `fn`'s output.
    """

    fn: Callable[[Mapping[str, PyTree[Array]], Any], PyTree[Array]] = eqx.field(
        static=True
    )
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True, converter=dict)
    output_spec: PortSpec = eqx.field(static=True)

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Return `fn(inputs, site_info)`."""
        output = self.fn(inputs, site_info)
        if return_aux:
            return output, None
        return output

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Evaluate `fn` once."""
        return self.fn(inputs, site_info), None

    def init_params(self, key: Key[Array, ""]) -> ParamsTree:
        return None

    def log_probability(
        self,
        inputs: Mapping[str, PyTree[Array]],
        outputs: PyTree[Array],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> Float[Array, ""] | tuple[Float[Array, ""], PyTree[Array]]:
        """`0` if `outputs == fn(inputs, site_info)`, else `-inf`."""
        equal = eqx.tree_equal(self.fn(inputs, site_info), outputs)
        log_prob = jnp.where(equal, 0.0, -jnp.inf)

        if return_aux:
            return log_prob, None
        return log_prob
```

---

## `torx/dfg.py`

**How it composes.** `Site` places a factor at an address; `DFG` validates the wiring,
topologically sorts it (Kahn), and walks it eagerly. Details that matter when writing
graphs:

- `parents` and a tuple `porting_fn` reject bare strings (`TypeError`): write
  `("coin",)`, not `"coin"`. A tuple `porting_fn` must name exactly the factor's input
  ports, one per parent; a callable receives `parent_outputs` as a **list** in parent
  order and returns the input dict.
- Validation at construction: unique site names, site names disjoint from input-port
  names, every parent and `output_name` a known address, no cycles.
- `sample` splits the key once per site (indexed by site position, not topological
  order), so each site's draw is reproducible from `(key, site index)`.
- `params[site.param_key]` selects each site's slice; `None` passes `None`. Any
  indexable pytree works, so circuits use integer keys into the thetas list.
  `init_params` initializes once per distinct `param_key`, which is how weight tying
  across sites works.
- `info` is `None` or a `DFGInfo`; `entries[info_key]` per site, otherwise
  `default_site_info` (pass an `AbstractSampler` there to route every gate's draws
  through one provider). With `expose_site_outputs=True` and `return_aux=True`, aux is
  `(dict of every site's output, per-site aux tuple)`.

### Source: `torx/dfg.py`

```python
from collections import deque
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from .factor import AbstractFactor, InfoTree, ParamsTree, PortSpec


def _convert_parents(parents):
    if isinstance(parents, str):
        raise TypeError(
            f"parents={parents!r} is a bare string; use [{parents!r}] for a "
            f"single parent."
        )
    return tuple(parents)


def _convert_porting_fn(porting_fn: Any) -> Any:
    """Pass a callable through; otherwise convert to a tuple of port names."""
    if callable(porting_fn):
        return porting_fn
    if isinstance(porting_fn, str):
        raise TypeError(
            f"porting_fn={porting_fn!r} is a bare string; use [{porting_fn!r}] "
            f"for a single port."
        )
    return tuple(porting_fn)


class Site(eqx.Module):
    """The placement of a `Factor` at one position in a DFG."""

    name: str = eqx.field(static=True)
    factor: AbstractFactor
    parents: tuple[str, ...] = eqx.field(
        static=True, converter=_convert_parents
    )  # parent addresses, in order
    porting_fn: (
        Callable[[Sequence[PyTree[Array]]], Mapping[str, PyTree[Array]]]
        | tuple[str, ...]
    ) = eqx.field(
        static=True, converter=_convert_porting_fn
    )  # routes parent outputs into the factor's inputs
    param_key: str | int | None = eqx.field(static=True)  # address into the params tree
    info_key: str | None = eqx.field(static=True)  # key into the DFG's info entries
    site_info: Any = eqx.field(static=True)  # per-site metadata for the factor


Site.__init__.__doc__ = """Construct a `Site`.

**Arguments:**

- `name`: This site's address in the DFG namespace. Unique among sites and
    disjoint from the input-port names.
- `factor`: The `Factor` placed here.
- `parents`: Addresses (input ports or site names) feeding this factor;
    `porting_fn` maps them onto its named input ports.
- `porting_fn`: How parents route into the factor's input dict: a tuple of
    port names (1:1 with `parents`), or a callable `parents -> input dict`
    for non-trivial routing.
- `param_key`: Address of this site's parameters within the DFG's `params`.
- `info_key`: Like `param_key`, for runtime info; `None` passes `info=None`.
- `site_info`: per-site metadata passed to the factor.
"""


DFGParams: TypeAlias = PyTree[Array]


class DFGInfo(eqx.Module):
    """DFG-level runtime info.

    Separate from per-site info, which lives in `entries` under each site's
    `info_key`; this configures how the DAG itself runs.

    **Arguments:**

    - `expose_site_outputs`: When `True` (and `aux` is requested), prepend a
        name-keyed dict of every site's main output to the aux return.
    - `entries`: Per-`info_key` mapping, scattered to sites by
        `distribute_info`. A child-`DFG` site's entry is itself a `DFGInfo`.
    - `default_site_info`: Runtime info for sites without an `info_key`.
        Pass an `AbstractSampler` here for factors that consume a distribution
        provider. Explicitly keyed entries take precedence.
    """

    expose_site_outputs: bool = eqx.field(static=True)
    entries: Mapping[str, PyTree[Array]] = eqx.field(default_factory=dict)
    default_site_info: InfoTree = None


class AbstractDFG(AbstractFactor):
    """A `Factor` built as a DAG of placed factors.

    Holds the structure and concretises `sample` / `sample_with_references`
    as an eager topological walk.

    The DFG owns a flat namespace of addresses, one per input port and one
    per `Site`, and the two sets must be disjoint. Every `parents` entry and
    `output_name` is such an address. A DFG is itself a `Factor`, so it can
    nest as the `factor` of a `Site`.
    """

    sites: eqx.AbstractVar[tuple[Site, ...]]
    output_name: eqx.AbstractVar[str]
    topological_order: eqx.AbstractVar[tuple[int, ...]]
    sites_by_name: eqx.AbstractVar[Mapping[str, int]]

    @staticmethod
    def _derive(
        sites: tuple[Site, ...],
        input_ports: Mapping[str, PortSpec],
        output_name: str,
    ) -> tuple[dict[str, PortSpec], dict[str, int], tuple[int, ...], PortSpec]:
        """Validate the DAG and derive the fields a final class must set."""
        input_ports = dict(input_ports)
        sites_by_name = AbstractDFG._validate(sites, input_ports, output_name)
        topological_order = AbstractDFG._topo_sort(sites, sites_by_name)
        if output_name in input_ports:
            output_spec = input_ports[output_name]
        else:
            output_spec = sites[sites_by_name[output_name]].factor.output_spec
        return input_ports, sites_by_name, topological_order, output_spec

    @staticmethod
    def _validate(
        sites: tuple[Site, ...],
        input_ports: Mapping[str, PortSpec],
        output_name: str,
    ) -> dict[str, int]:
        site_names = [s.name for s in sites]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for n in site_names:
            if n in seen:
                duplicates.add(n)
            seen.add(n)
        if duplicates:
            raise ValueError(f"Duplicate site names: {sorted(duplicates)}")

        # Disjointness with input port names.
        overlap = set(site_names) & set(input_ports.keys())
        if overlap:
            raise ValueError(
                f"Site names overlap with input port names: {sorted(overlap)}"
            )

        sites_by_name = {s.name: i for i, s in enumerate(sites)}

        valid_names = set(input_ports) | set(sites_by_name)
        if output_name not in valid_names:
            raise ValueError(
                f"output_name {output_name!r} is not a DFG address; input ports: "
                f"{tuple(input_ports)}, site names: {tuple(sites_by_name)}."
            )

        for site in sites:
            unknown_parents = tuple(p for p in site.parents if p not in valid_names)
            if unknown_parents:
                raise ValueError(
                    f"Site {site.name!r} has unknown parent(s) "
                    f"{unknown_parents}; input ports: {tuple(input_ports)}, "
                    f"site names: {tuple(sites_by_name)}."
                )

            if callable(site.porting_fn):
                continue

            port_names = tuple(site.porting_fn)
            if len(port_names) != len(site.parents):
                raise ValueError(
                    f"Site {site.name!r} has {len(site.parents)} parent(s) but "
                    f"tuple porting_fn has {len(port_names)} port(s)."
                )

            seen_ports = set()
            duplicate_ports = []
            for port in port_names:
                if port in seen_ports and port not in duplicate_ports:
                    duplicate_ports.append(port)
                seen_ports.add(port)
            if duplicate_ports:
                raise ValueError(
                    f"Site {site.name!r} tuple porting_fn contains duplicate "
                    f"port(s): {duplicate_ports}."
                )

            actual_ports = set(port_names)
            expected_ports = set(site.factor.input_ports)
            if actual_ports != expected_ports:
                missing = expected_ports - actual_ports
                extra = actual_ports - expected_ports
                raise ValueError(
                    f"Site {site.name!r} tuple porting_fn must match factor "
                    f"input ports; missing {missing}, extra {extra}."
                )

        return sites_by_name

    @staticmethod
    def _topo_sort(
        sites: tuple[Site, ...],
        sites_by_name: Mapping[str, int],
    ) -> tuple[int, ...]:
        """Kahn's algorithm"""
        in_degree = [0] * len(sites)
        adj = [[] for _ in range(len(sites))]
        for j, site in enumerate(sites):
            for parent_name in site.parents:
                if parent_name in sites_by_name:
                    i = sites_by_name[parent_name]
                    in_degree[j] += 1
                    adj[i].append(j)

        queue = deque(i for i in range(len(sites)) if in_degree[i] == 0)
        topo = []
        while queue:
            i = queue.popleft()
            topo.append(i)
            for j in adj[i]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

        if len(topo) != len(sites):
            remaining = [s.name for k, s in enumerate(sites) if in_degree[k] > 0]
            raise ValueError(
                f"Cycle detected in DAG. Sites still with incoming edges: "
                f"{remaining}"
            )
        return tuple(topo)

    @staticmethod
    def _init_params(sites: tuple[Site, ...], key: Key[Array, ""]) -> DFGParams:
        """Initialise parameters once per distinct `param_key` across `sites`."""
        seen = {}
        for site in sites:
            if site.param_key is not None and site.param_key not in seen:
                seen[site.param_key] = site.factor

        if not seen:
            return {}

        pkeys = list(seen.keys())
        rng_keys = jax.random.split(key, len(pkeys))
        return {
            pkey: seen[pkey].init_params(rng_keys[i]) for i, pkey in enumerate(pkeys)
        }

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: DFGParams,
        info: DFGInfo | None = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> PyTree[Array] | tuple[PyTree[Array], tuple[PyTree[Array], ...]]:
        """Run the DAG once and return the `output_name` value.

        Samples each site in topological order, routing parent outputs
        through its `porting_fn`.
        """
        values, auxes = self._run(
            key, inputs, params, info, n_references=None, with_aux=return_aux
        )
        output = values[self.output_name]
        if return_aux:
            return output, auxes
        return output

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: DFGParams,
        info: DFGInfo | None = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], tuple[PyTree[Array], ...]]:
        """Like `sample`, but draws each site via its factor's
        `sample_with_references`, so every site yields `n_references + 1`
        samples. Returns `(output, per_site_auxes)`.
        """
        values, auxes = self._run(
            key, inputs, params, info, n_references=n_references, with_aux=True
        )
        return values[self.output_name], auxes

    def _run(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: DFGParams,
        info: DFGInfo | None,
        n_references: int | None,
        with_aux: bool,
    ) -> tuple[dict[str, PyTree[Array]], tuple[PyTree[Array], ...]]:
        """Shared eager topological walk for `sample` / `sample_with_references`."""
        per_site_params = self.distribute_params(params)
        per_site_info = self.distribute_info(info)
        if len(self.sites) > 0:
            keys = jax.random.split(key, len(self.sites))
        else:
            keys = jnp.empty((0, 2), dtype=jnp.uint32)

        values = dict(inputs)
        auxes = [None] * len(self.sites)

        for site_idx in self.topological_order:
            site = self.sites[site_idx]
            parent_outputs = [values[a] for a in site.parents]
            if callable(site.porting_fn):
                site_inputs = site.porting_fn(parent_outputs)
            else:
                site_inputs = dict(zip(site.porting_fn, parent_outputs))
            k = keys[site_idx]
            par = per_site_params[site_idx]
            inf = per_site_info[site_idx]
            si = site.site_info

            if n_references is None:
                if with_aux:
                    output, aux = site.factor.sample(
                        k, site_inputs, par, inf, si, return_aux=True
                    )
                else:
                    output = site.factor.sample(
                        k, site_inputs, par, inf, si, return_aux=False
                    )
                    aux = None
            else:
                output, aux = site.factor.sample_with_references(
                    k,
                    site_inputs,
                    par,
                    inf,
                    si,
                    n_references=n_references,
                )

            values[site.name] = output
            auxes[site_idx] = aux

        aux_out = tuple(auxes)
        if with_aux and info is not None and info.expose_site_outputs:
            site_outputs = {s.name: values[s.name] for s in self.sites}
            if n_references is not None:
                n_samples = n_references + 1
                site_outputs = jax.tree.map(
                    lambda x: jnp.broadcast_to(
                        jnp.expand_dims(x, 0), (n_samples,) + x.shape
                    ),
                    site_outputs,
                )
            aux_out = (site_outputs, aux_out)

        return values, aux_out

    def distribute_params(self, params: DFGParams) -> tuple[ParamsTree, ...]:
        """Scatter the shared params mapping into a per-site tuple."""
        return tuple(
            params[site.param_key] if site.param_key is not None else None
            for site in self.sites
        )

    def gather_param_grads(
        self,
        params: DFGParams,
        site_grads: Mapping[int, ParamsTree],
    ) -> DFGParams:
        """Gather per-site parameter gradients into the shared params dict."""

        def dot_fn(p: DFGParams):
            per_site = self.distribute_params(p)
            total = 0.0
            for idx, g in site_grads.items():
                total = total + sum(
                    jnp.vdot(a, b)
                    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(per_site[idx]))
                )
            return total

        return eqx.filter_grad(dot_fn)(params)

    def distribute_info(self, info: DFGInfo | None) -> tuple[InfoTree, ...]:
        """Scatter `info.entries` into a per-site tuple."""
        if info is None:
            return tuple(None for _ in self.sites)
        return tuple(
            (
                info.entries[site.info_key]
                if site.info_key is not None
                else info.default_site_info
            )
            for site in self.sites
        )


class DFG(AbstractDFG):
    """A concrete, eagerly-walked DAG of factors."""

    sites: tuple[Site, ...]
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)
    output_name: str = eqx.field(static=True)
    topological_order: tuple[int, ...] = eqx.field(static=True)
    sites_by_name: Mapping[str, int] = eqx.field(static=True)

    def __init__(
        self,
        sites: tuple[Site, ...],
        input_ports: Mapping[str, PortSpec],
        output_name: str,
    ) -> None:
        input_ports, sites_by_name, topological_order, output_spec = self._derive(
            sites, input_ports, output_name
        )
        self.sites = sites
        self.input_ports = input_ports
        self.output_name = output_name
        self.topological_order = topological_order
        self.sites_by_name = sites_by_name
        self.output_spec = output_spec

    def init_params(self, key: Key[Array, ""]) -> DFGParams:
        """Initialise parameters once per distinct `param_key`."""
        return self._init_params(self.sites, key)
```

---

## `torx/composite_factors.py`

**How it composes.** Two wrappers that turn one factor into many draws without changing
the calling contract.

- `TiledFactor(base, n_tiles, weight_tied, *, batch_size=None, slice_info=False)`:
  every input port and the output gain a leading `n_tiles` axis; equivalent to
  vmapping `base.sample` over `jax.random.split(key, n_tiles)`. `weight_tied=False`
  means params carry a leading `n_tiles` axis (and `init_params` vmaps). A
  `batch_size` below `n_tiles` switches to `jax.lax.map` in chunks to bound memory.
- `ChainFactor(base, n_steps, feedback_porting_fn, weight_tied, *, slice_info=False)`:
  a `lax.scan` over `jax.random.split(key, n_steps)`. Feedback ports receive the
  previous step's output (the caller's value at step 0); all other ports are broadcast
  unchanged. A string `feedback_porting_fn` routes the whole output into that port. The
  feedback spec must match the port spec exactly (shape and dtype) or construction
  raises `ValueError`, so an int output can only feed an int port. `sample` returns the
  final step's output; with `return_aux=True` the per-step aux is stacked on a leading
  `n_steps` axis.

### Source: `torx/composite_factors.py`

```python
from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from ._utils import same_pytree_spec
from .factor import _SampleOutput, AbstractFactor, InfoTree, ParamsTree, PortSpec


def _prepend_axis(spec: PortSpec, n: int) -> PortSpec:
    """Prepend a leading axis of size `n` to every `ShapeDtypeStruct` in
    `spec`.
    """
    return jax.tree.map(
        lambda s: jax.ShapeDtypeStruct((n,) + tuple(s.shape), s.dtype),
        spec,
    )


class AbstractTiledFactor(AbstractFactor):
    """Base implementation for factors tiled across independent replicas.

    `batch_size` chooses the mapping mode: `None`, `0`, or any value
    `>= n_tiles` uses `jax.vmap` (all tiles at once); a smaller value uses
    `jax.lax.map` with chunks of that size.

    The `info` argument is the runtime info forwarded to each tile's
    `base.sample`. By default the same `info` is broadcast to every tile; set
    `slice_info=True` to instead pass per-tile info.

    **Arguments:**

    - `base`: The factor to replicate.
    - `n_tiles`: Number of tiles.
    - `weight_tied`: Whether all tiles share one parameter set (see above).
    - `batch_size`: `vmap` vs `lax.map` execution strategy (see above).
    - `slice_info`: Whether `info` is sliced per tile (see above).
    """

    base: eqx.AbstractVar[AbstractFactor]
    n_tiles: eqx.AbstractVar[int]
    weight_tied: eqx.AbstractVar[bool]
    batch_size: eqx.AbstractVar[int | None]
    slice_info: eqx.AbstractVar[bool]
    input_ports: eqx.AbstractVar[Mapping[str, PortSpec]]
    output_spec: eqx.AbstractVar[PortSpec]

    def __init__(
        self,
        base: AbstractFactor,
        n_tiles: int,
        weight_tied: bool,
        *,
        batch_size: int | None = None,
        slice_info: bool = False,
    ) -> None:
        self.base = base
        self.n_tiles = n_tiles
        self.weight_tied = weight_tied
        self.batch_size = batch_size
        self.slice_info = slice_info
        self.input_ports = {
            name: _prepend_axis(spec, n_tiles)
            for name, spec in base.input_ports.items()
        }
        self.output_spec = _prepend_axis(base.output_spec, n_tiles)

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Sample every tile and stack the results along a leading `n_tiles`
        axis. With `return_aux=True` each tile's aux is stacked likewise.
        """
        main, aux = self._run(
            key,
            inputs,
            params,
            info,
            site_info,
            n_references=None,
            with_aux=return_aux,
        )
        return (main, aux) if return_aux else main

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Per-tile [`torx.AbstractFactor.sample_with_references`][].

        Each tile calls `base.sample_with_references(n_references)` and the
        results are stacked across tiles. The `main` output leaves have shape
        `(n_tiles, *base_output_shape)` and aux leaves have shape
        `(n_references + 1, n_tiles, *base_aux_leaf_shape)`.
        """
        main, aux = self._run(
            key,
            inputs,
            params,
            info,
            site_info,
            n_references=n_references,
            with_aux=True,
        )
        # `_run` stacks per-tile results tile-first, giving aux leaves
        # `(n_tiles, n_references + 1, ...)`.
        aux = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), aux)
        return main, aux

    def _run(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree,
        site_info: Any,
        n_references: int | None,
        with_aux: bool,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """`vmap` or `lax.map` the base factor across tiles.

        Returns `(stacked_main, stacked_aux_or_None_tree)`.
        """
        keys = jax.random.split(key, self.n_tiles)
        base = self.base
        slice_info = self.slice_info
        batch_size = self.batch_size
        weight_tied = self.weight_tied

        def single(args):
            # `args[2]` is the per-tile `params` slice when untied (else `None`,
            # tied `params` are used); `args[3]` is the per-tile `info` slice when
            # `slice_info` (else `None`, and the broadcast `info` is used).
            k, inp = args[0], args[1]
            par = params if weight_tied else args[2]
            inf = args[3] if slice_info else info
            if n_references is None:
                if with_aux:
                    return base.sample(k, inp, par, inf, site_info, return_aux=True)
                return (
                    base.sample(k, inp, par, inf, site_info, return_aux=False),
                    None,
                )
            return base.sample_with_references(
                k, inp, par, inf, site_info, n_references=n_references
            )

        param_xs = None if weight_tied else params
        info_xs = info if slice_info else None
        xs = (keys, dict(inputs), param_xs, info_xs)
        if batch_size is None or batch_size == 0 or batch_size >= self.n_tiles:
            return jax.vmap(single)(xs)
        return jax.lax.map(single, xs, batch_size=batch_size)

    def init_params(self, key: Key[Array, ""]) -> ParamsTree:
        """`base.init_params` when `weight_tied`, else `vmap`-ed over `n_tiles`."""
        if self.weight_tied:
            return self.base.init_params(key)
        keys = jax.random.split(key, self.n_tiles)
        return jax.vmap(self.base.init_params)(keys)


class TiledFactor(AbstractTiledFactor):
    """Replication of a base factor across `n_tiles` independent tiles."""

    base: AbstractFactor
    n_tiles: int = eqx.field(static=True)
    weight_tied: bool = eqx.field(static=True)
    batch_size: int | None = eqx.field(static=True)
    slice_info: bool = eqx.field(static=True)
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)


class AbstractChainFactor(AbstractFactor):
    """Base implementation for factors chained via `jax.lax.scan`.

    At each step the base's input ports split into two disjoint sets:

    - Feedback ports: receive `feedback_porting_fn(previous step's main
        output)`; at step 0 they take the chain's initial state.
    - Broadcast ports: get the caller's value unchanged at every step.

    `feedback_porting_fn` is either a port-name `str` (for `lambda main: {str: main}`)
    or a `Callable` mapping the previous main output to a dict of feedback-port values.

    **Arguments:**

    - `base`: The factor applied at every step.
    - `n_steps`: Number of steps.
    - `feedback_porting_fn`: Port-name `str` or `Callable` routing each step's
        main output into the next step.
    - `weight_tied`: Whether all steps share one parameter set.
    - `slice_info`: Whether `info` is sliced per step.
    """

    base: eqx.AbstractVar[AbstractFactor]
    n_steps: eqx.AbstractVar[int]
    # `None` when built from the port-name `str` shorthand (the single feedback
    # port is then `feedback_ports[0]`); the user's callable otherwise.
    feedback_porting_fn: eqx.AbstractVar[
        Callable[[PyTree[Array]], Mapping[str, PyTree[Array]]] | None
    ]
    feedback_ports: eqx.AbstractVar[tuple[str, ...]]
    all_step_input_ports: eqx.AbstractVar[tuple[str, ...]]
    weight_tied: eqx.AbstractVar[bool]
    slice_info: eqx.AbstractVar[bool]
    input_ports: eqx.AbstractVar[Mapping[str, PortSpec]]
    output_spec: eqx.AbstractVar[PortSpec]

    def __init__(
        self,
        base: AbstractFactor,
        n_steps: int,
        feedback_porting_fn: (
            Callable[[PyTree[Array]], Mapping[str, PyTree[Array]]] | str
        ),
        weight_tied: bool,
        *,
        slice_info: bool = False,
    ) -> None:
        if isinstance(feedback_porting_fn, str):
            fb_port = feedback_porting_fn
            if fb_port not in base.input_ports:
                raise ValueError(
                    f"feedback_porting_fn '{fb_port}' is not an input port of "
                    f"base factor (available: {base.input_ports.keys()})."
                )
            feedback_ports = (fb_port,)
            feedback_struct = {fb_port: base.output_spec}
            stored_porting_fn = None
        elif callable(feedback_porting_fn):
            # dict from feedback-port name to spec
            feedback_struct = dict(
                eqx.filter_eval_shape(feedback_porting_fn, base.output_spec)
            )
            feedback_ports = tuple(feedback_struct.keys())
            stored_porting_fn = feedback_porting_fn

        missing = set(feedback_ports) - set(base.input_ports.keys())
        if missing:
            raise ValueError(
                f"feedback_porting_fn produces port(s) {sorted(missing)} not in "
                f"base.input_ports {base.input_ports.keys()}."
            )

        fb_in_spec = {k: base.input_ports[k] for k in feedback_ports}
        if not same_pytree_spec(feedback_struct, fb_in_spec):
            raise ValueError(
                f"feedback_porting_fn output spec "
                f"{eqx.tree_pformat(feedback_struct, struct_as_array=True)} does "
                f"not match base.input_ports spec "
                f"{eqx.tree_pformat(fb_in_spec, struct_as_array=True)}."
            )

        all_step_input_ports = tuple(
            sorted(set(base.input_ports.keys()) - set(feedback_ports))
        )

        self.base = base
        self.n_steps = n_steps
        self.feedback_porting_fn = stored_porting_fn
        self.feedback_ports = feedback_ports
        self.all_step_input_ports = all_step_input_ports
        self.weight_tied = weight_tied
        self.slice_info = slice_info
        self.input_ports = dict(base.input_ports)
        self.output_spec = base.output_spec

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Run the scan and return the final step's output. With
        `return_aux=True` additionally returns the per-step aux stacked along
        a leading axis of size `n_steps`.
        """
        final_main, aux_trace = self._run(
            key,
            inputs,
            params,
            info,
            site_info,
            n_references=None,
            with_aux=return_aux,
        )
        return (final_main, aux_trace) if return_aux else final_main

    def sample_with_references(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        n_references: int = 1,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Per-step [`torx.AbstractFactor.sample_with_references`][] along the
        scan.

        Each step calls `base.sample_with_references(n_references)`; the main
        output of the step is threaded as the scan carry.The returned `aux_trace`
        leaves have shape `(n_references + 1, n_steps, *base_aux_leaf_shape)`.
        """
        final_main, aux_trace = self._run(
            key,
            inputs,
            params,
            info,
            site_info,
            n_references=n_references,
            with_aux=True,
        )
        aux_trace = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), aux_trace)
        return final_main, aux_trace

    def _run(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree,
        site_info: Any,
        n_references: int | None,
        with_aux: bool,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        """Run the scan over base steps. Returns `(final_main, aux_trace)`."""
        keys = jax.random.split(key, self.n_steps)
        all_step_inputs = {k: inputs[k] for k in self.all_step_input_ports}
        initial_feedback = {k: inputs[k] for k in self.feedback_ports}
        initial_step_main = jax.tree.map(
            lambda s: jnp.zeros(s.shape, s.dtype), self.base.output_spec
        )
        base = self.base
        feedback_porting_fn = self.feedback_porting_fn
        feedback_ports = self.feedback_ports
        tied = self.weight_tied
        slice_info = self.slice_info

        def step_fn(carry, scan_input):
            step_key, step_params_in, step_info_in = scan_input
            step_params = params if tied else step_params_in
            step_info = step_info_in if slice_info else info
            feedback, _prev_main = carry
            step_inputs = feedback | all_step_inputs
            if n_references is None:
                if with_aux:
                    step_main, step_aux = base.sample(
                        step_key,
                        step_inputs,
                        step_params,
                        step_info,
                        site_info,
                        return_aux=True,
                    )
                else:
                    step_main = base.sample(
                        step_key,
                        step_inputs,
                        step_params,
                        step_info,
                        site_info,
                        return_aux=False,
                    )
                    step_aux = None
            else:
                step_main, step_aux = base.sample_with_references(
                    step_key,
                    step_inputs,
                    step_params,
                    step_info,
                    site_info,
                    n_references=n_references,
                )
            if feedback_porting_fn is not None:
                next_feedback = feedback_porting_fn(step_main)
            else:
                next_feedback = {feedback_ports[0]: step_main}
            return (next_feedback, step_main), step_aux

        params_xs = None if tied else params
        info_xs = info if slice_info else None
        scan_xs = (keys, params_xs, info_xs)
        initial_carry = (initial_feedback, initial_step_main)
        (_, final_main), aux_trace = jax.lax.scan(step_fn, initial_carry, scan_xs)
        return final_main, aux_trace

    def init_params(self, key: Key[Array, ""]) -> ParamsTree:
        """`vmap(base.init_params)` per step, or one shared init when
        `weight_tied`.
        """
        if self.weight_tied:
            return self.base.init_params(key)
        keys = jax.random.split(key, self.n_steps)
        return jax.vmap(self.base.init_params)(keys)


class ChainFactor(AbstractChainFactor):
    """Sequential composition of a base factor via `jax.lax.scan`."""

    base: AbstractFactor
    n_steps: int = eqx.field(static=True)
    feedback_porting_fn: (
        Callable[[PyTree[Array]], Mapping[str, PyTree[Array]]] | None
    ) = eqx.field(static=True)
    feedback_ports: tuple[str, ...] = eqx.field(static=True)
    all_step_input_ports: tuple[str, ...] = eqx.field(static=True)
    weight_tied: bool = eqx.field(static=True)
    slice_info: bool = eqx.field(static=True)
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)
```

---

## `torx/_sampler.py`, `torx/_utils.py`

**How it composes.** An `AbstractSampler` is an injectable distribution provider with
`bernoulli`, `categorical` and `normal`. `JaxPRNGSampler` wraps `jax.random`; `None`
everywhere resolves to it. Where it is consumed: discrete gates draw their output with
`info.categorical`, affine-Gaussian and controlled continuous gates draw noise with
`info.normal`, and `BranchingSimulator.sample` / `expval*` take `sampler=` (Bernoulli
branch draws when every gate has two branches, categorical otherwise). The sampler's
own parameters are never differentiated. `same_pytree_spec` is the shape-and-dtype
comparison behind the `ChainFactor` feedback check.

### Source: `torx/_sampler.py`

```python
"""Injectable distribution providers for stochastic factors and simulators."""

from abc import abstractmethod

import jax
import jax.numpy as jnp
from ihoop.eqx import AbstractStrictModule
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, Key


class AbstractSampler(AbstractStrictModule):

    @abstractmethod
    def bernoulli(
        self,
        key: Key[Array, ""],
        p: Float[Array, "..."],
        shape: tuple[int, ...] | None = None,
    ) -> Int[Array, "..."]:
        raise NotImplementedError

    @abstractmethod
    def categorical(
        self,
        key: Key[Array, ""],
        logits: Float[Array, "... K"],
        axis: int = -1,
        shape: tuple[int, ...] | None = None,
    ) -> Int[Array, "..."]:
        raise NotImplementedError

    @abstractmethod
    def normal(
        self,
        key: Key[Array, ""],
        shape: tuple[int, ...] = (),
        dtype: DTypeLike = float,
    ) -> Float[Array, "..."]:
        raise NotImplementedError


class JaxPRNGSampler(AbstractSampler):
    """Use ``jax.random`` with explicit PRNG keys."""

    def bernoulli(self, key, p, shape=None):
        return jax.random.bernoulli(key, p, shape=shape).astype(jnp.int32)

    def categorical(self, key, logits, axis=-1, shape=None):
        return jax.random.categorical(key, logits, axis=axis, shape=shape)

    def normal(self, key, shape=(), dtype=float):
        return jax.random.normal(key, shape, dtype=dtype)


DEFAULT_SAMPLER: AbstractSampler = JaxPRNGSampler()


def _resolve(sampler: AbstractSampler | None) -> AbstractSampler:
    return DEFAULT_SAMPLER if sampler is None else sampler
```

### Source: `torx/_utils.py`

```python
import jax


def same_pytree_spec(a, b, comparison_fn=None):
    """Return whether two pytrees have matching structure and leaf specs.

    By default, leaves match when both their `shape` and `dtype` match. This
    works for JAX arrays and `jax.ShapeDtypeStruct` specs.
    """
    a_leaves, a_def = jax.tree.flatten(a)
    b_leaves, b_def = jax.tree.flatten(b)

    if comparison_fn is None:

        def _default_cmp(x, y):
            return (x.shape == y.shape) and (x.dtype == y.dtype)

        comparison_fn = _default_cmp

    return a_def == b_def and all(
        comparison_fn(x, y) for x, y in zip(a_leaves, b_leaves)
    )
```

---

## `torx/psc/gates/_base.py`

**How it composes.** Every gate is a factor whose params are its *theta*. Gates store
`sites` and `dims` only.

| Class | Key API | Notes |
|-------|---------|-------|
| `AbstractPGate` | `init_params(key) -> theta`; class vars `_control_indices` (`()`, site positions, or `"all_but_last"`) and `_draw_label` for `draw_circuit` | base of all gates |
| `AbstractDiscreteGate` | `get_matrix(theta)`: **column-stochastic `P[out, in]`**; `get_log_probability_matrix = log(get_matrix.T)` (row = input); input port `"in"`, output an int vector over the gate's sites | a matrix factor; `sample` is a categorical draw on the input's row (final) |
| `AbstractGeneratorGate` | `get_generator(theta)` (rate matrix: off-diagonals >= 0, columns sum to 0), `dt(theta)`; `get_matrix = expm(Q * dt)` | `StateVectorSimulator` only |
| `AbstractKBranchGate` | `num_branches`, `branches` `(K, basis_size, num_sites)` lookup tables, `probs(theta) = softmax([0, theta])` (sigmoid for K = 2), `prob(theta)` (K = 2 only) | theta shape `(K - 1,)`, init zeros; the only family `BranchingSimulator` accepts |
| `AbstractSingleBinaryPGate` / `AbstractMultiBinaryPGate` | `dims` fixed to 2 per site | concrete subclasses declare `sites: int` / `sites: list[int]` |
| `AbstractSinglePditGate` / `AbstractMultiPditGate` | arbitrary `dims` | concrete subclasses declare `sites` and `dims` |
| `AbstractHybridGate` | `sites` is a `HybridSites` dict (`discrete`, `continuous`); input ports `{"discrete": int32 (n_discrete,), "continuous": float32 (sum(dims),)}`; output float32 `(sum(dims),)` | `discrete_dims` defaults to 2 per control site |
| `AbstractContinuousGate` | no discrete reads | **declares no fields and no `__init__`**: a custom subclass must declare `sites` and `dims` itself |
| `AbstractAffineGaussianGate` | implement `affine_parameters(theta) -> (A, b, log_var)`; `sample` draws `A x + b + N(0, diag(exp(log_var)))` (final) | enables `AffineGaussianSimulator` |
| `AbstractControlledContinuousGate` | `_get_discrete_control(inputs)` checks exactly one control site | single control only |

Basis order is row-major over `dims` (`itertools.product`), so site 0 is the most
significant digit: for two pbits the flat index is `2 * s0 + s1`.

### Source: `torx/psc/gates/_base.py`

```python
"""Abstract base classes for probabilistic gates."""

import itertools
from abc import abstractmethod
from typing import Any, ClassVar, Generic, Mapping, TypeAlias, TypedDict, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jaxtyping import Array, Float, Int, Key, PyTree

from ..._sampler import _resolve
from ...factor import (
    _SampleOutput,
    AbstractReferenceFactor,
    InfoTree,
    ParamsTree,
    PortSpec,
)
from ...tractable_prob_factors import AbstractMatrixFactor


class HybridSites(TypedDict):
    discrete: list[int]
    continuous: list[int]


DiscreteSites: TypeAlias = int | list[int]
_SiteType = TypeVar("_SiteType", bound=HybridSites | DiscreteSites)
_DiscreteSiteType = TypeVar("_DiscreteSiteType", bound=DiscreteSites)
_ThetaType = TypeVar("_ThetaType", bound=PyTree[Array])
_DimsType = TypeVar("_DimsType", bound=tuple[int, ...])


class AbstractPGate(AbstractReferenceFactor, Generic[_SiteType, _ThetaType, _DimsType]):
    """Base class for all probabilistic gates.

    For visualization, there are additional properties:

    - `_control_indices`: Indices of control sites for drawing.
        - `()` empty tuple: no controls, draw as spanning box (default)
        - `(0,)` tuple of ints: specific site indices are controls
        - `"all_but_last"`: all sites except the last are controls
    - `_draw_label`: Optional short label for visualization (defaults to class name).
    """

    _control_indices: ClassVar[tuple[int, ...] | str] = ()
    _draw_label: ClassVar[str | None] = None

    sites: eqx.AbstractVar[_SiteType]
    dims: eqx.AbstractVar[_DimsType]

    @abstractmethod
    def init_params(self, key: Key[Array, ""]) -> _ThetaType:
        """Return a freshly-initialised parameter pytree (``theta``).

        **Arguments:**

        - `key`: PRNG key.

        **Returns:**

        An initial ``theta`` pytree for this gate.
        """
        raise NotImplementedError


class AbstractDiscreteGate(
    AbstractMatrixFactor,
    AbstractPGate[_DiscreteSiteType, _ThetaType, _DimsType],
):
    """A discrete probabilistic gate, a finite-state matrix factor."""

    @abstractmethod
    def get_matrix(self, theta: _ThetaType) -> Float[Array, "dim1 dim2"]:
        """
        Get the matrix representation of the gate.

        **Arguments:**

        - `theta`: the gate's parameters.

        **Returns:**

        The column-stochastic transition matrix ``P[out, in]`` of the gate.
        """
        raise NotImplementedError

    @property
    def _basis(self) -> Int[Array, "n_states n_sites"]:
        """The ``prod(dims)`` basis configurations, row-major over `dims`."""
        return jnp.array(list(itertools.product(*(range(d) for d in self.dims))))

    @property
    def input_states(self) -> dict[str, PyTree[Array]]:  # type: ignore[override]
        return {"in": self._basis}

    @property
    def output_states(self) -> PyTree[Array]:
        return self._basis

    @property
    def input_ports(self) -> Mapping[str, PortSpec]:  # type: ignore[override]
        return super().input_ports

    def get_log_probability_matrix(
        self,
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
    ) -> Float[Array, "n_input_states n_output_states"]:
        """Row-stochastic ``log P[in, out] = log(get_matrix(params).T)``."""
        return jnp.log(self.get_matrix(params).T)

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Draw the gate's output configuration from its transition matrix."""
        i = self.input_state_to_index(inputs).astype(jnp.int32)
        row = self.get_log_probability_matrix(params)[i]
        output = self.get_nth_output_state(_resolve(info).categorical(key, row))
        if return_aux:
            return output, None
        return output


class AbstractGeneratorGate(
    AbstractDiscreteGate[_DiscreteSiteType, _ThetaType, _DimsType]
):
    r"""
    Discrete gate defined by a continuous-time Markov generator $Q$.

    Subclasses provide a rate matrix $Q$ via `get_generator` and a time
    step via `dt`. The transition matrix is then

    $$P = \exp(Q \cdot \Delta t).$$
    """

    @abstractmethod
    def dt(self, theta: _ThetaType) -> Float[Array, ""]:
        r"""The continuous-time step $\Delta t$, extracted from ``theta``."""
        raise NotImplementedError

    @abstractmethod
    def get_generator(self, theta: _ThetaType) -> Float[Array, "dim1 dim2"]:
        r"""
        Return the continuous-time Markov generator $Q$.

        $Q$ must be a rate matrix, which means off-diagonal entries are non-negative
        and column sums equal to zero.

        **Arguments:**

        - `theta`: the gate's parameters.

        **Returns:**

        The Markov generator $Q$.
        """
        raise NotImplementedError

    def get_matrix(self, theta: _ThetaType) -> Float[Array, "dim1 dim2"]:
        r"""Column-stochastic transition matrix $P = \exp(Q \cdot \Delta t)$.

        See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation.
        """
        return expm(self.get_generator(theta) * self.dt(theta))


class AbstractHybridGate(AbstractPGate[HybridSites, _ThetaType, _DimsType]):
    """Base class for hybrid gates.

    Gates specify which discrete and continuous sites they act on via the `sites`
    dict. As a `Factor`, the gate's `sample` takes the relevant substate as its
    `inputs` (a dict with ``"discrete"`` / ``"continuous"`` values at the gate's
    sites) and returns the new continuous substate.

    !!! note

        `torx` currently only supports discrete controlled continuous gates.
    """

    @property
    def discrete_dims(self) -> tuple[int, ...]:
        """Dimensions of discrete control sites, aligned with ``sites["discrete"]``."""
        return (2,) * len(self.sites.get("discrete", []))

    @property
    def input_ports(self) -> Mapping[str, PortSpec]:  # type: ignore[override]
        n_discrete = len(self.sites.get("discrete", []))
        cont_dim = sum(self.dims)
        return {
            "discrete": jax.ShapeDtypeStruct((n_discrete,), jnp.int32),
            "continuous": jax.ShapeDtypeStruct((cont_dim,), jnp.float32),
        }

    @property
    def output_spec(self) -> PortSpec:
        return jax.ShapeDtypeStruct((sum(self.dims),), jnp.float32)

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Sample the new continuous substate.

        **Arguments:**

        - `key`: JAX random key.
        - `inputs`: Substate dict with ``"discrete"`` / ``"continuous"`` keys
          holding the values at the gate's sites.
        - `params`: the gate's ``theta``.
        - `info`: optional `AbstractSampler` distribution provider.
        - `site_info`: unused.
        - `return_aux`: if ``True``, return ``(output, None)``.

        **Returns:**

        New values for the continuous sites (same shape as
        ``inputs["continuous"]``), or ``(output, None)`` when ``return_aux``.
        """
        raise NotImplementedError


class AbstractContinuousGate(AbstractHybridGate[_ThetaType, _DimsType]):
    """Gate that only touches continuous sites.

    Subclasses set sites to specify which continuous sites they act on.
    """

    pass


class AbstractAffineGaussianGate(AbstractContinuousGate[_ThetaType, _DimsType]):
    r"""Continuous gate with an exact affine-Gaussian channel.

    Subclasses expose their local continuous-state transition as

    $$
    x' = A x + b + \epsilon,\qquad
    \epsilon \sim \mathcal{N}(0, \operatorname{diag}(\exp(\text{log\_var}))),
    $$

    by implementing `affine_parameters`. The default `sample` draws from this
    channel, but exposing $(A, b, \log\text{var})$ also lets simulators reason
    about the exact Gaussian transition rather than only drawing samples.
    """

    @abstractmethod
    def affine_parameters(
        self,
        theta: _ThetaType,
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return ``(A, b, log_var)`` for the local affine-Gaussian channel.

        **Arguments:**

        - `theta`: the gate's parameters.

        **Returns:**

        A tuple ``(A, b, log_var)`` describing ``x' = A x + b + noise`` with
        ``noise ~ Normal(0, diag(exp(log_var)))``.
        """
        raise NotImplementedError

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: ParamsTree,
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        """Sample the affine-Gaussian channel exposed by `affine_parameters`."""
        x = inputs["continuous"]
        A, b, log_var = self.affine_parameters(params)
        noise = _resolve(info).normal(key, x.shape, dtype=log_var.dtype)
        mean = jnp.einsum("ij,...j->...i", A, x) + b
        output = mean + noise * jnp.sqrt(jnp.exp(log_var))
        return (output, None) if return_aux else output


class AbstractControlledContinuousGate(AbstractHybridGate[_ThetaType, _DimsType]):
    """Gate where discrete sites control the continuous transformation.

    !!! note

        Currently we only support control by a single discrete site (pbit/pdit),
        so dims only specifies continuous dimensions. Multi-site discrete control
        would require tracking discrete dims as well.
    """

    def _get_discrete_control(self, inputs: Mapping[str, PyTree[Array]]) -> Array:
        """Extract single discrete control value with validation."""
        disc = inputs["discrete"]
        if disc.shape != (1,):
            raise ValueError(
                f"{self.__class__.__name__} expects exactly 1 discrete control "
                f"site, got shape {disc.shape}"
            )
        return disc[0]


class AbstractKBranchGate(
    AbstractDiscreteGate[
        _DiscreteSiteType, Float[Array, " num_branch_params"], _DimsType
    ]
):
    """
    Gates with K branches selected via softmax over K-1 parameters.

    The probability of branch k is softmax([0, theta])[k], where theta has K-1
    learnable parameters for K branches. For K = 2, this reduces to the sigmoid
    parameterization: softmax([0, theta]) = [1 - sigmoid(theta), sigmoid(theta)].

    Subclasses must implement:

    - num_branches: The number of branches K (>= 2)

    - branches: A list of K lookup tables, one per branch
    """

    @property
    @abstractmethod
    def num_branches(self) -> int:
        """Return the number of branches K (must be >= 2)."""
        raise NotImplementedError

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " num_branch_params"]:
        """Initial ``theta`` of shape ``(num_branches - 1,)`` (zeros).

        See [`torx.psc.AbstractPGate.init_params`][].
        """
        return jnp.zeros((self.num_branches - 1,))

    def probs(self, theta: Float[Array, " num_branch_params"]) -> Float[Array, " K"]:
        """Branch probabilities for parameters ``theta``."""
        theta = jnp.atleast_1d(theta)

        if self.num_branches == 2:
            sig = jax.nn.sigmoid(theta[0])
            return jnp.stack([1 - sig, sig])

        padded = jnp.concatenate([jnp.zeros(1), theta])
        return jax.nn.softmax(padded)

    @property
    @abstractmethod
    def branches(self) -> Int[Array, "K basis_size num_sites"]:
        """
        Get the K lookup tables for all branches as a stacked array.

        Each lookup table (along axis 0) maps computational basis state index to
        output state values for each site. For example, for a PCNOT gate the
        operation branch is a CNOT gate. A CNOT gate's action on the computational
        basis states is 00 -> 00, 01 -> 01, 10 -> 11, and 11 -> 10, so the lookup
        table is [[0, 0], [0, 1], [1, 1], [1, 0]].

        The identity branch (often branches[0]) maps each state to itself.
        For the PCNOT example, the identity maps 00 -> 00, 01 -> 01, 10 -> 10,
        11 -> 11, so the lookup table is [[0, 0], [0, 1], [1, 0], [1, 1]].

        For binary gates, basis_size = 2**num_sites.
        For pdit gates, basis_size = prod(dims).

        **Returns:**

        A stacked array of shape (K, basis_size, num_sites) containing all branch
        lookup tables.
        """
        raise NotImplementedError

    # todo: do we want to remove this?
    def prob(self, theta: Float[Array, " num_branch_params"]) -> Float[Array, ""]:
        """Probability of applying the operation (branches[1]); K=2 only."""
        probs = self.probs(theta)
        if len(probs) > 2:
            raise ValueError("prob is undefined for K>2 branches.")
        return probs[1]


class AbstractSingleBinaryPGate(AbstractKBranchGate[int, tuple[int]]):
    """Abstract base class for single-site binary gates (dimension 2)."""

    sites: eqx.AbstractVar[int]

    @property
    def dims(self) -> tuple[int]:  # type: ignore[override]
        return (2,)


class AbstractMultiBinaryPGate(AbstractKBranchGate[list[int], tuple[int, ...]]):
    """Abstract base class for multi-site binary gates (dims = 2)."""

    sites: eqx.AbstractVar[list[int]]

    @property
    def dims(self) -> tuple[int, ...]:  # type: ignore[override]
        return (2,) * len(self.sites)


class AbstractSinglePditGate(AbstractKBranchGate[int, tuple[int, ...]]):
    """Abstract base class for single-site pdit gates (arbitrary dimension k)."""

    pass


class AbstractMultiPditGate(AbstractKBranchGate[list[int], tuple[int, ...]]):
    """Abstract base class for multi-site pdit gates (same dimension k per site)."""

    pass


AbstractSingleBinaryPGate.__init__.__doc__ = """**Arguments:**

- `sites`: The index of the site that this gate acts on.
"""

AbstractMultiBinaryPGate.__init__.__doc__ = """**Arguments:**

- `sites`: A list of site indices that this gate acts on.
"""

AbstractContinuousGate.__init__.__doc__ = """**Arguments:**

- `sites`: List of continuous site indices.
- `dims`: Tuple of continuous dimensions for each site.
"""

AbstractAffineGaussianGate.__init__.__doc__ = """**Arguments:**

- `sites`: List of continuous site indices.
- `dims`: Tuple of continuous dimensions for each site.
"""

AbstractControlledContinuousGate.__init__.__doc__ = """**Arguments:**

- `sites`: Tuple of (discrete_site, continuous_sites).
- `dims`: Tuple of continuous dimensions.
"""

AbstractSinglePditGate.__init__.__doc__ = """**Arguments:**

- `sites`: The index of the site that this gate acts on.
- `dims`: Tuple containing the dimension of the pdit.
"""

AbstractMultiPditGate.__init__.__doc__ = """**Arguments:**

- `sites`: A list of site indices that this gate acts on.
- `dims`: Tuple containing the dimensions of each pdit.
"""
```

---

## `torx/psc/gates/_binary.py`

**How it composes.** Concrete pbit gates. Theta is a `(1,)` logit and branch 1 fires
with probability `sigmoid(theta)`; pass `jnp.array([logit(p)])`.

| Gate | Sites | Branch 0 | Branch 1 |
|------|-------|----------|----------|
| `PNOT(i)` | 1 | identity | flip |
| `PReset(i)` | 1 | identity | reset to 0 (finite logit leaks: `P(1)` stays `1 - sigmoid(theta)`) |
| `PCNOT([c, t])` | 2 | identity | `10 -> 11, 11 -> 10` |
| `PSWAP([a, b])` | 2 | identity | `01 <-> 10` |
| `PJUMP([a, b])` | 2 | identity | `10 -> 01` only: moves a token to an empty target |
| `PCopy([src, dst])` | 2 | identity | `01 -> 00, 10 -> 11` |
| `POR([a, b])` | 2 | identity | first = `a OR b`, second = 0 |
| `PDEMUX([in, work])` | 2 | **reset work** (`01 -> 00, 11 -> 10`) | copy in to work and reset in (`10 -> 01, 11 -> 01`, `01 -> 00`) |
| `PMultiCNOT([c1, ..., t])` | k + 1 | identity | flip `t` iff all controls are 1 |
| `PCSWAP([c, a, b])` | 3 | identity | swap `a, b` iff `c = 1` |

### Source: `torx/psc/gates/_binary.py`

```python
"""Binary probabilistic gates."""

import itertools

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._base import AbstractMultiBinaryPGate, AbstractSingleBinaryPGate


class PNOT(AbstractSingleBinaryPGate):
    r"""
    A probabilistic NOT gate.

    This gate flips the value of a given pbit with probability $p = \sigma(\theta)$
    where $\theta$ is the gate parameter, and otherwise does nothing.

    The transition matrix is:

    $$\begin{pmatrix} 1-p & p \\ p & 1-p \end{pmatrix}$$
    """

    sites: int

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 2 1"]:
        """
        branches[0] is identity: 0 -> 0 and 1 -> 1.
        branches[1] is bit flip: 0 -> 1 and 1 -> 0.
        """
        return jnp.array(
            [
                [[0], [1]],  # identity
                [[1], [0]],  # flip
            ]
        )

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "2 2"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        return jnp.array(
            [
                [1 - self.prob(theta), self.prob(theta)],
                [self.prob(theta), 1 - self.prob(theta)],
            ]
        )


class PCNOT(AbstractMultiBinaryPGate):
    r"""
    A probabilistic CNOT gate.

    This gate performs a controlled-NOT operation on two given pbits with
    probability $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and
    otherwise does nothing. A controlled-NOT operation flips the value of
    the second pbit if the first pbit has the value `1`, and does nothing
    if the first pbit has the value `0`.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1-p & p \\ 0 & 0 & p & 1-p
    \end{pmatrix}$$
    """

    _control_indices = (0,)

    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00 -> 00, 01 -> 01, 10 -> 10, 11 -> 11.
        branches[1] is CNOT: 00->00, 01->01, 10->11, 11->10.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [0, 1], [1, 1], [1, 0]],  # CNOT
            ]
        )

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "4 4"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 - self.prob(theta), self.prob(theta)],
                [0, 0, self.prob(theta), 1 - self.prob(theta)],
            ]
        )


class PSWAP(AbstractMultiBinaryPGate):
    r"""
    A probabilistic SWAP gate.

    This gate swaps the values of two given pbits with probability
    $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and otherwise
    does nothing.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 \\ 0 & 1-p & p & 0 \\ 0 & p & 1-p & 0 \\ 0 & 0 & 0 & 1
    \end{pmatrix}$$
    """

    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00 -> 00, 01 -> 01, 10 -> 10, 11 -> 11.
        branches[1] is swap: 00->00, 01->10, 10->01, 11->11.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [1, 0], [0, 1], [1, 1]],  # swap
            ]
        )

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "4 4"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1 - self.prob(theta), self.prob(theta), 0],
                [0, self.prob(theta), 1 - self.prob(theta), 0],
                [0, 0, 0, 1],
            ]
        )


class PJUMP(AbstractMultiBinaryPGate):
    r"""
    A probabilistic JUMP gate.

    This gate performs a "jump" operation on two given pbits with probability
    $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and otherwise does
    nothing. A jump operation moves probability from $|10)$ to $|01)$
    if the first pbit has the value `1`.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 \\ 0 & 1 & p & 0 \\ 0 & 0 & 1-p & 0 \\ 0 & 0 & 0 & 1
    \end{pmatrix}$$
    """

    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00 -> 00, 01 -> 01, 10 -> 10, 11 -> 11.
        branches[1] is JUMP: 00->00, 01->01, 10->01, 11->11.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [0, 1], [0, 1], [1, 1]],  # JUMP
            ]
        )

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "4 4"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1, self.prob(theta), 0],
                [0, 0, 1 - self.prob(theta), 0],
                [0, 0, 0, 1],
            ]
        )


class PMultiCNOT(AbstractMultiBinaryPGate):
    r"""
    A probabilistic multi-controlled-NOT gate.

    This gate performs a multi-controlled-NOT operation on any number of given pbits
    with probability $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and
    otherwise does nothing. A multi-controlled-NOT operation flips the value
    of the last pbit if all the other pbits have the value `1`, and does
    nothing otherwise.

    The transition matrix is identity except for the last two rows/columns which
    swap with probability $p$.
    """

    _control_indices = "all_but_last"
    _draw_label = "PMCNOT"

    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 2**d d"]:
        """
        branches[0] is identity.
        branches[1] is multi-controlled-NOT: flips last bit when all others are 1.
        """
        identity = jnp.array(list(itertools.product([0, 1], repeat=len(self.sites))))
        op = identity.at[-2, -1].set(1).at[-1, -1].set(0)
        return jnp.stack([identity, op])

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "d d"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        return (
            jnp.eye(2 ** len(self.sites))
            .at[-2, -2]
            .set(1 - self.prob(theta))
            .at[-2, -1]
            .set(self.prob(theta))
            .at[-1, -2]
            .set(self.prob(theta))
            .at[-1, -1]
            .set(1 - self.prob(theta))
        )


class PDEMUX(AbstractMultiBinaryPGate):
    r"""
    A probabilistic demultiplexor.

    This gate has one input pbit and one "work" pbit. It either (i) copies the
    input to the work pbit and sets the input to `0`, or (ii) does nothing to
    the input and sets the work pbit to `0`. The probability of the former
    occurring is $p = \sigma(\theta)$ where $\theta$ is the gate parameter.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 1 & 0 & 0 \\ 0 & 0 & p & p \\ 0 & 0 & 1-p & 1-p \\ 0 & 0 & 0 & 0
    \end{pmatrix}$$
    """

    _control_indices = (0,)

    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] resets second pbit: 00 -> 00, 01 -> 00, 10 -> 10, 11 -> 10.
        branches[1] copies and resets: 00->00, 01->00, 10->01, 11->01.
        """
        return jnp.array(
            [
                [[0, 0], [0, 0], [1, 0], [1, 0]],  # resets second pbit
                [[0, 0], [0, 0], [0, 1], [0, 1]],  # copies and resets
            ]
        )

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "4 4"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        return jnp.array(
            [
                [1, 1, 0, 0],
                [0, 0, self.prob(theta), self.prob(theta)],
                [0, 0, 1 - self.prob(theta), 1 - self.prob(theta)],
                [0, 0, 0, 0],
            ]
        )


class POR(AbstractMultiBinaryPGate):
    r"""
    A probabilistic OR gate.

    This gate performs a modified OR operation on two given pbits with probability
    $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and otherwise does
    nothing. A modified OR operation sets the first pbit to the OR of the two input
    pbits, and sets the second pbit to `0`.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 \\ 0 & 1-p & 0 & 0 \\ 0 & p & 1 & p \\ 0 & 0 & 0 & 1-p
    \end{pmatrix}$$
    """

    _control_indices = "all_but_last"

    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00 -> 00, 01 -> 01, 10 -> 10, 11 -> 11.
        branches[1] is OR: 00->00, 01->10, 10->10, 11->10.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [1, 0], [1, 0], [1, 0]],  # OR
            ]
        )

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "4 4"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        # 00 -> 00, 01 -> 10, 10 -> 10, 11 -> 10

        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1 - self.prob(theta), 0, 0],
                [0, self.prob(theta), 1, self.prob(theta)],
                [0, 0, 0, 1 - self.prob(theta)],
            ]
        )


class PReset(AbstractSingleBinaryPGate):
    r"""
    A probabilistic reset gate.

    This gate sets the given pbit to `0` with probability $p = \sigma(\theta)$ where
    $\theta$ is the gate parameter, and otherwise does nothing.

    The transition matrix is:

    $$\begin{pmatrix} 1 & p \\ 0 & 1-p \end{pmatrix}$$
    """

    sites: int

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 2 1"]:
        """
        branches[0] is identity: 0 -> 0 and 1 -> 1.
        branches[1] resets to 0: 0 -> 0 and 1 -> 0.
        """
        return jnp.array(
            [
                [[0], [1]],  # identity
                [[0], [0]],  # reset
            ]
        )

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "2 2"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        return jnp.array([[1, self.prob(theta)], [0, 1 - self.prob(theta)]])


class PCopy(AbstractMultiBinaryPGate):
    r"""
    A probabilistic copy gate.

    This gate copies the first pbit to the second pbit with probability
    $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and otherwise
    does nothing.

    Transition matrix:

    $$\begin{pmatrix}
    1 & p & 0 & 0 \\ 0 & 1-p & 0 & 0 \\ 0 & 0 & 1-p & 0 \\ 0 & 0 & p & 1
    \end{pmatrix}$$
    """

    _control_indices = (0,)

    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00->00, 01->01, 10->10, 11->11.
        branches[1] copies first to second: 00->00, 01->00, 10->11, 11->11.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [0, 0], [1, 1], [1, 1]],  # copy
            ]
        )

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "4 4"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        return jnp.array(
            [
                [1, self.prob(theta), 0, 0],
                [0, 1 - self.prob(theta), 0, 0],
                [0, 0, 1 - self.prob(theta), 0],
                [0, 0, self.prob(theta), 1],
            ]
        )


class PCSWAP(AbstractMultiBinaryPGate):
    r"""
    A probabilistic controlled-SWAP gate (Fredkin gate).

    This gate swaps the values of two target pbits only when the control pbit
    is in state $|1)$, with probability $p = \sigma(\theta)$ where $\theta$
    is the gate parameter. If the control is $|0)$ or the gate is not applied
    (with probability $1-p$), nothing happens.

    The first site is the control, the second and third sites are the targets
    to be swapped.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1-p & p & 0 \\
    0 & 0 & 0 & 0 & 0 & p & 1-p & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
    \end{pmatrix}$$
    """

    _control_indices = (0,)

    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 8 3"]:
        """
        branches[0] is identity.
        branches[1] is controlled-SWAP: swap targets if control=1.
        """
        # Input states in order: 000, 001, 010, 011, 100, 101, 110, 111
        identity = jnp.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        # Output when control=1: swap the last two bits
        cswap = jnp.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
            ]
        )
        return jnp.stack([identity, cswap])

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "8 8"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        p = self.prob(theta)
        return jnp.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1 - p, p, 0],
                [0, 0, 0, 0, 0, p, 1 - p, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )


PNOT.__init__.__doc__ = """See [`torx.psc.AbstractSingleBinaryPGate.__init__`][]."""
PReset.__init__.__doc__ = """See [`torx.psc.AbstractSingleBinaryPGate.__init__`][]."""
PCNOT.__init__.__doc__ = """See [`torx.psc.AbstractMultiBinaryPGate.__init__`][]."""
PSWAP.__init__.__doc__ = """See [`torx.psc.AbstractMultiBinaryPGate.__init__`][]."""
PJUMP.__init__.__doc__ = """See [`torx.psc.AbstractMultiBinaryPGate.__init__`][]."""
PDEMUX.__init__.__doc__ = """See [`torx.psc.AbstractMultiBinaryPGate.__init__`][]."""
POR.__init__.__doc__ = """See [`torx.psc.AbstractMultiBinaryPGate.__init__`][]."""
PCopy.__init__.__doc__ = """See [`torx.psc.AbstractMultiBinaryPGate.__init__`][]."""
PMultiCNOT.__init__.__doc__ = (
    """See [`torx.psc.AbstractMultiBinaryPGate.__init__`][]."""
)
PCSWAP.__init__.__doc__ = """See [`torx.psc.AbstractMultiBinaryPGate.__init__`][]."""
```

---

## `torx/psc/gates/_pdit.py`

**How it composes.** `d`-state gates; `dims` accepts an int and is stored as a tuple.
`PditShift(sites=i, dims=d)` and `PditSWAP(sites=[a, b], dims=d)` are two-branch
(`(1,)` logit). `PditCycle(sites=i, dims=d)` has three branches
`[stay, forward, backward] = softmax([0, t0, t1])`, so
`theta = log([p_fwd / p_stay, p_back / p_stay])`.

### Source: `torx/psc/gates/_pdit.py`

```python
"""Pdit probabilistic gates."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._base import AbstractMultiPditGate, AbstractSinglePditGate


class PditShift(AbstractSinglePditGate):
    r"""
    A probabilistic cyclic shift gate for a single k-dimensional pdit.

    This gate shifts the state cyclically: i -> (i+1) mod k with probability
    $p = \sigma(\theta)$, and does nothing with probability $1-p$.

    The transition matrix for dimension k is:
    $(1 - p) I + p C$ where $C$ is the cyclic permutation matrix.
    """

    sites: int
    dims: tuple[int, ...] = eqx.field(
        converter=lambda d: (d,) if isinstance(d, int) else tuple(d)
    )

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 d 1"]:
        """
        branches[0] is identity: i -> i.
        branches[1] is cyclic shift: i -> (i+1) mod k.
        """
        d = self.dims[0]
        identity = jnp.array([[i] for i in range(d)])
        shift = jnp.array([[(i + 1) % d] for i in range(d)])
        return jnp.stack([identity, shift])

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "d d"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        d = self.dims[0]
        identity = jnp.eye(d)
        cyclic = jnp.roll(identity, shift=1, axis=0)
        p = self.prob(theta)
        return (1 - p) * identity + p * cyclic


class PditSWAP(AbstractMultiPditGate):
    r"""
    A probabilistic SWAP gate for two k-dimensional pdits.

    This gate swaps the values of two pdits with probability $p = \sigma(\theta)$,
    and does nothing with probability $1-p$. Both pdits must have the same dimension.

    For dimension k, the state space is $k^2$ and the SWAP permutes basis states
    $|i,j) \to |j,i)$.
    """

    sites: list[int]
    dims: tuple[int, ...] = eqx.field(
        converter=lambda d: (d, d) if isinstance(d, int) else tuple(d)
    )

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 d2 2"]:
        """
        branches[0] is identity: |i,j) -> |i,j).
        branches[1] is SWAP: |i,j) -> |j,i).
        """
        d = self.dims[0]
        identity = jnp.array([[i, j] for i in range(d) for j in range(d)])
        swap = jnp.array([[j, i] for i in range(d) for j in range(d)])
        return jnp.stack([identity, swap])

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "d2 d2"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        d = self.dims[0]
        d2 = d * d

        # Build SWAP matrix: |i,j> -> |j,i>
        swap = jnp.zeros((d2, d2))
        for i in range(d):
            for j in range(d):
                src = i * d + j
                dst = j * d + i
                swap = swap.at[dst, src].set(1.0)

        identity = jnp.eye(d2)
        p = self.prob(theta)
        return (1 - p) * identity + p * swap


class PditCycle(AbstractSinglePditGate):
    r"""
    A probabilistic 3-branch cyclic gate for a single d-dimensional pdit.

    This gate implements a random walk on a ring: with some probability stay
    in place, shift forward, or shift backward. Useful for modeling diffusion
    or random walks on discrete rings.

    This gate has 3 branches:
    - Branch 0: Identity (i -> i)
    - Branch 1: Forward cycle (i -> (i+1) mod d)
    - Branch 2: Backward cycle (i -> (i-1) mod d)

    The probabilities are determined by softmax([0, theta[0], theta[1]]).

    The transition matrix is:
    $p_0 I + p_1 C_{forward} + p_2 C_{backward}$

    where $C_{forward}$ is the forward cyclic permutation and $C_{backward}$
    is the backward cyclic permutation.

    """

    sites: int
    dims: tuple[int, ...] = eqx.field(
        converter=lambda d: (d,) if isinstance(d, int) else tuple(d)
    )

    @property
    def num_branches(self) -> int:
        return 3

    @property
    def branches(self) -> Int[Array, "3 d 1"]:
        """
        branches[0] is identity: i -> i.
        branches[1] is forward cycle: i -> (i+1) mod d.
        branches[2] is backward cycle: i -> (i-1) mod d.
        """
        d = self.dims[0]
        identity = jnp.array([[i] for i in range(d)])
        forward = jnp.array([[(i + 1) % d] for i in range(d)])
        backward = jnp.array([[(i - 1) % d] for i in range(d)])
        return jnp.stack([identity, forward, backward])

    def get_matrix(self, theta: Float[Array, "2"]) -> Float[Array, "d d"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        d = self.dims[0]
        p = self.probs(theta)  # [p_identity, p_forward, p_backward]
        identity = jnp.eye(d)
        forward = jnp.roll(identity, shift=1, axis=0)
        backward = jnp.roll(identity, shift=-1, axis=0)
        return p[0] * identity + p[1] * forward + p[2] * backward


PditShift.__init__.__doc__ = """See [`torx.psc.AbstractSinglePditGate.__init__`][]."""
PditSWAP.__init__.__doc__ = """See [`torx.psc.AbstractMultiPditGate.__init__`][]."""
PditCycle.__init__.__doc__ = """See [`torx.psc.AbstractSinglePditGate.__init__`][]."""
```

---

## `torx/psc/gates/_generator.py`

**How it composes.** `PISING([i, j])` with theta `[J, h1, h2, beta, dt]` (init zeros, the
identity). Spins `s = 2b - 1`, energy `E = -J s1 s2 - h1 s1 - h2 s2`, Glauber
single-flip rates `sigmoid(-beta * dE)`, transition `expm(Q * dt)`. Use it with
`StateVectorSimulator`; `BranchingSimulator` rejects it at `build_circuit`. A bond
kernel drives its *pair* toward the pair's Boltzmann law and ignores neighbors outside
the pair, so a sweep of per-edge `PISING` gates is not a Gibbs sampler of the whole
graph.

### Source: `torx/psc/gates/_generator.py`

```python
"""Generator-defined gates for torx circuits."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ._base import AbstractGeneratorGate


class PISING(AbstractGeneratorGate[list[int], Float[Array, "5"], tuple[int, int]]):
    r"""
    A probabilistic Ising interaction gate.

    Unlike the Boolean gates (PNOT, PCNOT, PSWAP, ...) which interpolate between
    identity and a logic operation, PISING is an **energy-based** gate: it drives
    two pbits toward the thermal equilibrium of a pairwise Ising interaction.
    Applying it repeatedly thermalizes the joint distribution toward the
    Boltzmann distribution $\pi_i \propto e^{-\beta E_i}$.

    !!! note "Simulator support"

        `PISING` is a generator-defined dense-matrix gate. Use it with
        [`StateVectorSimulator`][torx.psc.StateVectorSimulator]; it is not
        compatible with [`BranchingSimulator`][torx.psc.BranchingSimulator], which
        currently requires branch lookup-table gates.

    The Ising energy for two pbits $\sigma_1, \sigma_2 \in \{0, 1\}$ is:

    $$E(\sigma_1, \sigma_2) = -J \, s_1 s_2 - h_1 s_1 - h_2 s_2$$

    where $s_i = 2\sigma_i - 1 \in \{-1, +1\}$.

    The transition matrix $P = \exp(Q \cdot \Delta t)$ is the matrix exponential
    of a Markov generator $Q$ whose off-diagonal entries are Glauber
    single-spin-flip rates:

    $$Q_{ji} = \frac{1}{1 + e^{\beta \, \Delta E_{ij}}}$$

    Only single-spin-flip transitions are nonzero.

    **Parameters**:

    - `theta[0]`, $J$: coupling strength
      ($J > 0$ ferromagnetic, $J < 0$ antiferromagnetic)
    - `theta[1]`, $h_1$: external field on first pbit
    - `theta[2]`, $h_2$: external field on second pbit
    - `theta[3]`, $\beta$: inverse temperature
      ($\beta \to 0$: uniform, $\beta \to \infty$: ground state)
    - `theta[4]`, $\Delta t$: continuous-time step
      ($0$: identity, larger: closer to equilibrium)
    """

    sites: list[int]

    @property
    def dims(self) -> tuple[int, int]:  # type: ignore[override]
        return (2, 2)

    def init_params(self, key: Key[Array, ""]) -> Float[Array, "5"]:
        """Initial ``theta = [J, h1, h2, beta, dt]`` (zeros).

        See [`torx.psc.AbstractPGate.init_params`][].
        """
        return jnp.zeros(5)

    def J(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[0]

    def h1(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[1]

    def h2(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[2]

    def beta(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[3]

    def dt(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[4]

    @classmethod
    def _glauber_rate(cls, dE: Float[Array, ""], beta: Float[Array, ""]):
        """Glauber single-spin flip rate for energy change dE."""
        return jax.nn.sigmoid(-beta * dE)

    def _energies(self, theta: Float[Array, "5"]) -> Float[Array, "4"]:
        """Ising energies for each basis configuration.

        Maps {0,1} -> {-1,+1} via s = 2*sigma - 1.

        Returns energies for |00), |01), |10), |11).
        """
        J, h1, h2 = self.J(theta), self.h1(theta), self.h2(theta)
        return jnp.array(
            [
                -J * (-1) * (-1) - h1 * (-1) - h2 * (-1),  # |00): s=(-1,-1)
                -J * (-1) * (+1) - h1 * (-1) - h2 * (+1),  # |01): s=(-1,+1)
                -J * (+1) * (-1) - h1 * (+1) - h2 * (-1),  # |10): s=(+1,-1)
                -J * (+1) * (+1) - h1 * (+1) - h2 * (+1),  # |11): s=(+1,+1)
            ]
        )

    def get_generator(self, theta: Float[Array, "5"]) -> Float[Array, "4 4"]:
        r"""Build the 4x4 Glauber-dynamics Markov generator $Q$.

        $Q$ is a rate matrix with zero column sums and non-negative off-diagonal
        entries. Only single-spin-flip transitions are nonzero:
        $|00) \leftrightarrow |01)$,
        $|00) \leftrightarrow |10)$,
        $|01) \leftrightarrow |11)$,
        $|10) \leftrightarrow |11)$.

        The generator is independent of $\Delta t$.
        """
        E = self._energies(theta)
        beta = self.beta(theta)

        transitions = jnp.array([[0, 1], [0, 2], [1, 3], [2, 3]])

        Q = jnp.zeros((4, 4))
        for idx in range(4):
            i, j = transitions[idx, 0], transitions[idx, 1]
            dE_ij = E[j] - E[i]
            dE_ji = E[i] - E[j]
            Q = Q.at[j, i].add(self._glauber_rate(dE_ij, beta))
            Q = Q.at[i, j].add(self._glauber_rate(dE_ji, beta))

        Q = Q.at[jnp.diag_indices(4)].set(0.0)
        Q = Q - jnp.diag(Q.sum(axis=0))

        return Q


PISING.__init__.__doc__ = """See [`torx.psc.AbstractDiscreteGate.__init__`][]."""
```

---

## `torx/psc/gates/_continuous.py`

**How it composes.** pmode gates. `sites` accepts an int, a list, or a `HybridSites`
dict and is converted to the dict; hybrid gates accept `(discrete, continuous)`.

| Gate | Theta | Channel |
|------|-------|---------|
| `AffineGaussianGate(sites, dims)` | `{"A": (d, d), "b": (d,), "log_var": (d,)}`, init `A = I, b = 0, log_var = 0` | `A x + b + N(0, diag(exp(log_var)))` |
| `GaussianNoiseGate(sites, dims)` / `Diffuse(sites, dims)` | `(d,)` log-variance | `x + N(0, exp(theta))`; Brownian `D` over `t` is `log(2 D t)` |
| `Displace(sites, dims)` | `(d,)` | `x + theta` (noise-free) |
| `Scale(sites, dims)` | `(d,)` log-scale | `exp(theta) * x` (noise-free) |
| `Mix(sites=[a, b], dims=(1, 1))` | scalar angle | rotation of two scalar sites |
| `MixtureGaussianGate(sites=(disc, cont), dims, num_components=K)` | `{"means": (K, d), "log_vars": (K, d)}` | `x + means[k] + N(0, diag(exp(log_vars[k])))`, `k` the control value; `discrete_dims = (K,)` |
| `JumpDiffusionGate(sites=(disc, cont), dims)` | `{"diff_log_var", "jump_mean", "jump_log_var"}`, each `(d,)` | diffusion always, plus a Gaussian jump when the control is 1 |

`d = sum(dims)`. The noise-free gates use `log_var = -inf` internally. Mixture weights
are not in the theta: they come from the control site's distribution.

### Source: `torx/psc/gates/_continuous.py`

```python
"""Continuous and hybrid probabilistic gates."""

from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Key, PyTree

from ..._sampler import _resolve, AbstractSampler
from ...factor import _SampleOutput, InfoTree
from ._base import (
    AbstractAffineGaussianGate,
    AbstractControlledContinuousGate,
    HybridSites,
)


def _draw_normal(
    sampler: AbstractSampler | None,
    key: Key[Array, ""],
    shape: tuple[int, ...],
    dtype: DTypeLike = float,
) -> Float[Array, "..."]:
    return _resolve(sampler).normal(key, shape, dtype=dtype)


def _continuous_sites_converter(x: HybridSites | list[int]) -> HybridSites:
    if isinstance(x, dict):
        return x
    return {"discrete": [], "continuous": [x] if not isinstance(x, list) else x}


def _hybrid_sites_converter(
    x: HybridSites | tuple[int | list[int], int | list[int]],
) -> HybridSites:
    if isinstance(x, dict):
        return x
    discrete, continuous = x
    return {
        "discrete": [discrete] if isinstance(discrete, int) else list(discrete),
        "continuous": [continuous] if isinstance(continuous, int) else list(continuous),
    }


class GaussianNoiseGate(
    AbstractAffineGaussianGate[Float[Array, " d"], tuple[int, ...]]
):
    r"""
    Additive Gaussian noise gate.

    This gate adds independent Gaussian noise to the continuous state:
    $x' = x + \mathcal{N}(0, \exp(\theta))$

    The variance is parameterized as $\exp(\theta)$ to ensure positivity.
    """

    _draw_label = "Gauss"

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        """Initial ``theta`` (log-variance), zeros over the continuous dims."""
        return jnp.zeros(sum(self.dims))

    def affine_parameters(
        self,
        theta: Float[Array, " d"],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the additive Gaussian channel as ``(A, b, log_var)``."""
        dim = sum(self.dims)
        log_var = jnp.broadcast_to(theta, (dim,))
        return (
            jnp.eye(dim, dtype=log_var.dtype),
            jnp.zeros(dim, dtype=log_var.dtype),
            log_var,
        )


class AffineGaussianGate(AbstractAffineGaussianGate[dict[str, Array], tuple[int, ...]]):
    r"""
    Affine transformation with Gaussian noise.

    This gate applies an affine transformation followed by Gaussian noise:
    $x' = Ax + b + \mathcal{N}(0, \sigma^2)$ where $\sigma^2 = \exp$(`log_var`)

    The theta dict contains:

    - `A`: Transformation matrix of shape (d, d)
    - `b`: Bias vector of shape (d,)
    - `log_var`: Log-variance of shape (d,)
    """

    _draw_label = "AffineGauss"

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> dict[str, Array]:
        """Initial identity affine: ``A = I``, ``b = 0``, ``log_var = 0``."""
        d = sum(self.dims)
        return {"A": jnp.eye(d), "b": jnp.zeros(d), "log_var": jnp.zeros(d)}

    def affine_parameters(
        self,
        theta: dict[str, Array],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return ``(A, b, log_var)`` for this affine-Gaussian channel."""
        return theta["A"], theta["b"], theta["log_var"]


def _deterministic_log_var(dim: int, dtype) -> Array:
    """Log-variance of a noise-free channel (zero variance)."""
    return jnp.full(dim, -jnp.inf, dtype=dtype)


class Displace(AbstractAffineGaussianGate[Float[Array, " d"], tuple[int, ...]]):
    r"""Displace continuous sites by ``theta``.

    This deterministic affine-Gaussian channel applies

    $$
    x' = x + \theta.
    $$
    """

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        """Initial displacement ``theta`` of zeros over the continuous dims."""
        return jnp.zeros(sum(self.dims))

    def affine_parameters(
        self,
        theta: Float[Array, " d"],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the displacement channel as ``(A, b, log_var)``."""
        dim = sum(self.dims)
        return (
            jnp.eye(dim, dtype=theta.dtype),
            theta,
            _deterministic_log_var(dim, theta.dtype),
        )


class Scale(AbstractAffineGaussianGate[Float[Array, " d"], tuple[int, ...]]):
    r"""Scale continuous sites by $\exp(\theta)$.

    This deterministic affine-Gaussian channel applies

    $$
    x'_i = \exp(\theta_i) x_i.
    $$
    """

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        """Initial log-scale ``theta`` of zeros (i.e. unit scale)."""
        return jnp.zeros(sum(self.dims))

    def affine_parameters(
        self,
        theta: Float[Array, " d"],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the scaling channel as ``(A, b, log_var)``."""
        dim = sum(self.dims)
        return (
            jnp.diag(jnp.exp(theta)),
            jnp.zeros(dim, dtype=theta.dtype),
            _deterministic_log_var(dim, theta.dtype),
        )


class Mix(AbstractAffineGaussianGate[Float[Array, ""], tuple[int, ...]]):
    r"""Rotate exactly two scalar continuous sites by angle ``theta``.

    This deterministic affine-Gaussian channel applies

    $$
    \begin{bmatrix}x'_0\\x'_1\end{bmatrix}
    =
    \begin{bmatrix}
      \cos\theta & -\sin\theta\\
      \sin\theta & \cos\theta
    \end{bmatrix}
    \begin{bmatrix}x_0\\x_1\end{bmatrix}.
    $$

    Expects two scalar continuous sites (``dims = (1, 1)``).
    """

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, ""]:
        """Initial rotation angle ``theta`` of zero."""
        return jnp.zeros(())

    def affine_parameters(
        self,
        theta: Float[Array, ""],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the rotation channel as ``(A, b, log_var)``."""
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        matrix = jnp.array([[c, -s], [s, c]])
        return (
            matrix,
            jnp.zeros(2, dtype=matrix.dtype),
            _deterministic_log_var(2, matrix.dtype),
        )


class Diffuse(AbstractAffineGaussianGate[Float[Array, " d"], tuple[int, ...]]):
    r"""Apply Brownian diffusion with log-variance ``theta``.

    This additive affine-Gaussian channel applies

    $$
    x' = x + \epsilon,\qquad
    \epsilon \sim \mathcal{N}(0, \exp(\theta)).
    $$

    For a diffusion coefficient $D$ over time $t$, set ``theta`` to
    $\log(2 D t)$. Functionally this matches
    [`torx.psc.GaussianNoiseGate`][]; it is provided as a named
    Brownian-diffusion channel.
    """

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        """Initial log-variance ``theta`` of zeros over the continuous dims."""
        return jnp.zeros(sum(self.dims))

    def affine_parameters(
        self,
        theta: Float[Array, " d"],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the Brownian diffusion channel as ``(A, b, log_var)``."""
        dim = sum(self.dims)
        log_var = jnp.broadcast_to(theta, (dim,))
        return (
            jnp.eye(dim, dtype=log_var.dtype),
            jnp.zeros(dim, dtype=log_var.dtype),
            log_var,
        )


class MixtureGaussianGate(
    AbstractControlledContinuousGate[dict[str, Array], tuple[int, ...]]
):
    r"""
    Mixture Gaussian gate controlled by a discrete site.

    This gate samples from one of K Gaussian components, where the component
    is selected by the discrete control site. Given discrete state $k$:
    $x' = x + \mu_k + \mathcal{N}(0, \sigma_k^2)$ where
    $\sigma_k^2 = \exp$(`log_vars[k]`)

    The discrete site should take values in $\{0, 1, ..., K-1\}$ where K is
    the number of mixture components.

    !!! note

        dims only specifies continuous dimensions since we only support
        control by a single discrete site for now.

    The theta dict contains:

    - `means`: Component means of shape (K, d)
    - `log_vars`: Component log-variances of shape (K, d)
    """

    _draw_label = "MoG"

    sites: HybridSites = eqx.field(converter=_hybrid_sites_converter)
    dims: tuple[int, ...]
    num_components: int = eqx.field(static=True)

    @property
    def discrete_dims(self) -> tuple[int, ...]:
        return (self.num_components,)

    def init_params(self, key: Key[Array, ""]) -> dict[str, Array]:
        """Initial ``means`` and ``log_vars`` of zeros, shape ``(K, d)``."""
        zeros = jnp.zeros((self.num_components, sum(self.dims)))
        return {"means": zeros, "log_vars": zeros}

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: dict[str, Array],
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        x = inputs["continuous"]
        k = self._get_discrete_control(inputs)
        var = jnp.exp(params["log_vars"][k])
        mean = params["means"][k]
        noise = _draw_normal(info, key, x.shape, dtype=var.dtype)
        output = x + mean + noise * jnp.sqrt(var)
        return (output, None) if return_aux else output


class JumpDiffusionGate(
    AbstractControlledContinuousGate[dict[str, Array], tuple[int, ...]]
):
    r"""
    Jump-diffusion gate controlled by a discrete firing site.

    This gate applies continuous diffusion, and conditionally applies a jump
    based on the discrete control site. Given discrete state $j \in \{0, 1\}$:

    $$
    x' = x + \mathcal{N}(0, \sigma_d^2)
    + j \cdot (\mu_j + \mathcal{N}(0, \sigma_j^2))
    $$

    where $\sigma_d^2 = \exp$(`diff_log_var`) and
    $\sigma_j^2 = \exp$(`jump_log_var`).

    When $j = 0$, only diffusion is applied. When $j = 1$, both diffusion and
    the jump are applied.

    !!! note

        dims only specifies continuous dimensions since we only support
        control by a single discrete site for now.

    The theta dict contains:

    - `diff_log_var`: Diffusion log-variance of shape (d,)
    - `jump_mean`: Jump mean of shape (d,)
    - `jump_log_var`: Jump log-variance of shape (d,)
    """

    _draw_label = "JumpDiff"

    sites: HybridSites = eqx.field(converter=_hybrid_sites_converter)
    dims: tuple[int, ...]

    @property
    def discrete_dims(self) -> tuple[int, ...]:
        return (2,)

    def init_params(self, key: Key[Array, ""]) -> dict[str, Array]:
        """Initial ``diff_log_var``/``jump_mean``/``jump_log_var`` of zeros."""
        d = sum(self.dims)
        return {
            "diff_log_var": jnp.zeros(d),
            "jump_mean": jnp.zeros(d),
            "jump_log_var": jnp.zeros(d),
        }

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: dict[str, Array],
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        x = inputs["continuous"]
        do_jump = self._get_discrete_control(inputs)
        k1, k2 = jax.random.split(key)
        diff_var = jnp.exp(params["diff_log_var"])
        diff_noise = _draw_normal(info, k1, x.shape, dtype=diff_var.dtype)
        x = x + diff_noise * jnp.sqrt(diff_var)
        jump_var = jnp.exp(params["jump_log_var"])
        jump_noise = _draw_normal(info, k2, x.shape, dtype=jump_var.dtype)
        jump = params["jump_mean"] + jump_noise * jnp.sqrt(jump_var)
        output = x + do_jump.astype(x.dtype) * jump
        return (output, None) if return_aux else output


GaussianNoiseGate.__init__.__doc__ = (
    """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
)
AffineGaussianGate.__init__.__doc__ = (
    """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
)
Displace.__init__.__doc__ = (
    """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
)
Scale.__init__.__doc__ = """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
Mix.__init__.__doc__ = """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
Diffuse.__init__.__doc__ = """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
MixtureGaussianGate.__init__.__doc__ = (
    """See [`torx.psc.AbstractControlledContinuousGate.__init__`][]."""
)
JumpDiffusionGate.__init__.__doc__ = (
    """See [`torx.psc.AbstractControlledContinuousGate.__init__`][]."""
)
```

---

## `torx/psc/_circuit.py`

**How it composes.** A circuit is an `AbstractDFG` whose sites are generated: one `Site`
per gate per repetition, all reps of gate `i` sharing `param_key = i` (so `params` is
the thetas list and `reps` replays the same thetas), plus a terminal
`DeterministicFactor` named `"out"`. Facts that follow from the code:

- `DiscretePCircuit(gates, reps=1)`: `num_pdits = max site + 1`; a site no gate touches
  defaults to dimension 2; conflicting dims for one site raise `ValueError`. Input port
  `{"in": int32 (num_pdits,)}`, output `int32 (num_pdits,)`, so
  `circuit.sample(key, {"in": bits}, thetas)` samples without a simulator.
  `circuit_a + circuit_b` concatenates gate lists (both must have `reps == 1`).
- `HybridPCircuit(gates, reps=1)`: discrete and continuous site indices are separate
  namespaces; untouched continuous sites default to dimension 1. Input and output are
  `{"discrete": int32 (num_discrete_sites,), "continuous": float32 (continuous_state_dim,)}`;
  site `s` occupies `continuous_offsets[s] = (s, start, stop)` of the flat register.
- `init_params(key)` splits the key and returns one default theta per gate.

### Source: `torx/psc/_circuit.py`

```python
"""Probabilistic circuit classes."""

import functools
import itertools
from abc import abstractmethod
from typing import (
    Any,
    Generic,
    Iterator,
    Mapping,
    NamedTuple,
    Sequence,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, Key, PyTree

from ..dfg import AbstractDFG, Site
from ..factor import PortSpec
from ..tractable_prob_factors import DeterministicFactor
from .gates import AbstractDiscreteGate, AbstractHybridGate, AbstractPGate

_GateType = TypeVar("_GateType", bound=AbstractPGate)


class _Src(NamedTuple):
    """A wire's current source.

    ``values[writer][start : start + length]``.
    """

    writer: str
    start: int
    length: int


class _Gather(NamedTuple):
    """A slice of one parent's output.

    ``parent_outputs[parent_index][start : start + length]``.
    """

    parent_index: int
    start: int
    length: int


class _GateWiring(NamedTuple):
    """One gate's wire-level I/O.

    `reads` maps each of the gate's input ports to the wires gathered into it,
    `write_wires` are the wires the gate writes, `write_lens` their lengths
    within that output.
    """

    reads: Mapping[str, Sequence[Any]]
    write_wires: Sequence[Any]
    write_lens: Sequence[int]


class _GatherPorting(eqx.Module):
    """A `Site.porting_fn` that gathers per-wire slices from parent outputs.

    `spec` describes how to build each input port, including the dtype used
    when a port gathers no wires. Concretely it is:

        ( (port_name, (gather, gather, ...), dtype),
          (port_name, (gather, ...), dtype),
          ... )

    """

    spec: tuple[tuple[str, tuple[_Gather, ...], Any], ...] = eqx.field(static=True)

    def __call__(
        self, parent_outputs: Sequence[PyTree[Array]]
    ) -> Mapping[str, PyTree[Array]]:
        inputs = {}
        for port, gathers, dtype in self.spec:
            parts = [parent_outputs[pi][s : s + n] for pi, s, n in gathers]
            inputs[port] = jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype)
        return inputs


def _wire_sites(
    gates: Sequence[AbstractPGate],
    wirings: Sequence[_GateWiring],
    initial_src: Mapping[Any, _Src],
    output_wires: Sequence[Any] | Mapping[str, Sequence[Any]],
    output_spec: PortSpec,
    param_keys: Sequence[int],
) -> tuple[Site, ...]:
    """Build one executable `Site` per gate plus a terminal output site."""

    src = dict(initial_src)

    def parents_of(wire_lists):
        parents = []
        for wires in wire_lists:
            for w in wires:
                addr = src[w].writer
                if addr not in parents:
                    parents.append(addr)
        return tuple(parents)

    def gather(parents, wires):
        return tuple(
            _Gather(parents.index(src[w].writer), src[w].start, src[w].length)
            for w in wires
        )

    def porting(parents, reads, factor_input_ports):
        return _GatherPorting(
            tuple(
                (
                    port,
                    gather(parents, wires),
                    jax.tree.leaves(factor_input_ports[port])[0].dtype,
                )
                for port, wires in reads.items()
            )
        )

    sites = []
    for i, (gate, (reads, write_wires, write_lens), param_key) in enumerate(
        zip(gates, wirings, param_keys)
    ):
        name = f"g{i}"
        parents = parents_of(list(reads.values()))
        sites.append(
            Site(
                name=name,
                factor=gate,
                parents=parents,
                porting_fn=porting(parents, reads, gate.input_ports),
                param_key=param_key,
                info_key=None,
                site_info=None,
            )
        )
        offset = 0
        for w, length in zip(write_wires, write_lens):
            src[w] = _Src(name, offset, length)
            offset += length

    if isinstance(output_wires, Mapping):
        output_reads = output_wires
        output_input_ports = output_spec
        output_fn = lambda inputs, site_info: dict(inputs)
    else:
        output_reads = {"in": output_wires}
        output_input_ports = {"in": output_spec}
        output_fn = lambda inputs, site_info: inputs["in"]

    out_parents = parents_of(list(output_reads.values()))
    sites.append(
        Site(
            name="out",
            factor=DeterministicFactor(output_fn, output_input_ports, output_spec),
            parents=out_parents,
            porting_fn=porting(out_parents, output_reads, output_input_ports),
            param_key=None,
            info_key=None,
            site_info=None,
        )
    )
    return tuple(sites)


class AbstractPCircuit(AbstractDFG, Generic[_GateType]):
    """Abstract base class for probabilistic circuits."""

    gates: eqx.AbstractVar[list[_GateType]]
    reps: eqx.AbstractVar[int]

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Return iterator over circuit gates."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return number of gates."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> _GateType:
        """Return gate at a particular index in the circuit order."""
        raise NotImplementedError

    def init_params(self, key: Key[Array, ""]) -> list[PyTree[Array]]:
        """Default per-gate parameters, aligned with the circuit's gates.

        Splits `key` and calls each gate's `init_params`.

        **Arguments:**

        - `key`: PRNG key.

        **Returns:**

        A list of per-gate parameter pytrees, one per gate in circuit order.
        """
        keys = jax.random.split(key, max(len(self), 1))
        return [gate.init_params(k) for gate, k in zip(self, keys)]


class DiscretePCircuit(AbstractPCircuit[AbstractDiscreteGate]):
    """Probabilistic circuit for discrete gates."""

    gates: list[AbstractDiscreteGate]
    num_pdits: int
    dims: tuple[int, ...]
    reps: int
    sites: tuple[Site, ...]
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)
    output_name: str = eqx.field(static=True)
    topological_order: tuple[int, ...] = eqx.field(static=True)
    sites_by_name: Mapping[str, int] = eqx.field(static=True)

    def __init__(self, gates: list[AbstractDiscreteGate], reps: int = 1):
        """
        Initialize a probabilistic circuit.

        **Arguments:**

        - `gates`: the list of gates to be applied in the circuit
        - `reps`: the number of times the circuit is applied to the initial input
        """
        if reps < 1:
            raise ValueError("reps must be at least 1.")

        self.gates = gates
        self.reps = reps

        self.num_pdits = int(
            functools.reduce(
                lambda a, b: max(a, np.max(b.sites)),
                gates,
                -1,
            )
            + 1
        )

        dims_dict = {}
        for gate in gates:
            sites = gate.sites
            if isinstance(sites, int):
                sites = [sites]
            for i, site_idx in enumerate(sites):
                gate_dim = gate.dims[i]
                if site_idx in dims_dict and dims_dict[site_idx] != gate_dim:
                    raise ValueError(
                        f"Dimension mismatch at site {site_idx}: "
                        f"expected {dims_dict[site_idx]}, got {gate_dim}"
                    )
                dims_dict[site_idx] = gate_dim

        self.dims = tuple(dims_dict.get(i, 2) for i in range(self.num_pdits))

        # Each pdit is a wire holding a single value
        wirings = []
        for g in gates:
            wires = [g.sites] if isinstance(g.sites, int) else list(g.sites)
            wirings.append(_GateWiring({"in": wires}, wires, [1] * len(wires)))
        repeated_gates = list(gates) * reps
        repeated_wirings = wirings * reps
        param_keys = list(range(len(gates))) * reps
        dfg_sites = _wire_sites(
            repeated_gates,
            repeated_wirings,
            {p: _Src("in", p, 1) for p in range(self.num_pdits)},
            list(range(self.num_pdits)),
            jax.ShapeDtypeStruct((self.num_pdits,), jnp.int32),
            param_keys,
        )
        input_ports, sites_by_name, topological_order, output_spec = self._derive(
            dfg_sites,
            {"in": jax.ShapeDtypeStruct((self.num_pdits,), jnp.int32)},
            "out",
        )
        self.sites = dfg_sites
        self.input_ports = input_ports
        self.output_name = "out"
        self.sites_by_name = sites_by_name
        self.topological_order = topological_order
        self.output_spec = output_spec

    def __iter__(self) -> Iterator:
        """Return iterator over circuit gates."""
        return iter(self.gates)

    def __len__(self) -> int:
        """Return number of gates."""
        return len(self.gates)

    def __getitem__(self, index: int) -> AbstractDiscreteGate:
        """Return gate at a particular index in the circuit order."""
        return self.gates[index]

    def __add__(self, other):
        """Append another non-repeated circuit's gates to this circuit's gates."""
        if not isinstance(other, DiscretePCircuit):
            return NotImplemented
        if self.reps != 1 or other.reps != 1:
            raise ValueError("can only add DiscretePCircuit instances with reps == 1")
        return DiscretePCircuit(self.gates + other.gates)


_HybridGateType: TypeAlias = AbstractDiscreteGate | AbstractHybridGate


class HybridState(TypedDict):
    """Discrete and continuous registers of a hybrid circuit."""

    discrete: Int[Array, "... num_discrete_sites"]
    continuous: Float[Array, "... continuous_dim"]


class HybridPCircuit(AbstractPCircuit[_HybridGateType]):
    """Circuit supporting discrete, continuous, and hybrid gates.

    Stores both discrete and continuous dimensions separately.
    """

    gates: list[_HybridGateType]
    discrete_dims: tuple[int, ...]
    continuous_dims: tuple[int, ...]
    reps: int
    sites: tuple[Site, ...]
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)
    output_name: str = eqx.field(static=True)
    topological_order: tuple[int, ...] = eqx.field(static=True)
    sites_by_name: Mapping[str, int] = eqx.field(static=True)

    def __init__(
        self,
        gates: list[_HybridGateType],
        reps: int = 1,
    ):
        """
        Initialize a hybrid circuit.

        **Arguments:**

        - `gates`: List of gates (discrete, continuous, or hybrid).
        - `reps`: Number of times the circuit is repeated.
        """
        if reps < 1:
            raise ValueError("reps must be at least 1.")

        self.gates = gates
        self.reps = reps

        self.discrete_dims, self.continuous_dims = self._infer_dims(gates)

        # Discrete sites are width-1 wires; continuous site s is a width-
        # continuous_dims[s] wire at offset sum(continuous_dims[:s]) in the
        # continuous register.
        cont_dims = self.continuous_dims
        cont_off = self._continuous_starts

        initial_src = {
            ("d", p): _Src("discrete", p, 1) for p in range(len(self.discrete_dims))
        }
        initial_src.update(
            {
                ("c", s): _Src("continuous", cont_off[s], cont_dims[s])
                for s in range(len(cont_dims))
            }
        )

        def _wiring(gate: _HybridGateType) -> _GateWiring:
            if isinstance(gate, AbstractDiscreteGate):
                s = gate.sites
                pdits = [s] if isinstance(s, int) else list(s)
                dwires = [("d", p) for p in pdits]
                return _GateWiring({"in": dwires}, dwires, [1] * len(dwires))
            ds = gate.sites.get("discrete", [])
            cs = gate.sites.get("continuous", [])
            reads = {
                "discrete": [("d", p) for p in ds],
                "continuous": [("c", s) for s in cs],
            }
            cwires = [("c", s) for s in cs]
            return _GateWiring(reads, cwires, [cont_dims[s] for s in cs])

        repeated_gates = list(gates) * reps
        base_wirings = [_wiring(g) for g in gates]
        repeated_wirings = base_wirings * reps
        param_keys = list(range(len(gates))) * reps
        input_ports = {
            "discrete": jax.ShapeDtypeStruct((len(self.discrete_dims),), jnp.int32),
            "continuous": jax.ShapeDtypeStruct(
                (sum(self.continuous_dims),), jnp.float32
            ),
        }
        dfg_sites = _wire_sites(
            repeated_gates,
            repeated_wirings,
            initial_src,
            {
                "discrete": [("d", p) for p in range(len(self.discrete_dims))],
                "continuous": [("c", s) for s in range(len(cont_dims))],
            },
            input_ports,
            param_keys,
        )
        input_ports, sites_by_name, topological_order, output_spec = self._derive(
            dfg_sites, input_ports, "out"
        )
        self.sites = dfg_sites
        self.input_ports = input_ports
        self.output_name = "out"
        self.sites_by_name = sites_by_name
        self.topological_order = topological_order
        self.output_spec = output_spec

    @staticmethod
    def _infer_dims(
        gates: list[_HybridGateType],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        discrete_dims_dict = {}
        continuous_dims_dict = {}

        for gate in gates:
            if isinstance(gate, AbstractDiscreteGate):
                sites = gate.sites
                if isinstance(sites, int):
                    sites = [sites]
                for i, site_idx in enumerate(sites):
                    gate_dim = gate.dims[i]
                    if (
                        site_idx in discrete_dims_dict
                        and discrete_dims_dict[site_idx] != gate_dim
                    ):
                        raise ValueError(
                            f"Discrete dimension mismatch at site {site_idx}: "
                            f"expected {discrete_dims_dict[site_idx]}, got {gate_dim}"
                        )
                    discrete_dims_dict[site_idx] = gate_dim
            elif isinstance(gate, AbstractHybridGate):
                gate_sites = gate.sites
                disc_sites = gate_sites.get("discrete", [])
                cont_sites = gate_sites.get("continuous", [])

                if len(gate.discrete_dims) != len(disc_sites):
                    raise ValueError(
                        f"{gate.__class__.__name__} discrete_dims must align with "
                        f"discrete sites: expected {len(disc_sites)}, "
                        f"got {len(gate.discrete_dims)}"
                    )
                for i, site_idx in enumerate(disc_sites):
                    gate_dim = gate.discrete_dims[i]
                    discrete_dims_dict[site_idx] = gate_dim

                for i, site_idx in enumerate(cont_sites):
                    gate_dim = gate.dims[i] if i < len(gate.dims) else 1
                    if (
                        site_idx in continuous_dims_dict
                        and continuous_dims_dict[site_idx] != gate_dim
                    ):
                        raise ValueError(
                            f"Continuous dimension mismatch at site {site_idx}: "
                            f"expected {continuous_dims_dict[site_idx]}, got {gate_dim}"
                        )
                    continuous_dims_dict[site_idx] = gate_dim

        num_discrete = max(discrete_dims_dict.keys(), default=-1) + 1
        num_continuous = max(continuous_dims_dict.keys(), default=-1) + 1

        discrete_dims = tuple(discrete_dims_dict.get(i, 2) for i in range(num_discrete))
        continuous_dims = tuple(
            continuous_dims_dict.get(i, 1) for i in range(num_continuous)
        )

        return discrete_dims, continuous_dims

    @property
    def num_discrete_sites(self) -> int:
        """Number of discrete sites in the circuit."""
        return len(self.discrete_dims)

    @property
    def num_continuous_sites(self) -> int:
        """Number of continuous sites in the circuit."""
        return len(self.continuous_dims)

    @property
    def continuous_state_dim(self) -> int:
        """Total dimension of continuous state = sum(continuous_dims)."""
        return sum(self.continuous_dims)

    @property
    def _continuous_starts(self) -> tuple[int, ...]:
        """Cumulative start offset of each continuous site in the register.

        Each start is the running sum of the preceding sites' dims, so site `i`
        occupies register slots `[start_i, start_i + dim_i)`. For
        `continuous_dims = (2, 1, 3)` this returns `(0, 2, 3)`: site 0 spans
        slots 0-1, site 1 slot 2, site 2 slots 3-5.
        """
        return tuple(itertools.accumulate(self.continuous_dims, initial=0))[:-1]

    @property
    def continuous_offsets(self) -> tuple[tuple[int, int, int], ...]:
        """Per continuous site `(site, start, stop)` slice into the register."""
        return tuple(
            (site, start, start + dim)
            for site, (start, dim) in enumerate(
                zip(self._continuous_starts, self.continuous_dims)
            )
        )

    def __iter__(self):
        """Return iterator over circuit gates."""
        return iter(self.gates)

    def __len__(self) -> int:
        """Return number of gates."""
        return len(self.gates)

    def __getitem__(self, index: int) -> _HybridGateType:
        """Return gate at a particular index in the circuit order."""
        return self.gates[index]
```

---

## `torx/psc/simulation/base.py`

**How it composes.** `simulator.build_circuit(circuit, thetas)` is
`Compiled*.from_pcircuit(circuit, thetas)`: it checks `len(thetas) == len(circuit.gates)`
and returns a pytree whose `thetas` leaf you can swap with `eqx.tree_at`.
`compiled.to_pcircuit(structure)` rebuilds the circuit with the compiled `reps`.

### Source: `torx/psc/simulation/base.py`

```python
"""Abstract base classes for simulators and compiled circuits."""

from abc import abstractmethod
from typing import Generic, Type, TYPE_CHECKING, TypeVar

from ihoop.eqx import AbstractStrictModule
from jaxtyping import Array, PyTree
from typing_extensions import Self

from .._circuit import AbstractPCircuit

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


_CircuitType = TypeVar("_CircuitType", bound=AbstractPCircuit)


class AbstractCompiledPCircuit(AbstractStrictModule, Generic[_CircuitType]):
    """Abstract parent class for probabilistic circuits built for a specific backend."""

    @classmethod
    @abstractmethod
    def from_pcircuit(cls, circuit: _CircuitType, thetas: PyTree[Array]) -> Self:
        """
        Compile the given probabilistic circuit with the given parameters.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compile
        - `thetas`: The parameters to bake into the compiled circuit.

        **Returns:**

        The compiled circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def to_pcircuit(self, structure: _CircuitType) -> _CircuitType:
        """Return a circuit structure reconstructed by this compiled backend."""
        raise NotImplementedError


_CompiledType = TypeVar("_CompiledType", bound=AbstractCompiledPCircuit)


class AbstractSimulator(AbstractStrictModule, Generic[_CircuitType, _CompiledType]):
    """Abstract parent class for probabilistic circuit simulators."""

    circuit_backend: AbstractClassVar[Type[AbstractCompiledPCircuit]]

    @abstractmethod
    def expval(self, *args, **kwargs):
        """
        Compute the expectation value of the given index after circuit execution.
        """
        raise NotImplementedError

    @abstractmethod
    def expval_all(self, *args, **kwargs):
        """Compute the expectation value of all indices after circuit execution."""
        raise NotImplementedError

    @abstractmethod
    def build_circuit(
        self, circuit: _CircuitType, thetas: PyTree[Array]
    ) -> _CompiledType:
        """
        Build the circuit for this simulator with the given parameters.

        **Arguments:**

        - `circuit`: The circuit to build
        - `thetas`: The parameters to bake into the compiled circuit.

        **Returns:**

        The built circuit.
        """
        raise NotImplementedError
```

---

## `torx/psc/simulation/statevector.py`

**How it composes.** Exact and fully differentiable. `density(compiled, x)` reshapes the
flat float distribution `x` (length `prod(dims)`) to a rank-`num_pdits` tensor, applies
each gate's `get_matrix` with `tensordot`, and repeats the gate list with
`lax.fori_loop(0, reps, ...)`. `expval` / `expval_all` return `sum_i i * P(site = i)`
(for a pbit, `P(1)`). `apply_gate` is the one-gate step. Supports every
`AbstractDiscreteGate`, including generator gates.

Performance notes measured with this version on CPU: an un-jitted call re-traces and
recompiles every time, so wrap it once in `eqx.filter_jit`; and XLA's CPU compile of the
state-vector update is erratic in the number of sites (about 1.5 s at 7 sites, 50 s to
over 4 minutes at exactly 8, under 0.5 s at 9 to 12). Padding with an idle ninth site,
a `PNOT(8)` with theta `jnp.array([-jnp.inf])` (an exact identity), brings an 8-site
compile back under a second.

### Source: `torx/psc/simulation/statevector.py`

```python
"""Statevector simulators."""

from typing import ClassVar, Type

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import Self

from .._circuit import DiscretePCircuit
from .._custom_types import StateVector
from ..gates import AbstractDiscreteGate
from .base import AbstractCompiledPCircuit, AbstractSimulator


class CompiledStateVectorPCircuit(AbstractCompiledPCircuit[DiscretePCircuit]):
    """Compiled probabilistic circuit class for the state vector simulator."""

    gates: list[AbstractDiscreteGate]
    thetas: list[Float[Array, "..."]]
    num_pdits: int
    dims: tuple[int, ...]
    reps: int

    @classmethod
    def from_pcircuit(
        cls, circuit: DiscretePCircuit, thetas: list[Float[Array, "..."]]
    ) -> Self:
        """
        Compile the given probabilistic circuit with the given parameters.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compile
        - `thetas`: Per-gate parameters aligned with `circuit.gates`

        **Returns:**

        The compiled circuit.
        """
        if len(thetas) != len(circuit.gates):
            raise ValueError(
                f"expected {len(circuit.gates)} thetas aligned with "
                f"circuit.gates, got {len(thetas)}."
            )
        return cls(circuit.gates, thetas, circuit.num_pdits, circuit.dims, circuit.reps)

    def to_pcircuit(self, structure: DiscretePCircuit) -> DiscretePCircuit:
        """Return a `DiscretePCircuit` with this compiled circuit's gates and reps."""
        return DiscretePCircuit(self.gates, self.reps)


# todo: as the DFG matures, we can probably merge this into the PSC DFG .sample
class StateVectorSimulator(
    AbstractSimulator[DiscretePCircuit, CompiledStateVectorPCircuit]
):
    """Simulator for exact state vectors representing probability distributions."""

    circuit_backend: ClassVar[Type[AbstractCompiledPCircuit]] = (
        CompiledStateVectorPCircuit
    )

    def build_circuit(
        self, circuit: DiscretePCircuit, thetas: list[Float[Array, "..."]]
    ) -> CompiledStateVectorPCircuit:
        return CompiledStateVectorPCircuit.from_pcircuit(circuit, thetas)

    @staticmethod
    def apply_gate(
        state: StateVector,
        gate: AbstractDiscreteGate,
        theta: Float[Array, "..."],
        dims: tuple[int, ...],
    ) -> StateVector:
        """
        Apply gate to the state StateVector and return the resulting state.

        **Arguments:**

        - `state`: The state to apply the gate to
        - `gate`: The gate to apply
        - `theta`: The gate's parameters
        - `dims`: The dimensions of all sites in the circuit

        **Returns:**

        The state after applying the gate.
        """
        # Get site indices
        sites = gate.sites
        if isinstance(sites, int):
            sites = [sites]
        else:
            sites = list(sites)

        # Get dimensions for the sites this gate acts on
        sub_dims = tuple(dims[s] for s in sites)

        matrix = gate.get_matrix(theta)

        # Reshape state to tensor form
        reshaped_state = state.reshape(dims)

        # Reshape matrix from (prod(sub_dims), prod(sub_dims)) to
        # (d0, d1, ..., d0', ...)
        # First half are output dims, second half are input dims
        matrix_shape = sub_dims + sub_dims
        reshaped_matrix = matrix.reshape(matrix_shape)

        num_sites = len(sites)
        matrix_contract_axes = list(range(num_sites, 2 * num_sites))

        result = jnp.tensordot(
            reshaped_matrix, reshaped_state, axes=(matrix_contract_axes, sites)
        )

        # tensordot puts output axes first, move them back to original positions
        # result shape is (d0, d1, ..., remaining_dims...)
        result = jnp.moveaxis(result, list(range(num_sites)), sites)

        return result.reshape(-1)

    def density(
        self, circuit: CompiledStateVectorPCircuit, x: StateVector
    ) -> StateVector:
        """
        Compute the final distribution over computational basis states.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compute the distribution of
        - `x`: The initial state vector, also the initial distribution

        **Returns:**

        The final distribution of the circuit.
        """
        dims = circuit.dims
        state = x.reshape(dims)

        def body_fn(_: int, state: StateVector) -> StateVector:
            # todo: we could also allow an option to pad the gates out
            # this would reduce compilation time, but potentially increase memory
            for gate, theta in zip(circuit.gates, circuit.thetas):
                state = self.apply_gate(state.reshape(-1), gate, theta, dims).reshape(
                    dims
                )

            return state

        state = jax.lax.fori_loop(0, circuit.reps, body_fn, state)

        return state.reshape(-1)

    def expval(
        self, circuit: CompiledStateVectorPCircuit, x: StateVector, pbit: int
    ) -> Float[Array, ""]:
        r"""
        Compute the expectation value of the given discrete site after circuit
        execution.

        For a site with basis values $0, \ldots, d - 1$, this returns
        $\sum_i i \, P(x = i)$. For binary sites, this is equivalent to the
        probability of measuring ``1``.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial state vector, also the initial distribution
        - `pbit`: The index of the site to compute the final expectation value of

        **Returns:**

        The expectation value of the given site after circuit execution.
        """
        dims = circuit.dims
        state = self.density(circuit, x).reshape(dims)

        sum_axes = list(range(circuit.num_pdits))
        sum_axes.remove(pbit)
        marginal = jnp.sum(state, axis=tuple(sum_axes))
        values = jnp.arange(dims[pbit], dtype=marginal.dtype)
        return jnp.sum(values * marginal)

    def expval_all(
        self, circuit: CompiledStateVectorPCircuit, x: StateVector
    ) -> Float[Array, " pbits"]:
        r"""
        Compute the expectation value of all discrete sites after circuit execution.

        For each site with basis values $0, \ldots, d - 1$, this returns
        $\sum_i i \, P(x = i)$. For binary sites, this is equivalent to the
        probability of measuring ``1``.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial state vector, also the initial distribution

        **Returns:**

        An array containing the expectation values of all discrete sites.
        """
        dims = circuit.dims
        state = self.density(circuit, x).reshape(dims)

        expvals = []
        for pbit in range(circuit.num_pdits):
            sum_axes = list(range(circuit.num_pdits))
            sum_axes.remove(pbit)
            marginal = jnp.sum(state, axis=tuple(sum_axes))
            values = jnp.arange(dims[pbit], dtype=marginal.dtype)
            expvals.append(jnp.sum(values * marginal))

        return jnp.stack(expvals)
```

---

## `torx/psc/simulation/sampled.py`

**How it composes.** `BranchingSimulator(diff_method="param_shift_inf", num_samples=1)`
never builds a state vector. `build_circuit` lowers every gate to its branch lookup
tables padded to a common locality and basis size, and stacks the thetas into one
`(num_gates, max_branches - 1)` array padded with `-inf`; non-`AbstractKBranchGate`
gates (`PISING`) raise `ValueError` here. `sample_circuit` draws every branch choice up
front, `(reps, num_gates, num_samples)`, then scans gates and reps applying the chosen
tables; `sample` returns the `(num_samples, num_sites)` int states and `expval*` their
means. The initial bitstring must have length `num_pdits`.

Gradients are custom VJPs on the compiled `thetas` only:

| `diff_method` | Rule | Cost per backward | Limits |
|---------------|------|-------------------|--------|
| `"param_shift_inf"` | clamp each gate to each branch (`theta -> +-inf`): `dE/dtheta_k = p_k (E[O | k] - E[O])` | `num_gates * max_branches` extra estimates | **`reps == 1` only** (`NotImplementedError` otherwise) |
| `"param_shift_single"` | shifted-logit rule reusing the primal | `num_gates` extra estimates | **`reps == 1` and two-branch gates only** |
| `"param_shift_filter"` | partition the primal samples by the branch each gate took | none | any `reps`, any K; noisiest |

The compiled circuit stores `num_pdits`, `reps` and `max_branches` as ordinary int
fields, so passing it as an argument to plain `jax.jit` fails with a
`TracerBoolConversionError`. Use `eqx.filter_jit` or close over it.

### Source: `torx/psc/simulation/sampled.py`

```python
"""Branch-sampling simulator for lookup-table probabilistic gates."""

import functools
import math
from typing import ClassVar, Literal, Type

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Key, PyTree
from typing_extensions import Self

from ..._sampler import _resolve, AbstractSampler
from .._circuit import DiscretePCircuit
from .._custom_types import BitString
from ..gates import AbstractGeneratorGate, AbstractKBranchGate
from .base import AbstractCompiledPCircuit, AbstractSimulator


class CompiledBranchingPCircuit(AbstractCompiledPCircuit[DiscretePCircuit]):
    """Compiled probabilistic circuit for the branching simulator."""

    num_pdits: int
    reps: int
    max_branches: int

    branch_ops: Int[Array, "num_gates max_branches max_basis max_l"]
    num_branches: Int[Array, " num_gates"]  # K for each gate
    sites: Int[Array, "num_gates l"]  # index of sites (e.g. wires operated on)
    dims: Int[Array, "num_gates l"]  # per-site dimensions, padded
    basis_sizes: Int[Array, " num_gates"]  # prod(dims) for each gate
    thetas: Float[
        Array, "num_gates max_branches_minus_1"
    ]  # K-1 params per gate, padded with -inf

    @classmethod
    def from_pcircuit(
        cls,
        circuit: DiscretePCircuit,
        thetas: list[Float[Array, "..."]],
    ) -> Self:
        r"""
        Compile the given probabilistic circuit with the given parameters.

        This compiled form removes the list of probabilistic gates in favour
        of a more JIT-friendly representation. This representation uses the
        matrix forms of the branches of the probabilistic gates.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compile
        - `thetas`: Per-gate parameters aligned with `circuit.gates`; stacked
            and padded to `(num_gates, max_branches - 1)` with `-inf`.

        **Returns:**

        The compiled circuit.
        """
        if len(thetas) != len(circuit.gates):
            raise ValueError(
                f"expected {len(circuit.gates)} thetas aligned with "
                f"circuit.gates, got {len(thetas)}."
            )
        if not circuit.gates:
            raise ValueError("BranchingSimulator requires at least one gate.")

        # the maximum number of pdits that any single gate in the circuit acts on
        def _f(a, b):
            return max(a, len(b.sites) if isinstance(b.sites, list) else 1)

        locality = functools.reduce(
            _f,
            circuit.gates,
            1,
        )

        # getattr: pyright types jax.config attrs inconsistently across platforms
        dtype = jnp.int64 if getattr(jax.config, "jax_enable_x64") else jnp.int32

        gates = [
            gate for gate in circuit.gates if isinstance(gate, AbstractKBranchGate)
        ]
        if len(gates) != len(circuit.gates):
            unsupported_gates = [
                gate
                for gate in circuit.gates
                if not isinstance(gate, AbstractKBranchGate)
            ]
            unsupported = [type(gate).__name__ for gate in unsupported_gates]
            generator_gates = [
                type(gate).__name__
                for gate in unsupported_gates
                if isinstance(gate, AbstractGeneratorGate)
            ]
            generator_hint = (
                " Generator gates require StateVectorSimulator: "
                f"{', '.join(generator_gates)}."
                if generator_gates
                else ""
            )
            raise ValueError(
                "BranchingSimulator only supports AbstractKBranchGate lookup-table "
                f"gates; unsupported gates: {', '.join(unsupported)}."
                f"{generator_hint}"
            )

        max_branches = max(gate.num_branches for gate in gates)

        # compute per-gate dims and basis sizes
        gate_dims = []
        gate_basis_sizes = []
        for gate in gates:
            dims = list(gate.dims)
            num_sites = len(dims)
            # virtual padding sites
            dims_padded = dims + [2] * (locality - num_sites)
            gate_dims.append(dims_padded)
            gate_basis_sizes.append(math.prod(dims_padded))

        max_basis = max(gate_basis_sizes)
        dims_array = jnp.array(gate_dims, dtype=dtype)
        basis_sizes = jnp.array(gate_basis_sizes, dtype=dtype)

        # extend branch ops to max basis size, locality, and max_branches
        branch_ops_list = []
        num_branches_list = []
        for i, gate in enumerate(gates):
            K = gate.num_branches
            num_branches_list.append(K)

            # gate.branches has shape (K, basis_size, num_sites)
            # extend each branch to (max_basis, locality)
            extended_branches = jnp.stack(
                [
                    cls._extend_op(gate.branches[k], locality, max_basis, gate_dims[i])
                    for k in range(K)
                ]
            )

            # Padded branches are unreachable in normal sampling because their
            # logits are -inf. Deterministic gradient probes may index them to
            # keep vmaps rectangular, but their probabilities stay zero, so any
            # finite branch table is equivalent.
            if K < max_branches:
                padding = jnp.tile(extended_branches[0:1], (max_branches - K, 1, 1))
                extended_branches = jnp.concatenate(
                    [extended_branches, padding], axis=0
                )

            branch_ops_list.append(extended_branches)

        branch_ops = jnp.stack(branch_ops_list).astype(dtype)
        num_branches = jnp.array(num_branches_list, dtype=dtype)

        # the sites that the gates act on, extended to the appropriate locality
        sites = jnp.stack(
            [
                cls._extend_sites(gate.sites, locality, circuit.num_pdits)
                for gate in gates
            ]
        ).astype(dtype)

        # pad thetas to (num_gates, max_branches - 1) with -inf
        thetas_list = []
        for gate, theta in zip(gates, thetas):
            theta_vals = jnp.atleast_1d(theta)
            K = gate.num_branches
            # theta has K-1 values, pad to max_branches - 1 with -inf
            padded_theta = jnp.full(max_branches - 1, -jnp.inf)
            padded_theta = padded_theta.at[: K - 1].set(theta_vals[: K - 1])
            thetas_list.append(padded_theta)
        padded_thetas = jnp.stack(thetas_list)

        return cls(
            circuit.num_pdits,
            circuit.reps,
            max_branches,
            branch_ops,
            num_branches,
            sites,
            dims_array,
            basis_sizes,
            padded_thetas,
        )

    def to_pcircuit(self, structure: DiscretePCircuit) -> DiscretePCircuit:
        """Return a `DiscretePCircuit` with `structure`'s gates and compiled reps."""
        return DiscretePCircuit(structure.gates, self.reps)

    def to_thetas(self) -> list[Float[Array, "..."]]:
        """Recover the per-gate parameter list from the padded `thetas`.

        Inverse of the stacking done in `from_pcircuit`: each gate's theta is
        the first `K - 1` entries of its padded row.

        **Returns:**

        A list of per-gate thetas, aligned with the original gates.
        """
        num_gates = self.thetas.shape[0]
        return [
            self.thetas[i, : int(self.num_branches[i]) - 1] for i in range(num_gates)
        ]

    @staticmethod
    def _extend_op(
        op: Int[Array, "basis_size sites"],
        locality: int,
        max_basis: int,
        dims_padded: list[int],
    ) -> Int[Array, "max_basis locality"]:
        """
        Extend the dimensions of the lookup table of a gate.

        "Locality" in this context means the maximum number of pdits that any
        single gate in the circuit acts on. Every gate that acts on less
        than this number of pdits must be padded with identity operators.

        This function is necessary since only matrices with the same shape can
        be stacked with each other. For example, a PNOT gate has the lookup
        table [[1], [0]] but a PCNOT gate has [[0, 0], [0, 1], [1, 1], [1, 0]],
        which is a different shape. In this case, it is necessary to extend
        the PNOT to 2 pdits (with an identity operator on the second pdit).

        The number of rows is prod(dims). Rows are padded to max_basis with
        zeros (these rows should never be indexed if basis_sizes is used
        correctly).

        **Arguments:**

        - `op`: the lookup table of a probabilistic gate
        - `locality`: the number of pdits to extend the gate to
        - `max_basis`: the maximum basis size across all gates
        - `dims_padded`: the dimensions of each site, padded to locality with 1s

        **Returns:**

        The new lookup table with shape (max_basis, locality).
        """
        num_sites = op.shape[1]
        # basis_size = op.shape[0]

        # compute the repeat factor from padded dims
        # for each padded site, we need to repeat the rows
        repeat_factor = math.prod(dims_padded[num_sites:])

        # repeat rows for the padded dimensions
        new_op = jnp.repeat(op, repeat_factor, axis=0)

        # pad columns for extra sites
        new_op = jnp.pad(new_op, [(0, 0), (0, locality - num_sites)])

        # pad rows to max_basis (these should never be accessed)
        current_rows = new_op.shape[0]
        if current_rows < max_basis:
            new_op = jnp.pad(new_op, [(0, max_basis - current_rows), (0, 0)])

        return new_op

    @staticmethod
    def _extend_sites(
        sites: int | list[int], locality: int, num_sites: int
    ) -> Int[Array, " locality"]:
        """
        Extend the number of sites of a gate to the given locality.

        "Locality" in this context means the maximum number of sites that any
        single gate in the circuit acts on. Every gate that acts on less
        than this number of sites must be padded with identity operators.

        This function is necessary since only arrays with the same length can
        be stacked with each other. For example, a PNOT gate has a site
        list of length 1 but a PCNOT gate has a site list of length 2. Thus
        it is necessary to extend the PNOT to 2 sites (with an identity
        operator on the second site).

        **Arguments:**

        - `sites`: the sites that the probabilistic gate acts on
        - `locality`: the number of sites to extend the gate to
        - `num_sites`: the total number of sites in the circuit

        **Returns:**

        The new sites.
        """
        sts = jnp.array([sites]) if isinstance(sites, int) else jnp.array(sites)
        new_sites = jnp.concatenate(
            [sts, jnp.arange(num_sites, num_sites + locality - len(sts))]
        )
        return new_sites


_DiffMethod = Literal["param_shift_inf", "param_shift_single", "param_shift_filter"]


class BranchingSimulator(
    AbstractSimulator[DiscretePCircuit, CompiledBranchingPCircuit]
):
    r"""
    A branch-sampling simulator for lookup-table probabilistic circuits.

    Instead of storing full state vectors, it samples from distributions.

    ??? tip "Choosing a differentiation method"

        Three differentiation methods are available:

        - `"param_shift_inf"`: Uses the parameter shift rule with deterministic
          gates ($\theta \to \pm\infty$). Requires $2N$ circuit evaluations for
          $N$ parameters.

        - `"param_shift_single"`: Uses the parameter shift rule with primal reuse.
          Requires $N$ circuit evaluations.

        - `"param_shift_filter"`: Estimates gradients from a single forward pass
          by filtering samples based on which branch was taken at each gate. For
          each gate, samples are partitioned into those that applied the gate vs.
          those that did not, and the gradient is estimated from the difference
          in expectation values between these groups.

    """

    diff_method: _DiffMethod
    num_samples: int

    circuit_backend: ClassVar[Type[AbstractCompiledPCircuit]] = (
        CompiledBranchingPCircuit
    )

    def __init__(
        self,
        diff_method: _DiffMethod = "param_shift_inf",
        num_samples: int = 1,
    ):
        """
        Initialize the branching simulator.

        **Arguments:**

        - `diff_method`: method used for differentiating circuit parameters
        - `num_samples`: number of samples used to estimate expectation values
        """
        if diff_method not in [
            "param_shift_inf",
            "param_shift_single",
            "param_shift_filter",
        ]:
            raise ValueError(
                "diff_method must be param_shift_inf, param_shift_single, "
                "or param_shift_filter"
            )
        self.num_samples = num_samples
        self.diff_method = diff_method

    def sample(
        self,
        circuit: CompiledBranchingPCircuit,
        x: BitString,
        key: Key[Array, ""],
        *,
        sampler: AbstractSampler | None = None,
    ) -> Int[Array, "num_samples num_pbits"]:
        """
        Obtain samples from the final distribution of the probabilistic circuit.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial computational basis state of the circuit
        - `key`: The random key to use to obtain samples
        - `sampler`: Optional distribution provider, defaulting to `jax.random`.

        **Returns:**

        An integer array with shape (num_samples, num_pbits) containing
        the computational basis state samples.
        """
        if x.shape[0] != circuit.num_pdits:
            raise ValueError(
                f"Malformed bitstring, shape {x.shape[0]} should match "
                f"number of pbits {circuit.num_pdits}"
            )
        return sample_circuit(
            circuit, x, key, num_samples=self.num_samples, sampler=sampler
        )[0]

    def expval(
        self,
        circuit: CompiledBranchingPCircuit,
        x: BitString,
        pbit: int,
        key: Key[Array, ""],
        *,
        sampler: AbstractSampler | None = None,
    ) -> Float[Array, ""]:
        r"""
        Estimate the expectation value of the given discrete site after circuit
        execution.

        For a site with basis values $0, \ldots, d - 1$, this estimates
        $\mathbb{E}[x] = \sum_i i \, P(x = i)$.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial computational basis state of the circuit
        - `pbit`: The index of the site to estimate the final expectation value of
        - `key`: The random key to use to obtain samples
        - `sampler`: Optional distribution provider, defaulting to `jax.random`.

        **Returns:**

        The expectation value of the given site after circuit execution.
        """
        return self.expval_all(circuit, x, key, sampler=sampler)[pbit]

    def expval_all(
        self,
        circuit: CompiledBranchingPCircuit,
        x: BitString,
        key: Key[Array, ""],
        *,
        sampler: AbstractSampler | None = None,
    ) -> Float[Array, " num_pbits"]:
        r"""
        Estimate the expectation value of all discrete sites after circuit execution.

        For each site with basis values $0, \ldots, d - 1$, this estimates
        $\mathbb{E}[x] = \sum_i i \, P(x = i)$.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial computational basis state of the circuit
        - `key`: The random key to use to obtain samples
        - `sampler`: Runtime distribution provider used in forward and backward
          sampling. Its parameters are not differentiated.

        **Returns:**

        An array containing the expectation values of all discrete sites.
        """
        # we don't currently support differentiation of circuits with multiple
        # repetitions with the param_shift_inf and param_shift_single variants,
        # because this would involve unrolling the scan in the sampling function
        # function, which would lead to large compilation times
        if self.diff_method == "param_shift_inf":
            return sample_expval_all_param_shift_inf(
                circuit, x, key, num_samples=self.num_samples, sampler=sampler
            )

        if self.diff_method == "param_shift_single":
            return sample_expval_all_param_shift_single(
                circuit, x, key, num_samples=self.num_samples, sampler=sampler
            )

        return sample_expval_all_param_shift_filter(
            circuit, x, key, num_samples=self.num_samples, sampler=sampler
        )

    def build_circuit(
        self, circuit: DiscretePCircuit, thetas: list[Float[Array, "..."]]
    ) -> CompiledBranchingPCircuit:
        """Compile ``circuit`` for branch sampling.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compile.
        - `thetas`: Per-gate parameters aligned with `circuit.gates`.

        **Returns:**

        The compiled circuit.
        """
        return CompiledBranchingPCircuit.from_pcircuit(circuit, thetas)


def sample_circuit(
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> tuple[
    Int[Array, "num_samples num_pbits"], Int[Array, "reps num_gates num_samples"]
]:
    """
    Obtain samples from the final distribution of the probabilistic circuit.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain
    - `sampler`: Optional runtime distribution provider.

    **Returns:**

    A tuple of:
    - An integer array with shape (num_samples, num_pbits) containing the
      computational basis state samples.
    - An integer array with shape (reps, num_gates, num_samples) containing
      the branch indices that were sampled for each gate.
    """
    # (num_samples, num_pbits)
    state = jnp.tile(x[None, :], (num_samples, 1))

    def _apply_gate(
        state: Int[Array, " num_pbits"],
        branch_ops: Int[Array, "K B l"],
        sites: Int[Array, " l"],
        dims: Int[Array, " l"],
        branch_idx: Int[Array, ""],
    ) -> Int[Array, " num_pbits"]:
        # getattr: pyright types jax.config attrs inconsistently across platforms
        dtype = jnp.int64 if getattr(jax.config, "jax_enable_x64") else jnp.int32

        # (locality,)
        # Sites >= num_pbits are virtual padding sites
        # append zeros before indexing
        virtual_state = jnp.zeros((sites.shape[0],), dtype=state.dtype)
        padded_state = jnp.concatenate([state, virtual_state])
        substate = padded_state[sites].astype(dtype)

        # index = sum_i substate[i] * prod(dims[i+1:])
        # compute strides: [prod(dims[1:]), prod(dims[2:]), ..., 1]
        strides = jnp.cumprod(dims[::-1])[::-1]
        strides = jnp.concatenate([strides[1:], jnp.array([1], dtype=dtype)])
        index = jnp.sum(substate * strides)

        # select the branch operation based on branch_idx
        selected_op = branch_ops[branch_idx]
        new_substate = selected_op[index]

        return state.at[sites].set(new_substate)

    # inner loop scans over the gates of the circuit
    def inner_body_fn(
        carry: Int[Array, "num_samples num_pbits"], x: tuple
    ) -> tuple[Int[Array, "num_samples num_pbits"], None]:
        # (num_samples, num_pbits)
        state = carry

        # (K, B, locality), (locality,), (locality,), (num_samples,)
        branch_ops, sites, dims, branch_idx = x

        # (num_samples, num_pbits)
        state = jax.vmap(_apply_gate, in_axes=(0, None, None, None, 0))(
            state, branch_ops, sites, dims, branch_idx
        )
        return state, None

    # outer loop scans over the number of repetitions
    def outer_body_fn(
        carry: Int[Array, "num_samples num_pbits"],
        x: Int[Array, "num_gates num_samples"],
    ) -> tuple[Int[Array, "num_samples num_pbits"], None]:
        # (num_samples, num_pbits)
        state = carry

        # (num_gates, num_samples)
        branch_indices = x

        # (num_samples, num_pbits)
        state = jax.lax.scan(
            inner_body_fn, state, (branch_ops, sites, dims, branch_indices)
        )[0]
        return state, None

    # (num_gates, max_branches, B, locality)
    branch_ops = circuit.branch_ops

    # (num_gates, locality)
    sites = circuit.sites
    dims = circuit.dims

    num_gates = circuit.thetas.shape[0]
    sampler = _resolve(sampler)

    if circuit.max_branches == 2:
        probs = jax.nn.sigmoid(circuit.thetas[:, 0])
        branch_indices = sampler.bernoulli(
            key,
            probs[None, :, None],
            shape=(circuit.reps, num_gates, num_samples),
        ).astype(jnp.int32)
    else:
        padded_logits = jnp.concatenate(
            [jnp.zeros((num_gates, 1)), circuit.thetas], axis=1
        )
        branch_indices = sampler.categorical(
            key,
            padded_logits[None, :, None, :],
            axis=-1,
            shape=(circuit.reps, num_gates, num_samples),
        )

    # (num_samples, num_pbits)
    state = jax.lax.scan(outer_body_fn, state, branch_indices)[0]

    return state, branch_indices


def _expval_all(
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> Float[Array, " pbits"]:
    r"""
    Estimate the expectation value of all discrete sites after circuit execution.

    For each site with basis values $0, \ldots, d - 1$, this estimates
    $\mathbb{E}[x] = \sum_i i \, P(x = i)$.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain to estimate the expval

    **Returns:**

    An array containing the expectation values of all discrete sites.
    """
    samples = sample_circuit(circuit, x, key, num_samples, sampler)[0]
    val = jnp.mean(samples.astype(jnp.float32), axis=0)
    return val


@eqx.filter_custom_vjp
def sample_expval_all_param_shift_inf(
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> Float[Array, " num_pbits"]:
    r"""
    Estimate the expectation value of all pbits after circuit execution.

    This is a wrapper function for `_expval_all` so that a custom VJP rule can
    be defined without affecting core functionality. The VJP uses the
    parameter shift rule with deterministic gates. For K-branch gates with
    softmax probabilities, the gradient with respect to logits is:

    $$\frac{\partial \mathbb{E}[O]}{\partial \theta_k} =
        p_k \left( \mathbb{E}[O|k] - \mathbb{E}[O] \right)$$

    where $p_k = \text{softmax}([0, \theta])_k$ is the probability of branch $k$,
    and $\mathbb{E}[O|k]$ is the expected observable when branch $k$ is selected
    deterministically.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain to estimate the expval

    **Returns:**

    An array containing the expectation values of all the pbits.
    """
    return _expval_all(circuit, x, key, num_samples, sampler)


@sample_expval_all_param_shift_inf.def_fwd
def sample_expval_all_param_shift_inf_fwd(
    perturbed: CompiledBranchingPCircuit,
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> tuple[Float[Array, " num_pbits"], None]:
    """Perform the forward execution of sample_expval_all_param_shift_inf."""
    return _expval_all(circuit, x, key, num_samples, sampler), None


@sample_expval_all_param_shift_inf.def_bwd
def sample_expval_all_param_shift_inf_bwd(
    residuals: PyTree,
    grad_out: Float[Array, " num_pbits"],
    perturbed: CompiledBranchingPCircuit,
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> CompiledBranchingPCircuit:
    """Perform the backward execution of sample_expval_all_param_shift_inf."""
    if circuit.reps != 1:
        raise NotImplementedError(
            "Deterministic variant of parameter-shift for multiple repetitions "
            "is not yet implemented. For computing gradients of circuits with "
            "multiple repetitions, use 'param_shift_filter' method instead."
        )

    trainable = eqx.filter(circuit, perturbed)
    num_gates = circuit.thetas.shape[0]
    max_branches = circuit.max_branches

    # For each gate g and branch k, we need to evaluate the circuit with
    # gate g set to deterministically select branch k.
    # This requires num_gates * max_branches evaluations.
    def make_deterministic_theta(g, k, base_thetas):
        new_thetas = base_thetas.copy()
        if k == 0:
            new_thetas = new_thetas.at[g, :].set(-jnp.inf)
        else:
            new_thetas = new_thetas.at[g, :].set(-jnp.inf)
            new_thetas = new_thetas.at[g, k - 1].set(jnp.inf)
        return new_thetas

    thetas_list = []
    for g in range(num_gates):
        for k in range(max_branches):
            thetas_list.append(make_deterministic_theta(g, k, circuit.thetas))

    # Stack into (num_gates * max_branches, num_gates, max_branches-1)
    thetas_deterministic = jnp.stack(thetas_list)

    circuit_deterministic = jax.tree.unflatten(
        jax.tree.structure(trainable), (thetas_deterministic,)
    )
    circuit_deterministic = eqx.combine(circuit_deterministic, circuit)

    keys = jax.random.split(key, num_gates * max_branches)
    vmap_axes = (
        eqx.filter(trainable, perturbed, inverse=True, replace=0),
        None,
        0,
        None,
        None,
    )

    # (num_gates * max_branches, num_pbits)
    expvals_deterministic = eqx.filter_vmap(_expval_all, in_axes=vmap_axes)(
        circuit_deterministic, x, keys, num_samples, sampler
    )

    # Reshape to (num_gates, max_branches, num_pbits)
    expvals_per_branch = expvals_deterministic.reshape(num_gates, max_branches, -1)

    if max_branches == 2:
        sig = jax.nn.sigmoid(circuit.thetas[:, 0])
        # (num_gates, 2)
        probs = jnp.stack([1 - sig, sig], axis=-1)
    else:
        padded_logits = jnp.concatenate(
            [jnp.zeros((num_gates, 1)), circuit.thetas], axis=1
        )  # (num_gates, max_branches)
        probs = jax.nn.softmax(padded_logits, axis=-1)

    # grad_theta[g, j] = sum_k dp_k/dtheta_j * (expval_branch[g,k] @ grad_out)
    # where dp_k/dtheta_j = p_k * (delta_{k,j+1} - p_{j+1})
    # grad_theta[g, j] = p_{j+1} * (expval_branch[g,j+1] - expval_overall[g]) @ grad_out

    # weighted over branches
    # (num_gates, num_pbits)
    expval_overall = jnp.einsum("gk,gkp->gp", probs, expvals_per_branch)

    # gradient for each theta parameter
    # (num_gates, max_branches-1)
    # grad[g, j] = p[g, j+1] * (expval_branch[g, j+1] - expval_overall[g]) @ grad_out
    diff = (
        expvals_per_branch[:, 1:, :] - expval_overall[:, None, :]
    )  # (num_gates, max_branches-1, num_pbits)
    grads_in = probs[:, 1:] * jnp.einsum(
        "gjp,p->gj", diff, grad_out
    )  # (num_gates, max_branches-1)

    grad_circuit = jax.tree.unflatten(jax.tree.structure(trainable), (grads_in,))

    return grad_circuit


@eqx.filter_custom_vjp
def sample_expval_all_param_shift_single(
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> Float[Array, " num_pbits"]:
    r"""
    Estimate the expectation value of all pbits after circuit execution.

    This is a wrapper function for `_expval` so that a custom VJP rule can be
    defined without affecting core functionality. The VJP uses the
    parameter shift rule with reuse of the primal. This rule is given by:

    $$\frac{\partial U(\theta)}{\partial\theta} = U(\theta) -
        U\left( -\ln\left[(1 + \exp(-\theta))^2 - 1 \right]  \right)$$

    where $U(\theta)$ taken from the primal.

    **Note:** This method only supports 2-branch (K=2) gates. For circuits
    with K>2 gates, use `param_shift_inf` or `param_shift_filter` instead.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain to estimate the expval

    **Returns:**

    An array containing the expectation values of all the pbits.
    """
    return _expval_all(circuit, x, key, num_samples, sampler)


@sample_expval_all_param_shift_single.def_fwd
def sample_expval_all_param_shift_single_fwd(
    perturbed: CompiledBranchingPCircuit,
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> tuple[Float[Array, " num_pbits"], Float[Array, " num_pbits"]]:
    """Perform the forward execution of sample_expval_all_param_shift_single."""
    expval = _expval_all(circuit, x, key, num_samples, sampler)
    return expval, expval


@sample_expval_all_param_shift_single.def_bwd
def sample_expval_all_param_shift_single_bwd(
    residuals: PyTree,
    grad_out: Float[Array, " num_pbits"],
    perturbed: CompiledBranchingPCircuit,
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> CompiledBranchingPCircuit:
    """Perform the backward execution of sample_expval_all_param_shift_single."""
    # (num_pbits,)
    expval_primal = residuals

    if circuit.reps != 1:
        raise NotImplementedError(
            "Single-shift variant of parameter-shift for multiple repetitions "
            "is not yet implemented. For computing gradients of circuits with "
            "multiple repetitions, use 'param_shift_filter' method instead."
        )

    # Check if any gate has K > 2 branches
    if circuit.max_branches > 2:
        raise NotImplementedError(
            "param_shift_single only supports 2-branch gates. "
            "Use 'param_shift_inf' or 'param_shift_filter' for K-branch gates."
        )

    trainable = eqx.filter(circuit, perturbed)
    num_gates = circuit.thetas.shape[0]

    # For K=2 gates, thetas has shape (num_gates, 1)
    theta_flat = circuit.thetas[:, 0]  # (num_gates,)

    # (num_gates, num_gates, 1)
    shifted_thetas = jnp.tile(circuit.thetas[None, :, :], (num_gates, 1, 1))
    shift_values = -jnp.log((1 + jnp.exp(-theta_flat)) ** 2 - 1)  # (num_gates,)
    shifted_thetas = shifted_thetas.at[
        jnp.arange(num_gates), jnp.arange(num_gates), 0
    ].set(shift_values)

    param_shift_circuit = jax.tree.unflatten(
        jax.tree.structure(trainable), (shifted_thetas,)
    )
    param_shift_circuit = eqx.combine(param_shift_circuit, circuit)

    keys = jax.random.split(key, num_gates)
    vmap_axes = (
        eqx.filter(trainable, perturbed, inverse=True, replace=0),
        None,
        0,
        None,
        None,
    )

    # (num_gates, num_pbits)
    expvals = eqx.filter_vmap(_expval_all, in_axes=vmap_axes)(
        param_shift_circuit, x, keys, num_samples, sampler
    )

    # (num_gates,)
    grad_flat = (expval_primal - expvals) @ grad_out

    # Reshape to (num_gates, 1) to match thetas shape
    grads_in = grad_flat[:, None]

    grad_circuit = jax.tree.unflatten(jax.tree.structure(trainable), (grads_in,))

    return grad_circuit


@eqx.filter_custom_vjp
def sample_expval_all_param_shift_filter(
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> Float[Array, " num_pbits"]:
    r"""
    Estimate the expectation value of all pbits after circuit execution.

    This is a wrapper function for `_expval_all` so that a custom VJP rule can
    be defined without affecting core functionality. The VJP uses the
    parameter shift rule with deterministic gates, but with no additional
    circuit executions. For K-branch gates with softmax probabilities, the
    gradient with respect to logits is:

    $$\frac{\partial \mathbb{E}[O]}{\partial \theta_k} =
        p_k \left( \mathbb{E}[O|k] - \mathbb{E}[O] \right)$$

    where $p_k = \text{softmax}([0, \theta])_k$ is the probability of branch $k$.
    The difference between this VJP rule and `param_shift_inf` above is that,
    for this rule, the conditional expectations $\mathbb{E}[O|k]$ are estimated
    directly from the primal execution by filtering samples based on which
    branch was selected at each gate.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain to estimate the expval

    **Returns:**

    An array containing the expectation values of all the pbits.
    """
    return _expval_all(circuit, x, key, num_samples, sampler)


@sample_expval_all_param_shift_filter.def_fwd
def sample_expval_all_param_shift_filter_fwd(
    perturbed: CompiledBranchingPCircuit,
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> tuple[
    Float[Array, " pbits"],
    tuple[Int[Array, " samples pbits"], Int[Array, "reps num_gates num_samples"]],
]:
    """Perform the forward execution of sample_expval_all_param_shift_filter."""
    samples, branch_indices = sample_circuit(circuit, x, key, num_samples, sampler)
    val = jnp.mean(samples.astype(jnp.float32), axis=0)
    return val, (samples, branch_indices)


@sample_expval_all_param_shift_filter.def_bwd
def sample_expval_all_param_shift_filter_bwd(
    residuals: PyTree,
    grad_out: Float[Array, " num_pbits"],
    perturbed: CompiledBranchingPCircuit,
    circuit: CompiledBranchingPCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
    sampler: AbstractSampler | None = None,
) -> CompiledBranchingPCircuit:
    """Perform the backward execution of sample_expval_all_param_shift_filter."""
    # (num_samples, num_pbits), (reps, num_gates, num_samples)
    primal, branch_indices = residuals

    num_gates = circuit.thetas.shape[0]
    max_branches = circuit.max_branches

    if max_branches == 2:
        sig = jax.nn.sigmoid(circuit.thetas[:, 0])
        probs = jnp.stack([1 - sig, sig], axis=-1)  # (num_gates, 2)
    else:
        padded_logits = jnp.concatenate(
            [jnp.zeros((num_gates, 1)), circuit.thetas], axis=1
        )  # (num_gates, max_branches)
        probs = jax.nn.softmax(padded_logits, axis=-1)

    # For each gate g and branch k, compute the count and conditional expectation
    # branch_indices: (reps, num_gates, num_samples)
    # primal: (num_samples, num_pbits)

    # Create one-hot encoding of branch indices
    # (reps, num_gates, num_samples, max_branches)
    branch_one_hot = jax.nn.one_hot(branch_indices, max_branches)

    # Count samples per branch: (reps, num_gates, max_branches)
    branch_counts = jnp.sum(branch_one_hot, axis=2)

    # Compute conditional expectation E[O | branch=k] for each branch
    # We need to sum primal values weighted by indicator that branch k was selected

    # For each rep r, gate g, branch k:
    # E[O | branch=k] = sum_s (primal[s] * I[branch_indices[r,g,s] == k]) / count[r,g,k]

    # (reps, num_gates, max_branches, num_pbits)
    # Sum primal values where branch k was selected
    weighted_sum = jnp.einsum(
        "sp,rgsk->rgkp", primal.astype(branch_one_hot.dtype), branch_one_hot
    )

    safe_counts = jnp.where(branch_counts > 0, branch_counts, 1.0)

    # (reps, num_gates, max_branches, num_pbits)
    expval_per_branch = weighted_sum / safe_counts[:, :, :, None]

    # Set expval to 0 where count is 0
    expval_per_branch = jnp.where(
        branch_counts[:, :, :, None] > 0, expval_per_branch, 0.0
    )

    # (num_pbits,)
    expval_overall = jnp.mean(primal.astype(branch_one_hot.dtype), axis=0)

    # Shared gate parameters are reused in every repetition
    # sum over reps
    num_reps = branch_indices.shape[0]
    expval_per_branch_sum = jnp.sum(expval_per_branch, axis=0)

    # diff[g, k] = sum_r (E[O | branch_{r,g}=k] - E[O]) for branches 1..K-1.
    # (num_gates, max_branches-1, num_pbits)
    diff = expval_per_branch_sum[:, 1:, :] - num_reps * expval_overall[None, None, :]

    # grad[g, j] = p[g, j+1] * diff[g, j] @ grad_out
    # (num_gates, max_branches-1)
    grads_in = probs[:, 1:] * jnp.einsum("gjp,p->gj", diff, grad_out)

    trainable = eqx.filter(circuit, perturbed)
    grad_circuit = jax.tree.unflatten(jax.tree.structure(trainable), (grads_in,))

    return grad_circuit
```

---

## `torx/psc/simulation/gaussian.py`

**How it composes.** Exact moments for circuits made only of
`AbstractAffineGaussianGate`s (any other gate, discrete ones included, raises
`ValueError`). `propagate` starts from the given mean with zero covariance and lifts
each local `(A, b, log_var)` to the full register, `O(D**3)` per gate.
`condition(compiled, observations, *, initial_continuous, query_sites=None, jitter=0.0)`
conditions with a Cholesky solve: observed and queried sets must be disjoint static
Python collections, each observation must match its site's dimension, and a
non-positive-definite observed block raises unless `jitter > 0`. `GaussianMoments`
carries `mean`, `covariance`, `site_offsets`, `observed_sites`, and
`site_moments(site)`.

### Source: `torx/psc/simulation/gaussian.py`

```python
"""Exact Gaussian moment simulator for affine continuous hybrid circuits."""

from collections.abc import Mapping, Sequence
from typing import ClassVar, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float, Int, PyTree
from typing_extensions import Self

from .._circuit import _HybridGateType, HybridPCircuit
from ..gates import AbstractAffineGaussianGate
from .base import AbstractCompiledPCircuit, AbstractSimulator


class GaussianMoments(eqx.Module):
    """Joint Gaussian moments over continuous sites.

    `mean` and `covariance` follow the order given by `sites`. `site_offsets`
    maps each site to its `(start, stop)` slice into them.
    """

    mean: Float[Array, " continuous_dim"]
    covariance: Float[Array, "continuous_dim continuous_dim"]
    site_offsets: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    observed_sites: tuple[int, ...] = eqx.field(static=True, default=())
    _lookup: dict[int, tuple[int, int]] = eqx.field(static=True, init=False)

    def __post_init__(self):
        # precompute the site -> (start, stop) map once
        self._lookup = _offset_lookup(self.site_offsets)

    @property
    def sites(self) -> tuple[int, ...]:
        """Sites covered by this state, in `mean`/`covariance` order."""
        return tuple(site for site, _, _ in self.site_offsets)

    def site_indices(self, site: int) -> Int[Array, " d"]:
        """Return flat state-vector indices for one continuous site."""
        start, stop = self._lookup[int(site)]
        return jnp.arange(start, stop, dtype=jnp.int32)

    def site_moments(self, site: int) -> tuple[Float[Array, " d"], Float[Array, "d d"]]:
        """Return marginal mean and covariance for one continuous site."""
        idx = self.site_indices(site)
        return self.mean[idx], self.covariance[jnp.ix_(idx, idx)]

    def _indices(self, sites: Sequence[int]) -> Int[Array, " k"]:
        """Flat state-vector indices spanning `sites`, in order."""
        return _flatten_indices(self._lookup, sites)

    def _observation_vector(
        self, sites: Sequence[int], observations: Mapping[int, Array]
    ) -> Float[Array, " observed_dim"]:
        """Concatenate observation values for `sites`, in order.

        Each observation must match its site's dimension. A scalar or otherwise
        mis-shaped value is rejected rather than silently flattened/broadcast,
        which would condition on the wrong vector and return a bogus posterior.
        """
        values = []
        for site in sites:
            start, stop = self._lookup[int(site)]
            value = jnp.asarray(observations[site], dtype=self.mean.dtype).reshape(-1)
            expected = stop - start
            if value.shape[0] != expected:
                raise ValueError(
                    f"observation for site {site} has {value.shape[0]} "
                    f"value(s) but the site spans {expected} dimension(s)."
                )
            values.append(value)
        if not values:
            return jnp.zeros(0, dtype=self.mean.dtype)
        return jnp.concatenate(values)

    def _posterior_offsets(
        self, query_sites: Sequence[int]
    ) -> tuple[tuple[int, int, int], ...]:
        """Re-pack `query_sites` into contiguous `(site, start, stop)` slices."""
        lookup = self._lookup
        offsets = []
        start = 0
        for site in query_sites:
            site_start, site_stop = lookup[int(site)]
            stop = start + site_stop - site_start
            offsets.append((site, start, stop))
            start = stop
        return tuple(offsets)


class CompiledAffineGaussianPCircuit(AbstractCompiledPCircuit[HybridPCircuit]):
    """Compiled hybrid circuit for the affine Gaussian simulator."""

    gates: list[_HybridGateType]
    thetas: list[PyTree[Array]]
    site_offsets: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    reps: int

    @classmethod
    def from_pcircuit(
        cls, circuit: HybridPCircuit, thetas: list[PyTree[Array]]
    ) -> Self:
        """Compile `circuit` for exact affine Gaussian moment propagation.

        **Arguments:**

        - `circuit`: The hybrid circuit to compile.
        - `thetas`: Per-gate parameters aligned with `circuit.gates`.

        **Returns:**

        The compiled circuit.
        """
        if len(thetas) != len(circuit.gates):
            raise ValueError(
                f"expected {len(circuit.gates)} thetas aligned with "
                f"circuit.gates, got {len(thetas)}."
            )
        return cls(circuit.gates, thetas, circuit.continuous_offsets, circuit.reps)

    def to_pcircuit(self, structure: HybridPCircuit) -> HybridPCircuit:
        """Return a `HybridPCircuit` with `structure`'s gates and compiled reps."""
        return HybridPCircuit(structure.gates, self.reps)


class AffineGaussianSimulator(
    AbstractSimulator[HybridPCircuit, CompiledAffineGaussianPCircuit]
):
    """Exact moment simulator for the affine Gaussian fragment of hybrid circuits.

    Propagates the joint mean and covariance in closed form through each gate's
    affine Gaussian channel `(A, b, log_var)`, exposed via
    [`AbstractAffineGaussianGate`][torx.psc.AbstractAffineGaussianGate], and
    conditions on observed sites via the Schur complement. Only supports
    Gaussian gates.

    Dense reference implementation: `O(D^3)` per gate in the total continuous
    dimension `D`; intended for small affine-Gaussian circuits.
    """

    circuit_backend: ClassVar[Type[AbstractCompiledPCircuit]] = (
        CompiledAffineGaussianPCircuit
    )

    def build_circuit(
        self,
        circuit: HybridPCircuit,
        thetas: list[PyTree[Array]],
    ) -> CompiledAffineGaussianPCircuit:
        """Compile `circuit` for exact affine Gaussian moment propagation."""
        return CompiledAffineGaussianPCircuit.from_pcircuit(circuit, thetas)

    def propagate(
        self,
        circuit: CompiledAffineGaussianPCircuit,
        initial_continuous: Float[Array, " continuous_dim"],
    ) -> GaussianMoments:
        """Propagate joint Gaussian moments through an affine-Gaussian circuit.

        **Arguments:**

        - `circuit`: The compiled affine Gaussian circuit to execute.
        - `initial_continuous`: Flat initial continuous state.

        **Returns:**

        `GaussianMoments` over every continuous site.
        """
        site_offsets = circuit.site_offsets
        lookup = _offset_lookup(site_offsets)
        dim = site_offsets[-1][2] if site_offsets else 0
        initial = jnp.asarray(initial_continuous)
        default_dtype = _infer_default_dtype(circuit)
        with jax.numpy_dtype_promotion("standard"):
            state_dtype = jnp.result_type(initial.dtype, default_dtype)
        mean = initial.astype(state_dtype)
        covariance = jnp.zeros((dim, dim), dtype=mean.dtype)

        def body_fn(_: int, state: tuple) -> tuple:
            mean, covariance = state
            for gate, theta in zip(circuit.gates, circuit.thetas, strict=True):
                if isinstance(gate, AbstractAffineGaussianGate):
                    indices = _flatten_indices(lookup, gate.sites["continuous"])
                    A, b, log_var = gate.affine_parameters(theta)
                    mean, covariance = _apply_linear_gaussian(
                        mean,
                        covariance,
                        indices,
                        A,
                        b,
                        log_var,
                    )
                else:
                    msg = (
                        "AffineGaussianSimulator supports continuous gates "
                        "that implement AbstractAffineGaussianGate, got "
                        f"{type(gate).__name__}."
                    )
                    raise ValueError(msg)
            return mean, covariance

        mean, covariance = jax.lax.fori_loop(
            0, circuit.reps, body_fn, (mean, covariance)
        )
        return GaussianMoments(mean, _symmetrize(covariance), site_offsets)

    def condition(
        self,
        circuit: CompiledAffineGaussianPCircuit,
        observations: Mapping[int, Array] | None = None,
        *,
        initial_continuous: Float[Array, " continuous_dim"],
        query_sites: Sequence[int] | None = None,
        jitter: Float[Array, ""] | float = 0.0,
    ) -> GaussianMoments:
        """Condition final Gaussian moments on continuous-site observations.

        The circuit is propagated to a joint Gaussian over the final continuous
        state, then the queried sites are conditioned on the observed ones via
        the Schur complement. The solve uses a Cholesky factorization of the
        observed covariance plus `jitter` on the diagonal.
        Observed/query site membership must be static Python-level dict keys or
        `Sequence` entries; only observation values and `jitter` may be traced,
        so JIT/vmap over which sites are observed is unsupported by design.

        **Arguments:**

        - `circuit`: The compiled affine Gaussian circuit to execute.
        - `observations`: Mapping from each observed continuous site to its
          value.
        - `initial_continuous`: Flat initial continuous state.
        - `query_sites`: Sites to return. If `None`, all unobserved sites are
          queried.
        - `jitter`: Nonnegative diagonal regularizer for the observed
          covariance.

        **Returns:**

        `GaussianMoments` over the queried sites, with `observed_sites` set.
        """
        jitter = jnp.asarray(jitter)
        jitter = eqx.error_if(
            jitter, jnp.any(jitter < 0), "jitter must be nonnegative."
        )

        prior = self.propagate(circuit, initial_continuous)
        observations = {} if observations is None else observations
        observed_sites = tuple(sorted(int(site) for site in observations))
        if query_sites is None:
            query_sites = tuple(
                site for site in prior.sites if site not in observed_sites
            )
        else:
            query_sites = tuple(int(site) for site in query_sites)

        known = set(prior.sites)
        unknown = sorted((set(observed_sites) | set(query_sites)) - known)
        if unknown:
            raise ValueError(
                f"sites {unknown} are not continuous sites of this circuit; "
                f"available sites are {prior.sites}."
            )

        overlap = sorted(set(query_sites) & set(observed_sites))
        if overlap:
            raise ValueError(
                "query_sites and observed sites must be disjoint, "
                f"got overlap on {overlap}."
            )

        query_idx = prior._indices(query_sites)
        query_offsets = prior._posterior_offsets(query_sites)
        y = prior._observation_vector(observed_sites, observations)

        if len(query_sites) == 0 or len(observed_sites) == 0:
            mean = prior.mean[query_idx]
            covariance = prior.covariance[jnp.ix_(query_idx, query_idx)]
            return GaussianMoments(
                mean,
                _symmetrize(covariance),
                query_offsets,
                observed_sites,
            )

        observed_idx = prior._indices(observed_sites)
        mu_q = prior.mean[query_idx]
        mu_o = prior.mean[observed_idx]
        cov_qq = prior.covariance[jnp.ix_(query_idx, query_idx)]
        cov_qo = prior.covariance[jnp.ix_(query_idx, observed_idx)]
        cov_oo = prior.covariance[jnp.ix_(observed_idx, observed_idx)]

        # Standard Gaussian conditioning via the Schur complement. Cholesky-
        # factor cov_oo (+ jitter) once; `cho_solve((L, True), cov_qo.T)` gives
        # gain^T = cov_oo^{-1} cov_oq, so the posterior is
        #   mean = mu_q + gain @ (y - mu_o),  cov = cov_qq - cov_qo @ gain^T.
        eye = jnp.eye(cov_oo.shape[0], dtype=cov_oo.dtype)
        cov_oo_reg = cov_oo + jnp.asarray(jitter, dtype=cov_oo.dtype) * eye
        L = jnp.linalg.cholesky(cov_oo_reg)
        L = eqx.error_if(
            L,
            jnp.any(jnp.isnan(L)),
            "observed covariance is not positive definite, "
            "pass positive jitter to regularize",
        )
        gain_t = jsl.cho_solve((L, True), cov_qo.T)
        mean = mu_q + gain_t.T @ (y - mu_o)
        covariance = cov_qq - cov_qo @ gain_t
        return GaussianMoments(
            mean,
            _symmetrize(covariance),
            query_offsets,
            observed_sites,
        )

    def expval(
        self,
        circuit: CompiledAffineGaussianPCircuit,
        initial_continuous: Float[Array, " continuous_dim"],
        site: int = 0,
    ) -> Float[Array, " d"]:
        """Return the marginal mean of one continuous site.

        **Arguments:**

        - `circuit`: The compiled affine Gaussian circuit to execute.
        - `initial_continuous`: Flat initial continuous state.
        - `site`: The continuous site whose marginal mean to return.

        **Returns:**

        The marginal mean of `site`.
        """
        return self.propagate(circuit, initial_continuous).site_moments(site)[0]

    def expval_all(
        self,
        circuit: CompiledAffineGaussianPCircuit,
        initial_continuous: Float[Array, " continuous_dim"],
    ) -> Float[Array, " continuous_dim"]:
        """Return the joint mean over all continuous sites."""
        return self.propagate(circuit, initial_continuous).mean


def _infer_default_dtype(circuit: CompiledAffineGaussianPCircuit) -> jnp.dtype:
    """Promote affine gate dtypes to set the default state dtype.

    Uses `jax.eval_shape` to read parameter dtypes without materializing the
    `(A, b, log_var)` arrays. With no affine gates, falls back to the default
    float type, which honors `jax_enable_x64` (float64 under x64, else float32).
    """
    dtypes: list[jnp.dtype] = []
    for gate, theta in zip(circuit.gates, circuit.thetas, strict=True):
        if isinstance(gate, AbstractAffineGaussianGate):
            shapes = jax.eval_shape(gate.affine_parameters, theta)
            dtypes.extend(s.dtype for s in shapes)
    if not dtypes:
        return jnp.result_type(float)
    # Mixed fp32/fp64 gate params promote under standard rules (propagate casts
    # everything to the state dtype anyway); strict promotion would reject the
    # mix here before propagation gets the chance.
    with jax.numpy_dtype_promotion("standard"):
        return jnp.result_type(*dtypes)


def _offset_lookup(
    site_offsets: tuple[tuple[int, int, int], ...],
) -> dict[int, tuple[int, int]]:
    """Build the `site -> (start, stop)` slice map once, for shared reuse."""
    return {site: (start, stop) for site, start, stop in site_offsets}


def _flatten_indices(
    lookup: dict[int, tuple[int, int]],
    sites: Sequence[int],
) -> Int[Array, " k"]:
    """Concatenate the flat state-vector indices spanning `sites`, in order."""
    parts = []
    for site in sites:
        start, stop = lookup[int(site)]
        parts.append(jnp.arange(start, stop, dtype=jnp.int32))
    if not parts:
        return jnp.array([], dtype=jnp.int32)
    return jnp.concatenate(parts)


def _apply_linear_gaussian(
    mean: Float[Array, " continuous_dim"],
    covariance: Float[Array, "continuous_dim continuous_dim"],
    indices: Int[Array, " local_dim"],
    A: Float[Array, "local_dim local_dim"],
    b: Float[Array, " local_dim"],
    log_var: Float[Array, " local_dim"],
) -> tuple[
    Float[Array, " continuous_dim"],
    Float[Array, "continuous_dim continuous_dim"],
]:
    """Apply a local affine Gaussian channel by lifting it to dense state space."""
    # Gate params may be narrower than the promoted state dtype under strict mode.
    A = A.astype(mean.dtype)
    b = b.astype(mean.dtype)
    log_var = log_var.astype(mean.dtype)
    # dense lift: A on this gate's sites, identity elsewhere, plus bias and
    # diagonal noise
    dim = mean.shape[0]
    transform = jnp.eye(dim, dtype=mean.dtype).at[indices, :].set(0.0)
    transform = transform.at[jnp.ix_(indices, indices)].set(A)
    bias = jnp.zeros(dim, dtype=mean.dtype).at[indices].set(b)
    noise = jnp.zeros((dim, dim), dtype=mean.dtype)
    noise = noise.at[jnp.ix_(indices, indices)].set(jnp.diag(jnp.exp(log_var)))
    new_mean = transform @ mean + bias
    new_covariance = transform @ covariance @ transform.T + noise
    return new_mean, _symmetrize(new_covariance)


def _symmetrize(matrix: Float[Array, "d d"]) -> Float[Array, "d d"]:
    return 0.5 * (matrix + matrix.T)
```

---

## `torx/psc/visualization/text.py`

**How it composes.** `draw_circuit(circuit, *, output=True, unroll=False)` prints an
ASCII drawing (or returns a `TextDrawing` with `output=False`); `unroll=True` repeats
the body `reps` times instead of drawing an `xN` box. Custom gates are labelled by
`_draw_label` (default: class name) and drawn with control dots from
`_control_indices`. Only the public part of the module is shown; the grid rendering
helpers that follow `draw_circuit` are omitted.

### Source: `torx/psc/visualization/text.py`

```python
"""ASCII text rendering for probabilistic circuits.

The visual style and architecture is inspired by Qiskit's circuit visualization.
See: https://github.com/Qiskit/qiskit
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .._circuit import AbstractPCircuit, DiscretePCircuit, HybridPCircuit
from ..gates._base import (
    AbstractControlledContinuousGate,
    AbstractDiscreteGate,
    AbstractHybridGate,
    AbstractPGate,
)


@dataclass
class GateDrawInfo:
    """Drawing information for a gate."""

    label: str
    all_sites: list[int]
    control_sites: list[int]
    target_sites: list[int]
    is_hybrid: bool
    discrete_sites: list[int]
    continuous_sites: list[int]


class TextDrawing:
    """ASCII circuit drawing that can be printed."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    def __str__(self) -> str:
        return "\n".join(self._lines)

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        """Rich display for Jupyter notebooks."""
        return f"<pre>{self}</pre>"


def get_gate_label(gate: AbstractPGate) -> str:
    """Get display label for a gate."""
    label = gate._draw_label
    if label is not None:
        return label
    return gate.__class__.__name__


def _get_control_indices(gate: AbstractPGate, num_sites: int) -> list[int]:
    """Resolve _control_indices to actual indices."""
    ctrl = gate._control_indices
    if isinstance(ctrl, str):
        if ctrl == "all_but_last":
            return list(range(num_sites - 1))
        return []
    return list(ctrl)


def get_gate_draw_info(gate: AbstractPGate) -> GateDrawInfo:
    """Extract drawing info from a gate."""
    label = get_gate_label(gate)

    if isinstance(gate, AbstractDiscreteGate):
        sites = gate.sites if isinstance(gate.sites, list) else [gate.sites]

        control_indices = _get_control_indices(gate, len(sites))
        control_sites = [sites[i] for i in control_indices]
        target_sites = [s for i, s in enumerate(sites) if i not in control_indices]

        return GateDrawInfo(
            label=label,
            all_sites=sites,
            control_sites=control_sites,
            target_sites=target_sites,
            is_hybrid=False,
            discrete_sites=sites,
            continuous_sites=[],
        )

    elif isinstance(gate, AbstractHybridGate):
        gate_sites = gate.sites
        disc_sites = list(gate_sites.get("discrete", []))
        cont_sites = gate_sites.get("continuous", [])
        all_sites = disc_sites + cont_sites

        # For controlled continuous gates, discrete sites are controls
        if isinstance(gate, AbstractControlledContinuousGate):
            control_sites = disc_sites
            target_sites = cont_sites
        else:
            control_sites = []
            target_sites = cont_sites

        return GateDrawInfo(
            label=label,
            all_sites=all_sites,
            control_sites=control_sites,
            target_sites=target_sites,
            is_hybrid=True,
            discrete_sites=disc_sites,
            continuous_sites=cont_sites,
        )

    else:
        # Fallback for unknown gate types
        return GateDrawInfo(
            label=label,
            all_sites=[0],
            control_sites=[],
            target_sites=[0],
            is_hybrid=False,
            discrete_sites=[0],
            continuous_sites=[],
        )


def compute_layout(
    gate_infos: Sequence[GateDrawInfo],
) -> list[int]:
    """Assign each gate to a column, allowing parallel placement.

    Gates that don't share any sites can be placed in the same column.
    For controlled gates that span multiple wires, all intermediate wires
    are also blocked. Returns a list of column indices, one per gate.
    """
    site_frontier: dict[int, int] = {}
    gate_columns: list[int] = []

    for info in gate_infos:
        sites = info.all_sites

        if not sites:
            gate_columns.append(0)
            continue

        has_controls = len(info.control_sites) > 0 and len(info.target_sites) > 0
        spans_multiple = len(info.target_sites) > 1
        if has_controls or spans_multiple:
            min_wire = min(sites)
            max_wire = max(sites)
            blocked_sites = list(range(min_wire, max_wire + 1))
        else:
            blocked_sites = sites

        # Gate goes in the max of all blocked sites' frontiers
        col = max(site_frontier.get(s, 0) for s in blocked_sites)

        gate_columns.append(col)

        # Advance frontier for all blocked sites
        for s in blocked_sites:
            site_frontier[s] = col + 1

    return gate_columns


def draw_circuit(
    circuit: AbstractPCircuit, *, output: bool = True, unroll: bool = False
) -> TextDrawing | None:
    """Create ASCII representation of a circuit.

    **Arguments:**

    - `circuit`: The circuit to draw.
    - `output`: If True (default), print the circuit and return None.
        If False, return a TextDrawing object.
    - `unroll`: If True, repeat the circuit content `reps` times instead of
        showing a box with "xN". Default is False.

    **Returns:**

    A TextDrawing object if output=False, otherwise None.
    """
    if isinstance(circuit, DiscretePCircuit):
        drawing = _draw_discrete_circuit(circuit, unroll=unroll)
    elif isinstance(circuit, HybridPCircuit):
        drawing = _draw_hybrid_circuit(circuit, unroll=unroll)
    else:
        raise TypeError(f"Unsupported circuit type: {type(circuit)}")

    if output:
        print(drawing)
        return None
    return drawing

# ... ASCII grid rendering helpers omitted ...
```

