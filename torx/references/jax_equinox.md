# JAX / Equinox playbook for Torx

A checklist for writing or reviewing Torx code. Each item names the failure it
prevents and, where useful, points to the section of `source.md` that shows why
(each module has a heading of the form `` ## `torx/dfg.py` ``). Facts were
checked against torx 0.0.2.

## Static structure, traced arrays

- [ ] Gates, circuits, `Site`s and `DFG`s are Equinox modules whose structure (sites,
      dims, port specs, wiring, `reps`) is fixed at trace time. Only thetas, initial
      states, inputs and keys are traced. (source.md: `torx/psc/_circuit.py`: the circuit
      generates one `Site` per gate per repetition at construction.)
- [ ] Custom factors declare `input_ports` and `output_spec` as
      `eqx.field(static=True)`; custom gates store configuration switches (a Metropolis
      flag) the same way. *Prevents non-hashable-static errors and retracing.*
- [ ] Use `eqx.filter_jit` / `eqx.filter_vmap` / `eqx.filter_value_and_grad` when a
      module is an argument. With plain `jax.jit`, close over the module and pass arrays
      only. *A compiled `BranchingSimulator` circuit passed to `jax.jit` fails with
      `TracerBoolConversionError`: `max_branches`, `reps`, `num_pdits` are plain int
      fields* (source.md: `torx/psc/simulation/sampled.py`, `CompiledBranchingPCircuit`).
- [ ] Scalars that change between runs (temperature, step size, rate) are passed as
      traced arrays (`jnp.asarray(beta, jnp.float32)`), not Python floats baked into a
      closure. *Prevents one recompile per value.*
- [ ] Integer index arrays stored on a trainable module (`param_map`, readout sites) are
      left out of the optimizer with `eqx.filter(model, eqx.is_inexact_array)`.

## Strict-class rules (ihoop)

`AbstractFactor` and the simulators derive from `ihoop.eqx.AbstractStrictModule`
(source.md: "Module map and import surface"). Checked when your class is
defined:

- [ ] Never subclass a concrete Torx class (`PNOT`, `AffineGaussianGate`,
      `TiledFactor`, `DFG`, a simulator). Subclass the `Abstract*` base.
      *Prevents `TypeError: Cannot inherit from concrete (final) class`.*
- [ ] Implement abstract methods and `__init__` only; never override an inherited
      concrete method (`AbstractAffineGaussianGate.sample`, `sample_multiple`,
      `AbstractReferenceFactor.sample_with_references`,
      `AbstractHasExplicitOutputDistribution.log_probability`).
      *Prevents `TypeError: Cannot override concrete method`.*
- [ ] A fully implemented class must not be named `Abstract...`; one with abstract
      members left must be.
- [ ] Declare every abstract field as a real field: `input_ports` / `output_spec` on a
      custom factor; `input_states` / `output_states` on a matrix factor; `sites` and
      `dims` on a custom gate, since `AbstractContinuousGate` declares no fields and no
      constructor (source.md: `torx/psc/gates/_base.py`). Give `sites` a converter to the
      `HybridSites` dict, as the built-in continuous gates do
      (`_continuous_sites_converter` in source.md: `torx/psc/gates/_continuous.py`).
- [ ] `AbstractMatrixFactor` has no `sample`: implement it (categorical on the row,
      then `get_nth_output_state`).

## Parameters

- [ ] One theta per gate, in `circuit.gates` order; `assert len(thetas) == len(circuit.gates)`
      (`build_circuit` raises otherwise).
- [ ] Two-branch gates get `jnp.array([logit(p)])` (shape `(1,)`); `PditCycle` gets
      shape `(2,)`; `PISING` gets `[J, h1, h2, beta, dt]`; `Mix` gets a scalar; affine and
      mixture gates get dicts (tables in `source.md`, gate sections).
- [ ] Clip probabilities before `logit` when they come from a model
      (`jnp.clip(p, 1e-6, 1 - 1e-6)`). *Prevents `+-inf` logits and NaN gradients.*
- [ ] Compile once, then `eqx.tree_at(lambda c: c.thetas, compiled, new)`. For
      `BranchingSimulator` the replacement has shape `(num_gates, max_branches - 1)` and
      keeps `-inf` padding for gates with fewer branches (`from_pcircuit` in
      source.md: `torx/psc/simulation/sampled.py`).
- [ ] Learnable sharing goes through an index map built inside the loss
      (`shared[param_map]`). Repeated entries of a thetas list you differentiate are
      separate leaves with separate gradients.
- [ ] `log_var = -inf` (exact deterministic channel) is fine for propagation and
      conditioning, but clamp it to a large negative finite value before differentiating.

## Keys

- [ ] Typed keys: `jax.random.key(seed)`.
- [ ] Split before branching; never reuse a key. One key per chain, per sample, per
      time step (`jax.random.split(key, num_steps)` fed to `lax.scan`).
- [ ] `jax.random.fold_in(base_key, i)` for per-case keys (per regime, per start state).
- [ ] Identities for same-key tests: `sample_multiple` splits `n_samples` ways and
      vmaps `sample`; `TiledFactor` = vmap over `split(key, n_tiles)`; `ChainFactor` =
      scan over `split(key, n_steps)`; `DFG.sample` splits once per site, indexed by
      site position (source.md: `torx/factor.py`, `torx/composite_factors.py`,
      `torx/dfg.py`).
- [ ] Independent estimates (training batch versus validation, `sample` versus
      `expval_all`) get independent keys, so their noise is not shared.

## What can be differentiated

| Call | Gradient |
|------|----------|
| `StateVectorSimulator.density` / `expval` / `expval_all` | exact autodiff |
| `BranchingSimulator.expval` / `expval_all` | custom VJP on the compiled `thetas`, per `diff_method` |
| `BranchingSimulator.sample` | none (integer output) |
| `AffineGaussianSimulator.propagate` / `condition` | exact autodiff, finite `log_var` only |
| `circuit.sample` | not documented as differentiable; do not rely on it |

- [ ] `diff_method` limits (source.md: `torx/psc/simulation/sampled.py`):
      `"param_shift_inf"` needs `reps == 1` and costs `num_gates * max_branches` extra
      estimates; `"param_shift_single"` needs `reps == 1` and two-branch gates;
      `"param_shift_filter"` works for any `reps` and K from the primal samples alone.
      *Prevents `NotImplementedError` on the backward pass of repeated circuits.*
- [ ] Discrete objectives on whole samples: REINFORCE with `stop_gradient` on both the
      sampling logits and the centered reward; multiply by `M / (M - 1)` when the
      baseline is the same batch's mean.
- [ ] optax: `optimizer.update` negates and `apply_updates` adds, so pass the gradient of
      the loss being minimized. Plain SGD subtracts the raw gradient.

## Control flow inside custom factors and gates

- [ ] Branch on types and static fields with Python `if`; branch on array values with
      `jnp.where`, `lax.select`, `lax.cond`.
- [ ] Use stable primitives: `jax.nn.log_sigmoid`, `jax.nn.softplus`,
      `jax.scipy.special.logsumexp`, `jax.random.categorical` straight from logits.
- [ ] Derive drifts with `jax.grad` of the same energy that defines the target.
- [ ] Return `(output, None)` when `return_aux=True`.
- [ ] Output dtype matches `output_spec` (`int32` for pbits, spins and pdits, `float32`
      for pmodes); a `ChainFactor` feedback output must match its port's spec exactly or
      construction raises `ValueError`.
- [ ] Draw randomness through `info` (an `AbstractSampler`) when it is not `None`, so a
      caller can inject a provider with `DFGInfo(default_site_info=...)`.

## Loops, batching and compile time

- [ ] Simulator `reps` for identical repeated layers: `StateVectorSimulator` and
      `AffineGaussianSimulator` `fori_loop` over it and `BranchingSimulator` scans it.
- [ ] Direct `circuit.sample` **unrolls** `reps` into `reps * len(gates)` DFG sites:
      compile time grows linearly (19 s at `reps=200` for one `Diffuse`). For long chains
      `jax.lax.scan` over `sample` of a `reps=1` circuit (0.4 s for the same 200 steps).
      `ChainFactor` is already a scan.
- [ ] `jax.vmap` over chains or trajectories; `jit` the outer function once.
- [ ] Jit simulator calls with `eqx.filter_jit`: eager `StateVectorSimulator.density`
      re-traces and recompiles its `fori_loop` on every call.
- [ ] On CPU, an 8-site `StateVectorSimulator` circuit took 50 s to over 4 minutes to
      compile (7 sites 1.5 s; 9 to 12 sites under 0.5 s). Pad with an idle ninth site,
      `PNOT(8)` with theta `jnp.array([-jnp.inf])` (an exact identity).
- [ ] `TiledFactor(batch_size=b)` switches to chunked `lax.map` when vmapping every tile
      at once would exhaust memory.

## Testing

- [ ] Exact reference first: `StateVectorSimulator.density` or brute-force enumeration
      for up to roughly `2**16` states; `scipy.linalg.expm(Q t)` for CTMCs;
      `AffineGaussianSimulator` or a dense Schur complement for Gaussians; quadrature for
      1-D continuous targets. Worked examples 1 to 3 in `source.md` show each.
- [ ] Kernel sanity: gate matrices are column-stochastic; matrix-factor rows normalize.
- [ ] Separate Monte Carlo error (sample versus exact mean of the same discretized
      scheme) from discretization bias (that mean versus the continuous answer).
- [ ] Tolerances are multiples of the standard error: `0.5 / sqrt(N)` for pbit means,
      `5 / sqrt(N)` for KS distances, per-moment SEs for Gaussian moments.
- [ ] Gradients: parameter shift versus `jax.grad` of the exact loss on a tiny circuit
      (worked example 1), or versus a central finite difference on the same batch and key
      (about 30% relative tolerance with a few hundred samples).
- [ ] Composites: same-key equality against manual `vmap` and manual loop (worked
      example 4).
- [ ] Assert, do not just print, so drift fails the run.
