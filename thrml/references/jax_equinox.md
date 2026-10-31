# JAX & Equinox playbook for THRML

The rules that keep THRML code correct under `jit` / `vmap` / `grad`, distilled from the
patterns the library and its tests already follow. THRML is built on JAX and Equinox: its
models, programs, samplers, factors, and observers are `equinox.Module`s, and its state is
JAX arrays. These rules are the THRML-specific slice of general JAX/Equinox numerical practice —
load `SKILL.md` first; this file is the depth-on-demand companion with a checklist at the end.

Each rule is **Do / Don't / Why**, with a pointer into `references/source.md` (the annotated
library) for the line that proves it.

---

## 1. Structure is static; arrays are dynamic

- **Do:** Treat nodes, `Block`s, the graph, `SamplingSchedule`, `IsingEBM`, the program, and
  `IsingTrainingSpec` as static structure fixed at trace time. Let only `biases`, `weights`,
  `beta`, the sampling state, and PRNG keys flow through transforms.
- **Don't:** Build the working representation out of integer index arrays, or try to trace the
  structure. The "carry blocks of nodes" idiom *is* this partition — see `SKILL.md`.
- **Why:** JAX traces array leaves and bakes everything else in as a compile-time constant.
  Mixing the two causes tracer-leak errors or silent recompiles. `SamplingSchedule` is a
  hashable `dataclass`; programs/EBMs are `equinox.Module`s — all static (source: `block_sampling.py`
  `SamplingSchedule.__hash__`; `models/ising.py` `IsingEBM` fields `nodes`/`edges` are object lists).

## 2. Cross the `jit` boundary with `eqx.filter_*`

- **Do:** Wrap functions that take a THRML object as an argument with `eqx.filter_jit` /
  `eqx.filter_vmap` / `eqx.filter_grad`. They partition non-array leaves (the `SpinNode`
  objects, blocks, schedule) off as static automatically, so you can pass a whole `IsingEBM`,
  program, or spec straight in. `hinton_init` is exactly this: `@eqx.filter_jit` over a
  function whose argument is an `IsingEBM` (source: `models/ising.py` `hinton_init`).
- **Don't:** Force plain `jax.jit` onto a function that receives an `IsingEBM` as a traced
  argument — its leaves include `SpinNode`s, which JAX cannot trace.
- **Why:** `filter_jit` removes the friction entirely. Plain `jax.jit` is fine only when you
  have already reduced the call to array leaves — e.g. a training step that closes over
  `nodes`/`edges`/blocks and threads `biases`/`weights` through an optimizer, rebuilding the
  model and spec inside (construction is free at trace time).

## 3. Thread PRNG keys

- **Do:** Construct keys with the typed `jax.random.key(seed)`, `jax.random.split(key, n)`
  before branching into `n` chains/blocks, and `vmap` over exactly `n` split keys.
- **Don't:** Reuse a key, create keys inside traced code from Python ints, or rely on global RNG.
- **Why:** Determinism under `jit`/`vmap`/`scan` depends on explicit key flow. THRML splits a
  key per free block every Gibbs iteration and per phase in training (source: `block_sampling.py`
  `sample_blocks` / `_run_blocks`; `models/ising.py` `estimate_kl_grad`).

## 4. Never differentiate through the sampler

- **Do:** Train via the contrastive estimator. `estimate_kl_grad` returns `∂KL/∂θ` as a
  difference of *sampled* moments — `ΔW = −β(⟨sᵢsⱼ⟩₊ − ⟨sᵢsⱼ⟩₋)`, `Δb = −β(⟨sᵢ⟩₊ − ⟨sᵢ⟩₋)` —
  computed by sampling, not autodiff (source: `models/ising.py` `estimate_kl_grad`).
- **Don't:** Call `jax.grad(sample_states)` or otherwise backprop through the Gibbs chain.
- **Why:** The chain draws discrete spins; it has no useful gradient. The only legitimate use
  of autodiff here is on a tiny model's *closed-form* energy, to check the estimator (§7).

## 5. Update parameters on an immutable model

- **Do:** Swap array fields with `eqx.tree_at(lambda m: (m.weights, m.biases), model, (w, b))`,
  or rebuild `IsingEBM(...)`. For optimizers, keep `(weights, biases)` as the optax params and
  reconstruct the model each step; the real training loop scans minibatches with `lax.scan`
  and rebuilds via `eqx.tree_at` inside the body (source: `tests/test_train_mnist.py`).
- **Don't:** Mutate a field in place — `equinox.Module`s are frozen.
- **Why:** Immutability is what makes the model a clean pytree. `eqx.tree_at` updates only the
  array leaves and leaves the static `nodes`/`edges` untouched, so you never thread node lists
  through the optimizer. Wrap `optax.update` in `with jax.numpy_dtype_promotion("standard"):`
  to avoid JAX's strict-promotion error on mixed dtypes.

## 6. Custom samplers and factors run inside the trace

- **Do:** Dispatch on interaction *type* with `isinstance` (compile-time, static). Combine
  contributions across a block's interactions, masking padded appearances by multiplying with
  `active_flags` and summing `axis=-1`. Extend a built-in sampler by overriding
  `compute_parameters`, handling your interaction, and delegating the rest to `super()`
  (source: `models/discrete_ebm.py` `SpinGibbsConditional.compute_parameters`).
- **Don't:** Branch with a Python `if` on an array value (use `jnp.where`/`lax.select`/
  `lax.cond`), drop "inactive" terms with control flow, or hand-roll a duplicate of a sampler
  you could subclass.
- **Why:** Everything inside `sample`/`compute_parameters` is traced; data-dependent Python
  control flow either fails or silently specializes the trace. `active_flags` (shape
  `[n_nodes, max_appearances]`) is how heterogeneous graphs vectorize on the GPU.

## 7. Stable numerics and dtypes

- **Do:** Prefer stability-aware primitives when you write parameter/energy math by hand:

  | Prefer | Instead of |
  | --- | --- |
  | `jax.nn.log_sigmoid(x)` | `jnp.log(jax.nn.sigmoid(x))` |
  | `jax.nn.softplus(x)` | `jnp.log1p(jnp.exp(x))` |
  | `jax.nn.log_softmax(x)` | `jnp.log(jax.nn.softmax(x))` |
  | `jax.scipy.special.logsumexp(x)` | `jnp.log(jnp.sum(jnp.exp(x)))` |
  | `jax.random.categorical(key, logits)` | softmax → `jax.random.bernoulli`/`choice` |

- **Do:** Keep spin **state** as `bool` and categorical state as `uint8`; convert spins to
  `{−1,+1}` with `2 * s.astype(jnp.int8) - 1` before arithmetic (source: `models/discrete_ebm.py`
  `_spin_product`). Parameters (`biases`/`weights`/`beta`) are floats. Enable
  `jax.config.update("jax_enable_x64", True)` only if a model is large or ill-conditioned
  enough that float32 moment estimates drift.
- **Don't:** Compose `exp`/`log`/`sqrt` for cancellation-prone quantities, softmax-then-sample,
  or feed float `state` arrays where `bool`/`uint8` is expected (it raises a `RuntimeError`).
- **Why:** The built-ins (`BernoulliConditional` uses `jax.nn.sigmoid(2γ)`; `SoftmaxConditional`
  uses `jax.random.categorical` on logits) are already stable — match them.

---

## THRML pre-flight checklist

Structure / transforms
- [ ] Only arrays (`biases`, `weights`, `beta`, state, keys) cross `jit`/`vmap`/`grad`; nodes,
      blocks, schedule, program, spec stay static.
- [ ] Functions taking a THRML object use `eqx.filter_jit`/`filter_vmap`/`filter_grad`
      (or close over the object and pass arrays only).
- [ ] Sub-structures derived by set operations on node objects, never index-array slicing.

PRNG
- [ ] Keys built with `jax.random.key`, split before branching, never reused.
- [ ] `vmap` over N chains splits exactly N keys, mapped alongside per-chain init arrays.

Training
- [ ] Gradients come from `estimate_kl_grad` (sampled moments), never `jax.grad` of the chain.
- [ ] There are latent nodes; positive init is one array per positive block, never `[]`.
- [ ] Optimizer step uses `eqx.tree_at` (or rebuild) and `jax.numpy_dtype_promotion("standard")`;
      sign descends KL (optax `apply_updates` adds; raw SGD subtracts the gradient).

Custom samplers / factors
- [ ] `isinstance` only on interaction type; no Python `if` on array values.
- [ ] Padded appearances masked via `active_flags` and summed, never dropped by control flow.
- [ ] Stable primitives used; categoricals sampled from logits; spins converted `2s−1`.
- [ ] Undirected coupling emits two `InteractionGroup`s (one per endpoint as head).

Validation
- [ ] Small graph checked by exact `2ⁿ` enumeration of `p ∝ exp(−E)` (rel. error ≲ 0.02).
- [ ] Empirical moments match `estimate_moments`; MC `estimate_kl_grad` matches the exact
      `eqx.filter_value_and_grad` of the closed-form KL on a tiny model (≲ 0.01).
- [ ] `vmap` output matches a Python-loop run over the same keys.
- [ ] Spin state `bool`, categorical state `uint8`.