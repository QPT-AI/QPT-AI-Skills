# THRML — Annotated API

This is the THRML library API (test suite removed), each module preceded by a short note on what it is and how it composes with the rest. THRML is layered: the **Ising machine** sits on top and fills in everything beneath it, so for the common case you only touch `IsingEBM` + colored `Block`s + `sample_states`/`estimate_kl_grad`. Read the worked examples first — they show the whole stack in motion — then drop into a module's source for an exact signature, default, or update rule.

## Navigation

**Worked examples (start here):** Ising sampling · training an Ising / Boltzmann machine · Potts (categorical) · custom continuous + heterogeneous graphs.

**Modules, in reading order:**

- `thrml/thrml/pgm.py` — nodes (`SpinNode`, `CategoricalNode`, custom `AbstractNode`)
- `thrml/thrml/block_management.py` — `Block` (parallel update unit) and global/block state plumbing
- `thrml/thrml/interaction.py` — `InteractionGroup`, the primitive directed dependency
- `thrml/thrml/conditional_samplers.py` — conditional update rules (`SpinGibbsConditional`, softmax, custom)
- `thrml/thrml/factor.py` — `AbstractFactor`/`WeightedFactor` and `FactorSamplingProgram`
- `thrml/thrml/block_sampling.py` — `BlockGibbsSpec`, `SamplingSchedule`, `sample_states` run loop
- `thrml/thrml/observers.py` — `StateObserver`, `MomentAccumulatorObserver`
- `thrml/thrml/models/ebm.py` — EBM base classes
- `thrml/thrml/models/discrete_ebm.py` — `SpinEBMFactor`, `CategoricalEBMFactor` and their samplers
- `thrml/thrml/models/ising.py` — **the Ising machine + training** (`IsingEBM`, `IsingTrainingSpec`, `estimate_kl_grad`)
- `thrml/thrml/models/__init__.py` — `thrml.models` public surface
- `thrml/thrml/__init__.py` — `thrml` public surface

---

# Worked examples

These are the canonical end-to-end programs. Prefer adapting one of these to writing from scratch. Examples 1–2 are the center of gravity for THRML; 3–4 show the categorical and custom-variable paths.

## Example 1 — Sample from an Ising model

The full sampling stack with only built-in objects. `vmap` runs parallel chains; `jit` compiles.

```python
import jax, jax.numpy as jnp, networkx as nx
from thrml import Block, SpinNode, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# graph: one SpinNode per vertex of a grid
G = nx.grid_graph(dim=(16, 16), periodic=False)
nx.relabel_nodes(G, {c: SpinNode() for c in G.nodes}, copy=False)
nodes, edges = list(G.nodes), list(G.edges)

# color so neighbors never share a block (grid is bipartite -> 2 colors)
bicol = nx.bipartite.color(G)
free_blocks = [Block([n for n, c in bicol.items() if c == k]) for k in (0, 1)]

# parameterize E(s) = -beta * (sum b_i s_i + sum J_ij s_i s_j)
key = jax.random.key(0); key, kb, kw = jax.random.split(key, 3)
model = IsingEBM(nodes, edges,
                 jax.random.normal(kb, (len(nodes),)),   # biases b
                 jax.random.normal(kw, (len(edges),)),   # weights J
                 jnp.array(1.0))                          # beta

program  = IsingSamplingProgram(model, free_blocks, [])   # no clamped blocks
schedule = SamplingSchedule(n_warmup=100, n_samples=50, steps_per_sample=5)

n_chains = 64; key, ki = jax.random.split(key)
init_free = hinton_init(ki, model, free_blocks, (n_chains,))  # list, one (n_chains, block_len) array per block

run = jax.jit(jax.vmap(lambda init, k: sample_states(k, program, schedule, init, [], [Block(nodes)])))
samples = run(init_free, jax.random.split(key, n_chains))
# samples[0]: (n_chains, n_samples, n_nodes); read spins as True->+1, False->-1
```

## Example 2 — Train an Ising model (Boltzmann machine)

Training minimizes `KL(data || model)` via a contrast of two phases: **positive** (data clamped,
latents sampled) minus **negative** (everything sampled). Two structural facts: you need latent
nodes, and the two phases color differently (negative colors the whole graph; positive reuses that
coloring with the data nodes removed).

```python
import jax, numpy as np, networkx as nx
from thrml import Block, SamplingSchedule
from thrml.models import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init
# assume: `graph` with a SpinNode per vertex, `nodes`, `edges`, `model = IsingEBM(...)`,
#         and `data` of shape (batch, n_data)

# 1) visible/latent split
n_data = 100
data_inds  = np.random.choice(len(nodes), n_data, replace=False)
data_nodes = [nodes[i] for i in data_inds]; data_set = set(data_nodes)

# 2) color the FULL graph -> negative (free) blocks
coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
n_colors = max(coloring.values()) + 1
free_blocks = [Block([n for n, c in coloring.items() if c == k]) for k in range(n_colors)]

# 3) same coloring minus data nodes -> positive (clamped) blocks
clamped_blocks = [Block([n for n in blk.nodes if n not in data_set]) for blk in free_blocks]
clamped_blocks = [b for b in clamped_blocks if len(b.nodes) > 0]

schedule = SamplingSchedule(5, 100, 5)

# IsingTrainingSpec(ebm, data_blocks, conditioning_blocks,
#                   positive_sampling_blocks, negative_sampling_blocks, sched_pos, sched_neg)
training_spec = IsingTrainingSpec(
    model, [Block(data_nodes)], [], clamped_blocks, free_blocks, schedule, schedule)

batch_size = data.shape[0]
key = jax.random.key(0); key, k1, k2 = jax.random.split(key, 3)
init_free    = hinton_init(k1, model, free_blocks,    (batch_size,))      # each (batch, block_len)
init_clamped = hinton_init(k2, model, clamped_blocks, (1, batch_size))    # each (1, batch, block_len)

# estimate_kl_grad(key, spec, bias_nodes, weight_edges, data, conditioning_values,
#                  init_positive, init_negative)   ->  grad_w, grad_b, pos_moments, neg_moments
grad_w, grad_b, _, _ = estimate_kl_grad(
    key, training_spec, nodes, edges,
    [data], [],
    init_clamped,   # positive init — MUST NOT be []
    init_free)      # negative init

# one SGD step (use any optimizer); IsingEBM is immutable, so rebuild it
lr = 0.01
model = IsingEBM(nodes, edges, model.biases - lr*grad_b, model.weights - lr*grad_w, model.beta)
```

## Example 3 — Potts model (categorical)

Same skeleton with `CategoricalNode`s and an explicit pairwise factor. Identity coupling makes
neighbors agree, producing domain formation.

```python
import jax, jax.numpy as jnp, networkx as nx
from thrml import Block, CategoricalNode, BlockGibbsSpec, SamplingSchedule, sample_states, FactorSamplingProgram
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional

G = nx.grid_graph(dim=(20, 20), periodic=False)
nx.relabel_nodes(G, {c: CategoricalNode() for c in G.nodes}, copy=False)
bicol = nx.bipartite.color(G)
color0 = [n for n, c in bicol.items() if c == 0]
color1 = [n for n, c in bicol.items() if c == 1]
u, v = map(list, zip(*G.edges()))

n_cats, beta = 5, 1.0
weights = beta * jnp.broadcast_to(jnp.eye(n_cats)[None], (len(u), n_cats, n_cats))  # (n_edges, K, K)
coupling = CategoricalEBMFactor([Block(u), Block(v)], weights)

spec = BlockGibbsSpec([Block(color0), Block(color1)], [])
sampler = CategoricalGibbsConditional(n_cats)
prog = FactorSamplingProgram(spec, [sampler, sampler], [coupling], [])

schedule = SamplingSchedule(0, 100, 5)
key = jax.random.key(4242); n_chains = 100
init = []
for blk in spec.free_blocks:
    key, sub = jax.random.split(key)
    init.append(jax.random.randint(sub, (n_chains, len(blk.nodes)), 0, n_cats, dtype=jnp.uint8))
run = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], [Block(G.nodes)])))
samples = run(init, jax.random.split(key, n_chains))
```

## Example 4 — Custom continuous variables + heterogeneous graphs

When the variable is not spin/categorical, you define a node, the interactions (`eqx.Module`s),
factors that emit them, and a sampler implementing the conditional. A Gaussian conditional from a
linear term and a quadratic (inverse-variance) term:

```python
import jax, jax.numpy as jnp, equinox as eqx
from jaxtyping import Array
from thrml import (AbstractNode, Block, BlockGibbsSpec, FactorSamplingProgram,
                   AbstractFactor, InteractionGroup, AbstractConditionalSampler)

class ContinuousNode(AbstractNode): pass
class LinearInteraction(eqx.Module):    weights: Array
class QuadraticInteraction(eqx.Module): inverse_weights: Array

class QuadraticFactor(AbstractFactor):           # w * x^2  (no tail nodes)
    inverse_weights: Array
    def __init__(self, inv_w, block): super().__init__([block]); self.inverse_weights = inv_w
    def to_interaction_groups(self):
        return [InteractionGroup(QuadraticInteraction(self.inverse_weights), self.node_groups[0], [])]

class CouplingFactor(AbstractFactor):            # w * x_i * x_j  (each endpoint a head)
    weights: Array
    def __init__(self, w, blocks): super().__init__(list(blocks)); self.weights = w
    def to_interaction_groups(self):
        return [InteractionGroup(LinearInteraction(self.weights), self.node_groups[0], [self.node_groups[1]]),
                InteractionGroup(LinearInteraction(self.weights), self.node_groups[1], [self.node_groups[0]])]

class GaussianSampler(AbstractConditionalSampler):
    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):
        bias = jnp.zeros(output_sd.shape, output_sd.dtype)
        var  = jnp.zeros(output_sd.shape, output_sd.dtype)
        for active, interaction, state in zip(active_flags, interactions, states):
            if isinstance(interaction, LinearInteraction):
                prod = jnp.prod(jnp.stack(state, -1), -1) if state else jnp.array(1.0)
                bias -= jnp.sum(interaction.weights * active * prod, axis=-1)
            if isinstance(interaction, QuadraticInteraction):
                var = (active * interaction.inverse_weights)[..., 0]
        return jnp.sqrt(var) * jax.random.normal(key, output_sd.shape) + bias * var, sampler_state
    def init(self): return None

# custom node types need an explicit shape/dtype; mixed-type colors use SuperBlocks (tuples)
node_sd = {ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32)}
spec = BlockGibbsSpec(free_blocks, [], node_sd)              # free_blocks may be tuples for mixed types
prog = FactorSamplingProgram(spec, [GaussianSampler(), GaussianSampler()], factors, [])
```

For a graph that mixes node types within one color, make one block per type and group them as a
SuperBlock so they share input state but run their own samplers:
`BlockGibbsSpec([(Block(spins0), Block(cont0)), (Block(spins1), Block(cont1))], [], {SpinNode: sd, ContinuousNode: sd})`,
with one sampler per flattened block, in spec order.

---

# Annotated modules

## `thrml/thrml/pgm.py`

**Nodes — the atoms.** Every random variable is a node subclassing `AbstractNode`. `SpinNode` is binary `{-1,+1}` stored as `bool_` (`True`=+1); `CategoricalNode` is discrete `{0..K-1}` stored as `uint8`. A custom variable (e.g. continuous) is an empty `AbstractNode` subclass that you later give a shape/dtype in the spec. Nodes carry no state themselves — they are identity handles that the graph and blocks refer to.

```python
import abc
from dataclasses import dataclass, is_dataclass
from typing import ClassVar

import jax
from jax import numpy as jnp


class _CounterMeta(abc.ABCMeta):
    """Metaclass that automatically calls __post_init__ and provides unique ordering.

    Used internally by THRML for node identification and ordering.
    """

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if not is_dataclass(cls):
            post_init = getattr(instance, "__post_init__", None)
            if callable(post_init):
                post_init()
        return instance

    def __lt__(cls, other):
        # todo: make sure this is sufficient to distinguish and be unique for JAX
        if not isinstance(other, type):
            raise NotImplementedError
        return (cls.__module__, cls.__qualname__) < (other.__module__, other.__qualname__)


class _UniqueID(metaclass=_CounterMeta):
    """
    This is a way of ensuring that there is a unique identifier
    for subclasses, without them being required to call super().__init__().
    """

    __slots__ = ("_hash",)
    _counter: ClassVar[int] = 0
    _hash: int

    def __post_init__(self):
        self._hash = _UniqueID._counter
        _UniqueID._counter += 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _UniqueID):
            return False
        return self._hash == other._hash

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other):
        if isinstance(other, _UniqueID):
            return self._hash < other._hash
        raise RuntimeError("less than only defined between _UniqueIDs")


@dataclass(eq=False)
class AbstractNode(_UniqueID):
    """
    A node in a PGM.

    Every node used in a PGM must inherit from this class. When compiling a program, each node is assigned a
    shape and datatype that are used to organize the state of the sampling program in a jax-friendly way.
    """

    def __new__(cls, *args, **kwargs):
        if cls is AbstractNode:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)


class SpinNode(AbstractNode):
    """A node that represents a random variable that takes on a state in {-1, 1}."""

    pass


class CategoricalNode(AbstractNode):
    """A node that represents a random variable that may take on any one of K possible discrete states,
    represented by a positive integer in (0, K]."""

    pass


DEFAULT_NODE_SHAPE_DTYPES = {
    SpinNode: jax.ShapeDtypeStruct(tuple(), dtype=jnp.bool_),
    CategoricalNode: jax.ShapeDtypeStruct(tuple(), dtype=jnp.uint8),
}
```

## `thrml/thrml/block_management.py`

**Blocks and global<->block state.** A `Block` is a list of nodes *all of the same type*; it is the atomic unit of parallel update, so a block may never contain two interacting (neighboring) nodes. `BlockSpec` assigns every node a location in a packed global state (a few JAX arrays). The `*_global_state` / `block_state_*` helpers convert between the per-block view you work with and that packed representation. You rarely call these directly for Ising models — the wrappers do — but they explain the shapes you see inside samplers.

```python
from typing import (
    Generic,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PyTree, Shaped

from .pgm import AbstractNode

_Node = TypeVar("_Node", bound=AbstractNode)
# hashable description of general pytrees
_PyTreeStruct: TypeAlias = tuple[
    PyTree,
    tuple[jax.ShapeDtypeStruct, ...],
]
_GlobalState: TypeAlias = PyTree[Shaped[Array, "nodes_global ?*state"], "_GlobalState"]
_State = PyTree[Shaped[Array, "nodes ?*state"], "State"]
_Node_SD = Mapping[Type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]


class Block(Generic[_Node]):
    """
    A Block is the basic unit through which Gibbs sampling can operate.

    Each block represents a collection of nodes that can efficiently be sampled
    simultaneously in a JAX-friendly SIMD manner. In THRML, this means that the nodes must all be of the same type.

    **Attributes:**

    - `nodes`: the tuple of nodes that this block contains
    """

    nodes: tuple[_Node, ...]

    def __init__(self, nodes: Sequence[_Node]) -> None:
        nodes_tuple = tuple(nodes)
        if nodes_tuple:
            first_type = type(nodes_tuple[0])
            if {type(node) for node in nodes_tuple} != {first_type}:
                raise ValueError("All nodes in a block must be of the same type")
        self.nodes = nodes_tuple

    @property
    def node_type(self) -> Type[_Node]:
        if not self.nodes:
            raise ValueError(
                "Block is empty and doesn't have a node type. Most methods in thrml do not support empty blocks."
            )
        return type(self.nodes[0])

    def __getitem__(self, index: int) -> _Node:
        return self.nodes[index]

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[_Node]:
        return iter(self.nodes)

    def __contains__(self, item) -> bool:
        return item in self.nodes

    def __add__(self, other):
        if isinstance(other, Block):
            if self.nodes and other.nodes:
                if type(self.nodes[0]) is not type(other.nodes[0]):
                    raise ValueError("Cannot add Blocks of different node types")
            return Block(self.nodes + other.nodes)
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nodes={self.nodes!r})"


def _hash_pytree(x: PyTree[jax.ShapeDtypeStruct]) -> _PyTreeStruct:
    return (jax.tree.structure(x), tuple(jax.tree.leaves(x)))


class BlockSpec:
    """
    This contains the necessary mappings for logging indices of states and node types.

    This helps convert between block states and global states. A block state is a list
    of pytrees, where each pytree leaf has shape[0] = number of nodes in the block.
    The length of the block state is the number of blocks. The global state is a
    flattened version of this. Each pytree type is combined (regardless of which block
    they are in), to make a list of pytrees where each leaf shape[0] is the total
    number of nodes of that pytree shape. As an example, imagine an Ising model,
    every node is the same pytree (just a scalar array), as such the block state is
    a list of arrays where each array is the state of the block and the global state
    would be a length-1 list that contains an array of shape (total_nodes,).

    Why is this global/block representation necessary? The answer is that the global
    representation is preferred for operating over in many JAX cases, but requires
    careful indexing (to know where in this long array each block resides) and thus
    the block representation is more natural/easy to use for many users. Why is the
    global state easier to work with? Well consider sampling, in order to sample a
    block (or even just a node) we need to collect all the states of the neighboring
    nodes. If we only had the block state we would have to loop over the block state
    and collect from each block the neighbors, we would then pass this to the
    sampler. The sampler would then have to know the type of each block (to know
    what to do with the states) then for loop over the blocks in order to collect
    each. This (programmatically) is fine, but results in additional for loops that
    slow down JAX, compared to gathering indexes from a single array.


    **Attributes:**

    - `blocks`: the list of blocks this spec contains
    - `all_block_sds`: a SD is a single `_PyTreeStruct`. Each node/block has only
        one SD associated with it, but each node can have neighbors of many types.
        This is the SD of each block (in the same order as blocks, this internal
        ordering is quite important for bookkeeping). This list is just the list
        of SDs for each block (and thus has length = len(blocks)).
    - `global_sd_order`: the list of SDs, providing a SoT for the global ordering
    - `sd_index_map`: a dictionary mapping the SD to an integer in the
        `global_sd_order`. This is like calling `.index` on it.
    - `node_global_location_map`: a dictionary mapping a given node to a tuple.
        That tuple contains the global index (i.e. which element in the global
        list it is in) and the relative position in that pytree. That is to say,
        you can get the state of the node via
        `map(x[tuple[1]], global_repr[tuple[0]])`
    - `block_to_global_slice_spec`: a list over unique SDs (so length
        global_sd_order), where each list inside this is the list over blocks
        which contain that pytree. E.g. [[0, 1], [2]] indicates that blocks[0]
        and blocks[1] are both of pytree SD 0.
    - `node_shape_dtypes`: a dictionary mapping node types to hashable `_PyTreeStruct`
    - `node_shape_struct`: a dictionary mapping node types to pytrees of JAX-shaped
        dtype structs (just for user access, since the keys aren't hashable that
        creates issues for JAX in other areas.)

    """

    blocks: list[Block]
    all_block_sds: list[_PyTreeStruct]
    global_sd_order: list[_PyTreeStruct]
    sd_index_map: dict[_PyTreeStruct, int]
    node_global_location_map: dict[AbstractNode, tuple[int, int]]
    block_to_global_slice_spec: list[list[int]]
    node_shape_dtypes: dict[Type[AbstractNode], _PyTreeStruct]
    node_shape_struct: dict[Type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]

    def __init__(
        self,
        blocks: list[Block],
        node_shape_dtypes: _Node_SD,
    ) -> None:
        """
        Create a BlockSpec from blocks.

        Based on the information passed in via node_shape_dtypes, determine the minimal global state that can be used
        to represent the blocks.

        **Arguments:**

        - `blocks`: the list of `Block`s that this specification operates on
        - `node_shape_dtypes`: the mapping of node types to their structures. This
                should be a pytree of `jax.ShapeDtypeStruct`s.
        """
        # variable types are assigned to nodes based on their class
        self.node_shape_struct = dict(node_shape_dtypes)
        self.node_shape_dtypes = {i: _hash_pytree(j) for i, j in node_shape_dtypes.items()}

        self.blocks = blocks

        # come up with an ordering of SDs for the global representation of the blocks
        all_sds = list({sd for sd in self.node_shape_dtypes.values()})
        self.global_sd_order = all_sds

        # map from SD to location in global representation
        self.sd_index_map = {sd: i for i, sd in enumerate(self.global_sd_order)}

        for block in blocks:
            if len(block) == 0:
                raise ValueError("Encountered an empty block in BlockSpec.")

            if block.node_type not in node_shape_dtypes:
                raise ValueError(f"Block with node type {block.node_type} not found in node_shape_dtypes.")

        self.all_block_sds = [self.node_shape_dtypes[block.node_type] for block in blocks]

        block_to_global_slice_spec = [[] for _ in self.global_sd_order]

        node_global_location_map = {}
        arr_ind_tracker = [0 for _ in self.global_sd_order]
        for block_idx, (block, sds) in enumerate(zip(blocks, self.all_block_sds)):
            block_len = len(block)

            sd_ind = self.sd_index_map[sds]
            start_ind = arr_ind_tracker[sd_ind]
            arr_ind_tracker[sd_ind] += block_len
            block_to_global_slice_spec[sd_ind].append(block_idx)
            for k, node in enumerate(block.nodes):
                if node in node_global_location_map:
                    raise RuntimeError("A node should not show up twice in the blocks input to BlockSpec.")
                node_global_location_map[node] = (sd_ind, start_ind + k)
        self.block_to_global_slice_spec = block_to_global_slice_spec
        self.node_global_location_map = node_global_location_map


def _stack(*args):
    if eqx.is_array(args[0]):
        if args[0].shape == ():
            return jnp.stack(args)
        # concate across node dim
        return jnp.concatenate(args, axis=0)
    else:
        assert all(args[0] == arg for arg in args[1:])
        return args[0]


def block_state_to_global(block_state: list[_State], spec: BlockSpec) -> list[_GlobalState]:
    """
    Convert block-local state to the global stacked representation.

    The block representation is a list where ``block_state[i]`` contains the
    state of ``spec.blocks[i]`` and every node occupies index 0 of its leaf.

    The global representation is a shorter list (one entry per distinct
    PyTree structure) in which all blocks with the same structure are
    concatenated along their node axis.

    **Arguments:**

    - `block_state`: State organised per block, same length as
        ``spec.blocks``.
    - `spec`: The [`thrml.BlockSpec`][] that defines the mapping.

    **Returns:**

    A list whose length equals
    ``len(spec.global_sd_order)``—the stacked global state.
    """
    # list is len(blocks), global_sd_order = len(unique pytrees) <= len(blocks)
    global_state = []
    for sd_indexes in spec.block_to_global_slice_spec:
        if not sd_indexes:
            global_state.append(None)
            continue

        collected = [block_state[i] for i in sd_indexes]

        # todo: should probably expand dims to be 1 to be consistent?
        if len(collected) == 1:
            global_state.append(collected[0])
        else:
            global_state.append(jax.tree.map(_stack, *collected))

    return global_state


def get_node_locations(nodes: Block, spec: BlockSpec) -> tuple[int, Int[Array, " nodes"]]:
    """
    Locate a contiguous set of nodes inside the global state.

    **Arguments:**

    - `nodes`: A [`thrml.Block`][] whose nodes you want locations for.
    - `spec`: The [`thrml.BlockSpec`][] generated from the same graph.

    **Returns:**

    Tuple ``(sd_index, positions)`` where

    * *sd_index* is the position inside the global list returned by
      [`thrml.block_state_to_global`][], and
    * *positions* is a 1D array with the indices each node
      occupies inside that particular PyTree.
    """

    # Make sure all nodes are of the same type
    # if len(set([type(node) for node in nodes])) > 1:
    #     raise ValueError("All nodes must be of the same type")

    node_sds = spec.node_shape_dtypes[nodes.node_type]

    sd_inds = spec.sd_index_map[node_sds]
    global_locs = [spec.node_global_location_map[node][1] for node in nodes]
    slices = jnp.array(global_locs)

    return sd_inds, slices


def from_global_state(
    global_state: list[_GlobalState],
    spec_from: BlockSpec,
    blocks_to_extract: list[Block],
) -> list[_State]:
    """
    Extract the states for a subset of blocks from a global state.

    **Arguments:**

    - `global_state`: A state produced by
        [`thrml.block_state_to_global(spec_from)`][].
    - `spec_from`: The [`thrml.BlockSpec`][] associated with *global_state*.
    - `blocks_to_extract`: The blocks whose node states should be returned.

    **Returns:**

    A list with one element per *blocks_to_extract*—each element is a PyTree
    with exactly ``len(block)`` nodes in its leading dimension.
    """

    all_sd_inds = []
    all_sd_slices = []
    for block in blocks_to_extract:
        sd_inds, slices = get_node_locations(block, spec_from)
        all_sd_inds.append(sd_inds)
        all_sd_slices.append(slices)

    return [
        jax.tree.map(lambda x: jnp.take(x, sls, axis=0), global_state[_sd_ind])
        for _sd_ind, sls in zip(all_sd_inds, all_sd_slices)
    ]


def make_empty_block_state(
    blocks: list[Block],
    node_shape_dtypes: _Node_SD,
    batch_shape: Optional[tuple] = None,
) -> list[_State]:
    """
    Allocate a zero-initialised block state.

    **Arguments:**

    - `blocks`: All blocks in the graph (order is preserved).
    - `node_shape_dtypes`: Maps every node class to its
        `jax.ShapeDtypeStruct` PyTree template.
    - `batch_shape`: Optional batch dimension(s) to prepend to every leaf.

    **Returns:**

    A list of PyTrees—one per *block*—whose leaves are
    ``zeros(batch_shape + (len(block),) + leaf.shape)``.
    """
    state = []
    for block in blocks:
        types = node_shape_dtypes[block.node_type]
        if batch_shape is None:
            this_state = jax.tree.map(
                lambda x: jnp.zeros(shape=(len(block), *x.shape), dtype=x.dtype),
                types,
            )
        else:
            this_state = jax.tree.map(
                lambda x: jnp.zeros(shape=(*batch_shape, len(block), *x.shape), dtype=x.dtype),
                types,
            )
        state.append(this_state)
    return state


def _check_pytree_compat(
    spec_tree,
    data_tree,
) -> tuple[int, ...] | None:
    """
    Verify that a PyTree of arrays matches up with a PyTree of ShapeDtypeStructs, up to a uniform batch shape.

    **Arguments:**

    - `spec_tree`: Pytree with `jax.ShapeDtypeStruct` leaves (at positions you want checked).
    - `data_tree`: Pytree with arrays at matching positions.

    **Returns:**

    The extracted batch shape if the two pytrees are compatible
    """

    if not jax.tree.structure(spec_tree) == jax.tree.structure(data_tree):
        raise RuntimeError("Tree structure mismatch between shape/dtype spec and data")

    spec_leaves, _ = jax.tree.flatten_with_path(spec_tree)
    val_leaves, _ = jax.tree.flatten_with_path(data_tree)

    batch_shape = None

    for (path, spec_leaf), (_, val_leaf) in zip(spec_leaves, val_leaves):
        if isinstance(spec_leaf, jax.ShapeDtypeStruct):
            if not eqx.is_array(val_leaf):
                raise RuntimeError("Array missing from data")

            vshape, vdtype = val_leaf.shape, val_leaf.dtype
            sshape, sdtype = spec_leaf.shape, spec_leaf.dtype

            val_shape_without_batch = () if not len(sshape) else vshape[-(len(sshape)) :]

            if val_shape_without_batch != sshape:
                raise RuntimeError("Shape of data mismatched with spec")

            cur_batch = vshape[: len(vshape) - len(sshape)]
            if batch_shape is None:
                batch_shape = cur_batch
            elif cur_batch != batch_shape:
                raise RuntimeError("Inconsistent batch shape in data")

            if vdtype != sdtype:
                raise RuntimeError(f"Data has incorrect type {vdtype} vs {sdtype}")

    return batch_shape


def verify_block_state(
    blocks: list[Block], states: list[_State], node_shape_dtypes: _Node_SD, block_axis: Optional[int] = None
) -> None:
    """
    Check that a state is what it should be given some blocks and node shape/dtypes.

    Passing incompatible state information into THRML functions can lead to unintended casting/other weird silent
    errors, so we should always check this.

    **Arguments:**

    - `blocks`: A list of Blocks.
    - `states`: A list of states to verify against blocks.
    - `node_shape_dtypes`: Maps every node class to its
        `jax.ShapeDtypeStruct` PyTree template.
    - `block_axis`: Index in the state batch shape at which to expect the block length.

    **Returns:**

    None. Raises RuntimeError if blocks and states are incompatible.
    """

    if not len(blocks) == len(states):
        raise RuntimeError("Number of states not equal to number of blocks")

    for block, state in zip(blocks, states):
        expected_sd = node_shape_dtypes[type(block.nodes[0])]
        batch_shape = _check_pytree_compat(expected_sd, state)
        assert batch_shape is not None
        if block_axis is not None:
            if not batch_shape[block_axis] == len(block.nodes):
                raise RuntimeError("State shape did not match detected block length")
```

## `thrml/thrml/interaction.py`

**`InteractionGroup` — the primitive directed dependency.** Signature is `InteractionGroup(interaction, head_nodes, tail_nodes)` (interaction first). When `head_nodes[i]` is updated, its sampler receives the state of `tail_nodes[k][i]` for all `k` plus the `i`-th slice of every array in `interaction`. This is the lowest-level object a sampler ever sees; factors compile down to lists of these. The leading dim of every array in `interaction` must equal `len(head_nodes)`.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from thrml.block_management import Block


class InteractionGroup(eqx.Module):
    """Defines computational dependencies for conditional sampling updates.

    An `InteractionGroup` specifies information that is required to update the state of some subset
    of the nodes of a PGM during a block sampling routine.

    More concretely, when the state of the node at head_nodes[i] is being updated, the sampler will receive the current
    state of the nodes at tail_nodes[k][i] for all k, and the ith element of each array in the Interaction PyTree
    (sliced along the first dimension).

    **Attributes:**

    - `head_nodes`: these are the nodes whose conditional updates should be affected by this InteractionGroup.
    - `tail_nodes`: these are the nodes whose state information is required to update `head_nodes`.
    - `interaction`: this specifies the parametric (independent of the state of the sampling program)
        required to update 'head_nodes'.
    """

    head_nodes: Block
    tail_nodes: list[Block]
    interaction: PyTree

    def __init__(self, interaction: PyTree, head_nodes: Block, tail_nodes: list[Block]):
        """Create an `InteractionGroup`.

        An `InteractionGroup` implements a group of directed interactions between nodes
        in a PGM sampling program.

        **Arguments:**

        - `interaction`: A PyTree specifying the static information associated with the
            interaction. The first dimension of every Array in interaction must be equal
            to the length of `head_nodes`.
        - `head_nodes`: The nodes whose update is affected by the interaction.
        - `tail_nodes`: The groups of nodes whose state is required to update
            `head_nodes`. Each block in this list of blocks is intended to be parallel
            to `head_nodes`. i.e, to update the state of head_nodes[i] during sampling
            we need state info about tail_nodes[k][i] for all values of k.
        """

        interaction_size = len(head_nodes.nodes)

        for block in tail_nodes:
            if not len(block.nodes) == interaction_size:
                raise RuntimeError("All tail node blocks must have the same length as head_nodes")

        def _get_dim(x):
            return (-1 if not len(x.shape) else x.shape[0]) if isinstance(x, jnp.ndarray) else interaction_size

        dims = jax.tree.leaves(jax.tree.map(_get_dim, interaction))
        if not all(dim == interaction_size for dim in dims):
            raise RuntimeError(
                "All arrays in interaction must have leading dimension equal to the length of head_nodes"
            )

        self.interaction = interaction
        self.head_nodes = head_nodes
        self.tail_nodes = tail_nodes
```

## `thrml/thrml/conditional_samplers.py`

**Conditional samplers — the update rule.** A sampler's `sample(key, interactions, active_flags, states, sampler_state, output_sd)` is called once per block per iteration and returns `(new_state, new_sampler_state)`. Shapes are `[n_nodes, max_appearances, ...]` with `active_flags` masking padded appearances (padding is how heterogeneous graphs vectorize on a GPU). `SpinGibbsConditional` implements `P(s=1|nb)=σ(2·field)` and is what the Ising wrapper attaches automatically; `CategoricalGibbsConditional(K)` does the softmax update. `AbstractParametricConditionalSampler` splits the work into `compute_parameters` + `sample_given_parameters` (override `compute_parameters` to extend a built-in sampler with a new interaction type while delegating native ones to `super()`).

```python
import abc
from typing import TypeVar

import equinox as eqx
import jax
from jaxtyping import Array, Key, PyTree, Shaped

_State = PyTree[Shaped[Array, "nodes ?*state"], "State"]
_SamplerState = TypeVar("_SamplerState", bound=PyTree)


class AbstractConditionalSampler(eqx.Module):
    """
    Base class for all conditional samplers.

    A conditional sampler is used to update the state of a block of nodes during each iteration of a sampling algorithm.
    It takes in the states of all the neighbors and produces a sample for the current block of nodes.
    This can often be done exactly, but need not be. One could embed MCMC methods within this sampler
    (to do Metropolis within Gibbs, for example).
    """

    @abc.abstractmethod
    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[_State, _SamplerState]:
        """Draw a sample from this conditional.

        If this sampler is involved in a block sampling program, this function is called every iteration to update the
        state of a block of nodes.

        **Arguments:**

        - `key`: A RNG key that the sampler can use to sample from distributions using `jax.random`.
        - `interactions`: A list of interactions that influence the result of this block update. Each interaction
            is a PyTree. Each array in the PyTree will have shape [n, k, ...], where n is the number of nodes in
             the block that is being updated and k is the maximum number of times any node in this block was
             detected as a head node for this interaction.
        - `active_flags`: A list of arrays of flags that is parallel to interactions. Each array indicates which
            instances of a given interaction are active for each node in the block. This array has shape [n, k],
            and is False if a given instance is inactive (which means that it should be ignored during the
             computation that happens in this function).
        - `states`: A list of PyTrees that is parallel to interactions, representing the sampling state information
            that is relevant to computing the influence of each interaction. Every array in each PyTree will have
            shape [n, k, ...].
        - `sampler_state`: The current state of this sampler. Will be replaced by the second return from this function
            the next time it is called.
        - `output_sd`: A PyTree indicating the expected shape/dtype of the output of this function.

        **Returns:**

        A new state for the block of nodes, matching the template given by `output_sd`.
        """
        pass

    def init(self) -> None:
        """Initialize the sampler state before sampling begins.

        This is called before the first iteration of block sampling, after which the return of this method is
        superseded by the return from `sample`.

        Returns:
            the initial sampler state to use for the first iteration of block sampling.
        """
        return None


class AbstractParametricConditionalSampler(AbstractConditionalSampler):
    """A conditional sampler that leverages a parameterized distribution.

    When `sample` is called, this sampler will first compute a set of parameters, and then use those parameters
    to draw a sample from some distribution. This workflow is frequently useful in practical cases; for example, to
    sample from a Gaussian, we can first compute a mean vector and covariance matrix using any procedure, and then
    draw a sample from the corresponding Gaussian distribution by appropriately transforming a vector of standard
    normal random variables."""

    @abc.abstractmethod
    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: PyTree,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        """Compute the parameters of the distribution. For a description of the arguments, see
        [`thrml.AbstractConditionalSampler.sample`][]"""
        pass

    @abc.abstractmethod
    def sample_given_parameters(
        self, key: Key, parameters: PyTree, sampler_state: _SamplerState, output_sd: PyTree[jax.ShapeDtypeStruct]
    ) -> tuple[_State, _SamplerState]:
        """Produce a sample given the parameters of the distribution, passed in as the `parameters` argument."""
        pass

    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[_State, _SamplerState]:
        """Sample from the distribution by first computing the parameters and then generating
        a sample based off of them."""
        key, subkey = jax.random.split(key, 2)
        parameters, state = self.compute_parameters(
            subkey, interactions, active_flags, states, sampler_state, output_sd
        )
        return self.sample_given_parameters(key, parameters, state, output_sd)


class BernoulliConditional(AbstractParametricConditionalSampler):
    r"""Sample from a bernoulli distribution.

    This sampler is designed to sample from a spin-valued bernoulli distribution:

    $$\mathbb{P}(S=s) \propto e^{\gamma s}$$

    where $S$ is a spin-valued random variable, $s \in \{-1, 1\}$. The parameter $\gamma$ must be
    computed by `compute_parameters`.
    """

    @abc.abstractmethod
    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        r"""A concrete implementation of this function has to return a value of $\gamma$ for every node
        in the block that is being updated. This array should have shape [b]."""
        pass

    def sample_given_parameters(
        self, key: Key, parameters: PyTree, sampler_state: None, output_sd: PyTree[jax.ShapeDtypeStruct]
    ) -> tuple[_State, None]:
        r"""Sample from a spin-valued bernoulli distribution given the parameter $\gamma$. In THRML,
        1 is represented by the boolean value `True` and -1 is represented by `False`."""
        return jax.random.bernoulli(key, jax.nn.sigmoid(2 * parameters)), sampler_state


class SoftmaxConditional(AbstractParametricConditionalSampler):
    r"""Sample from a softmax distribution.

    This sampler samples from the standard softmax distribution:

    $$\mathbb{P}(X=k) \propto e^{\theta_k}$$

    where $X$ is a categorical random variable and $\theta$ is a vector that parameterizes the relative
    probabilities of each of the categories.
    """

    @abc.abstractmethod
    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        """A concrete implementation of this function has to return $\theta$ vector for every node
        in the block that is being updated. This array should have shape [b, M], where $M$ is the
        number of possible values that $X$ may take on."""
        pass

    def sample_given_parameters(
        self, key: Key, parameters: PyTree, sampler_state: None, output_sd: PyTree[jax.ShapeDtypeStruct]
    ) -> tuple[_State, None]:
        """Sample from a softmax distribution given the parameter vector $\theta$."""
        return jax.random.categorical(key, parameters, axis=-1).astype(output_sd.dtype), sampler_state
```

## `thrml/thrml/factor.py`

**Factors — undirected energy terms, plus the program builder.** `AbstractFactor` holds parallel `node_groups` (all equal length) and implements `to_interaction_groups()`, which emits the directed `InteractionGroup`s the term induces (e.g. a pairwise coupling becomes two interactions, one per endpoint as head). `WeightedFactor(weights, node_groups)` validates `weights.shape[0] == len(node_groups[0])`. `FactorSamplingProgram(gibbs_spec, samplers, factors, other_interaction_groups)` is the workhorse assembler: it just compiles every factor to interaction groups and hands them to a `BlockSamplingProgram`. This is the layer you build on for *non-Ising* energies.

```python
import abc
from typing import Sequence

import equinox as eqx
from jaxtyping import Array

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, BlockSamplingProgram
from thrml.conditional_samplers import AbstractConditionalSampler
from thrml.interaction import InteractionGroup


class AbstractFactor(eqx.Module):
    """A factor represents a batch of undirected interactions between sets of random variables.

    Concretely, this class implements a batch of factors defined over a bunch of parallel node groups. A single
    factor is defined over the nodes given by node_groups[k][i] for all k and a particular i. The defining trait of a
    factor is to produce InteractionGroups that affect each member of the factor in some way during the conditional
    updates of a block sampling program. As a user, you specify how this is done by implementing a
    concrete to_interaction_groups method for your child class.

    **Attributes:**

    - `node_groups`: the list of blocks that makes up this batch of factors.
    """

    node_groups: list[Block]

    def __init__(self, node_groups: list[Block]):
        """Create a batch of Factors.

        Practically, this just means writing down some parallel groups of nodes that the batch of Factors acts on.
        All of the functionality of the Factor is implemented by the method to_interaction_groups.

        **Arguments:**

        - `node_groups`: The node groups that this batch of factors acts on. A single Factor is defined
            over node_groups[k][i] for all values of k and a particular batch index i.
        """

        if not len(node_groups) > 0:
            raise RuntimeError("A factor should not be empty.")

        n_nodes = len(node_groups[0].nodes)

        for group in node_groups:
            if not len(group.nodes) == n_nodes:
                raise RuntimeError("Every block in node_groups must contain the same number of nodes.")

        self.node_groups = node_groups

    @abc.abstractmethod
    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Compile a factor to a set of directed interactions."""
        pass


class WeightedFactor(AbstractFactor):
    """A factor that is parameterized by a weight tensor.

    The leading dimension of the weights tensor must be the same length as the batch dimension of the factor (i.e
    the number of nodes in each of the node_groups).

    **Attributes:**

    - `weights`: the weight tensor.
    """

    weights: Array

    def __init__(self, weights: Array, node_groups: list[Block]):
        super().__init__(node_groups)
        if not weights.shape[0] == len(node_groups[0].nodes):
            raise RuntimeError(
                "The leading dimension of weights must have the same length as the number of nodes in each node group"
            )
        self.weights = weights


class FactorSamplingProgram(BlockSamplingProgram):
    """A sampling program built out of factors.

    This class simply breaks each factor passed to it down into interaction groups and uses them to build a
    BlockSamplingProgram.
    """

    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: list[AbstractConditionalSampler],
        factors: Sequence[AbstractFactor],
        other_interaction_groups: list[InteractionGroup],
    ):
        """Create a FactorSamplingProgram. Thin wrapper over `BlockSamplingProgram`.

        **Arguments:**

        - `gibbs_spec`: A division of some PGM into free and clamped blocks.
        - `samplers`: The update rule to use for each free block in gibbs_spec.
        - `factors`: The factors to use to build this sampling program.
        - `other_interaction_groups`: Other interaction groups to include in your program alongside what the
            factors produce.

        """

        all_interaction_groups = list(other_interaction_groups)
        for factor in factors:
            all_interaction_groups += factor.to_interaction_groups()

        super().__init__(gibbs_spec, samplers, all_interaction_groups)
```

## `thrml/thrml/block_sampling.py`

**Specs, programs, and the run loop.** `BlockGibbsSpec(free_super_blocks, clamped_blocks, node_shape_dtypes=...)` records which blocks are sampled vs held fixed, in order; a *SuperBlock* (a tuple of blocks) lets same-color nodes of different types share input state while using different samplers. `SamplingSchedule(n_warmup, n_samples, steps_per_sample)` controls burn-in and thinning. `sample_states(key, program, schedule, init_state_free, state_clamp, nodes_to_sample)` runs the chain and returns recorded states for the requested blocks; `sample_with_observation` is the general form taking an `Observer`. The Gibbs loop simply updates each free block in spec order, every iteration.

```python
import dataclasses
from collections import defaultdict
from typing import Mapping, Sequence, Type, TypeAlias

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree, Shaped

from thrml.block_management import (
    Block,
    BlockSpec,
    block_state_to_global,
    verify_block_state,
)
from thrml.interaction import InteractionGroup
from thrml.pgm import DEFAULT_NODE_SHAPE_DTYPES, AbstractNode

from .conditional_samplers import AbstractConditionalSampler, _SamplerState
from .observers import AbstractObserver, ObserveCarry, StateObserver

# A SuperBlock is a collection of blocks that will be sampled at the same "time"
# specifically, they will be sampled separately, but without updating the state
# in between (so same algorithmic, but not computation time)
# This could be used if you have need to break blocks up for e.g. different
# samplers/nodes
SuperBlock: TypeAlias = tuple[Block, ...] | Block
_SD: TypeAlias = Mapping[Type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]


class BlockGibbsSpec(BlockSpec):
    """
    A BlockGibbsSpec is a type of BlockSpec which contains additional information
    on free and clamped blocks.

    This entity also supports `SuperBlock`s, which are merely groups of blocks
    which are sampled at the same time algorithmically, but not programmatically.
    That is to say, superblock = (block1, block2) means that the states input to
    block1 and block2 are the same, but they are not executed at the same time.
    This may be because they are the same color on a graph, but require vastly
    different sampling methods such that JAX SIMD approaches are not feasible
    to parallelize them.

    A recurring theme in `thrml` is the importance of implicit indexing. One
    such example can be seen here. Because global states are created by
    concatenating lists of free and clamped blocks, providing the inputs
    in the same order as the blocks are defined is essential. This is almost
    always taken care of internally, but when writing custom functions or
    interfaces this is important to keep in mind.

    **Attributes:**

    - `free_blocks`: the list of free blocks (in order)
    - `sampling_order`: a list of `len(superblocks)` lists, where each
        `sampling_order[i]` is the index of `free_blocks` to sample.
        Sampling is done by iterating over this order and sampling each
        sublist of free blocks at the same algorithmic time.
    - `clamped_blocks`: the list of clamped blocks
    - `superblocks`: the list of superblocks
    """

    free_blocks: list[Block]
    sampling_order: list[list[int]]
    clamped_blocks: list[Block]
    superblocks: list[tuple[Block, ...]]

    def __init__(
        self,
        free_super_blocks: Sequence[SuperBlock],
        clamped_blocks: list[Block],
        node_shape_dtypes: _SD = DEFAULT_NODE_SHAPE_DTYPES,
    ):
        """Create a Gibbs specification from free and clamped blocks.

        **Arguments:**

        - `free_super_blocks`: An ordered sequence where each element is either
            a single `Block`, or a tuple of blocks that must share the same global
            state when calling their individual samplers.
        - `clamped_blocks`: Blocks whose nodes stay fixed during sampling.
        - `node_shape_dtypes`: Mapping from node class to a PyTree of
            `jax.ShapeDtypeStruct`; identical to the argument in `BlockSpec`.
        """
        free_blocks = []
        sampling_order = []
        superblocks = []
        i = 0
        for super_block in free_super_blocks:
            if isinstance(super_block, Block):
                blocks = (super_block,)
            else:
                blocks = super_block

            superblocks.append(blocks)
            sampling_group = []
            for block in blocks:
                free_blocks.append(block)
                sampling_group.append(i)
                i += 1
            sampling_order.append(sampling_group)

        super().__init__(free_blocks + clamped_blocks, node_shape_dtypes)
        self.free_blocks = free_blocks
        self.clamped_blocks = clamped_blocks
        self.sampling_order = sampling_order
        self.superblocks = superblocks


def _tree_slice(x, sl):
    if eqx.is_array(x):
        return jnp.take(x, sl, axis=0)
    return x


class BlockSamplingProgram(eqx.Module):
    """A PGM block-sampling program.

    This class encapsulates everything that is needed to run a PGM block sampling program in THRML.
    `per_block_interactions` and `per_block_interaction_active` are parallel to the free blocks in `gibbs_spec`, and
    their members are passed directly to a sampler when the state of the corresponding free block is being updated
    during a sampling program. `per_block_interaction_global_inds` and `per_block_interaction_global_slices` are
    also parallel to the free blocks, and are used to slice the global state of the program to produce the
    state information required to update the state of each block alongside the static information contained in the
    interactions.

    **Attributes:**

    - `gibbs_spec`: A division of some PGM into free and clamped blocks.
    - `samplers`: A sampler to use to update every free block in `gibbs_spec`.
    - `per_block_interactions`: All the interactions that touch each free block in `gibbs_spec`.
    - `per_block_interaction_active`: indicates which interactions are real
        and which interactions are not part of the model and have been added to pad data structures so that they
        can be rectangular.
    - `per_block_interaction_global_inds`: how to find the information required to update each block within the global
        state list
    - `per_block_interaction_global_slices`: how to slice each array in the global state list to find the information
        required to update each block
    """

    gibbs_spec: BlockGibbsSpec
    samplers: list[AbstractConditionalSampler]
    per_block_interactions: list[list[PyTree]]
    per_block_interaction_active: list[list[Array]]
    per_block_interaction_global_inds: list[list[list[int]]]
    per_block_interaction_global_slices: list[list[list[Array]]]

    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: list[AbstractConditionalSampler],
        interaction_groups: list[InteractionGroup],
    ):
        """Construct a `BlockSamplingProgram`.

        This code is the beating heart of THRML, and the chance that you should be
        modifying it or trying to understand it deeply are very low (as this would
        basically correspond to re-writing the library). This code takes in a set of
        information that implicitly defines a sampling program and manipulates it into
        a shape that is appropriate for practical vectorized block-sampling program.
        This involves reindexing, slicing, and often padding.

        **Arguments:**

        - `gibbs_spec`: A division of some PGM into free and clamped blocks.
        - `samplers`: The update rule to use for each free block in `gibbs_spec`.
        - `interaction_groups`: A list of `InteractionGroups` that define how the
            variables in your sampling program affect one another.
        """

        self.gibbs_spec = gibbs_spec
        self.samplers = samplers

        n_free_blocks = len(self.gibbs_spec.free_blocks)
        if len(self.samplers) != n_free_blocks:
            raise ValueError(f"Expected {n_free_blocks} samplers, received {len(self.samplers)}")

        # first, construct a map from every head node to each interaction it
        # shows up in and where it shows up in that interaction

        head_node_map = defaultdict(list)

        for i, interaction_group in enumerate(interaction_groups):
            for j, node in enumerate(interaction_group.head_nodes.nodes):
                head_node_map[node].append((i, j))

        # now, let's organize this information on the interactions into a block format

        interaction_inds = []
        max_n_interactions = []

        for block in gibbs_spec.free_blocks:
            this_block_interaction_info = [
                [[] for _ in range(len(block.nodes))] for _ in range(len(interaction_groups))
            ]
            for j, node in enumerate(block.nodes):
                this_node_interaction_info = head_node_map[node]
                for info in this_node_interaction_info:
                    this_block_interaction_info[info[0]][j].append(info[1])
            interaction_inds.append(this_block_interaction_info)
            this_max_n = [max([len(x) for x in this_int]) for this_int in this_block_interaction_info]
            max_n_interactions.append(this_max_n)

        # now, take the block-arranged interaction structure and use it to construct the block-arranged interactions
        # and slicers for the global state

        # if you are reading this, god help you

        per_block_interactions = []
        per_block_interaction_active = []
        per_block_interaction_global_inds = []
        per_block_interaction_global_slices = []

        for block, block_interact_inds, block_n_interactions in zip(
            gibbs_spec.free_blocks, interaction_inds, max_n_interactions
        ):
            this_block_interactions = []
            this_block_active = []
            this_block_global_inds = []
            this_block_global_slices = []
            for interaction_group, interact_inds, n_interactions in zip(
                interaction_groups, block_interact_inds, block_n_interactions
            ):
                if n_interactions > 0:
                    n_nodes = len(block.nodes)
                    interaction_slices = np.zeros((n_nodes, n_interactions), dtype=int)

                    global_inds = []
                    global_slices = []
                    for tail_block in interaction_group.tail_nodes:
                        global_inds.append(gibbs_spec.node_global_location_map[tail_block.nodes[0]][0])
                        global_slices.append(np.zeros((n_nodes, n_interactions), dtype=int))

                    active = np.zeros((n_nodes, n_interactions), dtype=bool)
                    for i, inds in enumerate(interact_inds):
                        for j, ind in enumerate(inds):
                            interaction_slices[i, j] = ind
                            active[i, j] = 1

                            for k, tail_block in enumerate(interaction_group.tail_nodes):
                                s = gibbs_spec.node_global_location_map[tail_block.nodes[ind]][1]
                                global_slices[k][i, j] = s

                    interaction_slices = jnp.array(interaction_slices)

                    sliced_interaction = jax.tree.map(
                        lambda x: _tree_slice(x, interaction_slices),  # shape -> (n, m, …)
                        interaction_group.interaction,
                    )

                    this_block_interactions.append(sliced_interaction)
                    this_block_active.append(jnp.array(active))
                    this_block_global_inds.append(global_inds)
                    this_block_global_slices.append([jnp.array(x) for x in global_slices])
            per_block_interactions.append(this_block_interactions)
            per_block_interaction_active.append(this_block_active)
            per_block_interaction_global_inds.append(this_block_global_inds)
            per_block_interaction_global_slices.append(this_block_global_slices)

        self.per_block_interactions = per_block_interactions
        self.per_block_interaction_active = per_block_interaction_active
        self.per_block_interaction_global_inds = per_block_interaction_global_inds
        self.per_block_interaction_global_slices = per_block_interaction_global_slices


_State: TypeAlias = PyTree[Shaped[Array, "nodes ?*state"], "_State"]


def sample_single_block(
    key: Key[Array, ""],
    state_free: list[_State],
    clamp_state: list[_State],
    program: BlockSamplingProgram,
    block: int,
    sampler_state: _SamplerState,
    global_state: list[PyTree] | None = None,
) -> tuple[_State, _SamplerState]:
    """Samples a single block within a Gibbs sampling program based on the current
    states and program configurations. It extracts neighboring states, processes
    required data, and applies a sampling function to generate output samples.

    **Arguments:**

    - `key`: Pseudo-random number generator key to ensure reproducibility of sampling.
    - `state_free`: Current states of free blocks, representing the values to be
        updated during sampling.
    - `clamp_state`: Clamped states that remain fixed during the sampling process.
    - `program`: The Gibbs sampling program containing specifications, samplers,
        neighborhood information, and parameters.
    - `block`: Index of the block to be sampled in the current iteration.
    - `sampler_state`: The current state of the sampler that will be used to
        perform the update.
    - `global_state`: Optionally precomputed global state for the concatenated
        free and clamped blocks; when omitted the function constructs it internally.

    **Returns:**

    - Updated block state and sampler state for the specified block.
    """
    # flatten the state and extract neighbor states
    if global_state is None:
        global_state = block_state_to_global(state_free + clamp_state, program.gibbs_spec)
    per_interaction_global_inds = program.per_block_interaction_global_inds[block]
    per_interaction_slices = program.per_block_interaction_global_slices[block]

    all_interaction_states = []
    for interaction_global_inds, interaction_slices in zip(per_interaction_global_inds, per_interaction_slices):
        this_interaction_states = []
        for ind, sl in zip(interaction_global_inds, interaction_slices):
            this_interaction_states.append(
                jax.tree.map(
                    lambda x: jnp.take(x, sl, axis=0),  # shape -> (n, m, …)
                    global_state[ind],
                )
            )
        all_interaction_states.append(this_interaction_states)

    this_block = program.gibbs_spec.free_blocks[block]

    node_type = this_block.node_type
    template_sd = program.gibbs_spec.node_shape_struct[node_type]

    def _resize_sd(leaf):
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct((len(this_block.nodes), *leaf.shape), leaf.dtype)
        return leaf

    sd_to_pass = jax.tree.map(_resize_sd, template_sd)

    sampler = program.samplers[block]
    out_samples, out_sampler_state = sampler.sample(
        key,
        program.per_block_interactions[block],
        program.per_block_interaction_active[block],
        all_interaction_states,
        sampler_state,
        sd_to_pass,
    )
    return out_samples, out_sampler_state


def sample_blocks(
    key: Key[Array, ""],
    state_free: list[_State],
    clamp_state: list[_State],
    program: BlockSamplingProgram,
    sampler_state: list[_SamplerState],
) -> tuple[list[_State], list[_SamplerState]]:
    """Perform one iteration of sampling, visiting every block.

    **Arguments:**

    - `key`: The JAX PRNG key.
    - `state_free`: The state of the free blocks.
    - `clamp_state`: The state of the clamped blocks.
    - `program`: The Gibbs program.
    - `sampler_state`: The state of the sampler.

    **Returns:**

    - Updated free-block state list and sampler-state list.
    """

    # gaurdrail state/block compatibility here as everything else calls this

    sds = {node_type: jax.tree.unflatten(*sd) for node_type, sd in program.gibbs_spec.node_shape_dtypes.items()}
    verify_block_state(program.gibbs_spec.free_blocks, state_free, sds, -1)
    verify_block_state(program.gibbs_spec.clamped_blocks, clamp_state, sds, -1)

    keys = jax.random.split(key, (len(program.gibbs_spec.free_blocks),))
    for sampling_group in program.gibbs_spec.sampling_order:
        global_state = block_state_to_global(state_free + clamp_state, program.gibbs_spec)
        state_updates = {}
        for i in sampling_group:
            state_updates[i], sampler_state[i] = sample_single_block(
                keys[i], state_free, clamp_state, program, i, sampler_state[i], global_state
            )

        for i, state in state_updates.items():
            state_free[i] = state
    return state_free, sampler_state


def _run_blocks(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    init_chain_state: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    n_iters: int,
    sampler_states: list[_SamplerState],
) -> tuple[list[PyTree[Shaped[Array, "n_iters nodes ?*state"]]], list[_SamplerState]]:
    """
    Perform `n_iters` steps of block sampling.
    """
    if n_iters == 0:
        return init_chain_state, sampler_states

    def body_fn(states, _key):
        state_free, sampler_state = states
        return sample_blocks(_key, state_free, state_clamp, program, sampler_state), None

    keys = jax.random.split(key, n_iters)

    return jax.lax.scan(body_fn, (init_chain_state, sampler_states), keys)[0]


@dataclasses.dataclass
class SamplingSchedule:
    """
    Represents a sampling schedule for a process.

    **Attributes:**

    - `n_warmup`: The number of warmup steps to run before collecting samples.
    - `n_samples`: The number of samples to collect.
    - `steps_per_sample`: The number of steps to run between each sample.
    """

    n_warmup: int
    n_samples: int
    steps_per_sample: int

    def __hash__(self) -> int:
        return hash((self.n_warmup, self.n_samples, self.steps_per_sample))


def sample_with_observation(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_chain_state: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    observation_carry_init: ObserveCarry,
    f_observe: AbstractObserver,
) -> tuple[ObserveCarry, list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]]:
    """Run the full chain and call an Observer after every recorded sample.

    **Arguments:**

    - `key`: RNG key.
    - `program`: The sampling program.
    - `schedule`: Warm-up length, number of samples, number of steps between samples.
    - `init_chain_state`: Initial free-block state.
    - `state_clamp`: Clamped-block state.
    - `observation_carry_init`: Initial carry handed to `f_observe`.
    - `f_observe`: Observer instance.

    **Returns:**

    - Tuple `(final_observer_carry, samples)` where `samples` is a PyTree whose
        leading axis has size `schedule.n_samples`.
    """
    # run warmup
    sampler_states = jax.tree.map(
        lambda x: x.init(),
        program.samplers,
        is_leaf=lambda a: isinstance(a, AbstractConditionalSampler),
    )
    key, subkey = jax.random.split(key, 2)
    warmup_state, warmup_sampler_states = _run_blocks(
        subkey,
        program,
        init_chain_state,
        state_clamp,
        schedule.n_warmup,
        sampler_states,
    )
    mem, warmup_observation = f_observe(program, warmup_state, state_clamp, observation_carry_init, jnp.array(0))

    if schedule.n_samples <= 1:
        warmup_observation = jax.tree.map(lambda x: x[None], warmup_observation)
        return mem, warmup_observation

    # collect samples

    def body_fn(carry, input):
        (prev_state, prev_sampler_state), _mem = carry

        _key, i = input

        new_state, new_sampler_state = _run_blocks(
            _key,
            program,
            prev_state,
            state_clamp,
            schedule.steps_per_sample,
            prev_sampler_state,
        )
        _mem, observe_out = f_observe(program, new_state, state_clamp, _mem, i)
        new_carry = ((new_state, new_sampler_state), _mem)
        return new_carry, observe_out

    keys = jax.random.split(key, schedule.n_samples - 1)
    outer_iters = jnp.arange(1, schedule.n_samples)

    inputs = (keys, outer_iters)

    (_, mem_out), observed_results = jax.lax.scan(body_fn, ((warmup_state, warmup_sampler_states), mem), inputs)

    # need to prepend the first observation from the warmup
    def prepend_warmup_observation(_warmup, _rest):
        return jnp.concatenate([_warmup[None], _rest], axis=0)

    observed_results = jax.tree.map(prepend_warmup_observation, warmup_observation, observed_results)

    return mem_out, observed_results


def sample_states(
    key: Key[Array, ""],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state_free: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    nodes_to_sample: list[Block],
) -> list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]:
    """Convenience wrapper to collect state information for *nodes_to_sample* only.

    Internally builds a [`thrml.StateObserver`][], runs
    [`thrml.sample_with_observation`][], and returns a stacked tensor of shape
    `(schedule.n_samples, ...)`.
    """
    f_observe = StateObserver(nodes_to_sample)
    carry_init = f_observe.init()

    mem_out, results_out = sample_with_observation(
        key,
        program,
        schedule,
        init_state_free,
        state_clamp,
        carry_init,
        f_observe,
    )

    return results_out
```

## `thrml/thrml/observers.py`

**Observers — what to record.** `StateObserver(blocks)` memorizes raw state each recorded step (this is what `sample_states` uses internally). `MomentAccumulatorObserver(moment_spec, f_transform)` instead accumulates products of node states — pass `moment_spec=[[(n1,),(n2,)],[(n1,n2)]]` for first and second moments, and an optional `f_transform` (e.g. bool->spin) applied before accumulation. Use it to estimate moments/covariances without materializing every sample.

```python
import abc
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Sequence, TypeVar

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Int, PyTree

from thrml.block_management import Block, block_state_to_global, from_global_state

if TYPE_CHECKING:
    from thrml.block_sampling import _State, BlockSamplingProgram

from thrml.pgm import AbstractNode

ObserveCarry = TypeVar("ObserveCarry", bound=PyTree)


class AbstractObserver(eqx.Module):
    """
    Interface for objects that inspect the sampling program while it is running.

    A concrete Observer is called once per block-sampling iteration and can maintain an
    arbitrary "carry" state across calls (e.g. running averages, histogram
    buffers, log-probs, etc.).
    """

    @abc.abstractmethod
    def __call__(
        self,
        program: "BlockSamplingProgram",
        state_free: list[PyTree[Array]],
        state_clamped: list[PyTree[Array]],
        carry: ObserveCarry,
        iteration: Int[Array, ""],
    ) -> tuple[ObserveCarry, PyTree]:
        """Make an observation.

        This function is called at the end of a block-sampling iteration and can record information about the
        current state of the sampling program that might be useful for something later.

        **Arguments:**

        - `program`: The sampling program that is running when this function is called.
        - `state_free`: The current state of the free nodes involved in the sampling program.
        - `state_clamped`: The state of the clamped nodes involved in the sampling program.
        - `carry`: The "memory" available to this observer. This function should modify this PyTree to record
            information about the sampling program.
        - `iteration`: How many iterations of block sampling have happened before this function was called.

        **Returns:**

        A tuple, where the first element is the updated carry, and the second is a PyTree that will be
        recorded by the sampler.

        """
        return NotImplemented

    def init(self) -> PyTree:
        """Initialize the memory for the observer. Defaults to None."""
        return None


class StateObserver(AbstractObserver):
    """
    Observer which logs the raw state of some set of nodes.

    **Attributes:**

    - `blocks_to_sample`: the list of `Block`s which the states are logged for
    """

    blocks_to_sample: list[Block]

    def __call__(
        self,
        program: "BlockSamplingProgram",
        state_free: list["_State"],
        state_clamped: list["_State"],
        carry: None,
        iteration: Int[Array, ""],
    ) -> tuple[None, PyTree]:
        """Simply returns the state of the blocks that are being logged to be recorded by the sampler."""
        global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)
        sampled_state = from_global_state(global_state, program.gibbs_spec, self.blocks_to_sample)
        return None, sampled_state


def _f_identity(*x):
    return x[0]


class MomentAccumulatorObserver(AbstractObserver):
    r"""
    Observer that accumulates and updates the provided moments.

    It doesn't log any samples, and will only accumulate moments. Note that this observer does not
    scale the accumulated values by the number of times it was called. It simply records a running sum of a product
    of some state variables,

    $$\sum_i f(x_1^i) f(x_2^i) \dots f(x_N^i)$$



    **Attributes:**

    - `blocks_to_sample`: the blocks to accumulate the moments over. These
        are for constructing the final state, and aren't truly "blocks"
        in the algorithmic sense (they can be connected to each other).
        There is one block per node type.
    - `flat_nodes_list`: a list of all of the nodes in the moments (each
        occurring only once, so len(set(x)) = len(x)).
    - `flat_to_type_slices_list`: a list over node types in which each element
        is an array of indices of the `flat_node_list` which that type
        corresponds to
    - `flat_to_full_moment_slices`: a list over moment types in which each
        element is a 2D array, which matches the shape of the `moment_spec[i]`
        and of which each element is the index in the `flat_node_list`.
    - `f_transform`: the element-wise transformation $f$ to apply to sample values before
        accumulation.

    """

    blocks_to_sample: list[Block]
    flat_nodes_list: list[AbstractNode]
    flat_to_type_slices_list: list[Int[Array, " nodes_in_slice"]]
    flat_to_full_moment_slices: list[Int[Array, "num_groups nodes_in_moment"]]
    f_transform: Callable

    def __init__(self, moment_spec: Sequence[Sequence[Sequence[AbstractNode]]], f_transform: Callable = _f_identity):
        r"""
        Create a MomentAccumulatorObserver.

        **Arguments:**

        - `moment_spec`: A 3 depth sequence. The first is a sequence
            over different moment types. A given moment type should have the same
            number of nodes in each moment. Then for each moment type, there is a
            sequence over moments. Each given moment is defined by a certain set
            of nodes.

            For example, to get the first and second moments on a simple o-o graph,
            it would be

            [
                [(node1,), (node2,)],
                [(node1, node2)]
            ]
        - `f_transform`: A function that takes in (state, blocks) and returns something with the same structure as
            state. This is used to apply functions to the samples before moments are computed. i.e this function
            defines a transformation of the state variable $y=f(x)$, such that the accumulated moments
            are of the form $\langle f(x_1) f(x_2) \rangle$.
        """

        self.f_transform = f_transform

        flat_nodes_list = []
        node_to_flat_idx = {}
        flat_to_full_moment_slices = []
        nodes_by_type = defaultdict(list)
        flat_to_type_slices = defaultdict(list)

        for i, moment in enumerate(moment_spec):
            # moment = tuple of “rows” => each row is a tuple of nodes
            shape = (len(moment), len(moment[0]))
            moment_slice = np.zeros(shape, dtype=int)

            for j, nodes in enumerate(moment):
                for k, node in enumerate(nodes):
                    # node_to_flat_idx[node] is the integer index assigned
                    idx = node_to_flat_idx.get(node, -1)
                    if idx == -1:
                        idx = len(flat_nodes_list)
                        node_to_flat_idx[node] = idx
                        flat_nodes_list.append(node)
                    moment_slice[j, k] = idx
                    nodes_by_type[node.__class__].append(node)
                    flat_to_type_slices[node.__class__].append(node_to_flat_idx[node])

            flat_to_full_moment_slices.append(jnp.array(moment_slice, dtype=int))

        blocks_to_sample = []
        flat_to_type_slices_list = []

        for node_type, nodes in nodes_by_type.items():
            blocks_to_sample.append(Block(nodes))
            type_slice = jnp.array(flat_to_type_slices[node_type], dtype=int)
            flat_to_type_slices_list.append(type_slice)

        self.flat_nodes_list = flat_nodes_list
        self.flat_to_full_moment_slices = flat_to_full_moment_slices
        self.blocks_to_sample = blocks_to_sample
        self.flat_to_type_slices_list = flat_to_type_slices_list

    def __call__(
        self,
        program: "BlockSamplingProgram",
        state_free: list[PyTree[Array]],
        state_clamped: list[PyTree[Array]],
        carry: list[Array],
        iteration: Int[Array, ""],
    ) -> tuple[list[Array], PyTree]:
        """Accumulate the moments via `carry`. Does not return anything for the sampler to write down."""
        global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)

        sampled_state = from_global_state(global_state, program.gibbs_spec, self.blocks_to_sample)

        sampled_state = self.f_transform(sampled_state, self.blocks_to_sample)
        sampled_state = list(sampled_state)

        flat_state = jnp.zeros(len(self.flat_nodes_list))
        result_type = jnp.result_type(*jax.tree.leaves(sampled_state))
        for i, type_slice in enumerate(self.flat_to_type_slices_list):
            if i == 0:
                flat_state = flat_state.astype(result_type)
            state = sampled_state[i]
            flat_state = flat_state.at[type_slice].set(state)

        def accumulate_moment(mem_entry, sl):
            update = jnp.prod(flat_state[sl], axis=1)
            return mem_entry.astype(update.dtype) + update

        mem = jax.tree.map(accumulate_moment, carry, self.flat_to_full_moment_slices)

        return mem, None

    def init(self) -> list[Array]:
        """Initialize the memory that will store the accumulated values."""
        return jax.tree.map(
            lambda x: jnp.zeros(x.shape[:1], dtype=float),
            self.flat_to_full_moment_slices,
        )
```

## `thrml/thrml/models/ebm.py`

**EBM base classes.** `AbstractEBM` / `AbstractFactorizedEBM` define the `P(x) ∝ exp(-E(x))` interface where `E` is a sum of factors. `IsingEBM` (next file) is the concrete factorized EBM you will actually use. You only touch these abstractly when defining a new model family.

```python
import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from thrml.block_management import Block, BlockSpec, block_state_to_global
from thrml.block_sampling import _SD, _State
from thrml.factor import AbstractFactor
from thrml.pgm import DEFAULT_NODE_SHAPE_DTYPES


class AbstractEBM(eqx.Module):
    """
    Something that has a well-defined energy function (map from a state to a scalar).
    """

    @abc.abstractmethod
    def energy(self, state: list[_State], blocks: list[Block]) -> Float[Array, ""]:
        """Evaluate the energy function of the EBM given some state information.

        **Arguments:**

        - `state`: The state for which to evaluate the energy function. Must be compatible with `blocks`.
        - `blocks`: Specifies how the information in `state` is organized.

        **Returns:**

        A scalar representing the energy value associated with `state`.
        """
        raise NotImplementedError


class EBMFactor(AbstractFactor):
    """A factor that defines an energy function."""

    @abc.abstractmethod
    def energy(self, global_state: list[Array], block_spec: BlockSpec) -> Float[Array, ""]:
        """Evaluate the energy function of the factor.

        **Arguments:**

        - `global_state`: The state information to use to evaluate the energy function.
            Is a global state of `block_spec`.
        - `block_spec`: The `BlockSpec` used to generate `global_state`.
        """
        raise NotImplementedError


class AbstractFactorizedEBM(AbstractEBM):
    r"""An EBM that is made up of Factors, i.e., an EBM with an energy function like,

    $$\mathcal{E}(x) = \sum_i \mathcal{E}^i(x)$$

    where the sum over $i$ is taken over factors.

    Child classes must define a property which returns a list of
    factors that substantiate the EBM.

    **Attributes:**

    - `node_shape_dtypes`: the shape/dtypes of the nodes involved in this EBM. Used to generate the BlockSpec that
        defines the global state that factors receive to compute energy.
    """

    node_shape_dtypes: _SD

    def __init__(self, node_shape_dtypes: _SD = DEFAULT_NODE_SHAPE_DTYPES):
        self.node_shape_dtypes = node_shape_dtypes

    def energy(self, state: list[_State], blocks: list[Block]) -> Float[Array, ""]:
        block_spec = BlockSpec(blocks, self.node_shape_dtypes)
        global_state = block_state_to_global(state, block_spec)
        energy = jnp.array(0.0)
        for factor in self.factors:
            energy += factor.energy(global_state, block_spec)
        return energy

    @property
    @abc.abstractmethod
    def factors(self) -> list[EBMFactor]:
        """A concrete implementation of this class must define this method that returns a list of factors that
        substantiate the EBM."""
        raise NotImplementedError


class FactorizedEBM(AbstractFactorizedEBM):
    """An EBM that is defined by a concrete list of factors.

    **Attributes:**

    - `_factors`: the list of factors that defines this EBM.
    """

    _factors: list[EBMFactor]

    def __init__(self, factors: list[EBMFactor], node_shape_dtypes: _SD = DEFAULT_NODE_SHAPE_DTYPES):
        super().__init__(node_shape_dtypes)
        self._factors = factors

    @property
    def factors(self):
        return self._factors
```

## `thrml/thrml/models/discrete_ebm.py`

**Built-in discrete factors and their samplers.** `SpinEBMFactor(node_groups, weights)` encodes `s_1*...*s_M` terms at any order: one block = biases `(n,)`, two = pairwise `(n_edges,)`, three = triplets `(n,)`. `CategoricalEBMFactor(node_groups, weights)` does the categorical analogue with `(n,K)` / `(n,K,K)` weights. `DiscreteEBMFactor` mixes spin and categorical; the `Square*` variants enforce square weight tensors. These are the factors an `IsingEBM` emits internally, and the building blocks for hand-built non-Ising spin/categorical models.

```python
from collections import defaultdict
from typing import Type

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree

from thrml.block_management import Block, BlockSpec, from_global_state
from thrml.block_sampling import _State
from thrml.conditional_samplers import (
    BernoulliConditional,
    SoftmaxConditional,
)
from thrml.factor import WeightedFactor
from thrml.interaction import InteractionGroup
from thrml.models.ebm import EBMFactor
from thrml.pgm import AbstractNode


class DiscreteEBMInteraction(eqx.Module):
    """An interaction that shows up when sampling from discrete-variable EBMs.

    **Attributes:**

    - `n_spin`: the number of spin states involved in the interaction.
    - `weights`: the weight tensor associated with this interaction.
    """

    n_spin: int
    weights: Array


def _spin_product(spin_vals: list[Array]):
    """Take the product of a bunch of spin states. If the input has length 0 return 1."""

    if len(spin_vals) == 0:
        return jnp.array(1)

    return jnp.prod(2 * jnp.stack(spin_vals, -1).astype(jnp.int8) - 1, axis=-1)


def _batch_gather(x, *idx):
    n = len(idx)

    if n == 0:
        return x

    batch_shape = x.shape[:-n]

    x_flat = x.reshape((-1, *x.shape[-n:]))
    idx_flat = tuple(y.flatten() for y in idx)

    batch_idx = jnp.arange(idx_flat[0].shape[0])

    return x_flat[(batch_idx,) + idx_flat].reshape(batch_shape)


class DiscreteEBMFactor(EBMFactor, WeightedFactor):
    """Implements batches of energy function terms of the form s_1 * ... * s_M * W[c_1, ..., c_N],
    where the s_i are spin variables and the c_i are categorical variables.

    No variable should show up twice in any given interaction. If this happens, the result of sampling from a model
    that includes the bad factor might not agree with the Boltzmann distribution. For example, the interaction
    w * s_1 * s_1 * s_2 would violate this rule because s_1 shows up twice. To allow you to do something weird if
    you want to, this condition has not been enforced in the code.


    **Attributes:**

    - `spin_node_groups`: the node groups involved in the batch of factors that represent spin-valued random variables.
    - `categorical_node_groups`: the node groups involved in the batch of factors that represent categorical-valued
        random variables.
    - `weights`: the batch of weight tensors W associated with the factors we are implementing. `weights` should have
        leading dimension b, where b is number of nodes in each element of `spin_node_groups` and
        `categorical_node_groups`. This tensor has shape [b, x_1, ..., x_N] where b is the number of nodes
        in each block and N is the length of `categorical_node_groups`.
    - `is_spin`: a map that indicates if a given node type represents a spin-valued random variable or not.
    """

    spin_node_groups: list[Block]
    categorical_node_groups: list[Block]
    weights: Array
    is_spin: dict[Type[AbstractNode], bool]

    def __init__(self, spin_node_groups: list[Block], categorical_node_groups: list[Block], weights: Array):
        """
        Create a `DiscreteEBMFactor`.

        **Arguments:**

        - `spin_node_groups`: The spin node groups
        - `categorical_node_groups`: The categorical node groups
        - `weights`: The interaction weight tensor
        """

        WeightedFactor.__init__(self, weights, spin_node_groups + categorical_node_groups)

        is_spin = defaultdict(lambda: False)

        # remember which node types are spin
        for group in spin_node_groups:
            is_spin[type(group.nodes[0])] = True

        for group in categorical_node_groups:
            curr_type = type(group.nodes[0])
            if is_spin[curr_type]:
                raise RuntimeError("A node cannot be both categorical and spin.")
            is_spin[curr_type] = False

        # don't want this to recognize nodes that haven't been seen here
        # not sure how that would happen but better to be careful
        self.is_spin = dict(is_spin)
        self.spin_node_groups = spin_node_groups
        self.categorical_node_groups = categorical_node_groups

        if not len(weights.shape) == 1 + len(categorical_node_groups):
            raise RuntimeError(
                "The shape of the weight tensor must be [b, x_1, ..., x_k], where"
                "k is the length of categorical_node_groups."
            )

        self.weights = weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Produce interaction groups that implement this factor.

        In this case, we have to treat the spin and categorical node groups slightly differently.
        """
        interaction_groups = []

        n_spin = len(self.spin_node_groups)
        n_cat = len(self.categorical_node_groups)
        n_total = n_spin + n_cat

        # handle the interaction groups with spin head nodes
        if n_spin > 0:
            spin_inds = list(range(len(self.spin_node_groups)))
            spin_combos = [(x, spin_inds[:i] + spin_inds[i + 1 :]) for i, x in enumerate(spin_inds)]

            all_head_nodes = []
            all_tail_nodes = [[] for _ in range(n_total - 1)]
            for combo in spin_combos:
                all_head_nodes += self.spin_node_groups[combo[0]].nodes
                for i, tail_ind in enumerate(combo[1]):
                    all_tail_nodes[i] += self.spin_node_groups[tail_ind].nodes
                for j, cat_group in enumerate(self.categorical_node_groups):
                    all_tail_nodes[n_spin - 1 + j] += cat_group.nodes

            tiler = [1] * len(self.weights.shape)
            tiler[0] = n_spin

            rep_weights = jnp.tile(self.weights, tiler)
            interaction_groups.append(
                InteractionGroup(
                    DiscreteEBMInteraction(n_spin - 1, rep_weights),
                    Block(all_head_nodes),
                    [Block(x) for x in all_tail_nodes],
                )
            )

        # handle the interaction groups with categorical head nodes
        if n_cat > 0:
            cat_inds = list(range(len(self.categorical_node_groups)))
            cat_combos = [(x, cat_inds[:i] + cat_inds[i + 1 :]) for i, x in enumerate(cat_inds)]

            for combo in cat_combos:
                head_nodes = self.categorical_node_groups[combo[0]]

                cat_blocks = [self.categorical_node_groups[i] for i in combo[1]]

                reind = (0, combo[0] + 1, *[x + 1 for x in combo[1]])

                weights_reind = jnp.moveaxis(self.weights, reind, list(range(len(reind))))

                interaction_groups.append(
                    InteractionGroup(
                        DiscreteEBMInteraction(n_spin, weights_reind), head_nodes, self.spin_node_groups + cat_blocks
                    )
                )

        return interaction_groups

    def energy(self, global_state: list[Array], block_spec: BlockSpec):
        """Compute the energy associated with this factor.

        In this case, that is the sum of terms like s_1 * ... * s_M * W[c_1, ..., c_N].
        """
        spin_vals = from_global_state(global_state, block_spec, self.spin_node_groups)
        cat_vals = from_global_state(global_state, block_spec, self.categorical_node_groups)
        spin_prod = _spin_product(spin_vals)
        weights = _batch_gather(self.weights, *cat_vals)
        return -jnp.sum(weights * spin_prod.astype(weights.dtype))


def _merge_groups(groups, n_tail_groups):
    if len(groups) == 0:
        return groups

    all_head = []
    all_tail = [[] for _ in range(n_tail_groups)]
    all_weights = []
    for group in groups:
        all_head += group.head_nodes.nodes
        for i, block in enumerate(group.tail_nodes):
            all_tail[i] += block.nodes
        all_weights.append(group.interaction.weights)

    return [
        InteractionGroup(
            DiscreteEBMInteraction(groups[0].interaction.n_spin, jnp.concatenate(all_weights, axis=0)),
            Block(all_head),
            [Block(x) for x in all_tail],
        )
    ]


class SquareDiscreteEBMFactor(DiscreteEBMFactor):
    """A discrete factor with a square interaction weight tensor (shape [b, x, x, ..., x]).

    If a discrete factor is square, the interaction groups corresponding to different choices of the head
    node blocks can be merged. This could yield smaller XLA programs and improved runtime performance via
    more efficient use of accelerators.
    """

    def __init__(self, spin_node_groups: list[Block], categorical_node_groups: list[Block], weights: Array):
        """Enforce that the weights are actually square."""
        super().__init__(spin_node_groups, categorical_node_groups, weights)

        if len(weights.shape) > 2:
            target_shape = weights.shape[1]
            for shape in weights.shape[1:]:
                if not shape == target_shape:
                    raise RuntimeError("Interaction tensor is not square.")

    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Call the parent class to_interaction_groups, and merge the results."""
        groups = super().to_interaction_groups()

        spin_groups = []
        cat_groups = []

        for group in groups:
            if self.is_spin[type(group.head_nodes[0])]:
                spin_groups.append(group)
            else:
                cat_groups.append(group)

        n_tail = len(self.node_groups) - 1

        return _merge_groups(spin_groups, n_tail) + _merge_groups(cat_groups, n_tail)


class SpinEBMFactor(SquareDiscreteEBMFactor):
    """A `DiscreteEBMFactor` that involves only spin variables."""

    def __init__(self, node_groups: list[Block], weights: Array):
        super().__init__(node_groups, [], weights)


class CategoricalEBMFactor(DiscreteEBMFactor):
    """A `DiscreteEBMFactor` that involves only categorical variables."""

    def __init__(self, node_groups: list[Block], weights: Array):
        super().__init__([], node_groups, weights)


class SquareCategoricalEBMFactor(SquareDiscreteEBMFactor):
    """A `DiscreteEBMFactor` that involves only categorical variables that also has a square weight tensor."""

    def __init__(self, node_groups: list[Block], weights: Array):
        super().__init__([], node_groups, weights)


def _batch_gather_with_k(x, *idx):
    n = len(idx)
    batch_shape = x.shape[:-n]

    if len(batch_shape) == 0:
        return x

    k = batch_shape[-1]

    new_idx = [jnp.broadcast_to(jnp.expand_dims(y, -1), (*y.shape, k)).flatten() for y in idx]

    return _batch_gather(x.reshape((np.prod(batch_shape), *x.shape[-n:])), *new_idx).reshape(batch_shape)


def _split_states(states, n_spin):
    states_spin, states_cat = states[:n_spin], states[n_spin:]

    # make sure spin values are actually bool, cat values are actually unsigned
    # violating this could lead to some weird behaviour

    def _validate(check_states, name, ex_type):
        for state in check_states:
            if not (len(state.shape) == 2):
                raise RuntimeError("{} states must be scalar.".format(name))
            if not jnp.isdtype(state.dtype, ex_type):
                raise RuntimeError("{} states must be {}.".format(name, ex_type))

    _validate(states_spin, "Spin", "bool")
    _validate(states_cat, "Categorical", "unsigned integer")

    return states_spin, states_cat


class SpinGibbsConditional(BernoulliConditional):
    r"""A conditional update for spin-valued random variables that will perform a Gibbs sampling update given one or
    more `DiscreteEBMInteractions`.

    This function can be extended to handle a broader class of interactions via inheritance. Specifically, a
    child class can override the `compute_parameters` method defined here, compute contributions to $\gamma$
    from other types of interactions, and then call this method to take into account the contributions from
    `DiscreteEBMInteractions`."""

    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        r"""Compute the parameter $\gamma$ of a spin-valued Bernoulli distribution given DiscreteEBMInteractions:

        $$\gamma = \sum_i s_1^i \dots s_K^i \: W^i[x_1^i, \dots, x_M^i]$$

        where the sum over $i$ is over all the `DiscreteEBMInteractions` seen by this function.
        """

        gamma = jnp.zeros(output_sd.shape)
        for i, (interaction, active, state) in enumerate(zip(interactions, active_flags, states)):
            if isinstance(interaction, DiscreteEBMInteraction):
                state_bin, state_cat = _split_states(state, interaction.n_spin)

                weights = _batch_gather(interaction.weights, *state_cat)
                spin_prod = _spin_product(state_bin).astype(weights.dtype)
                active = active.astype(weights.dtype)
                if i == 0:
                    gamma = gamma.astype(weights.dtype)

                gamma += jnp.sum(weights * active * spin_prod, axis=-1)
            else:
                raise RuntimeError("Unsupported interaction found")
        return gamma, sampler_state


class CategoricalGibbsConditional(SoftmaxConditional):
    """A conditional update for categorical random variables that will perform a Gibbs sampling update given one or
        more `DiscreteEBMInteractions`.

    This function can be extended to handle other interactions in the same way as [`thrml.models.SpinGibbsConditional`][].

    **Attributes:**

    - `n_categories`: how many categories are involved in the softmax distribution this sampler will sample from.
    """

    n_categories: int

    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        r"""Compute the parameter $\theta$ of a softmax distribution given DiscreteEBMInteractions:

        $$\theta = \sum_i s_1^i \dots s_K^i \: W^i[:, x_1^i, \dots, x_M^i]$$

        where the sum over $i$ is over all the `DiscreteEBMInteractions` seen by this function.
        """

        theta = jnp.zeros((*output_sd.shape, self.n_categories))
        for i, (interaction, active, state) in enumerate(zip(interactions, active_flags, states)):
            if isinstance(interaction, DiscreteEBMInteraction):
                state_bin, state_cat = _split_states(state, interaction.n_spin)

                weights = _batch_gather_with_k(interaction.weights, *state_cat)
                spin_prod = jnp.expand_dims(_spin_product(state_bin), -1).astype(weights.dtype)
                if i == 0:
                    theta = theta.astype(weights.dtype)

                theta += jnp.sum(spin_prod * weights * jnp.expand_dims(active, -1).astype(weights.dtype), axis=-2)

            else:
                raise RuntimeError("Unsupported interaction found")

        return theta, sampler_state
```

## `thrml/thrml/models/ising.py`

**The Ising machine — the center of THRML, and where training lives.** `IsingEBM(nodes, edges, biases, weights, beta)` is the model object: energy `E(s) = -β(Σ bᵢsᵢ + Σ Jᵢⱼsᵢsⱼ)`, and its `.factors` property auto-emits the bias + pairwise `SpinEBMFactor`s. `IsingSamplingProgram(model, free_blocks, clamped_blocks)` auto-builds the spec and attaches a `SpinGibbsConditional` to every free block, so you never assemble factors/samplers by hand. Training is contrastive: `IsingTrainingSpec(ebm, data_blocks, conditioning_blocks, positive_sampling_blocks, negative_sampling_blocks, sched_pos, sched_neg)` packages a positive-phase program (data clamped, latents sampled) and a negative-phase program (everything sampled). `estimate_kl_grad(key, spec, bias_nodes, weight_edges, data, conditioning_values, init_positive, init_negative)` runs both and returns `(grad_w, grad_b, pos_moments, neg_moments)` from `ΔW=-β(⟨sᵢsⱼ⟩₊-⟨sᵢsⱼ⟩₋)`, `Δb=-β(⟨sᵢ⟩₊-⟨sᵢ⟩₋)`. `hinton_init(key, model, blocks, batch_shape)` seeds chains from the marginal bias; `estimate_moments` is the per-phase moment estimator the gradient is built from. **Pitfall:** `init_positive` must have one array per positive block — never `[]` — because the positive phase samples the latents.

```python
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Key

from thrml.block_sampling import (
    Block,
    BlockGibbsSpec,
    BlockSamplingProgram,
    SamplingSchedule,
    SuperBlock,
    sample_with_observation,
)
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.models.ebm import AbstractFactorizedEBM, EBMFactor
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import AbstractNode

Edge = tuple[AbstractNode, AbstractNode]


class IsingEBM(AbstractFactorizedEBM):
    r"""An EBM with the energy function,

    $$\mathcal{E}(s) = -\beta \left( \sum_{i \in S_1} b_i s_i + \sum_{(i, j) \in S_2} J_{ij} s_i s_j \right)$$

    where $S_1$ and $S_2$ are the sets of biases and weights that make up the model, respectively.
    $b_i$ represents the bias associated with the spin $s_i$ and $J_{ij}$ is a weight that couples
    $s_i$ and $s_j$. $\beta$ is the usual temperature parameter.

    **Attributes:**

    - `nodes`: the nodes that have an associated bias (i.e $S_1$)
    - `biases`: the bias associated with each node in `nodes`.
    - `edges`: the edges that have an associated weight (i.e $S_2$)
    - `weights`: the weight associated with each pair of nodes in `edges`.
    - `beta`: the scalar temperature parameter for the model.

    """

    nodes: list[AbstractNode]
    biases: Array
    edges: list[Edge]
    weights: Array
    beta: Array

    def __init__(self, nodes: list[AbstractNode], edges: list[Edge], biases: Array, weights: Array, beta: Array):
        """Initialize an Ising EBM.

        **Arguments:**

        - `nodes`: List of nodes with associated biases
        - `edges`: List of edge pairs with associated weights
        - `biases`: Bias values for each node
        - `weights`: Weight values for each edge
        - `beta`: Temperature parameter
        """
        # nodes should be same type, should be at least one node passed in
        sd_map = {nodes[0].__class__: jax.ShapeDtypeStruct((), jnp.bool_)}

        super().__init__(sd_map)

        self.nodes = nodes
        self.edges = edges
        self.beta = beta
        self.weights = weights
        self.biases = biases

    @property
    def factors(self) -> list[EBMFactor]:
        return [
            SpinEBMFactor([Block(self.nodes)], self.beta * self.biases),
            SpinEBMFactor(
                [Block([x[0] for x in self.edges]), Block([x[1] for x in self.edges])], self.beta * self.weights
            ),
        ]


class IsingSamplingProgram(FactorSamplingProgram):
    """A very thin wrapper on FactorSamplingProgram that specializes it to the case of an Ising Model."""

    def __init__(self, ebm: IsingEBM, free_blocks: list[SuperBlock], clamped_blocks: list[Block]):
        """Initialize an Ising sampling program.

        **Arguments:**

        - `ebm`: The Ising EBM to sample from
        - `free_blocks`: List of super blocks that are free to vary
        - `clamped_blocks`: List of blocks that are held fixed
        """
        samp = SpinGibbsConditional()

        spec = BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)

        super().__init__(spec, [samp for _ in spec.free_blocks], ebm.factors, [])


class IsingTrainingSpec(eqx.Module):
    """Contains a complete specification of an Ising EBM that can be trained using sampling-based gradients.

    Defines sampling programs and schedules that allow for collection of the positive and negative phase samples
    required for Monte Carlo estimation of the gradient of the KL-divergence between the model and a data distribution.
    """

    ebm: IsingEBM
    program_positive: IsingSamplingProgram
    program_negative: IsingSamplingProgram
    schedule_positive: SamplingSchedule
    schedule_negative: SamplingSchedule

    def __init__(
        self,
        ebm: IsingEBM,
        data_blocks: list[Block],
        conditioning_blocks: list[Block],
        positive_sampling_blocks: list[SuperBlock],
        negative_sampling_blocks: list[SuperBlock],
        schedule_positive: SamplingSchedule,
        schedule_negative: SamplingSchedule,
    ):
        self.ebm = ebm

        self.program_positive = IsingSamplingProgram(ebm, positive_sampling_blocks, data_blocks + conditioning_blocks)
        self.program_negative = IsingSamplingProgram(ebm, negative_sampling_blocks, conditioning_blocks)

        self.schedule_positive = schedule_positive
        self.schedule_negative = schedule_negative


@eqx.filter_jit
def hinton_init(
    key: Key[Array, ""], model: IsingEBM, blocks: list[Block[AbstractNode]], batch_shape: tuple[int]
) -> list[Bool[Array, "batch_size block_size"]]:
    r"""
    Initialize the blocks according to the marginal bias.

    Each binary unit $i$ in a block is sampled independently as

    $$\mathbb{P}(S_i = 1) = \sigma(\beta h_i) = \frac{1}{1 + e^{-\beta h_i}}$$

    where $h_i$ is the bias of unit *i* and $\beta$ is the
    inverse-temperature scaling factor. See Hinton (2012) for a discussion of this initialization heuristic.

    Arguments:
        key: the JAX PRNG key to use
        model: the Ising model to initialize for
        blocks: the blocks that are to be initialized
        batch_shape: the pre-pended dimension

    Returns:
        the initialized blocks
    """
    node_map = {node: i for i, node in enumerate(model.nodes)}

    data = []
    keys = jax.random.split(key, len(blocks))
    for i, block in enumerate(blocks):
        if len(block) == 0:
            data.append(jnp.zeros((*batch_shape, 0), dtype=jnp.bool_))
            continue

        block_indices = jnp.array([node_map[node] for node in block], dtype=jnp.int32)
        block_biases = model.biases[block_indices]
        probs = jax.nn.sigmoid(model.beta * block_biases)

        block_data = jax.random.bernoulli(keys[i], p=probs, shape=(*batch_shape, len(block))).astype(jnp.bool_)

        data.append(block_data)

    return data


def estimate_moments(
    key: Key[Array, ""],
    first_moment_nodes: list[AbstractNode],
    second_moment_edges: list[Edge],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array],
    clamped_data: list[Array],
):
    """
    Estimates the first and second moments of an Ising model Boltzmann distribution via sampling.

    Arguments:
        key: the jax PRNG key
        first_moment_nodes: the nodes that represent the variables we want to estimate the first moments of
        second_moment_edges: the edges that connect the variables we want to estimate the second moments of
        program: the `BlockSamplingProgram` to be used for sampling
        schedule: the schedule to use for sampling
        init_state: the variable values to use to initialize the sampling
        clamped_data: the variable values to assign to the clamped nodes
    Returns:
        the first and second moment data
    """

    # add a layer of tuple
    first_moments = ((),)
    if len(first_moment_nodes) > 0:
        first_moments = [(node,) for node in first_moment_nodes]

    def _spin_transform(state, _):
        return [2 * x.astype(jnp.int8) - 1 for x in state]

    observer = MomentAccumulatorObserver((first_moments, second_moment_edges), _spin_transform)
    init_mem = observer.init()

    moments, _ = sample_with_observation(key, program, schedule, init_state, clamped_data, init_mem, observer)

    node_sums, edge_sums = moments
    node_moments = node_sums / schedule.n_samples
    edge_moments = edge_sums / schedule.n_samples

    return node_moments, edge_moments


def estimate_kl_grad(
    key: Key[Array, ""],
    training_spec: IsingTrainingSpec,
    bias_nodes: list[AbstractNode],
    weight_edges: list[Edge],
    data: list[Array],
    conditioning_values: list[Array],
    init_state_positive: list[Array],
    init_state_negative: list[Array],
) -> tuple:
    r"""
    Estimate the KL-gradients of an Ising model with respect to its weights and biases.

    Uses the standard two-term Monte Carlo estimator of the gradient of the KL-divergence between an Ising model and
    a data distribution

    The gradients are:

    $$\Delta W = -\beta (\langle s_i s_j \rangle_{+} - \langle s_i s_j \rangle_{-})$$

    $$\Delta b = -\beta (\langle s_i \rangle_{+} - \langle s_i \rangle_{-})$$

    Here, $\langle\cdot\rangle_{+}$ denotes an expectation under the
    *positive* phase (data-clamped Boltzmann distribution) and
    $\langle\cdot\rangle_{-}$ under the *negative* phase (model
    distribution).

    Arguments:
        key: the JAX PRNG key
        training_spec: the Ising EBM for which to estimate the gradients
        bias_nodes: the nodes for which to estimate the bias gradients
        weight_edges: the edges for which to estimate the weight gradients
        data: The data values to use for the positive phase of the gradient estimate. Each array has shape [batch nodes]
        conditioning_values: values to assign to the nodes that the model is conditioned on.
         Each array has shape [nodes]
        init_state_positive: initial state for the positive sampling chain. Each array has
         shape [n_chains_pos batch nodes]
        init_state_negative: initial state for the negative sampling chain. Each array has
         shape [n_chains_neg nodes]
    Returns:
        the weight gradients and the bias gradients
    """

    key_pos, key_neg = jax.random.split(key, 2)

    cond_batched_pos = jax.tree.map(lambda x: jnp.broadcast_to(x, (data[0].shape[0], *x.shape)), conditioning_values)

    keys_pos = jax.random.split(key_pos, init_state_positive[0].shape[:2])

    moms_b_pos, moms_w_pos = jax.vmap(
        lambda k_out, i_out: jax.vmap(
            lambda k, i, c: estimate_moments(
                k, bias_nodes, weight_edges, training_spec.program_positive, training_spec.schedule_positive, i, c
            )
        )(k_out, i_out, data + cond_batched_pos)
    )(keys_pos, init_state_positive)

    keys_neg = jax.random.split(key_neg, init_state_negative[0].shape[0])

    moms_b_neg, moms_w_neg = jax.vmap(
        lambda k, i: estimate_moments(
            k,
            bias_nodes,
            weight_edges,
            training_spec.program_negative,
            training_spec.schedule_negative,
            i,
            conditioning_values,
        )
    )(keys_neg, init_state_negative)

    float_type = training_spec.ebm.beta.dtype
    grad_b = -training_spec.ebm.beta * (
        jnp.mean(moms_b_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_b_neg, axis=0, dtype=float_type)
    )
    grad_w = -training_spec.ebm.beta * (
        jnp.mean(moms_w_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_w_neg, axis=0, dtype=float_type)
    )
    return grad_w, grad_b, (moms_b_pos, moms_w_pos), (moms_b_neg, moms_w_neg)
```

## `thrml/thrml/models/__init__.py`

**`thrml.models` public surface.** Re-exports the discrete factors, EBM bases, and Ising utilities. Import model classes from here (`from thrml.models import IsingEBM, ...`).

```python
from .discrete_ebm import CategoricalEBMFactor as CategoricalEBMFactor
from .discrete_ebm import CategoricalGibbsConditional as CategoricalGibbsConditional
from .discrete_ebm import DiscreteEBMFactor as DiscreteEBMFactor
from .discrete_ebm import DiscreteEBMInteraction as DiscreteEBMInteraction
from .discrete_ebm import SpinEBMFactor as SpinEBMFactor
from .discrete_ebm import SpinGibbsConditional as SpinGibbsConditional
from .discrete_ebm import SquareCategoricalEBMFactor as SquareCategoricalEBMFactor
from .discrete_ebm import SquareDiscreteEBMFactor as SquareDiscreteEBMFactor
from .ebm import AbstractEBM as AbstractEBM
from .ebm import AbstractFactorizedEBM as AbstractFactorizedEBM
from .ebm import EBMFactor as EBMFactor
from .ebm import FactorizedEBM as FactorizedEBM
from .ising import IsingEBM as IsingEBM
from .ising import IsingSamplingProgram as IsingSamplingProgram
from .ising import IsingTrainingSpec as IsingTrainingSpec
from .ising import estimate_kl_grad as estimate_kl_grad
from .ising import estimate_moments as estimate_moments
from .ising import hinton_init as hinton_init
```

## `thrml/thrml/__init__.py`

**`thrml` public surface.** Re-exports every core object. Import from here (`from thrml import Block, SpinNode, sample_states, ...`); never reach into a submodule path.

```python
import importlib.metadata

from . import models as models
from .block_management import Block as Block
from .block_management import BlockSpec as BlockSpec
from .block_management import block_state_to_global as block_state_to_global
from .block_management import from_global_state as from_global_state
from .block_management import get_node_locations as get_node_locations
from .block_management import make_empty_block_state as make_empty_block_state
from .block_management import verify_block_state as verify_block_state
from .block_sampling import BlockGibbsSpec as BlockGibbsSpec
from .block_sampling import BlockSamplingProgram as BlockSamplingProgram
from .block_sampling import SamplingSchedule as SamplingSchedule
from .block_sampling import sample_blocks as sample_blocks
from .block_sampling import sample_single_block as sample_single_block
from .block_sampling import sample_states as sample_states
from .block_sampling import sample_with_observation as sample_with_observation
from .conditional_samplers import AbstractConditionalSampler as AbstractConditionalSampler
from .conditional_samplers import AbstractParametricConditionalSampler as AbstractParametricConditionalSampler
from .conditional_samplers import BernoulliConditional as BernoulliConditional
from .conditional_samplers import SoftmaxConditional as SoftmaxConditional
from .factor import AbstractFactor as AbstractFactor
from .factor import FactorSamplingProgram as FactorSamplingProgram
from .factor import WeightedFactor as WeightedFactor
from .interaction import InteractionGroup as InteractionGroup
from .observers import AbstractObserver as AbstractObserver
from .observers import MomentAccumulatorObserver as MomentAccumulatorObserver
from .observers import StateObserver as StateObserver
from .pgm import AbstractNode as AbstractNode
from .pgm import CategoricalNode as CategoricalNode
from .pgm import SpinNode as SpinNode

__version__ = importlib.metadata.version("thrml")
```