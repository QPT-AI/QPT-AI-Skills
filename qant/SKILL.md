---
name: "qant-native-computing"
description: "Write correct code for the Q.ANT Native Computing Toolkit (photonic NPU inference in bfloat16, Python and C/DLPack). Use for Q.ANT/QANT, photonic NPU, or porting PyTorch models to Q.ANT hardware."
---

# Q.ANT Native Computing Toolkit

The toolkit is the official API for Q.ANT Native Processing Units (NPUs). These are
**photonic** accelerators: matrix products and a periodic optical nonlinearity run in
light, in **bfloat16**. The API is inference only. Every op is a forward pass (`*_fprop`),
nothing is differentiable, and nothing trains on the device. You train in PyTorch, then
run the forward pass on the NPU. Despite the name, Q.ANT is not quantum computing.

When asked to implement something, write the code. No emojis. When you explain a
concept, tie it to the concrete toolkit function that realizes it.

## Import rule

Import the five public submodules and `bfloat16` from `ml_dtypes`. Never import
`_wrapper`, `_backend` or `_utils`.

```python
import numpy as np
from ml_dtypes import bfloat16
from qant_native_computing_toolkit import ai, generic, info, native, utils
from qant_native_computing_toolkit.generic import DataRole
```

`DataRole` lives in `generic`. The docs write `backend.DataRole`, but that module is
private. `get_available_npus` is also in `generic`, not `info`. `ai.linear_fprop`,
`ai.linear_fprop_preloaded` and `ai.linear_fprop_preloaded_weights` are the same
objects as their `native.*` counterparts, so pick one module and use it consistently.

## Installation: hardware vs. cpu-backend

There are two builds with the same API:

| Build | Needs | Use it for |
|-------|-------|------------|
| default | Q.ANT NPU + `qant-native-computing-driver` .deb | real runs, timing, accuracy on hardware |
| `cpu-backend` | nothing | development, CI, unit tests, writing code without hardware |

```bash
# From release artifacts (GitHub releases of Q-ANT-GmbH/qant_native_computing_toolkit)
sudo apt install ./qant-native-computing-driver.<version>.deb     # hardware only
pip install qant_native_computing_toolkit.<version>.whl           # pick the cpu-backend wheel if no hardware

# From source without hardware (submodules are required for dlpack)
git clone --recurse-submodules https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit
cd qant_native_computing_toolkit
pip install "maturin>=1.14,<2" && maturin develop --release -F cpu-backend
# or: MATURIN_PEP517_ARGS="--features cpu-backend" pip install .
```

A plain `pip install .` without hardware fails at link time against
`libqant_native_computing_driver`. That is the most common install error, and the fix is
`-F cpu-backend`. `rust-toolchain.toml` pins a Rust channel. If rustup cannot download it
(offline or proxied), `RUSTUP_TOOLCHAIN=stable maturin develop ...` works.
`generic.get_available_npus()` returns `{0: 'cpu_backend'}` on the simulator, which is the
quickest way to tell which build is installed.

## How the API composes

| Module | Role | Main functions |
|--------|------|----------------|
| `utils` | host buffer prep | `align_ndarray_page_boundary` |
| `generic` | devices and device memory | `get_available_npus`, `init_npu`, `release_npu`, `allocate`, `upload`, `upload_into`, `download`, `DataRole` |
| `native` | primitives that run on the NPU | `linear_fprop` (+ `_preloaded`, `_preloaded_weights`), `mul_elementwise`, `calc_scaled_periodic_nl_fprop` |
| `ai` | NN layers built on the primitives | `conv_fprop`, `conv_transpose_fprop` (+ `upload_*_weights`, `*_preloaded_weights`), `add_bias_fprop`, `batchnorm2d_fprop`, `relu/sigmoid/softmax_fprop`, `maxpool2d/avgpool2d_fprop`, `adaptive_*pool2d_fprop`, `calc_kan_layer_fprop` |
| `info` | diagnostics | `get_driver_info`, `get_version_info`, `get_sensor_info`, `get_perf_counter`, `reset_perf_counter`, `setup_logging` |

Where the work actually runs determines performance and error behaviour:

- **On the NPU (photonic):** `linear_fprop*`, `conv_fprop*` and `conv_transpose_fprop*`
  (both lowered to batched matrix-vector products), `mul_elementwise`,
  `calc_scaled_periodic_nl_fprop`, `calc_kan_layer_fprop`, and the scale step of
  `batchnorm2d_fprop`.
- **On the host CPU, inside the toolkit:** `add_bias_fprop`, `relu_fprop`,
  `sigmoid_fprop`, `softmax_fprop`, all pooling ops, and batchnorm's normalize and shift.

Every non-`preloaded` call takes host numpy arrays and returns host numpy arrays, so each
call is one upload/compute/download round trip. Matrix work is what you are paying the
NPU for. For repeated layers, keep weights on the device (see "Upload once, run many").

## The three rules that every call obeys

**1. Everything is `ml_dtypes.bfloat16`.** Every array argument must be a numpy array with
`dtype == ml_dtypes.bfloat16`, including weights, biases, batchnorm statistics and `ks`.
Anything else raises `TypeError: ... must be of dtype <class 'ml_dtypes.bfloat16'>`.
There is no auto-cast. `np.float16` is not bfloat16, and `torch.bfloat16` tensors are not
numpy. Cast at the boundary and cast back for further math:

```python
x_bf = x_f32.astype(bfloat16)                          # numpy float -> bf16
w_bf = torch_w.detach().cpu().float().numpy().astype(bfloat16)  # torch -> bf16 (go via float32)
y_f32 = y_bf.astype(np.float32)                        # bf16 -> float32 for host-side math
```

bfloat16 keeps float32's range but only 8 bits of mantissa (about 2 to 3 significant
digits). Exact equality is never the right test. See "Testing what you build".

**2. Weight layouts are PyTorch's. Never transpose.**

| Op | `weights` shape | Same as |
|----|-----------------|---------|
| `linear_fprop` | `(n_out, n_in)`, computes `features @ weights.T` | `nn.Linear.weight` |
| `conv_fprop` / `upload_conv_weights` | `(out_ch, in_ch, kh, kw)` | `nn.Conv2d.weight` |
| `conv_transpose_fprop` / `upload_conv_transpose_weights` | `(in_ch, out_ch, kh, kw)` | `nn.ConvTranspose2d.weight` |
| `calc_kan_layer_fprop` `phis`, `ampls` | `(n_out, n_in, len(ks))` | - |

`linear_fprop` is **not** `features @ weights`. Passing `W.T` from a math-style
`(n_in, n_out)` matrix gives a shape error at best and silently wrong results when it is square.

**3. Know which ops accept a batch.** Image layout is channels-first `(C, H, W)` or
`(N, C, H, W)`. The documentation shows 4D inputs everywhere, but several ops reject `N > 1`
at runtime with `RuntimeError: ... does not support batched input.`

| Accepts any batch | Batch must be 1 (pass 3D `(C,H,W)` or `N == 1`) |
|-------------------|-----------------------------------------------|
| `linear_fprop*` (any leading dims), `calc_kan_layer_fprop` (any leading dims), `conv_fprop*`, `add_bias_fprop` (1-4D), `relu_fprop`, `sigmoid_fprop`, `mul_elementwise` (any shape) | `conv_transpose_fprop*`, `batchnorm2d_fprop`, `maxpool2d_fprop`, `avgpool2d_fprop`, `adaptive_maxpool2d_fprop`, `adaptive_avgpool2d_fprop`, `softmax_fprop` (1D, or 2D with one row) |

So a CNN forward pass loops over samples, which is what Q.ANT's own ResNet and U-Net
examples do. Note that `softmax_fprop` normalizes over **all** elements of its input. For
per-row softmax of logits `(B, K)`, call it once per row.

## Quick start: one dense layer, checked against a reference

```python
import numpy as np
from ml_dtypes import bfloat16
from qant_native_computing_toolkit import ai, generic, native

print(generic.get_available_npus())                 # {0: 'cpu_backend'} on the simulator

rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, (8, 64)).astype(bfloat16)     # (batch, n_in)
W = rng.uniform(-0.2, 0.2, (32, 64)).astype(bfloat16)  # (n_out, n_in), as in nn.Linear
b = rng.uniform(-0.1, 0.1, 32).astype(bfloat16)

y = ai.relu_fprop(ai.add_bias_fprop(native.linear_fprop(x, W), b))

ref = np.maximum(x.astype(np.float32) @ W.astype(np.float32).T + b.astype(np.float32), 0)
err = np.abs(y.astype(np.float32) - ref).max()
assert y.dtype == bfloat16 and y.shape == (8, 32) and err < 0.1, err
print(f"[OK] max abs error vs float32 reference = {err:.4f}")
```

## Upload once, run many: device memory and handles

Uploading weights dominates the cost of small calls. Keep them on the device:

| Want | Use |
|------|-----|
| reuse a dense weight matrix, features on host | `h = generic.upload(W, DataRole.Weights)` then `native.linear_fprop_preloaded_weights(x, h)` |
| features, weights and result all resident | `generic.allocate` / `upload_into` plus `native.linear_fprop_preloaded(f_h, w_h, r_h)` |
| reuse conv kernels | `h = ai.upload_conv_weights(K)` then `ai.conv_fprop_preloaded_weights(x, h, padding, stride, 1)` |
| reuse transposed-conv kernels | `ai.upload_conv_transpose_weights` then `ai.conv_transpose_fprop_preloaded_weights` |

Handle facts:

- `QantMemoryHandle` exposes `.rows`, `.cols`, `.shape`, `.npu_id`. Conv handles expose
  `.shape` (4D) and `.npu_id`.
- Device memory is **2D only**. `upload` / `upload_into` take a 2D bfloat16 array, and
  `upload_into` needs exactly `(handle.rows, handle.cols)`, so reshape a single vector to `(1, n)`.
- **Data roles must match the slot.** `linear_fprop_preloaded` expects
  `(Weights, Features, Features)` for (weights, features, result). Swapping them raises
  `Invalid data roles for matrix-vector multiplication`.
- A preloaded op runs **on the NPU that holds the weights**. Its features and result must
  live on that same NPU.
- In Python, memory is freed when the handle is garbage-collected or `del`eted. Keep handles
  alive for the model's lifetime. In C you must free them explicitly (see the C section).
- `linear_fprop_preloaded_weights` still uploads the features on every call. Only the
  fully resident `linear_fprop_preloaded` avoids all per-call allocation.

```python
import numpy as np
from ml_dtypes import bfloat16
from qant_native_computing_toolkit import generic, native
from qant_native_computing_toolkit.generic import DataRole

class ResidentLinear:
    """Weights uploaded once; feature and result buffers allocated once and reused."""
    def __init__(self, W: np.ndarray, max_batch: int = 1, device_id: int = 0):
        n_out, n_in = W.shape
        self.w = generic.upload(W, DataRole.Weights, device_id=device_id)
        self.f = generic.allocate(max_batch, n_in, DataRole.Features, device_id=device_id)
        self.r = generic.allocate(max_batch, n_out, DataRole.Features, device_id=device_id)

    def __call__(self, x: np.ndarray) -> np.ndarray:        # x: (max_batch, n_in) bf16
        generic.upload_into(x, self.f)
        native.linear_fprop_preloaded(self.f, self.w, self.r)
        return generic.download(self.r)                      # (max_batch, n_out) bf16

rng = np.random.default_rng(1)
W = rng.uniform(-0.2, 0.2, (10, 32)).astype(bfloat16)
layer = ResidentLinear(W, max_batch=4)
for _ in range(3):
    x = rng.uniform(-1, 1, (4, 32)).astype(bfloat16)
    y = layer(x)
    ref = x.astype(np.float32) @ W.astype(np.float32).T
    assert np.abs(y.astype(np.float32) - ref).max() < 0.1
print("[OK] resident linear layer")
```

**Page alignment.** The NPU wants host buffers aligned to 4096 bytes. Unaligned input is
copied by the runtime, which is correct but slower. For buffers you reuse in a hot loop,
align them once with `utils.align_ndarray_page_boundary(arr)`. It returns `arr` itself
when the array is already aligned, otherwise an aligned copy. Do not bother for one-off calls.

## Porting a trained PyTorch model (the core workflow)

1. **Train in PyTorch** in float32. Optionally make the model robust to the hardware: Q.ANT's
   own Fourier-layer example adds Gaussian noise (`std = 0.04`) to layer outputs during
   training, and you can round weights to bfloat16 in eval to preview quantization loss.
2. **Export**: for each tensor in `model.state_dict()`, do
   `t.detach().cpu().float().numpy().astype(bfloat16)`. Keep the PyTorch layouts (rule 2).
3. **Map layers**: `nn.Linear` to `linear_fprop` plus `add_bias_fprop`. `nn.Conv2d` to `conv_fprop`
   (`dilation=1`, `groups=1` only) plus `add_bias_fprop`. Eval-mode `nn.BatchNorm2d` to
   `batchnorm2d_fprop(x, running_mean, running_var, weight, bias, eps)`. `ReLU`, `Sigmoid`,
   `MaxPool2d`, `AvgPool2d` and `AdaptiveAvgPool2d` map to their `*_fprop` functions.
   `Flatten` is `x.reshape(-1)` on the host, and residual adds are `+` on bf16 arrays on the host.
   Anything else (GELU, LayerNorm, attention, grouped or dilated conv) runs on the host in
   numpy, between NPU calls.
4. **Upload weights once** at construction (handles), and run a per-sample loop for the
   batch-1 ops (rule 3).
5. **Validate** against the float32 PyTorch model on held-out data. Compare task metrics
   (accuracy, IoU), not only tensors.

```python
import numpy as np
from ml_dtypes import bfloat16
from qant_native_computing_toolkit import ai, generic
from qant_native_computing_toolkit.generic import DataRole

class QantSmallCNN:
    """conv3x3 -> BN -> ReLU -> maxpool2 -> flatten -> linear -> softmax, weights resident on the NPU.
    `sd` maps PyTorch state_dict names to bf16 numpy arrays in PyTorch layout."""
    def __init__(self, sd: dict, device_id: int = 0):
        self.sd = sd
        self.conv = ai.upload_conv_weights(sd["conv.weight"], device_id=device_id)
        self.fc = generic.upload(sd["fc.weight"], DataRole.Weights, device_id=device_id)

    def forward_one(self, x: np.ndarray) -> np.ndarray:      # x: (C, H, W) bf16
        sd = self.sd
        h = ai.conv_fprop_preloaded_weights(x, self.conv, padding=1, stride=1, dilation=1)
        h = ai.add_bias_fprop(h, sd["conv.bias"])
        h = ai.batchnorm2d_fprop(h, sd["bn.running_mean"], sd["bn.running_var"],
                                 sd["bn.weight"], sd["bn.bias"], 1e-5)
        h = ai.relu_fprop(h)
        h = ai.maxpool2d_fprop(h, kernel_size=2, stride=2, padding=0)
        logits = ai.add_bias_fprop(
            ai.linear_fprop_preloaded_weights(h.reshape(1, -1), self.fc), sd["fc.bias"])
        return ai.softmax_fprop(logits)[0]                  # one row -> per-sample softmax

    def __call__(self, batch: np.ndarray) -> np.ndarray:     # (N, C, H, W) bf16
        return np.stack([self.forward_one(x) for x in batch])

# Stand-in for an exported state_dict: {k: v.detach().cpu().float().numpy().astype(bfloat16)}
rng = np.random.default_rng(2)
r = lambda *s, a=0.3: rng.uniform(-a, a, s).astype(bfloat16)
sd = {"conv.weight": r(4, 1, 3, 3), "conv.bias": r(4, a=0.1),
      "bn.running_mean": r(4, a=0.1), "bn.running_var": rng.uniform(0.5, 1.5, 4).astype(bfloat16),
      "bn.weight": r(4, a=1.0), "bn.bias": r(4, a=0.1),
      "fc.weight": r(10, 4 * 4 * 4, a=0.2), "fc.bias": r(10, a=0.1)}
model = QantSmallCNN(sd)
probs = model(rng.uniform(0, 1, (3, 1, 8, 8)).astype(bfloat16))
assert probs.shape == (3, 10) and np.allclose(probs.astype(np.float32).sum(-1), 1, atol=0.06)
print("[OK] CNN forward:", probs.shape)
```

The PyTorch-side pattern from Q.ANT's examples is a module that trains with plain PyTorch
and routes only the eval forward to the NPU. Under `self.training`, return
`F.linear(x, self.weight, self.bias)`. Otherwise convert to bf16 numpy, call
`ai.linear_fprop`, add the bias, and return `torch.from_numpy(out.astype(np.float32))`.
Inputs must be float32 CPU tensors.

## The photonic nonlinearity and KAN / Fourier layers

`native.calc_scaled_periodic_nl_fprop(u, v)` computes `tcos(u) * v` elementwise on 1D
arrays of equal length. `tcos` is the NPU's optical transfer function: 2π-periodic, with
values in [-1, 1], and **cosine-like but not exactly cosine**. `ai.calc_kan_layer_fprop`
builds a Fourier-style KAN layer from it:

```
y[..., j] = sum_{i,l} tcos(ks[l] * x[..., i] + phis[j, i, l]) * ampls[j, i, l]
```

with `features (..., n_in)`, `phis` and `ampls` of shape `(n_out, n_in, len(ks))`, and `ks` 1D.
All are bf16. The frequencies `ks` are usually the integers `1..grid_size`. Train the layer
in PyTorch with `torch.cos` and output noise, then deploy with `calc_kan_layer_fprop`.
Add any bias on the host.

```python
import numpy as np
from ml_dtypes import bfloat16
from qant_native_computing_toolkit import ai

rng = np.random.default_rng(3)
n_in, n_out, G = 3, 2, 4
x = rng.uniform(-1, 1, (5, n_in)).astype(bfloat16)
ks = np.arange(1, G + 1).astype(bfloat16)
phis = rng.uniform(-np.pi / 2, np.pi / 2, (n_out, n_in, G)).astype(bfloat16)
ampls = (rng.standard_normal((n_out, n_in, G)) * 0.3).astype(bfloat16)
y = ai.calc_kan_layer_fprop(x, phis, ampls, ks)               # (5, n_out)

f = lambda a: a.astype(np.float32)
ref = (f(ampls)[None] * np.cos(f(ks)[None, None, None] * f(x)[:, None, :, None] + f(phis)[None])).sum((2, 3))
assert y.shape == (5, n_out) and np.abs(f(y) - ref).max() < 0.3   # upstream tests use atol ~0.27
print("[OK] KAN layer")
```

The `cpu-backend` implements `tcos` as an exact `cos`. Code that passes on the simulator can
therefore be off by the shape difference on hardware. Validate KAN and periodic models on
hardware against the PyTorch float32 model (task metric), and do not tighten tolerances
that were tuned on the simulator.

## Devices, processes and parallelism

- `generic.get_available_npus()` returns `{device_id: serial}`. Every op takes `device_id=`
  (default 0), and preloaded ops use the handle's NPU.
- NPUs initialise lazily on first use. Calling `init_npu(id)` is optional. A process holds an
  **exclusive lock** on each NPU it initialised until `release_npu(id)` or process exit. A second
  process cannot use that NPU in the meantime. Release explicitly in long-running services and notebooks.
- Within one process, concurrent calls to one NPU queue first-come, first-served.
- To use several NPUs, follow Q.ANT's `parallel_execution` example: one worker process per NPU,
  and pass large bf16 arrays through `multiprocessing.shared_memory` rather than pickling them.

## Diagnostics and performance

```python
from qant_native_computing_toolkit import info
info.reset_perf_counter(0)
# ... run workload on device 0 ...
pc = info.get_perf_counter(0)            # {'timebased_counter': s, 'stalling_counter': s}
busy = 1 - pc["stalling_counter"] / max(pc["timebased_counter"], 1e-12)
print(info.get_driver_info(0), info.get_version_info(0), info.get_sensor_info(0))
```

`stalling_counter` is idle time. A low busy fraction means the host (uploads, host-side
ops, Python loops) is the bottleneck, and preloading or batching the matrix ops is the fix.
`get_sensor_info` reports photodiode and DAC temperatures and rail voltages, so log it alongside
accuracy when results drift. `setup_logging(folder, loglevel)` (1 = debug ... 4 = error) writes
driver logs. It raises `RuntimeError` on the cpu-backend.

## C / C++ API

The C API (`#include "qant_native_computing_toolkit.h"`, link `libqant_native_computing_toolkit`)
mirrors Python with these differences:

- Tensors are `DLManagedTensorVersioned*` (DLPack, `kDLBfloat`, 16 bits, `kDLCPU`). Shapes are
  fixed rank: C `linear_fprop` takes 2D features `(n_batches, n_in)`. There is no ND flattening
  and no 3D-to-4D promotion, which the Python wrapper does for you.
- `npu_id` is the **first** argument (Python: `device_id=` last).
- Names differ: `mul_npu` (Python `mul_elementwise`, 1D only), and `npu_allocate`, `npu_upload`,
  `npu_upload_into`, `npu_download`, `npu_free` (Python `generic.allocate` and so on).
  `add_bias_for_conv2d_fprop` is the 4D variant that Python dispatches to automatically.
- In C++ the functions live in `qant_native_computing_toolkit::{generic,native,ai,info}`.
  In C they are plain `extern "C"` symbols.
- **Ownership:** a returned `DLManagedTensorVersioned*` is yours, so call
  `t->deleter(t)` when you are done. Handles from `npu_allocate` / `npu_upload` need exactly one
  `npu_free`. Conv handles need `conv_weights_free` / `conv_transpose_weights_free`.
- **Errors:** pointer-returning functions return `NULL`, and int-returning ones return a negative
  code. Check every call.
- C `linear_fprop_preloaded(features, weights, result)` returns `int` and **requires a
  pre-allocated** result handle (role `Features`, same NPU). It does not allocate one as Python does.

## Testing what you build

Analog bf16 hardware fails quietly. A wrong layout or a transposed weight still produces
plausible numbers, so test against references:

- **Develop on `cpu-backend`, then confirm on hardware.** Same code, same API. Put
  `generic.get_available_npus()` in your logs so every result records which backend produced it.
- **Compare against float32**, using tolerances like the upstream tests: linear/conv about
  `atol 0.01-0.2` or `rtol 0.05` depending on size, sigmoid `atol 0.005`, batchnorm `atol 0.015`,
  softmax sums to 1 within about `0.06`, KAN `atol 0.27`. Error grows with the reduction length
  `n_in`, so scale inputs to O(1) and check mean error as well as max error.
- **Separate quantization from hardware error.** Also compute the reference from the bf16-rounded
  inputs and weights cast to float32. The gap between that and the full-precision reference is
  bf16 rounding, and anything beyond it is the device.
- **Test shapes, not just values**: an unbatched `(C,H,W)` input returns `(C',H',W')`, ND linear
  inputs keep their leading dims, and a 1D vector into `linear_fprop` returns a 1D vector.
- For deployed models, the acceptance test is the **task metric** on held-out data versus the
  PyTorch float32 model.

## Failure checks

| Symptom | Cause and fix |
|---------|---------------|
| `ImportError: Failed to import the Q.ANT Toolkit ... Installed driver version: not installed` | hardware build without a driver. Install the driver .deb or use the cpu-backend build |
| linker error mentioning `libqant_native_computing_driver` during `pip install .` | build with `-F cpu-backend` |
| `TypeError: ... must be of dtype <class 'ml_dtypes.bfloat16'>` | cast with `.astype(bfloat16)`. This applies to BN stats, biases and `ks` too |
| `ValueError: ... must have dimension N` | `upload`/`upload_into` need 2D, `calc_scaled_periodic_nl_fprop` needs 1D, conv/pool/BN need 3D or 4D |
| `number of features (...) != number of weights input channels` | weights are `(n_out, n_in)`, so you passed the transpose |
| `does not support batched input.` | batch-1 op (rule 3). Loop over samples |
| `dilation != 1 not implemented yet` / `output_padding != 0 not implemented yet` | not supported. Use dilation 1 and output_padding 0, or run that layer on the host |
| `stride must be > 0` | stride 0 is invalid. Padding, stride and kernel sizes must be Python `int`s (`kernel_size` may be an `(h, w)` tuple), and padding is symmetric |
| `input must be symmetric` / `input size must be multiple integer of output size` | adaptive pooling needs square H == W and H % output_size == 0 |
| `pool window ... contains no real pixels` | padding is at least the kernel size, so reduce the padding |
| `Invalid data roles for matrix-vector multiplication` | upload weights with `DataRole.Weights`, and features and result with `DataRole.Features` |
| softmax output does not sum to 1 per row | `softmax_fprop` normalizes the whole input, so call it per row |
| second process hangs or fails to acquire the NPU | another process holds the lock, so call `release_npu(id)` there |
| `logging not available for cpu-backend` | expected on the simulator |

## Upstream references

- Python and C API docs: https://q-ant-gmbh.github.io/qant_native_computing_toolkit/
- Source, releases, CONTRIBUTING (build without hardware): https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit
- Official examples (intro, memory management, digit recognition, ResNet-18 with and without
  resident weights, U-Net segmentation, Fourier/KAN function learning, multi-NPU parallel
  execution, C++ mat-vec and perf measurement): https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit_examples
- DLPack: https://github.com/dmlc/dlpack
