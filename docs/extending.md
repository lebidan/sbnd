# Extending SBND

This document collects the conventions and contracts to follow when adding new components to the SBND codebase. The first section covers the most common case — implementing a new decoder architecture. Future sections will cover other extension points (e.g. registering a new automorphism family for training-time augmentation and test-time augmentation) as the corresponding interfaces stabilize.

**See also:** [README](../README.md) · [Training a model](training.md) · [Evaluating a model](evaluation.md)

## Contents

1. [Adding a decoder architecture](#1-adding-a-decoder-architecture)
   - [The `BaseDecoder` contract](#the-basedecoder-contract)
   - [Conventions](#conventions)
   - [Walk-through: the mocked decoder](#walk-through-the-mocked-decoder)
   - [Wiring the decoder into an experiment](#wiring-the-decoder-into-an-experiment)
2. [Future extension points](#2-future-extension-points)

## 1. Adding a decoder architecture

SBND decoders share a common abstract base class, [`BaseDecoder`](../src/decoder.py), which fixes the input/output contract and centralizes a small amount of boilerplate (output-size derivation, the example input array used by Lightning's model summary, optional `torch.compile` activation). To add a new architecture, subclass `BaseDecoder`, implement `forward(ym, s)`, and follow the conventions below.

The minimal working example shipped with SBND is [`MockedDecoder`](../src/mocked.py): a single linear layer mapping the concatenation of the input magnitude and syndrome vectors to the predicted error pattern. We recommend using it as a starting template.

### The `BaseDecoder` interface

A decoder consumes the channel-matched magnitude vector `ym` of shape `(B, n)` and the bipolar syndrome `s` of shape `(B, m)`, and returns LLR-like logits for the predicted error pattern. The output shape depends on the `error_space` argument — see below.

**Constructor arguments** inherited from `BaseDecoder`:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `code` | `LinearCode` | — (required) | The code whose errors the decoder is trained to predict. Used to derive `code.n` (length), `code.m` (number of parity-check equations), and `code.k` (message length). |
| `error_space` | `str` | `"codeword"` | `"codeword"` → output shape `(B, n)`, predicting the full n-bit error pattern `e_cw = ĉ ⊕ c`. `"message"` → output shape `(B, k)`, predicting the k-bit error pattern `e_msg = (G⁻¹·e_cw) mod 2` directly in the message space. The value MUST match the datamodule's `error_space`; a mismatch is caught at `trainer.fit` start. |
| `compile` | `bool` | `False` | If `True`, `self.compile()` is invoked by `_maybe_compile()` once the module is fully constructed, producing a traced graph for faster training. |

**Attributes set by the base class** (do not override):

| Attribute | Description |
| --- | --- |
| `self.error_space` | as passed in |
| `self.output_sz` | `code.n` if `error_space == "codeword"`, else `code.k` — use this to size the final projection layer |
| `self.example_input_array` | a `(zeros(1, n), zeros(1, m))` tuple, consumed by Lightning for shape inference in the model summary |

**Required override:**

```python
@abstractmethod
def forward(self, ym: Tensor, s: Tensor) -> Tensor: ...
```

The output is interpreted as logits in bipolar convention: a negative value at position `i` means "bit `i` is predicted to be 1 (in error)". This matches the loss computed by [`SBNDLitModule`](../src/model.py).

### Conventions

* **Call `self._maybe_compile()` LAST** in `__init__`, after every parameter, buffer, and submodule has been registered. `_maybe_compile()` is a no-op when `compile=False`; when `compile=True`, it captures the traced graph of the fully-constructed module. Calling it earlier traces an incomplete graph.
* **Size your output projection from `self.output_sz`**, not from `code.n` or `code.k` directly. This is what allows the same decoder class to be used in both codeword-level and message-level (iSBND) modes — see [Decoding modes](../README.md#decoding-modes).
* **Do not mutate `code`.** It is a shared object; treat it as read-only.

### Walk-through: the mocked decoder

The shipped [`MockedDecoder`](../src/mocked.py) is the minimal compliant implementation:

```python
class MockedDecoder(BaseDecoder):

    def __init__(
        self,
        code: LinearCode,
        error_space: str = "codeword",
        compile: bool = False,
    ) -> None:
        super().__init__(code, error_space=error_space, compile=compile)

        # --- replace this block with your architecture ---
        self.fc = nn.Linear(code.n + code.m, self.output_sz)
        # -------------------------------------------------

        # call last (compiles the forward graph once all submodules exist)
        self._maybe_compile()

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        # --- replace this block with your architecture ---
        x = torch.cat((ym, s), dim=1)
        return self.fc(x)
        # -------------------------------------------------
```

The two annotated regions are the only places you need to modify. Everything else — the `error_space` plumbing, the example input array, the optional compilation — is inherited.

### Wiring the decoder into an experiment

Once your decoder class is importable as e.g. `sbnd.my_decoder.MyDecoder`, point a `decoder:` block in an experiment config (under [`conf/exp/`](../conf/exp)) at it via Hydra's `_target_`:

```yaml
decoder:
  _target_: sbnd.my_decoder.MyDecoder
  # constructor kwargs other than `code`, `error_space`, `compile`
  embed_dim: 128
  n_layers: 6
  attn_dropout: 0.1
  compile: true
```

The `code`, `error_space`, and `compile` arguments are passed through automatically by [`SBNDLitModule`](../src/model.py); the rest of the block is forwarded to your `__init__`. See any of the existing experiment configs in [`conf/exp/`](../conf/exp) for working examples of the `decoder:` block.

For a quick smoke test, drop a minimal experiment config alongside the existing ones, copying the pattern of [`conf/exp/dev-test-mocked.yaml`](../conf/exp/dev-test-mocked.yaml), and run:

```
sbnd-train exp=my-experiment max_epochs=4
```

A run that completes without raising and reaches non-trivial validation accuracy is a good first sign that the decoder is correctly wired.

## 2. Future extension points

The following extension surfaces are stable in the codebase but not yet documented in detail here. They will be covered in a future revision of this guide:

* **Adding a code automorphism family** — implementing a new permutation class in [`src/transforms.py`](../src/transforms.py), used both for training-time data augmentation and for [test-time augmentation](evaluation.md#test-time-augmentation). The contract is small: subclass `BasePerms`, populate `self.perms` (shape `(n_perms, code.n)`) and `self.n_perms`, and provide a `__call__(y, e) → (yp, ep)` method for the training-augmentation path. The `sample_perms(bs) → (perms, perms_inv)` method used by the TTA path is inherited from `BasePerms`.
* **Adding a TTS variant** — implementing a new decoding strategy in [`src/tts.py`](../src/tts.py). The protocol is `decode(model, code, ym, syndromes, targets) → preds`, plus the `name`, `suffix`, and `validate(model, code)` attributes.
* **Adding a custom dataset format** — extending [`SBNDDataModule`](../src/data.py) to load datasets with a non-standard layout.

Contributions in any of these areas are welcome — see the [Contributing](../README.md#-contributing) section in the README.
