from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.utils import get_default_device


class LayerNormSingleLayerTranscoder(SingleLayerTranscoder):
    """Adapter for local trainer checkpoints that apply an input LayerNorm before encoding.

    This preserves checkpoint behavior for activation computation and interventions.
    Attribution through the pre-encoder LayerNorm is an approximation, because the encoder
    vectors returned by `encode_sparse` do not include the full input-dependent LayerNorm
    Jacobian.
    """

    def __init__(
        self,
        d_model: int,
        d_transcoder: int,
        activation_function,
        layer_idx: int,
        *,
        use_input_ln: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            d_model=d_model,
            d_transcoder=d_transcoder,
            activation_function=activation_function,
            layer_idx=layer_idx,
            skip_connection=False,
            device=device,
            dtype=dtype,
        )
        if use_input_ln:
            self.ln_weight = torch.nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))
            self.ln_bias = torch.nn.Parameter(torch.zeros(d_model, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter("ln_weight", None)
            self.register_parameter("ln_bias", None)

    def _normalize_input(self, input_acts: torch.Tensor) -> torch.Tensor:
        if self.ln_weight is None and self.ln_bias is None:
            return input_acts
        return F.layer_norm(
            input_acts.to(self.dtype),
            (self.d_model,),
            self.ln_weight,
            self.ln_bias,
        )

    def encode(self, input_acts, apply_activation_function: bool = True):
        normalized = self._normalize_input(input_acts)
        return super().encode(normalized, apply_activation_function=apply_activation_function)

    def encode_sparse(self, input_acts, zero_positions: slice = slice(0, 1)):
        normalized = self._normalize_input(input_acts)
        return super().encode_sparse(normalized, zero_positions=zero_positions)

    def to_safetensors(self, save_path: str):
        state_dict = {
            "W_enc": self.W_enc.cpu(),
            "W_dec": self.W_dec.cpu(),
            "b_enc": self.b_enc.cpu(),
            "b_dec": self.b_dec.cpu(),
        }
        if self.ln_weight is not None:
            state_dict["ln_weight"] = self.ln_weight.cpu()
        if self.ln_bias is not None:
            state_dict["ln_bias"] = self.ln_bias.cpu()
        from safetensors.torch import save_file

        save_file(state_dict, save_path)


def _placeholder_transcoder(
    layer: int,
    d_model: int,
    d_transcoder: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_input_ln: bool = True,
) -> LayerNormSingleLayerTranscoder:
    transcoder = LayerNormSingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=F.relu,
        layer_idx=layer,
        use_input_ln=use_input_ln,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        transcoder.W_enc.zero_()
        transcoder.W_dec.zero_()
        transcoder.b_enc.zero_()
        transcoder.b_dec.zero_()
        if transcoder.ln_weight is not None:
            transcoder.ln_weight.fill_(1.0)
        if transcoder.ln_bias is not None:
            transcoder.ln_bias.zero_()
    return transcoder


def load_local_qwen_plt_checkpoint(
    path: str,
    *,
    layer: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> LayerNormSingleLayerTranscoder:
    if device is None:
        device = get_default_device()

    obj = torch.load(path, map_location="cpu")
    state_dict = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    hidden_dim = int(obj.get("hidden_dim", state_dict["decoder.bias"].shape[0]))
    feature_dim = int(obj.get("feature_dim", state_dict["encoder.bias"].shape[0]))
    layer_idx = int(obj.get("layer", layer if layer is not None else 0))
    use_input_ln = "input_ln.weight" in state_dict or "input_ln.bias" in state_dict

    transcoder = LayerNormSingleLayerTranscoder(
        d_model=hidden_dim,
        d_transcoder=feature_dim,
        activation_function=F.relu,
        layer_idx=layer_idx,
        use_input_ln=use_input_ln,
        device=device,
        dtype=dtype,
    )

    adapted_state = {
        "W_enc": state_dict["encoder.weight"].to(device=device, dtype=dtype),
        "W_dec": state_dict["decoder.weight"].T.contiguous().to(device=device, dtype=dtype),
        "b_enc": state_dict["encoder.bias"].to(device=device, dtype=dtype),
        "b_dec": state_dict["decoder.bias"].to(device=device, dtype=dtype),
    }
    if use_input_ln:
        adapted_state["ln_weight"] = state_dict["input_ln.weight"].to(device=device, dtype=dtype)
        adapted_state["ln_bias"] = state_dict["input_ln.bias"].to(device=device, dtype=dtype)

    transcoder.load_state_dict(adapted_state, strict=False)
    return transcoder


def load_local_qwen_plt_transcoder_set(
    checkpoint_dir: str,
    *,
    n_layers: int = 32,
    feature_input_hook: str = "mlp.hook_in",
    feature_output_hook: str = "mlp.hook_out",
    scan: str | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    allow_missing: bool = True,
) -> TranscoderSet:
    if device is None:
        device = get_default_device()

    checkpoint_root = Path(checkpoint_dir)
    checkpoints: dict[int, str] = {}
    for layer in range(n_layers):
        best_path = checkpoint_root / f"transcoder_L{layer}_best.pt"
        plain_path = checkpoint_root / f"transcoder_L{layer}.pt"
        if best_path.exists():
            checkpoints[layer] = str(best_path)
        elif plain_path.exists():
            checkpoints[layer] = str(plain_path)

    if not checkpoints:
        raise FileNotFoundError(f"No local Qwen PLT checkpoints found in {checkpoint_dir}")

    probe = load_local_qwen_plt_checkpoint(next(iter(checkpoints.values())), device=device, dtype=dtype)
    d_model = probe.d_model
    d_transcoder = probe.d_transcoder

    transcoders = {}
    for layer in range(n_layers):
        ckpt = checkpoints.get(layer)
        if ckpt is not None:
            transcoders[layer] = load_local_qwen_plt_checkpoint(
                ckpt,
                layer=layer,
                device=device,
                dtype=dtype,
            )
            continue

        if not allow_missing:
            raise FileNotFoundError(
                f"Missing local Qwen PLT checkpoint for layer {layer} in {checkpoint_dir}"
            )

        warnings.warn(
            f"Missing local Qwen PLT checkpoint for layer {layer}; using a zero placeholder.",
            UserWarning,
        )
        transcoders[layer] = _placeholder_transcoder(
            layer,
            d_model=d_model,
            d_transcoder=d_transcoder,
            device=device,
            dtype=dtype,
        )

    return TranscoderSet(
        transcoders,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        scan=scan or f"local-qwen35-plt:{os.path.abspath(checkpoint_dir)}",
    )
