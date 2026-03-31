from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_NEUTRAL_PROMPT = "Describe what you see in this image."


@dataclass
class PromptCache:
    image_np: np.ndarray
    visual_positions: torch.Tensor
    decision_pos: int
    grid_h: int
    grid_w: int
    mlp_inputs: dict[int, torch.Tensor]
    generated_text: str
    first_token_text: str


def get_child_module(module, name: str):
    if module is None:
        return None
    modules = getattr(module, "_modules", None)
    if isinstance(modules, dict) and name in modules:
        return modules[name]
    try:
        return getattr(module, name)
    except AttributeError:
        return None


def get_model_input_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    hf_device_map = getattr(model, "hf_device_map", None) or {}
    for _, device in hf_device_map.items():
        if isinstance(device, str) and device.startswith("cuda"):
            return torch.device(device)
    return torch.device("cpu")


def get_text_model(model):
    outer_model = get_child_module(model, "model")
    text_model = get_child_module(outer_model, "language_model")
    if text_model is None:
        text_model = get_child_module(model, "language_model")
    if text_model is None:
        raise RuntimeError("Could not locate Qwen language model stack.")
    return text_model


def get_text_layers(model):
    text_model = get_text_model(model)
    layers = get_child_module(text_model, "layers")
    if layers is None:
        inner_model = get_child_module(text_model, "model")
        layers = get_child_module(inner_model, "layers")
    if layers is None:
        raise RuntimeError("Could not locate decoder layers for Qwen language model.")
    return layers


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device | str):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def build_prompt(processor, prompt: str, image_path: Path) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prepare_inputs(processor, image_path: Path, prompt: str):
    image = Image.open(image_path).convert("RGB")
    rendered = build_prompt(processor, prompt, image_path)
    batch = processor(
        text=[rendered],
        images=[str(image_path)],
        padding=True,
        return_tensors="pt",
    )
    return image, batch


def official_generation_config(max_new_tokens: int) -> dict:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
    }


def find_image_token_id(model, processor) -> int:
    token_id = getattr(model.config, "image_token_id", None)
    if token_id is not None:
        return int(token_id)
    tokenizer = processor.tokenizer
    for candidate in ("<|image_pad|>", "<image>", "<img>"):
        maybe = tokenizer.convert_tokens_to_ids(candidate)
        if maybe is not None and maybe != tokenizer.unk_token_id:
            return int(maybe)
    raise RuntimeError("Could not determine image token id.")


def compute_visual_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    visual_positions = torch.where(input_ids == image_token_id)[0]
    if visual_positions.numel() == 0:
        raise RuntimeError("No visual tokens found in the rendered prompt.")
    return visual_positions


def get_attention_capable_layers(model) -> list[int]:
    return [idx for idx, layer in enumerate(get_text_layers(model)) if hasattr(layer, "self_attn")]


def visual_grid_from_image_grid(image_grid_thw: torch.Tensor, model) -> tuple[int, int]:
    grid = image_grid_thw[0] if image_grid_thw.ndim == 2 else image_grid_thw
    _t, h, w = [int(x) for x in grid.tolist()]
    vision_config = getattr(model.config, "vision_config", None)
    merge_size = getattr(vision_config, "spatial_merge_size", 2) or 2
    return h // int(merge_size), w // int(merge_size)


def aggregate_attention_for_layer(
    layer_attn: torch.Tensor,
    query_token_idx: int,
    visual_positions: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    attn_slice = layer_attn[0, :, query_token_idx, visual_positions]
    per_visual = attn_slice.float().mean(dim=0)
    expected = grid_h * grid_w
    if per_visual.numel() > expected:
        per_visual = per_visual[:expected]
    if per_visual.numel() != expected:
        raise RuntimeError(f"Visual token count {per_visual.numel()} does not match grid {grid_h}x{grid_w}.")
    return per_visual.view(grid_h, grid_w)


def upsample_heatmap(grid: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    upsampled = F.interpolate(
        grid.unsqueeze(0).unsqueeze(0).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    heatmap = upsampled.cpu().numpy().astype(np.float32)
    heatmap = np.maximum(heatmap, 0.0)
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def save_overlay(
    image_np: np.ndarray,
    heatmap_raw: np.ndarray,
    output_path: Path,
    vmax: float,
    *,
    colorbar_label: str,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.imshow(image_np)
    overlay = ax.imshow(np.maximum(heatmap_raw, 0.0), cmap="turbo", alpha=0.5, vmin=0.0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(colorbar_label, rotation=90)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def capture_mlp_inputs(model, batch: dict[str, torch.Tensor], layers: list[int]) -> dict[int, torch.Tensor]:
    text_layers = get_text_layers(model)
    cache: dict[int, torch.Tensor] = {}
    hooks = []

    for layer in layers:
        mlp = text_layers[layer].mlp

        def pre_hook(_module, inputs, layer_idx=layer):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            cache[layer_idx] = x.detach().float().cpu()

        hooks.append(mlp.register_forward_pre_hook(pre_hook))

    with torch.no_grad():
        model(**batch)

    for hook in hooks:
        hook.remove()
    return cache


def capture_decode_step_attentions(model, batch: dict[str, torch.Tensor], layers: list[int], gen_cfg: dict):
    text_layers = get_text_layers(model)
    captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            attn_weights = None
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
            elif hasattr(output, "attn_weights"):
                attn_weights = output.attn_weights
            if attn_weights is not None:
                captured[layer_idx].append(attn_weights.detach().cpu())

        return hook

    for layer_idx in layers:
        hooks.append(getattr(get_text_layers(model)[layer_idx], "self_attn").register_forward_hook(make_hook(layer_idx)))

    try:
        local_cfg = dict(gen_cfg)
        local_cfg["max_new_tokens"] = 1
        with torch.no_grad():
            outputs = model.generate(**batch, **local_cfg)
    finally:
        for hook in hooks:
            hook.remove()

    return outputs, {layer_idx: values[-1] for layer_idx, values in captured.items() if values}


def compute_next_token_probs(model, processor, batch: dict[str, torch.Tensor], top_k: int = 5) -> list[dict]:
    with torch.no_grad():
        outputs = model(**batch, use_cache=False, return_dict=True)
    probs = torch.softmax(outputs.logits[0, -1].float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]))
    items = []
    for prob, token_id in zip(top_probs.tolist(), top_ids.tolist(), strict=False):
        items.append(
            {
                "token_id": int(token_id),
                "token_text": processor.tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "prob": float(prob),
            }
        )
    return items


def generate_outputs(processor, model, image_path: Path, prompt: str, seed: int) -> tuple[str, str]:
    _image, raw_batch = prepare_inputs(processor, image_path, prompt)
    batch = move_batch_to_device(raw_batch, get_model_input_device(model))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        generated = model.generate(**batch, **official_generation_config(max_new_tokens=50))
    prompt_len = int(batch["input_ids"].shape[1])
    new_ids = generated[0, prompt_len:]
    full_text = processor.tokenizer.decode(new_ids, skip_special_tokens=False)
    if new_ids.numel() == 0:
        return full_text, ""
    first_token_text = processor.tokenizer.decode([int(new_ids[0].item())], skip_special_tokens=False)
    return full_text, first_token_text


def prepare_prompt_cache(
    processor,
    model,
    image_path: Path,
    prompt: str,
    image_token_id: int,
    plt_layers: list[int],
    seed: int,
) -> PromptCache:
    image, raw_batch = prepare_inputs(processor, image_path, prompt)
    image_np = np.array(image)
    batch = move_batch_to_device(raw_batch, get_model_input_device(model))
    input_ids = batch["input_ids"][0].detach().cpu()
    visual_positions = compute_visual_positions(input_ids, image_token_id).detach().cpu()
    decision_pos = int(batch["input_ids"].shape[1] - 1)
    grid_h, grid_w = visual_grid_from_image_grid(batch["image_grid_thw"][0].detach().cpu(), model)
    mlp_inputs = capture_mlp_inputs(model, batch, plt_layers)
    generated_text, first_token_text = generate_outputs(processor, model, image_path, prompt, seed)
    return PromptCache(
        image_np=image_np,
        visual_positions=visual_positions,
        decision_pos=decision_pos,
        grid_h=grid_h,
        grid_w=grid_w,
        mlp_inputs=mlp_inputs,
        generated_text=generated_text,
        first_token_text=first_token_text,
    )


def build_first_token_attention_artifacts(
    processor,
    model,
    image_path: Path,
    prompt: str,
    image_token_id: int,
    attention_dir: Path,
    seed: int,
) -> dict:
    image, raw_batch = prepare_inputs(processor, image_path, prompt)
    image_np = np.array(image)
    batch = move_batch_to_device(raw_batch, get_model_input_device(model))
    input_len = int(batch["input_ids"].shape[1])
    visual_positions = compute_visual_positions(batch["input_ids"][0], image_token_id).detach().cpu()
    grid_h, grid_w = visual_grid_from_image_grid(batch["image_grid_thw"][0].detach().cpu(), model)
    layers = get_attention_capable_layers(model)
    outputs, attentions = capture_decode_step_attentions(model, batch, layers, official_generation_config(50))
    first_token_id = int(outputs[0, input_len].item())
    first_token_text = processor.tokenizer.decode([first_token_id], skip_special_tokens=False)
    top_probs = compute_next_token_probs(model, processor, batch)

    per_layer = {}
    positive_maps = []
    for layer in layers:
        layer_attn = attentions[layer]
        query_token_idx = 0 if layer_attn.shape[2] == 1 else int(layer_attn.shape[2] - 1)
        grid = aggregate_attention_for_layer(layer_attn, query_token_idx, visual_positions, grid_h, grid_w)
        heatmap = upsample_heatmap(grid, image.size)
        per_layer[layer] = heatmap
        positive_maps.append(np.maximum(heatmap, 0.0))

    all_layers_heat = np.mean(np.stack([per_layer[layer] for layer in layers], axis=0), axis=0)
    all_layers_heat = np.maximum(all_layers_heat, 0.0)
    positive_maps.append(all_layers_heat)
    stacked_positive = np.stack(positive_maps, axis=0)
    vmax = float(np.quantile(stacked_positive, 0.995)) if stacked_positive.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    overlay_paths = {}
    all_path = attention_dir / "first_token_attention_all.png"
    save_overlay(image_np, all_layers_heat, all_path, vmax, colorbar_label="First-token attention")
    overlay_paths["all"] = str(all_path)
    for layer in layers:
        layer_path = attention_dir / f"first_token_attention_L{layer}.png"
        save_overlay(image_np, per_layer[layer], layer_path, vmax, colorbar_label="First-token attention")
        overlay_paths[str(layer)] = str(layer_path)

    return {
        "first_token_text": first_token_text,
        "top_token_probs": top_probs,
        "available_layers": layers,
        "default_key": "all",
        "overlay_paths": overlay_paths,
    }


def parse_graph_feature_keys(graph_json_path: Path) -> list[tuple[int, int]]:
    graph = json.loads(graph_json_path.read_text())
    feature_keys = set()
    for node in graph.get("nodes", []):
        if node.get("feature_type") != "cross layer transcoder":
            continue
        parts = str(node.get("node_id", "")).split("_")
        if len(parts) >= 2:
            feature_keys.add((int(parts[0]), int(parts[1])))
    return sorted(feature_keys)


def discover_transcoder_layers(transcoder_dir: Path) -> list[int]:
    return [
        layer
        for layer in range(128)
        if (transcoder_dir / f"transcoder_L{layer}_best.pt").exists()
        or (transcoder_dir / f"transcoder_L{layer}.pt").exists()
    ]


def load_plt_encoder_payload(transcoder_dir: Path, layer: int) -> dict:
    best_path = transcoder_dir / f"transcoder_L{layer}_best.pt"
    plain_path = transcoder_dir / f"transcoder_L{layer}.pt"
    path = best_path if best_path.exists() else plain_path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    return {
        "input_ln_weight": state_dict["input_ln.weight"].float(),
        "input_ln_bias": state_dict["input_ln.bias"].float(),
        "encoder_weight": state_dict["encoder.weight"].float(),
        "encoder_bias": state_dict["encoder.bias"].float(),
    }


def compute_plt_activations(hidden_states: torch.Tensor, plt_payload: dict) -> torch.Tensor:
    normalized = F.layer_norm(
        hidden_states.float(),
        normalized_shape=(hidden_states.shape[-1],),
        weight=plt_payload["input_ln_weight"],
        bias=plt_payload["input_ln_bias"],
    )
    pre = F.linear(normalized, plt_payload["encoder_weight"], plt_payload["encoder_bias"])
    return torch.relu(pre)


def upsample_raw_grid(grid: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    upsampled = F.interpolate(
        grid.unsqueeze(0).unsqueeze(0).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return upsampled.cpu().numpy().astype(np.float32)


def build_feature_heatmaps(
    feature_keys: list[tuple[int, int]],
    plt_payloads: dict[int, dict],
    spatial_cache: PromptCache,
    neutral_cache: PromptCache,
    heatmap_dir: Path,
) -> dict[str, dict]:
    active_layers = sorted({layer for layer, _ in feature_keys if layer in plt_payloads})
    spatial_by_layer = {layer: compute_plt_activations(spatial_cache.mlp_inputs[layer], plt_payloads[layer]) for layer in active_layers}
    neutral_by_layer = {layer: compute_plt_activations(neutral_cache.mlp_inputs[layer], plt_payloads[layer]) for layer in active_layers}

    raw_entries: list[dict] = []
    for layer, feature_idx in feature_keys:
        if layer not in spatial_by_layer:
            continue
        spatial_acts = spatial_by_layer[layer]
        neutral_acts = neutral_by_layer[layer]
        spatial_decision = float(spatial_acts[0, spatial_cache.decision_pos, feature_idx].item())
        neutral_decision = float(neutral_acts[0, neutral_cache.decision_pos, feature_idx].item())
        spatial_visual = spatial_acts[0, spatial_cache.visual_positions, feature_idx]
        spatial_heat = upsample_raw_grid(
            spatial_visual.view(spatial_cache.grid_h, spatial_cache.grid_w),
            (spatial_cache.image_np.shape[1], spatial_cache.image_np.shape[0]),
        )
        raw_entries.append(
            {
                "layer": layer,
                "feature_idx": feature_idx,
                "decision_activation": spatial_decision,
                "neutral_activation": neutral_decision,
                "decision_delta": spatial_decision - neutral_decision,
                "spatial_heat_raw": spatial_heat,
            }
        )

    if not raw_entries:
        return {}

    positive_stack = np.stack([np.maximum(row["spatial_heat_raw"], 0.0) for row in raw_entries], axis=0)
    vmax = float(np.quantile(positive_stack, 0.995))
    if vmax <= 0:
        vmax = 1.0

    top_diff = max(raw_entries, key=lambda row: row["decision_delta"])
    metadata_map: dict[str, dict] = {}
    for row in raw_entries:
        key = f"{row['layer']}:{row['feature_idx']}"
        overlay_path = heatmap_dir / f"L{row['layer']}_F{row['feature_idx']}_spatial.png"
        save_overlay(
            spatial_cache.image_np,
            row["spatial_heat_raw"],
            overlay_path,
            vmax,
            colorbar_label="Feature activation",
        )
        metadata_map[key] = {
            "layer": row["layer"],
            "feature_idx": row["feature_idx"],
            "overlay_path": str(overlay_path),
            "decision_activation": row["decision_activation"],
            "neutral_activation": row["neutral_activation"],
            "decision_delta": row["decision_delta"],
            "kind": "spatial_bias_candidate"
            if row["layer"] == top_diff["layer"] and row["feature_idx"] == top_diff["feature_idx"]
            else "feature",
            "display_name": f"L{row['layer']} F{row['feature_idx']}",
        }
    return metadata_map


def patch_graph_json(
    graph_json_path: Path,
    *,
    model_output: str,
    default_attention_overlay_path: str | None,
    first_token_attention: dict,
    feature_heatmaps: dict[str, dict],
):
    graph = json.loads(graph_json_path.read_text())
    prompt_tokens = graph.get("metadata", {}).get("prompt_tokens", [])

    def is_semantic_text_embedding(node: dict) -> bool:
        if node.get("feature_type") != "embedding":
            return True
        ctx_idx = int(node.get("ctx_idx", -1))
        if ctx_idx < 0 or ctx_idx >= len(prompt_tokens):
            return False
        token = str(prompt_tokens[ctx_idx])
        stripped = token.strip()
        if not stripped:
            return False
        if token in {"<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|im_start|>", "<|im_end|>"}:
            return False
        if stripped in {"user", "assistant", "system"}:
            return False
        return True

    removed_node_ids = {
        node["node_id"]
        for node in graph.get("nodes", [])
        if node.get("feature_type") == "embedding" and not is_semantic_text_embedding(node)
    }
    if removed_node_ids:
        graph["nodes"] = [node for node in graph.get("nodes", []) if node.get("node_id") not in removed_node_ids]
        graph["links"] = [
            link
            for link in graph.get("links", [])
            if link.get("source") not in removed_node_ids and link.get("target") not in removed_node_ids
        ]

    metadata = graph.setdefault("metadata", {})
    metadata["model_output"] = model_output
    metadata["model_output_prob"] = (
        float(first_token_attention["top_token_probs"][0]["prob"])
        if first_token_attention.get("top_token_probs")
        else metadata.get("model_output_prob")
    )
    metadata["model_output_top_probs"] = first_token_attention.get("top_token_probs", [])
    metadata["first_token_attention"] = first_token_attention
    metadata["feature_heatmaps"] = feature_heatmaps
    metadata["default_attention_overlay_path"] = default_attention_overlay_path
    metadata["raw_input_image_path"] = metadata.get("image_path")
    metadata["hidden_visual_embedding_count"] = len(removed_node_ids)
    for node in graph.get("nodes", []):
        if node.get("feature_type") == "cross layer transcoder":
            node["clerp"] = ""
    graph_json_path.write_text(json.dumps(graph, indent=2))


def enrich_qwen_vlm_graph(
    *,
    model_id: str,
    prompt: str,
    image_path: str,
    graph_json_path: str,
    transcoder_dir: str | None = None,
    neutral_prompt: str = DEFAULT_NEUTRAL_PROMPT,
    seed: int = 0,
) -> dict:
    graph_json = Path(graph_json_path)
    output_dir = graph_json.parent.parent
    attention_dir = output_dir / "attention_maps"
    heatmap_dir = output_dir / "feature_heatmaps"
    image_path_obj = Path(image_path)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    image_token_id = find_image_token_id(model, processor)

    first_token_attention = build_first_token_attention_artifacts(
        processor,
        model,
        image_path_obj,
        prompt,
        image_token_id,
        attention_dir,
        seed,
    )

    feature_heatmaps: dict[str, dict] = {}
    if transcoder_dir:
        transcoder_root = Path(transcoder_dir)
        plt_layers = discover_transcoder_layers(transcoder_root)
        if plt_layers:
            plt_payloads = {layer: load_plt_encoder_payload(transcoder_root, layer) for layer in plt_layers}
            spatial_cache = prepare_prompt_cache(
                processor, model, image_path_obj, prompt, image_token_id, plt_layers, seed
            )
            neutral_cache = prepare_prompt_cache(
                processor, model, image_path_obj, neutral_prompt, image_token_id, plt_layers, seed
            )
            feature_keys = parse_graph_feature_keys(graph_json)
            feature_heatmaps = build_feature_heatmaps(
                feature_keys,
                plt_payloads,
                spatial_cache,
                neutral_cache,
                heatmap_dir,
            )
            model_output = spatial_cache.generated_text
        else:
            model_output, _ = generate_outputs(processor, model, image_path_obj, prompt, seed)
    else:
        model_output, _ = generate_outputs(processor, model, image_path_obj, prompt, seed)

    patch_graph_json(
        graph_json,
        model_output=model_output,
        default_attention_overlay_path=first_token_attention["overlay_paths"].get(first_token_attention["default_key"]),
        first_token_attention=first_token_attention,
        feature_heatmaps=feature_heatmaps,
    )

    return {
        "graph_json_path": str(graph_json),
        "attention_dir": str(attention_dir),
        "feature_heatmap_dir": str(heatmap_dir) if feature_heatmaps else None,
        "feature_count": len(feature_heatmaps),
        "model_output": model_output,
    }
