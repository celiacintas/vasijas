"""Print text + vision architecture parameters of the multimodal LLMs as a LaTeX table.

Models are loaded locally (mirroring load_multimodal_llms.py) so the table
captures both the language stack and the vision/vision-projector stack.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from janus.models import MultiModalityCausalLM


def _load_llava(device, dtype):
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    return model, model_id


def _load_qwen25_vl(device, dtype):
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    return model, model_id


def _load_gemma3(device, dtype):
    model_id = "google/gemma-3-4b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    return model, model_id


def _load_janus(device, dtype):
    sys.path.insert(0, "/tmp/janus")
    model_id = "deepseek-ai/Janus-1.3B"
    model = MultiModalityCausalLM.from_pretrained(model_id, trust_remote_code=True)
    if device == "cuda":
        model = model.to(torch.bfloat16).cuda().eval()
    else:
        model = model.eval()
    return model, model_id


LOADERS = {
    "Janus-1.3B": _load_janus,
    "LLaVA-1.5-7B": _load_llava,
    "Qwen2.5-VL-7B": _load_qwen25_vl,
    "Gemma-3-4B-IT": _load_gemma3,
}


def module_params(module):
    try:
        return module.num_parameters()
    except (AttributeError, TypeError):
        return sum(p.numel() for p in module.parameters())


def _first_truthy(*values):
    for v in values:
        if v is not None:
            return v
    return None


def find_vision_module(model):
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        for attr in ("vision_tower", "vision_model", "visual", "vision_encoder"):
            sub = getattr(root, attr, None)
            if sub is not None:
                return sub
    return None


def find_projector(model):
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        for attr in ("multi_modal_projector", "merger", "aligner"):
            sub = getattr(root, attr, None)
            if sub is not None:
                return sub
    return None


def probe_vision(vis, vision_cfg=None):
    cfg = getattr(vis, "config", None)
    layers = _first_truthy(
        getattr(cfg, "num_hidden_layers", None), getattr(cfg, "depth", None)
    )
    hidden = _first_truthy(
        getattr(cfg, "hidden_size", None),
        getattr(cfg, "d_model", None),
        getattr(cfg, "embed_dim", None),
    )
    heads = _first_truthy(
        getattr(cfg, "num_attention_heads", None), getattr(cfg, "num_heads", None)
    )
    image_size = getattr(cfg, "image_size", None)
    patch_size = getattr(cfg, "patch_size", None)
    merge_size = getattr(cfg, "spatial_merge_size", None)

    blocks = getattr(vis, "blocks", None)
    if layers is None and blocks is not None:
        layers = len(blocks)
    if hidden is None:
        proj = getattr(getattr(vis, "patch_embed", None), "proj", None)
        if proj is not None:
            hidden = getattr(proj, "out_channels", None)
    if heads is None and blocks is not None:
        attn = getattr(blocks[0], "attn", None)
        if attn is None:
            attn = getattr(getattr(blocks[0], "attention", None), "attn", None)
        if attn is not None:
            heads = getattr(attn, "num_heads", None)
    if patch_size is None:
        ps = getattr(getattr(vis, "patch_embed", None), "patch_size", None)
        if isinstance(ps, (tuple, list)):
            ps = ps[0] if ps else None
        patch_size = ps

    if image_size is None and vision_cfg is not None:
        if isinstance(vision_cfg, dict):
            image_size = vision_cfg.get("image_size")
        else:
            image_size = getattr(vision_cfg, "image_size", None)
    return layers, hidden, heads, image_size, patch_size, merge_size


VISION_NAMES = {
    "CLIPVisionModel": "CLIP ViT",
    "SiglipVisionModel": "SigLIP ViT",
    "Qwen2_5_VisionTransformer": "Qwen2.5 ViT",
    "SiglipVisionTransformer": "SigLIP-L ViT",
}


def extract_vision(model):
    vis = find_vision_module(model)
    if vis is None:
        return None
    cfg = getattr(model, "config", None)
    vision_cfg = getattr(cfg, "vision_config", None)
    layers, hidden, heads, image_size, patch_size, merge_size = probe_vision(
        vis, vision_cfg
    )
    return {
        "name": VISION_NAMES.get(type(vis).__name__, type(vis).__name__),
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "image_size": image_size,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "params": module_params(vis),
    }


def extract_projector(model):
    proj = find_projector(model)
    if proj is None:
        return None
    return {
        "name": type(proj).__name__,
        "params": module_params(proj),
    }


def extract_text(model):
    lang = getattr(model, "language_model", None) or getattr(model, "model", None)
    config = getattr(lang, "config", None)
    if config is None:
        config = getattr(model, "config", None)
        for attr in ("text_config", "language_config"):
            sub = getattr(config, attr, None)
            if sub is not None:
                config = sub
                break
    return {
        "layers": getattr(config, "num_hidden_layers", None),
        "hidden": getattr(config, "hidden_size", None),
        "heads": getattr(config, "num_attention_heads", None),
        "params": module_params(lang) if lang is not None else None,
    }


def extract_architecture(model):
    text = extract_text(model)
    vision = extract_vision(model)
    projector = extract_projector(model)
    return {
        "text": text,
        "vision": vision,
        "projector": projector,
        "params_b": model.num_parameters() / 1e9,
    }


def resolution_str(image_size, patch_size, merge_size):
    if image_size and patch_size:
        return f"{image_size}/{patch_size}"
    if patch_size:
        res = f"p{patch_size}"
        if merge_size:
            res += f" (merge {merge_size})"
        return res
    if image_size:
        return str(image_size)
    return "--"


def render_latex_table(rows):
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Architecture parameters of the evaluated multimodal LLMs.}",
        "\\label{tab:model-architecture}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l c c c c c c c c c}",
        "\\toprule",
        "Model & Vis. Enc. & Vis. Layers & Vis. Dim & Res. & Proj. & "
        "LLM Layers & LLM Dim & LLM Heads & Params (B) \\\\",
        "\\midrule",
    ]
    for name, arch in rows:
        v = arch["vision"]
        p = arch["projector"]
        res = resolution_str(
            v["image_size"] if v else None,
            v["patch_size"] if v else None,
            v["merge_size"] if v else None,
        )
        lines.append(
            f"{name} & {v['name'] if v else '--'} & "
            f"{v['layers'] if v else '--'} & {v['hidden'] if v else '--'} & {res} & "
            f"{p['name'] if p else '--'} & {arch['text']['layers']} & "
            f"{arch['text']['hidden']} & {arch['text']['heads']} & "
            f"{arch['params_b']:.2f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model_architecture_table.tex"),
        help="Output LaTeX file (default: model_architecture_table.tex)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    rows = []
    for name, loader in LOADERS.items():
        print(f"Loading {name} ...")
        model = loader(device, dtype)[0]
        arch = extract_architecture(model)
        rows.append((name, arch))
        print(f"  text:      {arch['text']}")
        print(f"  vision:    {arch['vision']}")
        print(f"  projector: {arch['projector']}")
        print(f"  total:     {arch['params_b']:.3f}B")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    tex = render_latex_table(rows)
    args.output.write_text(tex, encoding="utf-8")
    print(f"\nWrote LaTeX table to {args.output}")
    print(tex)


if __name__ == "__main__":
    main()
