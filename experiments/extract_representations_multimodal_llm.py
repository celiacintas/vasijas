"""Extract representations from Visual Encoder and Projection layers of multimodal LLMs.

Following the pattern from IBM/personas-llms-analysis/notebooks/representation_extraction.ipynb,
this script loads each model, processes images through the visual encoder and projection layers,
and saves the intermediate representations using forward hooks.
"""

import argparse
import pickle
import random
import traceback
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from experiments.load_multimodal_llms import get_cultural_samples
except ModuleNotFoundError:

    def get_cultural_samples(n_runs=3, seed=42):
        rng = random.Random(seed)
        data_root = Path("data")
        culture_folders = [
            "Iberian",
            "Predynastic-egyptian",
            "Kushite",
            "Andean",
            "East African",
            "West African",
        ]
        by_culture = {}
        for folder in culture_folders:
            folder_path = data_root / folder
            if not folder_path.is_dir():
                continue
            images = sorted(
                list(folder_path.glob("**/*.png")) + list(folder_path.glob("**/*.jpg"))
            )
            by_culture[folder.lower()] = [
                {
                    "culture": folder.lower(),
                    "filename": p.name,
                    "path": str(p),
                    "idx": idx,
                }
                for idx, p in enumerate(images)
            ]
        k = (
            min(len(pool) // n_runs for pool in by_culture.values())
            if by_culture
            else 0
        )
        runs = [[] for _ in range(n_runs)]
        for cul in sorted(by_culture):
            pool = by_culture[cul]
            chosen = rng.sample(pool, k * n_runs)
            for i in range(n_runs):
                runs[i].extend(chosen[i * k : (i + 1) * k])
        for run in runs:
            rng.shuffle(run)
            for entry in run:
                entry["pil_image"] = Image.open(entry["path"]).convert("RGB")
        return runs


BASE_CULTURES = [
    "Iberian",
    "Predynastic-egyptian",
    "Kushite",
    "Andean",
    "East African",
    "West African",
]


def build_culture_prompt(rng):
    """Build a classification prompt with shuffled culture names."""
    cultures = BASE_CULTURES.copy()
    rng.shuffle(cultures)
    culture_list = ", ".join(cultures[:-1]) + f", or {cultures[-1]}"
    return f"Classify this ceramic artifact into one culture: {culture_list}. Respond with only the culture name."


def _patch_janus_nn_getattr():
    """Patch nn.Module.__getattr__ for Janus model loading."""
    import torch.nn as nn

    orig_getattr = nn.Module.__getattr__

    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            keys = getattr(self, "_tied_weights_keys", {})
            return {} if keys is None else keys
        return orig_getattr(self, name)

    nn.Module.__getattr__ = _patched_getattr
    return orig_getattr


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
    # Qwen2.5-VL: projector (merger) is nested inside the vision transformer
    vision = find_vision_module(model)
    if vision is not None:
        for attr in ("merger", "aligner", "multi_modal_projector"):
            sub = getattr(vision, attr, None)
            if sub is not None:
                return sub
    return None


def get_module_name(module, prefix="model"):
    """Build a dotted name from module class hierarchy."""
    parts = [prefix]
    obj = module
    while obj is not None:
        parent = getattr(obj, "parent", None) or getattr(obj, "_parent", None)
        if parent is None:
            break
        for name, child in parent.named_children():
            if child is obj:
                parts.append(name)
                obj = parent
                break
        else:
            break
    return ".".join(reversed(parts))


class HookManager:
    """Register forward hooks and capture outputs from named modules."""

    def __init__(self):
        self.hooks = OrderedDict()
        self.outputs = OrderedDict()

    def register(self, name, module):
        def hook_fn(m, inp, out):
            self.outputs[name] = out

        h = module.register_forward_hook(hook_fn)
        self.hooks[name] = h

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self.hooks.values():
            h.remove()
        self.hooks.clear()
        self.outputs.clear()


def extract_hook_tensor(hook_output):
    """Extract the main tensor from a hook output (handles tensor, tuple, BaseModelOutput)."""
    if isinstance(hook_output, torch.Tensor):
        return hook_output
    if isinstance(hook_output, (list, tuple)):
        return hook_output[0]
    if hasattr(hook_output, "last_hidden_state"):
        return hook_output.last_hidden_state
    if hasattr(hook_output, "logits"):
        return hook_output.logits
    return hook_output


# ---------------------------------------------------------------------------
# Model-specific representation extraction
# ---------------------------------------------------------------------------


def extract_representations_llava(model, processor, images, device, model_name):
    vision = find_vision_module(model)
    projector = find_projector(model)
    if vision is None:
        raise RuntimeError("Cannot find vision tower in LLaVA model")
    if projector is None:
        raise RuntimeError("Cannot find projector in LLaVA model")

    vision_name = get_module_name(vision, "model")
    proj_name = get_module_name(projector, "model")
    print(f"  Vision: {vision_name}  Projector: {proj_name}")

    hook_mgr = HookManager()
    hook_mgr.register("vision_encoder", vision)
    hook_mgr.register("projector", projector)

    rng = random.Random(42)
    reps = {}
    for img_info in images:
        img = img_info["pil_image"]
        prompt = build_culture_prompt(rng)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)

        hook_mgr.clear()
        with torch.no_grad():
            model(**inputs)

        reps[img_info["filename"]] = {
            "culture": img_info["culture"],
            "vision_encoder": extract_hook_tensor(
                hook_mgr.outputs.get("vision_encoder")
            ).cpu(),
            "projector": extract_hook_tensor(hook_mgr.outputs.get("projector")).cpu(),
        }
    hook_mgr.remove()
    return reps


def extract_representations_qwen25vl(model, processor, images, device, model_name):
    vision = find_vision_module(model)
    projector = find_projector(model)
    if vision is None:
        raise RuntimeError("Cannot find vision module in Qwen2.5-VL model")
    if projector is None:
        raise RuntimeError("Cannot find projector in Qwen2.5-VL model")

    # For Qwen2.5-VL, the merger is inside the vision transformer.
    # Hooking model.visual would give the post-merger output, identical
    # to the merger output. Instead, hook the last transformer block
    # for the raw vision-encoder representation (pre-merger, vision-space).
    vision_blocks = getattr(vision, "blocks", None)
    if vision_blocks is not None and len(vision_blocks) > 0:
        vision_enc = vision_blocks[-1]
        print(
            f"  Vision: {get_module_name(vision, 'model')}.blocks[-1]  Projector: {get_module_name(projector, 'model')}"
        )
    else:
        vision_enc = vision
        print(
            f"  Vision: {get_module_name(vision, 'model')}  Projector: {get_module_name(projector, 'model')}"
        )

    hook_mgr = HookManager()
    hook_mgr.register("vision_encoder", vision_enc)
    hook_mgr.register("projector", projector)

    rng = random.Random(42)
    reps = {}
    for img_info in images:
        img = img_info["pil_image"]
        prompt = build_culture_prompt(rng)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[img], padding=True, return_tensors="pt"
        ).to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        hook_mgr.clear()
        with torch.no_grad():
            model(**inputs)

        reps[img_info["filename"]] = {
            "culture": img_info["culture"],
            "vision_encoder": extract_hook_tensor(
                hook_mgr.outputs.get("vision_encoder")
            ).cpu(),
            "projector": extract_hook_tensor(hook_mgr.outputs.get("projector")).cpu(),
        }
    hook_mgr.remove()
    return reps


def extract_representations_gemma3(model, processor, images, device, model_name):
    vision = find_vision_module(model)
    projector = find_projector(model)
    if vision is None:
        raise RuntimeError("Cannot find vision tower in Gemma-3 model")
    if projector is None:
        raise RuntimeError("Cannot find projector in Gemma-3 model")

    vision_name = get_module_name(vision, "model")
    proj_name = get_module_name(projector, "model")
    print(f"  Vision: {vision_name}  Projector: {proj_name}")

    hook_mgr = HookManager()
    hook_mgr.register("vision_encoder", vision)
    hook_mgr.register("projector", projector)

    rng = random.Random(42)
    reps = {}
    for img_info in images:
        img = img_info["pil_image"]
        prompt = build_culture_prompt(rng)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=img, return_tensors="pt").to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        hook_mgr.clear()
        with torch.no_grad():
            model(**inputs)

        reps[img_info["filename"]] = {
            "culture": img_info["culture"],
            "vision_encoder": extract_hook_tensor(
                hook_mgr.outputs.get("vision_encoder")
            ).cpu(),
            "projector": extract_hook_tensor(hook_mgr.outputs.get("projector")).cpu(),
        }
    hook_mgr.remove()
    return reps


def extract_representations_janus(model, processor, images, device, model_name):
    vision = find_vision_module(model)
    projector = find_projector(model)
    if vision is None:
        raise RuntimeError("Cannot find vision module in Janus model")
    if projector is None:
        raise RuntimeError("Cannot find projector in Janus model")

    vision_name = get_module_name(vision, "model")
    proj_name = get_module_name(projector, "model")
    print(f"  Vision: {vision_name}  Projector: {proj_name}")

    hook_mgr = HookManager()
    hook_mgr.register("vision_encoder", vision)
    hook_mgr.register("projector", projector)

    # tokenizer = processor.tokenizer
    rng = random.Random(42)
    reps = {}
    for img_info in images:
        img = img_info["pil_image"]
        prompt = build_culture_prompt(rng)
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [img],
            },
            {"role": "Assistant", "content": ""},
        ]
        prepare_inputs = processor(
            conversations=conversation, images=[img], force_batchify=True
        ).to(model.device)

        hook_mgr.clear()
        with torch.no_grad():
            model.prepare_inputs_embeds(**prepare_inputs)

        reps[img_info["filename"]] = {
            "culture": img_info["culture"],
            "vision_encoder": extract_hook_tensor(
                hook_mgr.outputs.get("vision_encoder")
            ).cpu(),
            "projector": extract_hook_tensor(hook_mgr.outputs.get("projector")).cpu(),
        }
    hook_mgr.remove()
    return reps


EXTRACTORS = {
    "LLaVA-1.5-7B": extract_representations_llava,
    "Qwen2.5-VL-7B": extract_representations_qwen25vl,
    "Gemma-3-4B-IT": extract_representations_gemma3,
    "Janus-1.3B": extract_representations_janus,
}


# ---------------------------------------------------------------------------
# Loading helpers (mirror load_multimodal_llms.py LOADERS)
# ---------------------------------------------------------------------------


def _load_llava(device, dtype):
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def _load_qwen25_vl(device, dtype):
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def _load_gemma3(device, dtype):
    model_id = "google/gemma-3-4b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def _load_janus(device, dtype):
    from janus.models import VLChatProcessor, MultiModalityCausalLM

    orig = _patch_janus_nn_getattr()
    model_id = "deepseek-ai/Janus-1.3B"
    try:
        model = MultiModalityCausalLM.from_pretrained(model_id, trust_remote_code=True)
    finally:
        import torch.nn as nn

        nn.Module.__getattr__ = orig

    vl_chat_processor = VLChatProcessor.from_pretrained(model_id)
    if (
        "<image_placeholder>"
        not in vl_chat_processor.tokenizer.additional_special_tokens
    ):
        vl_chat_processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<image_placeholder>"]}
        )

    if device == "cuda":
        model = model.to(torch.bfloat16).cuda().eval()
    else:
        model = model.eval()
    return model, vl_chat_processor, model_id


REPRESENTATION_LOADERS = {
    "LLaVA-1.5-7B": _load_llava,
    "Qwen2.5-VL-7B": _load_qwen25_vl,
    "Gemma-3-4B-IT": _load_gemma3,
    "Janus-1.3B": _load_janus,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract visual encoder and projector representations from multimodal LLMs."
    )
    parser.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="Fall back to CPU if model does not fit in GPU memory",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit number of images per culture (default: all available)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("representations"),
        help="Output directory for saved representations (default: representations/)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names to process (default: all active loaders)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load cultural samples (all images, no run splitting)
    print("Loading cultural samples ...")
    all_runs = get_cultural_samples(n_runs=1, seed=42)
    all_images = all_runs[0] if all_runs else []
    print(f"Total images: {len(all_images)}")

    # Limit samples if requested
    if args.n_samples is not None:
        all_images = all_images[: args.n_samples]
        print(f"Limited to {len(all_images)} images")

    if not all_images:
        print("No images found. Check data/ directory structure.")
        return

    # Select models
    model_names = args.models if args.models else list(REPRESENTATION_LOADERS.keys())
    model_names = [n for n in model_names if n in REPRESENTATION_LOADERS]
    print(f"Models: {model_names}")

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in model_names:
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    for name in model_names:
        loader = REPRESENTATION_LOADERS[name]
        extractor = EXTRACTORS[name]
        print(f"\n{'=' * 70}")
        print(f"  {name}")
        print(f"{'=' * 70}")

        current_device = device
        current_dtype = dtype
        try:
            result = loader(current_device, current_dtype)
            model, processor, model_id = result
        except (
            torch.cuda.OutOfMemoryError,
            torch.OutOfMemoryError,
            RuntimeError,
        ) as e:
            if args.cpu_fallback and current_device == "cuda":
                print(f"  GPU OOM for {name}, falling back to CPU ({e})")
                current_device = "cpu"
                current_dtype = torch.float32
                result = loader(current_device, current_dtype)
                model, processor, model_id = result
            else:
                print(f"  Failed to load {name}: {e}")
                traceback.print_exc()
                continue

        print(f"  Model ID: {model_id}")
        print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")

        try:
            representations = extractor(
                model, processor, all_images, current_device, name
            )

            out_path = args.output_dir / name / "representations.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(representations, f)

            n_images = len(representations)
            sample_key = next(iter(representations)) if representations else None
            if sample_key:
                sample = representations[sample_key]
                vision_shape = (
                    sample["vision_encoder"].shape
                    if "vision_encoder" in sample
                    else "N/A"
                )
                proj_shape = (
                    sample["projector"].shape if "projector" in sample else "N/A"
                )
            else:
                vision_shape = proj_shape = "N/A"
            print(f"  Saved {n_images} representations to {out_path}")
            print(f"  Vision encoder shape: {vision_shape}")
            print(f"  Projector shape:       {proj_shape}")

        except Exception as e:
            print(f"  Failed to extract representations for {name}: {e}")
            traceback.print_exc()

        del model
        if current_device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. Representations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
