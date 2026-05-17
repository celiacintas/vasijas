"""Load and test multimodal LLMs from Hugging Face"""

import csv
import random
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from janus.models import VLChatProcessor, MultiModalityCausalLM


def load_llava(device, dtype):
    """Load LLaVA-1.5-7B model and processor from Hugging Face."""
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def infer_llava(model, processor, image, prompt, device):
    """Run LLaVA inference with chat template, return decoded response."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def load_qwen25_vl(device, dtype):
    """Load Qwen2.5-VL-7B-Instruct model and processor from Hugging Face."""
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def infer_qwen25_vl(model, processor, image, prompt, device):
    """Run Qwen2.5-VL inference with chat template, return decoded response."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    output = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def load_gemma3(device, dtype):
    """Load Gemma-3-4B-IT model and processor from Hugging Face."""
    model_id = "google/gemma-3-4b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def infer_gemma3(model, processor, image, prompt, device):
    """Run Gemma-3 inference with chat template, return decoded response."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    output = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def load_moondream2(device, dtype):
    """Load Moondream2 model and tokenizer with monkey-patch for all_tied_weights_keys."""
    orig_getattr = nn.Module.__getattr__

    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            keys = getattr(self, "_tied_weights_keys", {})
            return {} if keys is None else keys
        return orig_getattr(self, name)

    nn.Module.__getattr__ = _patched_getattr

    model_id = "vikhyatk/moondream2"
    revision = "2025-06-21"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
    finally:
        nn.Module.__getattr__ = orig_getattr
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer, model_id


def infer_moondream2(model, tokenizer, image, prompt, device):
    """Run Moondream2 inference via query API."""
    return model.query(image, prompt)["answer"]


def load_janus(device, dtype):
    """Load Janus-1.3B model and processor from cloned repo at /tmp/janus."""
    sys.path.insert(0, "/tmp/janus")

    model_path = "deepseek-ai/Janus-1.3B"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    if (
        "<image_placeholder>"
        not in vl_chat_processor.tokenizer.additional_special_tokens
    ):
        vl_chat_processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<image_placeholder>"]}
        )

    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if device == "cuda":
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    else:
        vl_gpt = vl_gpt.eval()
    return vl_gpt, vl_chat_processor, model_path


def infer_janus(model, processor, image, prompt, device):
    """Run Janus-1.3B inference with image placeholder token, return decoded response."""
    tokenizer = processor.tokenizer
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [image],
        },
        {"role": "Assistant", "content": ""},
    ]

    prepare_inputs = processor(
        conversations=conversation, images=[image], force_batchify=True
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


LOADERS = {
    "LLaVA-1.5-7B": load_llava,
    "Qwen2.5-VL-7B": load_qwen25_vl,
    "Gemma-3-4B-IT": load_gemma3,
    "Janus-1.3B": load_janus,
    # "Moondream2": load_moondream2,
}

INFER = {
    "LLaVA-1.5-7B": infer_llava,
    "Qwen2.5-VL-7B": infer_qwen25_vl,
    "Gemma-3-4B-IT": infer_gemma3,
    "Janus-1.3B": infer_janus,
    # "Moondream2": infer_moondream2,
}


def get_cultural_samples(n=100, seed=42):
    """Sample n images evenly across culture folders, stratified by culture."""
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
            print(f"  Warning: {folder_path} not found, skipping")
            continue
        images = sorted(
            list(folder_path.glob("**/*.png")) + list(folder_path.glob("**/*.jpg"))
        )
        by_culture[folder.lower()] = [
            {"culture": folder.lower(), "filename": p.name, "path": str(p), "idx": idx}
            for idx, p in enumerate(images)
        ]
    num_cultures = len(by_culture)
    if num_cultures == 0:
        return []
    per_culture = n // num_cultures
    sampled = []
    for cul in sorted(by_culture):
        pool = by_culture[cul]
        k = min(per_culture, len(pool))
        for entry in rng.sample(pool, k):
            img = Image.open(entry["path"]).convert("RGB")
            sampled.append({**entry, "pil_image": img})
        print(f"  {cul}: {k} samples (from {len(pool)} available)")
    # Distribute remainder — one extra sample to random cultures with capacity
    remain = n - len(sampled)
    candidates = [c for c in sorted(by_culture) if len(by_culture[c]) > per_culture]
    for cul in rng.sample(candidates, min(remain, len(candidates))):
        pool = [
            e for e in by_culture[cul] if e["path"] not in {s["path"] for s in sampled}
        ]
        if pool:
            entry = rng.choice(pool)
            img = Image.open(entry["path"]).convert("RGB")
            sampled.append({**entry, "pil_image": img})
            print(f"  {cul}: +1 extra sample")
    rng.shuffle(sampled)
    return sampled


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    sample_images = get_cultural_samples(n=498)
    base_cultures = [
        "Iberian",
        "Predynastic-egyptian",
        "Kushite",
        "Andean",
        "East African",
        "West African",
    ]
    rng = random.Random(42)
    rows = []
    for name, loader in LOADERS.items():
        print(f"\n{'=' * 70}")
        print(f"  {name}")
        print(f"{'=' * 70}")
        try:
            result = loader(device, dtype)
            model = result[0]
            proc_tok = result[1]
            print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
            print("  Prompt: shuffled options with culture name response")
            print()
            infer_fn = INFER[name]
            for s in sample_images:
                cultures = base_cultures.copy()
                rng.shuffle(cultures)
                culture_list = ", ".join(cultures[:-1]) + f", or {cultures[-1]}"
                prompt = f"Classify this ceramic artifact into one culture: {culture_list}. Respond with only the culture name."
                response = infer_fn(model, proc_tok, s["pil_image"], prompt, device)
                culture_name = (
                    response.strip().lower().split("\n")[0]
                    if prompt not in response
                    else response.split(prompt)[-1].strip().lower()
                )
                rows.append([name, culture_name, s["culture"], s["filename"]])
                print(f"  [{s['filename']}] (ground truth: {s['culture']})")
                print(f"  {culture_name}")
                print()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed: {e}")
            traceback.print_exc()

    output_path = Path("evaluation_results.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "response", "groundtruth", "filename"])
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_path}")
