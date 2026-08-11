"""Load and test multimodal LLMs from Hugging Face"""

import argparse
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
    AutoModelForImageTextToText,
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
    # if len(vl_gpt.language_model.get_input_embeddings().weight) != len(
    #    vl_chat_processor.tokenizer
    # ):
    #    vl_gpt.language_model.resize_token_embeddings(len(vl_chat_processor.tokenizer))
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


def load_minicpm(device, dtype):
    """Load MiniCPM-V-4.6 model and processor from Hugging Face."""
    model_id = "openbmb/MiniCPM-V-4.6"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def infer_minicpm(model, processor, image, prompt, device):
    """Run MiniCPM-V-4.6 inference with chat template, return decoded response."""
    downsample_mode = "16x"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        downsample_mode=downsample_mode,
        max_slice_nums=36,
    ).to(model.device)
    generated_ids = model.generate(
        **inputs, downsample_mode=downsample_mode, max_new_tokens=512
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


LOADERS = {
    "Janus-1.3B": load_janus,
    "LLaVA-1.5-7B": load_llava,
    "Qwen2.5-VL-7B": load_qwen25_vl,
    "Gemma-3-4B-IT": load_gemma3,
    # "MiniCPM-V-4.6": load_minicpm,  # needs transformers>=? to support minicpmv4_6 arch
    # "Moondream2": load_moondream2,
}

INFER = {
    "Janus-1.3B": infer_janus,
    "LLaVA-1.5-7B": infer_llava,
    "Qwen2.5-VL-7B": infer_qwen25_vl,
    "Gemma-3-4B-IT": infer_gemma3,
    # "MiniCPM-V-4.6": infer_minicpm,  # needs transformers>=? to support minicpmv4_6 arch
    # "Moondream2": infer_moondream2,
}


def get_cultural_samples(n_runs=3, seed=42):
    """Split the image pool into n_runs disjoint, culture-stratified sample sets.

    Every image is assigned to exactly one run, so no filename is ever shared
    between runs. Each run stays balanced across cultures (~1/n_runs of each
    culture's images).
    """
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
    if len(by_culture) == 0:
        return [[] for _ in range(n_runs)]

    # Largest k so that every run can take k images from every culture
    # without reusing an image across runs (no filename overlaps) while
    # keeping the per-run class counts balanced.
    k = min(len(pool) // n_runs for pool in by_culture.values())
    if k == 0:
        raise ValueError(
            f"Not enough images per culture for {n_runs} disjoint runs: "
            + ", ".join(f"{c}={len(p)}" for c, p in by_culture.items())
        )

    runs = [[] for _ in range(n_runs)]
    for cul in sorted(by_culture):
        pool = by_culture[cul]
        chosen = rng.sample(pool, k * n_runs)
        for i in range(n_runs):
            runs[i].extend(chosen[i * k : (i + 1) * k])
        print(f"  {cul}: {k} per run (from {len(pool)} available)")
    for run in runs:
        rng.shuffle(run)
        for entry in run:
            entry["pil_image"] = Image.open(entry["path"]).convert("RGB")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="Fall back to CPU if model doesn't fit in GPU memory",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of independent runs with disjoint, class-balanced image samples",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base_cultures = [
        "Iberian",
        "Predynastic-egyptian",
        "Kushite",
        "Andean",
        "East African",
        "West African",
    ]
    print("Splitting cultural samples into disjoint runs:")
    sample_runs = get_cultural_samples(n_runs=args.n_runs)
    print(
        f"Runs: {[len(run) for run in sample_runs]} samples; "
        f"per-run per-class: {len(sample_runs[0]) // len(base_cultures)}"
    )
    all_paths = [s["path"] for run in sample_runs for s in run]
    assert len(all_paths) == len(set(all_paths)), "sample overlap across runs!"

    for run_idx, sample_images in enumerate(sample_runs):
        rng = random.Random(42 + run_idx)
        rows = []
        print(
            f"\n{'=' * 70}\n  Run {run_idx + 1} ({len(sample_images)} samples)\n{'=' * 70}"
        )
        for name, loader in LOADERS.items():
            current_device = device
            current_dtype = dtype
            print(f"\n{'=' * 70}")
            print(f"  {name}")
            print(f"{'=' * 70}")
            try:
                result = loader(current_device, current_dtype)
                model = result[0]
                proc_tok = result[1]
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
                    model = result[0]
                    proc_tok = result[1]
                else:
                    print(f"Failed: {e}")
                    traceback.print_exc()
                    continue
            print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
            print("  Prompt: shuffled options with culture name response")
            print()
            infer_fn = INFER[name]
            for s in sample_images:
                cultures = base_cultures.copy()
                rng.shuffle(cultures)
                culture_list = ", ".join(cultures[:-1]) + f", or {cultures[-1]}"
                prompt = f"Classify this ceramic artifact into one culture: {culture_list}. Respond with only the culture name."
                response = infer_fn(
                    model, proc_tok, s["pil_image"], prompt, current_device
                )
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
            if current_device == "cuda":
                torch.cuda.empty_cache()

        output_path = Path(f"evaluation_results_run{run_idx + 1}.csv")
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "response", "groundtruth", "filename"])
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {output_path}")
