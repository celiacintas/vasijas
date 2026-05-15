"""Load and test multimodal LLMs from Hugging Face"""

import random
import torch
from PIL import Image

from ceramic_dataset import CeramicArtifactDataset


def load_llava(device, dtype):
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def infer_llava(model, processor, image, prompt, device):
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
    output = model.generate(**inputs, max_new_tokens=128)
    return processor.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def load_qwen25_vl(device, dtype):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def infer_qwen25_vl(model, processor, image, prompt, device):
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
    output = model.generate(**inputs, max_new_tokens=10)
    return processor.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def load_glm4v(device, dtype):
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    orig_getattr = nn.Module.__getattr__

    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            keys = getattr(self, "_tied_weights_keys", {})
            return {} if keys is None else keys
        return orig_getattr(self, name)

    nn.Module.__getattr__ = _patched_getattr

    model_id = "THUDM/glm-4v-9b"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(config, "max_length") and hasattr(config, "seq_length"):
        config.max_length = config.seq_length
    try:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )
    finally:
        nn.Module.__getattr__ = orig_getattr
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(tokenizer, "batch_encode_plus"):
        import functools
        from transformers.tokenization_utils import PreTrainedTokenizer

        tokenizer.batch_encode_plus = functools.partial(
            PreTrainedTokenizer.batch_encode_plus, tokenizer
        )
    return model, tokenizer, model_id


def infer_glm4v(model, tokenizer, image, prompt, device):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    gen_kwargs = {"max_length": 200, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(outputs[0])


def load_gemma3(device, dtype):
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "google/gemma-3-4b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def infer_gemma3(model, processor, image, prompt, device):
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
    output = model.generate(**inputs, max_new_tokens=10)
    return processor.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


# def load_moondream2(device, dtype):
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#
#     model_id = "vikhyatk/moondream2"
#     revision = "2025-06-21"
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, revision=revision, torch_dtype=dtype, device_map=device, trust_remote_code=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
#     return model, tokenizer, model_id
#
#
# def infer_moondream2(model, tokenizer, image, prompt, device):
#     image_embeds = model.encode_image(image)
#     return model.answer_question(image_embeds, prompt, tokenizer)


LOADERS = {
    "LLaVA-1.5-7B": load_llava,
    "Qwen2.5-VL-7B": load_qwen25_vl,
    # "GLM-4V-9B": load_glm4v,
    "Gemma-3-4B-IT": load_gemma3,
    # "Moondream2": load_moondream2,
}

INFER = {
    "LLaVA-1.5-7B": infer_llava,
    "Qwen2.5-VL-7B": infer_qwen25_vl,
    # "GLM-4V-9B": infer_glm4v,
    "Gemma-3-4B-IT": infer_gemma3,
    # "Moondream2": infer_moondream2,
}


def get_cultural_samples(n=100, seed=42):
    dataset = CeramicArtifactDataset(image_dir="data")
    by_culture = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        by_culture.setdefault(sample["culture"], []).append({**sample, "idx": i})
    rng = random.Random(seed)
    per_culture = n // 2
    sampled = []
    for cul in ("iberian", "egyptian"):
        pool = by_culture.get(cul, [])
        k = min(per_culture, len(pool))
        for entry in rng.sample(pool, k):
            img = Image.open(dataset.image_paths[entry["idx"]]).convert("RGB")
            sampled.append({**entry, "pil_image": img})
        print(f"  {cul}: {k} samples (from {len(pool)} available)")
    rng.shuffle(sampled)
    return sampled


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    sample_images = get_cultural_samples(n=3)
    prompt = "In this image you can see an archeological ceramic artifact, can you tell me to which culture belongs to in two words?"

    for name, loader in LOADERS.items():
        print(f"\n{'=' * 70}")
        print(f"  {name}")
        print(f"{'=' * 70}")
        try:
            result = loader(device, dtype)
            model = result[0]
            proc_tok = result[1]
            print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
            print(f"  Images: {[s['filename'] for s in sample_images]}")
            print(f"  Prompt: {prompt}")
            print()
            infer_fn = INFER[name]
            for s in sample_images:
                response = infer_fn(model, proc_tok, s["pil_image"], prompt, device)
                clean = (
                    response.split(prompt)[-1].strip()
                    if prompt in response
                    else response
                )
                print(f"  [{s['filename']}] (ground truth: {s['culture']})")
                print(f"  {clean}")
                print()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()
