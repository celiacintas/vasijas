"""Load and test multimodal LLMs from Hugging Face"""

import torch
from pathlib import Path
from PIL import Image


def load_llava(device, dtype):
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, model_id


def infer_llava(model, processor, image, prompt, device):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=128)
    return processor.decode(output[0], skip_special_tokens=True)


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
    output = model.generate(**inputs, max_new_tokens=128)
    return processor.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def load_glm4v(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "THUDM/glm-4v-9b"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer, model_id


def infer_glm4v(model, tokenizer, image, prompt, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(device)
    output = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def load_gemma3(device, dtype):
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "google/gemma-3-4b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
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
    output = model.generate(**inputs, max_new_tokens=128)
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
    # "LLaVA-1.5-7B": load_llava,
    "Qwen2.5-VL-7B": load_qwen25_vl,
    "GLM-4V-9B": load_glm4v,
    # "Gemma-3-4B-IT": load_gemma3,
    # "Moondream2": load_moondream2,
}

INFER = {
    # "LLaVA-1.5-7B": infer_llava,
    "Qwen2.5-VL-7B": infer_qwen25_vl,
    "GLM-4V-9B": infer_glm4v,
    # "Gemma-3-4B-IT": infer_gemma3,
    # "Moondream2": infer_moondream2,
}


def get_sample_images():
    image_dir = Path("data/artifacts_with_descriptions")
    paths = sorted(image_dir.glob("*.png"))[:3]
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        images.append((p.name, img))
    return images


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    sample_images = get_sample_images()
    prompt = "Describe this image in one sentence."

    for name, loader in LOADERS.items():
        print(f"\n{'=' * 70}")
        print(f"  {name}")
        print(f"{'=' * 70}")
        try:
            result = loader(device, dtype)
            model = result[0]
            proc_tok = result[1]
            print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
            print(f"  Images: {[img_name for img_name, _ in sample_images]}")
            print(f"  Prompt: {prompt}")
            print()
            infer_fn = INFER[name]
            for img_name, img in sample_images:
                response = infer_fn(model, proc_tok, img, prompt, device)
                clean = (
                    response.split(prompt)[-1].strip()
                    if prompt in response
                    else response
                )
                print(f"  [{img_name}]")
                print(f"  {clean}")
                print()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()
