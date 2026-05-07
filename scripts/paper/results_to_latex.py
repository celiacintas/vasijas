import argparse
import json
import numpy as np
from pathlib import Path


def results_to_latex(results_file, output_file=None):
    with open(results_file) as f:
        results = json.load(f)

    rows = []
    for name, data in results.items():
        rank = data.get("rank", "?")
        fid = data["fid"]
        clip_per_image = data["clip_score_per_image"]
        clip_mean = np.mean(clip_per_image)
        clip_std = np.std(clip_per_image)

        num_gen = data["num_generated"]
        num_prompts = len(data["prompts"])
        gen_per_prompt = num_gen // num_prompts

        per_prompt_clip = []
        for p in range(num_prompts):
            start = p * gen_per_prompt
            end = start + gen_per_prompt
            scores = clip_per_image[start:end]
            per_prompt_clip.append((np.mean(scores), np.std(scores)))

        rows.append({
            "name": name,
            "rank": rank,
            "fid": fid,
            "clip_mean": clip_mean,
            "clip_std": clip_std,
            "per_prompt_clip": per_prompt_clip,
        })

    rows.sort(key=lambda r: r["rank"] if isinstance(r["rank"], (int, float)) else 9999)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{LoRA rank ablation: FID and CLIP Score (mean $\pm$ std) across different LoRA ranks.}")
    lines.append(r"\label{tab:lora_ablation}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{FID} $\downarrow$ & \textbf{CLIP Score} $\uparrow$ & \textbf{CLIP Score (all prompts)} $\uparrow$ \\")
    lines.append(r"\midrule")

    for row in rows:
        name = row["name"].replace("_", "\\_")
        fid_str = f"{row['fid']:.2f}"
        clip_str = f"{row['clip_mean']:.2f} $\\pm$ {row['clip_std']:.2f}"

        prompt_parts = []
        for m, s in row["per_prompt_clip"]:
            prompt_parts.append(f"{m:.2f} $\\pm$ {s:.2f}")
        all_prompts_str = " \\\\ ".join(prompt_parts)

        lines.append(f"{name} & {fid_str} & {clip_str} & \\makecell{{{all_prompts_str}}} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    if output_file:
        Path(output_file).write_text(latex)
        print(f"LaTeX table written to {output_file}")
    else:
        print(latex)

    return latex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    results_to_latex(args.results_file, args.output)
