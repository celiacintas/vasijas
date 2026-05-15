## Multimodal LLM Evaluation

The script `experiments/load_multimodal_llms.py` evaluates vision-language models (LLaVA, Qwen2.5-VL, GLM-4V, Gemma-3) on ceramic artifact culture classification.

Run with `uv`:

```bash
# Install dependencies
uv sync

# Run evaluation on 3 random samples
uv run python experiments/load_multimodal_llms.py
```

### Data source

Iberian pottery images are sourced from the doctoral thesis of Padilla.

```BibTeX
@phdthesis{padilla2019decoracion,
  title={Decoraci{\'o}n vascular y significaci{\'o}n social en los territorios {\'\i}beros. Los estilos y grupos pict{\'o}ricos de la cer{\'a}mica a torno del Alto Guadalquivir (siglos VI aNE-I dNE)},
  author={Padilla, Mar{\'\i}a Isabel Moreno},
  year={2019},
  school={Universidad de Ja{\'e}n}
}
```

Predynastic Egyptian pottery images are sourced from the Predynastic Online Database (PONDA).

```BibTeX
@misc{ponda2026,
  author = {Droux, Xavier},
  title = {Predynastic Online Database ({PONDA})},
  howpublished = {\url{https://ponda.org}},
  year = {2026},
  note = {ISSN 2813-7132},
  address = {Geneva, Switzerland},
}
```
