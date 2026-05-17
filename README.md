## Multimodal LLM Evaluation

The script `experiments/load_multimodal_llms.py` evaluates vision-language models (LLaVA, Qwen2.5-VL, Gemma-3, Janus-1.3B) on ceramic artifact culture classification.

> **Janus-1.3B** requires Python 3.9 — it is the only version that works with its dependencies (`attrdict`).

### Setup

```bash
# System dependencies
apt-get install python3-pip tmux
pip install uv

# Clone and checkout branch
git clone https://github.com/celiacintas/vasijas.git
cd vasijas
git checkout multimodal-bias

# Install Python dependencies
uv sync

# Authenticate with Hugging Face (required for Gemma-3)
uv run huggingface-cli login

# Install flash-attn (required for Janus-1.3B)
uv pip install flash-attn --no-build-isolation

# Clone Janus repo (required for Janus-1.3B)
git clone https://github.com/deepseek-ai/Janus.git /tmp/janus
```

### Run

```bash
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

Andean and Kushite pottery images are sourced from the online British Museum collection. East and West African pottery images also come from the British Museum, though we are not certain the museum holds the rights to those pictures and collection elements.

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
