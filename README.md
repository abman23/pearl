# PEARL: Peer‑Enhanced Adaptive Radio via On‑Device LLM
[![arXiv](https://img.shields.io/badge/arXiv-2509.24085-b31b1b.svg)](https://arxiv.org/abs/2509.24085)

This is the official repository for the paper **“PEARL: Peer‑Enhanced Adaptive Radio via On‑Device LLM”** accepted at **NeurIPS'25 (AI4NextG Workshop)** (arXiv:2509.24085, 2025).

## Demo Videos
<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://youtu.be/7m9UCowiKh4">
        <img src="https://img.youtube.com/vi/7m9UCowiKh4/0.jpg" width="280"><br>
        iPad Pro Demo (NeurIPS'25 @AI4NextG)
      </a>
    </td>
    <td align="center">
      <a href="https://youtu.be/yK59pwtLrj8">
        <img src="https://img.youtube.com/vi/yK59pwtLrj8/0.jpg" width="280"><br>
        Captioned Walkthrough Demo
      </a>
    </td>
  </tr>
</table>
</div>




## Highlights

- **Peer‑aware cross‑layer control** – first on‑device LLM framework that uses both publisher and subscriber context to guide Wi‑Fi Aware parameter selection.
- **Reward‑aligned training** – latency is normalized by application requirements and energy scaled by battery state, yielding soft labels for KL‑based fine‑tuning.
- **Efficient variants** – PEARL (Head + LoRA) attains the highest objective score (7.58), while PEARL‑Lite (Head only) delivers near‑identical performance (7.54) with inference under 20 ms.
- **Energy savings** – incorporating peer information reduces energy consumption by up to ~16 % in cooperative low‑battery scenarios without sacrificing latency.
- **Real‑world validation** – prototype iPadOS app integrates Apple’s on‑device Foundation Models and dynamically updates WA parameters during D2D communication.


## Overview

We introduce **PEARL**, a cooperative cross‑layer optimization framework that leverages a pre‑trained on‑device language model for device‑to‑device (D2D) communication. Unlike prior single‑device solutions, PEARL combines local context (e.g., battery level, application type) with **peer information** from the neighboring device, enabling two‑sided adaptation. 

A **context‑aware reward function** normalizes latency by application tolerances and scales energy by device battery state, providing richer supervision for KL‑based fine‑tuning. We present two variants—**PEARL** (Head + LoRA) and **PEARL‑Lite** (Head only)—that achieve superior objective scores compared to rule‑based and LoRA‑only baselines, while maintaining sub‑20 ms inference on mobile hardware.

## Quick Start

### 1. Clone
```sh
git clone https://github.com/yqlu1015/PEARL.git
cd PEARL
````

### 2. Environment

Create and activate a Python environment (tested with Python 3.10):

```sh
conda create -n pearl_env python=3.10
conda activate pearl_env
pip install -r requirements.txt
```

### 3. Train a Model

We provide scripts to fine‑tune the pre‑trained backbone on the WA dataset.

**PEARL (Head + LoRA)** – trains a classification head and LoRA adapters:

```sh
python training/wa_sft.py --with_head --new_model_name pearl
```

**PEARL‑Lite (Head only)** – trains only the classification head for faster inference:

```sh
python training/wa_sft.py --with_head --no_adapter --new_model_name pearl-lite
```

Additional arguments (loss type, epochs, etc.) can be specified via command‑line flags. See `training/wa_sft.py` for details.

### 4. Evaluate

After training, evaluate the model on the WA test set:

```sh
python evaluation/wa_eval.py --model output/pearl      # evaluate PEARL
python evaluation/wa_eval.py --model output/pearl-lite # evaluate PEARL‑Lite
```

The script reports objective, latency and energy scores as defined in the paper.

### 5. Demo App

A fully interactive app is located under `demo_app/WAParameterTuning/`. It requires two iOS or iPadOS devices and a Mac. The publisher runs the on‑device LLM agent, gathers local and peer context and updates WA parameters in real time. Refer to `demo_app/WAParameterTuning/README.md` for build instructions.

## Citation

```
@article{lee2025pearl,
  title   = {{PEARL}: Peer-Enhanced Adaptive Radio via On-Device LLM},
  author  = {Lee, Ju-Hyung and Lu, Yanqing and Doppler, Klaus},
  journal = {arXiv preprint arXiv:2509.24085},
  year    = {2025}
}
```

## Contributors

Special thanks to Yanqing Lu who contributed to the simulations and real‑world prototyping.
