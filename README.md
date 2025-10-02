# PEARL: Peer-Enhanced Adaptive Radio via On-Device LLM

This is the official repository for the paper **“PEARL: Peer-Enhanced Adaptive Radio via On-Device LLM”**.

## Demo Videos
<table>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=U9QFzw7fJMQ">
        <img src="https://img.youtube.com/vi/U9QFzw7fJMQ/0.jpg" width="280"><br>
        Outdoor Campus Demo
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=UR13hyJkE8k">
        <img src="https://img.youtube.com/vi/UR13hyJkE8k/0.jpg" width="280"><br>
        Indoor Street Demo
      </a>
    </td>
  </tr>
</table>


## Overview

We present **PEARL** (*Peer-Enhanced Adaptive Radio via On-Device LLM*), a framework for cooperative cross-layer optimization in device-to-device (D2D) communication. Unlike prior on-device LLM approaches limited to single-device context, PEARL leverages both publisher and subscriber states to guide Wi-Fi Aware parameter selection. A context-aware reward, which normalizes latency by application tolerances and modulates energy by device battery states, provides richer supervision for KL-based fine-tuning. We study two lightweight variants: **PEARL** (Head+LoRA) achieves the best overall performance, while **PEARL-Lite** (Head-only) delivers sub-20 ms inference at near-identical objective scores. Across synthetic scenarios grounded in real measurements, PEARL improves objective scores over heuristic and compact model baselines and reduces energy by up to 16% in cooperative low-battery cases. These results demonstrate that peer-aware context, reward-aligned training, and head-based efficiency make LLMs practical for always-on, on-device cross-layer control.


## Quick Start

### 1. Clone
   ```sh
   git clone https://github.com/yqlu1015/PEARL
   ```

### 2. Environment Setup
   ```sh
   cd PEARL/
   conda create -n pearl_env python=3.10
   conda activate pearl_env
   pip install -r requirements.txt
   ```
  
### 3. Post-training

**PEARL**:
```sh
python training/wa_sft.py --with_head --new_model_name pearl
```
**PEARL-Lite**:
   ```sh
python training/wa_sft.py --with_head --no_adapter --new_model_name pearl-lite
   ```

Check out the training script (`training/wa_sft.py`) for more details of the arguments.

### 4. Inference
```sh
python evaluation/wa_eval.py --model [output/pearl or output/pearl-lite]
```


## Play with Demo App
You need two iOS or iPadOS devices and one computer running MacOS to run the demo app. Please refer to `demo_app/WAParameterTuning/README.md` for more details.

## Citation
```

```