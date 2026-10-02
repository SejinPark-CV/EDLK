# [NeurIPS 2026] Overcoming Kernel Redundancy for Scaling Logic Gate Networks

### [Overcoming Kernel Redundancy for Scaling Logic Gate Networks](https://arxiv.org/abs/2610.01069)

**Sejin Park**, Hongjae Lee, Changwoo Han, Seung-Won Jung  
Conference on Neural Information Processing Systems (**NeurIPS 2026**)

<p align="center">
  <img src="EDLK.png" width="85%">
</p>

This repository provides the implementation of **Early-stage Dynamic Logic Kernel (EDLK)**.

EDLK addresses kernel redundancy in logic gate networks through input-dependent routing, while applying dynamic selection only to the first gate level of the logic kernel.

---

## Base Code

> [!NOTE]
> **This project is built upon [LogicIR](https://github.com/jimmy9704/LogicIR), which serves as the base codebase for our implementation.**

---

## 🛠️ Environment Setup

All experiments in this repository were conducted with **PyTorch 2.9.1 + CUDA 12.2**.

```bash
conda create -n edlk python=3.10
conda activate edlk
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

---

## Pretrained Model

The pretrained MNIST EDLK model is available at:

[Google Drive](https://drive.google.com/file/d/1_i7isdWbUNzN4_l_7B0sFeWD_DsxY7Tp/view?usp=sharing)

Place the downloaded checkpoint in:

```text
experiments/checkpoints/mnist_edlk_25.pt
```

---

## 🚀 Training

```bash
cd experiments
bash run_mnist.sh
```

---

## ✅ Inference

After downloading the pretrained model, run:

```bash
cd experiments
bash run_infer_mnist.sh
```

The inference script reports the MNIST test accuracy and routing statistics over the full test set.

---

## Logic Tree Edit Distance

We provide `analyze_logic_ted.py` to measure structural diversity between learned logic trees.

For MNIST EDLK:

```bash
cd experiments

CUDA_VISIBLE_DEVICES=0 python analyze_logic_ted.py \
  --architecture mnist_edlk \
  --dataset mnist \
  --resume checkpoints/mnist_edlk_25.pt \
  --layer conv3 \
  --device cuda \
  --save \
  --tag mnist_edlk
```

The script reports:

- **Mean TED**
- **Normalized Mean TED**

Higher TED values indicate greater structural diversity among logic kernels.
