# [NeurIPS 2026] Overcoming Kernel Redundancy for Scaling Logic Gate Networks

### Overcoming Kernel Redundancy for Scaling Logic Gate Networks

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
