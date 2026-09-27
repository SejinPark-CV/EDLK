# [NeurIPS'26] Overcoming Kernel Redundancy for Scaling Logic Gate Networks

### Overcoming Kernel Redundancy for Scaling Logic Gate Networks

**Sejin Park**, Hongjae Lee, Changwoo Han, Seung-Won Jung  
Conference on Neural Information Processing Systems (**NeurIPS 2026**)

<p align="center">
  <img src="EDLK.png" width="85%">
</p>

# Abstract

Logic gate networks provide highly efficient inference using Boolean operations, but their scaling behavior remains underexplored.

We find that naively increasing network width introduces redundant logic kernels, leading to inefficient capacity utilization and performance saturation.

To address this problem, we propose **Dynamic Logic Kernel (DLK)**, which performs input-dependent kernel routing entirely within the logic-gate domain. We further identify that kernel redundancy is most pronounced at the first gate level and introduce **Early-stage Dynamic Logic Kernel (EDLK)**, which concentrates dynamic routing on first-level gate groups.

Our approach improves kernel utilization and enables logic gate networks to benefit more effectively from width scaling.

# Performance and Efficiency

For direct comparison, we report both **inference-time parameter counts and Top-1 accuracy**, allowing performance and model efficiency to be compared together.

# Code Details

```text
Updating...
```

The code and pretrained models will be released.
