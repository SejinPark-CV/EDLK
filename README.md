# [NeurIPS'26] Overcoming Kernel Redundancy for Scaling Logic Gate Networks

### Overcoming Kernel Redundancy for Scaling Logic Gate Networks

**Sejin Park**, Hongjae Lee, Changwoo Han, Seung-Won Jung  
Conference on Neural Information Processing Systems (**NeurIPS 2026**)

<p align="center">
  <img src="EDLK.png" width="85%">
</p>

## Base Code

> [!NOTE]
> **This project is built upon [LogicIR](https://github.com/jimmy9704/LogicIR), which serves as the base codebase for our implementation.**  

# Abstract

Differentiable logic gate networks, which operate using only logic gates, have recently attracted attention as an efficient alternative to conventional neural networks. However, despite their efficiency, the scaling behavior of logic gate networks remains underexplored. By contrast, scaling model capacity is a central design principle in deep neural networks and typically leads to improved performance. This discrepancy raises a key question: Can similar scaling benefits also be achieved in logic gate networks? In this work, we focus on width as a primary scaling axis and conduct a systematic analysis of its behavior in logic gate networks. We observe that naive width scaling often introduces redundancy among logic kernels, limiting the effective use of additional kernels and leading to performance saturation. To address this limitation, we propose a dynamic logic kernel framework that reorganizes kernel utilization by promoting specialization across kernel groups. This enables the network to better utilize increased width via input-dependent kernel routing, while ensuring that both routing and computation are implemented entirely with gate-level Boolean operations at inference time. We further find that kernel redundancy is most pronounced at the first gate level, motivating an early-stage dynamic logic kernel strategy that concentrates adaptation at this level. Experimental results demonstrate that our approach improves kernel utilization and increases kernel diversity, leading to higher accuracy with improved parameter efficiency.


# Performance and Efficiency

For direct comparison, we report both **inference-time parameter counts and Top-1 accuracy**, allowing performance and model efficiency to be compared together.

# Code Details

```text
Updating...
```

The code and pretrained models will be released.
**
