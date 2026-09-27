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

𝗖𝗮𝗻 𝗟𝗼𝗴𝗶𝗰 𝗚𝗮𝘁𝗲 𝗡𝗲𝘁𝘄𝗼𝗿𝗸𝘀 𝘁𝘂𝗿𝗻 𝘄𝗶𝗱𝘁𝗵 𝗶𝗻𝘁𝗼 𝗲𝗳𝗳𝗲𝗰𝘁𝗶𝘃𝗲 𝗰𝗮𝗽𝗮𝗰𝗶𝘁𝘆?

We find that **naively increasing network width does not necessarily translate into effective model capacity**. Instead, additional logic kernels become increasingly redundant, leading to inefficient kernel utilization and performance saturation.

To address this, we introduce **Dynamic Logic Kernel (DLK)**, which dynamically routes each input to specialized kernel groups while keeping inference entirely within the logic-gate domain. Building on our level-wise analysis, we further propose **Early-stage Dynamic Logic Kernel (EDLK)**, which focuses dynamic routing on the first gate level where kernel redundancy is most pronounced.

By turning redundant width into specialized capacity, our approach improves kernel diversity, model utilization, and accuracy with better parameter efficiency.


# Pretrained Models

> [!IMPORTANT]
> **Pretrained models will be released soon.**
> 
# Logic Tree Edit Distance

> **Measure structural diversity between learned logic trees.**

We compute the minimum Tree Edit Distance (TED) for every pair of logic trees in the target layer.

- **Mean TED**: average TED across all tree pairs.
- **Normalized Mean TED**: Mean TED divided by the maximum possible TED.

Higher values indicate more diverse logic kernels, while lower values indicate greater redundancy.

```bash
python analyze_logic_ted.py \
  --resume path/to/checkpoint.pt \
  --architecture <architecture> \
  --dataset <dataset> \
  --layer conv3
```


