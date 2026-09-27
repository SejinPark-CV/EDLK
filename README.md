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

# Logic Tree Diversity Analysis

> **Measure redundancy where it occurs — directly in the learned logic trees.**

We evaluate the structural diversity of learned logic kernels using **Logic Tree Edit Distance (TED)**.

For every pair of logic trees in the target layer, we compute the **minimum distance** under unordered binary-tree matching. At each node, both the original and left/right-swapped subtree alignments are considered, and the lower-cost alignment is selected recursively.

The substitution cost between two logic gates is defined as the **Hamming distance between their 4-bit Boolean truth tables**, ranging from 0 to 4.

We report:

- **Mean TED** — the average minimum TED over all pairs of logic trees.
- **Normalized Mean TED** — Mean TED normalized by the maximum possible distance for a tree of depth \(d\):

\[
\mathrm{Normalized\ Mean\ TED}
=
\frac{\mathrm{Mean\ TED}}
{4(2^d - 1)}
\]

Higher TED indicates greater structural diversity among learned logic kernels, while lower TED indicates greater kernel redundancy.

## Usage

```bash
python analyze_logic_ted.py \
  --resume path/to/checkpoint.pt \
  --architecture <architecture> \
  --dataset <dataset> \
  --layer conv3 \
  --channels <channels> \
  --save
```

For dynamic logic layers, all logic trees across the experts are included in the pairwise TED analysis.

The script optionally saves:

```text
*_matrix.npy    # Pairwise TED matrix
*_summary.csv   # Mean TED and Normalized Mean TED
*_trees.csv     # Extracted logic-tree gate IDs
```

# Comparison with Baselines

> **Compare where it matters — accuracy vs. inference-time parameters.**

We report **Top-1 accuracy alongside inference-time parameter counts** for direct comparison with existing logic gate network baselines.

# Code Details

> [!IMPORTANT]
> **Code and pretrained models are being prepared for release.**

```text
Coming soon.
```
