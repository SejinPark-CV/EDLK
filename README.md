# [NeurIPS 2026] Overcoming Kernel Redundancy for Scaling Logic Gate Networks

### Overcoming Kernel Redundancy for Scaling Logic Gate Networks

**Sejin Park**, Hongjae Lee, Changwoo Han, Seung-Won Jung  
Conference on Neural Information Processing Systems (**NeurIPS 2026**)

This repository contains the code for our **Early-stage Dynamic Logic Kernel (EDLK)** experiments.

EDLK is designed to reduce redundancy among logic kernels by applying input-dependent routing only at the **first gate level** of a logic kernel. Candidate groups have independent first-level logic gates, while the remaining upper gate levels are shared.

---

## Base Code

> [!NOTE]
> This project is built upon [LogicIR](https://github.com/jimmy9704/LogicIR) and [difflogic](https://github.com/Felix-Petersen/difflogic).

The repository includes the `difflogic` source code and builds the CUDA extension locally through `setup.py`.

The current `setup.py` builds only the standard `difflogic_cuda` extension required by the MNIST EDLK experiments. The optional IWP CUDA extension is not built.

---

## Environment Setup

The provided `requirements.txt` uses **PyTorch 2.9.1** and Python 3.10 is recommended.

```bash
conda create -n edlk python=3.10
conda activate edlk

pip install -r requirements.txt
pip install -e . --no-build-isolation
```

Because `difflogic` contains a custom CUDA extension, a working CUDA toolkit and C++ compiler are required. You can check the environment with:

```bash
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
nvcc --version
g++ --version
```

If the editable CUDA build fails, the following command is useful for obtaining the first compiler error clearly:

```bash
MAX_JOBS=1 \
SETUPTOOLS_ENABLE_FEATURES=legacy-editable \
pip install -e . --no-build-isolation -v
```

After installation:

```bash
python -c "import difflogic; print('difflogic import OK')"
```

---

## Repository Structure

```text
EDLK/
├── difflogic/
│   ├── cuda/
│   ├── difflogic.py
│   ├── functional.py
│   ├── packbitstensor.py
│   └── ...
│
├── experiments/
│   ├── analyze_logic_ted.py
│   ├── infer_mnist.py
│   ├── mnist_dataset.py
│   ├── model_mnist.py
│   ├── train_mnist.py
│   ├── run_infer_mnist.sh
│   ├── run_mnist.sh
│   │
│   ├── checkpoints/
│   │   └── mnist_edlk_25.pt
│   │
│   ├── data-mnist/
│   └── logs/
│
├── requirements.txt
├── setup.py
└── README.md
```

The commands below assume that the environment has already been installed from the repository root and that you then move to the experiment directory:

```bash
cd experiments
```

---

## MNIST EDLK Architecture

The current MNIST implementation uses three convolutional logic stages. Only the final convolutional stage is dynamic.

```text
Input: MNIST 28 x 28
        │
        ▼
Static LogicConv
  width = W
  receptive field = 5 x 5
  tree depth = 3
  no padding
        │
        ▼
2 x 2 max pooling
        │
        ▼
Static LogicConv
  width = 3W
  receptive field = 3 x 3
  tree depth = 3
  padding
        │
        ▼
2 x 2 max pooling
        │
        ▼
EDLK LogicConv
  K candidate groups
  width per group = 9W / K
  total candidate-bank width = 9W
  receptive field = 3 x 3
  tree depth = 3
        │
        ▼
2 x 2 max pooling
        │
        ▼
LogicLayer: 9W/K x 3 x 3 -> 500W
        │
        ▼
LogicLayer: 500W -> 320W
        │
        ▼
LogicLayer: 320W -> 120W
        │
        ▼
GroupSum -> 10 classes
```

Inside the EDLK block:

- each candidate has its own `tree_layers.0.weights`;
- the upper logic-tree parameters are shared across candidates;
- the router produces one score per candidate group;
- training uses soft routing with `softmax(score / tau_router)`;
- evaluation uses deterministic hard routing with `argmax`.

For the default experiment in `run_mnist.sh`:

```text
W = 256
K = 2
router hidden width = 32
router bits = 15
router temperature = 1.0
```

The routed output width of the dynamic convolution is therefore `9W/K`.

> [!NOTE]
> The current CUDA evaluation path computes complete candidate-tree outputs and then applies the hard one-hot routing decision. This preserves the hard-routed output while avoiding operations on packed intermediate tensors. The reference code therefore does not implement sparse candidate execution for runtime benchmarking.

---

## Dataset

The training code automatically downloads MNIST to:

```text
experiments/data-mnist/
```

when it is not already available.

If you want to run inference on a fresh clone without training first, download the dataset from inside `experiments/`:

```bash
python - <<'PY'
import mnist_dataset

mnist_dataset.MNIST(
    "./data-mnist",
    train=True,
    download=True,
    remove_border=False,
)
PY
```

The MNIST downloader retrieves both the training and test raw files.

---

## Training

From the `experiments/` directory:

```bash
bash run_mnist.sh
```

The active configuration in `run_mnist.sh` is:

```bash
CUDA_VISIBLE_DEVICES=0 python train_mnist.py \
  --implementation cuda \
  --connections random \
  --tau 20 \
  --tau-router 1.0 \
  --router-bits 15 \
  --grad-factor 2.0 \
  --channels 256 \
  --K 2 \
  --router-hidden 32 \
  --num-iterations 200000 \
  --batch-size 64 \
  --eval-freq 2000 \
  --learning-rate 0.01 \
  --load-balance-weight 0.01 \
  --seed 0 \
  --curve-csv logs/mnist_edlk_20.csv \
  --save-ckpt checkpoints/mnist_edlk_20.pt
```

Training uses AdamW and the objective

```text
Cross-Entropy + lambda_bal * Load-Balancing Loss
```

with soft routing during training.

The training script evaluates the train, validation, and test splits every `--eval-freq` iterations. Test-set routing counts are accumulated over the entire test set.

Example output:

```text
[eval] iter=... train=... valid=... test=... best_test=...
[routing:test] conv3: fraction=[..., ...], counts=[..., ...], total=10000
```

A checkpoint is saved when validation accuracy improves. The checkpoint contains:

```text
model
optimizer
iter
best_valid_acc
best_test_acc
test_acc
args
```

The stored training arguments are later reused by the inference and TED-analysis scripts.

---

## Inference

A pretrained MNIST checkpoint is included at:

```text
experiments/checkpoints/mnist_edlk_25.pt
```

The included checkpoint was saved with:

```text
tau = 25
W = 256
K = 2
router bits = 15
router hidden width = 32
seed = 0
connections = random
```

To evaluate it:

```bash
bash run_infer_mnist.sh
```

which runs:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_mnist.py \
  --checkpoint checkpoints/mnist_edlk_25.pt \
  --batch-size 64 \
  --num-workers 4
```

`infer_mnist.py` reconstructs the model using the arguments stored in the checkpoint. In particular, it restores the training seed before model construction so that random logic connections are reproduced consistently.

The model state is loaded with strict matching:

```text
missing_keys=[]
unexpected_keys=[]
```

Inference uses hard routing and reports both classification accuracy and routing usage over the full MNIST test set.

Example:

```text
test accuracy: ...
total samples: 10000
[routing:test] conv3: fraction=[..., ...], counts=[..., ...], total=10000
```

---

## Load Balancing

For each dynamic EDLK block, the implementation tracks:

- the average soft routing mass across the mini-batch;
- the hard group assignment obtained from the largest routing weight.

The load-balancing term is added to cross-entropy during training:

```text
loss = CE + load_balance_weight * L_bal
```

The default shell script uses:

```text
load_balance_weight = 0.01
```

---

## Logic Tree Edit Distance

`analyze_logic_ted.py` measures structural diversity between learned logic trees.

For each pair of trees, the script computes an unordered fixed-topology tree distance. At every node it compares both possible child alignments and uses the lower-cost alignment. Gate substitution cost is the Hamming distance between the two 4-entry Boolean truth tables.

The main reported metrics are:

- **Mean TED**: mean pairwise distance over all unique tree pairs.
- **Normalized Mean TED**: Mean TED divided by the maximum possible distance, `4 * (2^tree_depth - 1)`.

For an EDLK block, selecting the block itself pools trees from all candidate groups.

### Analyze all candidate groups in `conv3`

```bash
CUDA_VISIBLE_DEVICES=0 python analyze_logic_ted.py \
  --architecture mnist_edlk \
  --dataset mnist \
  --resume checkpoints/mnist_edlk_25.pt \
  --layer conv3 \
  --device cuda \
  --save \
  --tag mnist_edlk
```

### Analyze one candidate group

```bash
CUDA_VISIBLE_DEVICES=0 python analyze_logic_ted.py \
  --architecture mnist_edlk \
  --dataset mnist \
  --resume checkpoints/mnist_edlk_25.pt \
  --layer conv3.trees.0 \
  --device cuda \
  --save \
  --tag mnist_edlk_tree0
```

When `--save` is enabled, results are written to:

```text
analysis_logic_ted/
```

including:

```text
*_logic_ted_matrix.npy
*_logic_ted_summary.csv
*_logic_ted_trees.csv
```

---

## Reproducibility

Random connection topology depends on the random seed used before model construction.

The training code sets:

```python
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
```

The inference and MNIST TED-analysis scripts restore the corresponding checkpoint arguments before constructing the model.

For checkpoint-based evaluation, use the model definition and `difflogic` implementation from the same repository revision that produced the checkpoint.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{park2026overcoming,
  title     = {Overcoming Kernel Redundancy for Scaling Logic Gate Networks},
  author    = {Park, Sejin and Lee, Hongjae and Han, Changwoo and Jung, Seung-Won},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## Acknowledgements

This project builds upon:

- [LogicIR](https://github.com/jimmy9704/LogicIR)
- [difflogic](https://github.com/Felix-Petersen/difflogic)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732)
