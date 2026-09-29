"""Pairwise Logic Tree Edit Distance (TED) analysis.

This script measures structural diversity between logic-kernel trees in a
trained Differentiable Logic Gate Network.

For every pair of trees in a selected convolutional logic layer, it computes
an unordered fixed-topology TED. At each node, both child matchings are tested:

    (left <-> left, right <-> right)
    (left <-> right, right <-> left)

and the lower-cost matching is used recursively.

Node substitution cost is the Hamming distance between the 4-entry Boolean
truth tables of the two gates (range: 0..4).

Main reported metrics:
    Mean TED
        Mean pairwise minimum TED over all unique tree pairs.

    Normalized Mean TED
        Mean TED divided by the maximum possible TED for the tree depth:
            4 * (2**tree_depth - 1)

Notes:
- Tree topology is assumed to be the same full binary topology for all trees.
- This is therefore a fixed-topology unordered substitution distance, not a
  general insertion/deletion tree edit distance.
- For a dynamic block, trees from all experts are pooled and compared.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from difflogic import LogicLayerIWP
from model_mnist import MNISTLastDynamicMatchedWidth

# -----------------------------------------------------------------------------
# Dataset / model helpers
# -----------------------------------------------------------------------------


def input_channels_of_dataset(dataset: str) -> int:
    return {
        "cifar-10-3-thresholds": 9,
        "cifar-10-31-thresholds": 93,
        "mnist": 1,
    }[dataset]


def num_classes_of_dataset(dataset: str) -> int:
    return {
        "cifar-10-3-thresholds": 10,
        "cifar-10-31-thresholds": 10,
        "mnist": 10,
    }[dataset]


def extract_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_checkpoint_state(path: str, device: str) -> dict:
    checkpoint = torch.load(path, map_location=device)
    return extract_state_dict_from_checkpoint(checkpoint)


def load_checkpoint_args(path: str, device: str) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        return checkpoint["args"]
    return {}


def apply_mnist_checkpoint_args(args, checkpoint_args: dict) -> None:
    """Use the exact MNIST training configuration saved in the checkpoint."""
    if not checkpoint_args:
        return

    keys = (
        "seed",
        "tau",
        "grad_factor",
        "implementation",
        "connections",
        "channels",
        "K",
        "router_hidden",
        "router_bits",
        "tau_router",
    )

    for key in keys:
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])

    print(
        "[mnist checkpoint args] "
        f"seed={args.seed}, "
        f"connections={args.connections}, "
        f"implementation={args.implementation}, "
        f"channels={args.channels}, "
        f"K={args.K}, "
        f"router_hidden={args.router_hidden}, "
        f"router_bits={args.router_bits}, "
        f"tau_router={args.tau_router}"
    )


def build_model(args) -> torch.nn.Module:
    common = dict(
        tau=args.tau,
        grad_factor=args.grad_factor,
        device=args.device,
        implementation=args.implementation,
        connections=args.connections,
        channels=args.channels,
    )

    if args.architecture == "logicnet_cifar3t":
        # Import only when CIFAR-10 analysis is actually requested.
        # MNIST analysis therefore does not depend on modelsClassification.
        from modelsClassification import LogicTreeNetS_CIFAR10

        if args.dataset not in {
            "cifar-10-3-thresholds",
            "cifar-10-31-thresholds",
        }:
            raise ValueError(
                "architecture='logicnet_cifar3t' requires a CIFAR-10 dataset."
            )

        model = LogicTreeNetS_CIFAR10(**common)

    elif args.architecture == "mnist_edlk":
        if args.dataset != "mnist":
            raise ValueError(
                "architecture='mnist_edlk' requires --dataset mnist."
            )

        print("[model] MNISTLastDynamicMatchedWidth")
        model = MNISTLastDynamicMatchedWidth(
            tau=args.tau,
            grad_factor=args.grad_factor,
            device=args.device,
            implementation=args.implementation,
            connections=args.connections,
            channels=args.channels,
            K=args.K,
            router_hidden=args.router_hidden,
            router_bits=args.router_bits,
            tau_router=args.tau_router,
            hard_infer=True,
            in_dim=1,
            num_classes=10,
        )

    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    return model.to(args.device)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    device: str,
    allow_mismatch: bool,
) -> None:
    state = load_checkpoint_state(path, device)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if not allow_mismatch and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing keys ({len(missing)}): {missing[:10]}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}\n"
            "Use the correct architecture/variant, or pass --allow-mismatch."
        )

    if missing:
        print(f"[warning] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warning] unexpected keys: {len(unexpected)}")


def resolve_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Resolve a dotted module path such as 'conv3' or 'conv3.trees.0'."""
    module = model

    for part in name.split("."):
        if part.isdigit():
            index = int(part)
            if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)):
                module = module[index]
            else:
                raise AttributeError(
                    f"Module {module.__class__.__name__} is not indexable, "
                    f"but path contains index '{part}'."
                )
        else:
            if not hasattr(module, part):
                raise AttributeError(
                    f"Module {module.__class__.__name__} has no attribute '{part}'."
                )
            module = getattr(module, part)

    return module


# -----------------------------------------------------------------------------
# Logic gate / tree helpers
# -----------------------------------------------------------------------------


def gate_id_from_logic_weights(weights: torch.Tensor) -> torch.Tensor:
    """Convert standard 16-way logic weights to discrete gate IDs."""
    return weights.argmax(dim=-1).to(torch.int64)


def gate_id_from_iwp_weights(weights: torch.Tensor) -> torch.Tensor:
    """Convert IWP 4-bit gate parameters to IDs in [0, 15]."""
    bits = (0.5 + 0.5 * torch.sin(weights) >= 0.5).to(torch.int64)
    return bits[:, 0] * 8 + bits[:, 1] * 4 + bits[:, 2] * 2 + bits[:, 3]


def layer_gate_ids(logic_layer: torch.nn.Module) -> torch.Tensor:
    weights = logic_layer.weights.detach()
    if isinstance(logic_layer, LogicLayerIWP) or weights.shape[-1] == 4:
        return gate_id_from_iwp_weights(weights)
    return gate_id_from_logic_weights(weights)


def gate_truth_table(gate_id: int) -> Tuple[int, int, int, int]:
    return (
        (gate_id >> 3) & 1,
        (gate_id >> 2) & 1,
        (gate_id >> 1) & 1,
        gate_id & 1,
    )


def build_gate_distance_matrix() -> np.ndarray:
    """Hamming distance between all pairs of 4-entry Boolean truth tables."""
    matrix = np.zeros((16, 16), dtype=np.int64)

    for i in range(16):
        truth_i = gate_truth_table(i)
        for j in range(16):
            truth_j = gate_truth_table(j)
            matrix[i, j] = sum(a != b for a, b in zip(truth_i, truth_j))

    return matrix


GATE_DISTANCE = build_gate_distance_matrix()


class TreeNode:
    __slots__ = ("gate_id", "left", "right")

    def __init__(
        self,
        gate_id: int,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ) -> None:
        self.gate_id = int(gate_id)
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def build_tree_from_per_level(
    levels: List[np.ndarray],
    level_idx: int,
    offset: int,
) -> TreeNode:
    gate_id = int(levels[level_idx][offset])

    if level_idx == 0:
        return TreeNode(gate_id)

    left = build_tree_from_per_level(levels, level_idx - 1, offset * 2)
    right = build_tree_from_per_level(levels, level_idx - 1, offset * 2 + 1)
    return TreeNode(gate_id, left, right)


def extract_trees_from_single_logic_conv(layer: torch.nn.Module) -> List[TreeNode]:
    required = ("tree_layers", "tree_d", "out_dim")
    if not all(hasattr(layer, attr) for attr in required):
        raise ValueError("Layer is not LogicConvLayer-like.")

    tree_depth = int(layer.tree_d)
    out_dim = int(layer.out_dim)
    per_level_gate_ids = []

    for depth_idx, tree_logic_layer in enumerate(layer.tree_layers):
        gate_ids = layer_gate_ids(tree_logic_layer).cpu().numpy()
        nodes_per_tree = 2 ** (tree_depth - 1 - depth_idx)
        expected = out_dim * nodes_per_tree

        if gate_ids.shape[0] != expected:
            raise RuntimeError(
                f"Level {depth_idx}: expected {expected} gates, got {gate_ids.shape[0]}."
            )

        per_level_gate_ids.append(gate_ids.reshape(out_dim, nodes_per_tree))

    trees = []
    for channel in range(out_dim):
        levels_for_tree = [level[channel] for level in per_level_gate_ids]
        trees.append(
            build_tree_from_per_level(
                levels_for_tree,
                level_idx=tree_depth - 1,
                offset=0,
            )
        )

    return trees


def extract_trees_from_layer(layer: torch.nn.Module) -> Tuple[List[TreeNode], dict]:
    """Extract trees from one LogicConvLayer or all experts in a dynamic block."""
    if all(hasattr(layer, attr) for attr in ("tree_layers", "tree_d", "out_dim")):
        trees = extract_trees_from_single_logic_conv(layer)
        return trees, {
            "source_mode": "single_logic_conv",
            "num_experts": 1,
            "trees_per_expert": [len(trees)],
            "tree_depth": int(layer.tree_d),
        }

    if hasattr(layer, "trees"):
        all_trees: List[TreeNode] = []
        trees_per_expert: List[int] = []
        tree_depth: Optional[int] = None

        for expert_idx, expert in enumerate(layer.trees):
            expert_trees = extract_trees_from_single_logic_conv(expert)
            expert_depth = int(expert.tree_d)

            if tree_depth is None:
                tree_depth = expert_depth
            elif expert_depth != tree_depth:
                raise RuntimeError(
                    "All experts must have the same tree depth. "
                    f"Expected {tree_depth}, got {expert_depth} at expert {expert_idx}."
                )

            all_trees.extend(expert_trees)
            trees_per_expert.append(len(expert_trees))

        if tree_depth is None:
            raise RuntimeError("Dynamic block contains no experts.")

        return all_trees, {
            "source_mode": "dynamic_block_all_experts",
            "num_experts": len(layer.trees),
            "trees_per_expert": trees_per_expert,
            "tree_depth": tree_depth,
        }

    raise ValueError(
        "Selected layer is neither a LogicConvLayer-like module nor a dynamic "
        "block containing '.trees'."
    )


# -----------------------------------------------------------------------------
# TED
# -----------------------------------------------------------------------------


def unordered_logic_ted(a: TreeNode, b: TreeNode) -> int:
    """Minimum unordered fixed-topology TED between two full binary trees."""
    substitution_cost = int(GATE_DISTANCE[a.gate_id, b.gate_id])

    if a.is_leaf and b.is_leaf:
        return substitution_cost

    if a.is_leaf != b.is_leaf:
        raise RuntimeError(
            "Tree topology mismatch. This analysis assumes equal full-binary topology."
        )

    no_swap = unordered_logic_ted(a.left, b.left) + unordered_logic_ted(
        a.right, b.right
    )
    swap = unordered_logic_ted(a.left, b.right) + unordered_logic_ted(
        a.right, b.left
    )

    return substitution_cost + min(no_swap, swap)


def pairwise_ted_matrix(trees: List[TreeNode]) -> np.ndarray:
    """Compute TED for every unique pair of trees."""
    num_trees = len(trees)
    matrix = np.zeros((num_trees, num_trees), dtype=np.int64)

    for i in range(num_trees):
        for j in range(i + 1, num_trees):
            distance = unordered_logic_ted(trees[i], trees[j])
            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix


def summarize_ted(matrix: np.ndarray, tree_depth: int) -> dict:
    """Return the two primary metrics used for kernel-diversity analysis."""
    num_trees = matrix.shape[0]
    pairwise_values = matrix[np.triu_indices(num_trees, k=1)]
    max_ted = 4 * ((2**tree_depth) - 1)

    mean_ted = float(pairwise_values.mean()) if pairwise_values.size else 0.0
    normalized_mean_ted = mean_ted / max_ted if max_ted > 0 else 0.0

    return {
        "num_trees": int(num_trees),
        "num_pairs": int(pairwise_values.size),
        "tree_depth": int(tree_depth),
        "max_ted": int(max_ted),
        "mean_ted": mean_ted,
        "normalized_mean_ted": float(normalized_mean_ted),
    }


def tree_to_preorder(node: TreeNode) -> List[int]:
    values = [node.gate_id]
    if not node.is_leaf:
        values.extend(tree_to_preorder(node.left))
        values.extend(tree_to_preorder(node.right))
    return values


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def save_outputs(
    out_dir: str,
    tag: str,
    layer_name: str,
    matrix: np.ndarray,
    summary: dict,
    trees: List[TreeNode],
    metadata: dict,
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{tag}_{layer_name.replace('.', '_')}_logic_ted"
    matrix_path = output_dir / f"{stem}_matrix.npy"
    summary_path = output_dir / f"{stem}_summary.csv"
    trees_path = output_dir / f"{stem}_trees.csv"

    np.save(matrix_path, matrix)

    summary_row = {
        **summary,
        "source_mode": metadata["source_mode"],
        "num_experts": metadata["num_experts"],
        "trees_per_expert": " ".join(map(str, metadata["trees_per_expert"])),
    }

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    with trees_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tree_idx", "preorder_gate_ids"])
        for tree_idx, tree in enumerate(trees):
            writer.writerow([tree_idx, " ".join(map(str, tree_to_preorder(tree)))])

    print(f"[save] matrix  : {matrix_path}")
    print(f"[save] summary : {summary_path}")
    print(f"[save] trees   : {trees_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    # MNIST EDLK checkpoints store the exact training args. Restore them before
    # seeding/model construction so random connection topology is reproduced.
    if args.architecture == "mnist_edlk" and args.resume is not None:
        checkpoint_args = load_checkpoint_args(args.resume, args.device)
        apply_mnist_checkpoint_args(args, checkpoint_args)

    seed_everything(args.seed)

    if args.resume is not None and not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    model = build_model(args)

    if args.resume is not None:
        load_checkpoint(
            model,
            args.resume,
            args.device,
            allow_mismatch=args.allow_mismatch,
        )
        print(f"[checkpoint] loaded: {args.resume}")
    else:
        print("[checkpoint] none; analyzing a newly initialized model")

    model.eval()

    target_layer = resolve_module(model, args.layer)
    trees, metadata = extract_trees_from_layer(target_layer)
    tree_depth = int(metadata["tree_depth"])

    matrix = pairwise_ted_matrix(trees)
    summary = summarize_ted(matrix, tree_depth)

    print("\n" + "=" * 68)
    print("LOGIC TED")
    print("=" * 68)
    print(f"layer                    : {args.layer}")
    print(f"source mode              : {metadata['source_mode']}")
    print(f"num experts              : {metadata['num_experts']}")
    print(f"trees per expert         : {metadata['trees_per_expert']}")
    print(f"num trees                : {summary['num_trees']}")
    print(f"num pairs                : {summary['num_pairs']}")
    print(f"tree depth               : {summary['tree_depth']}")
    print(f"max TED                  : {summary['max_ted']}")
    print("-" * 68)
    print(f"Mean TED                 : {summary['mean_ted']:.6f}")
    print(f"Normalized Mean TED      : {summary['normalized_mean_ted']:.6f}")
    print("=" * 68)

    if args.save:
        save_outputs(
            out_dir=args.out_dir,
            tag=args.tag,
            layer_name=args.layer,
            matrix=matrix,
            summary=summary,
            trees=trees,
            metadata=metadata,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all-pairs minimum unordered Logic TED for a selected "
            "logic-convolution layer."
        )
    )

    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path.")
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=[
            "logicnet_cifar3t",
            "mnist_edlk",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "cifar-10-3-thresholds",
            "cifar-10-31-thresholds",
            "mnist",
        ],
    )
    parser.add_argument(
        "--layer",
        type=str,
        required=True,
        help="Examples: conv1, conv3, conv3.trees.0",
    )

    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--grad-factor", type=float, default=2.0)
    parser.add_argument(
        "--implementation",
        type=str,
        default="cuda",
        choices=["cuda", "cuda_ste", "python"],
    )
    parser.add_argument(
        "--connections",
        type=str,
        default="random",
        choices=["random", "unique"],
    )

    # Dynamic-model options.
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--router-hidden", type=int, default=64)
    parser.add_argument("--router-bits", type=int, default=4)
    parser.add_argument("--tau-router", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow partial checkpoint loading. Avoid for final analysis.",
    )

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out-dir", type=str, default="analysis_logic_ted")
    parser.add_argument("--tag", type=str, default="run")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
