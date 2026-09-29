import argparse
import random

import numpy as np
import torch

import mnist_dataset
from model_mnist import MNISTLastDynamicMatchedWidth


torch.set_num_threads(1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    routing_counts = {}

    for x, y in loader:
        x = x.to("cuda").round()
        y = y.to("cuda")

        out = model(x)
        pred = out.argmax(dim=-1)

        correct += (pred == y).sum().item()
        total += y.numel()

        # Accumulate hard-routing decisions over the ENTIRE test set.
        for name, module in model.named_modules():
            if not hasattr(module, "last_group_scores"):
                continue
            if module.last_group_scores is None:
                continue
            if not hasattr(module, "K"):
                continue

            hard_idx = module.last_group_scores.argmax(dim=-1)

            counts = torch.bincount(
                hard_idx,
                minlength=module.K,
            ).cpu()

            if name not in routing_counts:
                routing_counts[name] = counts
            else:
                routing_counts[name] += counts

    acc = correct / total

    print(f"test accuracy: {acc:.6f}")
    print(f"total samples: {total}")

    for name, counts in routing_counts.items():
        fractions = counts.float() / counts.sum()

        print(
            f"[routing:test] {name}: "
            f"fraction={[round(v, 6) for v in fractions.tolist()]}, "
            f"counts={counts.tolist()}, "
            f"total={int(counts.sum())}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    ckpt = torch.load(
        args.checkpoint,
        map_location="cuda",
    )

    train_args = ckpt["args"]

    # Match training-time RNG state before model construction.
    seed = int(train_args.get("seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 72)
    print(f"checkpoint : {args.checkpoint}")
    print(f"iteration  : {ckpt.get('iter', 'unknown')}")
    print(f"best valid : {ckpt.get('best_valid_acc', 'unknown')}")
    print(f"best test  : {ckpt.get('best_test_acc', 'unknown')}")
    print(f"seed       : {seed}")
    print(f"connections: {train_args.get('connections', 'random')}")
    print(f"impl       : {train_args.get('implementation', 'cuda')}")
    print(f"channels   : {train_args.get('channels', 64)}")
    print(f"K          : {train_args.get('K', 2)}")
    print(f"router bits: {train_args.get('router_bits', 4)}")
    print(f"router tau : {train_args.get('tau_router', 0.1)}")
    print("=" * 72)

    model = MNISTLastDynamicMatchedWidth(
        tau=train_args.get("tau", 10.0),
        grad_factor=train_args.get("grad_factor", 2.0),
        device="cuda",
        implementation=train_args.get("implementation", "cuda"),
        connections=train_args.get("connections", "random"),
        channels=train_args.get("channels", 64),
        K=train_args.get("K", 2),
        router_hidden=train_args.get("router_hidden", 32),
        router_bits=train_args.get("router_bits", 4),
        tau_router=train_args.get("tau_router", 0.1),
        hard_infer=True,
        in_dim=1,
        num_classes=10,
    ).cuda()

    load_result = model.load_state_dict(
        ckpt["model"],
        strict=True,
    )
    print("[checkpoint] state_dict loaded strictly")
    print(f"[checkpoint] missing_keys={load_result.missing_keys}")
    print(f"[checkpoint] unexpected_keys={load_result.unexpected_keys}")

    model.eval()

    test_set = mnist_dataset.MNIST(
        "./data-mnist",
        train=False,
        remove_border=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    evaluate(
        model,
        test_loader,
    )


if __name__ == "__main__":
    main()
