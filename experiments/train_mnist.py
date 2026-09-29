import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import mnist_dataset
from model_mnist import MNISTLastDynamicMatchedWidth


torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}


def load_dataset(args):
    train_set = mnist_dataset.MNIST(
        "./data-mnist",
        train=True,
        download=True,
        remove_border=False,
    )
    test_set = mnist_dataset.MNIST(
        "./data-mnist",
        train=False,
        remove_border=False,
    )

    train_size = math.ceil((1.0 - args.valid_set_size) * len(train_set))
    valid_size = len(train_set) - train_size

    split_generator = torch.Generator().manual_seed(args.seed)
    train_set, valid_set = torch.utils.data.random_split(
        train_set,
        [train_size, valid_size],
        generator=split_generator,
    )

    common = dict(
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, drop_last=True, **common
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, shuffle=False, drop_last=False, **common
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, drop_last=False, **common
    )
    return train_loader, valid_loader, test_loader


def load_n(loader, n):
    i = 0
    while i < n:
        for batch in loader:
            yield batch
            i += 1
            if i >= n:
                break


def build_model(args):
    model = MNISTLastDynamicMatchedWidth(
        tau=args.tau,
        grad_factor=args.grad_factor,
        device="cuda",
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
    ).cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    return model, loss_fn, optimizer


def train_step(model, x, y, loss_fn, optimizer, load_balance_weight):
    model.train()
    out = model(x)
    ce_loss = loss_fn(out, y)

    load_balance_loss = ce_loss.new_zeros(())
    if load_balance_weight > 0.0 and hasattr(model, "get_load_balance_loss"):
        load_balance_loss = model.get_load_balance_loss()

    # Same loss as before.
    total_loss = ce_loss + load_balance_weight * load_balance_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "ce_loss": ce_loss.item(),
        "load_balance_loss": load_balance_loss.item(),
    }


@torch.no_grad()
def evaluate(model, loader, train_mode=False, collect_routing=False):
    original_mode = model.training
    model.train(mode=train_mode)

    correct = 0
    total = 0
    routing_counts = {}

    for x, y in loader:
        x = x.to("cuda").round()
        y = y.to("cuda")
        pred = model(x).argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()

        if collect_routing:
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
                ).to(torch.long).cpu()

                if name not in routing_counts:
                    routing_counts[name] = counts
                else:
                    routing_counts[name] += counts

    model.train(mode=original_mode)

    acc = correct / total

    if not collect_routing:
        return acc

    routing_stats = {}
    for name, counts in routing_counts.items():
        n = int(counts.sum().item())
        if n == 0:
            continue
        routing_stats[name] = {
            "counts": counts,
            "fractions": counts.float() / n,
            "total": n,
        }

    return acc, routing_stats


def init_csv(path):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "iter",
                "loss",
                "ce_loss",
                "load_balance_loss",
                "train_acc_eval_mode",
                "valid_acc_eval_mode",
                "test_acc_eval_mode",
            ])


def append_csv(path, row):
    if path is None:
        return
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def print_routing_stats(stats, split="test"):
    for name, s in stats.items():
        counts = s["counts"].tolist()
        fractions = [round(v, 6) for v in s["fractions"].tolist()]
        print(
            f"[routing:{split}] {name}: "
            f"fraction={fractions}, counts={counts}, total={s['total']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="MNIST matched-width EDLK: only the last conv layer is dynamic."
    )

    parser.add_argument("--implementation", default="cuda", choices=["cuda", "cuda_ste", "python"])
    parser.add_argument("--connections", default="random", choices=["random", "unique"])
    parser.add_argument("--training-bit-count", type=int, default=32, choices=[16, 32, 64])

    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--tau-router", type=float, default=0.1)
    parser.add_argument("--router-bits", type=int, default=4)
    parser.add_argument("--router-hidden", type=int, default=32)
    parser.add_argument("--grad-factor", type=float, default=2.0)

    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--K", type=int, default=2)

    parser.add_argument("--num-iterations", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--load-balance-weight", type=float, default=0.01)

    parser.add_argument("--valid-set-size", type=float, default=10000 / 60000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--curve-csv", type=str, default=None)
    parser.add_argument("--save-ckpt", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    if args.channels % args.K != 0:
        raise ValueError(
            f"Matched width requires channels % K == 0, got "
            f"channels={args.channels}, K={args.K}."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, valid_loader, test_loader = load_dataset(args)
    model, loss_fn, optimizer = build_model(args)

    num_params = sum(p.numel() for p in model.parameters())
    print("=" * 72)
    print("MNIST matched-width / last conv only dynamic")
    print(f"channels W               : {args.channels}")
    print(f"K                        : {args.K}")
    print(f"per-candidate conv3 width: {9 * args.channels // args.K}")
    print(f"candidate bank width     : {args.K * (9 * args.channels // args.K)} (= 9W)")
    print(f"routed conv3 width       : {9 * args.channels // args.K}")
    print(f"load balance weight      : {args.load_balance_weight}")
    print(f"trainable parameters     : {num_params:,}")
    print("=" * 72)

    start_iter = 0
    best_valid = -1.0
    best_test = -1.0
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = int(ckpt.get("iter", 0))
        best_valid = float(ckpt.get("best_valid_acc", -1.0))
        best_test = float(ckpt.get("best_test_acc", ckpt.get("test_acc", -1.0)))
        print(f"[resume] {args.resume}, iter={start_iter}")

    init_csv(args.curve_csv)

    iterator = enumerate(load_n(train_loader, args.num_iterations))
    for i, (x, y) in tqdm(iterator, total=args.num_iterations, desc="iteration"):
        global_i = start_iter + i + 1
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to("cuda")
        y = y.to("cuda")

        losses = train_step(
            model,
            x,
            y,
            loss_fn,
            optimizer,
            args.load_balance_weight,
        )

        if global_i == 1 or global_i % 1000 == 0:
            print(
                f"iter={global_i} "
                f"loss={losses['total_loss']:.6f} "
                f"ce={losses['ce_loss']:.6f} "
                f"lb={losses['load_balance_loss']:.6f}"
            )

        if global_i % args.eval_freq == 0:
            train_acc = evaluate(model, train_loader, train_mode=False)
            valid_acc = evaluate(model, valid_loader, train_mode=False)
            test_acc, test_routing = evaluate(
                model,
                test_loader,
                train_mode=False,
                collect_routing=True,
            )

            if test_acc > best_test:
                best_test = test_acc

            print(
                f"[eval] iter={global_i} "
                f"train={train_acc:.6f} valid={valid_acc:.6f} test={test_acc:.6f} "
                f"best_test={best_test:.6f} "
                f"loss={losses['total_loss']:.6f} ce={losses['ce_loss']:.6f} "
                f"lb={losses['load_balance_loss']:.6f}"
            )
            print_routing_stats(test_routing, split="test")

            append_csv(args.curve_csv, [
                global_i,
                losses["total_loss"],
                losses["ce_loss"],
                losses["load_balance_loss"],
                train_acc,
                valid_acc,
                test_acc,
            ])

            if valid_acc > best_valid:
                best_valid = valid_acc
                if args.save_ckpt is not None:
                    os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": global_i,
                        "best_valid_acc": best_valid,
                        "best_test_acc": best_test,
                        "test_acc": test_acc,
                        "args": vars(args),
                    }, args.save_ckpt)
                    print(
                        f"[best checkpoint] saved: {args.save_ckpt} "
                        f"valid={best_valid:.6f} test={test_acc:.6f} best_test={best_test:.6f}"
                    )


if __name__ == "__main__":
    main()
