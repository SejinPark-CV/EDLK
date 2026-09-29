import torch
import torch.nn as nn
import torch.nn.functional as F

from difflogic import LogicConvLayer, LogicLayer, GroupSum


class _StopAfterFirstLevel(RuntimeError):
    """Internal control-flow exception used to expose a LogicConvLayer first-level output."""

    def __init__(self, output):
        super().__init__("stop after first logic-tree level")
        self.output = output


class EDLK(nn.Module):
    """EDLK block: only the first gate level is expert-specific."""

    def __init__(
        self,
        in_dim,
        out_dim,
        tree_d,
        rf,
        padding,
        K=2,
        router_hidden=32,
        router_bits=4,
        tau_router=0.1,
        hard_infer=True,
        device="cuda",
        grad_factor=1.0,
        implementation=None,
        connections="random",
    ):
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if router_bits < 1:
            raise ValueError(f"router_bits must be >= 1, got {router_bits}")

        self.K = int(K)
        self.out_dim = int(out_dim)
        self.tree_d = int(tree_d)
        self.hard_infer = bool(hard_infer)
        self.tau_router = float(tau_router)
        self.router_bits = int(router_bits)

        llkw = dict(
            device=device,
            grad_factor=grad_factor,
            implementation=implementation,
            connections=connections,
        )

        # K candidate trees. Only tree_layers.0.weights remain independent.
        self.trees = nn.ModuleList()
        base_tree = LogicConvLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            tree_d=tree_d,
            rf=rf,
            padding=padding,
            **llkw,
        )
        self.trees.append(base_tree)

        for _ in range(1, self.K):
            tree = LogicConvLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                tree_d=tree_d,
                rf=rf,
                padding=padding,
                **llkw,
            )
            with torch.no_grad():
                if hasattr(tree, "cm") and hasattr(base_tree, "cm"):
                    tree.cm.copy_(base_tree.cm)
                if hasattr(tree, "ch") and hasattr(base_tree, "ch"):
                    tree.ch.copy_(base_tree.ch)
                if hasattr(tree, "cw") and hasattr(base_tree, "cw"):
                    tree.cw.copy_(base_tree.cw)
            if hasattr(tree, "_rebuild_tree_from_loaded_connections"):
                tree._rebuild_tree_from_loaded_connections()

            self._share_all_but_first_layer(base_tree, tree)
            self.trees.append(tree)

        rh = int(router_hidden)
        self.router_conv1 = LogicConvLayer(
            in_dim=in_dim,
            out_dim=rh,
            tree_d=tree_d,
            rf=3,
            padding=True,
            **llkw,
        )
        self.router_pool1 = nn.MaxPool2d(2, 2)

        self.router_conv2 = LogicConvLayer(
            in_dim=rh,
            out_dim=rh,
            tree_d=tree_d,
            rf=3,
            padding=True,
            **llkw,
        )
        self.router_pool2 = nn.MaxPool2d(2, 2)

        self.router_conv3 = LogicConvLayer(
            in_dim=rh,
            out_dim=self.K * self.router_bits,
            tree_d=tree_d,
            rf=1,
            padding=False,
            **llkw,
        )

        self.last_logits = None
        self.last_weight = None
        self.last_router_bits = None
        self.last_group_scores = None

    @staticmethod
    def _get_submodule_and_attr(module, dotted_name):
        parts = dotted_name.split(".")
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def _share_parameter_by_name(self, src_tree, dst_tree, param_name):
        src_parent, src_attr = self._get_submodule_and_attr(src_tree, param_name)
        dst_parent, dst_attr = self._get_submodule_and_attr(dst_tree, param_name)
        shared_param = getattr(src_parent, src_attr)

        if dst_attr in getattr(dst_parent, "_parameters", {}):
            del dst_parent._parameters[dst_attr]
        setattr(dst_parent, dst_attr, shared_param)

    def _share_all_but_first_layer(self, src_tree, dst_tree):
        src_params = dict(src_tree.named_parameters())
        dst_params = dict(dst_tree.named_parameters())

        if "tree_layers.0.weights" not in src_params:
            available = ", ".join(f"{n}:{tuple(p.shape)}" for n, p in src_params.items())
            raise RuntimeError(
                "Expected LogicConvLayer to expose tree_layers.0.weights. "
                f"Available parameters: {available}"
            )

        for param_name in list(dst_params.keys()):
            if param_name == "tree_layers.0.weights":
                continue
            if param_name in src_params:
                self._share_parameter_by_name(src_tree, dst_tree, param_name)

        if hasattr(src_tree, "weights") and hasattr(dst_tree, "weights"):
            if "weights" in getattr(dst_tree, "_parameters", {}):
                del dst_tree._parameters["weights"]
            dst_tree.weights = src_tree.weights

    @staticmethod
    def _safe_pool(h, pool):
        if h.size(-1) >= 2 and h.size(-2) >= 2:
            return pool(h)
        return h

    def _route_logits(self, x):
        h = self.router_conv1(x).float()
        h = self._safe_pool(h, self.router_pool1)
        h = self.router_conv2(h).float()
        h = self._safe_pool(h, self.router_pool2)
        h = self.router_conv3(h).float()
        h = F.adaptive_max_pool2d(h, 1).flatten(1)

        B = h.size(0)
        h_bits = h.view(B, self.K, self.router_bits)
        group_scores = h_bits.sum(dim=-1)

        self.last_router_bits = h_bits
        self.last_group_scores = group_scores
        return group_scores

    @staticmethod
    def _run_to_first_level(tree, x):
        """
        Run LogicConvLayer only until tree_layers[0] has produced its output.

        We intentionally go through LogicConvLayer.forward() rather than calling
        tree_layers[0] directly because LogicConvLayer may perform receptive-field
        indexing / reshaping before invoking the first tree level.
        """
        if not hasattr(tree, "tree_layers") or len(tree.tree_layers) < 1:
            raise RuntimeError(
                "EDLK requires LogicConvLayer.tree_layers[0] to represent the first gate level."
            )

        first_layer = tree.tree_layers[0]

        def stop_hook(_module, _inputs, output):
            raise _StopAfterFirstLevel(output)

        handle = first_layer.register_forward_hook(stop_hook)
        try:
            tree(x)
        except _StopAfterFirstLevel as stop:
            return stop.output
        finally:
            handle.remove()

        raise RuntimeError(
            "Failed to intercept LogicConvLayer after tree_layers[0]. "
            "Check the installed difflogic LogicConvLayer implementation."
        )

    @staticmethod
    def _run_shared_upper_levels(base_tree, x, first_level_output):
        """
        Continue through the shared upper gate levels exactly once.

        LogicConvLayer.forward() is reused so all of its internal tree wiring is
        preserved. During this call only, tree_layers[0].forward is replaced by
        the already-routed first-level activation.
        """
        if not hasattr(base_tree, "tree_layers") or len(base_tree.tree_layers) < 1:
            raise RuntimeError(
                "EDLK requires LogicConvLayer.tree_layers[0] to represent the first gate level."
            )

        first_layer = base_tree.tree_layers[0]
        original_forward = first_layer.forward

        def return_routed_first_level(*_args, **_kwargs):
            return first_level_output

        first_layer.forward = return_routed_first_level
        try:
            return base_tree(x)
        finally:
            first_layer.forward = original_forward

    def forward(self, x):
        B = x.size(0)
        logits = self._route_logits(x)
        self.last_logits = logits

        # Same routing rule as before:
        #   training -> soft routing
        #   inference -> hard one-hot routing
        if (not self.training) and self.hard_infer:
            idx = logits.argmax(dim=-1)
            weight = F.one_hot(idx, num_classes=self.K).to(dtype=logits.dtype)
        else:
            weight = F.softmax(logits / self.tau_router, dim=-1)

        self.last_weight = weight

        # In eval mode with the CUDA implementation, intermediate tree-level
        # activations can be represented as difflogic.PackBitsTensor. That
        # packed type is an inference-only internal representation and does not
        # expose Tensor methods such as .float().
        #
        # Because hard inference uses a one-hot routing vector, evaluating the
        # complete candidate trees and selecting their final outputs is
        # numerically equivalent to selecting the corresponding first-level
        # candidate and then applying the shared upper levels:
        #
        #   F(T_g*^(1)(x))
        #
        # This path therefore preserves the paper's hard-routing output while
        # avoiding any attempt to mix packed intermediate tensors.
        if (not self.training) and self.hard_infer:
            cand = [tree(x).float() for tree in self.trees]
            cand = torch.stack(cand, dim=1)  # [B, K, Cout, H, W]
            hard_weight = weight.view(B, self.K, 1, 1, 1)
            return (cand * hard_weight).sum(dim=1)

        # Training path (soft routing):
        #   1) compute each candidate's first gate level T_g^(1)(x)
        #   2) soft-route/mix at the first level
        #   3) run the shared remaining levels once
        first_candidates = [
            self._run_to_first_level(tree, x).float()
            for tree in self.trees
        ]
        first_candidates = torch.stack(first_candidates, dim=1)

        # LogicConvLayer internally flattens spatial positions before applying
        # tree_layers[0]. For conv3 on MNIST this is typically:
        #
        #   B = 64, H = W = 7
        #   first-level leading dimension = 64 * 7 * 7 = 3136
        #
        # Router weights are per image [B, K], so expand each image's routing
        # weights over all of its flattened spatial positions before mixing.
        first_n = first_candidates.size(0)

        if first_n == B:
            expanded_weight = weight
        elif first_n % B == 0:
            positions_per_image = first_n // B
            expanded_weight = weight.repeat_interleave(
                positions_per_image, dim=0
            )
        else:
            raise RuntimeError(
                "Cannot align router weights with first-level LogicConvLayer "
                f"output: batch={B}, first_level_leading_dim={first_n}, "
                f"first_candidates_shape={tuple(first_candidates.shape)}."
            )

        view_shape = [first_n, self.K] + [1] * (first_candidates.dim() - 2)
        routed_first = (
            first_candidates * expanded_weight.view(*view_shape)
        ).sum(dim=1)

        # Upper levels are shared across all groups, so execute them once.
        return self._run_shared_upper_levels(
            self.trees[0],
            x,
            routed_first,
        ).float()


class MNISTLastDynamicMatchedWidth(nn.Module):
    """
    MNIST-only matched-candidate-width EDLK.

    conv1: static, W
    conv2: static, 3W
    conv3: dynamic only (last conv layer)
        - K candidate groups
        - each group width = 9W/K
        - total candidate bank width = K * (9W/K) = 9W

    Routing is applied to the first gate-level output; the shared upper levels
    then produce one routed output with width 9W/K.
    """

    def __init__(
        self,
        tau=10.0,
        grad_factor=2.0,
        device="cuda",
        implementation="cuda",
        connections="random",
        channels=64,
        K=2,
        router_hidden=32,
        router_bits=4,
        tau_router=0.1,
        hard_infer=True,
        in_dim=1,
        num_classes=10,
    ):
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if channels % K != 0:
            raise ValueError(
                f"For matched width, channels must be divisible by K. "
                f"Got channels={channels}, K={K}."
            )

        self.channels = int(channels)
        self.K = int(K)
        self.routed_channels = self.channels // self.K

        W = self.channels
        G = self.K
        Wr = self.routed_channels
        tree_d = 3

        llkw = dict(
            device=device,
            grad_factor=grad_factor,
            implementation=implementation,
            connections=connections,
        )

        # Static early layers.
        self.conv1 = LogicConvLayer(
            in_dim=in_dim, out_dim=W, tree_d=tree_d, rf=5, padding=False, **llkw
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 28 -> 14

        self.conv2 = LogicConvLayer(
            in_dim=W, out_dim=3 * W, tree_d=tree_d, rf=3, padding=True, **llkw
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 14 -> 7

        # ONLY the last convolutional layer is dynamic.
        # Candidate-bank width: G * (9W/G) = 9W, matching the static baseline.
        self.conv3 = EDLK(
            in_dim=3 * W,
            out_dim=9 * Wr,
            tree_d=tree_d,
            rf=3,
            padding=True,
            K=G,
            router_hidden=router_hidden,
            router_bits=router_bits,
            tau_router=tau_router,
            hard_infer=hard_infer,
            **llkw,
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 7 -> 3

        # Routed output has 9W/K channels.
        fc_input_dim = 9 * Wr * 3 * 3

        # Same static-head output-width rule used by the matched MNIST baseline.
        self.fc1 = LogicLayer(in_dim=fc_input_dim, out_dim=500 * W, **llkw)
        self.fc2 = LogicLayer(in_dim=500 * W, out_dim=320 * W, **llkw)
        self.fc3 = LogicLayer(in_dim=320 * W, out_dim=120 * W, **llkw)
        self.classifier = GroupSum(k=num_classes, tau=tau)

    @staticmethod
    def _to_image(x):
        if x.dim() == 2:
            return x.view(x.size(0), 1, 28, 28)
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    def get_load_balance_loss(self):
        # Kept identical to the existing implementation.
        losses = []
        for module in self.modules():
            if isinstance(module, EDLK):
                if module.last_weight is None:
                    continue
                importance = module.last_weight.mean(dim=0)
                hard_idx = module.last_weight.argmax(dim=-1)
                hard_load = F.one_hot(
                    hard_idx, num_classes=module.K
                ).float().mean(dim=0)
                losses.append(module.K * torch.sum(importance * hard_load))

        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def get_load_balance_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, EDLK) and module.last_weight is not None:
                importance = module.last_weight.mean(dim=0)
                hard_idx = module.last_weight.argmax(dim=-1)
                hard_load = F.one_hot(
                    hard_idx, num_classes=module.K
                ).float().mean(dim=0)
                stats[name] = {
                    "importance": importance.detach().cpu(),
                    "load": hard_load.detach().cpu(),
                }
        return stats

    def forward(self, x):
        x = self._to_image(x)
        x = self.pool1(self.conv1(x).float())
        x = self.pool2(self.conv2(x).float())
        x = self.pool3(self.conv3(x).float())
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.classifier(x)
