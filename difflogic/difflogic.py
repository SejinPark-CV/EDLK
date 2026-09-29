import torch
import torch.nn as nn
import torch.nn.functional as F
import difflogic_cuda
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor
import math
#import difflogic_cuda_iwp

########################################################################################################################
class BinaryEncodingLayer(nn.Module):
    def __init__(self):
        super(BinaryEncodingLayer, self).__init__()

    def forward(self, x):
        # Ensure input is within range [0, 255]
        x = torch.clamp(255.*x, 0, 255).long()

        bit_masks = (1 << torch.arange(8, device=x.device)).view(1, 8, 1, 1)
        binary_encoded = (x.unsqueeze(1) & bit_masks).ne(0).long()
        binary_encoded = binary_encoded.reshape(-1, 8, x.size(2), x.size(3)) 
        return binary_encoded.float()

class KeepMSBPlanes(nn.Module):
    def __init__(self, num_msb=4):
        super().__init__()
        self.num_msb = num_msb
        self.encoder = BinaryEncodingLayer()

    def forward(self, x):
        bit_encoded = self.encoder(x)

        N, _, H, W = bit_encoded.shape
        C = x.size(1)
        bit_encoded = bit_encoded.view(N, C, 8, H, W)

        msb_start = 8 - self.num_msb
        bit_encoded[:, :, :msb_start, :, :] = 0 

        bit_masks = (1 << torch.arange(8, device=x.device)).view(1, 1, 8, 1, 1)
        reconstructed_int = (bit_encoded * bit_masks).sum(dim=2)
        reconstructed = reconstructed_int.float() / 255.

        return reconstructed
    


class LogicConvLayer(torch.nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            tree_d: int,
            rf: int,
            padding = True,
            stride: int = 1,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        #self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tree_d = tree_d
        self.rf = rf
        self.padding = padding
        self.stride = stride
        self.device = device
        self.grad_factor = grad_factor
        self.implementation = implementation
        cm, ch, cw = self.get_connections(in_dim, out_dim, rf, tree_d)
        self.register_buffer('cm', cm)
        self.register_buffer('ch', ch)
        self.register_buffer('cw', cw)

        ### for fast mode
        flat_indices = (self.cm.squeeze() * (rf * rf) + 
                       self.ch.squeeze() * rf + 
                       self.cw.squeeze())  # (out_dim, 2**tree_d)
        indices_a = flat_indices[:, ::2].reshape(-1).contiguous().to(torch.int64).to(device) 
        indices_b = flat_indices[:, 1::2].reshape(-1).contiguous().to(torch.int64).to(device)

        tree_layers = []
        for t in range(tree_d, 0, -1): 
            indices = (indices_a, indices_b) if t == tree_d else None
            tree_layers.append(LogicLayer(in_dim=int(out_dim*(2**t)), 
                                        out_dim=int(out_dim*(2**(t-1))), 
                                        do_perm=False,
                                        indices=indices,
                                        grad_factor=grad_factor,
                                        implementation=implementation, 
                                        connections='unique'))
        self.tree_layers = torch.nn.Sequential(*tree_layers)

    def _rebuild_tree_from_loaded_connections(self):
        """
        Rebuilds the internal `tree_layers` to match loaded `cm`, `ch`, `cw`
        buffers, and then reapplies weights from the old tree.
        """
        try:
            old_tree_weights = self.tree_layers.state_dict()
            device = self.cm.device
            rf = self.rf
            flat_indices = (self.cm.squeeze() * (rf * rf) +
                            self.ch.squeeze() * rf +
                            self.cw.squeeze())
            indices_a = flat_indices[:, ::2].reshape(-1).contiguous().to(torch.int64).to(device)
            indices_b = flat_indices[:, 1::2].reshape(-1).contiguous().to(torch.int64).to(device)

            new_tree_layers = []
            for t in range(self.tree_d, 0, -1):
                indices = (indices_a, indices_b) if t == self.tree_d else None
                new_tree_layers.append(
                    LogicLayer(in_dim=int(self.out_dim * (2**t)),
                               out_dim=int(self.out_dim * (2**(t - 1))),
                               do_perm=False,
                               indices=indices,
                               grad_factor=self.grad_factor,
                               implementation=self.implementation,
                               connections='unique')
                )
            self.tree_layers = torch.nn.Sequential(*new_tree_layers).to(device)
            self.tree_layers.load_state_dict(old_tree_weights)
            # print("[LogicConvLayer] Rebuilt tree from loaded connections.")
        except Exception as e:
            print(f"[LogicConvLayer] Failed to rebuild tree: {e}")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # First, load all tensors into the module
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        # Now that cm/ch/cw are loaded, rebuild tree to match them
        self._rebuild_tree_from_loaded_connections()

    def forward_fast(self, x):
        b, c, h, w = x.shape

        pad = self.rf // 2 if self.padding else 0
        x_padded = F.pad(x, (pad, pad, pad, pad))

        h_out = (h + 2 * pad - self.rf) // self.stride + 1
        w_out = (w + 2 * pad - self.rf) // self.stride + 1

        x_unfold = x_padded.unfold(2, self.rf, self.stride).unfold(3, self.rf, self.stride)
        x_unfold = x_unfold.permute(0, 2, 3, 1, 4, 5).reshape(int(b * h_out * w_out), -1)

        x_unfold_pack = PackBitsTensor(x_unfold.bool(), self.out_dim)
        out_flat = self.tree_layers(x_unfold_pack).unpack()

        out = out_flat.view(b, h_out, w_out, self.out_dim).permute(0, 3, 1, 2)

        return out

    def forward_train(self, x):
        b, _, h, w = x.shape

        pad = self.rf // 2 if self.padding else 0
        x = F.pad(x, (pad, pad, pad, pad))

        h_out = (h + 2 * pad - self.rf) // self.stride + 1
        w_out = (w + 2 * pad - self.rf) // self.stride + 1

        i = torch.arange(h_out, device=self.device, dtype=torch.long)
        j = torch.arange(w_out, device=self.device, dtype=torch.long)
        i = (i * self.stride).view(1, h_out, 1, 1, 1)
        j = (j * self.stride).view(1, 1, w_out, 1, 1)
        ch = self.ch + i
        cw = self.cw + j

        x = x.permute(0, 2, 3, 1)  # b, h, w, c
        x_gathered = x[:, ch, cw, self.cm]  # b, 1, h_out, w_out, out_dim, 2**tree
        reshaped = x_gathered.contiguous().view(-1, self.out_dim * (2**self.tree_d))

        out_flat = self.tree_layers(reshaped)

        out = out_flat.view(b, h_out, w_out, self.out_dim).permute(0, 3, 1, 2)
        return out

    def forward(self, x):
        if self.training:
            if self.implementation == 'cuda_ste':
                out_soft = self.forward_train(x)
                out_hard = self.forward_fast(x)
                return out_hard.detach() + out_soft - out_soft.detach()
            else:
                return self.forward_train(x)
        return self.forward_fast(x)

    
    
    #############################################################################################
    #############################################################################################
    #############################################################################################
    
    def get_connections(self, in_dim, out_dim, rf, tree_d, device='cuda'):
        num_groups = in_dim // 8  # Number of channel groups (m / 8)
        num_inputs_per_tree = int(2**tree_d)  # Binary tree with 2^tree_d inputs

        if num_groups < 2:
            CM = torch.zeros((out_dim, num_inputs_per_tree), dtype=torch.int32)
            for i in range(out_dim):
                CM[i] = torch.tensor(torch.randperm(in_dim)[:num_inputs_per_tree])

            if tree_d == 3:
                rf_range = torch.arange(rf)  
                grid_x, grid_y = torch.meshgrid(rf_range, rf_range, indexing='ij')
                valid_x = grid_x.flatten()
                valid_y = grid_y.flatten()

                CH = torch.zeros((out_dim, num_inputs_per_tree), dtype=torch.int32)
                CW = torch.zeros((out_dim, num_inputs_per_tree), dtype=torch.int32)

                for i in range(out_dim):
                    perm = torch.randperm(len(valid_x))[:num_inputs_per_tree] 
                    CH[i] = valid_x[perm]
                    CW[i] = valid_y[perm]
        else:
            # Ensure channels are split into groups
            channels_per_group = in_dim // num_groups  #%
            
            CM = torch.zeros((out_dim, num_inputs_per_tree), dtype=torch.int32)

            for i in range(out_dim):
                group_idx = i % num_groups
                start_channel = group_idx * channels_per_group
                end_channel = start_channel + channels_per_group

                CM[i] = torch.randint(low=start_channel, high=end_channel, size=(num_inputs_per_tree,))
            CH = torch.randint(low=0, high=rf, size=(out_dim, num_inputs_per_tree))  
            CW = torch.randint(low=0, high=rf, size=(out_dim, num_inputs_per_tree)) 
        CM = CM.long().unsqueeze(0).unsqueeze(1).unsqueeze(2)  
        CH = CH.long().unsqueeze(0).unsqueeze(1).unsqueeze(2)  
        CW = CW.long().unsqueeze(0).unsqueeze(1).unsqueeze(2)  

        return CM.to(device), CH.to(device), CW.to(device)




class LogicConvLayerRandom(torch.nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            tree_d: int,
            rf: int,
            padding = True,
            stride: int = 1,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        #self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tree_d = tree_d
        self.rf = rf
        self.padding = padding
        self.stride = stride
        self.device = device
        self.grad_factor = grad_factor
        self.implementation = implementation
        cm, ch, cw = self.get_connections(in_dim, out_dim, rf, tree_d)
        self.register_buffer('cm', cm)
        self.register_buffer('ch', ch)
        self.register_buffer('cw', cw)

        ### for fast mode
        flat_indices = (self.cm.squeeze() * (rf * rf) + 
                       self.ch.squeeze() * rf + 
                       self.cw.squeeze())  # (out_dim, 2**tree_d)
        indices_a = flat_indices[:, ::2].reshape(-1).contiguous().to(torch.int64).to(device) 
        indices_b = flat_indices[:, 1::2].reshape(-1).contiguous().to(torch.int64).to(device)

        tree_layers = []
        for t in range(tree_d, 0, -1): 
            indices = (indices_a, indices_b) if t == tree_d else None
            tree_layers.append(LogicLayerRandom(in_dim=int(out_dim*(2**t)), 
                                        out_dim=int(out_dim*(2**(t-1))), 
                                        do_perm=False,
                                        indices=indices,
                                        grad_factor=grad_factor,
                                        implementation=implementation, 
                                        connections='unique'))
        self.tree_layers = torch.nn.Sequential(*tree_layers)

    def _rebuild_tree_from_loaded_connections(self):
        """
        Rebuilds the internal `tree_layers` to match loaded `cm`, `ch`, `cw`
        buffers, and then reapplies weights from the old tree.
        """
        try:
            old_tree_weights = self.tree_layers.state_dict()
            device = self.cm.device
            rf = self.rf
            flat_indices = (self.cm.squeeze() * (rf * rf) +
                            self.ch.squeeze() * rf +
                            self.cw.squeeze())
            indices_a = flat_indices[:, ::2].reshape(-1).contiguous().to(torch.int64).to(device)
            indices_b = flat_indices[:, 1::2].reshape(-1).contiguous().to(torch.int64).to(device)

            new_tree_layers = []
            for t in range(self.tree_d, 0, -1):
                indices = (indices_a, indices_b) if t == self.tree_d else None
                new_tree_layers.append(
                    LogicLayerRandom(in_dim=int(self.out_dim * (2**t)),
                               out_dim=int(self.out_dim * (2**(t - 1))),
                               do_perm=False,
                               indices=indices,
                               grad_factor=self.grad_factor,
                               implementation=self.implementation,
                               connections='unique')
                )
            self.tree_layers = torch.nn.Sequential(*new_tree_layers).to(device)
            self.tree_layers.load_state_dict(old_tree_weights)
            # print("[LogicConvLayer] Rebuilt tree from loaded connections.")
        except Exception as e:
            print(f"[LogicConvLayer] Failed to rebuild tree: {e}")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # First, load all tensors into the module
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        # Now that cm/ch/cw are loaded, rebuild tree to match them
        self._rebuild_tree_from_loaded_connections()

    def forward_fast(self, x):
        b, c, h, w = x.shape

        pad = self.rf // 2 if self.padding else 0
        x_padded = F.pad(x, (pad, pad, pad, pad))

        h_out = (h + 2 * pad - self.rf) // self.stride + 1
        w_out = (w + 2 * pad - self.rf) // self.stride + 1

        x_unfold = x_padded.unfold(2, self.rf, self.stride).unfold(3, self.rf, self.stride)
        x_unfold = x_unfold.permute(0, 2, 3, 1, 4, 5).reshape(int(b * h_out * w_out), -1)

        x_unfold_pack = PackBitsTensor(x_unfold.bool(), self.out_dim)
        out_flat = self.tree_layers(x_unfold_pack).unpack()

        out = out_flat.view(b, h_out, w_out, self.out_dim).permute(0, 3, 1, 2)

        return out

    def forward_train(self, x):
        b, _, h, w = x.shape

        pad = self.rf // 2 if self.padding else 0
        x = F.pad(x, (pad, pad, pad, pad))

        h_out = (h + 2 * pad - self.rf) // self.stride + 1
        w_out = (w + 2 * pad - self.rf) // self.stride + 1

        i = torch.arange(h_out, device=self.device, dtype=torch.long)
        j = torch.arange(w_out, device=self.device, dtype=torch.long)
        i = (i * self.stride).view(1, h_out, 1, 1, 1)
        j = (j * self.stride).view(1, 1, w_out, 1, 1)
        ch = self.ch + i
        cw = self.cw + j

        x = x.permute(0, 2, 3, 1)  # b, h, w, c
        x_gathered = x[:, ch, cw, self.cm]  # b, 1, h_out, w_out, out_dim, 2**tree
        reshaped = x_gathered.contiguous().view(-1, self.out_dim * (2**self.tree_d))

        out_flat = self.tree_layers(reshaped)

        out = out_flat.view(b, h_out, w_out, self.out_dim).permute(0, 3, 1, 2)
        return out

    def forward(self, x):
        if self.training:
            if self.implementation == 'cuda_ste':
                out_soft = self.forward_train(x)
                out_hard = self.forward_fast(x)
                return out_hard.detach() + out_soft - out_soft.detach()
            else:
                return self.forward_train(x)
        return self.forward_fast(x)

    
    
    #############################################################################################
    #############################################################################################
    #############################################################################################
    
    def get_connections(self, in_dim, out_dim, rf, tree_d, device='cuda'):
        num_groups = in_dim // 1  # Number of channel groups (m / 8)
        num_inputs_per_tree = int(2**tree_d)  # Binary tree with 2^tree_d inputs

        if num_groups < 2:
            CM = torch.zeros((out_dim, num_inputs_per_tree), dtype=torch.int32)
            for i in range(out_dim):
                CM[i] = torch.tensor(torch.randperm(in_dim)[:num_inputs_per_tree])

            if tree_d == 3:
                rf_range = torch.arange(rf)  
                grid_x, grid_y = torch.meshgrid(rf_range, rf_range, indexing='ij')
                valid_x = grid_x.flatten()
                valid_y = grid_y.flatten()

                CH = torch.zeros((out_dim, num_inputs_per_tree), dtype=torch.int32)
                CW = torch.zeros((out_dim, num_inputs_per_tree), dtype=torch.int32)

                for i in range(out_dim):
                    perm = torch.randperm(len(valid_x))[:num_inputs_per_tree] 
                    CH[i] = valid_x[perm]
                    CW[i] = valid_y[perm]
        else:
            # Ensure channels are split into groups
            channels_per_group = in_dim // num_groups  #%
            
            CM = torch.zeros((out_dim, num_inputs_per_tree), dtype=torch.int32)

            for i in range(out_dim):
                group_idx = i % num_groups
                start_channel = group_idx * channels_per_group
                end_channel = start_channel + channels_per_group

                CM[i] = torch.randint(low=start_channel, high=end_channel, size=(num_inputs_per_tree,))
            CH = torch.randint(low=0, high=rf, size=(out_dim, num_inputs_per_tree))  
            CW = torch.randint(low=0, high=rf, size=(out_dim, num_inputs_per_tree)) 
        CM = CM.long().unsqueeze(0).unsqueeze(1).unsqueeze(2)  
        CH = CH.long().unsqueeze(0).unsqueeze(1).unsqueeze(2)  
        CW = CW.long().unsqueeze(0).unsqueeze(1).unsqueeze(2)  

        return CM.to(device), CH.to(device), CW.to(device)


class LogicLayerRandom(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            do_perm: bool = True,
            indices = None, 
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = "cuda",
            connections: str = 'random',
            group_size: int = None,
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        :param group_size: size of the group for the group connection
        """
        super().__init__()
        
        # or init
        
        #init_weights = torch.zeros(out_dim, 16, device=device)
        #init_weights[:,7] = 5
        #self.weights = torch.nn.parameter.Parameter(init_weights)
        
        init_weights = torch.randn(out_dim, 16, device=device)
        self.weights = torch.nn.parameter.Parameter(init_weights)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.do_perm = do_perm
        self.device = device
        self.grad_factor = grad_factor
        self.group_size = group_size

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'cuda_ste', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random', 'unique', 'group'], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == 'cuda' or self.implementation == 'cuda_ste':
            """
            Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            """
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

        self.fast_indices = indices if indices is not None else self.indices
            
        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            # assert not self.training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA, not for {}. ' \
                                          'If you want fast inference on CPU, please use CompiledDiffLogicModel.' \
                                          ''.format(self.device)

        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)
        if self.implementation == 'cuda' or 'cuda_ste':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        # elif self.implementation == 'cuda_ste':       
        #     if self.training == False:
        #         if isinstance(x, PackBitsTensor):
        #             return self.forward_cuda_eval(x)
        #     else:
        #         self.training = True
        #         out_soft = self.forward_cuda(x)
        #         # self.training = False
        #         # out_hard = self.forward_cuda(x)
        #         # out = out_hard.detach() + out_soft - out_soft.detach()
        #         # self.training = True
        #     return out_soft
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        # CUDA 커널은 float 타입만 지원하므로 변환
        if x.dtype != torch.float32:
            x = x.float()

        x = x.transpose(0, 1)
        x = x.contiguous()

        if self.indices == None:
            assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        if self.training:
            a, b = self.indices
            w = torch.nn.functional.softmax(self.weights - torch.max(self.weights, dim=-1, keepdim=True)[0], dim=-1).to(x.dtype)
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            a, b = self.fast_indices
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)

    def forward_cuda_eval(self, x: PackBitsTensor):
        """
        WARNING: this is an in-place operation.

        :param x:
        :return:
        """
        # assert not self.training
        assert isinstance(x, PackBitsTensor)

        a, b = self.fast_indices
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)

        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, device='cuda'):
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, self.do_perm, device)
        elif connections == 'group':
            # 각 그룹 내에서 연결 생성
            groups = self.group_size
            in_size = self.in_dim // groups
            out_size = self.out_dim // groups
            
            # out_size가 in_size보다 클 경우 반복 허용
            all_a_indices = []
            all_b_indices = []
            
            for g in range(groups):
                start = g * in_size
                end = start + in_size
                
                # 각 그룹에서 out_size만큼의 연결 생성
                if out_size <= in_size:
                    # 중복 없이 선택
                    a_idx = torch.randperm(in_size)[:out_size] + start
                    b_idx = torch.randperm(in_size)[:out_size] + start
                else:
                    # 중복 허용하여 선택
                    a_idx = torch.randint(start, end, (out_size,))
                    b_idx = torch.randint(start, end, (out_size,))
                
                all_a_indices.append(a_idx)
                all_b_indices.append(b_idx)

            a = torch.cat(all_a_indices)
            b = torch.cat(all_b_indices)
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        else:
            raise ValueError(connections)

class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            do_perm: bool = True,
            indices = None, 
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        
        # skip init
        init_weights = torch.zeros(out_dim, 16, device=device)
        init_weights[:,3] = 5
        self.weights = torch.nn.parameter.Parameter(init_weights)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.do_perm = do_perm
        self.device = device
        self.grad_factor = grad_factor

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'cuda_ste', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == 'cuda' or self.implementation == 'cuda_ste':
            """
            Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            """
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

        self.fast_indices = indices if indices is not None else self.indices
            
        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            # assert not self.training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA, not for {}. ' \
                                          'If you want fast inference on CPU, please use CompiledDiffLogicModel.' \
                                          ''.format(self.device)

        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)
        if self.implementation == 'cuda' or 'cuda_ste':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        # elif self.implementation == 'cuda_ste':       
        #     if self.training == False:
        #         if isinstance(x, PackBitsTensor):
        #             return self.forward_cuda_eval(x)
        #     else:
        #         self.training = True
        #         out_soft = self.forward_cuda(x)
        #         # self.training = False
        #         # out_hard = self.forward_cuda(x)
        #         # out = out_hard.detach() + out_soft - out_soft.detach()
        #         # self.training = True
        #     return out_soft
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        if self.indices == None:
            assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        if self.training:
            a, b = self.indices
            w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            a, b = self.fast_indices
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)

    def forward_cuda_eval(self, x: PackBitsTensor):
        """
        WARNING: this is an in-place operation.

        :param x:
        :return:
        """
        # assert not self.training
        assert isinstance(x, PackBitsTensor)

        a, b = self.fast_indices
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)

        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, device='cuda'):
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, self.do_perm, device)
        else:
            raise ValueError(connections)


########################################################################################################################


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)


########################################################################################################################


class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None
