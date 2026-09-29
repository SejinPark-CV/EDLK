#pragma once
#ifndef DIFFLOGIC_IWP_H
#define DIFFLOGIC_IWP_H

torch::Tensor iwp_logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);
torch::Tensor iwp_logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
);
torch::Tensor iwp_logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
);
torch::Tensor iwp_logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);

#endif // DIFFLOGIC_IWP_H