#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

#include "difflogic_iwp.h"

namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    m.def(
        "iwp_forward",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w) {
            return iwp_logic_layer_cuda_forward(x, a, b, w);
        },
        "iwp logic layer forward (CUDA)");
    m.def(
        "iwp_backward_w", [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor grad_y) {
            return iwp_logic_layer_cuda_backward_w(x, a, b, grad_y);
        },
        "iwp logic layer backward w (CUDA)");
    m.def(
        "iwp_backward_x",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w, torch::Tensor grad_y, torch::Tensor given_x_indices_of_y_start, torch::Tensor given_x_indices_of_y) {
            return iwp_logic_layer_cuda_backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y);
        },
        "iwp logic layer backward x (CUDA)");
    m.def(
        "iwp_eval",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w) {
            return iwp_logic_layer_cuda_eval(x, a, b, w);
        },
        "iwp logic layer eval (CUDA)");
    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}
