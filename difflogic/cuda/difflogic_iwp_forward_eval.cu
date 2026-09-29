#include "difflogic_shared.cuh"
#include "difflogic_iwp.h"

template <typename scalar_t>
__global__ void iwp_logic_layer_cuda_eval_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y
) {
    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {

            const auto idx_a = a[col];
            const auto idx_b = b[col];
            const auto a1 = x[idx_a][row];
            const auto b1 = x[idx_b][row];
            const auto a0 = ~a1;
            const auto b0 = ~b1;
            const auto w_ = w[col];
            y[col][row] = (
                  (w_[0] & (a0 & b0)  // 00
                +  w_[1] & (a0 & b1)) // 01
                + (w_[2] & (a1 & b0)  // 10
                +  w_[3] & (a1 & b1)) // 11
            );
        }
    }
}

torch::Tensor iwp_logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = w.size(0);

    // TODO: torch::empty should work aswell, since we overwrite every position anyway.
    auto y = torch::zeros({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "iwp_logic_layer_cuda_eval_kernel", ([&] {
                                   iwp_logic_layer_cuda_eval_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                       x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                       b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                       w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                                   );
                               }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return y;
}