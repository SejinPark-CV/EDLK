#include "difflogic_shared.cuh"
#include "difflogic_iwp.h"

template <typename scalar_t>
__global__ void
iwp_logic_layer_cuda_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_w_
) {

    const auto row_ = blockIdx.x * blockDim.x + threadIdx.x;

    for (  // neuron dim
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        col < grad_y.size(0);
        col += blockDim.y * gridDim.y
    ) {
        const auto idx_a = a[col];
        const auto idx_b = b[col];
        scalar_t grad_w_local_00 = 0;
        scalar_t grad_w_local_01 = 0;
        scalar_t grad_w_local_10 = 0;
        scalar_t grad_w_local_11 = 0;
        for (int row = row_; row < grad_y.size(1); row += BACKWARD_W_BATCH_THREADS) {  // batch dim
            const auto a1 = x[idx_a][row];
            const auto b1 = x[idx_b][row];
            
            const auto a0 = static_cast<scalar_t>(1) - a1;
            const auto b0 = static_cast<scalar_t>(1) - b1;
            const auto grad_y_ = grad_y[col][row];

            // compute grad_w
            grad_w_local_00 += (a0 * b0) * grad_y_; // 00
            grad_w_local_01 += (a0 * b1) * grad_y_; // 01
            grad_w_local_10 += (a1 * b0) * grad_y_; // 10
            grad_w_local_11 += (a1 * b1) * grad_y_; // 11
        }

        grad_w_[col][row_][0] = grad_w_local_00;
        grad_w_[col][row_][1] = grad_w_local_01;
        grad_w_[col][row_][2] = grad_w_local_10;
        grad_w_[col][row_][3] = grad_w_local_11;
    }
}


torch::Tensor iwp_logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(grad_y);


    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = grad_y.size(0);

    auto grad_w = torch::empty({out_size, BACKWARD_W_BATCH_THREADS, 4}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(BACKWARD_W_BATCH_THREADS, 1024 / BACKWARD_W_BATCH_THREADS);

    const dim3 blocks_per_grid(
        1,
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "iwp_logic_layer_cuda_backward_w", ([&] {
                           iwp_logic_layer_cuda_backward_w_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_w.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                       }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const auto grad_w_batch = grad_w.sum(1);

    return grad_w_batch;
}
