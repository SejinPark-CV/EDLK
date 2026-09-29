#include "difflogic_shared.cuh"
#include "difflogic_iwp.h"

template <typename scalar_t>
__global__ void
iwp_logic_layer_cuda_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y_start,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < grad_x.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < grad_x.size(0);
            col += blockDim.y * gridDim.y
        ) {

            scalar_t grad_x_ = 0;

            const auto start = given_x_indices_of_y_start[col];
            const auto end = given_x_indices_of_y_start[col + 1];

            // Sum over all neurons that x has a connection to
            for (int cur = start; cur < end; ++cur) {
                const auto idx_y = given_x_indices_of_y[cur];
                const auto idx_a = a[idx_y];
                const auto idx_b = b[idx_y];
                const auto grad_y_ = grad_y[idx_y][row];
                const auto idx_is_a = idx_a == col;
                const auto w_ = w[idx_y];

                // compute grad_x
                if (idx_is_a) {
                    const auto b1 = x[idx_b][row];
                    const auto b0 = static_cast<scalar_t>(1) - b1;
                    const auto dy_dx = (
                        b0 * (w_[2] - w_[0]) + 
                        b1 * (w_[3] - w_[1])
                    );
                    grad_x_ += dy_dx * grad_y_;
                } else {
                    const auto a1 = x[idx_a][row];
                    const auto a0 = static_cast<scalar_t>(1) - a1;
                    const auto dy_dx = (
                        a0 * (w_[1] - w_[0]) + 
                        a1 * (w_[3] - w_[2])
                    );
                    grad_x_ += dy_dx * grad_y_;
                }
            }
            grad_x[col][row] = grad_x_;
    }}
}


torch::Tensor iwp_logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(given_x_indices_of_y_start);
    CHECK_INPUT(given_x_indices_of_y);

    auto grad_x = torch::empty_like(x);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "iwp_logic_layer_cuda_backward_x", ([&] {
                           iwp_logic_layer_cuda_backward_x_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               given_x_indices_of_y_start.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               given_x_indices_of_y.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>()
                           );
                       }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return grad_x;
}
