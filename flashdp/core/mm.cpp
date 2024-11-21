#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/torch.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>

namespace py = pybind11;
#define DEFAULT_BLOCK_SIZE 16

torch::Tensor gemm(torch::Tensor A, torch::Tensor B, int block_size = DEFAULT_BLOCK_SIZE);


PYBIND11_MODULE(gemm, m) {
    m.def("gemm", &gemm, "A function that implements the Flash Attention forward pass",
          py::arg("A"), py::arg("B"), py::arg("block_size") = DEFAULT_BLOCK_SIZE);
}


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> 
__global__ void GEMM_kernel(
    float *C, float *A, float *B, 
    int wA, int wB) {

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = wA * BLOCK_SIZE * by;
  int aEnd   = aBegin + wA - 1;
  int aStep  = BLOCK_SIZE;
  int bBegin = BLOCK_SIZE * bx;
  int bStep  = BLOCK_SIZE * wB;

  float Csub = 0;
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}


torch::Tensor gemm(torch::Tensor A, torch::Tensor B, int block_size) {
    // Ensure tensors are on the GPU and are of type float
    if (A.dtype() != torch::kFloat || B.dtype() != torch::kFloat) {
        throw std::runtime_error("Input tensors must be of type CUDA float");
    }
    if (!A.is_cuda() || !B.is_cuda() || A.device() != B.device()) {
        throw std::runtime_error("Input tensors must be on the same CUDA device");
    }

    // Ensure A and B dimensions are compatible for matrix multiplication
    if (A.size(1) != B.size(0)) {
        throw std::runtime_error("Inner dimensions of A and B must match for matrix multiplication");
    }

    int hA = A.size(0);
    int wA = A.size(1);
    int hB = B.size(0);
    int wB = B.size(1);

    // Create output tensor C
    torch::Tensor C = torch::zeros({hA, wB}, A.options());

    // Check if the dimensions can be divided by block_size
    if (wA % block_size != 0 || wB % block_size != 0) {
        std::cerr << "Warning: The dimensions of the matrices are not multiples of the block size. Adjusting grid dimensions to fit all elements.\n";
    }

    // Calculate grid and block dimensions
    dim3 threads(block_size, block_size);
    dim3 grid((wB + block_size - 1) / block_size, (hA + block_size - 1) / block_size);

    // Launch the matrix multiplication kernel
    if (block_size == 64) {
        GEMM_kernel<64><<<grid, threads>>>(
            C.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            wA, wB
        );
    } else if (block_size == 32) {
        GEMM_kernel<32><<<grid, threads>>>(
            C.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            wA, wB
        );
    } else if (block_size == 16) {
        GEMM_kernel<16><<<grid, threads>>>(
            C.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            wA, wB
        );
    } else if (block_size == 8) {
        GEMM_kernel<8><<<grid, threads>>>(
            C.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            wA, wB
        );
    } else if (block_size == 4) {
        GEMM_kernel<4><<<grid, threads>>>(
            C.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            wA, wB
        );
    } else if (block_size == 2) {
        GEMM_kernel<2><<<grid, threads>>>(
            C.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            wA, wB
        );
    } else {
        GEMM_kernel<1><<<grid, threads>>>(
            C.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            wA, wB
        );
    }
    
    return C;
}