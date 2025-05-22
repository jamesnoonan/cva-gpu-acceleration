#include <stdio.h>
#include <cuda_runtime.h>
#include "../valuation.h"

#define HANDLE_ERROR(e)                                                       \
  do {                                                                        \
    if ((e) != cudaSuccess) {                                                 \
      fprintf(stderr, "[CUDA Error] %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(e));                                         \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

__device__ void merge_tuples_device(
    uint8_t *dst,
    const uint8_t *tuple1, uint32_t size1,
    const uint8_t *tuple2, uint32_t size2,
    const uint32_t *v2_indices, uint32_t common_size)
    {
    uint32_t idx = 0;

    // Copy all from tuple1
    for (uint32_t i = 0; i < size1; i++) {
        dst[idx++] = tuple1[i];
    }

    uint32_t pointer = 0; 
    for (uint32_t j = 0; j < size2; j++) {
        if (pointer < common_size && v2_indices[pointer] == j) {
            pointer++;
            continue;
        }
        dst[idx++] = tuple2[j];
    }
}

__global__ void combine_valuations_kernel(
    Valuation *dst,
    Valuation *v1,
    Valuation *v2,
    uint32_t dst_tuple_size,
    uint32_t *v2_indices,
    uint32_t common_var_size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = v1->size * v2->size;

    if (idx >= total) return;

    uint32_t i = idx / v2->size;
    uint32_t j = idx % v2->size;

    // Skip if any input row is incompatible
    if (v1->rows[i].incompatible || v2->rows[j].incompatible)
        return;

    ValuationRow *out_row = &dst->rows[idx];

    merge_tuples_device(
        out_row->tuple,
        v1->rows[i].tuple, v1->domain_size,
        v2->rows[j].tuple, v2->domain_size,
        v2_indices,
        common_var_size);

    out_row->value = v1->rows[i].value * v2->rows[j].value;
    out_row->incompatible = 0;
}


int combine_valuations(Valuation *device_dst, Valuation *device_v1, Valuation *device_v2, Valuation *dst, Valuation *v1, Valuation *v2) {
    if (!dst) return -1;

    float comp_millis = 0.f;
    cudaEvent_t comp_start, comp_end;
    HANDLE_ERROR(cudaEventCreate(&comp_start));
    HANDLE_ERROR(cudaEventCreate(&comp_end));

    uint32_t *v1_indices, *v2_indices;
    uint32_t common_var_size = get_common_vars(
        v1->domain, v1->domain_size,
        v2->domain, v2->domain_size,
        &v1_indices, &v2_indices);

    if (!v1_indices || !v2_indices) {
        free(v1_indices);
        free(v2_indices);
        return -1;
    }

    // Copy v2_indices to device (v1_indices not needed for merge)
    uint32_t *d_v2_indices;
    cudaMalloc(&d_v2_indices, sizeof(uint32_t) * common_var_size);
    cudaMemcpy(d_v2_indices, v2_indices, sizeof(uint32_t) * common_var_size, cudaMemcpyHostToDevice);

    // Compute total output size
    uint32_t total = v1->size * v2->size;
    dst->size = total;  // May trim later

    // Launch kernel
    uint32_t threads_per_block = 256;
    uint32_t blocks = (total + threads_per_block - 1) / threads_per_block;

    HANDLE_ERROR(cudaEventRecord(comp_start)); 
    combine_valuations_kernel<<<blocks, threads_per_block>>>(
        device_dst, device_v1, device_v2,
        dst->domain_size,
        d_v2_indices,
        common_var_size);

    HANDLE_ERROR(cudaEventRecord(comp_end));
    HANDLE_ERROR(cudaEventSynchronize(comp_end));
    HANDLE_ERROR(cudaEventElapsedTime(&comp_millis, comp_start, comp_end));
    HANDLE_ERROR(cudaEventDestroy(comp_start));
    HANDLE_ERROR(cudaEventDestroy(comp_end));

    cudaDeviceSynchronize();  // Ensure completion before returning

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[Combine CUDA Error] %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Computation time: %.3f ms\n", comp_millis);

    cudaFree(d_v2_indices);
    free(v1_indices);
    free(v2_indices);
    return 0;
}