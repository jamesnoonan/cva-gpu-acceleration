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
    uint32_t *v1_indices,
    uint32_t *v2_indices,
    uint32_t common_var_size)
{
    uint32_t work_per_thread = 1;
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = v1->size * v2->size;

    uint32_t start_idx = thread_id * work_per_thread;
    uint32_t end_idx = min(start_idx + work_per_thread, total);

    uint8_t incompatible = 0, v1_tuple_val, v2_tuple_val;
    for (uint32_t idx = start_idx; idx < end_idx; ++idx) {
        uint32_t i = idx / v2->size;
        uint32_t j = idx % v2->size;

        incompatible = v1->rows[i].incompatible + v2->rows[j].incompatible;

        for (uint32_t k = 0; k < common_var_size; ++k) {
            v1_tuple_val = v1->rows[i].tuple[v1_indices[k]];
            v2_tuple_val = v2->rows[j].tuple[v2_indices[k]];
            incompatible += abs(v1_tuple_val - v2_tuple_val);
        }

        ValuationRow *out_row = &dst->rows[idx];

        merge_tuples_device(
            out_row->tuple,
            v1->rows[i].tuple, v1->domain_size,
            v2->rows[j].tuple, v2->domain_size,
            v2_indices,
            common_var_size);

        out_row->value = v1->rows[i].value * v2->rows[j].value;
        out_row->incompatible = incompatible;
    }
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

    // Copy v1_indices to device
    uint32_t *d_v1_indices;
    cudaMalloc(&d_v1_indices, sizeof(uint32_t) * common_var_size);
    cudaMemcpy(d_v1_indices, v1_indices, sizeof(uint32_t) * common_var_size, cudaMemcpyHostToDevice);

    // Copy v2_indices to device
    uint32_t *d_v2_indices;
    cudaMalloc(&d_v2_indices, sizeof(uint32_t) * common_var_size);
    cudaMemcpy(d_v2_indices, v2_indices, sizeof(uint32_t) * common_var_size, cudaMemcpyHostToDevice);

    // Compute total output size
    uint32_t total = v1->size * v2->size;
    dst->size = total;

    uint32_t work_per_thread = 1;
    uint32_t threads_needed = (total + work_per_thread - 1) / work_per_thread;

    uint32_t threads_per_block = 256;
    uint32_t blocks = (threads_needed + threads_per_block - 1) / threads_per_block;

    // printf("Combining valuations with %d blocks and %d threads per block\n", blocks, threads_per_block);

    HANDLE_ERROR(cudaEventRecord(comp_start)); 
    combine_valuations_kernel<<<blocks, threads_per_block>>>(
        device_dst, device_v1, device_v2,
        dst->domain_size,
        d_v1_indices,
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

    // Print milliseconds
    printf("%.6f,", comp_millis);

    cudaFree(d_v1_indices);
    cudaFree(d_v2_indices);
    free(v1_indices);
    free(v2_indices);
    return 0;
}